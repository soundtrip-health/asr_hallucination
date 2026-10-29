"""High-level processing functions for batch transcription jobs."""

import json
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

from .config import setup_logging, get_uuid
from .audio import load_audio, audio_to_numpy, mix_audio, get_vad_probabilities
from .ami_utils import (
    get_audiofile,
    load_manual_transcript_words,
    load_transcript_words,
)
from .transcription import (
    whisperx_transcribe_audio,
    transcribe_rev,
    assemblyai_transcribe_audio,
    aws_transcribe_audio,
    check_aws_job_status,
)
from .s3_utils import upload_file, download_file, delete_uri

# S3 prefix for temporary AWS Transcribe input/output files, e.g. "s3://your-bucket/tmp".
# Can be overridden per call or with the AWS_S3_SCRATCH_URI environment variable.
S3_SCRATCH_ENV = "AWS_S3_SCRATCH_URI"


def process_session_audio(session_id, outbase, s3_session, datadir, snr=float("inf"), noisefn=None, rerun=False,
                          s3_scratch_uri=None):
    """
    Process a single AMI session: mix with noise, compute VAD, and transcribe with all available ASR systems.

    Note: the file names should be unique, but this assumes that the snr values are integers, or
    if they are floats that they don't conflict when rounded to an int.

    Args:
        session_id: AMI session ID
        outbase: output directory base path
        s3_session: boto3 session for AWS Transcribe
        datadir: base directory containing AMI corpus
        snr: signal-to-noise ratio in dB
        noisefn: path to noise file
        rerun: whether to rerun if output files exist
        s3_scratch_uri: S3 prefix for temporary AWS Transcribe files (e.g. "s3://your-bucket/tmp");
            defaults to the AWS_S3_SCRATCH_URI environment variable. AWS is skipped if unset.

    Returns:
        tuple of (vad_df, aws_df, aai_df, wsp_df, rev_df)
    """
    logger = setup_logging()
    s3_scratch_uri = s3_scratch_uri or os.environ.get(S3_SCRATCH_ENV)
    audio_path = get_audiofile(session_id, datadir)
    if not audio_path:
        logger.error(f"Audio file not found for {session_id=}.")
        return False

    outpath_session = Path(outbase) / f"{session_id}"
    outpath_session.mkdir(parents=True, exist_ok=True)

    if snr != float("inf") and noisefn is not None:
        noisefn_str = str(noisefn)
        if noisefn_str.startswith("s3://") and s3_session is None:
            logger.warning(f"Noise file {noisefn_str} is an S3 URI but AWS credentials are not available. Skipping noise mixing.")
            noisefn = "none"
        else:
            noise = audio_to_numpy(load_audio([noisefn_str], s3_session=s3_session).set_channels(1).set_frame_rate(16000))
            noisefn = Path(noisefn).stem
            snr_audio_path = outpath_session / f"audio_{noisefn}_snr{snr:+03.0f}.wav"
            if snr_audio_path.is_file() and not rerun:
                logger.info(f"File {snr_audio_path} exists and rerun is False. Using that audio file...")
            else:
                logger.info(f"Mixing in noise {snr_audio_path=}...")
                # Assumes the files are already mono 16kHz
                signal = audio_to_numpy(load_audio([str(audio_path)]))
                aseg = mix_audio(signal, noise, snr)
                aseg.export(snr_audio_path, format="wav")
            audio_path = snr_audio_path
    else:
        noisefn = "none"

    # VAD
    vad_path = outpath_session / f"vad_{noisefn}_snr{snr:+03.0f}.feather"
    if vad_path.is_file() and not rerun:
        logger.info(f"File {vad_path} exists and rerun is False. Skipping...")
        vad_df = pd.read_feather(vad_path)
    else:
        logger.info(f"Computing VAD {vad_path=}...")
        speech_probs, step = get_vad_probabilities(audio_path, kind="silero")
        vad_times_msec = (np.arange(len(speech_probs)) * step * 1000).round().astype(int)
        vad_df = pd.DataFrame(data={"time": vad_times_msec, "vad_prob": speech_probs})
        vad_df.to_feather(vad_path)

    # Transcribe with AWS
    aws_path = outpath_session / f"xscript_aws_{noisefn}_snr{snr:+03.0f}.json"
    aws_job_status = None
    aws_tmp_audio_uri = None
    transcribe_client = None
    result_uri = None
    if aws_path.is_file() and not rerun:
        logger.info(f"File {aws_path} exists and rerun is False. Skipping...")
        aws_df = load_transcript_words(aws_path, "aws")
    elif s3_session is None:
        logger.warning("AWS credentials not available. Skipping AWS transcription.")
        aws_df = pd.DataFrame()
    elif not s3_scratch_uri:
        logger.warning(f"No S3 scratch location (set {S3_SCRATCH_ENV} or --s3-scratch). Skipping AWS transcription.")
        aws_df = pd.DataFrame()
    else:
        # upload temp file. The file uri must be unique to avoid bugs when multi-threaded.
        job_name = get_uuid()
        scratch = s3_scratch_uri.rstrip("/")
        aws_tmp_audio_uri = f"{scratch}/{job_name}_audio.wav"
        upload_file(s3_session, aws_tmp_audio_uri, audio_path)
        # For a list of language code options: https://docs.aws.amazon.com/transcribe/latest/dg/supported-languages.html
        languages = ["en-US", "en-GB"]  # add en-AB (scottish?)
        result_uri = f"{scratch}/{job_name}.json"
        logger.info(f"Transcribing AWS {aws_path=}, {job_name=}, {aws_tmp_audio_uri=}, {result_uri=}, {languages=}...")
        aws_job_status, transcribe_client = aws_transcribe_audio(s3_session, aws_tmp_audio_uri, result_uri, languages=languages, job_name=job_name)

    # Transcribe with AssemblyAI
    aai_path = outpath_session / f"xscript_aai_{noisefn}_snr{snr:+03.0f}.json"
    if aai_path.is_file() and not rerun:
        logger.info(f"File {aai_path} exists and rerun is False. Skipping...")
        aai_df = load_transcript_words(aai_path, "aai")
    else:
        logger.info(f"Transcribing AssemblyAI {aai_path=}...")
        try:
            aai_result, aai_result_raw = assemblyai_transcribe_audio(audio_path)
            with open(aai_path, "wt") as fp:
                json.dump(aai_result, fp)
            # We want to save the raw aai output as well. That will have lots of useful metadata, diarization info, etc.
            with open(aai_path.with_stem(aai_path.stem + "_raw"), "wt") as fp:
                json.dump(aai_result_raw, fp)
            aai_df = load_transcript_words(aai_path, "aai")
        except (ValueError, ImportError) as e:
            logger.warning(f"AssemblyAI transcription skipped for {session_id=}: {e}")
            aai_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed for {session_id=}: {e}")
            aai_df = pd.DataFrame()

    # Transcribe with WhisperX
    wsp_path = outpath_session / f"xscript_wsp_{noisefn}_snr{snr:+03.0f}.json"
    if wsp_path.is_file() and not rerun:
        logger.info(f"File {wsp_path} exists and rerun is False. Skipping...")
        wsp_df = load_transcript_words(wsp_path, "wsp")
    else:
        logger.info(f"Transcribing WhisperX {wsp_path=}...")
        wsp_result = whisperx_transcribe_audio(audio_path)
        with open(wsp_path, "wt") as fp:
            json.dump(wsp_result, fp)
        wsp_df = load_transcript_words(wsp_path, "wsp")

    # Transcribe with Reverb
    rev_path = outpath_session / f"xscript_rev_{noisefn}_snr{snr:+03.0f}.json"
    if rev_path.is_file() and not rerun:
        logger.info(f"File {rev_path} exists and rerun is False. Skipping...")
        rev_df = load_transcript_words(rev_path, "rev")
    else:
        logger.info(f"Transcribing Reverb {rev_path=}...")
        rev_result = transcribe_rev(audio_path)
        with open(rev_path, "wt") as fp:
            json.dump(rev_result, fp)
        rev_df = load_transcript_words(rev_path, "rev")

    # Check on the AWS job
    if aws_job_status is not None and transcribe_client is not None and result_uri is not None:
        aws_job_status = check_aws_job_status(aws_job_status["TranscriptionJob"]["TranscriptionJobName"], transcribe_client)
        while aws_job_status["TranscriptionJob"]["TranscriptionJobStatus"] != "COMPLETED":
            time.sleep(2)
            aws_job_status = check_aws_job_status(aws_job_status["TranscriptionJob"]["TranscriptionJobName"], transcribe_client)
        download_file(s3_session, result_uri, aws_path)
        logger.info(f"Downloaded AWS transcription {result_uri=} to {aws_path=}...")
        aws_df = load_transcript_words(aws_path, "aws")
        if aws_tmp_audio_uri is not None:
            # clean up the temp file
            logger.debug(f"Deleting temp file {aws_tmp_audio_uri}...")
            delete_uri(s3_session, aws_tmp_audio_uri)

    return vad_df, aws_df, aai_df, wsp_df, rev_df


def merge_session_files(session_dir, num_snrs, noise_name):
    """Merge multiple SNR transcript files for a session.

    Args:
        session_dir: directory containing session files
        num_snrs: expected number of SNR files
        noise_name: noise file stem name
    """
    logger = setup_logging()
    session_dir = Path(session_dir)
    snr_files = [f for f in session_dir.glob(f"xscript_{noise_name}_snr*.feather")]
    # ensure that all the expected snr files are there
    if len(snr_files) != num_snrs:
        logger.debug(f"session {session_dir} incomplete ({len(snr_files)} of {num_snrs} expected files)")
        return
    logger.info(f"Merging {len(snr_files)} files for {session_dir=} and {noise_name=}...")
    dfs = []
    for fn in snr_files:
        dfs.append(pd.read_feather(fn))
    df = pd.concat(dfs)
    outfn = session_dir / f"xscript_{noise_name}.feather"
    df.to_feather(outfn)


def process_session(session_id, snr, outbase, s3_session, noisefn, datadir, num_snrs=None, reprocess=False,
                    s3_scratch_uri=None):
    """Process a single session at a specific SNR level.

    Args:
        session_id: AMI session ID
        snr: signal-to-noise ratio
        outbase: output directory base path
        s3_session: boto3 session
        noisefn: Path to noise file
        datadir: base directory containing AMI corpus
        num_snrs: total number of SNR levels (for merging)
        reprocess: whether to reprocess existing files
        s3_scratch_uri: S3 prefix for temporary AWS Transcribe files (see process_session_audio)
    """
    logger = setup_logging()
    out_file = outbase / session_id / f"xscript_{noisefn.stem}_snr{snr:+03.0f}.feather"
    if out_file.is_file() and not reprocess:
        logger.info(f"File {out_file} exists and reprocess is False. Skipping...")
        return
    _, aws_df, aai_df, wsp_df, rev_df = process_session_audio(
        session_id, outbase, s3_session, datadir, snr=snr, noisefn=noisefn, s3_scratch_uri=s3_scratch_uri
    )
    hum_df = load_manual_transcript_words(session_id, datadir)
    hum_df["snr"] = 0
    df = pd.concat((hum_df, aws_df, aai_df, wsp_df, rev_df)).reset_index(drop=True)
    df["dataset"] = "ami"
    df["session_id"] = session_id
    df["snr"] = snr
    df["noisefile"] = noisefn.name
    df.to_feather(out_file)
    if num_snrs is not None:
        # this will not do the merge if the session for given noise file is not complete
        merge_session_files(outbase / session_id, num_snrs, noisefn.stem)


def make_noises(outpath, source_file, s3_session=None):
    """
    Make a series of noise files using different parts of the source file.

    For the AMI analysis, the source file was a 4-hour room recording of an instrumental music
    playlist. There is some excessive room noise in the first few minutes, so we start
    at 7 minutes into the file. We'll use this to mix with audio files that should be at most
    60 minutes so we'll create files that are 75 minutes long to be sure.

    Args:
        outpath: output directory for noise files
        source_file: local path or S3 URI (e.g. "s3://your-bucket/noise_source.mp3") of the source audio
        s3_session: boto3 session (only needed if source_file is an S3 URI)
    """
    logger = setup_logging()
    if source_file.startswith("s3://") and s3_session is None:
        raise ValueError(f"Source file {source_file} is an S3 URI but AWS credentials are not available. Please provide AWS credentials or use a local file.")
    DURATION = 75*60000  # msec
    START = 7*60000  # msec
    noise = load_audio([source_file], s3_session=s3_session).set_channels(1).set_frame_rate(16000)
    for idx, start_time in enumerate(range(START, int(noise.duration_seconds * 1000)-DURATION, DURATION)):
        end_time = start_time + DURATION
        noise_seg = noise[start_time:end_time]
        fn = Path(outpath) / f"noise{idx+1}.flac"
        print(f"{str(fn)}: start at {start_time//60000} minutes (duration {noise_seg.duration_seconds/60:0.1f} minutes)")
        noise_seg.export(fn, format="flac")
