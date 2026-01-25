"""Transcription functions for multiple ASR systems."""

import os
import pathlib
import time
import numpy as np

from .config import ROOT_DIR, setup_logging, get_uuid
from .s3_utils import parse_s3_uri

# Optional imports for transcription services
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    import assemblyai as aai
    AAI_AVAILABLE = True
except ImportError:
    AAI_AVAILABLE = False

try:
    import torch
    # Monkey-patch torch.load to force weights_only=False for PyTorch 2.6+
    # This fixes compatibility with whisperx/pyannote models that use omegaconf.ListConfig
    # lightning_fabric explicitly passes weights_only=True, so we need to override it
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False

try:
    # The following is rev.com's forked version of wenet to run their open-source reverb model.
    # See https://github.com/revdotcom/reverb for install instructions.
    import wenet
    REVERB_AVAILABLE = True
except ImportError:
    REVERB_AVAILABLE = False


def whisperx_transcribe_audio(
    file_path: pathlib.Path,
    vad_options: dict | None = None,
    model_name: str = "large-v3",
    download_root: str | None = None,
) -> dict:
    """Transcribes an audio file using WhisperX and returns the result.

    NOTE: Running this on some systems may cause "kernel restarting" as soon as the translate job starts.
    Downgrading ctranslate2 to 4.4.0 fixed the issue (pip install ctranslate2==4.4.0)
    See https://github.com/SYSTRAN/faster-whisper/issues/752.

    Args:
        file_path: path to audio file
        vad_options: VAD options dict (default: {"vad_onset": 0.370, "vad_offset": 0.35})
        model_name: WhisperX model name
        download_root: directory to download models to

    Returns:
        dict with keys: segments, word_segments
    """
    if not WHISPERX_AVAILABLE:
        raise ImportError("whisperx is not installed. Install with: pip install whisperx")

    device = "cuda"
    batch_size = 4  # reduce if low on GPU mem
    vad_options = {"vad_onset": 0.370, "vad_offset": 0.35} if vad_options is None else vad_options
    asr_options = {"multilingual": False, "hotwords": None}
    download_root = ROOT_DIR / "models" if download_root is None else pathlib.Path(download_root)
    model = whisperx.load_model(
        model_name, device, download_root=download_root, language="en", vad_options=vad_options, asr_options=asr_options
    )
    model_align, metadata = whisperx.load_align_model(language_code="en", device=device)
    audio = whisperx.load_audio(file_path)
    result = model.transcribe(audio, batch_size=batch_size)
    result = whisperx.align(result["segments"], model_align, metadata, audio, device, return_char_alignments=False)
    return result


def load_rev_model(model_path=None, gpu=-1):
    """
    Load the rev.com open-source reverb transcription model.

    The model must be pre-downloaded and available at model_path. If model_path is not specified,
    it will be set to ROOT_DIR/reverb-asr.

    We re-implement the reverb load_model function as it doesn't expose some useful options (like gpu).
    Unfortunately, it doesn't seem to use the gpu even if that option is set.

    Args:
        model_path: path to model directory containing config.yaml and checkpoint .pt file
        gpu: GPU device ID (-1 for CPU)

    Returns:
        wenet.ReverbASR instance

    Raises:
        ImportError: If wenet (reverb) is not installed
        FileNotFoundError: If the model files are not found at the specified path
    """
    if not REVERB_AVAILABLE:
        raise ImportError("wenet (reverb) is not installed. See README.md for installation instructions.")

    logger = setup_logging()
    if model_path is None:
        model_path = ROOT_DIR / "reverb-asr"
    else:
        model_path = pathlib.Path(model_path)
    
    if not (model_path / "config.yaml").exists():
        raise FileNotFoundError(
            f"Reverb model not found at {model_path}. "
            "Please download the model following the instructions in README.md. "
            "The model should be downloaded from Hugging Face (Revai/reverb-asr) "
            "and placed in the specified directory."
        )
    
    config_path = (model_path / "config.yaml").resolve()
    checkpoint_files = list(model_path.glob("*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint file (.pt) found in {model_path}. "
            "Please ensure the model has been downloaded correctly."
        )
    checkpoint_path = checkpoint_files[0].resolve()
    logger.info(f"Loading the model with {config_path = } and {checkpoint_path = }")
    return wenet.ReverbASR(str(config_path), str(checkpoint_path))


def transcribe_rev(audio_file, verbatimicity=1.0, reverb_model=None):
    """
    Transcribe using Rev.com's open-source model reverb.
    https://github.com/revdotcom/reverb

    Note: The default mode (ctc_prefix_beam_search) doesn't return confidence scores
    and seems to generate a lower-quality transcript, so we use attention_rescoring instead.

    Args:
        audio_file: path to audio file
        verbatimicity: verbatimicity level (0.0-1.0)
        reverb_model: pre-loaded reverb model (will load if None)

    Returns:
        dict with key: word_segments
    """
    if not REVERB_AVAILABLE:
        raise ImportError("wenet (reverb) is not installed. See https://github.com/revdotcom/reverb for install instructions.")

    logger = setup_logging()
    if reverb_model is None:
        reverb_model = load_rev_model()
    MODE = "attention_rescoring"  # "ctc_prefix_beam_search", "attention_rescoring"
    try:
        output = reverb_model.transcribe(audio_file, MODE, format="ctm", verbatimicity=verbatimicity)
    except AssertionError as e:
        # I think this only happens for very low snr levels?
        logger.error(f"failed to transcribe {audio_file}: {e}")
        return {"word_segments": []}
    words = []
    for ln in output.split("\n"):
        try:
            parts = ln.split()
            filename, channel, start, duration, word, conf = parts
            start, duration, conf = float(start), float(duration), float(conf)
            # Currently not doing diarization, but capture speaker in case we enable this later.
            words.append({"speaker": channel, "start": start, "end": start + duration, "word": word, "score": conf})
        except Exception as e:
            logger.error(f"failed to parse {ln}: {e}")
    # Match the whisper output format
    return {"word_segments": words}


def assemblyai_transcribe_audio(
    audio_path: str, speakers_expected: int | None = None, language: str = "en", boost_words: list[str] = []
) -> dict:
    """Transcribes an audio file using AssemblyAI and returns the result.

    Args:
        audio_path: path to audio file
        speakers_expected: expected number of speakers for diarization
        language: language code
        boost_words: list of words to boost in recognition

    Returns:
        tuple of (results_dict, raw_json_response)
        results_dict has keys: segments, word_segments
    """
    if not AAI_AVAILABLE:
        raise ImportError("assemblyai is not installed. Install with: pip install assemblyai")

    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables. Please set it in your .env file or environment.")
    
    aai.settings.api_key = api_key

    config = aai.TranscriptionConfig(
        speaker_labels=True,
        entity_detection=True,
        disfluencies=True,
        word_boost=boost_words,
        boost_param="default",
        language_code=language,
        speakers_expected=speakers_expected,
    )

    transcriber = aai.Transcriber()

    with open(audio_path, "rb") as file:
        transcript = transcriber.transcribe(file, config)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(transcript.error)
    json_response = transcript.json_response

    # to be compatible with whisperx
    results = {
        "segments": [
            {
                "start": item["start"] / 1000,
                "end": item["end"] / 1000,
                "text": item["text"],
                "words": [],
            }
            for item in json_response["utterances"]
        ],
        "word_segments": [
            {
                "word": item["text"],
                "start": item["start"] / 1000,
                "end": item["end"] / 1000,
                "score": item["confidence"],
                "speaker": item["speaker"],
            }
            for item in json_response["words"]
        ],
    }

    return results, transcript.json_response


def aws_transcribe_audio(s3_session, audio_uri: str, result_uri: str, languages: None, job_name=None):
    """Transcribes an audio file using AWS Transcribe, saving results in S3.

    Audio_file must be in S3. If you know which languages might be present, specify as a list. E.g.,
    languages=["en-US", "en-GB"]

    Args:
        s3_session: boto3 session
        audio_uri: S3 URI of audio file
        result_uri: S3 URI for results
        languages: list of language codes or None
        job_name: job name (will generate unique ID if None)

    Returns:
        tuple of (status, transcribe_client)
    """
    if not AWS_AVAILABLE:
        raise ImportError("boto3 is not installed. Install with: pip install boto3")

    job_name = get_uuid() if job_name is None else job_name
    transcribe_client = s3_session.client("transcribe")
    bucket, key = parse_s3_uri(result_uri)
    args = {
        "TranscriptionJobName": job_name,
        "Media": {"MediaFileUri": audio_uri},
        "MediaFormat": audio_uri.split(".")[-1],
        "OutputBucketName": bucket,
        "OutputKey": key,
    }
    if languages:
        if len(languages) == 1:
            args["LanguageCode"] = languages[0]
        else:
            args["IdentifyMultipleLanguages"] = True
            args["LanguageOptions"] = languages

    status = transcribe_client.start_transcription_job(**args)
    return status, transcribe_client


def check_aws_job_status(job_name: str, client):
    """Checks the status of an AWS transcription job.

    Args:
        job_name: AWS Transcribe job name
        client: boto3 transcribe client

    Returns:
        job status dict
    """
    if not AWS_AVAILABLE:
        raise ImportError("boto3 is not installed. Install with: pip install boto3")

    return client.get_transcription_job(TranscriptionJobName=job_name)
