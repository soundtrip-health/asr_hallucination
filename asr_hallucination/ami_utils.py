"""Utilities specific to the AMI corpus."""

import json
import pathlib
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path

from .config import setup_logging


def get_splits():
    """
    Use the same splits as https://huggingface.co/datasets/edinburghcstr/ami/blob/main/ami.py.

    Returns:
        dict with keys: train, validation, test
    """
    curdir = pathlib.Path(__file__).parent.parent
    with open(curdir / "data" / "split_ids.json") as fp:
        splits = json.load(fp)
    return splits


def get_session_and_speaker(filename):
    """Extract session ID and speaker from AMI filename.

    Args:
        filename: Path object for AMI file (e.g., EN2001a.A.words.xml)

    Returns:
        tuple of (session_id, speaker)
    """
    # EN2001a.A.words.xml -> session="EN2001a", speaker="A"
    parts = filename.stem.split('.')
    return parts[0], parts[1]


def get_ami_sessions(basedir):
    """Get list of session IDs from AMI corpus.

    Args:
        basedir: base directory containing AMI corpus

    Returns:
        list of session IDs
    """
    path = Path(basedir) / "transcripts" / "manual" / "words"
    sessions = []
    for xml_file in path.glob("*.A.words.xml"):
        session_id, _ = get_session_and_speaker(xml_file)
        sessions.append(session_id)
    return sessions


def parse_ami_xml(xml_file, speaker):
    """
    Parse an AMI XML file and return a list of word segments.

    There are other items in the XML file in addition to the words (tag "w"). These include "vocalsound",
    which has no information about the sound so it's not very useful, "gap", which I have
    no idea what it specifies (the start/end timestamps are identical), and "disfmarker",
    which I'm guessing a disfluency marker, but not sure. Unfortunately these tags are not
    documented anywhere that I could find. But for our purposes, we just need the words
    so that's all that this code will parse out.

    Note that the sequence returned is just the word sequence for this speaker!

    Args:
        xml_file: path to AMI XML file
        speaker: speaker ID

    Returns:
        list of word dicts with keys: speaker, word, start_time, stop_time, confidence
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    words = []
    for word_elem in root.findall('.//w'):
        # Extract timing information from attributes
        start_time = float(word_elem.get('starttime', -1))
        end_time = float(word_elem.get('endtime', -1))

        # Convert times from seconds to milliseconds to match whisperx format
        word_data = {
            "speaker": speaker,
            "word": word_elem.text or "",
            "start_time": start_time if start_time >= 0 else np.nan,
            "stop_time": end_time if end_time >= 0 else np.nan,
            "confidence": 1.0,  # AMI transcripts are manual, so confidence is 1.0
        }
        words.append(word_data)

    return words


def get_audiofile(session_id, basedir):
    """Get audio file path for an AMI session.

    We prefer the "Mix-Lapel" audio, but will use Mix-Headset if lapel isn't available.

    Args:
        session_id: AMI session ID
        basedir: base directory containing AMI corpus

    Returns:
        Path to audio file or None if not found
    """
    logger = setup_logging()
    base = Path(basedir)
    fn = base / session_id / "audio" / f"{session_id}.Mix-Lapel.wav"
    if not fn.is_file():
        fn = base / session_id / "audio" / f"{session_id}.Mix-Headset.wav"
        if not fn.is_file():
            logger.error(f"Audio file not found: {fn}")
            return None
    return fn


def load_transcript_words(path, transcriber):
    """Load transcription words from a JSON file.

    Note: AWS provides rich language code info in audio_segments indicating which language
    (e.g., dialect of English) it detected in each chunk of speech. The audio_segments may
    also give insight into the AWS VAD model.

    Args:
        path: path to transcript JSON file
        transcriber: transcriber name ("aws", "aai", "wsp", "rev")

    Returns:
        DataFrame with columns: sequence, transcriber, word, start_time, stop_time, confidence, speaker
    """
    logger = setup_logging()
    if not path.is_file():
        logger.debug(f"File {path} does not exist.")
        return None

    if transcriber == "aws":
        with open(path) as fp:
            tmp = json.load(fp)
        words = [
            {
                "sequence": idx,
                "transcriber": transcriber,
                "word": data["alternatives"][0]["content"],
                "start_time": float(data.get("start_time", np.nan)),
                "stop_time": float(data.get("end_time", np.nan)),
                # Confidence for punctuation is 0; but it should be NaN so it doesn't bias downstream analyses.
                "confidence": float(data["alternatives"][0]["confidence"]) if data["type"] == "pronunciation" else np.nan,
                "speaker": "UNK",
            }
            for idx, data in enumerate(tmp["results"]["items"])
        ]
        df = pd.DataFrame(words)
        # Fill the missing punctuation start_time. Start by forward-filling end_times so that
        # a punctuation row has the previous non-punctuation row's end time. Then just copy end
        # time over to the start times that are missing. This has the effect of filling a missing
        # start time with the previous row's end time (or the row before that, if the previous row
        # is also missing the time.)
        if len(df) > 0:
            df.stop_time = df.stop_time.ffill()
            idx = df.start_time.isna()
            df.loc[idx, 'start_time'] = df.stop_time[idx]
    elif transcriber in ["aai", "wsp", "rev"]:
        with open(path) as fp:
            tmp = json.load(fp)
        words = [
            {
                "sequence": idx,
                "transcriber": transcriber,
                "word": data["word"],
                "start_time": float(data.get("start", np.nan)),
                "stop_time": float(data.get("end", np.nan)),
                "confidence": float(data.get("score", np.nan)),
                "speaker": "UNK",
            }
            for idx, data in enumerate(tmp["word_segments"])
        ]
        df = pd.DataFrame(words)
    else:
        logger.error(f"Unknown transcriber '{transcriber}'.")
        df = None
    return df


def load_manual_transcript_words(session_id, basedir):
    """Load manual (human) transcript words for an AMI session.

    Args:
        session_id: AMI session ID
        basedir: base directory containing AMI corpus

    Returns:
        DataFrame with columns: speaker, word, start_time, stop_time, confidence, sequence, transcriber
    """
    transcript_dir = Path(basedir) / "transcripts" / "manual" / "words"
    session_data = []
    for xml_file in transcript_dir.glob(f"{session_id}.*.words.xml"):
        _, speaker = get_session_and_speaker(xml_file)
        session_data += parse_ami_xml(xml_file, speaker)
    df = pd.DataFrame(session_data).sort_values(["start_time", "stop_time"]).reset_index(drop=True)
    df["sequence"] = df.index
    df["transcriber"] = "hum"
    return df


def load_ami_transcript(session_id, transcriber, snr, basedir, noise=None, outdir=None):
    """Load AMI transcript for a specific session, transcriber, noise, and SNR.

    Args:
        session_id: AMI session ID
        transcriber: transcriber name ("hum", "aws", "aai", "wsp", "rev")
        snr: signal-to-noise ratio (float("inf") for the original audio)
        basedir: base directory containing AMI corpus
        noise: noise file stem (e.g. "noise1"); ignored (and "none" is used) when snr is inf
        outdir: directory with the processed transcripts (default: basedir/out)

    Returns:
        DataFrame with transcript words, or None if the transcript file doesn't exist
    """
    if transcriber == "hum":
        df = load_manual_transcript_words(session_id, basedir)
        df["transcriber"] = "hum"
        df["snr"] = float("inf")
        df["dataset"] = "ami"
    else:
        if snr == float("inf"):
            noise = "none"
        elif noise is None:
            raise ValueError("noise must be specified for finite SNRs.")
        outdir = Path(basedir) / "out" if outdir is None else Path(outdir)
        path = outdir / session_id / f"xscript_{transcriber}_{noise}_snr{snr:+03.0f}.json"
        df = load_transcript_words(path, transcriber)
        if df is not None:
            df["session_id"] = session_id
            df["snr"] = snr
            df["noise"] = noise
            df["dataset"] = "ami"
    return df
