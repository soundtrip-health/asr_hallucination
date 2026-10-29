#!/usr/bin/env python3
"""CLI script for batch transcription of AMI corpus sessions."""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import joblib

# Add parent directory to path to import package
sys.path.insert(0, str(Path(__file__).parent.parent))

from asr_hallucination.config import setup_logging
from asr_hallucination.ami_utils import get_ami_sessions, get_splits
from asr_hallucination.processing import process_session

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False


DEFAULT_ENV = "~/.env"
DEFAULT_SESSION_TYPES = ",".join(["train", "validation", "test"])
DEFAULT_NOISEFILE = "./data/noise1.flac"
DEFAULT_SNRS = ",".join([str(i) for i in [np.inf, 20, 15, 10, 5, 0, -5, -10, -15, -20]])
DEFAULT_NJOBS = max(1, joblib.cpu_count() // 2)
S3_SCRATCH_ENV = "AWS_S3_SCRATCH_URI"


def main():
    parser = argparse.ArgumentParser(
        description='Process AMI corpus audio files for a specified set of SNR values. '
                    'The resulting transcripts and all the intermediary files are stored in outdir. '
                    'If a transcript exists there, it will not be regenerated. Note that you need AWS '
                    'credentials to download the noise file from S3 and to run AWS transcribe. These '
                    'are expected to be in the envfile.')
    parser.add_argument('datadir', type=str, help='Base directory containing AMI corpus data (see README for details)')
    parser.add_argument('--outdir', type=str, help='Where to put the processed files. Default is datadir/out.')
    parser.add_argument('--envfile', type=str, default=DEFAULT_ENV, help=f'Env file path; default is {DEFAULT_ENV}.')
    parser.add_argument('--noisefile', type=str, default=DEFAULT_NOISEFILE, help=f'Noise file; default is {DEFAULT_NOISEFILE}.')
    parser.add_argument('--sampfrac', type=float, help=f'Process a random sample this fraction of the total (seed is fixed).')
    parser.add_argument('--njobs', type=int, default=DEFAULT_NJOBS,
                        help=f'Number of jobs to run in parallel. Default is num cpus // 2 ({DEFAULT_NJOBS}).')
    parser.add_argument('--snrs', type=str, default=DEFAULT_SNRS,
                       help=f'comma-separated list of SNR values to process (default: {DEFAULT_SNRS})')
    parser.add_argument('--s3-scratch', type=str,
                        help=f'S3 prefix for temporary AWS Transcribe files, e.g. s3://your-bucket/tmp '
                             f'(default: ${S3_SCRATCH_ENV}). AWS Transcribe is skipped if not set.')
    parser.add_argument('--aws-region', type=str,
                        help='AWS region for Transcribe/S3 (default: $AWS_REGION, $AWS_DEFAULT_REGION, or your AWS config). '
                             'Should match the region of the scratch bucket.')
    parser.add_argument("--session-types", type=str, default=DEFAULT_SESSION_TYPES,
                        help=f"comma-separated list of session types to process (default: {DEFAULT_SESSION_TYPES})")
    args = parser.parse_args()

    load_dotenv(Path(args.envfile).expanduser(), override=True)

    logger = setup_logging()

    if not AWS_AVAILABLE:
        logger.warning("boto3 is not installed. AWS Transcribe will not be available.")
        s3_session = None
    else:
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if not aws_access_key or not aws_secret_key:
            logger.warning("AWS credentials not found in environment. AWS Transcribe and S3 operations will be skipped.")
            s3_session = None
        else:
            region = args.aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            s3_session = boto3.Session(
                aws_access_key,
                aws_secret_key,
                region_name=region
            )

    s3_scratch = args.s3_scratch or os.environ.get(S3_SCRATCH_ENV)
    if s3_session is not None and not s3_scratch:
        logger.warning(f"No S3 scratch location (--s3-scratch or ${S3_SCRATCH_ENV}). AWS Transcribe will be skipped.")

    datadir = Path(args.datadir)
    outbase = Path(args.outdir) if args.outdir else datadir / "out"

    if args.noisefile:
        noisefn = Path(args.noisefile)
    else:
        noisefn = None

    logger.info(f"Processing AMI corpus {datadir=}, {outbase=}, {noisefn=}, {args.njobs=}, {args.snrs=}...")

    sessions = get_ami_sessions(datadir)

    if args.session_types:
        session_types_to_process = args.session_types.split(",")
        all_session_types = get_splits()
        sessions_to_process = []
        for session_type in session_types_to_process:
            sessions_to_process.extend(all_session_types[session_type])
        sessions = [session for session in sessions if session in sessions_to_process]

    nsessions = len(sessions)

    logger.info(f"Found {nsessions} sessions in {str(datadir)}.")
    if args.sampfrac:
        # Set random seed for reproducibility and select sample of sessions
        np.random.seed(42)
        n_samples = min(nsessions, max(1, int(round(nsessions * args.sampfrac))))
        sessions = np.random.choice(sessions, size=n_samples, replace=False).tolist()
        logger.info(f"Selected a random sample of {n_samples} sessions for processing.")

    snrs = [float(x) for x in args.snrs.split(",")]
    n = len(sessions) * len(snrs)

    # Run parallel transcription jobs
    results = joblib.Parallel(n_jobs=args.njobs, verbose=1)(
        joblib.delayed(process_session)(
            session_id, snr, outbase, s3_session, noisefn, datadir, len(snrs), s3_scratch_uri=s3_scratch
        )
        for session_id in sessions
        for snr in snrs
    )

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
