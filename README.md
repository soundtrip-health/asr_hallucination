# ASR Hallucination Detection

Tools for analyzing ASR hallucinations using time-aligned word-level transcripts. This code accompanies a scientific paper analyzing hallucination behavior across multiple commercial and open-source ASR systems.

## Overview

This package provides:
- Transcription with multiple ASR systems (WhisperX, AWS Transcribe, AssemblyAI, Reverb)
- Audio processing utilities (mixing with noise at various SNR levels, VAD)
- Time-constrained word alignment for computing WER and detecting hallucinations
- Analysis tools for the AMI corpus with configurable noise and SNR levels

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/asr-hallucination.git
cd asr-hallucination

# Install core dependencies
pip install -e .

# Install optional transcription services as needed
pip install -e ".[whisperx]"     # For WhisperX
pip install -e ".[aws]"          # For AWS Transcribe
pip install -e ".[assemblyai]"   # For AssemblyAI
pip install -e ".[all]"          # Install all optional dependencies

# For Reverb, follow instructions at https://github.com/revdotcom/reverb
```

## Quick Start

### Batch Transcription

Process AMI corpus sessions with multiple ASR systems:

```bash
# Set up environment variables (AWS credentials, AssemblyAI API key, etc.)
cp .env.example ~/.env
# Edit ~/.env with your credentials

# Run transcriptions on a subset of sessions
python scripts/run_transcriptions.py /path/to/ami/corpus \
    --outdir ./output \
    --session-types train \
    --sampfrac 0.1 \
    --njobs 4
```

### Analysis

Generate figures and tables for the paper:

```bash
cd notebooks
python analysis.py
```

Results will be saved in the `figures/` directory.

## Project Structure

```
asr_hallucination/
├── asr_hallucination/      # Main package
│   ├── transcription.py    # ASR system interfaces
│   ├── audio.py           # Audio processing utilities
│   ├── alignment.py       # Word alignment algorithms
│   ├── ami_utils.py       # AMI corpus utilities
│   ├── processing.py      # High-level processing functions
│   └── config.py          # Configuration and logging
├── scripts/
│   └── run_transcriptions.py  # CLI for batch transcription
├── notebooks/
│   └── analysis.py        # Analysis and figure generation
├── data/                  # Data files (splits, noise samples)
└── figures/               # Generated figures and tables
```

## Requirements

- Python 3.9+
- FFmpeg (for audio processing with pydub)
- CUDA-capable GPU (recommended for WhisperX and Reverb)

### Optional Services

- **AWS Transcribe**: Requires AWS credentials in environment variables
- **AssemblyAI**: Requires API key in `ASSEMBLYAI_API_KEY` environment variable
- **WhisperX**: Requires installation via `pip install whisperx`
- **Reverb**: Requires special installation, see https://github.com/revdotcom/reverb

## AMI Corpus

This code is designed to work with the AMI Meeting Corpus. Download instructions:
```bash
# Manual transcripts and audio files
wget -r -np -nH --cut-dirs=3 http://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/
wget -r -np -nH --cut-dirs=3 http://groups.inf.ed.ac.uk/ami/AMICorpusMirror/annotation/
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{yourpaper2024,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.
