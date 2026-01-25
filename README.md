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

# For Reverb, see installation instructions below
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
- **Reverb**: Requires special installation (see below)

#### Reverb Installation

Reverb is Rev.com's semi-open-source ASR model. To use it:

1. **Install the reverb package**:
   ```bash
   git clone https://github.com/revdotcom/reverb.git
   cd reverb
   pip install .
   ```

2. **Set up Hugging Face access**:
   - Create a Hugging Face account at https://huggingface.co
   - Accept the model license at https://huggingface.co/Revai/reverb-asr
   - Install Git LFS: `git lfs install`
   - Log in to Hugging Face CLI: `hf login` (or `huggingface-cli login` if using older version)

3. **Download the model**:
   ```bash
   # Option 1: Using git (recommended)
   git clone https://huggingface.co/Revai/reverb-asr /path/to/asr-hallucination/reverb-asr
   
   # Option 2: Using huggingface_hub Python library
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Revai/reverb-asr', local_dir='reverb-asr')"
   ```

   The model will be downloaded to `reverb-asr/` in your project root by default, or you can specify a custom path when calling `load_rev_model()`.

**Note**: The model files are large (~2-3 GB) and require Git LFS. Make sure you have sufficient disk space and bandwidth.

## Security Considerations

### PyTorch `weights_only` Monkey Patch

This codebase includes a monkey patch in `asr_hallucination/transcription.py` that forces `torch.load(weights_only=False)` for compatibility with PyTorch 2.6+ and the WhisperX/pyannote models.

**What it does**: The patch overrides PyTorch's `torch.load` function to always set `weights_only=False`, even when libraries like `lightning_fabric` explicitly pass `weights_only=True`.

**Why it's necessary**: PyTorch 2.6 changed the default value of `weights_only` from `False` to `True` for security reasons. However, the WhisperX library (specifically its dependency on pyannote) loads model checkpoints that contain `omegaconf.listconfig.ListConfig` objects, which are not allowed when `weights_only=True`. Without this patch, WhisperX transcription would fail with an unpickling error.

**Security risk**: Setting `weights_only=False` allows arbitrary Python code execution when loading model checkpoints. This means that if a malicious checkpoint file is loaded, it could execute arbitrary code on your system.

**Why it's acceptable here**: 
- We only load models from trusted sources (official WhisperX/pyannote models from Hugging Face)
- The models are downloaded from reputable sources and verified repositories
- This is a research codebase where the models are known and trusted

**Recommendation**: Only use this codebase with models from trusted sources. Do not load arbitrary checkpoint files from untrusted sources. If you need to load untrusted checkpoints, consider removing the monkey patch and using PyTorch's `torch.serialization.add_safe_globals()` API instead (though this may require modifications to the underlying libraries).

## AMI Corpus

This code is designed to work with the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/). 

### Download Instructions

You can use the [AMI download tool](https://groups.inf.ed.ac.uk/ami/download/) to select which data you want. For convenience, we include a script that downloads the manual transcripts and lapel and headset mix audio files for all sessions.
```bash
./scripts/get_ami_corpus.sh
```

### Directory Structure Setup

After downloading the AMI corpus, make sure that your data directory is structured as follows:

```
/path/to/amicorpus/
├── transcripts/
│   └── manual/
│       └── words/
│           ├── EN2001a.A.words.xml
│           ├── EN2001a.B.words.xml
│           ├── EN2001a.C.words.xml
│           ├── EN2001a.D.words.xml
│           ├── EN2002a.A.words.xml
│           └── ... (one XML file per speaker per session)
├── EN2001a/
│   └── audio/
│       ├── EN2001a.Mix-Lapel.wav  (this is our preferred audio format)
│       └── EN2001a.Mix-Headset.wav  (fallback if Mix-Lapel not available)
├── EN2002a/
│   └── audio/
│       └── ...
└── ... (one folder per session)
```

**Important Notes:**
- **AMI corpus directory structure**:
  - The above director structure is the recommended AMI corpus directory structure and is what you will get by using the included download script.
- **Output directory**: The `out/` directory will be created automatically in your `datadir` (or `--outdir` if specified) - it stores all processed transcripts and intermediate files like:
  - `{session_id}/xscript_{transcriber}_snr{snr}.json` - Individual transcript files
  - `{session_id}/audio_{noise}_snr{snr}.wav` - Noise-mixed audio files
  - `{session_id}/vad_{noise}_snr{snr}.feather` - VAD probability files

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
