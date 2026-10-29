# ASR Hallucination Detection

Tools for analyzing ASR hallucinations using time-aligned word-level transcripts. This code accompanies the paper *Evaluating Hallucination Characteristics of Automated Speech Recognition Systems* (Dougherty, Shafiei, Alti, Leininger & Staples). The paper compares how four ASR systems behave as background noise increases. LaTeX source and a PDF are in [`latex/`](latex/).

## Overview

The pipeline takes meetings from the [AMI corpus](https://groups.inf.ed.ac.uk/ami/corpus/), mixes in realistic clinic background noise (office room noise or instrumental music) at SNRs from +20 to −20 dB, transcribes every version with four ASR systems, and compares the results to AMI's human, word-timestamped transcripts.

This package provides:
- **Transcription** with WhisperX, AWS Transcribe, AssemblyAI, and Rev's Reverb, all saved in a common word-level format
- **Noise mixing at a target SNR**, plus Silero VAD → [docs/snr_mixing.md](docs/snr_mixing.md)
- **Time-constrained word alignment**, which avoids the spurious long-range matches that plain edit-distance alignment makes on noisy transcripts → [docs/time_constrained_alignment.md](docs/time_constrained_alignment.md)
- **Hallucination detection**: labels every ASR word as match / substitution / insertion / insertion-during-silence, as the basis for finding hallucinated sequences → [docs/hallucination_detection.md](docs/hallucination_detection.md)
- **Perplexity analysis**: language-model fluency of hallucinated sequences compared with length-matched human sequences → [docs/perplexity_analysis.md](docs/perplexity_analysis.md)
- **Analysis code** for the paper's figures, tables, and statistics

See [`docs/`](docs/README.md) for the pipeline overview, output file formats, configuration, and known limitations.

## Installation

```bash
# Clone the repository
git clone https://github.com/soundtrip-health/asr_hallucination.git
cd asr_hallucination

# Install core dependencies
pip install -e .

# Install optional transcription services as needed
pip install -e ".[whisperx]"     # For WhisperX
pip install -e ".[aws]"          # For AWS Transcribe
pip install -e ".[assemblyai]"   # For AssemblyAI
pip install -e ".[perplexity]"   # For the perplexity analysis (transformers)
pip install -e ".[all]"          # Install all optional dependencies

# For Reverb, see installation instructions below
```

## Quick Start

### Batch Transcription

Process AMI corpus sessions with multiple ASR systems:

```bash
# Credentials are read from ~/.env (override with --envfile):
#   AWS_ACCESS_KEY_ID=...                     # AWS Transcribe (optional)
#   AWS_SECRET_ACCESS_KEY=...
#   AWS_REGION=us-east-1                      # region of your scratch bucket
#   AWS_S3_SCRATCH_URI=s3://your-bucket/tmp   # temp storage for AWS Transcribe jobs
#   ASSEMBLYAI_API_KEY=...                    # AssemblyAI (optional)

# Run transcriptions on a subset of sessions
python scripts/run_transcriptions.py /path/to/ami/corpus \
    --outdir ./output \
    --session-types train \
    --sampfrac 0.1 \
    --njobs 4 \
    --noisefile ./data/noise1.flac \
    --snrs inf,20,10,0,-10
```

Each `(session, SNR)` pair is processed independently, and existing outputs are skipped, so you can safely re-run after an interruption. AWS and AssemblyAI are skipped if their credentials are missing. AWS also needs an S3 scratch prefix (`--s3-scratch` or `AWS_S3_SCRATCH_URI`). WhisperX and Reverb must be installed (a CUDA GPU is recommended). The noise recordings used in the paper are in `data/` (`noise1.flac` = music, `noise2.flac` = office). See [docs/README.md](docs/README.md) for the output files.

### Alignment and hallucination labeling

```python
import copy
import pandas as pd
from asr_hallucination.alignment import (
    get_normalizer, normalize_xscript, align_using_time, compute_wer,
    compute_hallucinations, find_hallucination_sequences,
)

df = pd.read_feather("out/ES2002a/xscript_noise2_snr+00.feather")
norm = get_normalizer()   # whisper-normalizer + the paper's filler-word list
transcripts = {
    t: {"word_segments": normalize_xscript(g.sort_values("start_time"), norm).to_dict("records")}
    for t, g in df.groupby("transcriber")
}
original = copy.deepcopy(transcripts)
aligned = align_using_time(transcripts, time_tolerance=0.44)
print(compute_wer(aligned))                    # S/D/I/hits/WER per ASR
ops = compute_hallucinations(aligned, original)  # per-word "M"/"S"/"I"/"H" labels per ASR
seqs = find_hallucination_sequences(aligned["wsp"]["word_segments"], ops["wsp"])  # >=3 errors, gaps <=1 s
```

### Analysis

`notebooks/analysis.py` is a percent-format (`# %%`) notebook that produces the paper's figures, LaTeX tables, and t-tests. Open it in VS Code or Jupyter (via jupytext), or run it as a script. Set `AMI_OUT_DIR` to your transcript output directory (default `/data/transcripts/ami/out`). Outputs, including the timestamp-jitter histograms and the `jitter_quantiles.tex` table, are written to `figures/`.

`notebooks/perplexity.py` (same format, same `AMI_OUT_DIR`) reproduces `perplexity_distributions.png`. It finds hallucination sequences at 0 dB music noise, scores them with BERT-large as in the paper, and compares them with length-matched human sequences. It needs the `perplexity` extra, and a GPU is recommended. See [docs/perplexity_analysis.md](docs/perplexity_analysis.md).

## Project Structure

```
asr_hallucination/
├── asr_hallucination/      # Main package
│   ├── transcription.py    # ASR system interfaces
│   ├── audio.py           # Audio processing utilities
│   ├── alignment.py       # Word alignment algorithms
│   ├── ami_utils.py       # AMI corpus utilities
│   ├── perplexity.py      # LM perplexity and length-matched sampling
│   ├── processing.py      # High-level processing functions
│   ├── s3_utils.py        # S3 helpers (AWS Transcribe I/O)
│   └── config.py          # Configuration and logging
├── scripts/
│   └── run_transcriptions.py  # CLI for batch transcription
├── notebooks/
│   ├── analysis.py        # Analysis and figure generation
│   └── perplexity.py      # Perplexity analysis (perplexity_distributions.png)
├── data/                  # AMI split IDs and noise recordings (noise1.flac, noise2.flac)
├── docs/                  # Detailed method documentation
├── figures/               # Generated figures and tables
└── latex/                 # Paper source and PDF
```

## Requirements

- Python 3.10+
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
  - The above directory structure is the recommended AMI corpus directory structure and is what you will get by using the included download script.
- **Output directory**: The `out/` directory will be created automatically in your `datadir` (or `--outdir` if specified) - it stores all processed transcripts and intermediate files like:
  - `{session_id}/xscript_{transcriber}_{noise}_snr{snr}.json` - Individual transcript files
  - `{session_id}/audio_{noise}_snr{snr}.wav` - Noise-mixed audio files
  - `{session_id}/vad_{noise}_snr{snr}.feather` - VAD probability files
  - `{session_id}/xscript_{noise}_snr{snr}.feather` and `xscript_{noise}.feather` - Combined transcripts (all ASRs + human)

  SNRs are formatted with a sign and two digits (`+20`, `-05`, `+inf`). See [docs/README.md](docs/README.md#output-layout) for details.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{dougherty_asr_hallucination,
  title  = {Evaluating Hallucination Characteristics of Automated Speech Recognition Systems},
  author = {Dougherty, Robert F. and Shafiei, Mario and Alti, Merve and Leininger, Carly and Staples, Patrick},
  note   = {Code: https://github.com/soundtrip-health/asr_hallucination}
}
```

*(This entry will be updated with full publication details once the paper is published.)*

## License

MIT License - see LICENSE file for details.
