# Documentation

Detailed documentation for the `asr_hallucination` package. For installation and
a quick start, see the [top-level README](../README.md).

| Document | Contents |
|----------|----------|
| [Noise mixing and SNR](snr_mixing.md) | How background noise is mixed into AMI audio at a target SNR, the math, and how to interpret the SNR values |
| [Time-constrained alignment](time_constrained_alignment.md) | Why plain edit-distance alignment misleads on noisy audio, plus the two time-aware aligners, their inputs/outputs, and tolerance choice |
| [Hallucination detection](hallucination_detection.md) | The paper's hallucination definition, per-word M/S/I/H labeling, grouping into sequences, and reference-free heuristics |

## Pipeline overview

```
AMI Mix-Lapel.wav ──┐
                    ├─ mix_audio() at each SNR ──► audio_{noise}_snr{SNR}.wav
noise recording ────┘                                   │
                                                        ├─ Silero VAD ──► vad_{noise}_snr{SNR}.feather
                                                        │
                                                        └─ AWS / AssemblyAI / WhisperX / Reverb
                                                                 │
                                                                 ▼
                                               xscript_{asr}_{noise}_snr{SNR}.json
                                                                 │
AMI words XML (human reference) ─────────────────────────────────┤
                                                                 ▼
                                    xscript_{noise}_snr{SNR}.feather  (all ASRs + human, one SNR)
                                                                 │  (once every SNR is done)
                                                                 ▼
                                    xscript_{noise}.feather           (all SNRs for the session)
                                                                 │
                         ┌───────────────────────────────────────┴───────────────────────┐
                         ▼                                                               ▼
        jiwer word alignment → WER, S/D/I                    normalize_xscript → align_using_time
        (notebooks/analysis.py; paper figures/tables)         → compute_wer / compute_hallucinations
                                                              → find_hallucination_sequences
```

### Stage 1: transcription (`scripts/run_transcriptions.py`)

For each `(session, SNR)` pair, `processing.process_session()` runs in a
joblib worker. It:

1. Finds the session audio (`ami_utils.get_audiofile`: `Mix-Lapel`, falling
   back to `Mix-Headset`).
2. For finite SNRs, mixes in the noise ([details](snr_mixing.md)).
3. Computes Silero VAD probabilities.
4. Transcribes with each available ASR. AWS and AssemblyAI are skipped with
   a warning if their credentials are missing. AWS is also skipped if no S3
   scratch location is set (`--s3-scratch` / `AWS_S3_SCRATCH_URI`). AWS
   Transcribe reads its input from S3 and writes its result there, so each
   job uploads the audio to that prefix and deletes it afterwards. WhisperX (CUDA) and Reverb are always
   attempted, and the job fails if either is not installed. The AWS job is submitted first and collected last, so it runs
   concurrently with the local models.
5. Loads the human reference from the AMI word XML files and concatenates it
   with all ASR transcripts into a single long-format feather file.

Every stage skips files that already exist, so an interrupted run can be
restarted.

#### Output layout

All outputs go in `{outdir}/{session_id}/`, where `outdir` defaults to
`{datadir}/out`. SNRs are formatted as `+20`, `-05`, `+inf`, and so on.

| File | Contents |
|------|----------|
| `audio_{noise}_snr{SNR}.wav` | Noise-mixed audio (finite SNRs only) |
| `vad_{noise}_snr{SNR}.feather` | `time` (ms), `vad_prob` |
| `xscript_{asr}_{noise}_snr{SNR}.json` | Raw per-ASR result, normalized to a `word_segments` list (AWS keeps its native JSON) |
| `xscript_aai_{noise}_snr{SNR}_raw.json` | Full AssemblyAI response (diarization, entities, etc.) |
| `xscript_{noise}_snr{SNR}.feather` | Combined words for all transcribers at one SNR |
| `xscript_{noise}.feather` | All SNRs merged (written once every SNR for the session exists) |

For SNR = +∞ the per-ASR, audio, and VAD files use `none` as the noise name.
The combined feather files use the noise file's stem (e.g. `noise1`).

`asr` codes: `hum` (human reference), `aws` (AWS Transcribe), `aai`
(AssemblyAI), `wsp` (WhisperX large-v3), `rev` (Rev Reverb).

#### Combined feather schema

| Column | Description |
|--------|-------------|
| `transcriber` | `hum`, `aws`, `aai`, `wsp`, `rev` |
| `word` | raw token |
| `start_time`, `stop_time` | seconds |
| `confidence` | ASR confidence (`1.0` for human; `NaN` for AWS punctuation) |
| `speaker` | AMI speaker letter for `hum`; `UNK` for ASRs |
| `sequence` | word index within that transcript |
| `session_id`, `snr`, `noisefile`, `dataset` | identifiers |

### Stage 2: analysis (`notebooks/analysis.py`)

A percent-format (`# %%`) notebook meant to be run cell by cell in VS Code or
Jupyter (via jupytext). It loads the merged feather files for the AMI `train`
split, computes jiwer word-alignment metrics, runs paired t-tests with
Benjamini–Yekutieli FDR correction, and writes the figures and LaTeX tables in
`figures/`. It also computes word-timestamp differences ("jitter") between
matched human and ASR words, along with the length of each run of
consecutive matches, and writes `jitter_hist.png`, `jitter_hist_neg.png`, and
`jitter_quantiles.tex`. Set the data location with the `AMI_OUT_DIR`
environment variable (default `/data/transcripts/ami/out`).

### Configuration

| Setting | CLI flag | Environment variable |
|---------|----------|----------------------|
| AWS credentials | — | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| AWS region (must match the scratch bucket) | `--aws-region` | `AWS_REGION` / `AWS_DEFAULT_REGION` |
| S3 scratch prefix for AWS Transcribe, e.g. `s3://your-bucket/tmp` | `--s3-scratch` | `AWS_S3_SCRATCH_URI` |
| AssemblyAI key | — | `ASSEMBLYAI_API_KEY` |

The script loads environment variables from `~/.env` (`--envfile` to change).

### AMI splits

`data/split_ids.json` holds the train/validation/test session lists used by
the [Hugging Face AMI dataset](https://huggingface.co/datasets/edinburghcstr/ami).
Read it with `ami_utils.get_splits()`.

## Known limitations

- **Time-constrained alignment is greedy**, not globally optimal (see
  [the alignment doc](time_constrained_alignment.md#worked-example)).
- **The perplexity analysis** behind `perplexity_distributions.png` is not
  included.
- **WhisperX and Reverb are required** by `process_session_audio`. A
  session's job fails if either is not installed.
