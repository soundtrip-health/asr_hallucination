# Perplexity analysis

Code: [`asr_hallucination/perplexity.py`](../asr_hallucination/perplexity.py) and
the notebook [`notebooks/perplexity.py`](../notebooks/perplexity.py), which
produces `figures/perplexity_distributions.png`.

## Question

Could a person reading an ASR transcript spot hallucinations because they read
oddly? To find out, the paper scores each
[hallucination sequence](hallucination_detection.md) with a language model and
compares the scores with those of human-transcribed sequences of the same
length. If the two distributions overlap, the hallucinations are as fluent as
real speech and cannot be found by reading alone. The paper found that they
overlap heavily for all four ASRs.

## Method

The defaults reproduce the paper's figure: music noise (`noise1`) at 0 dB SNR,
using sessions from all AMI splits.

1. **Find hallucination sequences.** For each session, load
   `xscript_{noise}_snr{SNR}.feather`, normalize it with `normalize_xscript`,
   then run `align_using_time` (τ = 0.44 s), `compute_hallucinations`, and
   `find_hallucination_sequences` (≥ 3 `S`/`I`/`H` words, gaps ≤ 1 s). The
   sequence text is the space-joined normalized words.
2. **Score them.** `compute_perplexity()` returns each sequence's perplexity:
   exp(mean next-token negative log-likelihood). It reproduces the
   [Hugging Face `evaluate` perplexity metric](https://huggingface.co/spaces/evaluate-metric/perplexity)
   cited in the paper, using the paper's settings: model
   `google-bert/bert-large-uncased` and `add_start_token=False`, so the first
   token is context only and is not scored. Its output matches `evaluate` to
   within 4 × 10⁻⁶ relative error. See [Caveats](#caveats) for what scoring
   with BERT means.
3. **Build a length-matched human baseline.** Perplexity depends strongly on
   sequence length. So for each ASR and each length L that has k
   hallucination sequences, `sample_word_sequences()` draws
   k × `N_REF_PER_SEQ` (default 100) sequences of L consecutive words from the
   normalized human transcripts. Windows stay within one session, and sampling
   is without replacement when possible.
4. **Compare.** The figure plots normalized histograms of log(perplexity + 1)
   for ASR vs. human, using 50 bins from 0 to the 99.999th percentile of all
   values, as in the paper. Normalizing the pooled human sample
   is the same as averaging the counts over the 100 draws. Each panel title
   gives the **overlap**, `histogram_overlap()` = Σ min(p_asr, p_human) over
   shared bins (1 means identical distributions), and the number of
   hallucination sequences N.

The notebook also prints the median log perplexity per ASR and shows the most
fluent (lowest-perplexity) hallucinations for each.

## Running it

```bash
pip install -e ".[perplexity]"     # adds transformers
AMI_OUT_DIR=/data/transcripts/ami/out python notebooks/perplexity.py
```

Or run it cell by cell like `analysis.py`. Settings are at the top of the
notebook: `NOISE`, `SNR`, `TIME_TOLERANCE`, `MIN_LENGTH`, `MAX_GAP`,
`MODEL_ID`, `ADD_START_TOKEN`, `N_REF_PER_SEQ`, and `RANDOM_SEED`.

Every LM score is cached by text in
`$AMI_PERPLEXITY_CACHE/ppl_{model}.feather` (`ppl_{model}_bos.feather` with a start token) (default
`{AMI_OUT_DIR}/../cache/perplexities/`). Only texts that are not yet cached get
scored, so duplicates cost nothing and an interrupted run picks up where it
stopped. The hallucination sequences with their scores are also saved as
`hallucination_seqs_{noise}_snr{SNR}.feather`.

**Cost.** With `N_REF_PER_SEQ = 100`, the human baseline runs to millions of
short sequences. The paper's figure has about 36,000 hallucination sequences,
and the original cached results held about 3.6 million scores. BERT-large
(340M parameters) handles this on a GPU in a few hours at most. On a CPU, lower `N_REF_PER_SEQ` (even
1 gives a usable, noisier baseline).

## API

```python
from asr_hallucination.perplexity import (
    compute_perplexity, cached_perplexity, sample_word_sequences, histogram_overlap,
)

compute_perplexity(["yeah so i think", "i do not know"])            # BERT-large, as in the paper
compute_perplexity(texts, model_id="gpt2", add_start_token=True)      # a conventional causal LM instead
cached_perplexity(texts, "cache/ppl.feather")                         # same, with a persistent cache
sample_word_sequences(word_lists, length=4, n=100, rng=33)             # random 4-word windows
histogram_overlap(x, y, bins=np.linspace(0, 30, 51))                  # 0..1
```

`compute_perplexity` uses CUDA if available, then MPS, then the CPU. Pass
`device=` to override.

## Caveats

- **BERT is not a causal language model.** The original `get_perplexity()`
  helper called
  `evaluate.load("perplexity").compute(model_id="google-bert/bert-large-uncased", add_start_token=False, ...)`.
  The metric loads BERT with `AutoModelForCausalLM`, which gives
  `BertLMHeadModel`. Its masked-LM head is then scored on predicting each
  *next* token, even though BERT was trained to predict the token at its own
  (masked) position and its attention is bidirectional. The resulting
  "perplexities" are very large (log values of about 7–17 for short AMI
  phrases; GPT-2 gives about 5). So they are best read as a relative fluency
  score, not a true perplexity. The paper's claim compares ASR and human
  sequences scored the same way, so it does not depend on this. To try a
  conventional LM, set `MODEL_ID = "gpt2"` and `ADD_START_TOKEN = True`.
  These scores also depend on how `transformers` builds `BertLMHeadModel`, so
  a future library release could shift them.
- **Minor differences from the original pipeline.** The original code used its
  own text normalizer: a shorter filler-word list (`hmm mm mhm mmm uh um erm
  err er m ah ooh`) plus rewrites such as "dunno" → "do not know", "kinda" →
  "kind of", "'cause" → "because". It also ended a sequence at gaps of
  ≥ 1 s rather than > 1 s. This notebook uses the package's
  `get_normalizer()` and `find_hallucination_sequences()`, so N values and
  histograms may differ slightly from the published figure. The original's
  `onset` silence test changes only `I`/`H` labels, not which words count as
  errors, so it has no effect here.
- **Minimum length.** The published figure uses sequences of ≥ 3 words, the
  paper's definition (N = 3510 / 8804 / 19802 / 4199 for AssemblyAI / AWS /
  WhisperX / Reverb). The original notebook's last saved output used
  `min_length = 4` (N = 1384 / 3625 / 14612 / 1537) and is not the published
  version.
- **Human baseline draws.** One draw (seed 33) in the original
  `perplexity.py` sampled one human sequence per hallucination sequence. The
  published figure averaged the human histogram over repeated draws (a `seed`
  column, about 100 draws, matching the caption). The code that generated
  those draws was not recovered. Pooling `N_REF_PER_SEQ = 100` draws and
  normalizing gives the same mean histogram.
- Text is lowercased and unpunctuated (whisper-normalizer output). ASR and
  human sequences are normalized the same way.
