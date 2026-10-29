# Time-constrained word alignment

Code: [`asr_hallucination/alignment.py`](../asr_hallucination/alignment.py)

## Why not plain word alignment?

Standard WER tools (e.g. `jiwer`) align a hypothesis (ASR) transcript to a
reference (human) transcript by minimizing Levenshtein edit distance over the
word sequence. That ignores *when* words were spoken. For long, noisy
transcripts with high error rates this causes predictable failures:

- **Spurious matches across time.** A common word (`so`, `the`, `to`) in the
  hypothesis can be matched to the same word in the reference tens of seconds
  away, because doing so lowers the edit distance. (Paper example: `so` at
  61.21 s in the reference matched to `so` at 110.33 s from AWS.)
- **Spurious substitutions across time.** An inserted ASR word gets paired with
  a deleted reference word that is far away (`ok` at 25.61 s "substituting" for
  `so` at 52.49 s). One substitution is counted where an insertion plus a
  deletion actually happened.
- **Ties broken arbitrarily.** When several alignments have the same edit
  distance, the one picked may not be the temporally plausible one.

All three make WER look *better* than it is, and they hide insertions, the
error type that hallucinations consist of. For comparability with the
literature, the paper's headline WER numbers still use jiwer's word alignment
(computed in `notebooks/analysis.py`). The time-constrained alignment is used
to characterize error types and to find hallucinations.

## Input format

Both aligners work on word-level transcripts in which every word has:

| Field | Meaning |
|-------|---------|
| `word` | Raw token as emitted by the transcriber |
| `word_norm` | Normalized token (see below); this is what gets compared |
| `start_time`, `stop_time` | Seconds from the start of the recording |

### Normalization: `normalize_xscript(df, normalizer)`

Before alignment, each transcript DataFrame is passed through
`normalize_xscript` with the normalizer returned by `get_normalizer()`. That is a
[whisper-normalizer](https://pypi.org/project/whisper-normalizer/)
`EnglishTextNormalizer` configured with the package's filler-word list.
`normalize_xscript`:

1. Fills missing timestamps. A missing `start_time` becomes the previous
   word's end + 1 ms; a missing `stop_time` becomes start + 10 ms. A
   `stop_time` earlier than `start_time` is clamped to `start_time`.
2. Joins AMI's underscore acronyms (`A_M_I` → `AMI`).
3. Normalizes the word (lower-casing, removing bracketed annotations, removing
   filler words, standardizing spelling and numbers).
4. Drops words that normalize to nothing (e.g. `um`).
5. Splits words that normalize to several tokens (e.g. `don't` → `do not`) into
   separate rows. The original duration is divided evenly among them.

The filler words are defined once in `alignment.FILLER_WORDS` and match the
paper's Methods section:
`hmm, mm, mhm, mmm, uh, um, erm, err, er, m, ah, la, da, ooh, aum, rawr, om`.
The analysis notebook uses the same normalizer.

## Algorithm: `align_using_time(transcripts, time_tolerance)`

This is the aligner described in the paper. It aligns one human reference
against any number of ASR transcripts at once.

```python
transcripts = {
    "hum": {"word_segments": [...]},   # reference, required
    "aws": {"word_segments": [...]},
    "wsp": {"word_segments": [...]},
}
transcripts = align_using_time(transcripts, time_tolerance=0.44)  # or {"aws": 0.5, "wsp": 0.3, ...}
```

Every `word_segments` list **must be sorted by `start_time`**. Candidate
windows are found by binary search on `start_time` (`get_temporal_segment`).
The input dicts are **modified in place** (annotation keys are added to the word
dicts), so make a `copy.deepcopy` first if you need the originals. You will need
one for `compute_hallucinations`.

For a reference word $r$ and tolerance $\tau$, the **candidate window** is every
not-yet-consumed hypothesis word whose `start_time` falls in

$$[\,r.\mathrm{start} - \tau,\; r.\mathrm{stop} + \tau\,)$$

The algorithm runs in two passes over the reference words, independently for
each ASR transcript.

### Pass 1: exact matches

For each reference word, in time order:

1. Find the candidate window.
2. Hypothesis words that start *before* the window can no longer match any
   later reference word. They are moved to a leftover pool. Each one is
   stamped with `last_match_start`, the start time of the most recent match
   at the moment it was set aside.
3. If the window is empty, the reference word is marked
   **deleted** (`delete_in_<asr>`).
4. Otherwise, scan the window in time order for the first word whose
   `word_norm` equals the reference's `word_norm`.
   - **Match found:** pair them (`matches_<asr>` on the reference word,
     `matches` on the hypothesis word). Unmatched window words *before* the
     match go to the leftover pool. Words *after* the match stay available
     for later reference words.
   - **No match:** the whole window stays available. The reference word is
     left unresolved for pass 2.

Consuming everything up to and including a match keeps the alignment
monotonic: a later reference word can never match a hypothesis word that comes
before an earlier match.

### Pass 2: substitutions and remaining deletions

The leftovers are merged back with the unconsumed hypothesis words and
re-sorted. Then, for each reference word still unresolved after pass 1:

1. Search a window that starts no earlier than the previous match:
   `[max(last_match_time, r.start − τ), r.stop + τ)`.
2. If the window is empty → **deletion**.
3. If the first word in the window was set aside *after* the most recent match
   (`last_match_start >= last_match_time`), pair it with the reference word as
   a **substitution** (`substitute_by_<asr>` / `substitutes`). This check stops
   a substitution from crossing an existing match.
4. Otherwise → **deletion**.

### Insertions

Any hypothesis word that ends up neither matched nor substituting is flagged
`insert`.

### Output annotations

| Where | Key | Meaning |
|-------|-----|---------|
| reference word | `matches_<asr>` | the matched hypothesis word dict |
| reference word | `substitute_by_<asr>` | the substituting hypothesis word dict |
| reference word | `delete_in_<asr>` | `True`: deleted by that ASR |
| hypothesis word | `matches` / `substitutes` / `insert` | that word's role |

Note that after `align_using_time`, each ASR's `word_segments` list holds only
the **unmatched** words (insertions). Matched and substituting words can be
reached through the reference annotations. `compute_hallucinations` puts them
back (see [hallucination detection](hallucination_detection.md)).

### Error counts: `compute_wer(transcripts)`

Returns one dict per ASR with `nref`, `hits`, `substitutions`, `deletions`,
`insertions`, and `wer = (S + D + I) / nref`. Here `nref` is the number of
(normalized) reference words.

## Worked example

Reference `i really like apples`, and a hypothesis that also adds a phrase
during silence a few seconds later:

| | words (start–stop, s) |
|-|-|
| `hum` | i (0.0–0.2), really (0.3–0.7), like (0.8–1.1), apples (1.2–1.7) |
| `wsp` | i (0.05–0.2), love (0.8–1.0), red (1.0–1.15), apples (1.25–1.7), thank (5.0–5.3), you (5.3–5.5), so (5.5–5.6), much (5.6–5.9) |

Actual output for two tolerances:

| τ | really | like | red | thank you so much | S / D / I | WER |
|---|--------|------|-----|-------------------|-----------|-----|
| 0.10 s | deleted | → love (S) | inserted | inserted | 1 / 1 / 5 | 1.75 |
| 0.44 s | → love (S) | → red (S) | — | inserted | 2 / 0 / 4 | 1.50 |

With τ = 0.44 s, the window for `really` (0.3–0.7 s) reaches out to 1.14 s.
That brings `love` into range, and the greedy pass 2 takes it. Plain
edit-distance alignment would choose *really*→D, *like*→*love* (S), *red*→I.
This is the trade-off the paper notes: the algorithm is simple and enforces
temporal plausibility, but it is **greedy, not globally optimal**. A wider τ
tolerates more timestamp jitter but makes early, loose pairings like this more
likely.

## Choosing the tolerance

`align_using_time` has no default tolerance. The DataFrame variant below
defaults to 0.44 s. Pick τ based on how far the ASR timestamps typically
drift from the human ones:

- The paper's timestamp-variability analysis (`jitter_hist.png`) measures this
  from words inside long runs of consecutive matches. Those are almost
  certainly correct pairings, and for all four systems they show very few
  differences above ~2 s.
- AMI's human timestamps are themselves approximate.
- τ can be a number (same for every ASR) or set per transcriber by passing a
  dict, e.g. for an ASR whose
  timestamps run systematically late.

## Alternative implementation: `my_time_align(refdf, hypdf, delta=0.44)`

A simpler, single-pass, DataFrame-based variant. It is useful for quick
per-transcript counts:

```python
nref, del_indices, nsubins, nins, nmatch = my_time_align(refdf, hypdf, delta=0.44)
```

- Reference windows are `[start − δ, stop + δ)`. Hypothesis words are intervals
  `[start, stop)` and are in a window if they **overlap** it. This differs from
  the start-time-only test in `align_using_time`.
- For each reference word, the first not-yet-matched overlapping hypothesis word
  with the same `word_norm` is a match. Matching is not forced to be monotonic.
- `hypdf` gets `window_idx` (the last reference window it fell in) and
  `match_idx` columns, in place. `refdf` must have a default `RangeIndex`.

Its outputs do **not** separate substitutions from deletions/insertions:

| Return | Contains |
|--------|----------|
| `nref` | non-empty reference words |
| `del_indices` | reference words with no match, i.e. **S + D** |
| `nsubins` | unmatched hyp words inside some reference window, i.e. **S + in-window I** |
| `nins` | hyp words outside every window, i.e. insertions certainly not near any reference word |
| `nmatch` | hits |

`nins` is a useful rough count of hallucinated words, since these words were
emitted where no one was speaking.
