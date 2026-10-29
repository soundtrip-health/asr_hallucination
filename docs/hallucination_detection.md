# Hallucination detection

Code: `compute_hallucinations()` and `find_hallucination_sequences()` in
[`asr_hallucination/alignment.py`](../asr_hallucination/alignment.py). It builds
on the output of [time-constrained alignment](time_constrained_alignment.md).

## Definition

The paper uses this definition:

> A **hallucination** is a sequence of **three or more** inserted or substituted
> words in the ASR transcript where the temporal gap between consecutive words
> is **no longer than one second**.

The one-second gap rule keeps errors from separate, unconnected utterances from
being chained together. It mirrors how a person reviewing a transcript would not
read words separated by a long pause as one phrase.

Hallucinations are *fluent*. The paper's perplexity analysis
(`perplexity_distributions.png`) shows that hallucinated sequences from all four
ASRs are about as linguistically coherent as same-length human transcript
sequences. They usually cannot be spotted by reading the transcript alone,
which is why a time-aligned reference is needed to find them.

## Step 1: label every ASR word

```python
import copy
from asr_hallucination.alignment import align_using_time, compute_hallucinations

original = copy.deepcopy(transcripts)          # needed for the sanity check below
aligned = align_using_time(transcripts, time_tolerance=0.44)
operations = compute_hallucinations(aligned, original)          # silence_test="overlap" (default)
# operations["wsp"] -> ["M", "M", "S", "I", "H", "H", ...]
# operations["wsp"][i] labels aligned["wsp"]["word_segments"][i]
```

`compute_hallucinations` first rebuilds each ASR transcript. It adds the
matched and substituting words (held in the reference annotations after
alignment) back into the list of unmatched words and re-sorts by start time. It
then **asserts** that the rebuilt start times exactly equal the original
transcript's. This checks that the aligner accounted for every ASR word exactly
once.

Each ASR word then gets one label:

| Label | Meaning | How it is determined |
|-------|---------|----------------------|
| `M` | Match | aligned to an identical reference word |
| `S` | Substitution | aligned to a different reference word; *potential* hallucination |
| `I` | Insertion during speech | unaligned, but its `[start, stop)` span overlaps at least one reference word; *potential* hallucination |
| `H` | Insertion during silence | unaligned, and **no** reference word overlaps its span; treated as a hallucinated word |

`I` and `H` both count as insertions for WER. The split separates ASR words
added *while someone was talking* (often ordinary recognition errors, or speech
the human transcriber left out) from words produced where the reference has no
speech at all.

### Silence test

The `silence_test` argument controls how `I` and `H` are told apart:

| `silence_test` | An inserted word is `H` when… |
|----------------|-------------------------------|
| `"overlap"` (default) | no reference word overlaps the inserted word in time (reference `start < word.stop` and reference `stop > word.start`) |
| `"onset"` | no reference word **starts** within the inserted word's `[start, stop)` span |

`"onset"` was the original behavior. It mislabels inserted words that fall in
the middle of a long reference word as `H`. In the worked example in the
alignment doc (τ = 0.1 s), `red` (1.00–1.15 s) overlaps `like` (0.8–1.1 s).
`"overlap"` labels it `I`, while `"onset"` labels it `H`. Neither test applies a
time tolerance.

## Step 2: group labels into hallucination sequences

`find_hallucination_sequences()` applies the paper's definition to one ASR's
labeled words:

```python
from asr_hallucination.alignment import find_hallucination_sequences

seqs = find_hallucination_sequences(aligned["wsp"]["word_segments"], operations["wsp"],
                                    min_length=3, max_gap=1.0)
# [{"start_idx": 4, "end_idx": 8, "length": 4, "start_time": 5.0, "stop_time": 5.9,
#   "text": "thank you so much", "labels": "HHHH"}, ...]
```

Scanning the words in time order:

1. Words labeled `S`, `I`, or `H` (configurable with `error_labels`) extend
   the current run. An `M` ends it.
2. The run also ends when the gap between one word's `stop_time` and the next
   word's `start_time` is more than `max_gap` seconds (default 1.0).
3. Runs of at least `min_length` words (default 3) are returned as
   hallucination sequences, in time order.

The per-session counts of these sequences, by ASR, noise type, and SNR, are
what `hallucination_count.png` shows. To print a quick summary of the longest
sequences for each ASR (shown as `length@start_time`), pass `print_top_n=N` to
`compute_hallucinations`, or call `print_longest_hallucinations()` directly.

> The paper's perplexity analysis (`perplexity_distributions.png`) is not
> included in this repository.

## Heuristics for hallucinations without a reference

When no human reference exists (the real-world case), the paper suggests two
signals worth building on:

- **Repetition.** Repeated words or phrases are a strong warning sign,
  especially for Whisper. Example: at −10 dB (session IS1007c, ~781 s),
  WhisperX transcribed the start of an utterance correctly, then drifted into
  "he is got a nice hat" repeated over and over. The other three ASRs produced
  nothing for that span. Real speech also contains repetitions, though, so
  this signal needs care.
- **Cross-ASR agreement.** If most systems produce nothing (or something
  different) over a span where one system emits fluent text, that system is
  probably hallucinating. Agreement among several ASRs against the human
  reference can also expose *human* transcription errors. The paper found
  cases where all ASRs agreed on "yeah but" and the human transcript had
  omitted it.

The time-aligned, multi-transcriber structure of `align_using_time` (one
reference, many hypotheses, annotated in one pass) is designed to support
comparisons like these.

## Related signal: VAD

`process_session_audio` also runs Silero VAD on each (noise-mixed) audio file
and saves the per-frame speech probabilities, at 32 ms steps, to
`out/{session}/vad_{noise}_snr{SNR}.feather`. Nothing in the package uses
these yet. They are a natural reference-free way to flag ASR words emitted
where the VAD sees no speech.
