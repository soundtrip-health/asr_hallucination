"""Time-constrained word alignment, WER, and hallucination detection."""

import pandas as pd
import numpy as np

# Disfluencies / filler words removed during normalization (see the paper's Methods section).
FILLER_WORDS = ["hmm", "mm", "mhm", "mmm", "uh", "um", "erm", "err", "er", "m", "ah",
                "la", "da", "ooh", "aum", "rawr", "om"]
FILLER_PATTERN = r"\b(" + "|".join(FILLER_WORDS) + r")\b"


def get_normalizer():
    """Return the whisper-normalizer EnglishTextNormalizer configured with FILLER_WORDS."""
    from whisper_normalizer.english import EnglishTextNormalizer

    normalizer = EnglishTextNormalizer()
    normalizer.ignore_patterns = FILLER_PATTERN
    return normalizer


def normalize_xscript(df, normalizer):
    if len(df) == 0:
        return df
    out = []
    ts_delta, ts_duration = 0.001, 0.01
    prev_end = df.stop_time.iloc[0] if not np.isnan(df.stop_time.iloc[0]) else 0
    
    for row in df.to_dict(orient="records"):
        # Fix missing times
        if np.isnan(row["start_time"]):
            row["start_time"] = prev_end + ts_delta
        if np.isnan(row["stop_time"]):
            row["stop_time"] = row["start_time"] + ts_duration
        if row["stop_time"] < row["start_time"]:
            row["stop_time"] = row["start_time"]
        # Fix AMI unusual acronymn format (e.g., "A_M_I")
        if row["word"].isupper():
            row["word"] = row["word"].replace("_", "")
        norm = normalizer(row["word"]).split(" ")
        if len(norm) == 0 or norm[0] == "":
            continue
        row["word_norm"] = norm[0]
        if len(norm) > 1:
            delta = (row["stop_time"] - row["start_time"]) / len(norm)
            row["stop_time"] = row["start_time"] + delta
            out.append(row)
            cur_start = row["stop_time"]
            for i,w in enumerate(norm[1:]):
                nextrow = {k:v for k,v in row.items()}
                nextrow["word_norm"] = w
                nextrow["start_time"] = cur_start
                cur_start += delta
                nextrow["stop_time"] = cur_start
                out.append(nextrow)
        else:
            out.append(row)
        prev_end = row["stop_time"]

    return pd.DataFrame(out)


def my_time_align(refdf, hypdf, delta=0.44):
    """ Time-constrained word alignment.
        Note that hypdf will be modified in-place to add window_idx and match_idx columns.
    """
    # IntervalIndex is a pandas implamentation of an interval tree
    ref_time_window = pd.IntervalIndex.from_arrays(refdf.start_time-delta, refdf.stop_time+delta, closed='left')
    hyp_time_window = pd.IntervalIndex.from_arrays(hypdf.start_time, hypdf.stop_time, closed='left')
    # Making these numpy arrays will likely make for a big speedup because of the pandas df update overhead.
    hypdf["window_idx"] = pd.Series(np.nan, dtype='Int64')
    hypdf["match_idx"] = pd.Series(np.nan, dtype='Int64')
    nref = 0
    nmatch = 0
    del_indices = []
    for ref_idx, row in refdf.iterrows():
        if row.word == "":
            continue
        nref += 1
        overlap = hyp_time_window.overlaps(ref_time_window[ref_idx])
        # only get overlaps that haven't been matched yet
        hyp_rows = hypdf.loc[overlap & hypdf.match_idx.isna()]
        if len(hyp_rows) == 0:
            # no words in the temporal window, a delete
            del_indices.append(ref_idx)
            continue
        # tag all these hyp words as being in a ref word temporal window
        hypdf.loc[overlap, "window_idx"] = ref_idx
        # Check for matching word
        found_match = False
        for hyprow in hyp_rows.itertuples():
            if hyprow.word_norm == row.word_norm:
                hypdf.loc[hyprow.Index, "match_idx"] = ref_idx
                nmatch += 1
                found_match = True
                break
        if not found_match:
            # no matches in this window; a delete
            del_indices.append(ref_idx)

    # hyp words in a temporal window but not matching can be substitutes or inserts
    nsubins = ((hypdf.match_idx.isna()) & ~hypdf.window_idx.isna()).sum()
    # hyp words not in a window (and of course also not matching) are definitely inserts
    nins = hypdf.window_idx.isna().sum()
    return nref, del_indices, nsubins, nins, nmatch

# TODO: use interval tree data structure
def get_temporal_segment(
    words: list[dict], start_time: float, stop_time: float):
    """Gets a list of words and returns the words that are within the start and stop time.
    Expects the words to be sorted by start time.
    """
    idx_start, idx_end = 0, len(words) - 1
    while idx_start <= idx_end:
        idx_mid = (idx_start + idx_end) // 2
        if words[idx_mid]["start_time"] < start_time:
            idx_start = idx_mid + 1
        else:
            idx_end = idx_mid - 1

    idx_stop, idx_end = idx_start, len(words) - 1
    while idx_stop <= idx_end:
        idx_mid = (idx_stop + idx_end) // 2
        if words[idx_mid]["start_time"] < stop_time:
            idx_stop = idx_mid + 1
        else:
            idx_end = idx_mid - 1

    return words[idx_start:idx_stop], words[:idx_start], words[idx_stop:]


def align_using_time(transcripts: dict, time_tolerance: float | dict[str, float]):
    if "hum" not in transcripts:
        raise ValueError("Human transcript is required for time-informed alignment.")

    transcribers = [transcriber for transcriber in transcripts if transcriber != "hum"]

    if isinstance(time_tolerance, (int, float)):
        time_tolerance = {transcriber: time_tolerance for transcriber in transcribers}

    unmatched_leftovers = {transcriber: [] for transcriber in transcribers}
    last_match_time = {transcriber: 0 for transcriber in transcribers}

    for ref_word in transcripts["hum"]["word_segments"]:
        for transcriber in transcribers:
            target_segment, left_leftover, right_leftover = get_temporal_segment(
                transcripts[transcriber]["word_segments"],
                ref_word["start_time"] - time_tolerance[transcriber],
                ref_word["stop_time"] + time_tolerance[transcriber],
            )

            for left_word in left_leftover:
                left_word["last_match_start"] = last_match_time[transcriber]
                unmatched_leftovers[transcriber].append(left_word)

            if not target_segment:
                ref_word["delete_in_" + transcriber] = True
                transcripts[transcriber]["word_segments"] = right_leftover
                continue

            idx, idx_match = 0, -1
            while idx < len(target_segment):
                if target_segment[idx]["word_norm"] == ref_word["word_norm"]:
                    ref_word["has_match"] = True
                    ref_word["matches_" + transcriber] = target_segment[idx]
                    target_segment[idx]["matches"] = True
                    idx_match = idx
                    break
                idx += 1

            if idx_match == -1:
                transcripts[transcriber]["word_segments"] = target_segment + right_leftover
            else:
                transcripts[transcriber]["word_segments"] = target_segment[idx_match + 1 :] + right_leftover
                for target_word in target_segment[:idx_match]:
                    target_word["last_match_start"] = last_match_time[transcriber]
                    unmatched_leftovers[transcriber].append(target_word)
                last_match_time[transcriber] = target_segment[idx]["start_time"]

    for transcriber in transcribers:
        for word in transcripts[transcriber]["word_segments"]:
            word["last_match_start"] = last_match_time[transcriber]

        transcripts[transcriber]["word_segments"].extend(unmatched_leftovers[transcriber])
        transcripts[transcriber]["word_segments"].sort(key=lambda x: x["start_time"])
        unmatched_leftovers[transcriber] = []

    # take care of substitutions and deletions in reference
    for transcriber in transcribers:
        last_match_time = 0
        for word in transcripts["hum"]["word_segments"]:
            if "delete_in_" + transcriber in word:
                continue

            if "matches_" + transcriber in word:
                last_match_time = word["matches_" + transcriber]["start_time"]
                continue

            segment, left, right = get_temporal_segment(
                transcripts[transcriber]["word_segments"],
                max(last_match_time, word["start_time"] - time_tolerance[transcriber]),
                word["stop_time"] + time_tolerance[transcriber],
            )

            if not segment:
                word["delete_in_" + transcriber] = True
                unmatched_leftovers[transcriber].extend(left)
                transcripts[transcriber]["word_segments"] = right
                continue

            if segment[0]["last_match_start"] >= last_match_time:
                word["substitute_by_" + transcriber] = segment[0]
                segment[0]["substitutes"] = True
                unmatched_leftovers[transcriber].extend(left)
                transcripts[transcriber]["word_segments"] = segment[1:] + right
            else:
                word["delete_in_" + transcriber] = True

    for transcriber in transcribers:
        transcripts[transcriber]["word_segments"].extend(unmatched_leftovers[transcriber])
        transcripts[transcriber]["word_segments"].sort(key=lambda x: x["start_time"])

        for word in transcripts[transcriber]["word_segments"]:
            if "substitutes" not in word:
                word["insert"] = True

    return transcripts


def _has_overlap(starts, prefix_max_stops, start_time, stop_time):
    """True if any interval [starts[i], stops[i]) overlaps [start_time, stop_time).

    starts must be sorted; prefix_max_stops[i] is max(stops[:i+1]).
    """
    idx = np.searchsorted(starts, stop_time, side="left")
    if start_time == stop_time:
        # zero-length word: count a reference word starting exactly here as overlapping
        idx = np.searchsorted(starts, stop_time, side="right")
    return idx > 0 and prefix_max_stops[idx - 1] > start_time


def compute_hallucinations(
    transcripts: dict, transcripts_original: dict, print_top_n: int = 0, silence_test: str = "overlap"
) -> dict[str, list[str]]:
    """Label every ASR word as M (match), S (substitution), I (insertion during speech) or
    H (insertion during reference silence).

    Expects the output of align_using_time() and a deep copy of its input. Each ASR's
    word_segments list is rebuilt in place to contain all of its words in time order, so
    operations[asr][i] labels transcripts[asr]["word_segments"][i].

    Args:
        transcripts: aligned transcripts from align_using_time()
        transcripts_original: deep copy of the transcripts before alignment (sanity check)
        print_top_n: if > 0, print the N longest hallucination sequences per ASR
        silence_test: how to decide whether an inserted word is in reference silence (H):
            "overlap" (default): no reference word overlaps the inserted word's [start, stop)
            "onset": no reference word *starts* within the inserted word's [start, stop)

    Returns:
        dict mapping ASR name to a list of labels
    """
    if silence_test not in ("overlap", "onset"):
        raise ValueError(f"Unknown silence_test '{silence_test}'.")
    non_human_transcribers = [transcriber for transcriber in transcripts if transcriber != "hum"]
    for transcriber in non_human_transcribers:
        for word in transcripts["hum"]["word_segments"]:
            if "matches_" + transcriber in word:
                transcripts[transcriber]["word_segments"].append(word["matches_" + transcriber])
            elif "substitute_by_" + transcriber in word:
                transcripts[transcriber]["word_segments"].append(word["substitute_by_" + transcriber])
        transcripts[transcriber]["word_segments"].sort(key=lambda x: x["start_time"])

    for transcriber in transcripts_original:
        np.testing.assert_array_equal(
            [word["start_time"] for word in transcripts_original[transcriber]["word_segments"]],
            [word["start_time"] for word in transcripts[transcriber]["word_segments"]],
        )

    ref_words = sorted(transcripts["hum"]["word_segments"], key=lambda x: x["start_time"])
    ref_starts = np.array([word["start_time"] for word in ref_words], dtype=float)
    ref_prefix_max_stops = np.maximum.accumulate(np.array([word["stop_time"] for word in ref_words], dtype=float))

    operations = {}
    for transcriber in non_human_transcribers:
        operations[transcriber] = []
        for word in transcripts[transcriber]["word_segments"]:
            if "matches" in word:
                operations[transcriber].append("M")
            elif "substitutes" in word:
                operations[transcriber].append("S")  # potentially hallucination
            elif "insert" in word:
                if silence_test == "overlap":
                    in_speech = len(ref_words) > 0 and _has_overlap(
                        ref_starts, ref_prefix_max_stops, word["start_time"], word["stop_time"])
                else:
                    segment, *_ = get_temporal_segment(ref_words, word["start_time"], word["stop_time"])
                    in_speech = len(segment) > 0
                operations[transcriber].append("I" if in_speech else "H")  # I: potentially hallucination

    if print_top_n > 0:
        print_longest_hallucinations(transcripts, operations, print_top_n)

    return operations


def find_hallucination_sequences(
    words: list[dict],
    operations: list[str],
    min_length: int = 3,
    max_gap: float = 1.0,
    error_labels: tuple[str, ...] = ("S", "I", "H"),
) -> list[dict]:
    """Group per-word labels into hallucination sequences.

    Following the paper, a hallucination is a run of at least min_length consecutive error words
    (substitutions or insertions) in the ASR transcript where the gap between one word's stop_time
    and the next word's start_time is no more than max_gap seconds. A non-error word (match) or a
    longer gap ends the run.

    Args:
        words: one ASR's word_segments after compute_hallucinations() (sorted by start_time)
        operations: the matching list of labels from compute_hallucinations()
        min_length: minimum number of words in a sequence
        max_gap: maximum allowed gap between consecutive words, in seconds
        error_labels: labels that count as errors

    Returns:
        list of dicts (in time order) with keys: start_idx, end_idx (exclusive), length,
        start_time, stop_time, text, labels
    """
    if len(words) != len(operations):
        raise ValueError(f"words ({len(words)}) and operations ({len(operations)}) must have the same length.")

    sequences = []

    def close_run(run_start, run_end):
        if run_start is not None and run_end - run_start >= min_length:
            run = words[run_start:run_end]
            sequences.append({
                "start_idx": run_start,
                "end_idx": run_end,
                "length": run_end - run_start,
                "start_time": run[0]["start_time"],
                "stop_time": run[-1]["stop_time"],
                "text": " ".join(word.get("word_norm", word.get("word", "")) for word in run),
                "labels": "".join(operations[run_start:run_end]),
            })

    run_start = None
    for idx, (word, op) in enumerate(zip(words, operations)):
        if op not in error_labels:
            close_run(run_start, idx)
            run_start = None
            continue
        if run_start is not None and word["start_time"] - words[idx - 1]["stop_time"] > max_gap:
            close_run(run_start, idx)
            run_start = None
        if run_start is None:
            run_start = idx
    close_run(run_start, len(words))
    return sequences


def print_longest_hallucinations(transcripts: dict, operations: dict, top_n: int, **kwargs):
    """Print the top_n longest hallucination sequences (as length@start_time) for each ASR.

    Extra kwargs are passed to find_hallucination_sequences().
    """
    cell_width = 11
    headers = ["transcriber"] + ["Top " + str(i + 1) for i in range(top_n)]
    header_row = " | ".join(f"{h:^{cell_width}s}" for h in headers)
    print("=" * len(header_row))
    print("Longest hallucination sequences".center(len(header_row)))
    print("-" * len(header_row))
    print(header_row)
    print("=" * len(header_row))

    for transcriber, ops in operations.items():
        sequences = find_hallucination_sequences(transcripts[transcriber]["word_segments"], ops, **kwargs)
        sequences = sorted(sequences, key=lambda x: x["length"], reverse=True)[:top_n]
        row = [transcriber] + [f"{seq['length']}@{seq['start_time']:.1f}" for seq in sequences]
        row += [""] * (top_n + 1 - len(row))
        print(" | ".join(f"{cell:^{cell_width}s}" for cell in row))
        print("-" * len(header_row))


def compute_wer(transcripts: dict) -> dict[str, float]:
    metrics = []
    nref = len(transcripts["hum"]["word_segments"])
    
    for xscriber in transcripts:
        if xscriber == "hum":
            continue
        ndel = sum([1 for word in transcripts["hum"]["word_segments"] if "delete_in_" + xscriber in word])
        nhit = sum([1 for word in transcripts["hum"]["word_segments"] if "matches_" + xscriber in word])
        nsub = sum([1 for word in transcripts["hum"]["word_segments"] if "substitute_by_" + xscriber in word])
        nins = sum([1 for word in transcripts[xscriber]["word_segments"] if "insert" in word])
        metrics.append({
                "transcriber": xscriber,
                "nref": nref,
                "insertions": nins,
                "deletions": ndel,
                "substitutions": nsub,
                "hits": nhit,
                "wer": (ndel + nins + nsub) / nref,
        })
        
    return metrics

