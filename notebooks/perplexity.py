# %% [markdown]
# # Perplexity of hallucination sequences
# Compares the language-model perplexity of ASR hallucination sequences with that of same-length
# sequences sampled from the human reference transcripts (paper Figure `perplexity_distributions.png`).
#
# 1. For every session, align each ASR transcript to the human reference with `align_using_time`,
#    label words with `compute_hallucinations`, and group them with `find_hallucination_sequences`
#    (>= 3 error words, gaps <= 1 s).
# 2. Score each hallucination sequence with BERT-large via `compute_perplexity` (as in the paper).
# 3. For each hallucination sequence, sample `N_REF_PER_SEQ` human sequences of the same length
#    (perplexity depends strongly on length) and score those too.
# 4. Plot normalized histograms of log(perplexity + 1) for ASR vs. human and report their overlap.
#
# Requires `pip install -e ".[perplexity]"`. A GPU is strongly recommended for the human reference
# sample. All LM scores are cached in `CACHE_DIR`, so the notebook can be interrupted and re-run.

# %%
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed
from plotly.subplots import make_subplots

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().resolve()
    if REPO_ROOT.name == "notebooks":
        REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from asr_hallucination.ami_utils import get_splits
from asr_hallucination.alignment import (
    get_normalizer, normalize_xscript, align_using_time, compute_hallucinations, find_hallucination_sequences,
)
from asr_hallucination.perplexity import (
    DEFAULT_MODEL_ID, cached_perplexity, sample_word_sequences, histogram_overlap,
)

# Directory with per-session transcripts ({OUT_DIR}/{session_id}/xscript_{noise}_snr{SNR}.feather)
OUT_DIR = Path(os.environ.get("AMI_OUT_DIR", "/data/transcripts/ami/out"))
CACHE_DIR = Path(os.environ.get("AMI_PERPLEXITY_CACHE", OUT_DIR.parent / "cache" / "perplexities"))
FIGDIR = REPO_ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

NOISE = "noise1"            # music
SNR = 0
TIME_TOLERANCE = 0.44       # seconds, for align_using_time
MIN_LENGTH, MAX_GAP = 3, 1.0
MODEL_ID = DEFAULT_MODEL_ID  # google-bert/bert-large-uncased, as in the paper
ADD_START_TOKEN = False      # as in the paper; set True for causal LMs such as "gpt2"
N_REF_PER_SEQ = 100         # human sequences sampled per hallucination sequence
RANDOM_SEED = 33

XSCRIBER_MAP = {"aai": "AssemblyAI", "aws": "AWS Transcribe", "wsp": "WhisperX", "rev": "Reverb"}

# All AMI splits are used for this analysis.
splits = get_splits()
SESSIONS = splits["test"] + splits["validation"] + splits["train"]

normalizer = get_normalizer()

# %% [markdown]
# ## Find hallucination sequences and collect the normalized human transcripts

# %%
def process_session(fn, normalizer):
    df = pd.read_feather(fn)
    session_id = df.session_id.iloc[0]
    transcripts = {
        t: {"word_segments": normalize_xscript(g.sort_values("start_time"), normalizer).to_dict("records")}
        for t, g in df.groupby("transcriber")
    }
    hum_words = [w["word_norm"] for w in transcripts["hum"]["word_segments"]]
    original = copy.deepcopy(transcripts)
    aligned = align_using_time(transcripts, time_tolerance=TIME_TOLERANCE)
    ops = compute_hallucinations(aligned, original)
    seqs = []
    for xscriber, labels in ops.items():
        for seq in find_hallucination_sequences(aligned[xscriber]["word_segments"], labels,
                                                min_length=MIN_LENGTH, max_gap=MAX_GAP):
            seqs.append({"session_id": session_id, "transcriber": xscriber,
                         **{k: seq[k] for k in ("length", "start_time", "stop_time", "text", "labels")}})
    return session_id, hum_words, seqs


session_files = [OUT_DIR / s / f"xscript_{NOISE}_snr{SNR:+03.0f}.feather" for s in SESSIONS]
session_files = [fn for fn in session_files if fn.exists()]
print(f"{len(session_files)} of {len(SESSIONS)} sessions have {NOISE} at SNR {SNR}")

results = Parallel(n_jobs=-1, verbose=1)(delayed(process_session)(fn, normalizer) for fn in session_files)
hum_word_lists = [words for _, words, _ in results]
seqdf = pd.DataFrame([seq for *_, seqs in results for seq in seqs])
seqdf.groupby("transcriber").length.describe()

# %% [markdown]
# ## Perplexity of the ASR hallucination sequences

# %%
ppl_cache = CACHE_DIR / f"ppl_{MODEL_ID.replace('/', '_')}{'_bos' if ADD_START_TOKEN else ''}.feather"
ppl_kwargs = dict(model_id=MODEL_ID, add_start_token=ADD_START_TOKEN)
seqdf["perplexity"] = cached_perplexity(seqdf.text.tolist(), ppl_cache, **ppl_kwargs)
seqdf["log_ppl"] = np.log(seqdf.perplexity + 1)

# %% [markdown]
# ## Length-matched human reference sequences
# For each ASR and each sequence length L with k hallucination sequences, sample k * N_REF_PER_SEQ
# human sequences of L consecutive words (within a session). Normalizing the pooled histogram
# is the same as averaging the counts over the N_REF_PER_SEQ draws.

# %%
rng = np.random.default_rng(RANDOM_SEED)
refs = []
for xscriber in XSCRIBER_MAP:
    for length, count in seqdf.loc[seqdf.transcriber == xscriber, "length"].value_counts().sort_index().items():
        for text in sample_word_sequences(hum_word_lists, length, count * N_REF_PER_SEQ, rng):
            refs.append({"transcriber": xscriber, "length": length, "text": text})
refdf = pd.DataFrame(refs)
print(f"{len(refdf)} human sequences ({refdf.text.nunique()} unique)")

refdf["perplexity"] = cached_perplexity(refdf.text.tolist(), ppl_cache, **ppl_kwargs)
refdf["log_ppl"] = np.log(refdf.perplexity + 1)

# %% [markdown]
# ## Plot

# %%
FONTCOLOR = 'rgba(0.2,0.2,0.2,1.0)'
GRIDCOLOR = 'rgba(0.2,0.2,0.3,0.1)'
FONTSIZE = 16
HUM_COLOR, ASR_COLOR = "rgba(128,128,128,0.5)", "rgba(70,160,180,0.45)"

# As in the paper: 50 bins from 0 to the 99.999th percentile of all log perplexities
n_bins = 50
max_x = pd.concat([seqdf.log_ppl, refdf.log_ppl]).quantile(0.99999)
bins = np.linspace(0, max_x, n_bins + 1)
bin_width = bins[1] - bins[0]
centers = bins[:-1] + bin_width / 2

titles = []
for xscriber, name in XSCRIBER_MAP.items():
    asr = seqdf.loc[seqdf.transcriber == xscriber, "log_ppl"]
    hum = refdf.loc[refdf.transcriber == xscriber, "log_ppl"]
    titles.append(f"{name}<br>{histogram_overlap(asr, hum, bins):.0%} overlap (N={len(asr)})")

fig = make_subplots(rows=2, cols=2, subplot_titles=titles, shared_yaxes=True,
                    horizontal_spacing=0.08, vertical_spacing=0.15)
for i, xscriber in enumerate(XSCRIBER_MAP):
    row, col = i // 2 + 1, i % 2 + 1
    for label, values, color in [("Human", refdf.loc[refdf.transcriber == xscriber, "log_ppl"], HUM_COLOR),
                                 ("ASR", seqdf.loc[seqdf.transcriber == xscriber, "log_ppl"], ASR_COLOR)]:
        counts = np.histogram(values, bins=bins)[0] / len(values)
        fig.add_trace(go.Bar(x=centers, y=counts, width=bin_width, name=label, marker_color=color,
                             marker_line=dict(color="white", width=0.5), showlegend=(i == 0)),
                      row=row, col=col)

fig.update_layout(
    width=900, height=750,
    barmode="overlay",
    font=dict(size=FONTSIZE, color=FONTCOLOR),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(font_size=FONTSIZE, yanchor="top", y=0.95, xanchor="center", x=0.5),
    margin=dict(l=80, r=10, b=80, t=70),
)
fig.update_xaxes(range=[0, max_x * 1.05], gridcolor=GRIDCOLOR, zerolinecolor=GRIDCOLOR)
fig.update_yaxes(gridcolor=GRIDCOLOR, zerolinecolor=GRIDCOLOR)
fig.add_annotation(text="Log Perplexity", x=0.5, y=-0.1, xref="paper", yref="paper", showarrow=False)
fig.add_annotation(text="Normalized Count", x=-0.09, y=0.5, xref="paper", yref="paper", showarrow=False,
                   textangle=-90)

fig.write_image(FIGDIR / "perplexity_distributions.png", scale=3)
fig.show()

# %% [markdown]
# ## Summary table and the most fluent hallucinations

# %%
summary = pd.DataFrame([
    {"transcriber": XSCRIBER_MAP[x], "n_seqs": (seqdf.transcriber == x).sum(),
     "asr_median_log_ppl": seqdf.loc[seqdf.transcriber == x, "log_ppl"].median(),
     "hum_median_log_ppl": refdf.loc[refdf.transcriber == x, "log_ppl"].median()}
    for x in XSCRIBER_MAP
])
print(summary.to_string(index=False))
seqdf.sort_values("perplexity").groupby("transcriber").head(5)[["transcriber", "session_id", "start_time", "text", "perplexity"]]

# %%
seqdf.to_feather(CACHE_DIR / f"hallucination_seqs_{NOISE}_snr{SNR:+03.0f}.feather")
