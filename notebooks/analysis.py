# %%
# %%capture
# %pip install pandas jiwer plotly nbformat kaleido whisper-normalizer
# %pip install assemblyai whisperx silero_vad filetype pydub pyarrow


# %%
import json
import os
import sys
import re
import subprocess
import time
import pandas as pd
import numpy as np
import scipy.stats as stats
import jiwer
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from joblib import Parallel, delayed
from pathlib import Path

# Repo root: __file__ is undefined when run as a Jupyter notebook, so fall back to the cwd
# (assumed to be either the repo root or the notebooks/ directory).
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().resolve()
    if REPO_ROOT.name == "notebooks":
        REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from asr_hallucination.ami_utils import get_splits
from asr_hallucination.alignment import get_normalizer, normalize_xscript

pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 300)
pd.set_option("display.max_colwidth", 1000)

# Directory with per-session transcripts ({OUT_DIR}/{session_id}/xscript_{noise}.feather)
OUT_DIR = Path(os.environ.get("AMI_OUT_DIR", "/data/transcripts/ami/out"))
FIGDIR = REPO_ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# whisper-normalizer EnglishTextNormalizer with the paper's filler-word list
normalizer = get_normalizer()

XSCRIBER_MAP = {"aai": "AssemblyAI", "aws": "AWS Transcribe", "wsp": "WhisperX", "rev": "Reverb"}
NOISE_MAP = {"noise2": "Office", "noise1": "Music"}
METRIC_MAP = {"wer": "WER", "ins_norm": "Insertions", "del_norm": "Deletions", "sub_norm": "Substitutions"}
# keep a list to ensure the order is consistent even if dicts aren't ordered.
XSCRIBERS = list(XSCRIBER_MAP.values())

# %% [markdown]
# ## Load all transcripts
# The transcripts are generated with the cli script in ami.py. 
# 
# Note: AWS and AssemblyAI are run with diarization, so should have speaker labels that we are not loading. 

# %%
def load_all(out=OUT_DIR, split="train"):
    splits = get_splits()
    dfs = []
    session_dirs = []
    for session_id in splits[split]:
        session_dir = out / session_id
        # Rev transcriptions are still running, so many sessions don't yet have them.
        if not (session_dir / "xscript_rev_noise1_snr-20.json").exists():
            continue
        session_dirs.append(session_dir)
        for fn in session_dir.glob("xscript*.feather"):
            dfs.append(pd.read_feather(fn))
    
    return pd.concat(dfs)

# %% [markdown]
# ## Compute WER
# Run jiwer pairwise alignment to the human ground-truth transcript for each machine transcript at all snr levels and put the word error metrics into a data frame. 
# 
# Note that this version runs a bunch of jobs in parallel to speed things up, and each job reads the session data directly from the feather file. This is redundant with the code above, which loads all the data into one big df. That code is useful for getting a df to explore (e.g., down below), but this code is built for speed.

# %%
splits = get_splits()
session_files = []
for session_id in splits["train"]:
    fn = OUT_DIR / session_id / "xscript_noise1.feather"
    if fn.exists():
        session_files.append(fn)
    fn = OUT_DIR / session_id / "xscript_noise2.feather"
    if fn.exists():
        session_files.append(fn)
print(len(session_files))
    
def process_session_snr_from_file(fn, normalizer):
    df = pd.read_feather(fn)
    noisefn = Path(fn).stem.split("_")[1]
    metrics = []
    aligns = []
    xscript_str = {}
    session_id = df.session_id.iloc[0]
    for snr in df.snr.unique():
        # Get transcript strings for all transcribers for this snr value
        for xscriber in list(XSCRIBER_MAP.keys()) + ["hum"]:
            xscript_str[xscriber] = normalizer(" ".join(df.loc[(df.snr == snr) & (df.transcriber == xscriber), "word"]))
        
        # Calculate WER metrics
        for xscriber in XSCRIBER_MAP.keys():
            align = jiwer.process_words(xscript_str["hum"], xscript_str[xscriber])
            metrics.append({
                "session_id": session_id,
                "snr": snr,
                "noise": noisefn,
                "transcriber": xscriber,
                "nref": len(align.references[0]),
                "insertions": align.insertions,
                "deletions": align.deletions,
                "substitutions": align.substitutions,
                "hits": align.hits,
                "wer": align.wer,
                "mer": align.mer,
                "wil": align.wil,
                "wip": align.wip
            })
            adict = {
                "session_id": session_id,
                "snr": snr,
                "noise": noisefn,
                "transcriber": xscriber,
                "alignment": align,
            }
            aligns.append(adict)
    return metrics, aligns


results = Parallel(n_jobs=-1, verbose=1)(
    delayed(process_session_snr_from_file)(fn, normalizer)
    for fn in session_files
)

# unpack the alignments
alignments = [sublist[1] for sublist in results]

# Flatten metrics and create the dataframe
wdf = pd.DataFrame([item for sublist in results for item in sublist[0]])

# Add normalized columns
wdf["ins_norm"] = wdf.insertions / wdf.nref
wdf["del_norm"] = wdf.deletions / wdf.nref
wdf["sub_norm"] = wdf.substitutions / wdf.nref
wdf = wdf.drop_duplicates()
print(wdf.shape)
wdf.head()

# %% [markdown]
# # Make figures and tables

# %%
# Constants for plots below
#colors = ["cornflowerblue", "goldenrod", "seagreen", "indigo"]
FONTCOLOR = 'rgba(0.2,0.2,0.2,1.0)'
GRIDCOLOR = 'rgba(0.8,0.8,0.85,0.5)'
FONTSIZE = 16
CMAP = px.colors.qualitative.Pastel

def get_sum_df(df, noise=None):
    aggers = {"Word Error Rate": ("wer", "mean"), 
              "Insertions": ("ins_norm", "mean"),
              "Deletions": ("del_norm", "mean"),
              "Substitutions": ("sub_norm", "mean"),
              "ci_wor": ("wer", "sem"),
              "ci_ins": ("ins_norm", "sem"),
              "ci_del": ("del_norm", "sem"),
              "ci_sub": ("sub_norm", "sem"),
              "N": ("wer", "count"),
             }
    if noise:
        sumdf = df.loc[df.noise==noise].groupby(["snr", "transcriber"]).agg(**aggers).reset_index()
    else:
        sumdf = df.groupby(["snr", "noise", "transcriber"]).agg(**aggers).reset_index()
    print(f"Num sessions: {sumdf.N.min()}")
    sumdf.transcriber = sumdf.transcriber.replace(XSCRIBER_MAP)
    sumdf = sumdf.rename(columns={"transcriber": "Transcriber"})
    sumdf.loc[:, ["ci_wor", "ci_ins", "ci_del", "ci_sub"]] *= 1.96
    return sumdf

def get_subrules(tab, every_nrows, start_col, end_col, inserts=None, head=3, tail=4):
    lines = tab.split("\n")
    inserts = inserts if inserts else {}
    for i, line in enumerate(lines):    
        if i > head and i < len(lines) - tail and (i - head) % every_nrows == 0:
            inserts[i] = f"\\cmidrule(r){{{start_col}-{end_col}}}"
    return inserts

def add_subrules(tab, inserts):
    lines = tab.split("\n")
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)
        if i in inserts:
            new_lines.append(inserts[i])
    return '\n'.join(new_lines)

# %% [markdown]
# ## Word error rates
# Figure 1

# %%
noises = ["noise2", "noise1"]
tmpdf = get_sum_df(wdf)
metrics = ["Word Error Rate", "Deletions", "Insertions", "Substitutions"]
#titles = ["a.", "b.", "c.", "d.", "e.", "f.", "g.", "h."]
nrows = len(metrics)
ncols = 2
titles = [" " for _ in metrics*2]
titles[::2] = metrics
fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.05, horizontal_spacing=0.03,
                    subplot_titles=titles, x_title="Signal to Noise Ratio (dB)", y_title="Error Rate")

for noise_num, noise in enumerate(noises, start=1):
    noise_mask = tmpdf.noise == noise
    for metric_num, metric in enumerate(metrics, start=1):
        for transcriber_idx, transcriber in enumerate(XSCRIBERS):
            mask = noise_mask & (tmpdf["Transcriber"] == transcriber)
            ci_name = f"ci_{metric[:3].lower()}"
            fig.add_trace(
                go.Scatter(x=tmpdf[mask]["snr"], y=tmpdf[mask][metric], name=transcriber,
                           line=dict(color=CMAP[transcriber_idx]),
                           error_y=dict(type='data', array=tmpdf[mask][ci_name]),
                           showlegend=(noise_num==1)&(metric_num==1)),
                row=metric_num, col=noise_num)

for noise_num, noise in enumerate(noises, start=1):
    noise_mask = tmpdf.noise == noise
    for metric_num, metric in enumerate(metrics, start=1):
        for transcriber_idx, transcriber in enumerate(XSCRIBERS):
            mask = noise_mask & (tmpdf["Transcriber"] == transcriber)
            fig.add_hline(y=tmpdf[mask].loc[tmpdf[mask]["snr"]==float("inf"), metric].values[0], 
                          line_dash="dot", line_color=CMAP[transcriber_idx], row=metric_num, col=noise_num)
            # x0=0.05, x1=0.95, 
                          
fig.update_layout(
    width=1200, 
    height=300*len(metrics),
    font=dict(size=FONTSIZE, color=FONTCOLOR),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(font_size=FONTSIZE-2, yanchor="top", y=0.98, xanchor="left", x=0.3, title="Transcriber")
)
tvs = [-20, -10, 0, 10, 20, 25]
for c in [1, 2]:
    for r in range(1, len(metrics)+1):
        if c == 1:
            fig.update_xaxes(showgrid=True, showline=True, zerolinecolor=GRIDCOLOR, gridcolor=GRIDCOLOR, 
                             row=r, col=c, tickvals=tvs, ticktext=[str(x) if x!=25 else "Inf" for x in tvs])
        else:
            fig.update_xaxes(showgrid=True, showline=True, zerolinecolor=GRIDCOLOR, gridcolor=GRIDCOLOR, row=r, col=c)
        fig.update_yaxes(showgrid=True, showline=True, zerolinecolor=GRIDCOLOR, gridcolor=GRIDCOLOR, row=r, col=c)
     
fig['layout']['margin'] = go.layout.Margin(l=70, r=0, b=70, t=30)
for i in fig['layout']['annotations']:
    i['font'] = dict(size=FONTSIZE+2)
    if i['text'] in metrics:
        i['xanchor'], i['x'] = "center", 0.5
    
fig.write_image(FIGDIR / "error_types.png", scale=3)
fig.show()

# %% [markdown]
# ## Compute t-tests for all

# %%
metrics = ["wer", "ins_norm", "del_norm", "sub_norm"]
icols = ["session_id", "snr", "transcriber"]
snrs = wdf.snr.unique()

# compare office noise with music
vals = []
for metric in metrics:
    office = wdf.loc[(wdf.noise=="noise2"), ["session_id", "snr", "transcriber"]+metrics]
    music = wdf.loc[(wdf.noise=="noise1"), ["session_id", "snr", "transcriber"]+metrics]
    tmpdf = office.merge(music, on=icols, suffixes=("_office", "_music")).dropna()
    for snr in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
        for transcriber in XSCRIBER_MAP.keys():
            tmp2df = tmpdf.loc[(tmpdf.snr == snr) & (tmpdf.transcriber == transcriber)]
            ovals, mvals = tmp2df[f"{metric}_office"].values, tmp2df[f"{metric}_music"].values
            res = stats.ttest_rel(ovals, mvals)
            vals.append({"metric": metric,
                         "SNR": snr,
                         "Transcriber": XSCRIBER_MAP[transcriber],
                         "Mean Office": np.mean(ovals), 
                         "Mean Music": np.mean(mvals), 
                         "t-statistic": res.statistic, 
                         "p-value-unc": res.pvalue, 
                         "df": res.df})
office_music_tdf = pd.DataFrame(vals)


# compare snr 20 with snr inf
vals = []
for noise in NOISE_MAP.keys():
    idx = wdf.noise == noise
    for metric in metrics:
        for transcriber in XSCRIBER_MAP.keys():
            inf = wdf.loc[idx & (wdf.snr==float("inf")) & (wdf.transcriber==transcriber), ["session_id"] + metrics]
            s20 = wdf.loc[idx & (wdf.snr==float("20")) & (wdf.transcriber==transcriber), ["session_id"] + metrics]
            tmpdf = inf.merge(s20, on="session_id", suffixes=("_inf", "_s20")).dropna()
            inf_vals, s20_vals = tmpdf[f"{metric}_inf"], tmpdf[f"{metric}_s20"]
            res = stats.ttest_rel(inf_vals, s20_vals)
            vals.append({"Noise type": NOISE_MAP[noise], 
                         "Metric": METRIC_MAP[metric], 
                         "Transcriber": XSCRIBER_MAP[transcriber],
                         "Mean Original": np.mean(inf_vals), 
                         "Mean SNR 20": np.mean(s20_vals), 
                         "t-statistic": res.statistic, 
                         "p-value-unc": res.pvalue,
                         "df": res.df})
inf20_tdf = pd.DataFrame(vals)


# compare the different transcribers
vals = []
for noise_cnt, noise in enumerate(NOISE_MAP.keys()):
    idx_noise = wdf.noise == noise
    for snr in snrs:
        # To avoid duplicate tests for inf snr (ie. it's the same for all noise types)
        if snr == float("inf") and noise_cnt > 0:
            continue
        for metric in metrics:
            idx = idx_noise & (wdf.snr == snr)
            done = []
            for xs1 in XSCRIBER_MAP.keys():
                for xs2 in XSCRIBER_MAP.keys():
                    if (xs1, xs2) in done or (xs2, xs1) in done or xs1 == xs2:
                        continue
                    xs1_vals = wdf.loc[idx & (wdf.transcriber == xs1), metric]
                    xs2_vals = wdf.loc[idx & (wdf.transcriber == xs2), metric]
                    res = stats.ttest_rel(xs1_vals, xs2_vals)
                    vals.append({"Noise type": NOISE_MAP[noise],
                                 "SNR": snr,
                                 "Metric": METRIC_MAP[metric],
                                 "Transcriber1": XSCRIBER_MAP[xs1],
                                 "Transcriber2": XSCRIBER_MAP[xs2],
                                 "Mean1": xs1_vals.mean(),
                                 "Mean2": xs2_vals.mean(),
                                 "t-statistic": res.statistic, 
                                 "p-value-unc": res.pvalue, 
                                 "df": res.df})
                    done.append((xs1, xs2))
xscribers_tdf = pd.DataFrame(vals)

# %% [markdown]
# ## Apply FDR correction
# 
# We'll use the more conservative method of Benjaminini-Yekutieli (Theorem 1.3) as it is guaranteed to control the FDR even when the p-values are not from independent tests.

# %%
ps = (inf20_tdf["p-value-unc"].tolist() 
      + office_music_tdf["p-value-unc"].tolist() 
      + xscribers_tdf["p-value-unc"].tolist())
# Paired tests on identical values (e.g. a metric that is 0 for every session) give NaN p-values,
# which false_discovery_control rejects. There is no evidence of a difference in that case, so use p=1.
ps = np.nan_to_num(np.asarray(ps, dtype=float), nan=1.0)
ps_adjusted = stats.false_discovery_control(ps, method="by")
start, n = 0, len(inf20_tdf)
inf20_tdf["p-value"] = ps_adjusted[start:start + n]
start += n
n = len(office_music_tdf)
office_music_tdf["p-value"] = ps_adjusted[start:start + n]
start += n
n = len(xscribers_tdf)
xscribers_tdf["p-value"] = ps_adjusted[start:start + n]
assert (xscribers_tdf.df[0] == xscribers_tdf.df).all()

print(f"FDR adjustment to account for {len(ps)} t-tests; {(ps_adjusted<0.05).sum()} significant at 0.05.")
xscribers_tdf.head()

# %% [markdown]
# ## Noise type table

# %%
pformat = lambda p: f"\\textbf{{{p:#8.1e}}}" if p < 0.05 else f"{p:#8.1e}"
cols = ["SNR", "Transcriber", "Mean Office", "Mean Music", "t-statistic", "p-value"]
for metric, tdf in office_music_tdf[cols + ["metric"]].groupby("metric"):
    # don't show duplicated SNR values in consecutive rows
    tdf.SNR = tdf.SNR.astype('object')
    tdf.SNR = tdf.SNR.map(lambda x: f"{x:0.0f}")
    tdf.loc[tdf["SNR"] == tdf["SNR"].shift(1), "SNR"] = " "
    tab = tdf[cols].to_latex(index=False, 
                    formatters={"Mean Office": lambda x: f"{x:0.5f}", 
                                "Mean Music": lambda x: f"{x:0.5f}",
                                "t-statistic": lambda x: f"{x:0.2f}",
                                "p-value": pformat})
    
    tab = add_subrules(tab, get_subrules(tab, 4, 2, len(cols)))
    with open(FIGDIR / f"office_music_{metric}.tex", "wt") as fp:
        fp.write(tab)
if office_music_tdf.df.min() != office_music_tdf.df.max(): 
    print("WARNING! DFs are not all equal.")

# %% [markdown]
# ## SNR inf vs 20 table

# %%
cols = ["Noise type", "Metric", "Transcriber", "Mean Original", "Mean SNR 20", "t-statistic", "p-value"]
tmpdf = inf20_tdf.copy()
# don't show duplicated noise types and metrics in consecutive rows
tmpdf.loc[tmpdf["Noise type"] == tmpdf["Noise type"].shift(1), "Noise type"] = " "
tmpdf.loc[tmpdf["Metric"] == tmpdf["Metric"].shift(1), "Metric"] = " "
tab = tmpdf[cols].to_latex(index=False, 
                    formatters={"Mean Original": lambda x: f"{x:0.4f}", 
                                "Mean SNR 20": lambda x: f"{x:0.4f}",
                                "t-statistic": lambda x: f"{x:0.2f}",
                                "p-value": pformat})
inserts = get_subrules(tab, 4, 3, len(cols))
inserts = get_subrules(tab, 16, 2, len(cols), inserts)
tab = add_subrules(tab, inserts)
with open(FIGDIR / f"inf20.tex", "wt") as fp:
    fp.write(tab) 

# %% [markdown]
# ## Transcribers comparison tables
# Pairwise comparison of the four transcribers.
# 
# **TODO:** maybe put xs1 and xs2 in the same column as "{xs1} vs. {xs2}"
# 
# ### Inf SNR

# %%
cols = ["Metric", "Transcriber1", "Transcriber2", "Mean1", "Mean2", "t-statistic", "p-value"]

tmpdf = xscribers_tdf.loc[xscribers_tdf.SNR == float("inf")].copy()
tmpdf.SNR = tmpdf.SNR.astype('object')
tmpdf.SNR = tmpdf.SNR.map(lambda x: f"{x:0.0f}")
tmpdf.loc[tmpdf["SNR"] == tmpdf["SNR"].shift(1), "SNR"] = " "
tmpdf.loc[tmpdf["Metric"] == tmpdf["Metric"].shift(1), "Metric"] = " "
tmpdf.loc[tmpdf["Transcriber1"] == tmpdf["Transcriber1"].shift(1), "Transcriber1"] = " "
tmpdf.loc[tmpdf["Transcriber2"] == tmpdf["Transcriber2"].shift(1), "Transcriber2"] = " "
tmpdf.Mean1 = tmpdf.Mean1.astype('object').map(lambda x: f"{x:0.4f}")
tmpdf.loc[tmpdf["Mean1"] == tmpdf["Mean1"].shift(1), "Mean1"] = " "
tmpdf.Mean2 = tmpdf.Mean2.astype('object').map(lambda x: f"{x:0.4f}")
tmpdf.loc[tmpdf["Mean2"] == tmpdf["Mean2"].shift(1), "Mean2"] = " "
tab = tmpdf[cols].to_latex(index=False, formatters={
    "t-statistic": lambda x: f"{x:0.2f}", 
    "p-value": pformat})
inserts = get_subrules(tab, 6, 2, len(cols))
tab = add_subrules(tab, inserts)
with open(FIGDIR / f"xscribers_inf.tex", "wt") as fp:
    fp.write(tab)

# %% [markdown]
# ### With noise
# There are too many rows for one table, so we split it into the lower and higher SNR levels. We had to drop one level to fit neatly into two tables with 4 snr levels each, so we drop snr -20 since it is basically unusable so not very interesting.

# %%
cols = ["Noise type", "SNR", "Transcriber1", "Transcriber2", "t-statistic", "p-value"]
show_snrs_list = {"low": [-15, -10, -5, 0], "high": [5, 10, 15, 20]}

for metric, tdf in xscribers_tdf.groupby("Metric"):
    for label, show_snrs in show_snrs_list.items():
        tmpdf = tdf.loc[tdf.SNR.isin(show_snrs)].copy().sort_values(["Noise type", "SNR"])
        tmpdf.loc[tmpdf["Transcriber1"] == tmpdf["Transcriber1"].shift(1), "Transcriber1"] = " "
        tmpdf.loc[tmpdf["Transcriber2"] == tmpdf["Transcriber2"].shift(1), "Transcriber2"] = " "
        tmpdf.SNR = tmpdf.SNR.astype('object')
        tmpdf.SNR = tmpdf.SNR.map(lambda x: f"{x:0.0f}")
        tmpdf.loc[tmpdf["Noise type"] == tmpdf["Noise type"].shift(1), "Noise type"] = " "
        tmpdf.loc[tmpdf["SNR"] == tmpdf["SNR"].shift(1), "SNR"] = " "
        
        tab = tmpdf[cols].to_latex(index=False, formatters={"t-statistic": lambda x: f"{x:0.2f}", 
                                                            "p-value": pformat})
        inserts = get_subrules(tab, 6, 3, len(cols))
        inserts = get_subrules(tab, 6*len(show_snrs), 2, len(cols), inserts)
        tab = add_subrules(tab, inserts)
        with open(FIGDIR / f"xscribers_{label}_{metric.lower()}.tex", "wt") as fp:
            fp.write(tab)

# %% [markdown]
# ## ASR timestamp analysis

# %% [markdown]
# Word timestamp differences ("jitter") between the human reference and each ASR transcript.
# For each session file, SNR, and ASR, align the normalized word sequences with jiwer and, for every
# run of consecutive matching words ("equal" chunk), record the ASR minus reference start time of each
# matched word pair, along with the length of the run it belongs to.

# %%
def compute_jitter_from_file(fn, normalizer):
    df = pd.read_feather(fn)
    noisefn = Path(fn).stem.split("_")[1]
    session_id = df.session_id.iloc[0]
    rows = []

    def norm_words(sub):
        if len(sub) == 0:
            return sub
        # Normalize in stored order (missing-time filling relies on neighbors), then sort by time
        sub = normalize_xscript(sub.reset_index(drop=True), normalizer)
        if len(sub) == 0:
            return sub
        return sub.sort_values("start_time", kind="stable").reset_index(drop=True)

    for snr in df.snr.unique():
        snr_df = df.loc[df.snr == snr]
        ref = norm_words(snr_df.loc[snr_df.transcriber == "hum"])
        if len(ref) == 0:
            continue
        for xscriber in XSCRIBER_MAP.keys():
            hyp = norm_words(snr_df.loc[snr_df.transcriber == xscriber])
            if len(hyp) == 0:
                continue
            align = jiwer.process_words(" ".join(ref.word_norm), " ".join(hyp.word_norm))
            # word_norm is a single non-empty token, so jiwer's default transform (split on
            # whitespace) must index one-to-one with the DataFrame rows.
            assert len(align.references[0]) == len(ref) and len(align.hypotheses[0]) == len(hyp)
            ref_start = ref.start_time.to_numpy()
            hyp_start = hyp.start_time.to_numpy()
            for chunk in align.alignments[0]:
                if chunk.type != "equal":
                    continue
                length = chunk.ref_end_idx - chunk.ref_start_idx
                jit = hyp_start[chunk.hyp_start_idx:chunk.hyp_end_idx] - ref_start[chunk.ref_start_idx:chunk.ref_end_idx]
                rows.extend({"session_id": session_id,
                             "noise_file": noisefn,
                             "snr": snr,
                             "transcriber": xscriber,
                             "length": length,
                             "jitter": j} for j in jit)
    return rows


results = Parallel(n_jobs=-1, verbose=1)(
    delayed(compute_jitter_from_file)(fn, normalizer)
    for fn in session_files
)
jitdf = pd.DataFrame([row for sublist in results for row in sublist])
jitdf["delta_ms"] = jitdf.jitter * 1000
del results
print(jitdf.shape)
jitdf.head()

# %% [markdown]
# ## Jitter histogram plot

# %%
noise = "noise2"
DELTA_THRESH = 10
snrs = [float("inf"), 0]
matlens = [0, 25, 50]
#cmap = ["rgba(.2,.2,.3,.5)", "rgba(.0,.6,.1,.5)", "rgba(.4,.4,.0,.5)"] #["rgba(.0,.1,.7,.5)", "rgba(.1,.5,.1,.5)"]
cmap = ["rgba(.7,.7,.8,.5)", "rgba(.0,.6,.1,.5)", "rgba(0,.1,.7,.5)"]
titles = ["a.", "b.", "c.", "d.", "e.", "f.", "g.", "h."]
nrows = len(XSCRIBER_MAP)
ncols = 2

xbins = dict(start=0, end=DELTA_THRESH+0.1, size=0.1)
noise_idx = jitdf.noise_file == noise

fig = make_subplots(rows=nrows, cols=ncols, vertical_spacing=0.05, shared_xaxes=True, shared_yaxes=True, 
                    x_title="Timestamp Difference (seconds)", y_title="Count",
                    subplot_titles=titles,
                   )

for snr_cnt, snr in enumerate(snrs):
    snr_idx = jitdf.snr == snr

    for len_cnt, len_thresh in enumerate(matlens):
        len_idx = jitdf.length > len_thresh
        label = f">{len_thresh} matches" if len_thresh > 0 else "All matches"
        
        for xscriber_cnt, xscriber in enumerate(XSCRIBER_MAP.keys()):
            if len_thresh == 0:
                print(f"{snr=}, {label=}, {xscriber=}, {titles[xscriber_cnt*ncols + snr_cnt]}")
            x = jitdf.loc[(jitdf.transcriber == xscriber) & noise_idx & snr_idx & len_idx, "jitter"].abs()
            fig.add_trace(go.Histogram(x=np.clip(x, 0, DELTA_THRESH), name=label, marker_color=cmap[len_cnt], #opacity=0.75, 
                                       autobinx=False, xbins=xbins, showlegend=(snr_cnt==0)&(xscriber_cnt==0)), 
                          row=xscriber_cnt+1, col=snr_cnt+1)

for r in range(nrows):
    for c in range(ncols):
        fig.update_xaxes(showgrid=True, showline=True, zerolinecolor=GRIDCOLOR, gridcolor=GRIDCOLOR, row=r+1, col=c+1)
        fig.update_yaxes(showgrid=True, showline=True, zerolinecolor=GRIDCOLOR, gridcolor=GRIDCOLOR, row=r+1, col=c+1, type="log")

fig.update_layout(
    height=900, width=1000,
    font=dict(size=FONTSIZE, color=FONTCOLOR),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    barmode='overlay',
    showlegend=True,
    legend=dict(font_size=FONTSIZE-2, yanchor="top", y=1.0, xanchor="right", x=0.99, title="Match Length")
)

fig['layout']['margin'] = go.layout.Margin(l=70, r=0, b=70, t=30)
# adjust title positions
for a in fig['layout']['annotations']:
    if a["text"] in titles[::2]:
        #a["y"] -= 0.05
        a["x"], a["xanchor"] = 0.01, "left"
    elif a["text"] in titles[1::2]:
        a["x"], a["xanchor"] = 0.56, "left"
    #a['font'] = dict(size=FONTSIZE+2)

# Sometimes plotly shows extra ticks for a log axis that we can clean up with the following
#fig.update_layout(yaxis4=dict(tickvals=[1, 10, 100, 1000]))

fig.write_image(FIGDIR / "jitter_hist.png", scale=3)
    
#fig.show()

# %%
noise = "noise2"
DELTA_THRESH = 5
xbins = dict(start=-DELTA_THRESH-0.1, end=DELTA_THRESH+0.1, size=0.1)
noise_idx = jitdf.noise_file == noise

fig = make_subplots(rows=nrows, cols=ncols, vertical_spacing=0.05, shared_xaxes=True, shared_yaxes=True, 
                    x_title="Timestamp Difference (seconds)", y_title="Count",
                    subplot_titles=titles,
                   )

for snr_cnt, snr in enumerate(snrs):
    snr_idx = jitdf.snr == snr

    for len_cnt, len_thresh in enumerate(matlens):
        len_idx = jitdf.length > len_thresh
        label = f">{len_thresh} matches" if len_thresh > 0 else "All matches"
        
        for xscriber_cnt, xscriber in enumerate(XSCRIBER_MAP.keys()):
            if len_thresh == 0:
                print(f"{snr=}, {label=}, {xscriber=}, {titles[xscriber_cnt*ncols + snr_cnt]}")
            x = jitdf.loc[(jitdf.transcriber == xscriber) & noise_idx & snr_idx & len_idx, "jitter"]
            fig.add_trace(go.Histogram(x=np.clip(x, -DELTA_THRESH, DELTA_THRESH), name=label, marker_color=cmap[len_cnt], 
                                       autobinx=False, xbins=xbins, showlegend=(snr_cnt==0)&(xscriber_cnt==0)), 
                          row=xscriber_cnt+1, col=snr_cnt+1)

for r in range(nrows):
    for c in range(ncols):
        fig.update_xaxes(showgrid=True, showline=True, zerolinecolor=GRIDCOLOR, gridcolor=GRIDCOLOR, row=r+1, col=c+1)
        fig.update_yaxes(showgrid=True, showline=True, zerolinecolor=GRIDCOLOR, gridcolor=GRIDCOLOR, row=r+1, col=c+1, type="log")

fig.update_layout(
    height=900, width=1000,
    font=dict(size=FONTSIZE, color=FONTCOLOR),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    barmode='overlay',
    showlegend=True,
    legend=dict(font_size=FONTSIZE-2, yanchor="top", y=1.0, xanchor="right", x=0.99, title="Match Length")
)

fig['layout']['margin'] = go.layout.Margin(l=70, r=0, b=70, t=30)
# adjust title positions
for a in fig['layout']['annotations']:
    if a["text"] in titles[::2]:
        #a["y"] -= 0.05
        a["x"], a["xanchor"] = 0.01, "left"
    elif a["text"] in titles[1::2]:
        a["x"], a["xanchor"] = 0.56, "left"
    #a['font'] = dict(size=FONTSIZE+2)

# Sometimes plotly shows extra ticks for a log axis that we can clean up with the following
#fig.update_layout(yaxis4=dict(tickvals=[1, 10, 100, 1000]))

fig.write_image(FIGDIR / "jitter_hist_neg.png", scale=3)

# %% [markdown]
# ## Jitter distribution table
# 95% and 99% quantiles of |jitter| for each ASR at various consecutive word-match thresholds, for the
# original audio (SNR inf) and with each noise type at SNR 0.

# %%
matlens = [0, 25, 50]
# SNR inf audio is identical for both noise files, so only use one copy of it to avoid double counting.
conditions = [("Original", "noise2", float("inf"))] + [(f"{NOISE_MAP[n]}, SNR 0", n, 0) for n in ["noise2", "noise1"]]
vals = []
for cond_label, noise_file, snr in conditions:
    cond_idx = (jitdf.noise_file == noise_file) & (jitdf.snr == snr)
    for len_thresh in matlens:
        len_idx = jitdf.length > len_thresh
        for xscriber in XSCRIBER_MAP.keys():
            x = jitdf.loc[cond_idx & len_idx & (jitdf.transcriber == xscriber), "jitter"].abs()
            vals.append({"Condition": cond_label,
                         "Matches": f">{len_thresh}" if len_thresh > 0 else "All",
                         "Transcriber": XSCRIBER_MAP[xscriber],
                         "95% Quantile": x.quantile(0.95),
                         "99% Quantile": x.quantile(0.99),
                         "N": len(x)})
jitq_df = pd.DataFrame(vals)

cols = ["Condition", "Matches", "Transcriber", "95% Quantile", "99% Quantile", "N"]
tmpdf = jitq_df.copy()
# don't show duplicated conditions and match thresholds in consecutive rows
tmpdf.loc[tmpdf["Matches"] == tmpdf["Matches"].shift(1), "Matches"] = " "
tmpdf.loc[tmpdf["Condition"] == tmpdf["Condition"].shift(1), "Condition"] = " "
# escape "%" for LaTeX; quantiles are in seconds
tmpdf = tmpdf[cols].rename(columns={"95% Quantile": "95\\% Quantile (s)", "99% Quantile": "99\\% Quantile (s)"})
tab = tmpdf.to_latex(index=False,
                     formatters={"95\\% Quantile (s)": lambda x: f"{x:0.3f}",
                                 "99\\% Quantile (s)": lambda x: f"{x:0.3f}",
                                 "N": lambda x: f"{x:d}"})
n_xs = len(XSCRIBER_MAP)
inserts = get_subrules(tab, n_xs, 2, len(cols))
inserts = get_subrules(tab, n_xs * len(matlens), 1, len(cols), inserts)
tab = add_subrules(tab, inserts)
with open(FIGDIR / "jitter_quantiles.tex", "wt") as fp:
    fp.write(tab)
jitq_df
