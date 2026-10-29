"""Language-model perplexity of word sequences, used to compare the fluency of ASR
hallucination sequences with same-length human transcript sequences.

Requires the optional `transformers` dependency (pip install -e ".[perplexity]").
"""

from pathlib import Path

import numpy as np
import pandas as pd

# The paper scored sequences with BERT-large through the Hugging Face `evaluate` perplexity metric
# (add_start_token=False). BERT is a masked LM; loaded as a causal LM it is used to predict each
# next token, so its perplexities are much higher than a true causal LM's (e.g. gpt2).
DEFAULT_MODEL_ID = "google-bert/bert-large-uncased"


def compute_perplexity(texts, model_id=DEFAULT_MODEL_ID, batch_size=64, add_start_token=False,
                       max_length=None, device=None):
    """Per-text perplexity under a causal language model.

    Reproduces the Hugging Face `evaluate` perplexity metric (cited in the paper; the defaults
    match the paper's call): each text is tokenized (optionally prefixed with the BOS token so the
    first word is also scored) and its perplexity is exp(mean next-token negative log-likelihood).
    Without a start token, each text needs at least two tokens.

    Args:
        texts: list of strings
        model_id: Hugging Face causal LM id
        batch_size: texts per forward pass
        add_start_token: prepend the BOS token to each text (needs a model with a BOS token)
        max_length: truncate texts to this many tokens (default: model limit)
        device: torch device (default: cuda, then mps, then cpu)

    Returns:
        numpy array of perplexities, one per text
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if max_length is None:
        max_length = model.config.max_position_embeddings
    if add_start_token:
        max_length -= 1  # leave room for BOS

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    ppls = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(list(texts[start:start + batch_size]), add_special_tokens=False, padding=True,
                        truncation=True, max_length=max_length, return_tensors="pt").to(device)
        input_ids, attn_mask = enc.input_ids, enc.attention_mask
        if add_start_token:
            bos = torch.full((input_ids.size(0), 1), tokenizer.bos_token_id, device=device)
            input_ids = torch.cat([bos, input_ids], dim=1)
            attn_mask = torch.cat([torch.ones_like(bos), attn_mask], dim=1)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attn_mask).logits

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attn_mask[:, 1:].contiguous()
        nll = loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_mask
        ppls.append(torch.exp(nll.sum(1) / shift_mask.sum(1)).float().cpu().numpy())

    return np.concatenate(ppls) if ppls else np.array([])


def cached_perplexity(texts, cache_path, **kwargs):
    """compute_perplexity() with a persistent text -> perplexity cache (a feather file).

    Only texts missing from the cache are scored, so duplicates are free and an
    interrupted run can be resumed. kwargs are passed to compute_perplexity(); keep
    them fixed for a given cache file.
    """
    cache_path = Path(cache_path)
    cache = {}
    if cache_path.exists():
        cdf = pd.read_feather(cache_path)
        cache = dict(zip(cdf.text, cdf.perplexity))
    missing = list(dict.fromkeys(t for t in texts if t not in cache))
    if missing:
        cache.update(zip(missing, compute_perplexity(missing, **kwargs)))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"text": list(cache.keys()), "perplexity": list(cache.values())}).to_feather(cache_path)
    return np.array([cache[t] for t in texts])


def sample_word_sequences(word_lists, length, n, rng=None):
    """Randomly sample n sequences of `length` consecutive words.

    Windows never cross list boundaries (e.g. sessions). Sampling is without replacement
    when there are enough windows, otherwise with replacement.

    Args:
        word_lists: list of word lists (e.g. one normalized human transcript per session)
        length: words per sequence
        n: number of sequences
        rng: numpy Generator or seed

    Returns:
        list of n space-joined strings
    """
    rng = np.random.default_rng(rng)
    n_windows = np.array([max(len(words) - length + 1, 0) for words in word_lists])
    total = n_windows.sum()
    if total == 0:
        raise ValueError(f"No word list has {length} or more words.")
    flat_idx = rng.choice(total, size=n, replace=n > total)
    offsets = np.cumsum(n_windows)
    list_idx = np.searchsorted(offsets, flat_idx, side="right")
    starts = flat_idx - np.concatenate([[0], offsets[:-1]])[list_idx]
    return [" ".join(word_lists[li][s:s + length]) for li, s in zip(list_idx, starts)]


def histogram_overlap(x, y, bins):
    """Overlap (0-1) of the normalized histograms of x and y: sum over bins of min(p_x, p_y)."""
    px = np.histogram(x, bins=bins)[0] / len(x)
    py = np.histogram(y, bins=bins)[0] / len(y)
    return np.minimum(px, py).sum()
