"""Shared helpers for agentic-graph evaluation (extracted from run_agentic_eval.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_graph_monthly import build_factor
from scripts.run_rq3_mild_rerun import mild_neutralize
from scripts.run_us_ema_mr_to_jp_mvp import rank_ic_series, summarize


def build_agentic_adj(agentic_pairs, target_syms, peer_syms):
    """Return (strength_matrix, sampled_jp_set).

    strength_matrix[i, j] = agent linkage_strength for pair (target_syms[i], peer_syms[j]),
    or NaN if the pair was not assessed by the agent.
    """
    t2i = {s: i for i, s in enumerate(target_syms)}
    # US peer uses ticker base; similarity cache has suffix like ".OQ". Match on base.
    base2j: dict[str, int] = {}
    for j, s in enumerate(peer_syms):
        base = s.split(".")[0]
        if base not in base2j:
            base2j[base] = j

    S = np.full((len(target_syms), len(peer_syms)), np.nan)
    sampled = set()
    for r in agentic_pairs:
        if "linkage_type" not in r:
            continue
        jp = r["jp_ticker"]
        us_base = r["us_ticker"].split(".")[0]
        i = t2i.get(jp)
        j = base2j.get(us_base)
        if i is None or j is None:
            continue
        sampled.add(jp)
        strength = float(r.get("linkage_strength", 0.0) or 0.0)
        S[i, j] = strength
    return S, sampled


def variant_emb_only(emb_w, S, sampled_idx):
    return emb_w.copy()


def variant_filter(emb_w, S, sampled_idx, min_strength=0.3):
    """For sampled JP rows, keep edges with strength >= min_strength; drop others.
    If a row becomes empty, fall back to original emb_w row."""
    W = emb_w.copy()
    for i in sampled_idx:
        row_S = S[i]
        assessed = ~np.isnan(row_S)
        if not assessed.any():
            continue
        keep = assessed & (row_S >= min_strength)
        new_row = emb_w[i].copy()
        # drop the top-K assessed edges that did not pass filter
        new_row[assessed & ~keep] = 0.0
        if new_row.sum() < 1e-9:
            continue  # fallback to emb_w row
        W[i] = new_row
    return W


def variant_strength_boost(emb_w, S, sampled_idx, beta=2.0):
    """Multiplicative boost by agent strength: W = emb * (1 + beta * strength).
    Unassessed edges unchanged."""
    W = emb_w.copy()
    for i in sampled_idx:
        row_S = S[i]
        assessed = ~np.isnan(row_S)
        if not assessed.any():
            continue
        boost = np.where(assessed, 1.0 + beta * np.nan_to_num(row_S, nan=0.0), 1.0)
        W[i] = emb_w[i] * boost
    return W


def variant_confirmed_edge_boost(emb_w, S, sampled_idx, beta=0.2, min_strength=None):
    """Multiplicative boost by confirmed Linker edge: W = emb * (1 + beta * A).

    A confirmed edge is an assessed pair returned by the Linker. If min_strength
    is provided, only assessed pairs with strength >= min_strength are boosted.
    """
    W = emb_w.copy()
    for i in sampled_idx:
        row_S = S[i]
        assessed = ~np.isnan(row_S)
        if min_strength is not None:
            assessed = assessed & (row_S >= min_strength)
        if not assessed.any():
            continue
        W[i] = emb_w[i] * (1.0 + beta * assessed.astype(np.float64))
    return W


def variant_filter_and_boost(emb_w, S, sampled_idx, min_strength=0.3, beta=2.0):
    """Drop weak edges AND boost surviving ones by strength."""
    W = emb_w.copy()
    for i in sampled_idx:
        row_S = S[i]
        assessed = ~np.isnan(row_S)
        if not assessed.any():
            continue
        keep = assessed & (row_S >= min_strength)
        new_row = emb_w[i].copy()
        new_row[assessed & ~keep] = 0.0
        boost = np.where(keep, 1.0 + beta * np.nan_to_num(row_S, nan=0.0), 1.0)
        new_row = new_row * boost
        if new_row.sum() < 1e-9:
            continue
        W[i] = new_row
    return W


def static_graph_boost(emb_w, target_syms, peer_syms, opus_path, beta=0.2):
    opus = json.load(open(opus_path))
    t2i = {s: i for i, s in enumerate(target_syms)}
    u2j = {}
    for uc in opus:
        for j, s in enumerate(peer_syms):
            if s.startswith(uc + ".") or s == uc:
                u2j[uc] = j
                break
    A = np.zeros_like(emb_w)
    for uc, info in opus.items():
        j = u2j.get(uc)
        if j is None:
            continue
        for c in info.get("connections", []):
            i = t2i.get(c.get("jp_ticker", ""))
            if i is not None:
                A[i, j] = 1.0
    return emb_w * (1.0 + beta * A)


def eval_weights_subuniverse(name, W, target_syms, peer_syms, peer_returns,
                              months, labels, lncap, jp_sectors, sub_syms=None):
    fdf = build_factor(W, target_syms, peer_syms, peer_returns, months)
    if sub_syms is not None:
        fdf = fdf.filter(pl.col("SYMBOL").is_in(list(sub_syms)))
    f_n = mild_neutralize(fdf, lncap, jp_sectors)
    ic_n = rank_ic_series(f_n, labels)
    return summarize(ic_n, name)
