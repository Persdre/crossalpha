#!/usr/bin/env python3
"""Evaluate agentic-weighted peer graphs against emb_only and static graph baselines.

Loads agentic_linkages_n50_k20.json, builds four agent-weighted variants of the
w128 similarity matrix (applied only to the 50 sampled JP targets; other 403
targets retain the embedding baseline), and compares monthly cross-sectional
Rank IC under MILD neutralization.

Usage:
    python scripts/run_agentic_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.kernels import load_monthly_close_prices, load_symbol_sectors, sigmoid_weights
from scripts.kernels.returns import sector_relative_return_df
from scripts.similarity_cache import load_similarity_cache
from scripts.h5_data.h5_utils import h5_load
from scripts.run_graph_monthly import build_true_lncap
from scripts.agentic_eval_helpers import (
    build_agentic_adj, variant_emb_only, variant_filter, variant_strength_boost,
    variant_filter_and_boost, static_graph_boost, eval_weights_subuniverse,
)

DATA_ROOT = Path("${CROSSALPHA_DATA_ROOT}")
CACHE_SIM_DIR = DATA_ROOT / "cache" / "similarity"
SYMBOL_DICT_PATH = PROJECT_ROOT / "docs" / "SymbolDict.csv"
TARGET_REGION = "jp_topix_500"
PEER_REGION = "us_russell_1000"
START, END = "2015-01-01", "2025-12-31"
C, K = 0.99, 50.0
LOOKBACK = 12

AGENTIC_JSON = PROJECT_ROOT / "backtest_results_daily" / "agentic_linkages_n50_k20.json"


def main():
    print("=" * 100)
    print("AGENTIC SEARCH — EVALUATION vs emb_only and static graph baselines")
    print("=" * 100)

    # Data
    jp_sectors = (
        pl.read_csv(SYMBOL_DICT_PATH)
        .filter(pl.col("region") == TARGET_REGION)
        .select(["SYMBOL", "gics_sector"])
        .filter(pl.col("gics_sector").is_not_null() & (pl.col("gics_sector") != ""))
    )
    jp_m = load_monthly_close_prices(TARGET_REGION, "2012-01-01", END)
    us_m = load_monthly_close_prices(PEER_REGION, "2012-01-01", END)
    months = sorted(
        jp_m.filter((pl.col("DATETIME") >= START) & (pl.col("DATETIME") <= END))
        .select("DATETIME").unique()["DATETIME"].to_list()
    )
    labels = h5_load(
        {"source_path": str(PROJECT_ROOT / "labels"), "product": "Equity",
         "region": TARGET_REGION, "freq": "1mo", "source": "tradable_close_return"},
        start_date=START, end_date=END, keys=["tradable_close_return"],
    )
    lncap = build_true_lncap(TARGET_REGION, months)

    us_sectors = load_symbol_sectors(PEER_REGION)
    cum_us = (
        us_m.sort(["SYMBOL", "DATETIME"])
        .with_columns(
            ((pl.col("close") / pl.col("close").shift(LOOKBACK)) - 1)
            .over("SYMBOL").alias("cum_return")
        )
        .select(["DATETIME", "SYMBOL", "cum_return"])
        .filter(pl.col("DATETIME").is_in(months))
        .join(us_sectors, on="SYMBOL", how="left")
    )
    sr = sector_relative_return_df(cum_us, sector_col="sector", return_col="cum_return")
    peer_returns = {}
    for m in months:
        md = sr.filter(pl.col("DATETIME") == m)
        if md.height > 0:
            peer_returns[m] = (md["SYMBOL"].to_list(),
                               md["sector_relative_return"].to_numpy())

    cached = load_similarity_cache(TARGET_REGION, PEER_REGION, "w128", cache_dir=CACHE_SIM_DIR)
    emb_w = sigmoid_weights(cached.pct_ranks, k=K, c=C)
    target_syms, peer_syms = cached.target_symbols, cached.peer_symbols
    t2i = {s: i for i, s in enumerate(target_syms)}

    # Agentic
    agentic = json.load(open(AGENTIC_JSON))
    S, sampled = build_agentic_adj(agentic["pairs"], target_syms, peer_syms)
    sampled_idx = [t2i[s] for s in sampled if s in t2i]
    print(f"Sampled JP targets with agentic weights: {len(sampled)}")
    print(f"Agentic edges assessed: {int((~np.isnan(S)).sum())}")
    print(f"Agentic strong edges (strength>=0.6): {int(np.nansum(S >= 0.6))}")

    opus_path = PROJECT_ROOT / "backtest_results_daily" / "supply_chain_opus_full.json"

    variants = {
        "emb_only": variant_emb_only(emb_w, S, sampled_idx),
        "static_boost_02": static_graph_boost(emb_w, target_syms, peer_syms, opus_path, 0.2),
        "agentic_filter_03": variant_filter(emb_w, S, sampled_idx, 0.3),
        "agentic_filter_06": variant_filter(emb_w, S, sampled_idx, 0.6),
        "agentic_strength_b10": variant_strength_boost(emb_w, S, sampled_idx, 1.0),
        "agentic_strength_b20": variant_strength_boost(emb_w, S, sampled_idx, 2.0),
        "agentic_strength_b50": variant_strength_boost(emb_w, S, sampled_idx, 5.0),
        "agentic_filter_boost_03_b20": variant_filter_and_boost(emb_w, S, sampled_idx, 0.3, 2.0),
        "agentic_filter_boost_06_b20": variant_filter_and_boost(emb_w, S, sampled_idx, 0.6, 2.0),
    }

    # Sub-universe eval: only the 50 sampled JP targets
    print("\n--- SUB-UNIVERSE EVAL (50 JP sampled targets only) ---")
    print(f"{'Variant':<32}{'n_mo':>6}{'IC':>10}{'ICIR':>9}{'t':>7}")
    print("-" * 70)
    sub_results = {}
    for name, W in variants.items():
        s = eval_weights_subuniverse(
            name + "_sub", W, target_syms, peer_syms, peer_returns,
            months, labels, lncap, jp_sectors, sub_syms=sampled
        )
        sub_results[name] = s
        print(f"{name:<32}{s.get('n_months',0):>6}"
              f"{s.get('mean_ic', float('nan')):>+10.4f}"
              f"{s.get('icir_annual', float('nan')):>+9.3f}"
              f"{s.get('t_stat', float('nan')):>+7.2f}")

    # Full-universe eval: 453 JP, but only 50 have modified weights
    print("\n--- FULL-UNIVERSE EVAL (all 453 JP; agentic applied to 50 sampled rows only) ---")
    print(f"{'Variant':<32}{'n_mo':>6}{'IC':>10}{'ICIR':>9}{'t':>7}")
    print("-" * 70)
    full_results = {}
    for name, W in variants.items():
        s = eval_weights_subuniverse(
            name + "_full", W, target_syms, peer_syms, peer_returns,
            months, labels, lncap, jp_sectors, sub_syms=None
        )
        full_results[name] = s
        print(f"{name:<32}{s.get('n_months',0):>6}"
              f"{s.get('mean_ic', float('nan')):>+10.4f}"
              f"{s.get('icir_annual', float('nan')):>+9.3f}"
              f"{s.get('t_stat', float('nan')):>+7.2f}")

    # Summary
    base_sub = sub_results["emb_only"]["icir_annual"]
    base_full = full_results["emb_only"]["icir_annual"]
    print("\n" + "=" * 100)
    print("SUMMARY — ΔICIR vs emb_only baseline (MILD neutralization)")
    print("=" * 100)
    print(f"{'Variant':<32}{'SubUni ICIR':>12}{'Δ sub':>9}{'Full ICIR':>11}{'Δ full':>9}")
    print("-" * 80)
    for name in variants:
        s = sub_results[name]["icir_annual"]
        f = full_results[name]["icir_annual"]
        ds = s - base_sub
        df = f - base_full
        print(f"{name:<32}{s:>+12.3f}{ds:>+9.3f}{f:>+11.3f}{df:>+9.3f}")

    out = PROJECT_ROOT / "backtest_results_daily" / "agentic_eval.json"
    json.dump(
        {"sub_universe": sub_results, "full_universe": full_results,
         "n_sampled_jp": len(sampled)},
        open(out, "w"), indent=2, default=str,
    )
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
