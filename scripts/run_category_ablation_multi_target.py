#!/usr/bin/env python3
"""Multi-target per-category ablation under MILD neutralization.

For each target T in {JP, TW, KR, HK} and each peer P in {US, T (self-masked)}:
  - For each of 10 Distiller categories, build single-category US->T or T->T factor
  - Apply mild_neutralize (sector + lnCap residual)
  - Run full cross-section backtest
  - Plus all_10_equal joint factor for reference

This generalises run_category_ablation_mild_neutralized.py to all 4 Asian targets,
giving the per-category × cross-market vs same-market matrix that supports the
"cross > single market" claim across all targets in RQ1.

Output: backtest_results_daily/category_ablation_multi_target_12mo.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.kernels import load_monthly_close_prices, load_symbol_sectors, sigmoid_weights
from scripts.kernels.similarity import cosine_similarity_matrix
from scripts.kernels.ranking import percentile_rank_fast
from scripts.kernels.returns import sector_relative_return_df
from scripts.h5_data.h5_utils import h5_load
from scripts.similarity_cache import load_similarity_cache
from scripts.backtesting.backtest_utils import run_cross_section_backtest
from scripts.run_graph_monthly import build_factor, build_true_lncap
from scripts.run_rq3_mild_rerun import mild_neutralize

DATA_ROOT = Path("${CROSSALPHA_DATA_ROOT}")
CACHE_SIM_DIR = DATA_ROOT / "cache" / "similarity"
SYMBOL_DICT_PATH = PROJECT_ROOT / "docs" / "SymbolDict.csv"
START, END = "2015-01-01", "2025-12-31"
C, K = 0.99, 50.0
EMBEDDING_DIR = "128_dimensions"
LOOKBACK = 12
FEES = [0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.0010, 0.0015, 0.0020]
LAYERS, LAG, ANNUAL_DAYS = 5, 1, 12

TARGETS = ["jp_topix_500", "tw_twse", "kr_kospi", "hk_main"]

CATEGORIES = [
    "main_business_segments",
    "core_technologies_production_methods",
    "primary_customers_markets",
    "geographic_coverage",
    "supply_chain_position",
    "strategic_focus_rd_direction",
    "revenue_model",
    "key_competitors_industry_positioning",
    "financial_scale_growth_profile",
    "value_proposition_product_differentiation",
]

SCALAR = ["rank_ic", "rank_ic_p_value", "rank_icir_annual", "rank_ic_winratio"]
COSTED = [
    "long_short_ret_annual", "long_short_ret_sharpe", "long_short_ret_max_dd",
    "long_ret_annual", "long_ret_sharpe", "long_ret_max_dd",
]


def load_category_embeddings(region, category):
    path = PROJECT_ROOT / "embeddings" / EMBEDDING_DIR / category / f"{region}_{category}.h5"
    if not path.exists():
        return None, None
    with h5py.File(path, "r") as f:
        symbols = json.loads(f["embeddings"].attrs["symbols"])
        embeddings = f["embeddings"][:].astype(np.float64)
    return symbols, embeddings


def self_mask_matrix(target_symbols, peer_symbols):
    p2i = {s: i for i, s in enumerate(peer_symbols)}
    m = np.zeros((len(target_symbols), len(peer_symbols)), dtype=np.float64)
    for i, s in enumerate(target_symbols):
        j = p2i.get(s)
        if j is not None:
            m[i, j] = 1.0
    return m


def peer_returns_for_region(region, lookback, months):
    pm = load_monthly_close_prices(region, "2012-01-01", END)
    secs = load_symbol_sectors(region)
    cum = (pm.sort(["SYMBOL", "DATETIME"])
        .with_columns(((pl.col("close") / pl.col("close").shift(lookback)) - 1)
                      .over("SYMBOL").alias("cum_return"))
        .select(["DATETIME", "SYMBOL", "cum_return"])
        .filter(pl.col("DATETIME").is_in(months))
        .join(secs, on="SYMBOL", how="left"))
    sr = sector_relative_return_df(cum, sector_col="sector", return_col="cum_return")
    out = {}
    for m in months:
        md = sr.filter(pl.col("DATETIME") == m)
        if md.height > 0:
            out[m] = (md["SYMBOL"].to_list(), md["sector_relative_return"].to_numpy())
    return out


def target_sectors_df(region: str) -> pl.DataFrame:
    return (pl.read_csv(SYMBOL_DICT_PATH)
            .filter(pl.col("region") == region)
            .select(["SYMBOL", "gics_sector"])
            .filter(pl.col("gics_sector").is_not_null() & (pl.col("gics_sector") != "")))


def target_months(region: str):
    pm = load_monthly_close_prices(region, "2012-01-01", END)
    return sorted(pm.filter((pl.col("DATETIME") >= START) & (pl.col("DATETIME") <= END))
                  .select("DATETIME").unique()["DATETIME"].to_list())


def target_labels(region: str):
    for src, key in [("tradable_close_return", "tradable_close_return"),
                     ("close_return", "close_return")]:
        if (PROJECT_ROOT / "labels" / "Equity" / region / "1mo" / src).exists():
            return h5_load({"source_path": str(PROJECT_ROOT / "labels"), "product": "Equity",
                "region": region, "freq": "1mo", "source": src},
                start_date=START, end_date=END, keys=[key])
    raise FileNotFoundError(f"no labels for {region}")


def extract_metrics(p: Path) -> dict:
    if not p.exists():
        return {}
    metrics = json.load(open(p)).get("metrics", {})
    out = {}
    for k in SCALAR:
        v = metrics.get(k, {}).get("by_cost", {}).get("0.0 bp")
        out[k] = float(v) if v is not None and np.isfinite(v) else None
    for k in COSTED:
        for c in ("0.0 bp", "2.0 bp"):
            v = metrics.get(k, {}).get("by_cost", {}).get(c)
            out[f"{k}@{c}"] = float(v) if v is not None and np.isfinite(v) else None
    return out


def run_full_backtest(factor_df, labels, target_region, name) -> dict:
    if factor_df is None or factor_df.is_empty():
        return {}
    label_col = [c for c in labels.columns if c not in ("DATETIME", "SYMBOL")][0]
    df = factor_df.join(labels, on=["DATETIME", "SYMBOL"], how="inner")
    if df.is_empty():
        return {}
    out_dir = PROJECT_ROOT / "backtest_results_monthly" / target_region / name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cross_section_backtest(
        df=df, factor_column_name="factor_value",
        datetime_column_name="DATETIME", symbol_column_name="SYMBOL",
        raw_label_column_name=label_col, freq="1mo", layers_use=LAYERS,
        fees=FEES, backtest_mode="long/short_layers",
        backtest_params={"long_layer_index": [1], "short_layer_index": [LAYERS]},
        lag=LAG, annual_days=ANNUAL_DAYS,
        factor_info_dict={"factor_name": name}, output_dir=out_dir,
    )
    return extract_metrics(out_dir / "summary_df.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    args = ap.parse_args()

    print("=" * 110)
    print(f"MULTI-TARGET PER-CATEGORY ABLATION × MILD NEUTRALIZED — lookback={LOOKBACK}mo")
    print(f"  targets = {args.targets}")
    print("=" * 110)

    results = []

    for t_region in args.targets:
        print(f"\n{'#' * 90}\n# TARGET: {t_region}\n{'#' * 90}")

        try:
            t_secs = target_sectors_df(t_region)
            t_months = target_months(t_region)
            t_labels = target_labels(t_region)
            t_lncap = build_true_lncap(t_region, t_months)
        except Exception as e:
            print(f"  ERROR loading target {t_region}: {e}")
            continue
        if t_lncap is None:
            print(f"  WARNING: no shares for {t_region}, skip")
            continue

        for peer_region in ["us_russell_1000", t_region]:
            is_self = peer_region == t_region
            corridor = "SAME" if is_self else "CROSS_US"
            print(f"\n  -- {corridor} ({peer_region}{' [SELF-MASKED]' if is_self else ''})")
            print(f"  {'Category':<42}  {'mild_ICIR':>10}  {'p':>9}  {'LO_Shp@2bp':>12}")
            print("  " + "-" * 80)

            try:
                peer_ret = peer_returns_for_region(peer_region, LOOKBACK, t_months)
            except Exception as e:
                print(f"    peer return load failed: {e}")
                continue

            for cat in CATEGORIES:
                t_syms, t_emb = load_category_embeddings(t_region, cat)
                p_syms, p_emb = load_category_embeddings(peer_region, cat)
                if t_syms is None or p_syms is None:
                    continue
                sim = cosine_similarity_matrix(t_emb, p_emb)
                pct = percentile_rank_fast(sim, axis=1)
                w = sigmoid_weights(pct, k=K, c=C)
                if is_self:
                    w *= (1.0 - self_mask_matrix(t_syms, p_syms))
                fdf = build_factor(w, t_syms, p_syms, peer_ret, t_months)
                mild = mild_neutralize(fdf, t_lncap, t_secs)
                tag = f"cat_multi_{t_region}_{corridor.lower()}_{cat}_12mo"
                m = run_full_backtest(mild, t_labels, t_region, tag)
                results.append({"target": t_region, "corridor": corridor,
                                "peer_region": peer_region, "category": cat,
                                "lookback": LOOKBACK, "mild": m})
                icir = (m or {}).get("rank_icir_annual")
                p = (m or {}).get("rank_ic_p_value")
                lo = (m or {}).get("long_ret_sharpe@2.0 bp")
                def f(v, fmt=".3f"):
                    return f"{v:>{8}.3f}" if isinstance(v, (int, float)) and np.isfinite(v) else f"{'N/A':>10}"
                print(f"  {cat:<42}  {f(icir):>10}  {f(p):>9}  {f(lo):>12}")

            # all_10_equal
            try:
                cached = load_similarity_cache(t_region, peer_region, "w128", cache_dir=CACHE_SIM_DIR)
                w_all = sigmoid_weights(cached.pct_ranks, k=K, c=C)
                if is_self:
                    w_all *= (1.0 - self_mask_matrix(cached.target_symbols, cached.peer_symbols))
                fdf_all = build_factor(w_all, cached.target_symbols, cached.peer_symbols, peer_ret, t_months)
                mild_all = mild_neutralize(fdf_all, t_lncap, t_secs)
                tag = f"cat_multi_{t_region}_{corridor.lower()}_all10_equal_12mo"
                m_all = run_full_backtest(mild_all, t_labels, t_region, tag)
                results.append({"target": t_region, "corridor": corridor,
                                "peer_region": peer_region, "category": "all_10_equal",
                                "lookback": LOOKBACK, "mild": m_all})
                icir = (m_all or {}).get("rank_icir_annual")
                p = (m_all or {}).get("rank_ic_p_value")
                lo = (m_all or {}).get("long_ret_sharpe@2.0 bp")
                def f(v, fmt=".3f"):
                    return f"{v:>{8}.3f}" if isinstance(v, (int, float)) and np.isfinite(v) else f"{'N/A':>10}"
                print(f"  {'all_10_equal':<42}  {f(icir):>10}  {f(p):>9}  {f(lo):>12}")
            except Exception as e:
                print(f"    all_10_equal failed: {e}")

    # Summary matrix: one row per (target, category) with cross + same ICIRs
    print(f"\n{'=' * 110}")
    print("SUMMARY: cross-market (US->T) vs same-market (T->T) mild ICIR per category")
    print(f"{'=' * 110}")
    by_key = {(r["target"], r["corridor"], r["category"]): r for r in results}
    for t in args.targets:
        print(f"\n  TARGET: {t}")
        print(f"  {'Category':<44}{'cross (US->T)':>16}{'same (T->T)':>16}")
        print("  " + "-" * 76)
        for cat in CATEGORIES + ["all_10_equal"]:
            cross_r = by_key.get((t, "CROSS_US", cat))
            same_r  = by_key.get((t, "SAME", cat))
            cross_icir = (cross_r["mild"] if cross_r else {}).get("rank_icir_annual") if cross_r else None
            same_icir  = (same_r["mild"] if same_r else {}).get("rank_icir_annual") if same_r else None
            def f(v):
                return f"{v:>14.3f}" if isinstance(v, (int, float)) and np.isfinite(v) else f"{'N/A':>14}"
            mark = ""
            if (cross_icir is not None) and (same_icir is not None):
                mark = "  ✓" if cross_icir > same_icir else "  ✗"
            print(f"  {cat:<44}{f(cross_icir):>16}{f(same_icir):>16}{mark}")

    out = PROJECT_ROOT / "backtest_results_daily" / "category_ablation_multi_target_12mo.json"
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
