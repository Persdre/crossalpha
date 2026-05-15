#!/usr/bin/env python3
"""5x5 cross-vs-same market matrix under MILD neutralization.

For each (target, source) in {US, JP, TW, KR, HK} x {same}, builds the
joint all_10_equal factor (sigmoid top-1% peer momentum, sector-relative
peer returns, self-mask on diagonal cells), applies the paper-stated
Mild neutralization (sector demean + lnCap residual via mild_neutralize),
runs the standard cross-section backtest, and snapshots the full metric set.

Lookback fixed at 12mo, monthly frequency, LAG=1, fees up to 20bp.

Output: backtest_results_daily/cross_vs_same_market_matrix_12mo.json

Each row:
  {target, source, mild: {<all summary metrics>}, raw: {<all summary metrics>},
   diagonal: bool}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.kernels import load_monthly_close_prices, load_symbol_sectors, sigmoid_weights
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
FEES = [0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.0010, 0.0015, 0.0020]
LAYERS = 5
LAG = 1
ANNUAL_DAYS = 12
LOOKBACK = 12
MARKETS = ["us_russell_1000", "jp_topix_500", "tw_twse", "kr_kospi", "hk_main"]

COST_LEVELS = ["0.0 bp", "2.0 bp"]
SCALAR_METRICS = [
    "ic", "ic_p_value", "ic_winratio",
    "icir", "icir_annual",
    "rank_ic", "rank_ic_p_value", "rank_ic_winratio",
    "rank_icir", "rank_icir_annual",
    "acf_1", "acf_halflife",
]
COSTED_METRICS = [
    "long_short_ret_annual", "long_short_ret_sharpe",
    "long_short_ret_max_dd", "long_short_ret_calmar",
    "long_short_ret_sortino", "long_short_ret_sharpe_per_turnover",
    "long_short_turnover_ratio", "long_short_win_rate",
    "long_ret_annual", "long_ret_sharpe", "long_ret_max_dd",
    "long_ret_calmar", "long_ret_sortino",
    "long_turnover_ratio", "long_win_rate", "long_excess_ret_annual",
    "short_ret_annual", "short_ret_sharpe", "short_ret_max_dd",
    "short_turnover_ratio", "short_win_rate", "short_excess_ret_annual",
    "passive_ret_annual", "passive_ret_sharpe", "passive_ret_max_dd",
]


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
    return (
        pl.read_csv(SYMBOL_DICT_PATH)
        .filter(pl.col("region") == region)
        .select(["SYMBOL", "gics_sector"])
        .filter(pl.col("gics_sector").is_not_null() & (pl.col("gics_sector") != ""))
    )


def target_months(region: str):
    pm = load_monthly_close_prices(region, "2012-01-01", END)
    return sorted(
        pm.filter((pl.col("DATETIME") >= START) & (pl.col("DATETIME") <= END))
        .select("DATETIME").unique()["DATETIME"].to_list()
    )


def target_labels(region: str):
    # Prefer tradable_close_return; fall back to close_return for markets that
    # only ship the latter (e.g., us_russell_1000).
    for src, key in [("tradable_close_return", "tradable_close_return"),
                     ("close_return", "close_return")]:
        src_dir = PROJECT_ROOT / "labels" / "Equity" / region / "1mo" / src
        if src_dir.exists():
            return h5_load(
                {"source_path": str(PROJECT_ROOT / "labels"), "product": "Equity",
                 "region": region, "freq": "1mo", "source": src},
                start_date=START, end_date=END, keys=[key],
            )
    raise FileNotFoundError(f"No 1mo return labels found for {region}")


def extract_full_metrics(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    payload = json.load(open(summary_path))
    metrics = payload.get("metrics", {})
    out: dict = {}
    for k in SCALAR_METRICS:
        v = metrics.get(k, {}).get("by_cost", {}).get("0.0 bp")
        out[k] = float(v) if v is not None and np.isfinite(v) else None
    for k in COSTED_METRICS:
        for c in COST_LEVELS:
            v = metrics.get(k, {}).get("by_cost", {}).get(c)
            out[f"{k}@{c}"] = float(v) if v is not None and np.isfinite(v) else None
    out["_n_obs"] = payload.get("n_obs")
    return out


def run_full_backtest(factor_df: pl.DataFrame, labels: pl.DataFrame, target_region: str, name: str) -> dict:
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
    return extract_full_metrics(out_dir / "summary_df.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=LOOKBACK)
    ap.add_argument("--targets", nargs="+", default=MARKETS)
    ap.add_argument("--sources", nargs="+", default=MARKETS)
    args = ap.parse_args()
    lookback = args.lookback

    print("=" * 110)
    print(f"CROSS-VS-SAME MARKET MATRIX — all_10_equal × MILD (sector + lnCap) — lookback={lookback}mo")
    print(f"  targets = {args.targets}")
    print(f"  sources = {args.sources}")
    print("=" * 110)

    results = []

    for t_region in args.targets:
        print(f"\n{'#' * 90}")
        print(f"# TARGET: {t_region}")
        print(f"{'#' * 90}")

        try:
            t_sectors = target_sectors_df(t_region)
            t_months = target_months(t_region)
            t_labels = target_labels(t_region)
            t_lncap = build_true_lncap(t_region, t_months)
        except Exception as e:
            print(f"  ERROR loading target {t_region}: {e}")
            continue
        if t_lncap is None:
            print(f"  WARNING: no shares_outstanding for {t_region}; skipping target")
            continue

        print(f"  months={len(t_months)}, sectors={t_sectors.height}, "
              f"labels={t_labels.height}, lncap={t_lncap.height}")

        for s_region in args.sources:
            is_self = s_region == t_region
            print(f"\n  -- source: {s_region}{'  [SELF-MASKED]' if is_self else ''}")

            try:
                cached = load_similarity_cache(t_region, s_region, "w128",
                                               cache_dir=CACHE_SIM_DIR)
            except Exception as e:
                print(f"    similarity cache load FAILED ({e}); skipping cell")
                continue

            try:
                peer_ret = peer_returns_for_region(s_region, lookback, t_months)
            except Exception as e:
                print(f"    peer-return load FAILED ({e}); skipping cell")
                continue

            w = sigmoid_weights(cached.pct_ranks, k=K, c=C)
            if is_self:
                w *= (1.0 - self_mask_matrix(cached.target_symbols, cached.peer_symbols))

            raw_fdf = build_factor(w, cached.target_symbols, cached.peer_symbols,
                                   peer_ret, t_months)
            mild_fdf = mild_neutralize(raw_fdf, t_lncap, t_sectors)

            tag = f"{t_region}_from_{s_region}_{lookback}mo"
            raw_metrics = run_full_backtest(raw_fdf, t_labels, t_region,
                                            f"matrix_{tag}_raw")
            mild_metrics = run_full_backtest(mild_fdf, t_labels, t_region,
                                             f"matrix_{tag}_mild")

            row = {"target": t_region, "source": s_region, "lookback": lookback,
                   "diagonal": is_self,
                   "raw": raw_metrics, "mild": mild_metrics}
            results.append(row)

            mild_icir = (mild_metrics or {}).get("rank_icir_annual")
            mild_lss = (mild_metrics or {}).get("long_short_ret_sharpe@2.0 bp")
            mild_lo = (mild_metrics or {}).get("long_ret_sharpe@2.0 bp")
            mild_lsret = (mild_metrics or {}).get("long_short_ret_annual@2.0 bp")
            def _f(v):
                return f"{v:>8.3f}" if isinstance(v, (int, float)) and np.isfinite(v) else f"{'  N/A':>8}"
            print(f"    mild ICIR={_f(mild_icir)}  LS Shp={_f(mild_lss)}  "
                  f"LO Shp={_f(mild_lo)}  LS Ret={_f(mild_lsret)}")

    # Summary matrix print
    print(f"\n{'=' * 110}")
    print(f"SUMMARY: 5×5 mild ICIR matrix (rows=target, cols=source)")
    print(f"{'=' * 110}")
    print(f"  {'target \\\\ source':<22} | " + " | ".join(f"{s:>14}" for s in args.sources))
    print("  " + "-" * (22 + 17 * len(args.sources) + 4))
    by_pair = {(r["target"], r["source"]): r for r in results}
    for t in args.targets:
        cells = []
        for s in args.sources:
            r = by_pair.get((t, s))
            if r is None:
                cells.append("       N/A      ")
            else:
                v = (r["mild"] or {}).get("rank_icir_annual")
                marker = "  ⓢ" if r["diagonal"] else ""
                if isinstance(v, (int, float)) and np.isfinite(v):
                    cells.append(f"{v:>11.3f}{marker:<3}")
                else:
                    cells.append(f"{'   N/A':>11}{marker:<3}")
        print(f"  {t:<22} | " + " | ".join(cells))
    print("  (ⓢ = self-masked diagonal — same-market baseline)")

    out = PROJECT_ROOT / "backtest_results_daily" / f"cross_vs_same_market_matrix_{lookback}mo.json"
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
