#!/usr/bin/env python3
"""SIMILARITY-LEVEL pooling under MILD on each target market.

Correct pooling: concat per-source similarity matrices along the peer axis,
percentile-rank globally, sigmoid (top ~1% of pooled peers), apply self-mask
where (target_symbol == peer_symbol & peer_source == target market).

Pool definitions per target T (similar to factor-level pool tests, but now
done at the similarity layer):
  - same_market_only:   T as the only source, self-masked diagonal baseline
  - best_single:        the strongest single source for T (read from matrix)
  - best_pair:          T's two strongest sources (single ICIR)
  - all_cross:          all sources except T
  - all_plus_dom:       all 5 sources including T-as-source (self-masked)

Output: backtest_results_daily/pooled_similarity_mild_{lookback}mo.json
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
from scripts.backtesting.backtest_utils import run_cross_section_backtest
from scripts.run_graph_monthly import build_true_lncap
from scripts.run_rq3_mild_rerun import mild_neutralize

DATA_ROOT = Path("${CROSSALPHA_DATA_ROOT}")
SYMBOL_DICT_PATH = PROJECT_ROOT / "docs" / "SymbolDict.csv"
START, END = "2015-01-01", "2025-12-31"
C, K = 0.99, 50.0
LOOKBACK = 12
FEES = [0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.0010, 0.0015, 0.0020]
LAYERS, LAG, ANNUAL_DAYS = 5, 1, 12
EMBEDDING_DIR = "128_dimensions"
MARKETS = ["us_russell_1000", "jp_topix_500", "tw_twse", "kr_kospi", "hk_main"]

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
# all_10_equal weights
CAT_WEIGHTS = {c: 1.0 / len(CATEGORIES) for c in CATEGORIES}

SCALAR_METRICS = ["rank_ic", "rank_ic_p_value", "rank_icir_annual", "rank_ic_winratio"]
COSTED_METRICS = [
    "long_short_ret_annual", "long_short_ret_sharpe", "long_short_ret_max_dd",
    "long_short_ret_calmar", "long_short_turnover_ratio", "long_short_win_rate",
    "long_ret_annual", "long_ret_sharpe", "long_ret_max_dd",
    "long_ret_calmar", "long_turnover_ratio", "long_win_rate",
]


def load_category_embeddings(region, category):
    path = PROJECT_ROOT / "embeddings" / EMBEDDING_DIR / category / f"{region}_{category}.h5"
    if not path.exists():
        return None, None
    with h5py.File(path, "r") as f:
        symbols = json.loads(f["embeddings"].attrs["symbols"])
        embeddings = f["embeddings"][:].astype(np.float64)
    return symbols, embeddings


def build_weighted_similarity(target_region: str, source_region: str):
    """Compute weighted-equal sum-of-cosines across 10 categories.
    Returns (sim, target_symbols, source_symbols).
    Only stocks present in ALL 10 category h5 files for that region are kept."""
    t_syms = None
    s_syms = None
    t_cats = {}
    s_cats = {}
    for cat in CATEGORIES:
        ts, te = load_category_embeddings(target_region, cat)
        ss, se = load_category_embeddings(source_region, cat)
        if ts is None or ss is None:
            continue
        t_cats[cat] = (ts, te)
        s_cats[cat] = (ss, se)
    if not t_cats or not s_cats:
        raise RuntimeError(f"missing category embeddings for {target_region}/{source_region}")

    # Intersect symbol lists across categories (should be identical but be safe)
    t_common = sorted(set.intersection(*[set(v[0]) for v in t_cats.values()]))
    s_common = sorted(set.intersection(*[set(v[0]) for v in s_cats.values()]))
    n_t, n_s = len(t_common), len(s_common)
    sim = np.zeros((n_t, n_s), dtype=np.float64)
    for cat, w in CAT_WEIGHTS.items():
        if cat not in t_cats or cat not in s_cats:
            continue
        ts, te = t_cats[cat]
        ss, se = s_cats[cat]
        t_idx = [ts.index(s) for s in t_common]
        s_idx = [ss.index(s) for s in s_common]
        sim_cat = cosine_similarity_matrix(te[t_idx], se[s_idx])
        sim += w * sim_cat
    return sim, t_common, s_common


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
    for k in SCALAR_METRICS:
        v = metrics.get(k, {}).get("by_cost", {}).get("0.0 bp")
        out[k] = float(v) if v is not None and np.isfinite(v) else None
    for k in COSTED_METRICS:
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


def build_pooled_factor(target_region: str, target_symbols: list[str],
                        sims: list[tuple[str, np.ndarray, list[str]]],
                        peer_returns_per_source: dict[str, dict[str, tuple[list[str], np.ndarray]]],
                        months: list[str]) -> pl.DataFrame:
    """Similarity-level pooling.
    sims: list of (source_region, sim_matrix [n_t, n_s], source_symbols).
    Returns a (DATETIME, SYMBOL, factor_value) DataFrame.
    """
    # Concat similarity matrices along the peer axis
    sim_concat = np.concatenate([s for (_, s, _) in sims], axis=1)
    # Global per-row percentile rank
    pct = percentile_rank_fast(sim_concat, axis=1)
    w_pool = sigmoid_weights(pct, k=K, c=C)

    # Self-mask: zero weight where peer == target (within the matching source = target_region)
    cursor = 0
    for s_region, s_mat, s_syms in sims:
        n_s = s_mat.shape[1]
        if s_region == target_region:
            # within this block, mask diagonals
            t2i_in_block = {sym: idx for idx, sym in enumerate(s_syms)}
            for ti, t_sym in enumerate(target_symbols):
                j_in_block = t2i_in_block.get(t_sym)
                if j_in_block is not None:
                    w_pool[ti, cursor + j_in_block] = 0.0
        cursor += n_s

    # Concat peer symbols (global ordering matches sim_concat columns)
    peer_blocks = []  # list of (source_region, source_symbols)
    for s_region, s_mat, s_syms in sims:
        peer_blocks.append((s_region, s_syms))

    # Build per-month peer return vector aligned to global peer ordering
    recs = []
    for m in months:
        # Construct a [n_peer_total] returns array, NaN where missing
        ret_vec = np.full(sim_concat.shape[1], np.nan)
        cursor = 0
        for s_region, s_syms in peer_blocks:
            n_s = len(s_syms)
            block_returns = np.full(n_s, np.nan)
            if (m in peer_returns_per_source.get(s_region, {})):
                ps, pv = peer_returns_per_source[s_region][m]
                pos = {sym: idx for idx, sym in enumerate(s_syms)}
                for sym, r in zip(ps, pv):
                    j = pos.get(sym)
                    if j is not None:
                        block_returns[j] = r
            ret_vec[cursor:cursor + n_s] = block_returns
            cursor += n_s
        valid = ~np.isnan(ret_vec)
        ret_filled = np.where(valid, ret_vec, 0.0)
        ws = w_pool @ ret_filled
        wd = w_pool @ valid.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            sc = np.where(wd > 1, ws / wd, np.nan)
        for sym, val in zip(target_symbols, sc):
            if np.isfinite(val):
                recs.append({"DATETIME": m, "SYMBOL": sym, "factor_value": float(val)})
    return pl.DataFrame(recs) if recs else pl.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=MARKETS)
    args = ap.parse_args()

    # Per-target single-source ICIR — used to define best_pair
    matrix_path = PROJECT_ROOT / "backtest_results_daily" / "cross_vs_same_market_matrix_12mo.json"
    matrix = json.load(open(matrix_path))
    single_icir = {(r["target"], r["source"]): (r["mild"] or {}).get("rank_icir_annual")
                   for r in matrix}

    print("=" * 110)
    print(f"SIMILARITY-LEVEL POOLED MILD MATRIX — concat sims, global rank, sigmoid top-1%")
    print("=" * 110)

    all_results = []

    # Build per-source peer return dicts once per source (reused across targets)
    peer_ret_cache: dict[str, dict] = {}
    for s in MARKETS:
        if s not in peer_ret_cache:
            print(f"  loading peer returns for {s}...")
            peer_ret_cache[s] = peer_returns_for_region(s, LOOKBACK, target_months(s))

    # Pre-build similarity matrices for each (target, source) pair
    for t in args.targets:
        print(f"\n{'#' * 90}\n# TARGET: {t}\n{'#' * 90}")
        try:
            t_secs = target_sectors_df(t)
            t_months = target_months(t)
            t_labels = target_labels(t)
            t_lncap = build_true_lncap(t, t_months)
        except Exception as e:
            print(f"  ERROR target {t}: {e}")
            continue
        if t_lncap is None:
            print(f"  WARNING: no shares for {t}")
            continue

        # Build per-source similarity matrix using the CURRENT target's symbol list
        per_source_sim = {}
        for s in MARKETS:
            try:
                sim, t_syms, s_syms = build_weighted_similarity(t, s)
                per_source_sim[s] = (sim, t_syms, s_syms)
            except Exception as e:
                print(f"  build sim {t}<-{s} FAILED: {e}")
                continue

        # All targets share the same target_symbols (intersection across cats),
        # but the sim matrices may have slightly different t_syms ordering.
        # Use the t_syms from one source as the canonical target order.
        # We reuse the first source's t_syms.
        any_source = next(iter(per_source_sim.keys()))
        canon_tsyms = per_source_sim[any_source][1]

        # Use per-source peer returns aligned to peer symbol order
        ret_by_src = {s: peer_ret_cache[s] for s in MARKETS}

        # Per-target peer-source-mapping
        sources_x = [s for s in MARKETS if s != t]
        ranked = sorted([(s, single_icir.get((t, s)) or float("-inf"))
                         for s in sources_x], key=lambda x: -x[1])
        best_pair_sources = [ranked[0][0], ranked[1][0]]

        pool_definitions = {
            "same_market_only": [t],
            "best_single":      [ranked[0][0]],
            "best_pair":        best_pair_sources,
            "all_cross":        sources_x,
            "all_plus_dom":     MARKETS,
        }

        for pool_name, srcs in pool_definitions.items():
            present = [s for s in srcs if s in per_source_sim]
            if not present:
                continue
            sims_list = [(s, per_source_sim[s][0], per_source_sim[s][2]) for s in present]
            fdf = build_pooled_factor(t, canon_tsyms, sims_list, ret_by_src, t_months)
            mild = mild_neutralize(fdf, t_lncap, t_secs)
            tag = f"poolsim_{pool_name}_{t}_12mo"
            metrics = run_full_backtest(mild, t_labels, t, tag)
            row = {"target": t, "pool": pool_name, "sources": present,
                   "lookback": LOOKBACK, "mild": metrics}
            all_results.append(row)
            icir = (metrics or {}).get("rank_icir_annual")
            pv = (metrics or {}).get("rank_ic_p_value")
            ls = (metrics or {}).get("long_short_ret_sharpe@2.0 bp")
            lo = (metrics or {}).get("long_ret_sharpe@2.0 bp")
            def f(v, fmt=".3f"):
                return f"{v:>9{fmt}}" if isinstance(v, (int, float)) and np.isfinite(v) else f"{'N/A':>9}"
            srcs_str = "+".join(s.split("_")[0].upper() for s in present)
            print(f"  pool {pool_name:<18} ({srcs_str:<25})  "
                  f"ICIR={f(icir)}  p={f(pv)}  LS={f(ls)}  LO={f(lo)}")

    # Summary
    print(f"\n{'=' * 110}\nSIMILARITY-LEVEL POOL ICIR — per target × pool combo\n{'=' * 110}")
    pools = ["same_market_only", "best_single", "best_pair", "all_cross", "all_plus_dom"]
    print(f"  {'target':<22}" + " | ".join(f"{p:>14}" for p in pools))
    print("  " + "-" * 100)
    by_pair = {(r["target"], r["pool"]): r for r in all_results}
    for t in args.targets:
        cells = []
        for p in pools:
            r = by_pair.get((t, p))
            v = (r["mild"] or {}).get("rank_icir_annual") if r else None
            if isinstance(v, (int, float)) and np.isfinite(v):
                cells.append(f"{v:>14.3f}")
            else:
                cells.append(f"{'   N/A':>14}")
        print(f"  {t:<22}" + " | ".join(cells))

    out = PROJECT_ROOT / "backtest_results_daily" / "pooled_similarity_mild_12mo.json"
    json.dump(all_results, open(out, "w"), indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
