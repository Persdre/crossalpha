#!/usr/bin/env python3
"""Daily pooled multi-market peer momentum pipeline for Japan.

Pools peer momentum signals from multiple markets (US, TW, HK, KR, VN)
at daily frequency to predict Japan (jp_topix_500). Tests IC/IR > 0.5
gate for production inclusion.

Usage:
    python scripts/daily_pooled_pipeline.py
    python scripts/daily_pooled_pipeline.py --lookback 5 20
    python scripts/daily_pooled_pipeline.py --pooled-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.kernels import load_daily_close_prices, sigmoid_weights
from scripts.similarity_cache import (
    encode_ck_params,
    load_similarity_cache,
)
from scripts.h5_data.h5_utils import H5Params, h5_save
from scripts.backtesting.backtest_utils import run_cross_section_backtest
from scripts.kernels import percentile_rank_fast
from scripts.kernels.return_correlation import (
    compute_correlation_from_prices,
    blend_pct_ranks_aligned,
    save_correlation_cache,
    load_correlation_cache,
    corr_cache_exists,
)
from scripts.similarity_cache import CachedSimilarity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_REGION = "jp_topix_500"

PEER_MARKETS = {
    "us": "us_russell_1000",
    "tw": "tw_twse",
    "hk": "hk_hsci",
    "kr": "kr_kospi",
}
# `vn_hose` is intentionally omitted because Vietnam market support is paused.

LOOKBACK_PERIODS = [1, 5, 20, 60]  # trading days

DEFAULT_DATA_ROOT = "${CROSSALPHA_DATA_ROOT}"

DEFAULT_C = 0.99
DEFAULT_K = 50.0
DEFAULT_EMBEDDING = "w128"

# Backtest configuration
FEES = [0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.0010, 0.0015, 0.0020]
LAYERS = 5
LAG = 2  # t signal → t+2 execution (use t+1 close), per PM feedback 2025-03-25
ANNUAL_DAYS = 252
FREQ = "1d"
BACKTEST_MODE = "long/short_layers"
BACKTEST_PARAMS = {"long_layer_index": [1], "short_layer_index": [5]}

ICIR_GATE = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily pooled multi-market peer momentum for Japan",
    )
    parser.add_argument(
        "--lookback", type=int, nargs="+", default=None,
        help=f"Lookback period(s) in trading days (default: {LOOKBACK_PERIODS})",
    )
    parser.add_argument(
        "--data-root", type=str, default=DEFAULT_DATA_ROOT,
        help=f"Root path for daily market data (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--embedding-type", type=str, default=DEFAULT_EMBEDDING,
        help=f"Embedding type (default: {DEFAULT_EMBEDDING})",
    )
    parser.add_argument(
        "--c", type=float, default=DEFAULT_C,
        help=f"Sigmoid center (default: {DEFAULT_C})",
    )
    parser.add_argument(
        "--k", type=float, default=DEFAULT_K,
        help=f"Sigmoid steepness (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--similarity", type=str, default="embedding",
        choices=["embedding", "return_corr", "blend"],
        help="Similarity source: embedding (default), return_corr, or blend",
    )
    parser.add_argument(
        "--alpha", type=float, nargs="+", default=[0.5],
        help="Blend weight(s) for embedding ranks (0=pure corr, 1=pure emb). Only used with --similarity blend",
    )
    parser.add_argument(
        "--corr-lookback", type=int, default=252,
        help="Trailing trading days for return correlation (default: 252)",
    )
    parser.add_argument(
        "--peers", type=str, nargs="+", default=None,
        choices=list(PEER_MARKETS.keys()),
        help=f"Subset of peer markets to use (default: all {list(PEER_MARKETS.keys())})",
    )
    parser.add_argument(
        "--rebalance-every", type=int, default=1,
        help="Rebalance every N trading days (default: 1 = daily)",
    )
    parser.add_argument(
        "--neutralize", action="store_true", default=True,
        help="Cross-sectional demean the factor (market neutralization, on by default)",
    )
    parser.add_argument(
        "--no-neutralize", dest="neutralize", action="store_false",
        help="Disable cross-sectional demean (not recommended)",
    )
    parser.add_argument(
        "--composite", action="store_true",
        help="Build multi-lookback composite factor (z-score + average across lookbacks)",
    )
    parser.add_argument(
        "--pooled-only", action="store_true",
        help="Skip individual peer backtests, only run pooled",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip factors/backtests that already exist",
    )
    parser.add_argument(
        "--start", type=str, default="2015-01-01",
        help="Start date (default: 2015-01-01)",
    )
    parser.add_argument(
        "--end", type=str, default="2025-12-31",
        help="End date (default: 2025-12-31)",
    )
    parser.add_argument(
        "--factors-dir", type=str, default=None,
        help="Factors output dir (default: factors_daily/)",
    )
    parser.add_argument(
        "--backtest-dir", type=str, default=None,
        help="Backtest output dir (default: backtest_results_daily/)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stage 1: Load similarity weights
# ---------------------------------------------------------------------------

class PrecomputedWeights(NamedTuple):
    weights: np.ndarray          # (n_target, n_peer) sigmoid weights
    target_symbols: list[str]
    peer_symbols: list[str]
    peer_symbol_to_idx: dict[str, int]


class MonthlyCorrelation(NamedTuple):
    """Container for monthly correlation results with both weights and raw ranks."""
    weights: dict[str, PrecomputedWeights]
    pct_ranks: dict[str, tuple[np.ndarray, list[str], list[str]]]


def load_and_apply_sigmoid(
    peer_region: str,
    embedding_type: str,
    c: float,
    k: float,
    cache_dir: Path | None = None,
) -> PrecomputedWeights:
    """Load cached percentile ranks and apply sigmoid weighting."""
    cached = load_similarity_cache(
        TARGET_REGION, peer_region, embedding_type, cache_dir=cache_dir
    )
    weights = sigmoid_weights(cached.pct_ranks, k=k, c=c)
    peer_symbol_to_idx = {s: i for i, s in enumerate(cached.peer_symbols)}
    return PrecomputedWeights(
        weights=weights,
        target_symbols=cached.target_symbols,
        peer_symbols=cached.peer_symbols,
        peer_symbol_to_idx=peer_symbol_to_idx,
    )


def compute_monthly_corr_weights(
    target_prices: pl.DataFrame,
    peer_prices: pl.DataFrame,
    target_region: str,
    peer_region: str,
    corr_lookback: int,
    k: float,
    c: float,
    cache_dir: Path | None = None,
) -> MonthlyCorrelation:
    """Compute correlation-based sigmoid weights for each month.

    For each calendar month, computes pairwise return correlation using
    the trailing `corr_lookback` trading days, converts to percentile
    ranks, then applies sigmoid weighting. Both the raw pct_ranks and
    the sigmoid weights are returned — the ranks are needed for blending.

    The cache key uses the last trading day of each month as rebalance_month
    (e.g. "2024-06-28"); _build_corr_cache_path truncates to "2024-06".
    """
    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "cache" / "return_corr"

    months = (
        target_prices.select("DATETIME")
        .unique()
        .sort("DATETIME")
        .with_columns(pl.col("DATETIME").str.slice(0, 7).alias("month"))
        .group_by("month")
        .agg(pl.col("DATETIME").max().alias("month_end"))
        .sort("month")
    )

    monthly_weights: dict[str, PrecomputedWeights] = {}
    monthly_ranks: dict[str, tuple[np.ndarray, list[str], list[str]]] = {}

    sorted_months = months.sort("month")["month"].to_list()

    for i, row in enumerate(months.sort("month").iter_rows(named=True)):
        month_key = row["month"]
        month_end = row["month_end"]

        # Use PREVIOUS month-end to avoid look-ahead bias:
        # Correlation computed from data up to prev month-end,
        # then applied to current month's factor computation.
        if i == 0:
            # First month: use its own month_end (no previous available)
            # This month will have slight look-ahead — acceptable for burn-in
            corr_cutoff = month_end
        else:
            corr_cutoff = months.sort("month").row(i - 1, named=True)["month_end"]

        rebalance_str = str(corr_cutoff)

        if corr_cache_exists(target_region, peer_region, rebalance_str, cache_dir):
            cached = load_correlation_cache(
                target_region, peer_region, rebalance_str, cache_dir
            )
            corr_matrix = cached.corr_matrix
            target_symbols = cached.target_symbols
            peer_symbols = cached.peer_symbols
        else:
            target_sub = target_prices.filter(pl.col("DATETIME") <= corr_cutoff)
            peer_sub = peer_prices.filter(pl.col("DATETIME") <= corr_cutoff)

            corr_matrix, target_symbols, peer_symbols = compute_correlation_from_prices(
                target_sub, peer_sub,
                corr_lookback=corr_lookback,
                min_overlap=max(60, corr_lookback // 4),
            )

            save_correlation_cache(
                corr_matrix, target_symbols, peer_symbols,
                target_region, peer_region, rebalance_str, cache_dir,
            )

        # Pass NaN-containing corr_matrix directly to percentile_rank_fast.
        # The ranking function now handles NaN correctly (excludes from ranking,
        # uses valid_count as denominator). NaN ranks become NaN weights (sigmoid=0).
        pct_ranks = percentile_rank_fast(corr_matrix, axis=1)
        weights = sigmoid_weights(pct_ranks, k=k, c=c)
        # NaN pct_ranks → NaN sigmoid → replace with 0 weight
        weights = np.nan_to_num(weights, nan=0.0)

        peer_symbol_to_idx = {s: i for i, s in enumerate(peer_symbols)}
        monthly_weights[month_key] = PrecomputedWeights(
            weights=weights,
            target_symbols=target_symbols,
            peer_symbols=peer_symbols,
            peer_symbol_to_idx=peer_symbol_to_idx,
        )
        monthly_ranks[month_key] = (pct_ranks, target_symbols, peer_symbols)

    return MonthlyCorrelation(weights=monthly_weights, pct_ranks=monthly_ranks)


# ---------------------------------------------------------------------------
# Stage 1: Daily factor construction
# ---------------------------------------------------------------------------

def compute_daily_returns(
    prices: pl.DataFrame,
    lookback: int,
) -> pl.DataFrame:
    """Compute rolling returns from daily close prices.

    Args:
        prices: DataFrame with [DATETIME, SYMBOL, close]
        lookback: Number of trading days to look back

    Returns:
        DataFrame with [DATETIME, SYMBOL, daily_return]
    """
    return (
        prices.sort(["SYMBOL", "DATETIME"])
        .with_columns(
            ((pl.col("close") / pl.col("close").shift(lookback)) - 1)
            .over("SYMBOL")
            .alias("daily_return")
        )
        .drop_nulls("daily_return")
        .select(["DATETIME", "SYMBOL", "daily_return"])
    )


def compute_daily_factor_single_peer(
    precomputed: PrecomputedWeights,
    peer_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Compute daily peer momentum factor for one peer market.

    For each target stock on each day, computes:
        score = sum(sigmoid_weight * peer_return) / sum(sigmoid_weight)

    Uses matrix multiplication for vectorized computation across all days.

    Args:
        precomputed: Sigmoid weights and symbol mappings
        peer_returns: DataFrame with [DATETIME, SYMBOL, daily_return]

    Returns:
        DataFrame with [DATETIME, SYMBOL, factor_value]
    """
    n_peer = len(precomputed.peer_symbols)

    # Pivot to wide format: (n_days, n_symbols_in_data) — fast via Polars
    wide = peer_returns.pivot(
        index="DATETIME", on="SYMBOL", values="daily_return"
    ).sort("DATETIME")

    wide_dates = wide["DATETIME"].to_list()
    wide_syms = [c for c in wide.columns if c != "DATETIME"]

    # Align to precomputed peer symbol order: (n_days, n_peer)
    returns_matrix = np.full((len(wide_dates), n_peer), np.nan, dtype=np.float64)
    for col_idx, sym in enumerate(wide_syms):
        p_idx = precomputed.peer_symbol_to_idx.get(sym)
        if p_idx is not None:
            returns_matrix[:, p_idx] = wide[sym].to_numpy()

    # Compute scores via matrix multiply: weights @ returns
    valid_mask = ~np.isnan(returns_matrix)  # (n_days, n_peer)
    returns_filled = np.where(valid_mask, returns_matrix, 0.0)

    # weights: (n_target, n_peer), returns_filled.T: (n_peer, n_days)
    weighted_sum = precomputed.weights @ returns_filled.T  # (n_target, n_days)
    weight_sum = precomputed.weights @ valid_mask.T.astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(weight_sum > 1.0, weighted_sum / weight_sum, np.nan)
    # scores: (n_target, n_days)

    # Convert to long-format DataFrame via NumPy meshgrid (vectorized)
    n_days_out = len(wide_dates)
    tgt_indices, day_indices = np.meshgrid(
        np.arange(len(precomputed.target_symbols)),
        np.arange(n_days_out),
        indexing="ij",
    )
    flat_scores = scores.ravel()
    valid = ~np.isnan(flat_scores)

    dates_arr = np.array(wide_dates)
    syms_arr = np.array(precomputed.target_symbols)

    return pl.DataFrame({
        "DATETIME": dates_arr[day_indices.ravel()[valid]],
        "SYMBOL": syms_arr[tgt_indices.ravel()[valid]],
        "factor_value": flat_scores[valid],
    })


def compute_daily_factor_monthly_rebalance(
    monthly_weights: dict[str, PrecomputedWeights],
    peer_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Compute daily peer momentum with monthly-rebalanced weights.

    For each calendar month, uses the corresponding weight matrix.
    Days in months without weights are silently dropped.
    """
    peer_returns = peer_returns.with_columns(
        pl.col("DATETIME").str.slice(0, 7).alias("_month")
    )

    chunks = []
    for month_key in sorted(monthly_weights.keys()):
        month_returns = peer_returns.filter(pl.col("_month") == month_key).drop("_month")
        if month_returns.height == 0:
            continue
        factor = compute_daily_factor_single_peer(
            monthly_weights[month_key], month_returns
        )
        chunks.append(factor)

    if not chunks:
        return pl.DataFrame({"DATETIME": [], "SYMBOL": [], "factor_value": []})
    return pl.concat(chunks)


# ---------------------------------------------------------------------------
# Stage 2: Pooling
# ---------------------------------------------------------------------------

def pool_peer_factors(
    peer_factors: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Pool multiple peer factor DataFrames via nanmean.

    Args:
        peer_factors: Dict of {peer_short: DataFrame[DATETIME, SYMBOL, factor_value]}

    Returns:
        DataFrame with [DATETIME, SYMBOL, factor_value] — pooled signal
    """
    if len(peer_factors) == 1:
        return list(peer_factors.values())[0]

    # Concat all peer factors and group_by nanmean
    all_factors = pl.concat(list(peer_factors.values()))
    return (
        all_factors
        .group_by(["DATETIME", "SYMBOL"])
        .agg(pl.col("factor_value").mean())
        .drop_nulls("factor_value")
    )


# ---------------------------------------------------------------------------
# Stage 3: Daily labels
# ---------------------------------------------------------------------------

def compute_daily_labels(prices: pl.DataFrame) -> pl.DataFrame:
    """Compute 1-day forward return labels.

    label_T = close_{T+1} / close_T - 1

    Args:
        prices: DataFrame with [DATETIME, SYMBOL, close]

    Returns:
        DataFrame with [DATETIME, SYMBOL, daily_return]
    """
    return (
        prices.sort(["SYMBOL", "DATETIME"])
        .with_columns(
            ((pl.col("close").shift(-1) / pl.col("close")) - 1)
            .over("SYMBOL")
            .alias("daily_return")
        )
        .drop_nulls("daily_return")
        .select(["DATETIME", "SYMBOL", "daily_return"])
    )


# ---------------------------------------------------------------------------
# Stage 4: Factor transforms
# ---------------------------------------------------------------------------

def neutralize_factor(factor_df: pl.DataFrame) -> pl.DataFrame:
    """Cross-sectional demean: subtract daily cross-sectional mean.

    Removes market-level exposure so the factor is pure stock-specific signal.

    Args:
        factor_df: DataFrame with [DATETIME, SYMBOL, factor_value]

    Returns:
        DataFrame with demeaned factor_value
    """
    return (
        factor_df.with_columns(
            (pl.col("factor_value") - pl.col("factor_value").mean().over("DATETIME"))
            .alias("factor_value")
        )
    )


def build_composite_factor(
    lookback_factors: dict[int, pl.DataFrame],
) -> pl.DataFrame:
    """Build multi-lookback composite: z-score each lookback, then average.

    Args:
        lookback_factors: Dict of {lookback_days: DataFrame[DATETIME, SYMBOL, factor_value]}

    Returns:
        DataFrame with [DATETIME, SYMBOL, factor_value] — composite signal
    """
    zscored = []
    for lb, df in lookback_factors.items():
        z = df.with_columns(
            ((pl.col("factor_value") - pl.col("factor_value").mean().over("DATETIME"))
             / pl.col("factor_value").std().over("DATETIME"))
            .alias("factor_value")
        ).drop_nulls("factor_value")
        zscored.append(z)

    all_z = pl.concat(zscored)
    return (
        all_z
        .group_by(["DATETIME", "SYMBOL"])
        .agg(pl.col("factor_value").mean())
        .drop_nulls("factor_value")
    )


# ---------------------------------------------------------------------------
# Stage 5: Backtest
# ---------------------------------------------------------------------------

def subsample_factor(
    factor_df: pl.DataFrame,
    rebalance_every: int,
) -> pl.DataFrame:
    """Subsample factor to every Nth trading day.

    Keeps only every Nth unique date, reducing turnover.

    Args:
        factor_df: DataFrame with [DATETIME, SYMBOL, factor_value]
        rebalance_every: Keep every Nth date (1 = keep all)

    Returns:
        Subsampled DataFrame
    """
    if rebalance_every <= 1:
        return factor_df
    dates = factor_df.select("DATETIME").unique().sort("DATETIME")["DATETIME"].to_list()
    keep_dates = dates[::rebalance_every]
    return factor_df.filter(pl.col("DATETIME").is_in(keep_dates))


def build_factor_name_daily(
    peer_short: str | None,
    lookback: int,
    embedding_type: str,
    c: float,
    k: float,
    rebalance_every: int = 1,
    peers_tag: str | None = None,
) -> str:
    """Build factor name for daily pipeline.

    Args:
        peer_short: Peer market short name, or None for pooled
        lookback: Lookback in trading days
        embedding_type: e.g. 'w128'
        c: Sigmoid center
        k: Sigmoid steepness
        rebalance_every: Rebalance frequency suffix
        peers_tag: Optional tag for peer subset (e.g. 'us_kr')

    Returns:
        Name like 'peer_us_momentum_5d_w128_c0_99_k50' or
        'pooled_peer_momentum_5d_w128_c0_99_k50'
    """
    ck = encode_ck_params(c, k)
    rebal_suffix = f"_r{rebalance_every}d" if rebalance_every > 1 else ""
    if peer_short is None:
        prefix = f"pooled_{peers_tag}_" if peers_tag else "pooled_peer_"
        return f"{prefix}momentum_{lookback}d_{embedding_type}_{ck}{rebal_suffix}"
    return f"peer_{peer_short}_momentum_{lookback}d_{embedding_type}_{ck}{rebal_suffix}"


def backtest_exists(factor_name: str, backtest_dir: Path) -> bool:
    return (backtest_dir / TARGET_REGION / factor_name / "summary_df.json").exists()


def run_daily_backtest(
    factor_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    factor_name: str,
    backtest_dir: Path,
    skip_existing: bool = False,
) -> dict:
    """Run daily cross-sectional backtest.

    Args:
        factor_df: DataFrame with [DATETIME, SYMBOL, factor_value]
        labels_df: DataFrame with [DATETIME, SYMBOL, daily_return]
        factor_name: Factor name for output
        backtest_dir: Output directory root
        skip_existing: Skip if results exist

    Returns:
        Dict with backtest metrics
    """
    if skip_existing and backtest_exists(factor_name, backtest_dir):
        print(f"  SKIP backtest: {factor_name}")
        return {"factor": factor_name, "skipped": True}

    df = factor_df.join(labels_df, on=["DATETIME", "SYMBOL"], how="inner")
    if df.height == 0:
        print(f"  ERROR: No overlapping data for {factor_name}")
        return {"factor": factor_name, "error": "No overlapping data"}

    print(f"  Backtesting {factor_name}: {df.height} obs ... ", end="", flush=True)

    save_dir = backtest_dir / TARGET_REGION / factor_name
    save_dir.mkdir(parents=True, exist_ok=True)

    run_cross_section_backtest(
        df=df,
        factor_column_name="factor_value",
        datetime_column_name="DATETIME",
        symbol_column_name="SYMBOL",
        raw_label_column_name="daily_return",
        freq=FREQ,
        layers_use=LAYERS,
        fees=FEES,
        backtest_mode=BACKTEST_MODE,
        backtest_params=BACKTEST_PARAMS,
        lag=LAG,
        annual_days=ANNUAL_DAYS,
        factor_info_dict={"factor_name": factor_name, "region": TARGET_REGION},
        output_dir=save_dir,
    )

    with open(save_dir / "summary_df.json") as f:
        summary = json.load(f)

    metrics = summary.get("metrics", {})
    rank_ic = metrics.get("rank_ic", {}).get("by_cost", {}).get("0.0 bp")
    rank_icir = metrics.get("rank_icir_annual", {}).get("by_cost", {}).get("0.0 bp")
    sharpe_2bp = (
        metrics.get("long_short_ret_sharpe", {}).get("by_cost", {}).get("2.0 bp")
    )
    annual_ret_2bp = (
        metrics.get("long_short_ret_annual", {}).get("by_cost", {}).get("2.0 bp")
    )

    rank_ic = float(rank_ic) if rank_ic is not None else float("nan")
    rank_icir = float(rank_icir) if rank_icir is not None else float("nan")
    sharpe_2bp = float(sharpe_2bp) if sharpe_2bp is not None else float("nan")
    annual_ret_2bp = float(annual_ret_2bp) if annual_ret_2bp is not None else float("nan")

    gate = "PASS" if rank_icir > ICIR_GATE else "FAIL"
    print(f"IC={rank_ic:.4f}, ICIR={rank_icir:.3f} [{gate}], Sharpe={sharpe_2bp:.3f}")

    return {
        "factor": factor_name,
        "rank_ic": rank_ic,
        "rank_icir": rank_icir,
        "sharpe_2bp": sharpe_2bp,
        "annual_ret_2bp": annual_ret_2bp,
        "gate": gate,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    lookbacks = args.lookback if args.lookback else LOOKBACK_PERIODS
    data_root = Path(args.data_root)
    market_data_path = str(data_root / "market_data")
    cache_dir = data_root / "cache" / "similarity"
    factors_dir = Path(args.factors_dir) if args.factors_dir else PROJECT_ROOT / "factors_daily"
    backtest_dir = Path(args.backtest_dir) if args.backtest_dir else PROJECT_ROOT / "backtest_results_daily"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    rebalance_every = args.rebalance_every

    # Filter peer markets if --peers specified
    active_peers = {k: v for k, v in PEER_MARKETS.items() if k in args.peers} if args.peers else dict(PEER_MARKETS)
    peers_tag = "_".join(sorted(active_peers.keys())) if args.peers else None

    print("=" * 70)
    print("Daily Pooled Multi-Market Peer Momentum — Japan")
    print("=" * 70)
    print(f"  Target:       {TARGET_REGION}")
    print(f"  Peers:        {list(active_peers.keys())}")
    print(f"  Lookbacks:    {lookbacks} (trading days)")
    print(f"  Sigmoid:      c={args.c}, k={args.k}")
    print(f"  Embedding:    {args.embedding_type}")
    print(f"  Similarity:   {args.similarity}")
    if args.similarity == "blend":
        print(f"  Alpha(s):     {args.alpha}")
    if args.similarity in ("return_corr", "blend"):
        print(f"  Corr lookback:{args.corr_lookback}d")
    print(f"  Data root:    {data_root}")
    print(f"  Date range:   {args.start} to {args.end}")
    print(f"  Rebalance:    every {rebalance_every}d")
    print(f"  Neutralize:   {args.neutralize}")
    print(f"  Composite:    {args.composite}")
    print(f"  Pooled only:  {args.pooled_only}")
    print(f"  ICIR gate:    {ICIR_GATE}")
    print("=" * 70)

    # --- Load JP daily prices and compute labels ---
    print("\nLoading JP daily prices...")
    jp_prices = load_daily_close_prices(
        TARGET_REGION, args.start, args.end, source_path=market_data_path
    )
    print(f"  {jp_prices.height} price records")

    print("Computing daily forward return labels...")
    labels_df = compute_daily_labels(jp_prices)
    print(f"  {labels_df.height} label records")

    # --- Load peer daily prices (once, reused by all modes) ---
    print("\nLoading peer daily prices...")
    peer_prices_loaded: dict[str, pl.DataFrame] = {}
    failed_peers: list[str] = []
    for peer_short, peer_region in active_peers.items():
        try:
            prices = load_daily_close_prices(
                peer_region, args.start, args.end, source_path=market_data_path
            )
            # Forward-fill to handle cross-market calendar gaps
            prices = (
                prices.pivot(index="DATETIME", on="SYMBOL", values="close")
                .sort("DATETIME")
                .fill_null(strategy="forward")
                .unpivot(index="DATETIME", variable_name="SYMBOL", value_name="close")
                .drop_nulls("close")
            )
            peer_prices_loaded[peer_short] = prices
            print(f"  {peer_short}: {prices.height} records")
        except Exception as e:
            print(f"  {peer_short}: ERROR ({e}), skipping")
            failed_peers.append(peer_short)
    for p in failed_peers:
        active_peers.pop(p, None)

    # --- Similarity mode routing ---
    print(f"\nSimilarity mode: {args.similarity}")

    peer_weights_static: dict[str, PrecomputedWeights] = {}  # embedding mode
    monthly_corr_map: dict[str, MonthlyCorrelation] = {}     # return_corr/blend
    emb_cached_map: dict[str, CachedSimilarity] = {}         # blend mode

    if args.similarity == "embedding":
        for peer_short, peer_region in active_peers.items():
            try:
                pw = load_and_apply_sigmoid(
                    peer_region, args.embedding_type, args.c, args.k,
                    cache_dir=cache_dir,
                )
                peer_weights_static[peer_short] = pw
                print(f"  {peer_short}: {pw.weights.shape}")
            except FileNotFoundError:
                print(f"  {peer_short}: CACHE NOT FOUND, skipping")

        if not peer_weights_static:
            print("ERROR: No peer weights loaded. Ensure similarity caches exist.")
            sys.exit(1)

    if args.similarity in ("return_corr", "blend"):
        corr_cache = data_root / "cache" / "return_corr"
        for peer_short, peer_region in active_peers.items():
            if peer_short not in peer_prices_loaded:
                continue
            print(f"  Computing monthly correlations for {peer_short}...")
            mc = compute_monthly_corr_weights(
                jp_prices, peer_prices_loaded[peer_short],
                TARGET_REGION, peer_region,
                corr_lookback=args.corr_lookback,
                k=args.k, c=args.c,
                cache_dir=corr_cache,
            )
            monthly_corr_map[peer_short] = mc
            print(f"    {len(mc.weights)} months computed")

        if not monthly_corr_map:
            print("ERROR: No correlation weights computed.")
            sys.exit(1)

    if args.similarity == "blend":
        for peer_short, peer_region in active_peers.items():
            try:
                cached = load_similarity_cache(
                    TARGET_REGION, peer_region, args.embedding_type,
                    cache_dir=cache_dir,
                )
                emb_cached_map[peer_short] = cached
                print(f"  {peer_short} embedding pct_ranks: {cached.pct_ranks.shape}")
            except FileNotFoundError:
                print(f"  {peer_short}: EMBEDDING CACHE NOT FOUND, skipping blend")

    # --- Process each lookback ---
    all_results: list[dict] = []
    pooled_by_lookback: dict[int, pl.DataFrame] = {}

    for lookback in lookbacks:
        print(f"\n{'─' * 70}")
        print(f"Lookback: {lookback}d")
        print(f"{'─' * 70}")

        # --- Compute per-peer factors based on similarity mode ---
        peer_factors: dict[str, pl.DataFrame] = {}

        if args.similarity == "embedding":
            for peer_short in peer_weights_static:
                if peer_short not in peer_prices_loaded:
                    continue
                print(f"  Computing {peer_short} {lookback}d factor...")
                returns = compute_daily_returns(peer_prices_loaded[peer_short], lookback)
                factor = compute_daily_factor_single_peer(
                    peer_weights_static[peer_short], returns
                )
                peer_factors[peer_short] = factor
                print(f"    {factor.height} factor values")

        elif args.similarity == "return_corr":
            for peer_short in monthly_corr_map:
                if peer_short not in peer_prices_loaded:
                    continue
                print(f"  Computing {peer_short} {lookback}d factor (return_corr)...")
                returns = compute_daily_returns(peer_prices_loaded[peer_short], lookback)
                factor = compute_daily_factor_monthly_rebalance(
                    monthly_corr_map[peer_short].weights, returns
                )
                peer_factors[peer_short] = factor
                print(f"    {factor.height} factor values")

        elif args.similarity == "blend":
            # Blend mode: loop over alpha values, each produces its own pooled factor
            for alpha in args.alpha:
                print(f"\n  Blend alpha={alpha:.2f}")
                blend_peer_factors: dict[str, pl.DataFrame] = {}

                for peer_short in monthly_corr_map:
                    if peer_short not in emb_cached_map:
                        continue
                    if peer_short not in peer_prices_loaded:
                        continue
                    mc = monthly_corr_map[peer_short]
                    emb_c = emb_cached_map[peer_short]

                    # Blend pct_ranks per month, apply sigmoid
                    blended_monthly: dict[str, PrecomputedWeights] = {}
                    for mk, (corr_ranks, corr_t, corr_p) in mc.pct_ranks.items():
                        blended, t_syms, p_syms = blend_pct_ranks_aligned(
                            emb_c.pct_ranks, emb_c.target_symbols, emb_c.peer_symbols,
                            corr_ranks, corr_t, corr_p,
                            alpha=alpha,
                        )
                        w = sigmoid_weights(blended, k=args.k, c=args.c)
                        p_idx = {s: i for i, s in enumerate(p_syms)}
                        blended_monthly[mk] = PrecomputedWeights(w, t_syms, p_syms, p_idx)

                    returns = compute_daily_returns(peer_prices_loaded[peer_short], lookback)
                    factor = compute_daily_factor_monthly_rebalance(blended_monthly, returns)
                    blend_peer_factors[peer_short] = factor
                    print(f"    {peer_short}: {factor.height} values")

                # Pool and backtest
                if not blend_peer_factors:
                    continue
                pooled = pool_peer_factors(blend_peer_factors)
                if args.neutralize:
                    pooled = neutralize_factor(pooled)

                alpha_str = f"{alpha:.2f}".replace(".", "_")
                pooled_name = build_factor_name_daily(
                    None, lookback, f"blend_a{alpha_str}", args.c, args.k,
                    rebalance_every=rebalance_every, peers_tag=peers_tag,
                )
                if args.neutralize:
                    pooled_name += "_neutral"

                bt_pooled = subsample_factor(pooled, rebalance_every)
                result = run_daily_backtest(
                    bt_pooled, labels_df, pooled_name, backtest_dir,
                    skip_existing=args.skip_existing,
                )
                result["lookback"] = lookback
                result["type"] = f"blend_a{alpha}"
                result["peer"] = "all"
                all_results.append(result)
            continue  # Skip the default pooling/backtest below for blend mode

        # --- For embedding and return_corr modes: individual backtests + pooling ---

        # Backtest individual peer factors (unless --pooled-only)
        if not args.pooled_only:
            emb_tag = "retcorr" if args.similarity == "return_corr" else args.embedding_type
            for peer_short, factor_df in peer_factors.items():
                if args.neutralize:
                    factor_df = neutralize_factor(factor_df)
                factor_name = build_factor_name_daily(
                    peer_short, lookback, emb_tag, args.c, args.k,
                    rebalance_every=rebalance_every,
                )
                if args.neutralize:
                    factor_name += "_neutral"
                bt_factor = subsample_factor(factor_df, rebalance_every)
                result = run_daily_backtest(
                    bt_factor, labels_df, factor_name, backtest_dir,
                    skip_existing=args.skip_existing,
                )
                result["lookback"] = lookback
                result["type"] = "individual"
                result["peer"] = peer_short
                all_results.append(result)

        # Pool and backtest
        if not peer_factors:
            continue
        print(f"\n  Pooling {len(peer_factors)} peer signals...")
        pooled = pool_peer_factors(peer_factors)
        print(f"    {pooled.height} pooled factor values")

        if args.neutralize:
            pooled = neutralize_factor(pooled)
            print(f"    Neutralized (cross-sectional demean)")

        if args.composite:
            pooled_by_lookback[lookback] = pooled

        emb_tag = "retcorr" if args.similarity == "return_corr" else args.embedding_type
        pooled_name = build_factor_name_daily(
            None, lookback, emb_tag, args.c, args.k,
            rebalance_every=rebalance_every, peers_tag=peers_tag,
        )
        if args.neutralize:
            pooled_name += "_neutral"

        # Save pooled factor
        factor_h5_params: H5Params = {
            "source_path": str(factors_dir),
            "product": "Equity",
            "region": TARGET_REGION,
            "freq": "1d",
            "source": pooled_name,
        }
        h5_save(
            pooled.rename({"factor_value": pooled_name}),
            factor_h5_params,
            compression="gzip", compression_opts=4, overwrite=True,
        )

        bt_pooled = subsample_factor(pooled, rebalance_every)
        result = run_daily_backtest(
            bt_pooled, labels_df, pooled_name, backtest_dir,
            skip_existing=args.skip_existing,
        )
        result["lookback"] = lookback
        result["type"] = "pooled"
        result["peer"] = "all"
        all_results.append(result)

    # --- Composite factor (multi-lookback) ---
    if args.composite and len(pooled_by_lookback) > 1:
        print(f"\n{'─' * 70}")
        print(f"Composite: {list(pooled_by_lookback.keys())}d combined")
        print(f"{'─' * 70}")

        composite = build_composite_factor(pooled_by_lookback)
        print(f"  {composite.height} composite factor values")

        lb_str = "_".join(str(lb) for lb in sorted(pooled_by_lookback.keys()))
        composite_name = f"composite_{lb_str}d"
        if peers_tag:
            composite_name = f"composite_{peers_tag}_{lb_str}d"
        if args.neutralize:
            composite_name += "_neutral"
        composite_name += f"_{args.embedding_type}_{encode_ck_params(args.c, args.k)}"

        bt_composite = subsample_factor(composite, rebalance_every)
        result = run_daily_backtest(
            bt_composite, labels_df, composite_name, backtest_dir,
            skip_existing=args.skip_existing,
        )
        result["lookback"] = lb_str
        result["type"] = "composite"
        result["peer"] = "all"
        all_results.append(result)

    # --- Summary table ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'Type':<12}{'Peer':<8}{'Lookback':<10}"
        f"{'IC':<10}{'ICIR':<10}{'Sharpe':<10}{'AnnRet':<10}{'Gate':<6}"
    )
    print("-" * 76)

    passed = []
    for r in all_results:
        if r.get("skipped") or r.get("error"):
            continue
        tp = r.get("type", "?")
        peer = r.get("peer", "?")
        lb = f"{r.get('lookback', '?')}d"
        ic = r.get("rank_ic", float("nan"))
        icir = r.get("rank_icir", float("nan"))
        sh = r.get("sharpe_2bp", float("nan"))
        ar = r.get("annual_ret_2bp", float("nan"))
        gate = r.get("gate", "?")

        print(
            f"{tp:<12}{peer:<8}{lb:<10}"
            f"{ic:<10.4f}{icir:<10.3f}{sh:<10.3f}{ar:<10.3%}{gate:<6}"
        )

        if gate == "PASS":
            passed.append(r)

    print(f"\n{'=' * 70}")
    if passed:
        print(f"PRODUCTION GATE PASSED: {len(passed)} factor(s) with ICIR > {ICIR_GATE}")
        for r in passed:
            print(f"  {r['factor']} (ICIR={r['rank_icir']:.3f})")
    else:
        print(f"No factors passed ICIR > {ICIR_GATE} gate.")
    print(f"{'=' * 70}")

    # Save all results
    results_path = backtest_dir / TARGET_REGION / "daily_pooled_summary.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
