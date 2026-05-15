"""Return correlation computation for peer similarity."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import NamedTuple

import h5py


def compute_return_correlation_matrix(
    target_returns: np.ndarray,
    peer_returns: np.ndarray,
    min_overlap: int = 120,
) -> np.ndarray:
    """Compute pairwise Pearson correlation of daily returns.

    Uses pairwise-complete observations: for each (target, peer) pair,
    only days where both have valid (non-NaN) returns are used.

    Args:
        target_returns: Shape (n_days, n_target) daily returns
        peer_returns: Shape (n_days, n_peer) daily returns
        min_overlap: Minimum valid overlapping days required.
                     Below this, correlation is set to NaN.

    Returns:
        Shape (n_target, n_peer) correlation matrix.
    """
    target = np.asarray(target_returns, dtype=np.float64)
    peer = np.asarray(peer_returns, dtype=np.float64)

    n_days, n_target = target.shape
    _, n_peer = peer.shape

    corr = np.full((n_target, n_peer), np.nan, dtype=np.float64)

    target_valid = ~np.isnan(target)
    peer_valid = ~np.isnan(peer)

    for i in range(n_target):
        t_mask = target_valid[:, i]
        t_vals = target[:, i]

        for j in range(n_peer):
            p_mask = peer_valid[:, j]
            both_valid = t_mask & p_mask
            n_valid = both_valid.sum()

            if n_valid < min_overlap:
                continue

            t_sub = t_vals[both_valid]
            p_sub = peer[both_valid, j]

            t_mean = t_sub.mean()
            p_mean = p_sub.mean()

            t_centered = t_sub - t_mean
            p_centered = p_sub - p_mean

            num = np.dot(t_centered, p_centered)
            denom = np.sqrt(np.dot(t_centered, t_centered) * np.dot(p_centered, p_centered))

            if denom > 0:
                corr[i, j] = num / denom
            else:
                corr[i, j] = 0.0

    return corr


class CachedCorrelation(NamedTuple):
    """Container for cached correlation data."""

    corr_matrix: np.ndarray
    target_symbols: list[str]
    peer_symbols: list[str]


def _build_corr_cache_path(
    target_region: str,
    peer_region: str,
    rebalance_month: str,
    cache_dir: Path,
) -> Path:
    """Build path for correlation cache file."""
    month_key = rebalance_month[:7]  # "2024-06-30" -> "2024-06"
    subdir = cache_dir / f"{target_region}_from_{peer_region}"
    return subdir / f"{month_key}.h5"


def save_correlation_cache(
    corr_matrix: np.ndarray,
    target_symbols: list[str],
    peer_symbols: list[str],
    target_region: str,
    peer_region: str,
    rebalance_month: str,
    cache_dir: Path | None = None,
) -> Path:
    """Save correlation matrix to cache."""
    if cache_dir is None:
        from scripts.kernels.loaders import PROJECT_ROOT
        cache_dir = PROJECT_ROOT / "cache" / "return_corr"

    path = _build_corr_cache_path(target_region, peer_region, rebalance_month, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset(
            "corr_matrix",
            data=corr_matrix.astype(np.float64),
            compression="gzip",
            compression_opts=4,
        )
        f.attrs["target_symbols"] = json.dumps(target_symbols)
        f.attrs["peer_symbols"] = json.dumps(peer_symbols)

    return path


def load_correlation_cache(
    target_region: str,
    peer_region: str,
    rebalance_month: str,
    cache_dir: Path | None = None,
) -> CachedCorrelation:
    """Load correlation matrix from cache."""
    if cache_dir is None:
        from scripts.kernels.loaders import PROJECT_ROOT
        cache_dir = PROJECT_ROOT / "cache" / "return_corr"

    path = _build_corr_cache_path(target_region, peer_region, rebalance_month, cache_dir)

    if not path.exists():
        raise FileNotFoundError(f"Correlation cache not found: {path}")

    with h5py.File(path, "r") as f:
        corr_matrix = f["corr_matrix"][:].astype(np.float64)
        target_symbols = json.loads(f.attrs["target_symbols"])
        peer_symbols = json.loads(f.attrs["peer_symbols"])

    return CachedCorrelation(
        corr_matrix=corr_matrix,
        target_symbols=target_symbols,
        peer_symbols=peer_symbols,
    )


def corr_cache_exists(
    target_region: str,
    peer_region: str,
    rebalance_month: str,
    cache_dir: Path | None = None,
) -> bool:
    """Check if correlation cache exists for a given month."""
    if cache_dir is None:
        from scripts.kernels.loaders import PROJECT_ROOT
        cache_dir = PROJECT_ROOT / "cache" / "return_corr"

    path = _build_corr_cache_path(target_region, peer_region, rebalance_month, cache_dir)
    return path.exists()


def compute_correlation_from_prices(
    target_prices: "pl.DataFrame",
    peer_prices: "pl.DataFrame",
    corr_lookback: int = 252,
    min_overlap: int = 120,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute return correlation matrix from price DataFrames.

    Computes daily returns from close prices, then pairwise Pearson
    correlation using the last `corr_lookback` trading days.

    Args:
        target_prices: DataFrame with [DATETIME, SYMBOL, close]
        peer_prices: DataFrame with [DATETIME, SYMBOL, close]
        corr_lookback: Number of trailing trading days to use
        min_overlap: Minimum valid overlapping days for correlation

    Returns:
        Tuple of (corr_matrix, target_symbols, peer_symbols)
        corr_matrix shape: (n_target, n_peer)
    """
    import polars as pl

    def _to_return_wide(prices_df: pl.DataFrame):
        """Convert long-format prices to wide return DataFrame with DATETIME index."""
        return (
            prices_df.sort(["SYMBOL", "DATETIME"])
            .with_columns(
                ((pl.col("close") / pl.col("close").shift(1)) - 1)
                .over("SYMBOL")
                .alias("ret")
            )
            .drop_nulls("ret")
            .pivot(index="DATETIME", on="SYMBOL", values="ret")
            .sort("DATETIME")
        )

    target_wide = _to_return_wide(target_prices)
    peer_wide = _to_return_wide(peer_prices)

    # Align to common dates (intersection of trading calendars)
    common = target_wide.join(peer_wide.select("DATETIME"), on="DATETIME", how="inner")
    peer_common = peer_wide.join(target_wide.select("DATETIME"), on="DATETIME", how="inner")

    # Take last corr_lookback rows
    common = common.tail(corr_lookback)
    peer_common = peer_common.tail(corr_lookback)

    target_symbols = sorted([c for c in common.columns if c != "DATETIME"])
    peer_symbols = sorted([c for c in peer_common.columns if c != "DATETIME"])

    target_matrix = common.select(target_symbols).to_numpy().astype(np.float64)
    peer_matrix = peer_common.select(peer_symbols).to_numpy().astype(np.float64)

    corr = compute_return_correlation_matrix(
        target_matrix, peer_matrix, min_overlap=min_overlap
    )
    return corr, target_symbols, peer_symbols


def blend_pct_ranks(
    emb_ranks: np.ndarray,
    corr_ranks: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend embedding and correlation percentile ranks.

    Formula: blended = alpha * emb_ranks + (1 - alpha) * corr_ranks

    Args:
        emb_ranks: Embedding percentile ranks (n_target, n_peer)
        corr_ranks: Correlation percentile ranks (n_target, n_peer)
        alpha: Weight for embedding ranks (0=pure corr, 1=pure embedding)

    Returns:
        Blended percentile ranks (n_target, n_peer)
    """
    return alpha * emb_ranks + (1.0 - alpha) * corr_ranks


def blend_pct_ranks_aligned(
    emb_ranks: np.ndarray,
    emb_target_symbols: list[str],
    emb_peer_symbols: list[str],
    corr_ranks: np.ndarray,
    corr_target_symbols: list[str],
    corr_peer_symbols: list[str],
    alpha: float = 0.5,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Blend percentile ranks with symbol alignment.

    Uses intersection of symbols from both sources.

    Args:
        emb_ranks: Embedding percentile ranks (n_emb_target, n_emb_peer)
        emb_target_symbols: Target symbols for embedding matrix
        emb_peer_symbols: Peer symbols for embedding matrix
        corr_ranks: Correlation percentile ranks (n_corr_target, n_corr_peer)
        corr_target_symbols: Target symbols for correlation matrix
        corr_peer_symbols: Peer symbols for correlation matrix
        alpha: Weight for embedding ranks

    Returns:
        Tuple of (blended_ranks, target_symbols, peer_symbols)
    """
    target_syms = sorted(set(emb_target_symbols) & set(corr_target_symbols))
    peer_syms = sorted(set(emb_peer_symbols) & set(corr_peer_symbols))

    emb_t_idx = {s: i for i, s in enumerate(emb_target_symbols)}
    emb_p_idx = {s: i for i, s in enumerate(emb_peer_symbols)}
    corr_t_idx = {s: i for i, s in enumerate(corr_target_symbols)}
    corr_p_idx = {s: i for i, s in enumerate(corr_peer_symbols)}

    emb_t_sel = [emb_t_idx[s] for s in target_syms]
    emb_p_sel = [emb_p_idx[s] for s in peer_syms]
    corr_t_sel = [corr_t_idx[s] for s in target_syms]
    corr_p_sel = [corr_p_idx[s] for s in peer_syms]

    emb_aligned = emb_ranks[np.ix_(emb_t_sel, emb_p_sel)]
    corr_aligned = corr_ranks[np.ix_(corr_t_sel, corr_p_sel)]

    blended = blend_pct_ranks(emb_aligned, corr_aligned, alpha)
    return blended, target_syms, peer_syms
