"""Weighting kernels for momentum score computation (Numba-accelerated)."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# Default benchmark: equal weight on all ten distilled disclosure categories.
# Category-specific or learned-weight variants should be treated as ablations.
CATEGORY_WEIGHTS: dict[str, float] = {
    "main_business_segments": 0.10,
    "core_technologies_production_methods": 0.10,
    "primary_customers_markets": 0.10,
    "geographic_coverage": 0.10,
    "supply_chain_position": 0.10,
    "strategic_focus_rd_direction": 0.10,
    "revenue_model": 0.10,
    "key_competitors_industry_positioning": 0.10,
    "financial_scale_growth_profile": 0.10,
    "value_proposition_product_differentiation": 0.10,
}

CATEGORIES = list(CATEGORY_WEIGHTS.keys())


@njit(cache=True, parallel=True)
def _sigmoid_weights_kernel(
    percentile_ranks: np.ndarray,
    k: float,
    c: float,
) -> np.ndarray:
    """Apply sigmoid weighting element-wise."""
    n_rows, n_cols = percentile_ranks.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    for i in prange(n_rows):
        for j in range(n_cols):
            x = percentile_ranks[i, j]
            result[i, j] = 1.0 / (1.0 + np.exp(-k * (x - c)))
    return result


def sigmoid_weights(
    percentile_ranks: np.ndarray,
    k: float = 50.0,
    c: float = 0.99,
) -> np.ndarray:
    """Apply sigmoid weighting to percentile ranks.

    Formula: w(x) = 1 / (1 + exp(-k * (x - c)))

    With k=50, c=0.99, this focuses on top ~1% most similar stocks.

    Args:
        percentile_ranks: Array of percentile ranks in [0, 1]
        k: Steepness parameter (default: 50)
        c: Center parameter (default: 0.99)

    Returns:
        Sigmoid weights in (0, 1)
    """
    arr = np.ascontiguousarray(percentile_ranks, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        return _sigmoid_weights_kernel(arr, k, c).ravel()
    return _sigmoid_weights_kernel(arr, k, c)


def threshold_weights(
    percentile_ranks: np.ndarray,
    threshold: float = 0.95,
) -> np.ndarray:
    """Apply hard threshold weighting to percentile ranks.

    Returns 1.0 for ranks >= threshold, 0.0 otherwise.
    This is the threshold equivalent of sigmoid_weights().

    Args:
        percentile_ranks: Array of percentile ranks in [0, 1]
        threshold: Cutoff percentile (e.g. 0.95 = top 5%)

    Returns:
        Binary weight array (same shape as input), dtype float64
    """
    arr = np.asarray(percentile_ranks, dtype=np.float64)
    return np.where(arr >= threshold, 1.0, 0.0)


@njit(cache=True, parallel=True)
def _compute_momentum_core(
    pct_ranks: np.ndarray,
    returns: np.ndarray,
    k: float,
    c: float,
) -> np.ndarray:
    """Core momentum computation.

    Args:
        pct_ranks: Shape (n_jp, n_us) percentile ranks
        returns: Shape (n_us,) sector-relative returns
        k: Sigmoid steepness
        c: Sigmoid center

    Returns:
        Shape (n_jp,) momentum scores
    """
    n_jp, n_us = pct_ranks.shape
    result = np.empty(n_jp, dtype=np.float64)

    for i in prange(n_jp):
        weighted_sum = 0.0
        weight_sum = 0.0
        for j in range(n_us):
            r = returns[j]
            if not np.isnan(r):
                x = pct_ranks[i, j]
                w = 1.0 / (1.0 + np.exp(-k * (x - c)))
                weighted_sum += w * r
                weight_sum += w

        if weight_sum > 0:
            result[i] = weighted_sum / weight_sum
        else:
            result[i] = np.nan

    return result


def compute_momentum_score(
    similarity_matrix: np.ndarray,
    us_sector_relative_returns: np.ndarray,
    k: float = 50.0,
    c: float = 0.99,
) -> np.ndarray:
    """Compute momentum scores for JP stocks.

    Args:
        similarity_matrix: Shape (n_jp, n_us) similarity scores
        us_sector_relative_returns: Shape (n_us,) sector-relative returns
        k: Sigmoid steepness (default: 50)
        c: Sigmoid center (default: 0.99)

    Returns:
        Momentum scores of shape (n_jp,)
    """
    from scripts.kernels.ranking import percentile_rank_fast

    sim = np.ascontiguousarray(similarity_matrix, dtype=np.float64)
    returns = np.ascontiguousarray(us_sector_relative_returns, dtype=np.float64)

    pct_ranks = percentile_rank_fast(sim, axis=1)
    return _compute_momentum_core(pct_ranks, returns, k, c)
