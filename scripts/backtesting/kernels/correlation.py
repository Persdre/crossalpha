"""Vectorized correlation utilities with Numba acceleration.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(nogil=True, cache=True, parallel=True)
def _rank_with_ties_numba(data: np.ndarray) -> np.ndarray:
    """Row-wise ranking with average tie-breaking (NaNs replaced with +inf).

    Parameters
    ----------
    data:
        2D array where NaN values have been replaced with +inf.

    Returns
    -------
    2D array of ranks (1-indexed); +inf positions get rank 0.0 (caller maps to NaN).
    """
    n_rows, n_cols = data.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for i in prange(n_rows):
        row = data[i]
        order = np.argsort(row)
        sorted_vals = row[order]

        ranks = np.empty(n_cols, dtype=np.float64)
        j = 0
        while j < n_cols:
            if sorted_vals[j] == np.inf:
                for k in range(j, n_cols):
                    ranks[order[k]] = 0.0
                break

            tie_start = j
            while j < n_cols - 1 and sorted_vals[j] == sorted_vals[j + 1]:
                j += 1
            tie_end = j

            avg_rank = (tie_start + tie_end + 2) / 2.0
            for k in range(tie_start, tie_end + 1):
                ranks[order[k]] = avg_rank
            j += 1

        result[i] = ranks

    return result


@njit(nogil=True, cache=True, parallel=True)
def _corr_numba(A: np.ndarray, B: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation with NaN handling."""
    n_rows, n_cols = A.shape
    result = np.empty(n_rows, dtype=np.float64)

    for i in prange(n_rows):
        sum_a = 0.0
        sum_b = 0.0
        n = 0

        for j in range(n_cols):
            if valid[i, j]:
                sum_a += A[i, j]
                sum_b += B[i, j]
                n += 1

        if n <= 1:
            result[i] = np.nan
            continue

        mean_a = sum_a / n
        mean_b = sum_b / n

        cov = 0.0
        var_a = 0.0
        var_b = 0.0

        for j in range(n_cols):
            if valid[i, j]:
                da = A[i, j] - mean_a
                db = B[i, j] - mean_b
                cov += da * db
                var_a += da * da
                var_b += db * db

        denom = np.sqrt(var_a) * np.sqrt(var_b)
        result[i] = cov / denom if denom > 0.0 else np.nan

    return result


def rank(X: np.ndarray) -> np.ndarray:
    """Row-wise ranking with NaN handling and average tie-breaking.

    Uses a fast path (pure NumPy) when no ties are detected; falls back to a
    Numba kernel when tie handling is required.
    """
    squeeze = X.ndim == 1
    if squeeze:
        X = X.reshape(1, -1)

    n_rows, n_cols = X.shape
    valid = ~np.isnan(X)
    X_filled = np.where(valid, X, np.inf)

    order = np.argsort(X_filled, axis=1)
    row_idx = np.arange(n_rows)[:, None]
    sorted_vals = X_filled[row_idx, order]
    base_ranks = np.arange(1, n_cols + 1, dtype=float)

    is_tie = (sorted_vals[:, 1:] == sorted_vals[:, :-1]) & (sorted_vals[:, :-1] != np.inf)
    if not is_tie.any():
        ranks = np.empty((n_rows, n_cols), dtype=float)
        ranks[row_idx, order] = base_ranks
        result = np.where(valid, ranks, np.nan)
        return result.squeeze() if squeeze else result

    ranked = _rank_with_ties_numba(X_filled.astype(np.float64))
    result = np.where(valid, ranked, np.nan)
    return result.squeeze() if squeeze else result


def corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation with NaN handling."""
    squeeze_a = A.ndim == 1
    if squeeze_a:
        A = A.reshape(1, -1)
        B = B.reshape(1, -1)

    valid = ~(np.isnan(A) | np.isnan(B))
    n = valid.sum(axis=1)

    result = _corr_numba(A.astype(np.float64), B.astype(np.float64), valid)
    result = np.where(n > 1, result, np.nan)
    return result.item() if squeeze_a else result


def spearman(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Spearman correlation = Pearson correlation on ranks."""
    return corr(rank(A), rank(B))

