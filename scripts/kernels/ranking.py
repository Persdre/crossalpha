"""Percentile rank transformation kernels (Numba-accelerated)."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _argsort(arr: np.ndarray) -> np.ndarray:
    """Return indices that would sort the array."""
    n = len(arr)
    indices = np.arange(n)
    # Simple insertion sort for stability with ties
    for i in range(1, n):
        key_idx = indices[i]
        key_val = arr[key_idx]
        j = i - 1
        while j >= 0 and arr[indices[j]] > key_val:
            indices[j + 1] = indices[j]
            j -= 1
        indices[j + 1] = key_idx
    return indices


@njit(cache=True)
def _rank_row(row: np.ndarray) -> np.ndarray:
    """Compute percentile rank for a single row (with tie handling)."""
    n = len(row)
    result = np.empty(n, dtype=np.float64)

    # Count valid (non-NaN) values
    valid_count = 0
    for i in range(n):
        if not np.isnan(row[i]):
            valid_count += 1

    if valid_count == 0:
        for i in range(n):
            result[i] = np.nan
        return result

    # Get sort order for valid values
    valid_vals = np.empty(valid_count, dtype=np.float64)
    valid_idx = np.empty(valid_count, dtype=np.int64)
    vi = 0
    for i in range(n):
        if not np.isnan(row[i]):
            valid_vals[vi] = row[i]
            valid_idx[vi] = i
            vi += 1

    # Sort and assign ranks
    order = _argsort(valid_vals)
    ranks = np.empty(valid_count, dtype=np.float64)
    for i in range(valid_count):
        ranks[order[i]] = (i + 1) / valid_count

    # Fill result
    for i in range(n):
        result[i] = np.nan
    for i in range(valid_count):
        result[valid_idx[i]] = ranks[i]

    return result


@njit(cache=True, parallel=True)
def _percentile_rank_rows(matrix: np.ndarray) -> np.ndarray:
    """Compute percentile rank for each row."""
    n_rows, n_cols = matrix.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    for i in prange(n_rows):
        result[i] = _rank_row(matrix[i])
    return result


def percentile_rank(
    matrix: np.ndarray,
    axis: int = 1,
) -> np.ndarray:
    """Convert values to percentile ranks along specified axis.

    Args:
        matrix: Input matrix of shape (n, m)
        axis: Axis along which to compute ranks (default: 1)

    Returns:
        Percentile ranks in range [0, 1] with same shape as input.
        Ties are handled by ordinal rank (no averaging).
    """
    arr = np.ascontiguousarray(matrix, dtype=np.float64)
    if axis == 0:
        arr = arr.T
    result = _percentile_rank_rows(arr)
    if axis == 0:
        result = result.T
    return result


@njit(cache=True, parallel=True)
def _percentile_rank_fast_rows(matrix: np.ndarray) -> np.ndarray:
    """Fast percentile rank using argsort (no tie handling).

    NaN values are excluded from ranking. Denominator uses valid_count
    (number of non-NaN values) so that percentile ranks span (0, 1].
    """
    n_rows, n_cols = matrix.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for i in prange(n_rows):
        row = matrix[i]
        # Check for NaN and count valid values
        filled = np.empty(n_cols, dtype=np.float64)
        has_nan = np.empty(n_cols, dtype=np.bool_)
        valid_count = 0
        for j in range(n_cols):
            if np.isnan(row[j]):
                filled[j] = -np.inf
                has_nan[j] = True
            else:
                filled[j] = row[j]
                has_nan[j] = False
                valid_count += 1

        if valid_count == 0:
            for j in range(n_cols):
                result[i, j] = np.nan
            continue

        # Double argsort to get ranks
        order = np.argsort(filled)
        ranks = np.empty(n_cols, dtype=np.int64)
        for j in range(n_cols):
            ranks[order[j]] = j

        # Rebase ranks: NaN entries got the lowest ranks (filled with -inf).
        # We need to subtract the number of NaN entries so valid ranks start at 0.
        n_nan = n_cols - valid_count

        # Convert to percentile and restore NaN
        for j in range(n_cols):
            if has_nan[j]:
                result[i, j] = np.nan
            else:
                result[i, j] = (ranks[j] - n_nan + 1) / valid_count

    return result


def percentile_rank_fast(
    matrix: np.ndarray,
    axis: int = 1,
) -> np.ndarray:
    """Fast percentile rank using argsort (no tie handling).

    Args:
        matrix: Input matrix of shape (n, m)
        axis: Axis along which to compute ranks (default: 1)

    Returns:
        Percentile ranks in range (0, 1] with same shape as input.
    """
    arr = np.ascontiguousarray(matrix, dtype=np.float64)
    if axis == 0:
        arr = arr.T
    result = _percentile_rank_fast_rows(arr)
    if axis == 0:
        result = result.T
    return result
