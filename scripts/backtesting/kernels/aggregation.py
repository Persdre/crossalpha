"""Numba kernels for aggregation (row -> datetime -> daily)."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True, parallel=True)
def sum_by_group_4(
    x0: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sum 4 aligned vectors over contiguous groups defined by offsets."""
    n_groups = offsets.shape[0] - 1
    out0 = np.empty(n_groups, dtype=np.float64)
    out1 = np.empty(n_groups, dtype=np.float64)
    out2 = np.empty(n_groups, dtype=np.float64)
    out3 = np.empty(n_groups, dtype=np.float64)

    for g in prange(n_groups):
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        start = offsets[g]
        end = offsets[g + 1]
        for i in range(start, end):
            s0 += x0[i]
            s1 += x1[i]
            s2 += x2[i]
            s3 += x3[i]
        out0[g] = s0
        out1[g] = s1
        out2[g] = s2
        out3[g] = s3

    return out0, out1, out2, out3


@njit(cache=True, nogil=True, parallel=True)
def sum_by_group_1(x: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Sum a vector over contiguous groups defined by offsets."""
    n_groups = offsets.shape[0] - 1
    out = np.empty(n_groups, dtype=np.float64)

    for g in prange(n_groups):
        s = 0.0
        start = offsets[g]
        end = offsets[g + 1]
        for i in range(start, end):
            s += x[i]
        out[g] = s

    return out

