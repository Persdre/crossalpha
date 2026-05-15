"""Numba kernels for turnover computation."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True, parallel=True)
def turnover_obs_cs_4(
    pos_ls: np.ndarray,
    pos_long: np.ndarray,
    pos_short: np.ndarray,
    pos_passive: np.ndarray,
    sym_offsets: np.ndarray,
    sym_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-row turnover contributions (abs pos change) for CS mode."""
    n_rows = pos_ls.shape[0]
    n_syms = sym_offsets.shape[0] - 1

    out_ls = np.empty(n_rows, dtype=np.float64)
    out_long = np.empty(n_rows, dtype=np.float64)
    out_short = np.empty(n_rows, dtype=np.float64)
    out_passive = np.empty(n_rows, dtype=np.float64)

    for s in prange(n_syms):
        prev_ls = 0.0
        prev_long = 0.0
        prev_short = 0.0
        prev_passive = 0.0

        for k in range(sym_offsets[s], sym_offsets[s + 1]):
            i = sym_rows[k]

            v = pos_ls[i]
            out_ls[i] = abs(v - prev_ls)
            prev_ls = v

            v = pos_long[i]
            out_long[i] = abs(v - prev_long)
            prev_long = v

            v = pos_short[i]
            out_short[i] = abs(v - prev_short)
            prev_short = v

            v = pos_passive[i]
            out_passive[i] = abs(v - prev_passive)
            prev_passive = v

    return out_ls, out_long, out_short, out_passive

