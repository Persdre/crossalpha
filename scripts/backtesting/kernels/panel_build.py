"""Panel construction kernels (NumPy long format and scatter)."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True, parallel=True)
def compact_cs_panel_long(
    factor_mat: np.ndarray,
    label_mat: np.ndarray,
    valid_label_mask: np.ndarray,
    dt_offsets_rows: np.ndarray,
    sym_code_out: np.ndarray,
    factor_out: np.ndarray,
    label_out: np.ndarray,
) -> None:
    """Compact a dense (dt×sym) panel into long arrays.

    Row order is datetime-major, then symbol-major, matching the legacy Polars
    construction (after filtering invalid labels).
    """
    n_dt, n_sym = valid_label_mask.shape
    for d in prange(n_dt):
        out_pos = int(dt_offsets_rows[d])
        k = 0
        for s in range(n_sym):
            if valid_label_mask[d, s]:
                idx = out_pos + k
                sym_code_out[idx] = s
                factor_out[idx] = factor_mat[d, s]
                label_out[idx] = label_mat[d, s]
                k += 1


@njit(cache=True, nogil=True, parallel=True)
def scatter_long_to_dense_2(
    factor: np.ndarray,
    label: np.ndarray,
    sym_code: np.ndarray,
    dt_offsets_rows: np.ndarray,
    n_symbols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter long arrays back to dense (dt×sym) matrices filled with NaNs."""
    n_dt = dt_offsets_rows.shape[0] - 1
    factor_mat = np.empty((n_dt, n_symbols), dtype=np.float64)
    label_mat = np.empty((n_dt, n_symbols), dtype=np.float64)
    factor_mat[:] = np.nan
    label_mat[:] = np.nan

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])
        for i in range(start, end):
            s = int(sym_code[i])
            factor_mat[d, s] = factor[i]
            label_mat[d, s] = label[i]

    return factor_mat, label_mat


@njit(cache=True, nogil=True, parallel=True)
def scatter_long_to_dense_1(
    values: np.ndarray,
    sym_code: np.ndarray,
    dt_offsets_rows: np.ndarray,
    n_symbols: int,
) -> np.ndarray:
    """Scatter long arrays back to a dense (dt×sym) matrix filled with NaNs."""
    n_dt = dt_offsets_rows.shape[0] - 1
    out = np.empty((n_dt, n_symbols), dtype=np.float64)
    out[:] = np.nan

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])
        for i in range(start, end):
            s = int(sym_code[i])
            out[d, s] = values[i]

    return out
