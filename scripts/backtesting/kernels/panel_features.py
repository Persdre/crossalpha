"""Per-symbol/time-series sequential features (lag and reference price)."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True, parallel=True)
def cs_lag_and_ref_price(
    factor_raw: np.ndarray,
    label: np.ndarray,
    sym_offsets: np.ndarray,
    sym_rows: np.ndarray,
    lag: int,
    factor_out: np.ndarray,
    ref_price_out: np.ndarray,
) -> None:
    """Apply per-symbol lag and build per-symbol reference prices.

    Semantics match the legacy Polars pipeline:
    - lag is applied *after* label filtering, within each symbol's remaining rows
    - ref_price is cumprod(1+label) shifted by 1, starting at 1.0 for each symbol
    """
    n_symbols = sym_offsets.shape[0] - 1
    use_lag = lag > 0

    for s in prange(n_symbols):
        start = int(sym_offsets[s])
        end = int(sym_offsets[s + 1])

        prev_ref = 1.0
        for k in range(start, end):
            row = int(sym_rows[k])
            ref_price_out[row] = prev_ref
            prev_ref *= 1.0 + label[row]

        if not use_lag:
            for k in range(start, end):
                row = int(sym_rows[k])
                factor_out[row] = factor_raw[row]
            continue

        seq_len = end - start
        for t in range(seq_len):
            row = int(sym_rows[start + t])
            if t < lag:
                factor_out[row] = np.nan
            else:
                prev_row = int(sym_rows[start + t - lag])
                factor_out[row] = factor_raw[prev_row]


@njit(cache=True, nogil=True)
def ts_lag_and_ref_price(
    factor_raw: np.ndarray,
    label: np.ndarray,
    lag: int,
    factor_out: np.ndarray,
    ref_price_out: np.ndarray,
) -> None:
    """Apply lag and build reference price for a single time series."""
    n = factor_raw.shape[0]
    if lag <= 0:
        for i in range(n):
            factor_out[i] = factor_raw[i]
    else:
        for i in range(min(lag, n)):
            factor_out[i] = np.nan
        for i in range(lag, n):
            factor_out[i] = factor_raw[i - lag]

    prev_ref = 1.0
    for i in range(n):
        ref_price_out[i] = prev_ref
        prev_ref *= 1.0 + label[i]

