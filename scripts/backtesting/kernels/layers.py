"""Numba kernels for layer bookkeeping."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True, parallel=True)
def assign_layers_ordinal_desc(
    factor: np.ndarray,
    dt_offsets_rows: np.ndarray,
    n_layers: int,
) -> np.ndarray:
    """Assign ordinal, descending ranks to quantile layers per datetime group.

    Matches the legacy Polars logic:
    - ranks are computed per datetime on finite factors only
    - rank method is "ordinal", descending
    - ties are broken deterministically by original row order (stable)
    - layers are assigned by floor(n_valid * frac) thresholds

    Returns `layer_code` as int16: 1..n_layers for valid factor rows, else 0.
    """
    n_dt = dt_offsets_rows.shape[0] - 1
    n_rows = factor.shape[0]
    out = np.zeros(n_rows, dtype=np.int16)

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])
        group_size = end - start
        if group_size <= 0:
            continue

        vals = np.empty(group_size, dtype=np.float64)
        rows = np.empty(group_size, dtype=np.int64)
        m = 0
        for i in range(start, end):
            v = factor[i]
            if np.isfinite(v):
                vals[m] = v
                rows[m] = i
                m += 1

        if m == 0:
            continue

        order = np.argsort(vals[:m])

        layer = 1
        upper = int(m * layer / n_layers)
        rank = 1

        p = m - 1
        while p >= 0:
            v = vals[order[p]]
            tie_start = p
            while tie_start > 0 and vals[order[tie_start - 1]] == v:
                tie_start -= 1

            k = p - tie_start + 1
            tie_rows = np.empty(k, dtype=np.int64)
            for t in range(k):
                tie_rows[t] = rows[order[tie_start + t]]
            tie_rows = np.sort(tie_rows)

            for t in range(k):
                while rank > upper and layer < n_layers:
                    layer += 1
                    upper = int(m * layer / n_layers)
                out[int(tie_rows[t])] = np.int16(layer)
                rank += 1

            p = tie_start - 1

    return out


@njit(cache=True, nogil=True, parallel=True)
def count_layers_by_dt(
    layer_code: np.ndarray,
    dt_offsets_rows: np.ndarray,
    n_layers: int,
) -> np.ndarray:
    """Count layer membership per datetime group.

    `layer_code` is 1..n_layers, or 0 for null/unknown.
    """
    n_dt = dt_offsets_rows.shape[0] - 1
    out = np.zeros((n_dt, n_layers), dtype=np.int32)

    for d in prange(n_dt):
        start = dt_offsets_rows[d]
        end = dt_offsets_rows[d + 1]
        for i in range(start, end):
            lc = layer_code[i]
            if 1 <= lc <= n_layers:
                out[d, lc - 1] += 1

    return out
