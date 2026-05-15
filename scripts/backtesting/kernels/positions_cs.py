"""Position construction kernels for cross-sectional strategies."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True, parallel=True)
def positions_cs_from_raw(
    raw: np.ndarray,
    include: np.ndarray,
    dt_offsets_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply legacy-equivalent position constraints per datetime group.

    Parameters
    ----------
    raw:
        Raw signal per observation (e.g., layer membership or normalized factor).
    include:
        Boolean mask: True for rows participating in long/short constraints.
    dt_offsets_rows:
        Contiguous datetime group offsets into the long arrays.

    Returns
    -------
    (pos_ls, pos_long, pos_short, pos_passive)
    """
    n_rows = raw.shape[0]
    n_dt = dt_offsets_rows.shape[0] - 1

    pos_ls = np.zeros(n_rows, dtype=np.float64)
    pos_long = np.zeros(n_rows, dtype=np.float64)
    pos_short = np.zeros(n_rows, dtype=np.float64)
    pos_passive = np.zeros(n_rows, dtype=np.float64)

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])
        group_size = end - start
        if group_size <= 0:
            continue

        passive = 1.0 / group_size
        for i in range(start, end):
            pos_passive[i] = passive

        sum_raw = 0.0
        cnt = 0
        for i in range(start, end):
            if include[i]:
                sum_raw += raw[i]
                cnt += 1
        if cnt == 0:
            continue

        mean_raw = sum_raw / cnt

        gross = 0.0
        for i in range(start, end):
            if include[i]:
                gross += abs(raw[i] - mean_raw)
        if gross <= 0.0:
            continue

        pos_sum = 0.0
        neg_gross = 0.0
        for i in range(start, end):
            if include[i]:
                v = (raw[i] - mean_raw) / gross
                pos_ls[i] = v
                if v > 0.0:
                    pos_sum += v
                elif v < 0.0:
                    neg_gross += -v

        if pos_sum > 0.0:
            inv = 1.0 / pos_sum
            for i in range(start, end):
                v = pos_ls[i]
                if include[i] and v > 0.0:
                    pos_long[i] = v * inv

        if neg_gross > 0.0:
            inv = 1.0 / neg_gross
            for i in range(start, end):
                v = pos_ls[i]
                if include[i] and v < 0.0:
                    pos_short[i] = v * inv

    return pos_ls, pos_long, pos_short, pos_passive

