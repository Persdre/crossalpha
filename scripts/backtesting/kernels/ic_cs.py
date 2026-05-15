"""Cross-sectional IC kernels (by datetime, by datetime×layer)."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True)
def _rank_average(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    ranks = np.empty(n, dtype=np.float64)
    order = np.argsort(values)

    p = 0
    while p < n:
        v = values[order[p]]
        q = p
        while q + 1 < n and values[order[q + 1]] == v:
            q += 1

        avg_rank = (p + q + 2) / 2.0
        for t in range(p, q + 1):
            ranks[order[t]] = avg_rank
        p = q + 1

    return ranks


@njit(cache=True, nogil=True)
def _spearman_1d(a: np.ndarray, b: np.ndarray) -> float:
    n = a.shape[0]
    if n <= 1:
        return np.nan

    ra = _rank_average(a)
    rb = _rank_average(b)

    sum_a = 0.0
    sum_b = 0.0
    for i in range(n):
        sum_a += ra[i]
        sum_b += rb[i]
    mean_a = sum_a / n
    mean_b = sum_b / n

    cov = 0.0
    var_a = 0.0
    var_b = 0.0
    for i in range(n):
        da = ra[i] - mean_a
        db = rb[i] - mean_b
        cov += da * db
        var_a += da * da
        var_b += db * db

    denom = np.sqrt(var_a) * np.sqrt(var_b)
    return cov / denom if denom > 0.0 else np.nan


@njit(cache=True, nogil=True)
def _rank_average_nan(values: np.ndarray) -> np.ndarray:
    """Rank with average tie-breaking, treating NaNs as missing."""
    n = values.shape[0]
    filled = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = values[i]
        filled[i] = np.inf if np.isnan(v) else v

    ranks = np.empty(n, dtype=np.float64)
    order = np.argsort(filled)

    p = 0
    while p < n:
        v = filled[order[p]]
        if v == np.inf:
            for t in range(p, n):
                ranks[order[t]] = np.nan
            break

        q = p
        while q + 1 < n and filled[order[q + 1]] == v:
            q += 1

        avg_rank = (p + q + 2) / 2.0
        for t in range(p, q + 1):
            ranks[order[t]] = avg_rank
        p = q + 1

    return ranks


@njit(cache=True, nogil=True, parallel=True)
def spearman_by_dt(
    factor: np.ndarray,
    label: np.ndarray,
    dt_offsets_rows: np.ndarray,
) -> np.ndarray:
    """Spearman IC for each datetime group.

    Notes
    -----
    This matches the legacy + existing semantics: each variable is
    ranked *independently* over its non-NaN values, then Pearson correlation is
    computed over the intersection of non-NaN ranks.
    """
    n_dt = dt_offsets_rows.shape[0] - 1
    out = np.empty(n_dt, dtype=np.float64)
    out[:] = np.nan

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])
        n = end - start
        if n <= 1:
            continue

        rf = _rank_average_nan(factor[start:end])
        rl = _rank_average_nan(label[start:end])

        sum_f = 0.0
        sum_l = 0.0
        m = 0
        for i in range(n):
            a = rf[i]
            b = rl[i]
            if not np.isnan(a) and not np.isnan(b):
                sum_f += a
                sum_l += b
                m += 1

        if m <= 1:
            continue

        mean_f = sum_f / m
        mean_l = sum_l / m

        cov = 0.0
        var_f = 0.0
        var_l = 0.0
        for i in range(n):
            a = rf[i]
            b = rl[i]
            if not np.isnan(a) and not np.isnan(b):
                da = a - mean_f
                db = b - mean_l
                cov += da * db
                var_f += da * da
                var_l += db * db

        denom = np.sqrt(var_f) * np.sqrt(var_l)
        out[d] = cov / denom if denom > 0.0 else np.nan

    return out


@njit(cache=True, nogil=True, parallel=True)
def corr_by_dt(
    factor: np.ndarray,
    label: np.ndarray,
    dt_offsets_rows: np.ndarray,
) -> np.ndarray:
    """Pearson IC for each datetime group (NaN-safe for factor)."""
    n_dt = dt_offsets_rows.shape[0] - 1
    out = np.empty(n_dt, dtype=np.float64)
    out[:] = np.nan

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])
        if end - start <= 1:
            continue

        sum_a = 0.0
        sum_b = 0.0
        n = 0
        for i in range(start, end):
            a = factor[i]
            if not np.isnan(a) and not np.isnan(label[i]):
                sum_a += a
                sum_b += label[i]
                n += 1

        if n <= 1:
            continue

        mean_a = sum_a / n
        mean_b = sum_b / n

        cov = 0.0
        var_a = 0.0
        var_b = 0.0
        for i in range(start, end):
            a = factor[i]
            if not np.isnan(a) and not np.isnan(label[i]):
                da = a - mean_a
                db = label[i] - mean_b
                cov += da * db
                var_a += da * da
                var_b += db * db

        denom = np.sqrt(var_a) * np.sqrt(var_b)
        out[d] = cov / denom if denom > 0.0 else np.nan

    return out


@njit(cache=True, nogil=True, parallel=True)
def spearman_by_dt_layer(
    factor: np.ndarray,
    label: np.ndarray,
    layer_code: np.ndarray,
    dt_offsets_rows: np.ndarray,
    n_layers: int,
) -> np.ndarray:
    """Spearman IC for each (dt, layer) group.

    Returns
    -------
    float64 array of shape (n_dt, n_layers), filled with NaNs where undefined.
    """
    n_dt = dt_offsets_rows.shape[0] - 1
    out = np.empty((n_dt, n_layers), dtype=np.float64)
    out[:] = np.nan

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])
        if end - start <= 1:
            continue

        counts = np.zeros(n_layers, dtype=np.int32)
        for i in range(start, end):
            lc = int(layer_code[i])
            if 1 <= lc <= n_layers:
                counts[lc - 1] += 1

        for l in range(n_layers):
            m = int(counts[l])
            if m <= 1:
                continue

            a = np.empty(m, dtype=np.float64)
            b = np.empty(m, dtype=np.float64)
            idx = 0
            target = l + 1
            for i in range(start, end):
                if int(layer_code[i]) == target:
                    a[idx] = factor[i]
                    b[idx] = label[i]
                    idx += 1

            out[d, l] = _spearman_1d(a, b)

    return out
