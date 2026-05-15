"""Cross-sectional factor transforms (winsorize/normalize) on long arrays."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True)
def _median_sorted(sorted_vals: np.ndarray) -> float:
    n = sorted_vals.shape[0]
    if n == 0:
        return np.nan
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_vals[mid])
    return 0.5 * (float(sorted_vals[mid - 1]) + float(sorted_vals[mid]))


@njit(cache=True, nogil=True)
def _quantile_nearest(sorted_vals: np.ndarray, q: float) -> float:
    n = sorted_vals.shape[0]
    if n == 0:
        return np.nan
    if n == 1:
        return float(sorted_vals[0])
    if q <= 0.0:
        return float(sorted_vals[0])
    if q >= 1.0:
        return float(sorted_vals[n - 1])
    idx = int(q * (n - 1) + 0.5)
    if idx < 0:
        idx = 0
    elif idx >= n:
        idx = n - 1
    return float(sorted_vals[idx])


@njit(cache=True, nogil=True, parallel=True)
def winsorize_cs_std(values: np.ndarray, dt_offsets_rows: np.ndarray, n: float) -> np.ndarray:
    out = values.copy()
    n_dt = dt_offsets_rows.shape[0] - 1

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])

        s = 0.0
        cnt = 0
        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                s += v
                cnt += 1
        if cnt == 0:
            continue

        mean = s / cnt
        if cnt <= 1:
            continue

        ss = 0.0
        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                dv = v - mean
                ss += dv * dv
        var = ss / (cnt - 1)
        if var <= 0.0:
            continue

        std = np.sqrt(var)
        lower = mean - n * std
        upper = mean + n * std

        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                if v < lower:
                    out[i] = lower
                elif v > upper:
                    out[i] = upper

    return out


@njit(cache=True, nogil=True, parallel=True)
def winsorize_cs_mad(values: np.ndarray, dt_offsets_rows: np.ndarray, n: float) -> np.ndarray:
    out = values.copy()
    n_dt = dt_offsets_rows.shape[0] - 1

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])
        group_size = end - start

        tmp = np.empty(group_size, dtype=np.float64)
        m = 0
        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                tmp[m] = v
                m += 1
        if m == 0:
            continue

        tmp = np.sort(tmp[:m])
        center = _median_sorted(tmp)

        dev = np.empty(m, dtype=np.float64)
        for j in range(m):
            dv = tmp[j] - center
            dev[j] = dv if dv >= 0.0 else -dv
        dev = np.sort(dev)
        spread = _median_sorted(dev)
        if not np.isfinite(spread) or spread <= 0.0:
            continue

        lower = center - n * spread
        upper = center + n * spread
        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                if v < lower:
                    out[i] = lower
                elif v > upper:
                    out[i] = upper

    return out


@njit(cache=True, nogil=True, parallel=True)
def winsorize_cs_quantile(
    values: np.ndarray, dt_offsets_rows: np.ndarray, q: float
) -> np.ndarray:
    out = values.copy()
    n_dt = dt_offsets_rows.shape[0] - 1

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])
        group_size = end - start

        tmp = np.empty(group_size, dtype=np.float64)
        m = 0
        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                tmp[m] = v
                m += 1
        if m == 0:
            continue

        tmp = np.sort(tmp[:m])
        lower = _quantile_nearest(tmp, q)
        upper = _quantile_nearest(tmp, 1.0 - q)

        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                if v < lower:
                    out[i] = lower
                elif v > upper:
                    out[i] = upper

    return out


@njit(cache=True, nogil=True, parallel=True)
def normalize_cs_zscore(values: np.ndarray, dt_offsets_rows: np.ndarray) -> np.ndarray:
    out = values.copy()
    n_dt = dt_offsets_rows.shape[0] - 1

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])

        s = 0.0
        cnt = 0
        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                s += v
                cnt += 1
        if cnt == 0:
            continue

        mean = s / cnt
        if cnt <= 1:
            for i in range(start, end):
                if np.isfinite(values[i]):
                    out[i] = 0.0
            continue

        ss = 0.0
        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                dv = v - mean
                ss += dv * dv
        var = ss / (cnt - 1)
        if var <= 0.0:
            for i in range(start, end):
                if np.isfinite(values[i]):
                    out[i] = 0.0
            continue

        inv_std = 1.0 / np.sqrt(var)
        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                out[i] = (v - mean) * inv_std

    return out


@njit(cache=True, nogil=True, parallel=True)
def normalize_cs_minmax(values: np.ndarray, dt_offsets_rows: np.ndarray) -> np.ndarray:
    out = values.copy()
    n_dt = dt_offsets_rows.shape[0] - 1

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])

        vmin = np.inf
        vmax = -np.inf
        cnt = 0
        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                cnt += 1
                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v
        if cnt == 0:
            continue

        span = vmax - vmin
        if span <= 0.0 or not np.isfinite(span):
            for i in range(start, end):
                if np.isfinite(values[i]):
                    out[i] = 0.0
            continue

        inv_span = 1.0 / span
        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                out[i] = 2.0 * (v - vmin) * inv_span - 1.0

    return out


@njit(cache=True, nogil=True, parallel=True)
def normalize_cs_demean(values: np.ndarray, dt_offsets_rows: np.ndarray) -> np.ndarray:
    out = values.copy()
    n_dt = dt_offsets_rows.shape[0] - 1

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])

        s = 0.0
        cnt = 0
        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                s += v
                cnt += 1
        if cnt == 0:
            continue

        mean = s / cnt
        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                out[i] = v - mean

    return out


@njit(cache=True, nogil=True, parallel=True)
def normalize_cs_rank(values: np.ndarray, dt_offsets_rows: np.ndarray) -> np.ndarray:
    out = values.copy()
    n_dt = dt_offsets_rows.shape[0] - 1

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])
        group_size = end - start

        vals = np.empty(group_size, dtype=np.float64)
        rows = np.empty(group_size, dtype=np.int64)
        m = 0
        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                vals[m] = v
                rows[m] = i
                m += 1
        if m == 0:
            continue
        if m == 1:
            out[int(rows[0])] = 0.0
            continue

        order = np.argsort(vals[:m])
        p = 0
        while p < m:
            v = vals[order[p]]
            q = p
            while q + 1 < m and vals[order[q + 1]] == v:
                q += 1

            avg_rank = (p + q + 2) / 2.0
            normed = 2.0 * (avg_rank - 1.0) / (m - 1.0) - 1.0
            for t in range(p, q + 1):
                out[int(rows[order[t]])] = normed
            p = q + 1

    return out


@njit(cache=True, nogil=True, parallel=True)
def winsorize_std_then_zscore(
    values: np.ndarray, dt_offsets_rows: np.ndarray, n: float
) -> np.ndarray:
    """Fused CS winsorize(std) + normalize(zscore).

    This avoids allocating and scanning an intermediate winsorized array while
    matching the legacy two-step semantics:
    1) clip at mean ± n*std (std is sample, requires >=2 samples)
    2) z-score on the winsorized values (sample std, requires >=2 samples; else 0)
    """
    out = values.copy()
    n_dt = dt_offsets_rows.shape[0] - 1

    for d in prange(n_dt):
        start = int(dt_offsets_rows[d])
        end = int(dt_offsets_rows[d + 1])

        s = 0.0
        cnt = 0
        for i in range(start, end):
            v = values[i]
            if np.isfinite(v):
                s += v
                cnt += 1
        if cnt == 0:
            continue

        mean = s / cnt
        std = np.nan
        if cnt > 1:
            ss = 0.0
            for i in range(start, end):
                v = values[i]
                if np.isfinite(v):
                    dv = v - mean
                    ss += dv * dv
            var = ss / (cnt - 1)
            if var > 0.0:
                std = np.sqrt(var)

        sum_w = 0.0
        sum_w2 = 0.0
        if np.isfinite(std) and std > 0.0:
            lower = mean - n * std
            upper = mean + n * std
            for i in range(start, end):
                v = values[i]
                if np.isfinite(v):
                    w = v
                    if w < lower:
                        w = lower
                    elif w > upper:
                        w = upper
                    out[i] = w
                    sum_w += w
                    sum_w2 += w * w
        else:
            for i in range(start, end):
                w = values[i]
                if np.isfinite(w):
                    sum_w += w
                    sum_w2 += w * w

        if cnt <= 1:
            for i in range(start, end):
                if np.isfinite(values[i]):
                    out[i] = 0.0
            continue

        mean_w = sum_w / cnt
        var_w = (sum_w2 - sum_w * mean_w) / (cnt - 1)
        if not np.isfinite(var_w) or var_w <= 0.0:
            for i in range(start, end):
                if np.isfinite(values[i]):
                    out[i] = 0.0
            continue

        inv_std_w = 1.0 / np.sqrt(var_w)
        for i in range(start, end):
            if np.isfinite(values[i]):
                out[i] = (out[i] - mean_w) * inv_std_w

    return out
