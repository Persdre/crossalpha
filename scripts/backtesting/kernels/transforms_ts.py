"""Rolling (time-series) transforms for TS backtests."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _rolling_mean_std(
    values: np.ndarray,
    window: int,
    *,
    min_count_mean: int,
    min_count_std: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean/std on finite values (trailing window).

    Non-finite values are treated as missing (excluded from statistics).
    Mean/std entries that do not meet the `min_count_*` requirement are NaN.

    Std is sample std (ddof=1), matching Polars.
    """
    n = values.shape[0]
    mean = np.empty(n, dtype=np.float64)
    std = np.empty(n, dtype=np.float64)

    if window <= 0:
        mean[:] = np.nan
        std[:] = np.nan
        return mean, std

    sum_x = 0.0
    sum_x2 = 0.0
    count = 0

    for i in range(n):
        x = values[i]
        if np.isfinite(x):
            sum_x += x
            sum_x2 += x * x
            count += 1

        j = i - window
        if j >= 0:
            old = values[j]
            if np.isfinite(old):
                sum_x -= old
                sum_x2 -= old * old
                count -= 1

        if count >= min_count_mean and count > 0:
            m = sum_x / count
            mean[i] = m
        else:
            mean[i] = np.nan

        if count >= min_count_std and count > 1:
            m = sum_x / count
            var = (sum_x2 - sum_x * m) / (count - 1)
            if var > 0.0:
                std[i] = np.sqrt(var)
            else:
                std[i] = 0.0
        else:
            std[i] = np.nan

    return mean, std


@njit(cache=True, nogil=True)
def rolling_winsorize_std(values: np.ndarray, window: int, n: float) -> np.ndarray:
    """Rolling winsorization via mean ± n*std (std requires >=2 samples)."""
    out = values.copy()
    mean, std = _rolling_mean_std(values, window, min_count_mean=1, min_count_std=2)

    for i in range(values.shape[0]):
        x = values[i]
        if not np.isfinite(x):
            continue
        s = std[i]
        if not np.isfinite(s) or s <= 0.0:
            continue
        lower = mean[i] - n * s
        upper = mean[i] + n * s
        if x < lower:
            out[i] = lower
        elif x > upper:
            out[i] = upper

    return out


@njit(cache=True, nogil=True)
def _median_sorted(sorted_vals: np.ndarray) -> float:
    m = sorted_vals.shape[0]
    if m == 0:
        return np.nan
    mid = m // 2
    if m % 2 == 1:
        return float(sorted_vals[mid])
    return 0.5 * (float(sorted_vals[mid - 1]) + float(sorted_vals[mid]))


@njit(cache=True, nogil=True)
def rolling_winsorize_mad(values: np.ndarray, window: int, n: float) -> np.ndarray:
    """Rolling winsorization via median ± n*(std*1.4826).

    This matches the legacy Polars implementation for the rolling "mad" mode.
    """
    out = values.copy()
    _, std = _rolling_mean_std(values, window, min_count_mean=1, min_count_std=2)
    scale = 1.4826

    n_obs = values.shape[0]
    for i in range(n_obs):
        x = values[i]
        if not np.isfinite(x):
            continue

        start = 0 if i + 1 <= window else i + 1 - window
        tmp = np.empty(i - start + 1, dtype=np.float64)
        m = 0
        for j in range(start, i + 1):
            v = values[j]
            if np.isfinite(v):
                tmp[m] = v
                m += 1
        if m == 0:
            continue

        tmp = np.sort(tmp[:m])
        center = _median_sorted(tmp)

        s = std[i]
        if not np.isfinite(s) or s <= 0.0:
            continue
        spread = s * scale
        lower = center - n * spread
        upper = center + n * spread

        if x < lower:
            out[i] = lower
        elif x > upper:
            out[i] = upper

    return out


@njit(cache=True, nogil=True)
def _quantile_nearest(sorted_vals: np.ndarray, q: float) -> float:
    m = sorted_vals.shape[0]
    if m == 0:
        return np.nan
    if m == 1:
        return float(sorted_vals[0])
    if q <= 0.0:
        return float(sorted_vals[0])
    if q >= 1.0:
        return float(sorted_vals[m - 1])
    idx = int(q * (m - 1) + 0.5)
    if idx < 0:
        idx = 0
    elif idx >= m:
        idx = m - 1
    return float(sorted_vals[idx])


@njit(cache=True, nogil=True)
def rolling_winsorize_quantile(values: np.ndarray, window: int, q: float) -> np.ndarray:
    """Rolling winsorization via [q, 1-q] quantiles (nearest interpolation)."""
    out = values.copy()
    n_obs = values.shape[0]

    for i in range(n_obs):
        x = values[i]
        if not np.isfinite(x):
            continue

        start = 0 if i + 1 <= window else i + 1 - window
        tmp = np.empty(i - start + 1, dtype=np.float64)
        m = 0
        for j in range(start, i + 1):
            v = values[j]
            if np.isfinite(v):
                tmp[m] = v
                m += 1
        if m == 0:
            continue

        tmp = np.sort(tmp[:m])
        lower = _quantile_nearest(tmp, q)
        upper = _quantile_nearest(tmp, 1.0 - q)

        if x < lower:
            out[i] = lower
        elif x > upper:
            out[i] = upper

    return out


@njit(cache=True, nogil=True)
def rolling_normalize_zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score normalization (mean>=1 sample, std>=2 samples)."""
    out = values.copy()
    mean, std = _rolling_mean_std(values, window, min_count_mean=1, min_count_std=2)

    for i in range(values.shape[0]):
        x = values[i]
        if not np.isfinite(x):
            continue
        s = std[i]
        out[i] = (x - mean[i]) / s if np.isfinite(s) and s > 0.0 else 0.0

    return out


@njit(cache=True, nogil=True)
def rolling_normalize_rank(values: np.ndarray, window: int) -> np.ndarray:
    """Legacy rolling "rank": tanh of z-score with full-window stats."""
    out = values.copy()
    mean, std = _rolling_mean_std(values, window, min_count_mean=window, min_count_std=window)

    for i in range(values.shape[0]):
        x = values[i]
        if not np.isfinite(x):
            continue
        s = std[i]
        if np.isfinite(s) and s > 0.0:
            out[i] = np.tanh((x - mean[i]) / s)
        else:
            out[i] = 0.0

    return out


@njit(cache=True, nogil=True)
def rolling_normalize_demean(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling demean with full-window mean (default Polars min_samples)."""
    out = values.copy()
    mean, _ = _rolling_mean_std(values, window, min_count_mean=window, min_count_std=2)

    for i in range(values.shape[0]):
        x = values[i]
        if not np.isfinite(x):
            continue
        m = mean[i]
        out[i] = x - m if np.isfinite(m) else np.nan

    return out


@njit(cache=True, nogil=True)
def rolling_normalize_minmax(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling min-max normalization to [-1, 1] (min/max require >=1 sample)."""
    out = values.copy()
    n_obs = values.shape[0]

    for i in range(n_obs):
        x = values[i]
        if not np.isfinite(x):
            continue

        start = 0 if i + 1 <= window else i + 1 - window
        vmin = np.inf
        vmax = -np.inf
        cnt = 0
        for j in range(start, i + 1):
            v = values[j]
            if np.isfinite(v):
                cnt += 1
                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v

        if cnt == 0:
            out[i] = np.nan
            continue

        span = vmax - vmin
        out[i] = 2.0 * (x - vmin) / span - 1.0 if span > 0.0 else 0.0

    return out

