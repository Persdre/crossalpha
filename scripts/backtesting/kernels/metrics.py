"""Numba-friendly metric helpers (matches legacy definitions)."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def nansum(x: np.ndarray) -> float:
    s = 0.0
    n = x.shape[0]
    for i in range(n):
        v = x[i]
        if not np.isnan(v):
            s += v
    return s


@njit(cache=True, nogil=True)
def nanmean(x: np.ndarray) -> float:
    s = 0.0
    n = 0
    size = x.shape[0]
    for i in range(size):
        v = x[i]
        if not np.isnan(v):
            s += v
            n += 1
    return s / n if n > 0 else 0.0


@njit(cache=True, nogil=True, inline="always")
def _nanstd_from_mean(x: np.ndarray, mean: float) -> float:
    s2 = 0.0
    n = 0
    size = x.shape[0]
    for i in range(size):
        v = x[i]
        if not np.isnan(v):
            dv = v - mean
            s2 += dv * dv
            n += 1
    return np.sqrt(s2 / n) if n > 0 else 0.0


@njit(cache=True, nogil=True)
def nanstd(x: np.ndarray) -> float:
    """Population std (ddof=0) ignoring NaNs, matching np.nanstd default."""
    m = nanmean(x)
    return _nanstd_from_mean(x, m)


@njit(cache=True, nogil=True)
def annual_ret(daily_ret: np.ndarray, num_years: float) -> float:
    if daily_ret.shape[0] == 0 or num_years <= 0.0:
        return 0.0
    return nansum(daily_ret) / num_years


@njit(cache=True, nogil=True)
def sharpe(daily_ret: np.ndarray, dates_to_years: float) -> float:
    if daily_ret.shape[0] == 0:
        return 0.0
    m = nanmean(daily_ret)
    s = _nanstd_from_mean(daily_ret, m)
    return m / s * np.sqrt(dates_to_years) if s > 0.0 else 0.0


@njit(cache=True, nogil=True)
def sortino(daily_ret: np.ndarray, dates_to_years: float) -> float:
    if daily_ret.shape[0] == 0:
        return 0.0
    total_sum = 0.0
    total_n = 0
    neg_sum = 0.0
    n = 0

    size = daily_ret.shape[0]
    for i in range(size):
        v = daily_ret[i]
        if not np.isnan(v):
            total_sum += v
            total_n += 1
            if v < 0.0:
                neg_sum += v
                n += 1

    m = total_sum / total_n if total_n > 0 else 0.0

    if n == 0:
        downside_std = 0.0
    else:
        neg_mean = neg_sum / n
        s2 = 0.0
        for i in range(size):
            v = daily_ret[i]
            if not np.isnan(v) and v < 0.0:
                dv = v - neg_mean
                s2 += dv * dv
        downside_std = np.sqrt(s2 / n)
    return m / downside_std * np.sqrt(dates_to_years) if downside_std > 0.0 else 0.0


@njit(cache=True, nogil=True)
def max_dd(daily_ret: np.ndarray) -> float:
    if daily_ret.shape[0] == 0:
        return 0.0

    running_max = 0.0
    cumulative = 0.0
    max_drawdown = 0.0

    for i in range(daily_ret.shape[0]):
        v = daily_ret[i]
        if np.isnan(v):
            v = 0.0
        cumulative += v
        if i == 0 or cumulative > running_max:
            running_max = cumulative
        dd = (cumulative - running_max) / (running_max + 1.0)
        if dd < max_drawdown:
            max_drawdown = dd

    return max(max_drawdown, -1.0)


@njit(cache=True, nogil=True)
def calmar(annual_ret_value: float, max_dd_value: float) -> float:
    return annual_ret_value / abs(max_dd_value) if max_dd_value != 0.0 else 0.0


@njit(cache=True, nogil=True)
def sharpe_per_turnover(sharpe_value: float, turnover_value: float) -> float:
    return sharpe_value / turnover_value if turnover_value > 0.0 else 0.0
