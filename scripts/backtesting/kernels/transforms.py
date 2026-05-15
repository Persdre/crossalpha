"""Array-first winsorize/normalize/squash helpers.

These thin wrappers are kept for internal convenience. Engines should prefer
calling the specialized modules:
- `kernels/transforms_cs.py` for cross-sectional (grouped by datetime) transforms
- `kernels/transforms_ts.py` for rolling (time-series) transforms
"""

from __future__ import annotations

import numpy as np

from .transforms_cs import (
    normalize_cs_demean,
    normalize_cs_minmax,
    normalize_cs_rank,
    normalize_cs_zscore,
    winsorize_cs_mad,
    winsorize_cs_quantile,
    winsorize_cs_std,
)
from .transforms_ts import (
    rolling_normalize_demean,
    rolling_normalize_minmax,
    rolling_normalize_rank,
    rolling_normalize_zscore,
    rolling_winsorize_mad,
    rolling_winsorize_quantile,
    rolling_winsorize_std,
)


def winsorize_cs(
    values: np.ndarray, dt_offsets_rows: np.ndarray, *, method: str = "std", n: float = 3.0
) -> np.ndarray:
    if method == "std":
        return winsorize_cs_std(values, dt_offsets_rows, float(n))
    if method == "mad":
        return winsorize_cs_mad(values, dt_offsets_rows, float(n))
    if method == "quantile":
        return winsorize_cs_quantile(values, dt_offsets_rows, float(n))
    raise ValueError(f"Unknown winsorize method: {method}")


def normalize_cs(
    values: np.ndarray, dt_offsets_rows: np.ndarray, *, method: str = "zscore"
) -> np.ndarray:
    if method == "zscore":
        return normalize_cs_zscore(values, dt_offsets_rows)
    if method == "minmax":
        return normalize_cs_minmax(values, dt_offsets_rows)
    if method == "rank":
        return normalize_cs_rank(values, dt_offsets_rows)
    if method == "demean":
        return normalize_cs_demean(values, dt_offsets_rows)
    raise ValueError(f"Unknown normalization method: {method}")


def winsorize_rolling(
    values: np.ndarray, *, method: str = "std", n: float = 3.0, window: int = 252
) -> np.ndarray:
    if method == "std":
        return rolling_winsorize_std(values, int(window), float(n))
    if method == "mad":
        return rolling_winsorize_mad(values, int(window), float(n))
    if method == "quantile":
        return rolling_winsorize_quantile(values, int(window), float(n))
    raise ValueError(f"Unknown winsorize method: {method}")


def normalize_rolling(
    values: np.ndarray, *, method: str = "zscore", window: int = 252
) -> np.ndarray:
    if method == "zscore":
        return rolling_normalize_zscore(values, int(window))
    if method == "minmax":
        return rolling_normalize_minmax(values, int(window))
    if method == "rank":
        return rolling_normalize_rank(values, int(window))
    if method == "demean":
        return rolling_normalize_demean(values, int(window))
    raise ValueError(f"Unknown normalization method: {method}")


def squash(values: np.ndarray, *, method: str = "tanh") -> np.ndarray:
    if method == "tanh":
        return np.tanh(values)
    if method == "clip":
        return np.clip(values, -1.0, 1.0)
    raise ValueError(f"Unknown squash method: {method}")

