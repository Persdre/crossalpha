"""Array containers for the backtesting engines.

The engines are array-first: all heavy compute happens on NumPy arrays
in Numba kernels, and only the final outputs are converted back to pandas
objects and Python lists to match the legacy API contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .adapters.input_builders import CSRIndex


@dataclass(frozen=True, slots=True)
class CSPanelArrays:
    """Cross-sectional (panel) arrays in long format."""

    dt_list: List  # list[datetime.datetime]
    dt_offsets_rows: np.ndarray  # int64, shape (n_dt + 1,)
    dt_trading_dates: List  # list[datetime.date], shape (n_dt,)

    day_list: List  # list[datetime.date]
    day_offsets_dt: np.ndarray  # int64, shape (n_days + 1,)

    sym_list: List[str]  # sorted unique symbols
    sym_csr: CSRIndex

    sym_code: np.ndarray  # int32, shape (n_rows,)
    factor: np.ndarray  # float64, shape (n_rows,) (lagged)
    label: np.ndarray  # float64, shape (n_rows,)
    ref_price: np.ndarray  # float64, shape (n_rows,)


@dataclass(frozen=True, slots=True)
class TSPanelArrays:
    """Time-series arrays after filtering and lag alignment."""

    dt_list: List  # list[datetime.datetime]
    trading_dates: List  # list[datetime.date], shape (n_dt,)

    day_list: List  # list[datetime.date]
    day_offsets_dt: np.ndarray  # int64, shape (n_days + 1,)

    factor: np.ndarray  # float64, shape (n_dt,) (lagged)
    label: np.ndarray  # float64, shape (n_dt,)
    ref_price: np.ndarray  # float64, shape (n_dt,)

