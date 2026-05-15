"""(Numba-first) drop-in backtesting framework.

This package provides a drop-in compatible API with the legacy implementation
in `scripts.backtesting`, while moving heavy computations into Numba kernels.
"""

from .backtest_utils import run_cross_section_backtest, run_time_series_backtest

__all__ = ["run_cross_section_backtest", "run_time_series_backtest"]

