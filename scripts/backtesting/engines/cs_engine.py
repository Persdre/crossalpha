"""Cross-sectional backtest engine (array-first, Numba-accelerated).

This engine matches the legacy output contract of
`scripts.backtesting.backtest.CS_Backtest` while minimizing Python/Polars
overhead by running compute-heavy stages on NumPy arrays in Numba kernels.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..adapters.array_builders import build_cs_panel_arrays
from ..core import CoreName, detect_core
from ..kernels.aggregation import sum_by_group_4
from ..kernels.correlation import corr
from ..kernels.ic_cs import corr_by_dt, spearman_by_dt, spearman_by_dt_layer
from ..kernels.layers import assign_layers_ordinal_desc, count_layers_by_dt
from ..kernels.metrics import (
    annual_ret,
    calmar,
    max_dd,
    sharpe,
    sharpe_per_turnover,
    sortino,
)
from ..kernels.panel_build import scatter_long_to_dense_1
from ..kernels.pnl import (
    sum_tpnl_by_dt_and_layer,
    tpnl_dt_components_cs_4_multi_fee,
    tpnl_obs_layer_membership_cs,
)
from ..kernels.positions_cs import positions_cs_from_raw
from ..kernels.transforms_cs import (
    normalize_cs_demean,
    normalize_cs_minmax,
    normalize_cs_rank,
    normalize_cs_zscore,
    winsorize_cs_mad,
    winsorize_cs_quantile,
    winsorize_cs_std,
    winsorize_std_then_zscore,
)
from ..kernels.turnover import turnover_obs_cs_4
from ..kernels.time_metrics import compute_time_metrics
from ..structures import CSPanelArrays


class CS_Backtest:
    """Drop-in equivalent of `scripts.backtesting.backtest.CS_Backtest`.

    The public behavior (inputs/outputs) must match the legacy implementation,
    while compute-heavy stages are delegated to Numba kernels.
    """

    def __init__(self, config: dict):
        self.config = config
        self.factor = config["factor"]
        self.label = config["label"]
        self.core: CoreName = detect_core(self.factor)
        if detect_core(self.label) != self.core:
            raise TypeError("factor/label backends must match")
        self.layers: int = int(config["layers"])
        self.fees: List[float] = list(config.get("fees", [0.0]))
        self.backtest_mode: str = config.get("backtest_mode", "long/short_layers")
        self.backtest_params: dict = config.get("backtest_params", {})
        self.lag: int = int(config.get("lag", 1))
        self.annual_days = config.get("annual_days")
        self.weight = config.get("weight")  # optional pivoted weight DataFrame

        self.panel: CSPanelArrays | None = None
        self._weight_long: np.ndarray | None = None  # compacted weight array

        self.time_metrics: Dict[str, Any] = {}
        self.fee_independent_metrics: Dict[str, Any] = {}

        self.rank_ic: Any = None
        self.rank_ic_daily: Any = None
        self.ic: Any = None
        self.ic_daily: Any = None
        self.acf: Dict[int, float] | None = None

        self.turnover: Dict[str, Any] | None = None
        self.turnover_daily: Dict[str, Any] | None = None

        self.ret: Dict[str, Dict[str, Any]] | None = None
        self.ret_daily: Dict[str, Dict[str, Any]] | None = None

        self.ret_layer: Dict[str, Any] | None = None
        self.ret_layer_daily: Dict[str, Any] | None = None
        self.rank_ic_layer: Dict[str, Any] | None = None
        self.rank_ic_layer_daily: Dict[str, Any] | None = None

        self.summary_df: Any = None

        self._rank_ic_dt_values: np.ndarray | None = None
        self._rank_ic_daily_index: List | None = None
        self._rank_ic_daily_values: np.ndarray | None = None

        self._dt_list: List | None = None
        self._dt_offsets_rows: np.ndarray | None = None
        self._dt_trading_dates: List | None = None
        self._day_list: List | None = None
        self._day_offsets_dt: np.ndarray | None = None
        self._sym_list: List[str] | None = None
        self._sym_offsets: np.ndarray | None = None
        self._sym_rows: np.ndarray | None = None

        self._layer_code: np.ndarray | None = None
        self._pos_ls: np.ndarray | None = None
        self._pos_long: np.ndarray | None = None
        self._pos_short: np.ndarray | None = None
        self._pos_passive: np.ndarray | None = None

        self._factor_matrix: np.ndarray | None = None

        self._tpnl_dt_by_fee: Dict[str, Dict[str, np.ndarray]] = {}
        self._tpnl_daily_by_fee: Dict[str, Dict[str, np.ndarray]] = {}
        self._tpnl_excess_dt_by_fee: Dict[str, Dict[str, np.ndarray]] = {}
        self._tpnl_excess_daily_by_fee: Dict[str, Dict[str, np.ndarray]] = {}

    @staticmethod
    def _pl_timeseries(
        time_col: str,
        time_values: List,
        values: np.ndarray,
        *,
        value_col: str = "value",
    ) -> Any:
        try:
            import polars as pl  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("polars is required for polars outputs") from exc

        return pl.DataFrame(
            {
                time_col: time_values,
                value_col: np.asarray(values, dtype=np.float64),
            }
        )

    def run(self) -> Tuple[Dict[str, Any], Any]:
        self._prepare_data()
        self._assign_layers()
        self._calculate_positions()
        self._build_grouping_metadata()
        self._compute_time_metrics()

        self._add_rank_ic_metrics()
        self._add_acf_metrics()
        self._add_turnover_metrics()

        self._run_fee_scenarios()
        if self.core == "pandas":
            self._compute_ret_series()

        self.ret_layer, self.ret_layer_daily = self._calculate_layer_returns()
        (
            self.rank_ic_layer,
            self.rank_ic_layer_daily,
        ) = self._calculate_layer_rank_ic()

        return self._compile_results(), self.summary_df

    # ──────────────────────────────── Data Preparation ────────────────────────────────

    def _prepare_data(self) -> None:
        self.panel = build_cs_panel_arrays(self.factor, self.label, lag=self.lag)

        self._dt_list = self.panel.dt_list
        self._dt_offsets_rows = self.panel.dt_offsets_rows
        self._dt_trading_dates = self.panel.dt_trading_dates
        self._day_list = self.panel.day_list
        self._day_offsets_dt = self.panel.day_offsets_dt
        self._sym_list = self.panel.sym_list
        self._sym_offsets = self.panel.sym_csr.offsets
        self._sym_rows = self.panel.sym_csr.rows

        # Compact optional weight data into panel's long format
        if self.weight is not None:
            self._weight_long = self._compact_weight(self.weight)

    def _compact_weight(self, weight: Any) -> np.ndarray:
        """Compact a pivoted weight DataFrame into the panel's long format."""
        assert self.panel is not None
        core = detect_core(weight)
        if core == "pandas":
            # Align to panel's datetime/symbol ordering
            dt_index = pd.DatetimeIndex(self.panel.dt_list)
            sym_cols = self.panel.sym_list
            # Reindex to match panel, fill missing with NaN
            weight_aligned = weight.reindex(index=dt_index, columns=sym_cols)
            weight_mat = weight_aligned.to_numpy(dtype=np.float64, copy=False)
        else:
            import polars as pl
            # For polars weight DataFrames
            dt_strs = [str(d) for d in self.panel.dt_list]
            sym_cols = self.panel.sym_list
            cols_present = [c for c in sym_cols if c in weight.columns]
            cols_missing = [c for c in sym_cols if c not in weight.columns]
            if cols_present:
                w_sub = weight.select(["datetime", *cols_present])
            else:
                w_sub = weight.select(["datetime"])
            # Add missing columns as NaN
            for c in cols_missing:
                w_sub = w_sub.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
            w_sub = w_sub.select(["datetime", *sym_cols]).sort("datetime")
            weight_mat = w_sub.select(sym_cols).to_numpy()
            weight_mat = np.asarray(weight_mat, dtype=np.float64)

        # Extract values in compacted long format using panel's sym_code
        n_rows = self.panel.sym_code.shape[0]
        dt_sizes = np.diff(self.panel.dt_offsets_rows).astype(np.int64, copy=False)
        row_dt_code = np.repeat(np.arange(len(dt_sizes), dtype=np.int64), dt_sizes)
        weight_long = weight_mat[row_dt_code, self.panel.sym_code]
        # Replace NaN/negative with 0
        weight_long = np.where(np.isfinite(weight_long) & (weight_long > 0), weight_long, 0.0)
        return weight_long

    def _build_grouping_metadata(self) -> None:
        if self.panel is None:
            raise RuntimeError("panel not initialized")

    def _assign_layers(self) -> None:
        assert self.panel is not None
        if self.panel.factor.size == 0 or self.layers <= 0:
            self._layer_code = np.empty(self.panel.factor.size, dtype=np.int16)
            return

        assert self._dt_offsets_rows is not None
        self._layer_code = assign_layers_ordinal_desc(
            self.panel.factor,
            self._dt_offsets_rows,
            int(self.layers),
        )

    def _winsorize_cs(self, values: np.ndarray) -> np.ndarray:
        assert self._dt_offsets_rows is not None
        method = self.backtest_params.get("winsorize_method", "std")
        n = float(self.backtest_params.get("winsorize_n", 3.0))

        if method == "std":
            return winsorize_cs_std(values, self._dt_offsets_rows, n)
        if method == "mad":
            return winsorize_cs_mad(values, self._dt_offsets_rows, n)
        if method == "quantile":
            return winsorize_cs_quantile(values, self._dt_offsets_rows, n)

        raise ValueError(f"Unknown winsorize method: {method}")

    def _normalize_cs(self, values: np.ndarray) -> np.ndarray:
        assert self._dt_offsets_rows is not None
        method = self.backtest_params.get("normalization_method", "rank")

        if method == "zscore":
            return normalize_cs_zscore(values, self._dt_offsets_rows)
        if method == "minmax":
            return normalize_cs_minmax(values, self._dt_offsets_rows)
        if method == "rank":
            return normalize_cs_rank(values, self._dt_offsets_rows)
        if method == "demean":
            return normalize_cs_demean(values, self._dt_offsets_rows)

        raise ValueError(f"Unknown normalization method: {method}")

    def _calculate_positions(self) -> None:
        assert self.panel is not None
        assert self._dt_offsets_rows is not None
        assert self._layer_code is not None

        n_rows = int(self.panel.factor.shape[0])
        if n_rows == 0:
            self._pos_ls = np.empty(0, dtype=np.float64)
            self._pos_long = np.empty(0, dtype=np.float64)
            self._pos_short = np.empty(0, dtype=np.float64)
            self._pos_passive = np.empty(0, dtype=np.float64)
            return

        factor_valid = np.isfinite(self.panel.factor)

        if self.backtest_mode == "long/short_layers":
            long_indices = self.backtest_params.get("long_layer_index", [1])
            short_indices = self.backtest_params.get(
                "short_layer_index", [int(self.layers)]
            )

            raw = np.zeros(n_rows, dtype=np.float64)
            if self._weight_long is not None:
                # Use market cap (or other weight) as raw signal magnitude
                # Kernel normalization will produce weight-proportional positions
                w = np.where(self._weight_long > 0, self._weight_long, 1.0)
                for idx in long_indices:
                    mask = (self._layer_code == int(idx)) & factor_valid
                    raw[mask] = w[mask]
                for idx in short_indices:
                    mask = (self._layer_code == int(idx)) & factor_valid
                    raw[mask] = -w[mask]
            else:
                for idx in long_indices:
                    raw[(self._layer_code == int(idx)) & factor_valid] = 1.0
                for idx in short_indices:
                    raw[(self._layer_code == int(idx)) & factor_valid] = -1.0

        elif self.backtest_mode == "long/short_normalization":
            wins_method = self.backtest_params.get("winsorize_method", "std")
            norm_method = self.backtest_params.get("normalization_method", "rank")
            if wins_method == "std" and norm_method == "zscore":
                wins_n = float(self.backtest_params.get("winsorize_n", 3.0))
                raw = winsorize_std_then_zscore(
                    self.panel.factor, self._dt_offsets_rows, wins_n
                )
            else:
                factor_winsorized = self._winsorize_cs(self.panel.factor)
                raw = self._normalize_cs(factor_winsorized)

        else:
            raise ValueError(f"Unknown backtest_mode: {self.backtest_mode}")

        (
            self._pos_ls,
            self._pos_long,
            self._pos_short,
            self._pos_passive,
        ) = positions_cs_from_raw(raw, factor_valid, self._dt_offsets_rows)

    def _compute_time_metrics(self) -> None:
        assert self._dt_list is not None
        assert self._day_list is not None

        n_obs = int(self.panel.factor.shape[0]) if self.panel is not None else 0
        self.time_metrics = compute_time_metrics(
            self._dt_list,
            self._day_list,
            n_obs,
            annual_days=self.annual_days,
        )

    # ──────────────────────────────── Fee-independent Metrics ────────────────────────────────

    def _daily_mean_components_from_dt_values(
        self, values: np.ndarray
    ) -> tuple[List, np.ndarray]:
        assert self._day_list is not None
        assert self._day_offsets_dt is not None

        if values.size == 0:
            return [], np.empty(0, dtype=np.float64)

        starts = self._day_offsets_dt[:-1]
        valid = ~np.isnan(values)
        sums = np.add.reduceat(np.where(valid, values, 0.0), starts)
        counts = np.add.reduceat(valid.astype(np.int64), starts)
        keep = counts > 0
        if not np.any(keep):
            return [], np.empty(0, dtype=np.float64)

        out_vals = sums[keep] / counts[keep]
        out_dates = [self._day_list[i] for i in np.flatnonzero(keep)]
        return out_dates, out_vals

    def _daily_mean_from_dt_values(self, values: np.ndarray) -> pd.Series:
        out_dates, out_vals = self._daily_mean_components_from_dt_values(values)
        if len(out_dates) == 0:
            return pd.Series(dtype=float)
        return pd.Series(out_vals, index=out_dates)

    @staticmethod
    def _daily_mean_keep_all(values: np.ndarray, day_offsets_dt: np.ndarray) -> np.ndarray:
        n_days = int(day_offsets_dt.shape[0] - 1)
        out = np.full(n_days, np.nan, dtype=np.float64)
        if values.size == 0:
            return out

        starts = day_offsets_dt[:-1]
        valid = ~np.isnan(values)
        sums = np.add.reduceat(np.where(valid, values, 0.0), starts)
        counts = np.add.reduceat(valid.astype(np.int64), starts)
        mask = counts > 0
        out[mask] = sums[mask] / counts[mask]
        return out

    def _add_rank_ic_metrics(self) -> None:
        assert self.panel is not None
        assert self._dt_list is not None
        assert self._dt_offsets_rows is not None
        assert self._day_list is not None
        assert self._day_offsets_dt is not None

        n_symbols = len(self.panel.sym_list)
        if self.panel.factor.size == 0 or n_symbols == 0:
            if self.core == "pandas":
                self.rank_ic = pd.Series(dtype=float)
                self.ic = pd.Series(dtype=float)
                self.rank_ic_daily = pd.Series(dtype=float)
                self.ic_daily = pd.Series(dtype=float)
            else:
                self.rank_ic = self._pl_timeseries(
                    "datetime", [], np.empty(0, dtype=np.float64)
                )
                self.ic = self._pl_timeseries(
                    "datetime", [], np.empty(0, dtype=np.float64)
                )
                self.rank_ic_daily = self._pl_timeseries(
                    "date", [], np.empty(0, dtype=np.float64)
                )
                self.ic_daily = self._pl_timeseries(
                    "date", [], np.empty(0, dtype=np.float64)
                )
            self._rank_ic_dt_values = np.empty(0, dtype=np.float64)
            self._rank_ic_daily_index = []
            self._rank_ic_daily_values = np.empty(0, dtype=np.float64)
            self._factor_matrix = None
            return

        factor_mat = scatter_long_to_dense_1(
            self.panel.factor,
            self.panel.sym_code,
            self._dt_offsets_rows,
            int(n_symbols),
        )
        self._factor_matrix = factor_mat

        rank_ic_arr = spearman_by_dt(self.panel.factor, self.panel.label, self._dt_offsets_rows)
        ic_arr = corr_by_dt(self.panel.factor, self.panel.label, self._dt_offsets_rows)

        self._rank_ic_dt_values = rank_ic_arr
        if self.core == "pandas":
            self.rank_ic = pd.Series(rank_ic_arr, index=self._dt_list)
            self.ic = pd.Series(ic_arr, index=self._dt_list)
        else:
            self.rank_ic = self._pl_timeseries("datetime", self._dt_list, rank_ic_arr)
            self.ic = self._pl_timeseries("datetime", self._dt_list, ic_arr)

        valid_mask = np.isfinite(rank_ic_arr)
        rank_ic_valid = rank_ic_arr[valid_mask]
        if rank_ic_valid.size > 0:
            rank_ic_mean = float(np.mean(rank_ic_valid))
            rank_ic_std = float(np.std(rank_ic_valid, ddof=1))
            n = int(rank_ic_valid.size)
            rank_ic_z = (
                rank_ic_mean / (rank_ic_std / np.sqrt(n)) if rank_ic_std > 0 else 0.0
            )
            rank_ic_p_value = float(2 * (1 - norm.cdf(abs(rank_ic_z))))
            rank_ic_winratio = float(np.sum(rank_ic_valid > 0) / n)

            daily_dates, daily_vals = self._daily_mean_components_from_dt_values(
                rank_ic_arr
            )
            self._rank_ic_daily_index = daily_dates
            self._rank_ic_daily_values = daily_vals
            if self.core == "pandas":
                self.rank_ic_daily = (
                    pd.Series(daily_vals, index=daily_dates)
                    if daily_dates
                    else pd.Series(dtype=float)
                )
                daily_mean = float(self.rank_ic_daily.mean()) if daily_dates else 0.0
                daily_std = float(self.rank_ic_daily.std()) if daily_dates else float("nan")
            else:
                self.rank_ic_daily = self._pl_timeseries("date", daily_dates, daily_vals)
                daily_mean = float(np.mean(daily_vals)) if daily_vals.size else 0.0
                daily_std = float(np.std(daily_vals, ddof=1)) if daily_vals.size else float("nan")

            rank_icir = abs(daily_mean) / daily_std if daily_std > 0 else 0.0
            rank_icir_annual = rank_icir * np.sqrt(self.time_metrics["dates_to_years"])

            self.fee_independent_metrics.update(
                {
                    "rank_ic": rank_ic_mean,
                    "rank_icir": float(rank_icir),
                    "rank_icir_annual": float(rank_icir_annual),
                    "rank_ic_p_value": rank_ic_p_value,
                    "rank_ic_winratio": rank_ic_winratio,
                }
            )
        else:
            self.fee_independent_metrics.update(
                {
                    "rank_ic": 0.0,
                    "rank_icir": 0.0,
                    "rank_icir_annual": 0.0,
                    "rank_ic_p_value": 1.0,
                    "rank_ic_winratio": 0.0,
                }
            )
            self._rank_ic_daily_index = []
            self._rank_ic_daily_values = np.empty(0, dtype=np.float64)
            if self.core == "pandas":
                self.rank_ic_daily = pd.Series(dtype=float)
            else:
                self.rank_ic_daily = self._pl_timeseries(
                    "date", [], np.empty(0, dtype=np.float64)
                )

        ic_valid_mask = np.isfinite(ic_arr)
        ic_valid = ic_arr[ic_valid_mask]
        if ic_valid.size > 0:
            ic_mean = float(np.mean(ic_valid))
            ic_std = float(np.std(ic_valid, ddof=1))
            n = int(ic_valid.size)
            ic_z = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0
            ic_p_value = float(2 * (1 - norm.cdf(abs(ic_z))))
            ic_winratio = float(np.sum(ic_valid > 0) / n)

            ic_daily_dates, ic_daily_vals = self._daily_mean_components_from_dt_values(
                ic_arr
            )
            if self.core == "pandas":
                self.ic_daily = (
                    pd.Series(ic_daily_vals, index=ic_daily_dates)
                    if ic_daily_dates
                    else pd.Series(dtype=float)
                )
                daily_mean = float(self.ic_daily.mean()) if ic_daily_dates else 0.0
                daily_std = float(self.ic_daily.std()) if ic_daily_dates else float("nan")
            else:
                self.ic_daily = self._pl_timeseries(
                    "date", ic_daily_dates, ic_daily_vals
                )
                daily_mean = float(np.mean(ic_daily_vals)) if ic_daily_vals.size else 0.0
                daily_std = float(np.std(ic_daily_vals, ddof=1)) if ic_daily_vals.size else float("nan")

            icir = abs(daily_mean) / daily_std if daily_std > 0 else 0.0
            icir_annual = icir * np.sqrt(self.time_metrics["dates_to_years"])

            self.fee_independent_metrics.update(
                {
                    "ic": ic_mean,
                    "icir": float(icir),
                    "icir_annual": float(icir_annual),
                    "ic_p_value": ic_p_value,
                    "ic_winratio": ic_winratio,
                }
            )
        else:
            self.fee_independent_metrics.update(
                {
                    "ic": 0.0,
                    "icir": 0.0,
                    "icir_annual": 0.0,
                    "ic_p_value": 1.0,
                    "ic_winratio": 0.0,
                }
            )
            if self.core == "pandas":
                self.ic_daily = pd.Series(dtype=float)
            else:
                self.ic_daily = self._pl_timeseries(
                    "date", [], np.empty(0, dtype=np.float64)
                )

    def _add_acf_metrics(self) -> None:
        if self._factor_matrix is None:
            self.acf = {}
            self.fee_independent_metrics.update({"acf_1": 0, "acf_halflife": 0})
            return

        factor_matrix = self._factor_matrix
        n_dates = int(factor_matrix.shape[0])
        acf: Dict[int, float] = {}
        lags = list(range(1, 21)) + list(range(25, 51, 5)) + [100, 150, 200]

        for lag in lags:
            if n_dates <= lag:
                acf[lag] = np.nan
                continue
            A = factor_matrix[:-lag]
            B = factor_matrix[lag:]
            correlations = corr(A, B)
            acf[lag] = float(np.nanmean(correlations))

        acf_halflife = None
        for lag in sorted(acf.keys()):
            if acf[lag] < 0.5:
                acf_halflife = lag
                break

        self.fee_independent_metrics.update(
            {
                "acf_1": acf.get(1, 0),
                "acf_halflife": acf_halflife if acf_halflife else max(lags),
            }
        )
        self.acf = acf

    def _add_turnover_metrics(self) -> None:
        assert self.panel is not None
        assert self._dt_list is not None
        assert self._dt_offsets_rows is not None
        assert self._day_list is not None
        assert self._day_offsets_dt is not None
        assert self._sym_offsets is not None
        assert self._sym_rows is not None
        assert self._pos_ls is not None
        assert self._pos_long is not None
        assert self._pos_short is not None
        assert self._pos_passive is not None

        obs_ls, obs_long, obs_short, obs_passive = turnover_obs_cs_4(
            self._pos_ls,
            self._pos_long,
            self._pos_short,
            self._pos_passive,
            self._sym_offsets,
            self._sym_rows,
        )

        dt_ls, dt_long, dt_short, dt_passive = sum_by_group_4(
            obs_ls,
            obs_long,
            obs_short,
            obs_passive,
            self._dt_offsets_rows,
        )

        day_sum_ls, day_sum_long, day_sum_short, day_sum_passive = sum_by_group_4(
            dt_ls,
            dt_long,
            dt_short,
            dt_passive,
            self._day_offsets_dt,
        )
        day_counts = np.diff(self._day_offsets_dt).astype(np.float64)
        day_ls = day_sum_ls / day_counts
        day_long = day_sum_long / day_counts
        day_short = day_sum_short / day_counts
        day_passive = day_sum_passive / day_counts

        if self.core == "pandas":
            self.turnover = {
                "ls": pd.Series(dt_ls, index=self._dt_list),
                "long": pd.Series(dt_long, index=self._dt_list),
                "short": pd.Series(dt_short, index=self._dt_list),
                "passive": pd.Series(dt_passive, index=self._dt_list),
            }
            self.turnover_daily = {
                "ls": pd.Series(day_ls, index=self._day_list),
                "long": pd.Series(day_long, index=self._day_list),
                "short": pd.Series(day_short, index=self._day_list),
                "passive": pd.Series(day_passive, index=self._day_list),
            }
        else:
            self.turnover = {
                "ls": self._pl_timeseries("datetime", self._dt_list, dt_ls),
                "long": self._pl_timeseries("datetime", self._dt_list, dt_long),
                "short": self._pl_timeseries("datetime", self._dt_list, dt_short),
                "passive": self._pl_timeseries("datetime", self._dt_list, dt_passive),
            }
            self.turnover_daily = {
                "ls": self._pl_timeseries("date", self._day_list, day_ls),
                "long": self._pl_timeseries("date", self._day_list, day_long),
                "short": self._pl_timeseries("date", self._day_list, day_short),
                "passive": self._pl_timeseries("date", self._day_list, day_passive),
            }

        if self.turnover is not None and self.turnover_daily is not None:
            self.turnover["short_passive"] = self.turnover["passive"]
            self.turnover_daily["short_passive"] = self.turnover_daily["passive"]

        self.fee_independent_metrics.update(
            {
                "long_short_turnover_ratio": float(np.mean(dt_ls)) if len(dt_ls) else 0.0,
                "long_turnover_ratio": float(np.mean(dt_long)) if len(dt_long) else 0.0,
                "short_turnover_ratio": float(np.mean(dt_short)) if len(dt_short) else 0.0,
                "passive_turnover_ratio": float(np.mean(dt_passive))
                if len(dt_passive)
                else 0.0,
                "short_passive_turnover_ratio": float(np.mean(dt_passive))
                if len(dt_passive)
                else 0.0,
            }
        )

    # ──────────────────────────────── Fee Scenarios ────────────────────────────────

    def _run_fee_scenarios(self) -> None:
        assert self.panel is not None
        assert self._dt_list is not None
        assert self._dt_offsets_rows is not None
        assert self._day_list is not None
        assert self._day_offsets_dt is not None
        assert self._sym_offsets is not None
        assert self._sym_rows is not None
        assert self._pos_ls is not None
        assert self._pos_long is not None
        assert self._pos_short is not None
        assert self._pos_passive is not None

        num_years = float(self.time_metrics["num_years"])
        dates_to_years = float(self.time_metrics["dates_to_years"])

        fees_arr = np.asarray(self.fees, dtype=np.float64)
        dt_base, dt_abs_trade, wins_by_fee, active = tpnl_dt_components_cs_4_multi_fee(
            self._pos_ls,
            self._pos_long,
            self._pos_short,
            self._pos_passive,
            self.panel.ref_price,
            self._sym_offsets,
            self._sym_rows,
            self._dt_offsets_rows,
            fees_arr,
        )

        day_base_ls, day_base_long, day_base_short, day_base_passive = sum_by_group_4(
            dt_base[:, 0],
            dt_base[:, 1],
            dt_base[:, 2],
            dt_base[:, 3],
            self._day_offsets_dt,
        )
        day_abs_ls, day_abs_long, day_abs_short, day_abs_passive = sum_by_group_4(
            dt_abs_trade[:, 0],
            dt_abs_trade[:, 1],
            dt_abs_trade[:, 2],
            dt_abs_trade[:, 3],
            self._day_offsets_dt,
        )

        summary_dict: Dict[str, pd.Series] = {}
        summary_rows: List[Dict[str, Any]] = []
        for fee_idx, fee in enumerate(fees_arr):
            fee_bp = f"{round(float(fee) * 1e4, 1)} bp"

            dt_ls = dt_base[:, 0] - fee * dt_abs_trade[:, 0]
            dt_long = dt_base[:, 1] - fee * dt_abs_trade[:, 1]
            dt_short = dt_base[:, 2] - fee * dt_abs_trade[:, 2]
            dt_long_passive = dt_base[:, 3] - fee * dt_abs_trade[:, 3]
            dt_short_passive = -dt_base[:, 3] - fee * dt_abs_trade[:, 3]

            day_ls = day_base_ls - fee * day_abs_ls
            day_long = day_base_long - fee * day_abs_long
            day_short = day_base_short - fee * day_abs_short
            day_long_passive = day_base_passive - fee * day_abs_passive
            day_short_passive = -day_base_passive - fee * day_abs_passive

            self._tpnl_dt_by_fee[fee_bp] = {
                "ls": dt_ls,
                "long": dt_long,
                "short": dt_short,
                "passive": dt_long_passive,
                "short_passive": dt_short_passive,
            }
            self._tpnl_daily_by_fee[fee_bp] = {
                "ls": day_ls,
                "long": day_long,
                "short": day_short,
                "passive": day_long_passive,
                "short_passive": day_short_passive,
            }
            self._tpnl_excess_dt_by_fee[fee_bp] = {
                "long": dt_long - dt_long_passive,
                "short": dt_short - dt_short_passive,
            }
            self._tpnl_excess_daily_by_fee[fee_bp] = {
                "long": day_long - day_long_passive,
                "short": day_short - day_short_passive,
            }

            wins = wins_by_fee[fee_idx]

            metrics: Dict[str, Any] = {}
            portfolio_types = ["ls", "long", "short", "passive", "short_passive"]
            prefix_map = {
                "ls": "long_short",
                "long": "long",
                "short": "short",
                "passive": "passive",
                "short_passive": "short_passive",
            }

            for i_pt, pt in enumerate(portfolio_types):
                prefix = prefix_map[pt]

                pt_daily = self._tpnl_daily_by_fee[fee_bp][pt]
                annual = annual_ret(pt_daily, num_years)
                dd = max_dd(pt_daily)
                s = sharpe(pt_daily, dates_to_years)
                so = sortino(pt_daily, dates_to_years)
                c = calmar(annual, dd)
                turnover_ratio = float(
                    self.fee_independent_metrics.get(f"{prefix}_turnover_ratio", 0)
                )
                spt = sharpe_per_turnover(s, turnover_ratio)
                wr = float(wins[i_pt] / active[i_pt]) if active[i_pt] > 0 else 0.0

                metrics[f"{prefix}_ret_annual"] = float(annual)
                if pt == "long":
                    metrics[f"{prefix}_excess_ret_annual"] = float(
                        annual_ret(pt_daily - day_long_passive, num_years)
                    )
                elif pt == "short":
                    metrics[f"{prefix}_excess_ret_annual"] = float(
                        annual_ret(pt_daily - day_short_passive, num_years)
                    )
                metrics[f"{prefix}_ret_sharpe"] = float(s)
                metrics[f"{prefix}_ret_max_dd"] = float(dd)
                metrics[f"{prefix}_ret_calmar"] = float(c)
                metrics[f"{prefix}_ret_sortino"] = float(so)
                metrics[f"{prefix}_ret_sharpe_per_turnover"] = float(spt)
                metrics[f"{prefix}_win_rate"] = float(wr)

            metrics.update(self.fee_independent_metrics)
            if self.core == "pandas":
                summary_dict[fee_bp] = pd.Series(metrics)
            else:
                summary_rows.append({"cost": fee_bp, **metrics})

        def _reorder(cols: List[str]) -> List[str]:
            ordered = cols.copy()
            for prefix in ["long_short", "long", "short", "passive", "short_passive"]:
                turnover_col = f"{prefix}_turnover_ratio"
                maxdd_col = f"{prefix}_ret_max_dd"
                if turnover_col in ordered and maxdd_col in ordered:
                    ordered.remove(turnover_col)
                    ordered.insert(ordered.index(maxdd_col) + 1, turnover_col)
            return ordered

        if self.core == "pandas":
            self.summary_df = pd.concat(summary_dict, axis=1).T
            self.summary_df.index.name = "cost"
            cols = _reorder(self.summary_df.columns.tolist())
            self.summary_df = self.summary_df[cols]
        else:
            try:
                import polars as pl  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("polars is required for polars outputs") from exc

            self.summary_df = pl.DataFrame(summary_rows)
            cols = _reorder([c for c in self.summary_df.columns if c != "cost"])
            self.summary_df = self.summary_df.select(["cost", *cols])

    def _compute_ret_series(self) -> None:
        assert self._dt_list is not None
        assert self._day_list is not None

        self.ret = {pt: {} for pt in ["ls", "long", "short", "passive", "short_passive"]}
        self.ret_daily = {
            pt: {} for pt in ["ls", "long", "short", "passive", "short_passive"]
        }

        for fee_bp in self._tpnl_dt_by_fee.keys():
            for pt in ["ls", "long", "short", "passive", "short_passive"]:
                self.ret[pt][fee_bp] = pd.Series(
                    self._tpnl_dt_by_fee[fee_bp][pt].tolist(), index=self._dt_list
                )
                self.ret_daily[pt][fee_bp] = pd.Series(
                    self._tpnl_daily_by_fee[fee_bp][pt].tolist(), index=self._day_list
                )

    # ──────────────────────────────── Layer Outputs ────────────────────────────────

    def _calculate_layer_returns(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert self.panel is not None
        assert self._dt_list is not None
        assert self._dt_offsets_rows is not None
        assert self._day_list is not None
        assert self._day_offsets_dt is not None
        assert self._sym_offsets is not None
        assert self._sym_rows is not None
        assert self._layer_code is not None

        n_rows = int(self.panel.factor.shape[0])
        if n_rows == 0 or self.layers <= 0:
            return {}, {}

        layer_code = self._layer_code.astype(np.int32, copy=False)

        dt_sizes = np.diff(self._dt_offsets_rows).astype(np.int64, copy=False)
        row_dt_code = np.repeat(np.arange(dt_sizes.size, dtype=np.int64), dt_sizes)
        layer_counts = count_layers_by_dt(layer_code, self._dt_offsets_rows, int(self.layers))

        layer_pos = np.zeros(n_rows, dtype=np.float64)
        valid = layer_code > 0
        if np.any(valid):
            if self._weight_long is not None:
                # Market-cap (or other) weighted positions within each layer
                w = self._weight_long[valid]
                dt_code = row_dt_code[valid]
                layer_idx = layer_code[valid] - 1
                # Sum weights per (datetime, layer) group
                n_dt = int(self._dt_offsets_rows.shape[0] - 1)
                weight_sums = np.zeros((n_dt, int(self.layers)), dtype=np.float64)
                np.add.at(weight_sums, (dt_code, layer_idx), w)
                denom = weight_sums[dt_code, layer_idx]
                # Where weight sum > 0, use proportional weight; else fall back to equal
                has_weight = denom > 0
                layer_pos_valid = np.where(has_weight, w / denom, 0.0)
                # Fall back to equal weight for groups with no weight data
                if not np.all(has_weight):
                    counts = layer_counts[dt_code, layer_idx]
                    layer_pos_valid = np.where(has_weight, layer_pos_valid, 1.0 / counts)
                layer_pos[valid] = layer_pos_valid
            else:
                dt_code = row_dt_code[valid]
                layer_idx = layer_code[valid] - 1
                counts = layer_counts[dt_code, layer_idx]
                layer_pos[valid] = 1.0 / counts

        tpnl_obs, prev_layer = tpnl_obs_layer_membership_cs(
            layer_code,
            layer_pos,
            self.panel.ref_price,
            self._sym_offsets,
            self._sym_rows,
        )
        tpnl_dt_mat = sum_tpnl_by_dt_and_layer(
            tpnl_obs, prev_layer, self._dt_offsets_rows, int(self.layers)
        )

        day_starts = self._day_offsets_dt[:-1]
        ret_layer: Dict[str, Any] = {}
        ret_layer_daily: Dict[str, Any] = {}
        for layer_num in range(1, self.layers + 1):
            key = f"layer_{layer_num}"
            dt_vals = tpnl_dt_mat[:, layer_num - 1]
            day_vals = np.add.reduceat(dt_vals, day_starts)
            if self.core == "pandas":
                ret_layer[key] = pd.Series(dt_vals, index=self._dt_list)
                ret_layer_daily[key] = pd.Series(day_vals, index=self._day_list)
            else:
                ret_layer[key] = self._pl_timeseries("datetime", self._dt_list, dt_vals)
                ret_layer_daily[key] = self._pl_timeseries("date", self._day_list, day_vals)

        return ret_layer, ret_layer_daily

    def _calculate_layer_rank_ic(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert self.panel is not None
        assert self._dt_list is not None
        assert self._dt_offsets_rows is not None
        assert self._day_list is not None
        assert self._day_offsets_dt is not None
        assert self._layer_code is not None

        n_dt = int(self._dt_offsets_rows.shape[0] - 1)
        if n_dt == 0 or self.layers <= 0:
            return {}, {}

        layer_code = self._layer_code.astype(np.int16, copy=False)
        rank_ic_dt_layer = spearman_by_dt_layer(
            self.panel.factor,
            self.panel.label,
            layer_code,
            self._dt_offsets_rows,
            int(self.layers),
        )

        dt_has_layer = (
            np.add.reduceat((layer_code > 0).astype(np.int64), self._dt_offsets_rows[:-1])
            > 0
        )
        dt_index = [self._dt_list[i] for i in np.flatnonzero(dt_has_layer)]
        if len(dt_index) == 0:
            return {}, {}

        day_has_layer = (
            np.add.reduceat(dt_has_layer.astype(np.int64), self._day_offsets_dt[:-1]) > 0
        )
        date_index = [self._day_list[i] for i in np.flatnonzero(day_has_layer)]

        layer_counts_total = np.bincount(
            layer_code.astype(np.int64, copy=False), minlength=self.layers + 1
        )
        present_layers = np.flatnonzero(layer_counts_total[1:] > 0) + 1

        rank_ic_layer: Dict[str, Any] = {}
        rank_ic_layer_daily: Dict[str, Any] = {}

        for layer_num in present_layers.tolist():
            key = f"layer_{int(layer_num)}"
            dt_vals = rank_ic_dt_layer[:, layer_num - 1][dt_has_layer]
            if self.core == "pandas":
                rank_ic_layer[key] = pd.Series(dt_vals, index=dt_index)
            else:
                rank_ic_layer[key] = self._pl_timeseries("datetime", dt_index, dt_vals)

            daily_full = self._daily_mean_keep_all(
                rank_ic_dt_layer[:, layer_num - 1], self._day_offsets_dt
            )
            daily_vals = daily_full[day_has_layer]
            if self.core == "pandas":
                rank_ic_layer_daily[key] = pd.Series(daily_vals, index=date_index)
            else:
                rank_ic_layer_daily[key] = self._pl_timeseries(
                    "date", date_index, daily_vals
                )

        return rank_ic_layer, rank_ic_layer_daily

    # ──────────────────────────────── Output ────────────────────────────────

    def _compile_results(self) -> Dict[str, Any]:
        assert self.panel is not None
        assert self._dt_list is not None
        assert self._dt_offsets_rows is not None
        assert self._dt_trading_dates is not None
        assert self._sym_list is not None
        assert self._layer_code is not None
        assert self._pos_ls is not None
        assert self._pos_long is not None
        assert self._pos_short is not None
        assert self._pos_passive is not None

        n_rows = int(self.panel.factor.shape[0])
        row_index = np.arange(n_rows, dtype=np.int64)

        dt_sizes = np.diff(self._dt_offsets_rows).astype(np.int64, copy=False)
        trading_dates = np.repeat(np.asarray(self._dt_trading_dates, dtype=object), dt_sizes)

        sym_arr = np.asarray(self._sym_list, dtype=object)
        symbol_rows = sym_arr[self.panel.sym_code.astype(np.int64, copy=False)]

        if self.core == "pandas":
            ret = self.ret
            ret_daily = self.ret_daily
            excess_ret: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}
            excess_ret_daily: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}
            for fee_bp in self._tpnl_excess_dt_by_fee:
                for pt in ["long", "short"]:
                    excess_ret[pt][fee_bp] = pd.Series(
                        self._tpnl_excess_dt_by_fee[fee_bp][pt], index=self._dt_list
                    )
                    excess_ret_daily[pt][fee_bp] = pd.Series(
                        self._tpnl_excess_daily_by_fee[fee_bp][pt], index=self._day_list
                    )
        else:
            ret = {pt: {} for pt in ["ls", "long", "short", "passive", "short_passive"]}
            ret_daily = {
                pt: {} for pt in ["ls", "long", "short", "passive", "short_passive"]
            }

            for fee_bp, dt_map in self._tpnl_dt_by_fee.items():
                for pt in ["ls", "long", "short", "passive", "short_passive"]:
                    ret[pt][fee_bp] = self._pl_timeseries(
                        "datetime", self._dt_list, dt_map[pt]
                    )
                    ret_daily[pt][fee_bp] = self._pl_timeseries(
                        "date", self._day_list, self._tpnl_daily_by_fee[fee_bp][pt]
                    )

            excess_ret = {"long": {}, "short": {}}
            excess_ret_daily = {"long": {}, "short": {}}
            for fee_bp in self._tpnl_excess_dt_by_fee:
                for pt in ["long", "short"]:
                    excess_ret[pt][fee_bp] = self._pl_timeseries(
                        "datetime", self._dt_list, self._tpnl_excess_dt_by_fee[fee_bp][pt]
                    )
                    excess_ret_daily[pt][fee_bp] = self._pl_timeseries(
                        "date", self._day_list, self._tpnl_excess_daily_by_fee[fee_bp][pt]
                    )

        if self.core == "pandas":
            index = pd.RangeIndex(n_rows)
            pos = {
                "ls": pd.Series(self._pos_ls, index=index, name="pos"),
                "long": pd.Series(self._pos_long, index=index, name="pos"),
                "short": pd.Series(self._pos_short, index=index, name="pos"),
                "passive": pd.Series(self._pos_passive, index=index, name="pos"),
                "short_passive": pd.Series(-self._pos_passive, index=index, name="pos"),
            }
            factor = pd.Series(self.panel.factor, index=index, name="factor")
            label = pd.Series(self.panel.label, index=index, name="label")
            symbol = pd.Series(symbol_rows, index=index, name="symbol")
            trading_date = pd.Series(trading_dates, index=index, name="trading_date")
            datetime_obj: Any = pd.Series(self._dt_list, name="datetime")
        else:
            try:
                import polars as pl  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("polars is required for polars outputs") from exc

            pos = {
                "ls": pl.DataFrame({"row": row_index, "value": self._pos_ls}),
                "long": pl.DataFrame({"row": row_index, "value": self._pos_long}),
                "short": pl.DataFrame({"row": row_index, "value": self._pos_short}),
                "passive": pl.DataFrame({"row": row_index, "value": self._pos_passive}),
                "short_passive": pl.DataFrame(
                    {"row": row_index, "value": -self._pos_passive}
                ),
            }
            factor = pl.DataFrame({"row": row_index, "value": self.panel.factor})
            label = pl.DataFrame({"row": row_index, "value": self.panel.label})
            symbol = pl.DataFrame({"row": row_index, "symbol": symbol_rows.tolist()})
            trading_date = pl.DataFrame(
                {"row": row_index, "trading_date": trading_dates.tolist()}
            )
            datetime_obj = pl.DataFrame({"datetime": self._dt_list})

        return {
            "ret": ret,
            "ret_daily": ret_daily,
            "excess_ret": excess_ret,
            "excess_ret_daily": excess_ret_daily,
            "rank_ic": self.rank_ic,
            "rank_ic_daily": self.rank_ic_daily,
            "ic": self.ic,
            "ic_daily": self.ic_daily,
            "turnover": self.turnover,
            "turnover_daily": self.turnover_daily,
            "pos": pos,
            "factor": factor,
            "label": label,
            "acf": self.acf,
            "datetime": datetime_obj,
            "symbol": symbol,
            "trading_date": trading_date,
            "time_metrics": self.time_metrics,
            "ret_layer": self.ret_layer,
            "ret_layer_daily": self.ret_layer_daily,
            "rank_ic_layer": self.rank_ic_layer,
            "rank_ic_layer_daily": self.rank_ic_layer_daily,
        }
