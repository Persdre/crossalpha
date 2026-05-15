"""Time-series backtest engine (array-first, Numba-accelerated).

This engine matches the legacy output contract of
`scripts.backtesting.backtest.TS_Backtest` while running compute-heavy stages on
NumPy arrays and Numba kernels.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm

from ..adapters.array_builders import build_ts_panel_arrays
from ..core import CoreName, detect_core
from ..kernels.correlation import corr, spearman
from ..kernels.metrics import (
    annual_ret,
    calmar,
    max_dd,
    sharpe,
    sharpe_per_turnover,
    sortino,
)
from ..kernels.pnl import tpnl_ts_4
from ..kernels.time_metrics import compute_time_metrics
from ..kernels.transforms_ts import (
    rolling_normalize_demean,
    rolling_normalize_minmax,
    rolling_normalize_rank,
    rolling_normalize_zscore,
    rolling_winsorize_mad,
    rolling_winsorize_quantile,
    rolling_winsorize_std,
)
from ..structures import TSPanelArrays


class TS_Backtest:
    """Drop-in equivalent of `scripts.backtesting.backtest.TS_Backtest`."""

    def __init__(self, config: dict):
        self.config = config
        self.factor_series = config["factor"]
        self.label_series = config["label"]
        self.core: CoreName = detect_core(self.factor_series)
        if detect_core(self.label_series) != self.core:
            raise TypeError("factor/label backends must match")
        self.backtest_mode: str = config["backtest_mode"]
        self.backtest_params: dict = config.get("backtest_params", {})
        self.fees: List[float] = list(config["fees"])
        self.lag: int = int(config["lag"])
        self.annual_days = config.get("annual_days")

        self.panel: TSPanelArrays | None = None

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

        self.summary_df: Any = None

        self._rank_ic_daily_index: List | None = None
        self._rank_ic_daily_values: np.ndarray | None = None

        self._dt_list: List | None = None
        self._trading_dates: List | None = None
        self._day_list: List | None = None
        self._day_offsets_dt: np.ndarray | None = None

        self._pos_ls: np.ndarray | None = None
        self._pos_long: np.ndarray | None = None
        self._pos_short: np.ndarray | None = None
        self._pos_passive: np.ndarray | None = None
        self._pos_short_passive: np.ndarray | None = None

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
        self._calculate_positions()
        self._compute_time_metrics()

        self._add_rank_ic_metrics()
        self._add_acf_metrics()
        self._add_turnover_metrics()

        self._run_fee_scenarios()
        if self.core == "pandas":
            self._compute_ret_series()

        return self._compile_results(), self.summary_df

    # ──────────────────────────────── Pipeline Steps ────────────────────────────────

    def _prepare_data(self) -> None:
        self.panel = build_ts_panel_arrays(self.factor_series, self.label_series, lag=self.lag)
        self._dt_list = self.panel.dt_list
        self._trading_dates = self.panel.trading_dates
        self._day_list = self.panel.day_list
        self._day_offsets_dt = self.panel.day_offsets_dt

    def _winsorize_rolling(self, values: np.ndarray) -> np.ndarray:
        method = self.backtest_params.get("winsorize_method", "std")
        n = float(self.backtest_params.get("winsorize_n", 3.0))
        window = int(self.backtest_params.get("winsorize_rolling_window_n", 252))

        if method == "std":
            return rolling_winsorize_std(values, window, n)
        if method == "mad":
            return rolling_winsorize_mad(values, window, n)
        if method == "quantile":
            return rolling_winsorize_quantile(values, window, n)
        raise ValueError(f"Unknown winsorize method: {method}")

    def _normalize_rolling(self, values: np.ndarray) -> np.ndarray:
        method = self.backtest_params.get("normalization_method", "zscore")
        window = int(self.backtest_params.get("normalization_rolling_window_n", 252))

        if method == "zscore":
            return rolling_normalize_zscore(values, window)
        if method == "minmax":
            return rolling_normalize_minmax(values, window)
        if method == "rank":
            return rolling_normalize_rank(values, window)
        if method == "demean":
            return rolling_normalize_demean(values, window)
        raise ValueError(f"Unknown normalization method: {method}")

    def _squash(self, values: np.ndarray) -> np.ndarray:
        method = self.backtest_params.get("squash_method", "tanh")
        if method == "tanh":
            return np.tanh(values)
        if method == "clip":
            return np.clip(values, -1.0, 1.0)
        raise ValueError(f"Unknown squash method: {method}")

    def _calculate_positions(self) -> None:
        assert self.panel is not None

        factor = self.panel.factor
        n = int(factor.shape[0])
        if n == 0:
            self._pos_ls = np.empty(0, dtype=np.float64)
            self._pos_long = np.empty(0, dtype=np.float64)
            self._pos_short = np.empty(0, dtype=np.float64)
            self._pos_passive = np.empty(0, dtype=np.float64)
            self._pos_short_passive = np.empty(0, dtype=np.float64)
            return

        factor_valid = np.isfinite(factor)

        if self.backtest_mode == "weights":
            pos_ls = np.where(factor_valid, factor, 0.0)

        elif self.backtest_mode == "long/short_threshold":
            lt = float(self.backtest_params["long_threshold"])
            st = float(self.backtest_params["short_threshold"])
            raw = np.zeros(n, dtype=np.float64)
            raw[factor >= lt] = 1.0
            raw[factor <= st] = -1.0
            pos_ls = np.where(factor_valid, raw, 0.0)

        elif self.backtest_mode == "gradual_long/short_threshold":
            es = float(self.backtest_params["end_short_threshold"])
            ss = float(self.backtest_params["start_short_threshold"])
            sl = float(self.backtest_params["start_long_threshold"])
            el = float(self.backtest_params["end_long_threshold"])

            raw = np.zeros(n, dtype=np.float64)
            raw[factor <= es] = -1.0
            raw[factor >= el] = 1.0

            mask = (factor > es) & (factor < ss)
            raw[mask] = -1.0 + (factor[mask] - es) / (ss - es)

            mask = (factor > sl) & (factor < el)
            raw[mask] = (factor[mask] - sl) / (el - sl)

            pos_ls = np.where(factor_valid, raw, 0.0)

        elif self.backtest_mode == "long/short_normalization":
            wins = self._winsorize_rolling(factor)
            normed = self._normalize_rolling(wins)
            squashed = self._squash(normed)
            pos_ls = np.where(factor_valid, squashed, 0.0)

        else:
            raise ValueError(f"Unknown backtest_mode: {self.backtest_mode}")

        self._pos_ls = pos_ls.astype(np.float64, copy=False)
        self._pos_long = np.where(pos_ls > 0.0, pos_ls, 0.0).astype(np.float64, copy=False)
        self._pos_short = np.where(pos_ls < 0.0, pos_ls, 0.0).astype(np.float64, copy=False)
        self._pos_passive = np.ones(n, dtype=np.float64)
        self._pos_short_passive = -np.ones(n, dtype=np.float64)

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

    def _daily_mean_from_dt_values(self, values: np.ndarray) -> pd.Series:
        out_dates, out_vals = self._daily_mean_components_from_dt_values(values)
        if len(out_dates) == 0:
            return pd.Series(dtype=float)
        return pd.Series(out_vals, index=out_dates)

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

    def _add_rank_ic_metrics(self) -> None:
        assert self.panel is not None
        assert self._dt_list is not None
        assert self._day_list is not None
        assert self._day_offsets_dt is not None

        window = 126
        f = self.panel.factor
        l = self.panel.label
        n = int(f.shape[0])

        rank_ic_dt = np.full(n, np.nan, dtype=np.float64)
        ic_dt = np.full(n, np.nan, dtype=np.float64)
        if n >= window:
            f_win = sliding_window_view(f, window)
            l_win = sliding_window_view(l, window)
            rank_ic_vals = spearman(f_win, l_win)
            ic_vals = corr(f_win, l_win)
            rank_ic_dt[window - 1 :] = rank_ic_vals.astype(np.float64, copy=False)
            ic_dt[window - 1 :] = ic_vals.astype(np.float64, copy=False)

        if self.core == "pandas":
            self.rank_ic = pd.Series(rank_ic_dt, index=self._dt_list)
            self.ic = pd.Series(ic_dt, index=self._dt_list)
            self.rank_ic_daily = self._daily_mean_from_dt_values(rank_ic_dt)
            self.ic_daily = self._daily_mean_from_dt_values(ic_dt)
        else:
            self.rank_ic = self._pl_timeseries("datetime", self._dt_list, rank_ic_dt)
            self.ic = self._pl_timeseries("datetime", self._dt_list, ic_dt)

            daily_dates, daily_vals = self._daily_mean_components_from_dt_values(rank_ic_dt)
            self._rank_ic_daily_index = daily_dates
            self._rank_ic_daily_values = daily_vals
            self.rank_ic_daily = self._pl_timeseries("date", daily_dates, daily_vals)

            ic_daily_dates, ic_daily_vals = self._daily_mean_components_from_dt_values(ic_dt)
            self.ic_daily = self._pl_timeseries("date", ic_daily_dates, ic_daily_vals)

        valid_rank_ic = rank_ic_dt[~np.isnan(rank_ic_dt)]
        if valid_rank_ic.size > 0:
            rank_ic_mean = float(np.mean(valid_rank_ic))
            rank_ic_std = float(np.std(valid_rank_ic))
            n_obs = int(valid_rank_ic.size)
            rank_ic_z = (
                rank_ic_mean / (rank_ic_std / np.sqrt(n_obs)) if rank_ic_std > 0 else 0
            )
            rank_ic_p = float(2 * (1 - norm.cdf(abs(rank_ic_z))))

            daily_arr = (
                self._rank_ic_daily_values
                if self._rank_ic_daily_values is not None
                else self._daily_mean_components_from_dt_values(rank_ic_dt)[1]
            )
            daily_mean, daily_std = (
                (float(np.mean(daily_arr)), float(np.std(daily_arr)))
                if daily_arr.size > 0
                else (0.0, 1.0)
            )
            rank_icir = abs(daily_mean) / daily_std if daily_std > 0 else 0
            rank_icir_annual = rank_icir * np.sqrt(self.time_metrics["dates_to_years"])

            self.fee_independent_metrics.update(
                {
                    "rank_ic": rank_ic_mean,
                    "rank_icir": rank_icir,
                    "rank_icir_annual": rank_icir_annual,
                    "rank_ic_p_value": rank_ic_p,
                    "rank_ic_winratio": float(np.sum(valid_rank_ic > 0) / n_obs),
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

        valid_ic = ic_dt[~np.isnan(ic_dt)]
        if valid_ic.size > 0:
            ic_mean = float(np.mean(valid_ic))
            ic_std = float(np.std(valid_ic))
            n_obs = int(valid_ic.size)
            ic_z = ic_mean / (ic_std / np.sqrt(n_obs)) if ic_std > 0 else 0
            ic_p = float(2 * (1 - norm.cdf(abs(ic_z))))

            daily_arr = self._daily_mean_components_from_dt_values(ic_dt)[1]
            daily_mean, daily_std = (
                (float(np.mean(daily_arr)), float(np.std(daily_arr)))
                if daily_arr.size > 0
                else (0.0, 1.0)
            )
            icir = abs(daily_mean) / daily_std if daily_std > 0 else 0
            icir_annual = icir * np.sqrt(self.time_metrics["dates_to_years"])

            self.fee_independent_metrics.update(
                {
                    "ic": ic_mean,
                    "icir": icir,
                    "icir_annual": icir_annual,
                    "ic_p_value": ic_p,
                    "ic_winratio": float(np.sum(valid_ic > 0) / n_obs),
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

    def _add_acf_metrics(self) -> None:
        assert self.panel is not None
        f = self.panel.factor
        n = int(f.shape[0])

        lags = list(range(1, 21)) + list(range(25, 51, 5)) + [100, 150, 200]
        acf: Dict[int, float] = {}
        for lag in lags:
            if n > lag:
                A = f[:-lag].reshape(1, -1)
                B = f[lag:].reshape(1, -1)
                acf[lag] = float(spearman(A, B)[0])
            else:
                acf[lag] = np.nan

        acf_halflife = next((lag for lag in sorted(acf.keys()) if acf[lag] < 0.5), None)
        self.fee_independent_metrics.update(
            {
                "acf_1": acf.get(1, 0),
                "acf_halflife": acf_halflife if acf_halflife else max(lags),
            }
        )
        self.acf = acf

    def _add_turnover_metrics(self) -> None:
        assert self._dt_list is not None
        assert self._day_list is not None
        assert self._day_offsets_dt is not None
        assert self._pos_ls is not None
        assert self._pos_long is not None
        assert self._pos_short is not None
        assert self._pos_passive is not None

        self.turnover = {}
        self.turnover_daily = {}

        pos_types = ["ls", "long", "short", "passive", "short_passive"]
        prefix_map = {
            "ls": "long_short",
            "long": "long",
            "short": "short",
            "passive": "passive",
            "short_passive": "short_passive",
        }
        pos_map = {
            "ls": self._pos_ls,
            "long": self._pos_long,
            "short": self._pos_short,
            "passive": self._pos_passive,
            "short_passive": self._pos_short_passive,
        }

        for pt in pos_types:
            pos = pos_map[pt]
            diff = np.abs(np.diff(pos, prepend=0.0))
            if self.core == "pandas":
                self.turnover[pt] = pd.Series(diff, index=self._dt_list)
            else:
                self.turnover[pt] = self._pl_timeseries("datetime", self._dt_list, diff)
            self.fee_independent_metrics[f"{prefix_map[pt]}_turnover"] = float(
                np.nanmean(diff)
            )

            day_sum = np.add.reduceat(diff, self._day_offsets_dt[:-1])
            day_count = np.diff(self._day_offsets_dt).astype(np.float64)
            day_mean = day_sum / day_count
            if self.core == "pandas":
                self.turnover_daily[pt] = pd.Series(day_mean.tolist(), index=self._day_list)
            else:
                self.turnover_daily[pt] = self._pl_timeseries(
                    "date", self._day_list, day_mean
                )

    def _run_fee_scenarios(self) -> None:
        assert self.panel is not None
        assert self._day_offsets_dt is not None
        assert self._dt_list is not None
        assert self._day_list is not None
        assert self._pos_ls is not None
        assert self._pos_long is not None
        assert self._pos_short is not None
        assert self._pos_passive is not None
        assert self._pos_short_passive is not None

        num_years = float(self.time_metrics["num_years"])
        dates_to_years = float(self.time_metrics["dates_to_years"])

        summary_dict: Dict[str, pd.Series] = {}
        summary_rows: List[Dict[str, Any]] = []

        for fee in self.fees:
            fee_bp = f"{round(fee * 1e4, 1)} bp"
            (
                tpnl_ls,
                tpnl_long,
                tpnl_short,
                tpnl_passive,
                wins,
                active,
            ) = tpnl_ts_4(
                self._pos_ls,
                self._pos_long,
                self._pos_short,
                self._pos_passive,
                self.panel.ref_price,
                float(fee),
            )

            day_ls = np.add.reduceat(tpnl_ls, self._day_offsets_dt[:-1])
            day_long = np.add.reduceat(tpnl_long, self._day_offsets_dt[:-1])
            day_short = np.add.reduceat(tpnl_short, self._day_offsets_dt[:-1])
            day_passive = np.add.reduceat(tpnl_passive, self._day_offsets_dt[:-1])

            tpnl_short_passive = np.empty_like(tpnl_passive)
            if tpnl_short_passive.size:
                fee_f = float(fee)
                tpnl_short_passive[0] = -fee_f
                if tpnl_short_passive.size > 1:
                    prev_label = self.panel.label[:-1]
                    tpnl_short_passive[1:] = -prev_label - np.abs(prev_label) * fee_f
            day_short_passive = np.add.reduceat(tpnl_short_passive, self._day_offsets_dt[:-1])

            self._tpnl_dt_by_fee[fee_bp] = {
                "ls": tpnl_ls,
                "long": tpnl_long,
                "short": tpnl_short,
                "passive": tpnl_passive,
                "short_passive": tpnl_short_passive,
            }
            self._tpnl_daily_by_fee[fee_bp] = {
                "ls": day_ls,
                "long": day_long,
                "short": day_short,
                "passive": day_passive,
                "short_passive": day_short_passive,
            }
            self._tpnl_excess_dt_by_fee[fee_bp] = {
                "long": tpnl_long - tpnl_passive,
                "short": tpnl_short - tpnl_short_passive,
            }
            self._tpnl_excess_daily_by_fee[fee_bp] = {
                "long": day_long - day_passive,
                "short": day_short - day_short_passive,
            }

            wins_short_passive = int(np.sum(tpnl_short_passive > 0.0))
            active_short_passive = int(tpnl_short_passive.size)
            wins_all = np.empty(5, dtype=np.int64)
            active_all = np.empty(5, dtype=np.int64)
            wins_all[:4] = wins
            active_all[:4] = active
            wins_all[4] = wins_short_passive
            active_all[4] = active_short_passive

            metrics: Dict[str, Any] = {}
            portfolio_types = ["ls", "long", "short", "passive", "short_passive"]
            prefix_map_metrics = {
                "ls": "long_short",
                "long": "long",
                "short": "short",
                "passive": "passive",
                "short_passive": "short_passive",
            }

            for i_pt, pt in enumerate(portfolio_types):
                prefix = prefix_map_metrics[pt]
                pt_daily = self._tpnl_daily_by_fee[fee_bp][pt]

                annual = annual_ret(pt_daily, num_years)
                dd = max_dd(pt_daily)
                s = sharpe(pt_daily, dates_to_years)
                so = sortino(pt_daily, dates_to_years)
                c = calmar(annual, dd)
                turnover_val = float(
                    self.fee_independent_metrics.get(f"{prefix}_turnover", 0)
                )
                spt = sharpe_per_turnover(s, turnover_val)
                wr = (
                    float(wins_all[i_pt] / active_all[i_pt])
                    if active_all[i_pt] > 0
                    else 0.0
                )

                metrics[f"{prefix}_ret_annual"] = float(annual)
                if pt == "long":
                    metrics[f"{prefix}_excess_ret_annual"] = float(
                        annual_ret(pt_daily - day_passive, num_years)
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

        if self.core == "pandas":
            self.summary_df = pd.concat(summary_dict, axis=1).T
            self.summary_df.index.name = "cost"
        else:
            try:
                import polars as pl  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("polars is required for polars outputs") from exc
            self.summary_df = pl.DataFrame(summary_rows)

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

    def _compile_results(self) -> Dict[str, Any]:
        assert self.panel is not None
        assert self._dt_list is not None
        assert self._trading_dates is not None
        assert self._day_list is not None
        assert self._pos_ls is not None
        assert self._pos_long is not None
        assert self._pos_short is not None
        assert self._pos_passive is not None
        assert self._pos_short_passive is not None

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
            pos = {
                "ls": pd.Series(self._pos_ls.tolist(), index=self._dt_list),
                "long": pd.Series(self._pos_long.tolist(), index=self._dt_list),
                "short": pd.Series(self._pos_short.tolist(), index=self._dt_list),
                "passive": pd.Series(self._pos_passive.tolist(), index=self._dt_list),
                "short_passive": pd.Series(
                    self._pos_short_passive.tolist(), index=self._dt_list
                ),
            }
            factor = pd.Series(self.panel.factor.tolist(), index=self._dt_list)
            label = pd.Series(self.panel.label.tolist(), index=self._dt_list)
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

            pos = {
                "ls": self._pl_timeseries("datetime", self._dt_list, self._pos_ls),
                "long": self._pl_timeseries("datetime", self._dt_list, self._pos_long),
                "short": self._pl_timeseries("datetime", self._dt_list, self._pos_short),
                "passive": self._pl_timeseries(
                    "datetime", self._dt_list, self._pos_passive
                ),
                "short_passive": self._pl_timeseries(
                    "datetime", self._dt_list, self._pos_short_passive
                ),
            }
            factor = self._pl_timeseries("datetime", self._dt_list, self.panel.factor)
            label = self._pl_timeseries("datetime", self._dt_list, self.panel.label)

        if self.core == "pandas":
            datetime_obj: Any = pd.Series(self._dt_list, name="datetime")
            trading_date_obj: Any = pd.Series(self._trading_dates, name="trading_date")
            symbol_obj: Any = pd.Series([self.config.get("symbol", "UNKNOWN")], name="symbol")
        else:
            try:
                import polars as pl  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("polars is required for polars outputs") from exc

            datetime_obj = pl.DataFrame({"datetime": self._dt_list})
            trading_date_obj = pl.DataFrame({"trading_date": self._trading_dates})
            symbol_obj = pl.DataFrame({"symbol": [self.config.get("symbol", "UNKNOWN")]})

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
            "symbol": symbol_obj,
            "trading_date": trading_date_obj,
            "time_metrics": self.time_metrics,
        }
