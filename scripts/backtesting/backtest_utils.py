"""Public API for backtesting (drop-in compatible with legacy API)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .backtest_doc import CS_Backtest_Doc, TS_Backtest_Doc
from .core import detect_core
from .html import CSReportGenerator, TSReportGenerator
from .plotting import (
    plot_cumulative_returns,
    plot_factor_autocorrelation,
    plot_factor_distribution,
    plot_layer_annual_returns,
    plot_layer_cumulative_returns,
    plot_layer_rank_ic,
    plot_rank_ic_cumulative,
    plot_rank_ic_interval_analysis,
    plot_turnover,
)
from .utils import validate_cs_params, validate_ts_params

from .engines.cs_engine import CS_Backtest
from .engines.ts_engine import TS_Backtest


def _json_key(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return str(value)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()

    if isinstance(value, pd.Series):
        return {_json_key(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, pd.DataFrame):
        return _to_jsonable(value.to_dict(orient="split"))

    try:
        import polars as pl  # type: ignore

        if isinstance(value, pl.Series):
            return value.to_list()
        if isinstance(value, pl.DataFrame):
            return value.to_dicts()
    except Exception:
        pass

    if isinstance(value, dict):
        return {_json_key(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    try:
        import numpy as np  # local import: optional dependency here

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass

    return str(value)


def _save_json(output_path: Path, payload: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, separators=(",", ":"))


def _as_finite_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(v):
        return None
    if v == float("inf") or v == float("-inf"):
        return None
    return v


def _summary_df_to_metrics_by_cost(
    summary_df: Any,
) -> Dict[str, Dict[str, Dict[str, float | None]]]:
    metrics: Dict[str, Dict[str, Dict[str, float | None]]] = {}
    core = detect_core(summary_df)

    if core == "pandas":
        fee_labels = [str(idx) for idx in summary_df.index]
        for metric_name in sorted(summary_df.columns.astype(str).tolist()):
            by_cost: Dict[str, float | None] = {}
            for fee_label in fee_labels:
                by_cost[fee_label] = _as_finite_float(
                    summary_df.loc[fee_label, metric_name]
                )
            metrics[metric_name] = {"by_cost": by_cost}
        return metrics

    try:
        import polars as pl  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("polars is required for polars summary_df") from exc

    if not isinstance(summary_df, pl.DataFrame):
        raise TypeError(f"Unsupported summary_df type: {type(summary_df)}")
    if "cost" not in summary_df.columns:
        raise ValueError("polars summary_df must include a 'cost' column")

    fee_labels = [str(v) for v in summary_df.get_column("cost").to_list()]
    rows_by_cost = {str(row["cost"]): row for row in summary_df.to_dicts()}
    metric_cols = [c for c in summary_df.columns if c != "cost"]
    for metric_name in sorted([str(c) for c in metric_cols]):
        by_cost = {
            fee_label: _as_finite_float(rows_by_cost.get(fee_label, {}).get(metric_name))
            for fee_label in fee_labels
        }
        metrics[metric_name] = {"by_cost": by_cost}

    return metrics


def _evaluate_metrics(
    metrics: Dict[str, Dict[str, Dict[str, float | None]]],
) -> tuple[bool, str, List[str]]:
    issues: List[str] = []

    for metric_name, metric_payload in metrics.items():
        by_cost = metric_payload.get("by_cost", {})
        for fee_label, val in by_cost.items():
            if val is None:
                issues.append(f"missing_or_nan:{fee_label}:{metric_name}")
                continue

            if metric_name.endswith("ret_annual") and abs(val) > 5.0:
                issues.append(f"annual_ret_too_large:{fee_label}:{metric_name}:{val}")
            if metric_name.endswith("ret_sharpe") and abs(val) > 10.0:
                issues.append(f"sharpe_too_large:{fee_label}:{metric_name}:{val}")
            if metric_name in {"rank_ic"} and abs(val) > 0.5:
                issues.append(f"rank_ic_too_large:{fee_label}:{metric_name}:{val}")
            if metric_name in {"rank_icir", "rank_icir_annual"} and abs(val) > 10.0:
                issues.append(f"icir_too_large:{fee_label}:{metric_name}:{val}")

            if metric_name.endswith("ret_max_dd") and not (-1.0 <= val <= 0.0):
                issues.append(f"max_dd_out_of_range:{fee_label}:{metric_name}:{val}")

    is_valid = len(issues) == 0

    fee_pref = metrics.get("long_short_ret_sharpe", {}).get("by_cost", {}).keys()
    fee_pref = sorted(fee_pref)[0] if fee_pref else "0.0 bp"
    sharpe = metrics.get("long_short_ret_sharpe", {}).get("by_cost", {}).get(fee_pref)
    annual_ret = metrics.get("long_short_ret_annual", {}).get("by_cost", {}).get(fee_pref)
    rank_ic = metrics.get("rank_ic", {}).get("by_cost", {}).get(fee_pref)

    if not is_valid:
        remarks = "Invalid backtest metrics; see issues for details."
    else:
        parts: List[str] = ["Backtest completed."]
        if sharpe is not None:
            parts.append(f"Sharpe({fee_pref})={sharpe:.3f}")
        if annual_ret is not None:
            parts.append(f"AnnualRet({fee_pref})={annual_ret:.3f}")
        if rank_ic is not None:
            parts.append(f"RankIC={rank_ic:.4f}")

        if sharpe is not None and annual_ret is not None:
            if sharpe >= 1.0 and annual_ret > 0:
                parts.append("Overall: strong performance.")
            elif sharpe > 0 and annual_ret > 0:
                parts.append("Overall: moderate performance.")
            else:
                parts.append("Overall: weak/negative performance.")

        remarks = " ".join(parts)

    return is_valid, remarks, issues


def _build_summary_payload(summary_df: Any) -> Dict[str, Any]:
    metrics = _summary_df_to_metrics_by_cost(summary_df)
    result_valid, result_remarks, issues = _evaluate_metrics(metrics)
    return {
        "result_valid": result_valid,
        "result_remarks": result_remarks,
        "issues": issues,
        "metrics": metrics,
    }


def run_cross_section_backtest(
    df: Any,
    factor_column_name: str,
    datetime_column_name: str,
    symbol_column_name: str,
    raw_label_column_name: str,
    freq: str,
    layers_use: int | None = None,
    fees: List[float] = None,
    backtest_mode: str = "long/short_layers",
    backtest_params: dict | None = None,
    lag: int = 2,
    annual_days: int | None = None,
    factor_info_dict: dict | None = None,
    *,
    weight_column_name: str | None = None,
    output_dir: str | Path | None = None,
) -> Tuple[Dict[str, Any], Any]:
    """Drop-in compatible CS backtest with optimized compute kernels."""
    if fees is None:
        fees = [0.0e-4, 2.0e-4, 5.0e-4]

    weight_df = None
    core = detect_core(df)
    if core == "pandas":
        feat = (
            df.rename(
                columns={
                    datetime_column_name: "datetime",
                    symbol_column_name: "symbol",
                    factor_column_name: "pred_ret",
                }
            )
            .loc[:, ["datetime", "symbol", "pred_ret"]]
            .assign(datetime=lambda _df: pd.to_datetime(_df["datetime"]))
        )
        factor_df = (
            feat.pivot(index="datetime", columns="symbol", values="pred_ret").sort_index()
        )

        label = (
            df.rename(
                columns={
                    datetime_column_name: "datetime",
                    symbol_column_name: "symbol",
                    raw_label_column_name: "label",
                }
            )
            .loc[:, ["datetime", "symbol", "label"]]
            .assign(datetime=lambda _df: pd.to_datetime(_df["datetime"]))
        )
        label_df = label.pivot(index="datetime", columns="symbol", values="label").sort_index()

        common_idx = factor_df.index.intersection(label_df.index)
        common_cols = factor_df.columns.intersection(label_df.columns)
        if common_idx.empty or common_cols.empty:
            raise ValueError("No overlapping dates or symbols between factors and labels.")

        start_dt = common_idx.min()
        end_dt = common_idx.max()

        factor_df = factor_df.loc[common_idx, common_cols]
        label_df = label_df.loc[common_idx, common_cols]

        weight_df = None
        if weight_column_name is not None and weight_column_name in df.columns:
            wt = (
                df.rename(
                    columns={
                        datetime_column_name: "datetime",
                        symbol_column_name: "symbol",
                        weight_column_name: "weight",
                    }
                )
                .loc[:, ["datetime", "symbol", "weight"]]
                .assign(datetime=lambda _df: pd.to_datetime(_df["datetime"]))
            )
            weight_df = (
                wt.pivot(index="datetime", columns="symbol", values="weight")
                .sort_index()
                .loc[common_idx, common_cols]
            )

        common_cols_list = list(common_cols)
        n_datetimes = int(len(common_idx))

    else:
        try:
            import polars as pl  # type: ignore
        except Exception as exc:
            raise RuntimeError("polars is required for polars inputs") from exc
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"Unsupported df type: {type(df)}")

        def _ensure_datetime_col(frame: pl.DataFrame, col: str) -> pl.DataFrame:
            dtype = frame.schema.get(col)
            if dtype == pl.Utf8:
                return frame.with_columns(
                    pl.col(col).str.strptime(pl.Datetime, strict=False).alias(col)
                )
            return frame.with_columns(pl.col(col).cast(pl.Datetime).alias(col))

        feat = df.rename(
            {
                datetime_column_name: "datetime",
                symbol_column_name: "symbol",
                factor_column_name: "pred_ret",
            }
        ).select(["datetime", "symbol", "pred_ret"])
        feat = _ensure_datetime_col(feat, "datetime")
        factor_df = (
            feat.pivot(index="datetime", columns="symbol", values="pred_ret")
            .sort("datetime")
        )

        label = df.rename(
            {
                datetime_column_name: "datetime",
                symbol_column_name: "symbol",
                raw_label_column_name: "label",
            }
        ).select(["datetime", "symbol", "label"])
        label = _ensure_datetime_col(label, "datetime")
        label_df = label.pivot(index="datetime", columns="symbol", values="label").sort(
            "datetime"
        )

        common_cols = sorted(
            set(factor_df.columns).intersection(label_df.columns) - {"datetime"}
        )
        if not common_cols:
            raise ValueError("No overlapping symbols between factors and labels.")

        common_dt = (
            factor_df.select("datetime")
            .join(label_df.select("datetime"), on="datetime", how="inner")
            .unique()
            .sort("datetime")
        )
        if common_dt.height == 0:
            raise ValueError("No overlapping datetimes between factors and labels.")

        start_dt = common_dt.select(pl.col("datetime").min()).item()
        end_dt = common_dt.select(pl.col("datetime").max()).item()
        n_datetimes = int(common_dt.height)

        factor_df = (
            factor_df.join(common_dt, on="datetime", how="semi")
            .select(["datetime", *common_cols])
            .sort("datetime")
        )
        label_df = (
            label_df.join(common_dt, on="datetime", how="semi")
            .select(["datetime", *common_cols])
            .sort("datetime")
        )

        weight_df = None
        if weight_column_name is not None and weight_column_name in df.columns:
            wt = df.rename({
                datetime_column_name: "datetime",
                symbol_column_name: "symbol",
                weight_column_name: "weight",
            }).select(["datetime", "symbol", "weight"])
            wt = _ensure_datetime_col(wt, "datetime")
            weight_df = (
                wt.pivot(index="datetime", columns="symbol", values="weight")
                .sort("datetime")
            )
            # Align to common datetimes and symbols
            weight_cols = [c for c in common_cols if c in weight_df.columns]
            missing_cols = [c for c in common_cols if c not in weight_df.columns]
            weight_df = (
                weight_df.join(common_dt, on="datetime", how="semi")
                .sort("datetime")
            )
            select_cols = ["datetime"] + weight_cols
            weight_df = weight_df.select(select_cols)
            # Add missing symbols as null columns
            for c in missing_cols:
                weight_df = weight_df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
            weight_df = weight_df.select(["datetime", *common_cols])

        common_cols_list = common_cols

    if layers_use is None:
        layers_use = min(5, len(common_cols_list))

    backtest_params = validate_cs_params(
        backtest_mode=backtest_mode,
        backtest_params=backtest_params,
        layers_use=layers_use,
        df=df,
        datetime_col=datetime_column_name,
        symbol_col=symbol_column_name,
    )

    cfg_cs = {
        "factor": factor_df,
        "label": label_df,
        "layers": layers_use,
        "freq": freq,
        "fees": fees,
        "backtest_mode": backtest_mode,
        "backtest_params": backtest_params,
        "lag": lag,
        "annual_days": annual_days,
    }
    if weight_df is not None:
        cfg_cs["weight"] = weight_df

    cs_backtest = CS_Backtest(cfg_cs)
    result_dict, summary_df = cs_backtest.run()

    plot_width = 380
    plot_height = 320
    plot_info = [
        (
            "plot1",
            "Long/Short Cumulative Returns",
            plot_cumulative_returns(
                result_dict["ret_daily"]["ls"],
                "Long/Short Cumulative Returns",
                plot_width,
                plot_height,
            ),
        ),
        (
            "plot2",
            "Long Only Cumulative Returns",
            plot_cumulative_returns(
                result_dict["ret_daily"]["long"],
                "Long Only Cumulative Returns",
                plot_width,
                plot_height,
            ),
        ),
        (
            "plot3",
            "Long Only Excess Cumulative Returns",
            plot_cumulative_returns(
                result_dict["excess_ret_daily"]["long"],
                "Long Only Excess Cumulative Returns",
                plot_width,
                plot_height,
            ),
        ),
        (
            "plot4",
            "Short Only Cumulative Returns",
            plot_cumulative_returns(
                result_dict["ret_daily"]["short"],
                "Short Only Cumulative Returns",
                plot_width,
                plot_height,
            ),
        ),
        (
            "plot5",
            "Short Only Excess Cumulative Returns",
            plot_cumulative_returns(
                result_dict["excess_ret_daily"]["short"],
                "Short Only Excess Cumulative Returns",
                plot_width,
                plot_height,
            ),
        ),
        (
            "plot6",
            "Long Passive Investment Cumulative Returns",
            plot_cumulative_returns(
                result_dict["ret_daily"]["passive"],
                "Long Passive Investment Cumulative Returns",
                plot_width,
                plot_height,
            ),
        ),
        (
            "plot7",
            "Short Passive Investment Cumulative Returns",
            plot_cumulative_returns(
                result_dict["ret_daily"]["short_passive"],
                "Short Passive Investment Cumulative Returns",
                plot_width,
                plot_height,
            ),
        ),
        ("plot8", "Layer Cumulative Returns", plot_layer_cumulative_returns(result_dict, plot_width, plot_height)),
        ("plot9", "Layer Annual Returns", plot_layer_annual_returns(result_dict, plot_width, plot_height)),
        ("plot10", "Layer Rank IC", plot_layer_rank_ic(result_dict, plot_width, plot_height)),
        ("plot11", "Rank IC Cumulative", plot_rank_ic_cumulative(result_dict["rank_ic_daily"], plot_width, plot_height)),
        ("plot12", "Rank IC Interval Analysis", plot_rank_ic_interval_analysis(result_dict["rank_ic_daily"], plot_width, plot_height)),
        ("plot13", "Factor Distribution", plot_factor_distribution(result_dict["factor"], plot_width, plot_height)),
        ("plot14", "Factor Autocorrelation", plot_factor_autocorrelation(result_dict["acf"], plot_width, plot_height)),
        ("plot15", "Long/Short Turnover Ratio", plot_turnover(result_dict["turnover_daily"]["ls"], "Long/Short Turnover Ratio", plot_width, plot_height)),
        ("plot16", "Long Only Turnover Ratio", plot_turnover(result_dict["turnover_daily"]["long"], "Long Only Turnover Ratio", plot_width, plot_height)),
        ("plot17", "Short Only Turnover Ratio", plot_turnover(result_dict["turnover_daily"]["short"], "Short Only Turnover Ratio", plot_width, plot_height)),
    ]

    try:
        doc_compiler = CS_Backtest_Doc()
        documentation_html = doc_compiler.compile_documentation(backtest_mode)
    except Exception:
        documentation_html = (
            '<div class="documentation-content"><p>Documentation not available.</p></div>'
        )

    output_dir = Path(output_dir or Path.cwd() / "saved_files")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "factor_info_dict": factor_info_dict,
        "fees": fees,
        "backtest_mode": backtest_mode,
        "backtest_params": backtest_params,
        "alpha_name": factor_column_name,
        "n_symbols": len(common_cols_list),
        "start_dt": start_dt,
        "end_dt": end_dt,
        "n_datetimes": n_datetimes,
        "freq": freq,
        "layers_use": layers_use,
        "lag": lag,
        "annual_days": annual_days,
    }

    generator = CSReportGenerator(result_dict, summary_df, config)
    generator.generate(plot_info, output_dir, documentation_html)
    _save_json(output_dir / "summary_df.json", _build_summary_payload(summary_df))

    return result_dict, summary_df


def run_time_series_backtest(
    df: Any,
    factor_column_name: str,
    datetime_column_name: str,
    symbol_column_name: str,
    raw_label_column_name: str,
    freq: str,
    backtest_mode: str = "long/short_threshold",
    backtest_params: dict = None,
    fees: List[float] = None,
    lag: int = 2,
    annual_days: int | None = None,
    factor_info_dict: dict | None = None,
    *,
    output_dir: str | Path | None = None,
) -> Tuple[Dict[str, Any], Any]:
    """Drop-in compatible TS backtest with optimized compute kernels."""
    if fees is None:
        fees = [0.0e-4, 2.0e-4, 5.0e-4]

    if backtest_params is None:
        if backtest_mode == "long/short_threshold":
            backtest_params = {"long_threshold_quantile": 0.8, "short_threshold_quantile": 0.2}
        elif backtest_mode == "gradual_long/short_threshold":
            backtest_params = {
                "end_short_quantile": 0.1,
                "start_short_quantile": 0.3,
                "start_long_quantile": 0.7,
                "end_long_quantile": 0.9,
            }
        elif backtest_mode == "long/short_normalization":
            backtest_params = {
                "normalization_method": "rank",
                "normalization_rolling_window_n": 252,
                "winsorize_method": "std",
                "winsorize_n": 3.0,
                "winsorize_rolling_window_n": 252,
                "squash_method": "tanh",
            }
        else:
            backtest_params = {}

    core = detect_core(df)
    if core == "pandas":
        processed_df = df.rename(
            columns={
                datetime_column_name: "datetime",
                symbol_column_name: "symbol",
                factor_column_name: "factor",
                raw_label_column_name: "label",
            }
        ).copy()

        processed_df["datetime"] = pd.to_datetime(processed_df["datetime"])
        processed_df = processed_df.sort_values("datetime")

        unique_symbols = processed_df["symbol"].unique()
        symbol = unique_symbols[0]
        if len(unique_symbols) > 1:
            raise ValueError(
                "Time series backtest only supports single symbol. "
                "Use cross-sectional backtest for multiple symbols."
            )

        factor_series = processed_df.set_index("datetime")["factor"]
        label_series = processed_df.set_index("datetime")["label"]

        common_idx = factor_series.index.intersection(label_series.index)
        if common_idx.empty:
            raise ValueError("No overlapping dates between factors and labels.")

        start_dt = common_idx.min()
        end_dt = common_idx.max()
        n_datetimes = int(len(common_idx))

        factor_input = factor_series.loc[common_idx]
        label_input = label_series.loc[common_idx]

    else:
        try:
            import polars as pl  # type: ignore
        except Exception as exc:
            raise RuntimeError("polars is required for polars inputs") from exc
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"Unsupported df type: {type(df)}")

        processed_df = df.rename(
            {
                datetime_column_name: "datetime",
                symbol_column_name: "symbol",
                factor_column_name: "factor",
                raw_label_column_name: "label",
            }
        ).select(["datetime", "symbol", "factor", "label"])

        dtype = processed_df.schema.get("datetime")
        if dtype == pl.Utf8:
            processed_df = processed_df.with_columns(
                pl.col("datetime").str.strptime(pl.Datetime, strict=False).alias("datetime")
            )
        else:
            processed_df = processed_df.with_columns(pl.col("datetime").cast(pl.Datetime).alias("datetime"))

        processed_df = processed_df.sort("datetime")

        unique_symbols = (
            processed_df.get_column("symbol").unique(maintain_order=True).to_list()
        )
        if not unique_symbols:
            raise ValueError("No symbol values found for time series backtest.")
        symbol = unique_symbols[0]
        if len(unique_symbols) > 1:
            raise ValueError(
                "Time series backtest only supports single symbol. "
                "Use cross-sectional backtest for multiple symbols."
            )

        factor_df = processed_df.select(["datetime", "factor"])
        label_df = processed_df.select(["datetime", "label"])

        common_dt = (
            factor_df.select("datetime")
            .join(label_df.select("datetime"), on="datetime", how="inner")
            .unique()
            .sort("datetime")
        )
        if common_dt.height == 0:
            raise ValueError("No overlapping dates between factors and labels.")

        start_dt = common_dt.select(pl.col("datetime").min()).item()
        end_dt = common_dt.select(pl.col("datetime").max()).item()
        n_datetimes = int(common_dt.height)

        factor_input = (
            factor_df.join(common_dt, on="datetime", how="semi")
            .select(["datetime", "factor"])
            .sort("datetime")
        )
        label_input = (
            label_df.join(common_dt, on="datetime", how="semi")
            .select(["datetime", "label"])
            .sort("datetime")
        )

    backtest_params_with_thresholds = backtest_params.copy()
    long_threshold = None
    short_threshold = None

    if backtest_mode == "long/short_threshold":
        if core == "pandas":
            long_threshold = factor_input.quantile(
                backtest_params.get("long_threshold_quantile", 0.8)
            )
            short_threshold = factor_input.quantile(
                backtest_params.get("short_threshold_quantile", 0.2)
            )
        else:
            import polars as pl  # type: ignore

            long_q = float(backtest_params.get("long_threshold_quantile", 0.8))
            short_q = float(backtest_params.get("short_threshold_quantile", 0.2))
            long_threshold = factor_input.select(
                pl.col("factor").quantile(long_q, interpolation="linear")
            ).item()
            short_threshold = factor_input.select(
                pl.col("factor").quantile(short_q, interpolation="linear")
            ).item()
        backtest_params_with_thresholds["long_threshold"] = long_threshold
        backtest_params_with_thresholds["short_threshold"] = short_threshold

    elif backtest_mode == "gradual_long/short_threshold":
        if core == "pandas":
            backtest_params_with_thresholds["end_short_threshold"] = factor_input.quantile(
                backtest_params.get("end_short_quantile", 0.1)
            )
            backtest_params_with_thresholds["start_short_threshold"] = factor_input.quantile(
                backtest_params.get("start_short_quantile", 0.3)
            )
            backtest_params_with_thresholds["start_long_threshold"] = factor_input.quantile(
                backtest_params.get("start_long_quantile", 0.7)
            )
            backtest_params_with_thresholds["end_long_threshold"] = factor_input.quantile(
                backtest_params.get("end_long_quantile", 0.9)
            )
        else:
            import polars as pl  # type: ignore

            esq = float(backtest_params.get("end_short_quantile", 0.1))
            ssq = float(backtest_params.get("start_short_quantile", 0.3))
            slq = float(backtest_params.get("start_long_quantile", 0.7))
            elq = float(backtest_params.get("end_long_quantile", 0.9))
            q_expr = pl.col("factor").quantile
            backtest_params_with_thresholds["end_short_threshold"] = factor_input.select(
                q_expr(esq, interpolation="linear")
            ).item()
            backtest_params_with_thresholds["start_short_threshold"] = factor_input.select(
                q_expr(ssq, interpolation="linear")
            ).item()
            backtest_params_with_thresholds["start_long_threshold"] = factor_input.select(
                q_expr(slq, interpolation="linear")
            ).item()
            backtest_params_with_thresholds["end_long_threshold"] = factor_input.select(
                q_expr(elq, interpolation="linear")
            ).item()
        long_threshold = backtest_params_with_thresholds["end_long_threshold"]
        short_threshold = backtest_params_with_thresholds["end_short_threshold"]

    backtest_params_with_thresholds = validate_ts_params(
        backtest_mode=backtest_mode,
        backtest_params=backtest_params_with_thresholds,
    )

    cfg_ts = {
        "factor": factor_input,
        "label": label_input,
        "freq": freq,
        "fees": fees,
        "backtest_mode": backtest_mode,
        "backtest_params": backtest_params_with_thresholds,
        "lag": lag,
        "annual_days": annual_days,
        "symbol": symbol,
    }

    ts_backtest = TS_Backtest(cfg_ts)
    result_dict, summary_df = ts_backtest.run()

    plot_width = 450
    plot_height = 400
    plot_info = [
        (
            "plot1",
            "Long/Short Cumulative Returns",
            plot_cumulative_returns(
                result_dict["ret_daily"]["ls"],
                "Long/Short Cumulative Returns",
                plot_width,
                plot_height,
            ),
        ),
        (
            "plot2",
            "Long Only Cumulative Returns",
            plot_cumulative_returns(
                result_dict["ret_daily"]["long"],
                "Long Only Cumulative Returns",
                plot_width,
                plot_height,
            ),
        ),
        (
            "plot3",
            "Long Only Excess Cumulative Returns",
            plot_cumulative_returns(
                result_dict["excess_ret_daily"]["long"],
                "Long Only Excess Cumulative Returns",
                plot_width,
                plot_height,
            ),
        ),
        (
            "plot4",
            "Short Only Cumulative Returns",
            plot_cumulative_returns(
                result_dict["ret_daily"]["short"],
                "Short Only Cumulative Returns",
                plot_width,
                plot_height,
            ),
        ),
        (
            "plot5",
            "Short Only Excess Cumulative Returns",
            plot_cumulative_returns(
                result_dict["excess_ret_daily"]["short"],
                "Short Only Excess Cumulative Returns",
                plot_width,
                plot_height,
            ),
        ),
        (
            "plot6",
            "Long Passive Investment Cumulative Returns",
            plot_cumulative_returns(
                result_dict["ret_daily"]["passive"],
                "Long Passive Investment Cumulative Returns",
                plot_width,
                plot_height,
            ),
        ),
        (
            "plot7",
            "Short Passive Investment Cumulative Returns",
            plot_cumulative_returns(
                result_dict["ret_daily"]["short_passive"],
                "Short Passive Investment Cumulative Returns",
                plot_width,
                plot_height,
            ),
        ),
        ("plot8", "Rank IC Cumulative", plot_rank_ic_cumulative(result_dict["rank_ic_daily"], plot_width, plot_height)),
        ("plot9", "Rank IC Interval Analysis", plot_rank_ic_interval_analysis(result_dict["rank_ic_daily"], plot_width, plot_height)),
        ("plot10", "Factor Distribution", plot_factor_distribution(result_dict["factor"], plot_width, plot_height)),
        ("plot11", "Factor Autocorrelation", plot_factor_autocorrelation(result_dict["acf"], plot_width, plot_height)),
        ("plot12", "Long/Short Turnover", plot_turnover(result_dict["turnover_daily"]["ls"], "Long/Short Turnover", plot_width, plot_height)),
        ("plot13", "Long Only Turnover", plot_turnover(result_dict["turnover_daily"]["long"], "Long Only Turnover", plot_width, plot_height)),
        ("plot14", "Short Only Turnover", plot_turnover(result_dict["turnover_daily"]["short"], "Short Only Turnover", plot_width, plot_height)),
    ]

    try:
        doc_compiler = TS_Backtest_Doc()
        documentation_html = doc_compiler.compile_documentation(backtest_mode)
    except Exception:
        documentation_html = (
            '<div class="documentation-content"><p>Documentation not available.</p></div>'
        )

    output_dir = Path(output_dir or Path.cwd() / "saved_files")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "factor_info_dict": factor_info_dict,
        "fees": fees,
        "backtest_mode": backtest_mode,
        "backtest_params": backtest_params,
        "alpha_name": factor_column_name,
        "symbol": symbol,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "n_datetimes": n_datetimes,
        "freq": freq,
        "lag": lag,
        "annual_days": annual_days,
    }

    thresholds: Dict[str, Any] = {}
    if long_threshold is not None:
        thresholds["long_threshold"] = long_threshold
    if short_threshold is not None:
        thresholds["short_threshold"] = short_threshold

    generator = TSReportGenerator(result_dict, summary_df, config)
    generator.generate(plot_info, output_dir, thresholds, documentation_html)
    _save_json(output_dir / "summary_df.json", _build_summary_payload(summary_df))

    return result_dict, summary_df
