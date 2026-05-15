"""Plotting functions for backtesting visualization.

These helpers are backend-aware:
- pandas mode: accepts `pd.Series`/`pd.DataFrame` outputs (legacy contract).
- polars mode: accepts `pl.DataFrame` time-series outputs from engines.

The plotting layer avoids creating pandas objects when inputs are polars.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import holoviews as hv

from .core import is_pandas_series, is_polars_df

hv.extension("bokeh")


ACF_LAGS = list(range(1, 21)) + list(range(25, 51, 5)) + [100, 150, 200]

POS_TYPES = ["ls", "long", "short", "passive"]
PREFIX_MAP = {"ls": "long_short", "long": "long", "short": "short", "passive": "passive"}

FEE_COLORS = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#34495e",
    "#16a085",
]

LAYER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]


def _cumsum_skipna(values: np.ndarray) -> np.ndarray:
    out = np.nancumsum(values)
    nan_mask = np.isnan(values)
    if np.any(nan_mask):
        out[nan_mask] = np.nan
    return out


def _timeseries_xy(obj: Any, *, time_col: str | None = None) -> Tuple[List, np.ndarray]:
    if is_pandas_series(obj):
        x = list(obj.index)
        y = np.asarray(obj.to_numpy(dtype=np.float64, copy=False), dtype=np.float64)
        return x, y

    if is_polars_df(obj):
        cols = obj.columns
        if time_col and time_col in cols:
            t_col = time_col
        elif "date" in cols:
            t_col = "date"
        elif "datetime" in cols:
            t_col = "datetime"
        else:
            t_col = cols[0]

        if "value" in cols:
            v_col = "value"
        else:
            value_cols = [c for c in cols if c != t_col]
            v_col = value_cols[0] if value_cols else t_col

        x = obj.get_column(t_col).to_list()
        y = np.asarray(obj.get_column(v_col).to_numpy(), dtype=np.float64)
        return x, y

    y = np.asarray(obj, dtype=np.float64)
    x = list(range(int(y.shape[0])))
    return x, y


def plot_cumulative_returns(
    ret_daily: Mapping[str, Any],
    title: str,
    width: int,
    height: int,
) -> hv.Overlay:
    curves = []
    for i, (fee_label, series_like) in enumerate(ret_daily.items()):
        x, y = _timeseries_xy(series_like, time_col="date")
        y_cum = _cumsum_skipna(y)
        color = FEE_COLORS[i % len(FEE_COLORS)]
        curve = hv.Curve(
            (x, y_cum),
            kdims=["date"],
            vdims=["cumulative_return"],
            label=str(fee_label),
        ).opts(color=color, line_width=2.5, tools=["hover"])
        curves.append(curve)

    return hv.Overlay(curves).opts(
        width=width,
        height=height,
        xlabel="Date",
        ylabel="Cumulative Return",
        title=title,
        legend_position="top_left",
        show_grid=True,
    )


def plot_turnover(
    turnover_daily: Any,
    title: str,
    width: int,
    height: int,
) -> hv.Overlay:
    x, y = _timeseries_xy(turnover_daily, time_col="date")
    scatter = hv.Scatter((x, y), kdims=["date"], vdims=["ratio"], label="Turnover").opts(
        width=width,
        height=height,
        xlabel="Date",
        ylabel="Turnover Ratio",
        title=title,
        tools=["hover"],
        show_grid=True,
        size=5,
        color="#1f77b4",
    )

    if not x:
        return scatter

    mean_val = float(np.nanmean(y))
    min_val = float(np.nanmin(y))
    max_val = float(np.nanmax(y))
    x_range = [x[0], x[-1]]

    mean_line = hv.Curve(
        (x_range, [mean_val, mean_val]),
        kdims=["date"],
        vdims=["ratio"],
        label=f"Mean: {round(mean_val * 100, 1)}%",
    ).opts(
        color="#d62728",
        line_width=2,
        line_dash="dashed",
        tools=["hover"],
    )
    min_line = hv.Curve(
        (x_range, [min_val, min_val]),
        kdims=["date"],
        vdims=["ratio"],
        label=f"Min: {round(min_val * 100, 1)}%",
    ).opts(
        color="#2ca02c",
        line_width=2,
        line_dash="dotted",
        tools=["hover"],
    )
    max_line = hv.Curve(
        (x_range, [max_val, max_val]),
        kdims=["date"],
        vdims=["ratio"],
        label=f"Max: {round(max_val * 100, 1)}%",
    ).opts(
        color="#ff7f0e",
        line_width=2,
        line_dash="dotted",
        tools=["hover"],
    )

    return (scatter * mean_line * min_line * max_line).opts(legend_position="top_left")


def plot_rank_ic_cumulative(rank_ic_daily: Any, width: int, height: int) -> hv.Curve:
    x, y = _timeseries_xy(rank_ic_daily, time_col="date")
    y_cum = _cumsum_skipna(y)
    return hv.Curve((x, y_cum), kdims=["date"], vdims=["cumulative_rank_ic"]).opts(
        width=width,
        height=height,
        xlabel="Date",
        ylabel="Cumulative Rank IC",
        title="Rank IC Cumulative",
        color="#1f77b4",
        line_width=2.5,
        tools=["hover"],
        show_grid=True,
    )


def plot_rank_ic_interval_analysis(
    rank_ic_daily: Any,
    width: int,
    height: int,
    mean_decimals: int = 5,
) -> hv.Overlay:
    x, y = _timeseries_xy(rank_ic_daily, time_col="date")
    mask = np.isfinite(y)
    x = [x[i] for i in np.flatnonzero(mask).tolist()]
    y = y[mask]

    if not x:
        bars = hv.Bars(([], []), kdims=["timeperiod"], vdims=["rank_ic"]).opts(
            width=width,
            height=height,
            xlabel="Time Period",
            ylabel="Rank IC",
            title="Rank IC Interval Analysis (no data)",
            tools=["hover"],
            show_grid=True,
            xrotation=40,
        )
        return bars

    intervals = 15
    step = max(1, len(y) // intervals)
    idx = list(range(0, len(y), step))
    if len(idx) < 2:
        bars = hv.Bars(([], []), kdims=["timeperiod"], vdims=["rank_ic"]).opts(
            width=width,
            height=height,
            xlabel="Time Period",
            ylabel="Rank IC",
            title=f"Rank IC Interval Analysis (Mean: {round(float(np.mean(y)), mean_decimals)})",
            tools=["hover"],
            show_grid=True,
            xrotation=40,
        )
        return bars

    cumsum = np.cumsum(y)
    interval_vals = (cumsum[idx][1:] - cumsum[idx][:-1]) / float(step)
    interval_times = [x[i].strftime("%Y-%m-%d") for i in idx[1:]]

    mean_val = float(np.mean(y))
    bars = hv.Bars((interval_times, interval_vals), kdims=["timeperiod"], vdims=["rank_ic"]).opts(
        width=width,
        height=height,
        xlabel="Time Period",
        ylabel="Rank IC",
        title=f"Rank IC Interval Analysis (Mean: {round(mean_val, mean_decimals)})",
        tools=["hover"],
        show_grid=True,
        xrotation=40,
    )

    mean_line = hv.Curve(
        ([interval_times[0], interval_times[-1]], [mean_val, mean_val]),
        kdims=["timeperiod"],
        vdims=["rank_ic"],
        label=f"Mean: {mean_val:.4f}",
    ).opts(color="#d62728", line_width=2, line_dash="dashed", tools=["hover"])

    return (bars * mean_line).opts(legend_position="top_right")


def plot_factor_distribution(
    factor: Any,
    width: int,
    height: int,
    bins: int = 50,
) -> hv.Bars:
    _, y = _timeseries_xy(factor)
    finite = y[np.isfinite(y)]
    frequencies, edges = np.histogram(finite, bins=bins)

    if finite.size:
        mean_val = float(np.mean(finite))
        centered = finite - mean_val
        var = float(np.mean(centered**2))
        if var > 0:
            skew_val = float(np.mean(centered**3) / (var ** 1.5))
            kurt_val = float(np.mean(centered**4) / (var**2) - 3.0)
        else:
            skew_val, kurt_val = 0.0, 0.0
    else:
        mean_val, skew_val, kurt_val = 0.0, 0.0, 0.0

    title_text = (
        "Factor Distribution\n"
        f"(mean={round(mean_val, 2)}, skew={round(skew_val, 2)}, kurtosis={round(kurt_val, 2)})"
    )

    return hv.Bars(
        (edges[:-1], frequencies),
        kdims=["factor_value"],
        vdims=["frequency"],
    ).opts(
        width=width,
        height=height,
        xlabel="Factor Value",
        ylabel="Frequency",
        title=title_text,
        tools=["hover"],
        show_grid=True,
        color="#1f77b4",
    )


def plot_factor_autocorrelation(acf: Dict[int, float], width: int, height: int) -> hv.Overlay:
    lags = [lag for lag in ACF_LAGS if lag in acf]
    coeffs = [acf.get(lag, np.nan) for lag in lags]
    periods = [str(lag) for lag in lags]

    bars = hv.Bars((periods, coeffs), kdims=["periods"], vdims=["coefficient"]).opts(
        width=width,
        height=height,
        xlabel="Lag (Periods)",
        ylabel="Autocorrelation Coefficient",
        title="Factor Autocorrelation",
        tools=["hover"],
        show_grid=True,
        color="#2ca02c",
    )
    if not periods:
        return bars

    ref_line = hv.Curve(
        ([periods[0], periods[-1]], [0.5, 0.5]),
        kdims=["periods"],
        vdims=["coefficient"],
        label="Reference (0.5)",
    ).opts(color="#d62728", line_width=2, line_dash="dashed", tools=["hover"])

    return (bars * ref_line).opts(legend_position="top_right")


def plot_layer_cumulative_returns(
    results: Dict[str, Any],
    width: int = 800,
    height: int = 500,
) -> hv.Overlay:
    curves = []
    layer_series = results.get("ret_layer_daily", {})
    for i, (layer, series_like) in enumerate(layer_series.items()):
        x, y = _timeseries_xy(series_like, time_col="date")
        y_cum = _cumsum_skipna(y)
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        curves.append(
            hv.Curve((x, y_cum), kdims=["date"], vdims=["cumulative_return"], label=str(layer)).opts(
                color=color, line_width=2.5, tools=["hover"]
            )
        )

    return hv.Overlay(curves).opts(
        width=width,
        height=height,
        xlabel="Date",
        ylabel="Cumulative Return",
        title="Layer Cumulative Returns",
        legend_position="top_left",
        show_grid=True,
    )


def plot_layer_annual_returns(
    results: Dict[str, Any],
    width: int = 800,
    height: int = 500,
) -> hv.Bars:
    layer_series = results.get("ret_layer_daily", {})
    num_years = float(results.get("time_metrics", {}).get("num_years", 1) or 1.0)

    layers: List[str] = []
    annual_returns: List[float] = []
    colors: List[str] = []

    for i, (layer, series_like) in enumerate(layer_series.items()):
        _, y = _timeseries_xy(series_like, time_col="date")
        layers.append(str(layer))
        annual_returns.append(float(np.nansum(y) / num_years))
        colors.append(LAYER_COLORS[i % len(LAYER_COLORS)])

    bars = hv.Bars(
        list(zip(layers, annual_returns, colors)),
        kdims=["layer"],
        vdims=["annual_return", "color"],
    ).opts(
        width=width,
        height=height,
        xlabel="Layer",
        ylabel="Annual Return",
        title="Layer Annual Returns",
        tools=["hover"],
        show_grid=True,
        color=hv.dim("color"),
    )
    return bars


def plot_layer_rank_ic(
    results: Dict[str, Any],
    width: int = 800,
    height: int = 500,
) -> hv.Bars:
    layer_series = results.get("rank_ic_layer", {})

    layers: List[str] = []
    means: List[float] = []
    colors: List[str] = []

    for i, (layer, series_like) in enumerate(layer_series.items()):
        _, y = _timeseries_xy(series_like, time_col="datetime")
        layers.append(str(layer))
        means.append(float(np.nanmean(y)) if y.size else np.nan)
        colors.append(LAYER_COLORS[i % len(LAYER_COLORS)])

    bars = hv.Bars(
        list(zip(layers, means, colors)),
        kdims=["layer"],
        vdims=["rank_ic", "color"],
    ).opts(
        width=width,
        height=height,
        xlabel="Layer",
        ylabel="Rank Information Coefficient",
        title="Layer Rank IC",
        tools=["hover"],
        show_grid=True,
        color=hv.dim("color"),
    )
    return bars


__all__ = [
    "ACF_LAGS",
    "POS_TYPES",
    "PREFIX_MAP",
    "FEE_COLORS",
    "LAYER_COLORS",
    "plot_cumulative_returns",
    "plot_turnover",
    "plot_rank_ic_cumulative",
    "plot_rank_ic_interval_analysis",
    "plot_factor_distribution",
    "plot_factor_autocorrelation",
    "plot_layer_cumulative_returns",
    "plot_layer_annual_returns",
    "plot_layer_rank_ic",
]
