"""Array builders: pandas/polars inputs -> array containers for Numba kernels."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd

from ..core import detect_core, is_pandas_df, is_pandas_series, is_polars_df
from .input_builders import build_symbol_csr_from_codes
from ..kernels.panel_build import compact_cs_panel_long
from ..kernels.panel_features import cs_lag_and_ref_price, ts_lag_and_ref_price
from ..structures import CSPanelArrays, TSPanelArrays


def _align_cs_inputs(
    factor: pd.DataFrame, label: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common_index = factor.index.intersection(label.index)
    common_cols = factor.columns.intersection(label.columns)
    if not len(common_index) or not len(common_cols):
        raise ValueError("No common index between factor and label")

    common_index = common_index.sort_values()
    common_cols = common_cols.sort_values()
    return factor.loc[common_index, common_cols], label.loc[common_index, common_cols]


def _build_day_offsets_from_sorted_days(day_dt64: np.ndarray) -> tuple[list, np.ndarray]:
    if day_dt64.size == 0:
        return [], np.array([0], dtype=np.int64)

    day_change = day_dt64[1:] != day_dt64[:-1]
    day_starts = np.empty(int(day_change.sum()) + 1, dtype=np.int64)
    day_starts[0] = 0
    day_starts[1:] = np.flatnonzero(day_change).astype(np.int64, copy=False) + 1

    day_offsets = np.empty(day_starts.size + 1, dtype=np.int64)
    day_offsets[:-1] = day_starts
    day_offsets[-1] = day_dt64.shape[0]

    day_list = day_dt64[day_starts].astype(object).tolist()
    return day_list, day_offsets


def _dt64ns_to_py_datetimes(dt_dt64: np.ndarray) -> list:
    if dt_dt64.size == 0:
        return []
    try:
        dt_ns = dt_dt64.astype("datetime64[ns]", copy=False)
    except TypeError:
        dt_ns = np.asarray(dt_dt64, dtype="datetime64[ns]")
    # NumPy may materialize `datetime64[ns]` as raw int nanoseconds; convert to
    # microsecond resolution (Python datetime max) before converting to objects.
    return dt_ns.astype("datetime64[us]").astype(object).tolist()


def _dt64d_to_py_dates(dt_dt64: np.ndarray) -> list:
    if dt_dt64.size == 0:
        return []
    try:
        dt_d = dt_dt64.astype("datetime64[D]", copy=False)
    except TypeError:
        dt_d = np.asarray(dt_dt64, dtype="datetime64[D]")
    return dt_d.astype(object).tolist()


def build_cs_panel_arrays(
    factor: Any,
    label: Any,
    *,
    lag: int,
) -> CSPanelArrays:
    """Build CS long arrays and grouping metadata.

    The output ordering matches the legacy engine:
    datetime-major, then symbol-major, after filtering invalid labels.
    """
    core = detect_core(factor)
    if core != detect_core(label):
        raise TypeError("factor/label backends must match")

    if core == "pandas":
        if not is_pandas_df(factor) or not is_pandas_df(label):
            raise TypeError("Expected pandas DataFrames for factor/label")
        factor_aligned, label_aligned = _align_cs_inputs(factor, label)
        dt_index = factor_aligned.index
        sym_list = list(label_aligned.columns.astype(str))

        factor_mat = factor_aligned.to_numpy(dtype=np.float64, copy=False)
        label_mat = label_aligned.to_numpy(dtype=np.float64, copy=False)

    elif core == "polars":
        if not is_polars_df(factor) or not is_polars_df(label):
            raise TypeError("Expected polars DataFrames for factor/label")
        try:
            import polars as pl  # type: ignore
        except Exception as exc:
            raise RuntimeError("polars is required for polars inputs") from exc

        if "datetime" not in factor.columns or "datetime" not in label.columns:
            raise ValueError("polars CS factor/label must include a 'datetime' column")

        factor_cols = [c for c in factor.columns if c != "datetime"]
        label_cols = [c for c in label.columns if c != "datetime"]
        common_cols = sorted(set(factor_cols).intersection(label_cols))
        if not common_cols:
            raise ValueError("No common symbols between factor and label")

        common_dt = (
            factor.select("datetime")
            .join(label.select("datetime"), on="datetime", how="inner")
            .unique()
            .sort("datetime")
        )
        if common_dt.height == 0:
            raise ValueError("No common datetimes between factor and label")

        factor_aligned = (
            factor.join(common_dt, on="datetime", how="semi")
            .sort("datetime")
            .select(["datetime", *common_cols])
        )
        label_aligned = (
            label.join(common_dt, on="datetime", how="semi")
            .sort("datetime")
            .select(["datetime", *common_cols])
        )

        dt_index = factor_aligned.get_column("datetime").to_numpy()
        sym_list = common_cols

        factor_mat = factor_aligned.select(common_cols).to_numpy()
        label_mat = label_aligned.select(common_cols).to_numpy()
        factor_mat = np.asarray(factor_mat, dtype=np.float64)
        label_mat = np.asarray(label_mat, dtype=np.float64)

    else:  # pragma: no cover
        raise TypeError(f"Unsupported core: {core}")

    valid_label_mask = np.isfinite(label_mat) & (label_mat > -1.0)

    row_counts = valid_label_mask.sum(axis=1).astype(np.int64, copy=False)
    keep_dt = row_counts > 0
    if not np.any(keep_dt):
        empty = np.array([0], dtype=np.int64)
        return CSPanelArrays(
            dt_list=[],
            dt_offsets_rows=empty,
            dt_trading_dates=[],
            day_list=[],
            day_offsets_dt=empty,
            sym_list=[],
            sym_csr=build_symbol_csr_from_codes(np.empty(0, dtype=np.int32), n_symbols=0),
            sym_code=np.empty(0, dtype=np.int32),
            factor=np.empty(0, dtype=np.float64),
            label=np.empty(0, dtype=np.float64),
            ref_price=np.empty(0, dtype=np.float64),
        )

    factor_mat = factor_mat[keep_dt]
    label_mat = label_mat[keep_dt]
    valid_label_mask = valid_label_mask[keep_dt]
    row_counts = row_counts[keep_dt]
    dt_index = np.asarray(dt_index)[keep_dt]

    dt_offsets_rows = np.empty(row_counts.size + 1, dtype=np.int64)
    dt_offsets_rows[0] = 0
    dt_offsets_rows[1:] = np.cumsum(row_counts, dtype=np.int64)
    n_rows = int(dt_offsets_rows[-1])

    n_symbols = int(label_mat.shape[1])
    sym_code = np.empty(n_rows, dtype=np.int32)
    factor_raw = np.empty(n_rows, dtype=np.float64)
    label_long = np.empty(n_rows, dtype=np.float64)
    compact_cs_panel_long(
        factor_mat,
        label_mat,
        valid_label_mask,
        dt_offsets_rows,
        sym_code,
        factor_raw,
        label_long,
    )

    sym_csr = build_symbol_csr_from_codes(
        sym_code.astype(np.int64, copy=False), n_symbols=n_symbols
    )

    factor_lagged = np.empty(n_rows, dtype=np.float64)
    ref_price = np.empty(n_rows, dtype=np.float64)
    cs_lag_and_ref_price(
        factor_raw,
        label_long,
        sym_csr.offsets,
        sym_csr.rows,
        int(lag),
        factor_lagged,
        ref_price,
    )

    if isinstance(dt_index, pd.Index):
        dt_dt64 = dt_index.to_numpy(dtype="datetime64[ns]", copy=False)
    else:
        dt_dt64 = np.asarray(dt_index, dtype="datetime64[ns]")
    dt_list = _dt64ns_to_py_datetimes(dt_dt64)

    day_dt64 = dt_dt64.astype("datetime64[D]")
    dt_trading_dates = _dt64d_to_py_dates(day_dt64)
    day_list, day_offsets_dt = _build_day_offsets_from_sorted_days(day_dt64)

    return CSPanelArrays(
        dt_list=dt_list,
        dt_offsets_rows=dt_offsets_rows,
        dt_trading_dates=dt_trading_dates,
        day_list=day_list,
        day_offsets_dt=day_offsets_dt,
        sym_list=sym_list,
        sym_csr=sym_csr,
        sym_code=sym_code,
        factor=factor_lagged,
        label=label_long,
        ref_price=ref_price,
    )


def build_ts_panel_arrays(
    factor: Any,
    label: Any,
    *,
    lag: int,
) -> TSPanelArrays:
    """Build TS arrays and day grouping metadata."""
    core = detect_core(factor)
    if core != detect_core(label):
        raise TypeError("factor/label backends must match")

    if core == "pandas":
        if not is_pandas_series(factor) or not is_pandas_series(label):
            raise TypeError("Expected pandas Series for factor/label")
        common_index = factor.index.intersection(label.index)
        if not len(common_index):
            raise ValueError("No common index between factor and label")

        common_index = common_index.sort_values()
        factor_s = factor.loc[common_index]
        label_s = label.loc[common_index]

        factor_raw = factor_s.to_numpy(dtype=np.float64, copy=False)
        label_raw = label_s.to_numpy(dtype=np.float64, copy=False)
        dt_index = common_index

    elif core == "polars":
        if not is_polars_df(factor) or not is_polars_df(label):
            raise TypeError("Expected polars DataFrames for factor/label")
        try:
            import polars as pl  # type: ignore
        except Exception as exc:
            raise RuntimeError("polars is required for polars inputs") from exc

        if "datetime" not in factor.columns or "datetime" not in label.columns:
            raise ValueError("polars TS factor/label must include a 'datetime' column")

        factor_vals = [c for c in factor.columns if c != "datetime"]
        label_vals = [c for c in label.columns if c != "datetime"]
        if len(factor_vals) != 1 or len(label_vals) != 1:
            raise ValueError("polars TS factor/label must have exactly one value column")
        factor_col = factor_vals[0]
        label_col = label_vals[0]

        common_dt = (
            factor.select("datetime")
            .join(label.select("datetime"), on="datetime", how="inner")
            .unique()
            .sort("datetime")
        )
        if common_dt.height == 0:
            raise ValueError("No common datetimes between factor and label")

        factor_aligned = (
            factor.join(common_dt, on="datetime", how="semi")
            .sort("datetime")
            .select(["datetime", factor_col])
        )
        label_aligned = (
            label.join(common_dt, on="datetime", how="semi")
            .sort("datetime")
            .select(["datetime", label_col])
        )

        factor_raw = np.asarray(factor_aligned.get_column(factor_col).to_numpy(), dtype=np.float64)
        label_raw = np.asarray(label_aligned.get_column(label_col).to_numpy(), dtype=np.float64)
        dt_index = factor_aligned.get_column("datetime").to_numpy()

    else:  # pragma: no cover
        raise TypeError(f"Unsupported core: {core}")

    valid_label = np.isfinite(label_raw) & (label_raw > -1.0)
    if not np.any(valid_label):
        empty = np.array([0], dtype=np.int64)
        return TSPanelArrays(
            dt_list=[],
            trading_dates=[],
            day_list=[],
            day_offsets_dt=empty,
            factor=np.empty(0, dtype=np.float64),
            label=np.empty(0, dtype=np.float64),
            ref_price=np.empty(0, dtype=np.float64),
        )

    factor_raw = factor_raw[valid_label]
    label_raw = label_raw[valid_label]
    dt_index = np.asarray(dt_index)[valid_label]

    n = int(factor_raw.shape[0])
    factor_lagged = np.empty(n, dtype=np.float64)
    ref_price = np.empty(n, dtype=np.float64)
    ts_lag_and_ref_price(factor_raw, label_raw, int(lag), factor_lagged, ref_price)

    if isinstance(dt_index, pd.Index):
        dt_dt64 = dt_index.to_numpy(dtype="datetime64[ns]", copy=False)
    else:
        dt_dt64 = np.asarray(dt_index, dtype="datetime64[ns]")
    dt_list = _dt64ns_to_py_datetimes(dt_dt64)

    day_dt64 = dt_dt64.astype("datetime64[D]")
    trading_dates = _dt64d_to_py_dates(day_dt64)
    day_list, day_offsets_dt = _build_day_offsets_from_sorted_days(day_dt64)

    return TSPanelArrays(
        dt_list=dt_list,
        trading_dates=trading_dates,
        day_list=day_list,
        day_offsets_dt=day_offsets_dt,
        factor=factor_lagged,
        label=label_raw,
        ref_price=ref_price,
    )
