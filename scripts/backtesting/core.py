"""Backend (pandas/polars) detection helpers."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

CoreName = Literal["pandas", "polars"]


def _polars_types() -> tuple[type | None, type | None]:
    try:
        import polars as pl  # type: ignore
    except Exception:
        return None, None
    return pl.DataFrame, pl.Series


def is_pandas_df(obj: Any) -> bool:
    return isinstance(obj, pd.DataFrame)


def is_pandas_series(obj: Any) -> bool:
    return isinstance(obj, pd.Series)


def is_polars_df(obj: Any) -> bool:
    df_type, _ = _polars_types()
    return df_type is not None and isinstance(obj, df_type)


def is_polars_series(obj: Any) -> bool:
    _, series_type = _polars_types()
    return series_type is not None and isinstance(obj, series_type)


def detect_core(value: Any) -> CoreName:
    if is_pandas_df(value) or is_pandas_series(value):
        return "pandas"
    if is_polars_df(value) or is_polars_series(value):
        return "polars"
    raise TypeError(f"Unsupported input type: {type(value)}")

