"""Parameter validation utilities for backtesting."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def validate_cs_params(
    backtest_mode: str,
    backtest_params: Dict,
    layers_use: int,
    df: Any,
    datetime_col: str,
    symbol_col: str
) -> Dict:
    """
    Validate cross-sectional backtest parameters.

    Args:
        backtest_mode: Backtest mode string
        backtest_params: Dictionary of backtest parameters
        layers_use: Number of layers to use
        df: Source DataFrame
        datetime_col: Datetime column name
        symbol_col: Symbol column name

    Returns:
        Validated and possibly defaulted backtest_params dict

    Raises:
        ValueError: If parameters are invalid
    """
    if backtest_params is None:
        if backtest_mode == "long/short_layers":
            backtest_params = {"long_layer_index": [1], "short_layer_index": [layers_use]}
        elif backtest_mode == "long/short_normalization":
            backtest_params = {
                "normalization_method": "rank",
                "winsorize_method": "std",
                "winsorize_n": 3.0
            }
        else:
            raise ValueError(f"Unknown backtest_mode: {backtest_mode}")

    if backtest_mode == "long/short_layers":
        if "long_layer_index" not in backtest_params:
            raise ValueError(
                "backtest_params must contain 'long_layer_index' for 'long/short_layers' mode"
            )
        if "short_layer_index" not in backtest_params:
            raise ValueError(
                "backtest_params must contain 'short_layer_index' for 'long/short_layers' mode"
            )

        long_indices = backtest_params["long_layer_index"]
        short_indices = backtest_params["short_layer_index"]

        if not isinstance(long_indices, list) or not isinstance(short_indices, list):
            raise ValueError("'long_layer_index' and 'short_layer_index' must be lists")

        long_set = set(long_indices)
        short_set = set(short_indices)
        if long_set.intersection(short_set):
            duplicates = long_set.intersection(short_set)
            raise ValueError(
                f"Duplicate indices found between long_layer_index and "
                f"short_layer_index: {duplicates}"
            )

        all_indices = long_indices + short_indices
        if any(idx < 1 or idx > layers_use for idx in all_indices):
            raise ValueError(
                f"All layer indices must be between 1 and {layers_use}, got: {all_indices}"
            )

        if isinstance(df, pd.DataFrame):
            symbols_per_datetime = df.groupby(datetime_col)[symbol_col].nunique()
            min_symbols = int(symbols_per_datetime.min())
        else:
            try:
                import polars as pl  # type: ignore
            except Exception as exc:
                raise TypeError(
                    "df must be a pandas or polars DataFrame for CS validation"
                ) from exc

            if not isinstance(df, pl.DataFrame):
                raise TypeError(
                    "df must be a pandas or polars DataFrame for CS validation"
                )

            min_symbols = (
                df.group_by(datetime_col)
                .agg(pl.col(symbol_col).n_unique().alias("n_symbols"))
                .select(pl.col("n_symbols").min())
                .item()
            )
            min_symbols = int(min_symbols) if min_symbols is not None else 0
        if min_symbols < layers_use:
            raise ValueError(
                f"At least one timestamp has fewer than {layers_use} "
                f"unique symbols (minimum found: {min_symbols}). "
                f"Cannot create {layers_use} layers."
            )

    elif backtest_mode == "long/short_normalization":
        if "normalization_method" not in backtest_params:
            raise ValueError(
                "backtest_params must contain 'normalization_method' for "
                "'long/short_normalization' mode"
            )

        norm_method = backtest_params["normalization_method"]
        valid_methods = ["minmax", "zscore", "rank", "demean"]
        if norm_method not in valid_methods:
            raise ValueError(
                f"normalization_method must be one of {valid_methods}, got: {norm_method}"
            )

        if 'winsorize_method' in backtest_params or 'winsorize_n' in backtest_params:
            winsorize_method = backtest_params.get('winsorize_method', 'std')
            winsorize_n = backtest_params.get('winsorize_n', 3.0)

            valid_winsorize_methods = ['std', 'mad', 'quantile']
            if winsorize_method not in valid_winsorize_methods:
                raise ValueError(
                    f"winsorize_method must be one of {valid_winsorize_methods}, "
                    f"got: {winsorize_method}"
                )

            if winsorize_method == 'quantile':
                if not (0 < winsorize_n < 0.5):
                    raise ValueError(
                        f"For 'quantile' method, winsorize_n must be between 0 and 0.5, "
                        f"got: {winsorize_n}"
                    )
            elif winsorize_method in ['std', 'mad']:
                if winsorize_n <= 0:
                    raise ValueError(
                        f"For '{winsorize_method}' method, winsorize_n must be positive, "
                        f"got: {winsorize_n}"
                    )
    else:
        raise ValueError(
            f"backtest_mode must be either 'long/short_layers' or "
            f"'long/short_normalization', got: {backtest_mode}"
        )

    return backtest_params


def validate_ts_params(
    backtest_mode: str,
    backtest_params: Dict
) -> Dict:
    """
    Validate time-series backtest parameters.

    Args:
        backtest_mode: Backtest mode string
        backtest_params: Dictionary of backtest parameters

    Returns:
        Validated and possibly defaulted backtest_params dict

    Raises:
        ValueError: If parameters are invalid
    """
    if backtest_params is None:
        if backtest_mode == "long/short_threshold":
            backtest_params = {"long_quantile": 0.7, "short_quantile": 0.3}
        elif backtest_mode == "gradual_long/short_threshold":
            backtest_params = {"long_quantile": 0.7, "short_quantile": 0.3}
        elif backtest_mode == "long/short_normalization":
            backtest_params = {
                "normalization_method": "zscore",
                "window": 20,
                "winsorize_method": "std",
                "winsorize_n": 3.0
            }
        elif backtest_mode == "weights":
            backtest_params = {}
        else:
            raise ValueError(f"Unknown backtest_mode: {backtest_mode}")

    if backtest_mode in ["long/short_threshold", "gradual_long/short_threshold"]:
        long_q = backtest_params.get("long_quantile", 0.7)
        short_q = backtest_params.get("short_quantile", 0.3)

        if not (0 < long_q < 1):
            raise ValueError(f"long_quantile must be between 0 and 1, got: {long_q}")
        if not (0 < short_q < 1):
            raise ValueError(f"short_quantile must be between 0 and 1, got: {short_q}")
        if long_q <= short_q:
            raise ValueError(
                f"long_quantile ({long_q}) must be greater than short_quantile ({short_q})"
            )

    elif backtest_mode == "long/short_normalization":
        if "normalization_method" not in backtest_params:
            raise ValueError(
                "backtest_params must contain 'normalization_method' for "
                "'long/short_normalization' mode"
            )

        norm_method = backtest_params["normalization_method"]
        valid_methods = ["minmax", "zscore", "rank", "demean"]
        if norm_method not in valid_methods:
            raise ValueError(
                f"normalization_method must be one of {valid_methods}, got: {norm_method}"
            )

        if "window" not in backtest_params:
            raise ValueError(
                "backtest_params must contain 'window' for time-series "
                "'long/short_normalization' mode"
            )

        window = backtest_params["window"]
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got: {window}")

        if 'winsorize_method' in backtest_params or 'winsorize_n' in backtest_params:
            winsorize_method = backtest_params.get('winsorize_method', 'std')
            winsorize_n = backtest_params.get('winsorize_n', 3.0)

            valid_winsorize_methods = ['std', 'mad', 'quantile']
            if winsorize_method not in valid_winsorize_methods:
                raise ValueError(
                    f"winsorize_method must be one of {valid_winsorize_methods}, "
                    f"got: {winsorize_method}"
                )

            if winsorize_method == 'quantile':
                if not (0 < winsorize_n < 0.5):
                    raise ValueError(
                        f"For 'quantile' method, winsorize_n must be between 0 and 0.5, "
                        f"got: {winsorize_n}"
                    )
            elif winsorize_method in ['std', 'mad']:
                if winsorize_n <= 0:
                    raise ValueError(
                        f"For '{winsorize_method}' method, winsorize_n must be positive, "
                        f"got: {winsorize_n}"
                    )

    elif backtest_mode == "weights":
        pass

    else:
        raise ValueError(
            f"backtest_mode must be one of 'long/short_threshold', "
            f"'gradual_long/short_threshold', 'long/short_normalization', or 'weights', "
            f"got: {backtest_mode}"
        )

    return backtest_params
