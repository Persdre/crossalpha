"""Return calculation kernels (Numba-accelerated where applicable)."""

from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit


def _ensure_date_col(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize DATETIME to Polars Date for calendar joins."""
    dtype = df.schema.get("DATETIME")
    if dtype == pl.Utf8:
        return df.with_columns(pl.col("DATETIME").str.strptime(pl.Date, strict=False))
    if dtype == pl.Datetime:
        return df.with_columns(pl.col("DATETIME").dt.date())
    if dtype == pl.Object:
        return df.with_columns(
            pl.col("DATETIME").map_elements(
                lambda v: getattr(v, "date", lambda: v)(),
                return_dtype=pl.Date,
            )
        )
    return df


def monthly_return(
    close_prices: pl.DataFrame,
) -> pl.DataFrame:
    """Compute monthly returns from close prices.

    Args:
        close_prices: DataFrame with columns [DATETIME, SYMBOL, close]
                      sorted by [SYMBOL, DATETIME]

    Returns:
        DataFrame with columns [DATETIME, SYMBOL, close_return]
    """
    return (
        close_prices.sort(["SYMBOL", "DATETIME"])
        .with_columns(
            ((pl.col("close") / pl.col("close").shift(1)) - 1)
            .over("SYMBOL")
            .alias("close_return")
        )
        .select(["DATETIME", "SYMBOL", "close_return"])
    )


def tradable_monthly_return(
    monthly_close_prices: pl.DataFrame,
    daily_close_prices: pl.DataFrame,
) -> pl.DataFrame:
    """Compute executable monthly returns using first tradable close of month.

    For month M, the return is:
        month_end_close(M) / first_trading_day_close(M) - 1

    This removes the untradeable month-boundary gap between the prior month-end
    close and the first tradable close in month M.

    Args:
        monthly_close_prices: DataFrame with columns [DATETIME, SYMBOL, close]
            at monthly frequency, where DATETIME is the month bucket.
        daily_close_prices: DataFrame with columns [DATETIME, SYMBOL, close]
            at daily frequency.

    Returns:
        DataFrame with columns [DATETIME, SYMBOL, tradable_close_return]
    """
    monthly_close_prices = _ensure_date_col(monthly_close_prices)
    daily_close_prices = _ensure_date_col(daily_close_prices)

    first_trading_close = (
        daily_close_prices.sort(["SYMBOL", "DATETIME"])
        .with_columns(pl.col("DATETIME").dt.truncate("1mo").alias("DATETIME"))
        .group_by(["DATETIME", "SYMBOL"], maintain_order=True)
        .agg(pl.col("close").first().alias("_entry_close"))
    )

    return (
        monthly_close_prices
        .join(first_trading_close, on=["DATETIME", "SYMBOL"], how="inner")
        .with_columns(
            (pl.col("close") / pl.col("_entry_close") - 1).alias("tradable_close_return"),
            pl.col("DATETIME").dt.strftime("%Y-%m-%d"),
        )
        .select(["DATETIME", "SYMBOL", "tradable_close_return"])
    )


def cumulative_return(
    close_prices: pl.DataFrame,
    lookback: int,
) -> pl.DataFrame:
    """Compute N-month cumulative returns.

    Args:
        close_prices: DataFrame with columns [DATETIME, SYMBOL, close]
        lookback: Number of months to look back

    Returns:
        DataFrame with columns [DATETIME, SYMBOL, cum_return]
    """
    return (
        close_prices.sort(["SYMBOL", "DATETIME"])
        .with_columns(
            ((pl.col("close") / pl.col("close").shift(lookback)) - 1)
            .over("SYMBOL")
            .alias("cum_return")
        )
        .select(["DATETIME", "SYMBOL", "cum_return"])
    )


@njit(cache=True)
def _sector_relative_return_core(
    returns: np.ndarray,
    sector_ids: np.ndarray,
    n_sectors: int,
) -> np.ndarray:
    """Compute sector-relative returns.

    Args:
        returns: Shape (n_stocks,) stock returns
        sector_ids: Shape (n_stocks,) integer sector IDs (0 to n_sectors-1, -1 for missing)
        n_sectors: Number of unique sectors

    Returns:
        Shape (n_stocks,) sector-relative returns
    """
    n = len(returns)

    # Compute sector means
    sector_sums = np.zeros(n_sectors, dtype=np.float64)
    sector_counts = np.zeros(n_sectors, dtype=np.int64)

    for i in range(n):
        sid = sector_ids[i]
        if sid >= 0 and not np.isnan(returns[i]):
            sector_sums[sid] += returns[i]
            sector_counts[sid] += 1

    sector_means = np.zeros(n_sectors, dtype=np.float64)
    for s in range(n_sectors):
        if sector_counts[s] > 0:
            sector_means[s] = sector_sums[s] / sector_counts[s]

    # Compute relative returns
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        sid = sector_ids[i]
        if sid >= 0:
            result[i] = returns[i] - sector_means[sid]
        else:
            result[i] = returns[i]

    return result


def sector_relative_return(
    returns: np.ndarray,
    sectors: np.ndarray,
) -> np.ndarray:
    """Compute sector-relative returns (equal-weighted, include self).

    Args:
        returns: Array of shape (n_stocks,) with stock returns
        sectors: Array of shape (n_stocks,) with sector labels

    Returns:
        Array of shape (n_stocks,) with sector-relative returns
    """
    returns_arr = np.ascontiguousarray(returns, dtype=np.float64)

    # Convert sectors to integer IDs
    unique_sectors = []
    sector_to_id = {}
    for s in sectors:
        if s is not None and s != "" and s not in sector_to_id:
            sector_to_id[s] = len(unique_sectors)
            unique_sectors.append(s)

    n_sectors = len(unique_sectors)
    sector_ids = np.array([
        sector_to_id.get(s, -1) if s is not None and s != "" else -1
        for s in sectors
    ], dtype=np.int64)

    return _sector_relative_return_core(returns_arr, sector_ids, n_sectors)


def sector_relative_return_df(
    returns_df: pl.DataFrame,
    sector_col: str = "sector",
    return_col: str = "cum_return",
) -> pl.DataFrame:
    """Compute sector-relative returns using Polars.

    Args:
        returns_df: DataFrame with columns [DATETIME, SYMBOL, sector, cum_return]
        sector_col: Column name for sector
        return_col: Column name for return values

    Returns:
        DataFrame with additional column 'sector_relative_return'
    """
    # Compute sector means excluding NaN values
    sector_means = (
        returns_df.filter(pl.col(return_col).is_not_nan())
        .group_by(["DATETIME", sector_col])
        .agg(pl.col(return_col).mean().alias("_sector_mean"))
    )

    # Join sector means back and compute relative return
    return (
        returns_df.join(sector_means, on=["DATETIME", sector_col], how="left")
        .with_columns(
            (pl.col(return_col) - pl.col("_sector_mean")).alias("sector_relative_return")
        )
        .drop("_sector_mean")
    )
