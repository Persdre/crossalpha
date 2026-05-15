"""Similarity-weighted and GICS-based industry neutralization helpers."""

from __future__ import annotations

import numpy as np
import polars as pl


def weighted_demean(
    factor_df: pl.DataFrame,
    sim_matrix: np.ndarray,
    sim_symbols: list[str],
    top_k: int = 20,
    col: str = "factor_value",
) -> pl.DataFrame:
    """Demean factor by similarity-weighted top-K peer mean (per date).

    For each (date, stock i): pick the top_k peers by sim_matrix (excluding self),
    normalize their similarities to sum to 1, subtract the weighted peer mean from
    factor_i. Stocks not in sim_symbols are dropped from the output. If some top-K
    peers are absent on a given date, weights are renormalized over present peers.

    Args:
        factor_df: DataFrame with [DATETIME, SYMBOL, col]
        sim_matrix: (n, n) similarity matrix; diagonal is ignored
        sim_symbols: List of n symbols matching sim_matrix rows/cols
        top_k: Number of nearest peers to use
        col: Name of the factor value column

    Returns:
        DataFrame with [DATETIME, SYMBOL, col] containing demeaned values.
    """
    sym_to_idx = {s: i for i, s in enumerate(sim_symbols)}

    n = sim_matrix.shape[0]
    sim_no_self = sim_matrix.copy()
    np.fill_diagonal(sim_no_self, -np.inf)

    if top_k >= n:
        raise ValueError(f"top_k ({top_k}) must be < n_symbols ({n})")

    peer_idx = np.argpartition(-sim_no_self, kth=top_k - 1, axis=1)[:, :top_k]
    row_idx = np.arange(n)[:, None]
    peer_sims = sim_no_self[row_idx, peer_idx]

    sims_sum = peer_sims.sum(axis=1, keepdims=True)
    sims_sum = np.where(sims_sum == 0, 1.0, sims_sum)
    peer_weights = peer_sims / sims_sum

    df = factor_df.filter(pl.col("SYMBOL").is_in(sim_symbols))

    out_rows = []
    for dt in df["DATETIME"].unique().to_list():
        day = df.filter(pl.col("DATETIME") == dt)
        present_syms = day["SYMBOL"].to_list()
        present_vals = day[col].to_numpy().astype(np.float64)
        present_set = set(present_syms)
        sym_to_val = dict(zip(present_syms, present_vals))

        for sym, val in zip(present_syms, present_vals):
            i = sym_to_idx[sym]
            peers_i = peer_idx[i]
            weights_i = peer_weights[i]

            mask = np.array([sim_symbols[j] in present_set for j in peers_i])
            if not mask.any():
                local_mean = 0.0
            else:
                w = weights_i[mask]
                w = w / w.sum() if w.sum() > 0 else w
                v = np.array([sym_to_val[sim_symbols[j]] for j in peers_i[mask]])
                local_mean = float(np.dot(w, v))

            out_rows.append({"DATETIME": dt, "SYMBOL": sym, col: float(val - local_mean)})

    return pl.DataFrame(out_rows)


def gics_demean(
    factor_df: pl.DataFrame,
    sectors_df: pl.DataFrame,
    col: str = "factor_value",
) -> pl.DataFrame:
    """Demean factor by GICS sector within each date.

    Stocks without a sector mapping are dropped from the output.

    Args:
        factor_df: DataFrame with [DATETIME, SYMBOL, col]
        sectors_df: DataFrame with [SYMBOL, gics_sector]
        col: Name of the factor value column

    Returns:
        DataFrame with [DATETIME, SYMBOL, col] containing sector-demeaned values.
    """
    merged = factor_df.join(sectors_df, on="SYMBOL", how="inner")
    return (
        merged.with_columns(
            (pl.col(col) - pl.col(col).mean().over(["DATETIME", "gics_sector"]))
            .alias(col)
        )
        .select(factor_df.columns)
    )


def cs_zscore(factor_df: pl.DataFrame, col: str = "factor_value") -> pl.DataFrame:
    """Cross-sectional z-score per date (population std, ddof=0).

    Drops rows where the resulting z-score is null (e.g., singleton dates
    where std is zero or undefined).

    Args:
        factor_df: DataFrame with [DATETIME, SYMBOL, col]
        col: Name of the factor value column

    Returns:
        DataFrame with [DATETIME, SYMBOL, col] containing per-date z-scored values.
    """
    return (
        factor_df.with_columns(
            ((pl.col(col) - pl.col(col).mean().over("DATETIME"))
             / pl.col(col).std(ddof=0).over("DATETIME"))
            .alias(col)
        )
        .drop_nulls(col)
    )
