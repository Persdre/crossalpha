#!/usr/bin/env python3
"""Mild neutralization helper for full-universe RQ3 evaluations."""

from __future__ import annotations

import numpy as np
import polars as pl

MIN_STOCKS = 10


def _winsorize(x: np.ndarray, p: float = 1.0) -> np.ndarray:
    return np.clip(x, np.nanpercentile(x, p), np.nanpercentile(x, 100 - p))


def _zscore(x: np.ndarray) -> np.ndarray:
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / sd if sd > 1e-12 else np.zeros_like(x)


def mild_neutralize(
    factor_df: pl.DataFrame,
    lncap_df: pl.DataFrame,
    sectors_df: pl.DataFrame,
) -> pl.DataFrame:
    """Neutralize each monthly cross-section by sector and ln(mcap).

    This is the "mild" convention used in the RQ3 tables: first demean the
    factor within GICS sector, then regress the standardized residual factor on
    standardized ln(mcap) and return the cross-sectional residual.
    """
    if factor_df is None or factor_df.is_empty():
        return pl.DataFrame()
    merged = (
        factor_df.join(lncap_df, on=["DATETIME", "SYMBOL"], how="inner")
        .join(sectors_df, on="SYMBOL", how="inner")
    )
    if merged.is_empty():
        return pl.DataFrame()

    recs: list[dict[str, object]] = []
    for dt in merged["DATETIME"].unique().sort().to_list():
        d = merged.filter(pl.col("DATETIME") == dt)
        y = d["factor_value"].to_numpy().astype(np.float64)
        lc = d["lncap"].to_numpy().astype(np.float64)
        secs = d["gics_sector"].to_list()
        syms = d["SYMBOL"].to_list()

        mask = np.isfinite(y) & np.isfinite(lc)
        if mask.sum() < MIN_STOCKS:
            continue

        y2 = y[mask]
        lc2 = lc[mask]
        syms2 = [s for s, ok in zip(syms, mask) if ok]
        secs2 = [s for s, ok in zip(secs, mask) if ok]

        yy = y2.copy()
        for sec in set(secs2):
            sec_mask = np.array([s == sec for s in secs2])
            if sec_mask.any():
                yy[sec_mask] -= np.nanmean(y2[sec_mask])

        yy = _zscore(_winsorize(yy))
        lc2 = _zscore(_winsorize(lc2))
        x = np.column_stack([np.ones(len(yy)), lc2])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(x, yy, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = yy - x @ coeffs

        for sym, val in zip(syms2, resid):
            recs.append({"DATETIME": dt, "SYMBOL": sym, "factor_value": float(val)})

    return pl.DataFrame(recs) if recs else pl.DataFrame()
