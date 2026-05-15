#!/usr/bin/env python3
"""Paper multi-market monthly metrics for the US→Asian headline factor.

Runs the same monthly peer-momentum construction for JP, TW, KR, and HK targets
at 6- and 12-month lookbacks, then reports raw, mild (sector + lnCap), and
3-style neutralized metrics.

Writes:
  - backtest_results_daily/paper_multi_market_full_metrics.json
  - backtest_results_daily/multi_market_full_neutralization.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.h5_data.h5_utils import h5_load
from scripts.kernels import load_monthly_close_prices, load_symbol_sectors, sigmoid_weights
from scripts.kernels.returns import sector_relative_return_df
from scripts.run_baselines_monthly import (
    build_monthly_factor,
    build_true_lncap,
    run_bt,
    three_style_neutralize,
)
from scripts.run_rq3_mild_rerun import mild_neutralize
from scripts.similarity_cache import load_similarity_cache

DATA_ROOT = Path("${CROSSALPHA_DATA_ROOT}")
CACHE_SIM_DIR = DATA_ROOT / "cache" / "similarity"
SYMBOL_DICT_PATH = PROJECT_ROOT / "docs" / "SymbolDict.csv"

PEER_REGION = "us_russell_1000"
TARGETS = {
    "jp": "jp_topix_500",
    "tw": "tw_twse",
    "kr": "kr_kospi",
    "hk": "hk_main",
}
LOOKBACKS = [6, 12]
START = "2015-01-01"
END = "2025-12-31"
C, K = 0.99, 50.0


def _target_sectors(region: str) -> pl.DataFrame:
    return (
        pl.read_csv(SYMBOL_DICT_PATH)
        .filter(pl.col("region") == region)
        .select(["SYMBOL", "gics_sector"])
        .filter(pl.col("gics_sector").is_not_null() & (pl.col("gics_sector") != ""))
    )


def _monthly_labels(region: str) -> pl.DataFrame:
    label_source = "tradable_close_return" if region == "jp_topix_500" else "close_return"
    return h5_load(
        {
            "source_path": str(PROJECT_ROOT / "labels"),
            "product": "Equity",
            "region": region,
            "freq": "1mo",
            "source": label_source,
        },
        start_date=START,
        end_date=END,
        keys=[label_source],
    )


def _style_controls(target_prices: pl.DataFrame, target_sectors: pl.DataFrame,
                    months: list[str], lookback: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    ind_mom = (
        target_prices.sort(["SYMBOL", "DATETIME"])
        .with_columns(
            ((pl.col("close") / pl.col("close").shift(lookback)) - 1)
            .over("SYMBOL")
            .alias("ret")
        )
        .drop_nulls("ret")
        .join(target_sectors, on="SYMBOL", how="inner")
    )
    sector_avg = ind_mom.group_by(["DATETIME", "gics_sector"]).agg(
        pl.col("ret").mean().alias("industry_mom")
    )
    ind_mom = (
        ind_mom.join(sector_avg, on=["DATETIME", "gics_sector"], how="left")
        .select(["DATETIME", "SYMBOL", "industry_mom"])
        .drop_nulls("industry_mom")
        .filter(pl.col("DATETIME").is_in(months))
    )
    own_mom = (
        target_prices.sort(["SYMBOL", "DATETIME"])
        .with_columns(
            ((pl.col("close") / pl.col("close").shift(lookback)) - 1)
            .over("SYMBOL")
            .alias("own_mom")
        )
        .drop_nulls("own_mom")
        .select(["DATETIME", "SYMBOL", "own_mom"])
        .filter(pl.col("DATETIME").is_in(months))
    )
    return ind_mom, own_mom


def _peer_returns(peer_prices: pl.DataFrame, months: list[str], lookback: int) -> dict[str, tuple[list[str], np.ndarray]]:
    us_sectors = load_symbol_sectors(PEER_REGION)
    cum_ret = (
        peer_prices.sort(["SYMBOL", "DATETIME"])
        .with_columns(
            ((pl.col("close") / pl.col("close").shift(lookback)) - 1)
            .over("SYMBOL")
            .alias("cum_return")
        )
        .select(["DATETIME", "SYMBOL", "cum_return"])
        .filter(pl.col("DATETIME").is_in(months))
        .join(us_sectors, on="SYMBOL", how="left")
    )
    sr = sector_relative_return_df(cum_ret, sector_col="sector", return_col="cum_return")
    out = {}
    for month in months:
        md = sr.filter(pl.col("DATETIME") == month)
        if md.height > 0:
            out[month] = (md["SYMBOL"].to_list(), md["sector_relative_return"].to_numpy())
    return out


def _record(metrics: dict, market: str, lookback: int, version: str, r2: float | None = None) -> dict:
    row = {**metrics, "market": market, "lookback": lookback, "version": version}
    if r2 is not None:
        row["r2"] = float(r2)
    return row


def main() -> None:
    print("=" * 100)
    print("PAPER MULTI-MARKET MONTHLY METRICS — US→Asian targets")
    print("=" * 100)

    peer_prices = load_monthly_close_prices(PEER_REGION, "2012-01-01", END)
    paper_rows: list[dict] = []
    neutral_rows: list[dict] = []

    for market, target_region in TARGETS.items():
        print(f"\nTarget: {market.upper()} ({target_region})")
        target_prices = load_monthly_close_prices(target_region, "2012-01-01", END)
        months = sorted(
            target_prices.filter((pl.col("DATETIME") >= START) & (pl.col("DATETIME") <= END))
            .select("DATETIME")
            .unique()["DATETIME"]
            .to_list()
        )
        labels = _monthly_labels(target_region)
        sectors = _target_sectors(target_region)
        lncap = build_true_lncap(target_region, months)
        print(f"  months={len(months)} lncap_rows={lncap.height if lncap is not None else 0}")

        cached = load_similarity_cache(target_region, PEER_REGION, "w128", cache_dir=CACHE_SIM_DIR)
        weights = sigmoid_weights(cached.pct_ranks, k=K, c=C)

        for lookback in LOOKBACKS:
            print(f"  lookback={lookback}mo")
            returns = _peer_returns(peer_prices, months, lookback)
            factor = build_monthly_factor(
                weights, cached.target_symbols, cached.peer_symbols, returns, months
            )
            raw = run_bt(factor, labels, f"paper_us_to_{market}_raw_{lookback}mo", target_region)
            raw_row = _record(raw, market, lookback, "raw")
            paper_rows.append(raw_row)
            neutral_rows.append(raw_row)
            print(f"    raw:  ICIR={raw.get('rank_icir', float('nan')):+.3f} "
                  f"Shp={raw.get('ls_sharpe', float('nan')):+.2f} "
                  f"Ret={raw.get('ls_ret', float('nan')):+.2%}")

            if lncap is not None and not lncap.is_empty():
                mild_factor = mild_neutralize(factor, lncap, sectors)
                mild = run_bt(mild_factor, labels, f"paper_us_to_{market}_mild_{lookback}mo", target_region)
                mild_row = _record(mild, market, lookback, "mild")
                neutral_rows.append(mild_row)
                print(f"    mild: ICIR={mild.get('rank_icir', float('nan')):+.3f} "
                      f"Shp={mild.get('ls_sharpe', float('nan')):+.2f} "
                      f"Ret={mild.get('ls_ret', float('nan')):+.2%}")

                ind_mom, own_mom = _style_controls(target_prices, sectors, months, lookback)
                factor_3s, r2 = three_style_neutralize(factor, lncap, sectors, ind_mom, own_mom)
                three = run_bt(factor_3s, labels, f"paper_us_to_{market}_3style_{lookback}mo", target_region)
                three_row = _record(three, market, lookback, "3style", r2)
                paper_rows.append(three_row)
                neutral_rows.append(three_row)
                print(f"    3sty: ICIR={three.get('rank_icir', float('nan')):+.3f} "
                      f"Shp={three.get('ls_sharpe', float('nan')):+.2f} "
                      f"Ret={three.get('ls_ret', float('nan')):+.2%} R2={r2:.3f}")

    out_dir = PROJECT_ROOT / "backtest_results_daily"
    out_dir.mkdir(exist_ok=True)
    paper_path = out_dir / "paper_multi_market_full_metrics.json"
    neutral_path = out_dir / "multi_market_full_neutralization.json"
    json.dump(paper_rows, open(paper_path, "w"), indent=2, default=str)
    json.dump(neutral_rows, open(neutral_path, "w"), indent=2, default=str)
    print(f"\nSaved: {paper_path}")
    print(f"Saved: {neutral_path}")


if __name__ == "__main__":
    main()
