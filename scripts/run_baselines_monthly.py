#!/usr/bin/env python3
"""Monthly baseline comparison — different similarity sources.

Compares similarity methods for cross-market peer momentum at monthly frequency:
  1. OpenAI embedding (w128) — our method
  2. Return correlation (252d) — price-based
  3. GICS sector equal-weight — industry classification
  4. Domestic peer (JP→JP) — single market
  5. Random shuffle — placebo

All methods use the same monthly factor construction pipeline:
  - 12mo (or 6mo) sector-relative US peer returns
  - Sigmoid(k=50, c=0.99) weighting
  - Three-style neutralization with true market cap
  - LAG=1 month, 5 quintile backtest

Usage:
    python scripts/run_baselines_monthly.py
    python scripts/run_baselines_monthly.py --lookback 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.kernels import (
    load_monthly_close_prices,
    load_symbol_sectors,
    sigmoid_weights,
)
from scripts.kernels.ranking import percentile_rank_fast
from scripts.kernels.returns import sector_relative_return_df
from scripts.similarity_cache import load_similarity_cache
from scripts.h5_data.h5_utils import h5_load
from scripts.backtesting.backtest_utils import run_cross_section_backtest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT = Path("${CROSSALPHA_DATA_ROOT}")
CACHE_SIM_DIR = DATA_ROOT / "cache" / "similarity"
MARKET_DATA_PATH = str(DATA_ROOT / "market_data")
SYMBOL_DICT_PATH = PROJECT_ROOT / "docs" / "SymbolDict.csv"

TARGET_REGION = "jp_topix_500"
PEER_REGION = "us_russell_1000"

START = "2015-01-01"
END = "2025-12-31"
C, K = 0.99, 50.0
CORR_WINDOW = 252

FEES = [0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.0010, 0.0015, 0.0020]
LAYERS = 5
LAG = 1
ANNUAL_DAYS = 12
FREQ = "1mo"
MIN_STOCKS = 10


# ---------------------------------------------------------------------------
# Three-style neutralization (same as other monthly scripts)
# ---------------------------------------------------------------------------

def _winsorize(x, p=1.0):
    return np.clip(x, np.nanpercentile(x, p), np.nanpercentile(x, 100 - p))

def _zscore(x):
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / sd if sd > 1e-12 else np.zeros_like(x)

def three_style_neutralize(factor_df, lncap_df, sectors_df, ind_mom_df, own_mom_df):
    merged = (factor_df.join(lncap_df, on=["DATETIME", "SYMBOL"], how="inner")
              .join(sectors_df, on="SYMBOL", how="inner")
              .join(ind_mom_df, on=["DATETIME", "SYMBOL"], how="inner")
              .join(own_mom_df, on=["DATETIME", "SYMBOL"], how="inner"))
    recs, r2s = [], []
    for dt in merged["DATETIME"].unique().sort().to_list():
        d = merged.filter(pl.col("DATETIME") == dt)
        y = d["factor_value"].to_numpy().astype(np.float64)
        lc = d["lncap"].to_numpy().astype(np.float64)
        im = d["industry_mom"].to_numpy().astype(np.float64)
        om = d["own_mom"].to_numpy().astype(np.float64)
        secs = d["gics_sector"].to_list()
        syms = d["SYMBOL"].to_list()
        m = np.isfinite(y) & np.isfinite(lc) & np.isfinite(im) & np.isfinite(om)
        if m.sum() < MIN_STOCKS:
            continue
        y2, lc2, im2, om2 = y[m], lc[m], im[m], om[m]
        s2 = [s for s, mm in zip(syms, m) if mm]
        sec2 = [s for s, mm in zip(secs, m) if mm]
        yy = y2.copy()
        for sec in set(sec2):
            sm = np.array([s == sec for s in sec2])
            if sm.any():
                yy[sm] -= np.nanmean(y2[sm])
        yy, lc2, im2, om2 = _zscore(_winsorize(yy)), _zscore(_winsorize(lc2)), _zscore(_winsorize(im2)), _zscore(_winsorize(om2))
        X = np.column_stack([np.ones(len(yy)), lc2, im2, om2])
        try:
            b, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
        except:
            continue
        resid = yy - X @ b
        ss_r, ss_t = np.sum(resid ** 2), np.sum((yy - yy.mean()) ** 2)
        r2s.append(1 - ss_r / ss_t if ss_t > 1e-12 else 0)
        for s, v in zip(s2, resid):
            recs.append({"DATETIME": dt, "SYMBOL": s, "factor_value": float(v)})
    return pl.DataFrame(recs) if recs else pl.DataFrame(), np.mean(r2s) if r2s else 0


def build_true_lncap(region, months):
    shares_path = PROJECT_ROOT / "market_data" / "shares_outstanding" / f"{region}.parquet"
    if not shares_path.exists():
        return None
    shares = pl.read_parquet(shares_path).with_columns(pl.col("DATETIME").str.slice(0, 7).alias("_ym"))
    sm = shares.sort("DATETIME").group_by(["_ym", "SYMBOL"]).agg(pl.col("shares_outstanding").last())
    all_yms = sorted(set([m[:7] for m in months]))
    recs = []
    for sym in shares["SYMBOL"].unique().to_list():
        sd = sm.filter(pl.col("SYMBOL") == sym).sort("_ym")
        yms, vals = sd["_ym"].to_list(), sd["shares_outstanding"].to_list()
        if not yms:
            continue
        lv = None
        for ym in all_yms:
            for i in range(len(yms) - 1, -1, -1):
                if yms[i] <= ym:
                    lv = vals[i]
                    break
            if lv is not None:
                recs.append({"DATETIME": ym + "-01", "SYMBOL": sym, "shares_outstanding": lv})
    sf = pl.DataFrame(recs)
    prices = load_monthly_close_prices(region, "2013-01-01", END)
    return (prices.filter(pl.col("DATETIME").is_in(months))
            .join(sf, on=["DATETIME", "SYMBOL"], how="inner")
            .with_columns((pl.col("close") * pl.col("shares_outstanding")).log().alias("lncap"))
            .select(["DATETIME", "SYMBOL", "lncap"]).drop_nulls("lncap").filter(pl.col("lncap").is_finite()))


# ---------------------------------------------------------------------------
# Factor construction helpers
# ---------------------------------------------------------------------------

def build_monthly_factor(weights, target_syms, peer_syms, peer_returns_by_month, months):
    """Build monthly factor from weight matrix and peer returns."""
    p2i = {s: i for i, s in enumerate(peer_syms)}
    recs = []
    for m in months:
        if m not in peer_returns_by_month:
            continue
        ps, pv = peer_returns_by_month[m]
        a = np.full(len(peer_syms), np.nan)
        for s, r in zip(ps, pv):
            if s in p2i:
                a[p2i[s]] = r
        v = ~np.isnan(a)
        rm = np.where(v, a, 0.0)
        ws = weights @ rm
        wd = weights @ v.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            sc = np.where(wd > 1, ws / wd, np.nan)
        for sym, val in zip(target_syms, sc):
            if np.isfinite(val):
                recs.append({"DATETIME": m, "SYMBOL": sym, "factor_value": float(val)})
    return pl.DataFrame(recs)


def extract_metrics(summary_path):
    m = json.load(open(summary_path)).get("metrics", {})
    def g(k, c="0.0 bp"):
        v = m.get(k, {}).get("by_cost", {}).get(c)
        return float(v) if v is not None else float("nan")
    c2 = "2.0 bp"
    return {
        "rank_ic": g("rank_ic"), "rank_icir": g("rank_icir_annual"),
        "ic_pvalue": g("rank_ic_p_value"), "ic_winratio": g("rank_ic_winratio"),
        "ls_sharpe_0bp": g("long_short_ret_sharpe"),
        "ls_sharpe": g("long_short_ret_sharpe", c2),
        "ls_ret": g("long_short_ret_annual", c2),
        "ls_max_dd": g("long_short_ret_max_dd", c2),
        "ls_sortino": g("long_short_ret_sortino", c2),
        "ls_turnover": g("long_short_turnover_ratio", c2),
        "long_sharpe": g("long_ret_sharpe", c2),
        "long_ret": g("long_ret_annual", c2),
    }


def run_bt(fdf, labels, name, region):
    label_col = [c for c in labels.columns if c not in ("DATETIME", "SYMBOL")][0]
    df = fdf.join(labels, on=["DATETIME", "SYMBOL"], how="inner")
    if df.height == 0:
        return {}
    d = PROJECT_ROOT / "backtest_results_monthly" / region / name
    d.mkdir(parents=True, exist_ok=True)
    run_cross_section_backtest(
        df=df, factor_column_name="factor_value",
        datetime_column_name="DATETIME", symbol_column_name="SYMBOL",
        raw_label_column_name=label_col, freq=FREQ, layers_use=LAYERS, fees=FEES,
        backtest_mode="long/short_layers",
        backtest_params={"long_layer_index": [1], "short_layer_index": [LAYERS]},
        lag=LAG, annual_days=ANNUAL_DAYS,
        factor_info_dict={"factor_name": name, "region": region}, output_dir=d,
    )
    return extract_metrics(d / "summary_df.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Monthly baseline comparison")
    parser.add_argument("--lookback", type=int, default=12, choices=[3, 6, 12])
    args = parser.parse_args()
    lookback = args.lookback

    print(f"{'=' * 90}")
    print(f"MONTHLY BASELINE COMPARISON — JP TOPIX 500, {lookback}mo lookback")
    print(f"{'=' * 90}")

    # Load data
    print("\nLoading data...", flush=True)
    jp_m = load_monthly_close_prices(TARGET_REGION, "2012-01-01", END)
    us_m = load_monthly_close_prices(PEER_REGION, "2012-01-01", END)
    months = sorted(jp_m.filter((pl.col("DATETIME") >= START) & (pl.col("DATETIME") <= END))
                    .select("DATETIME").unique()["DATETIME"].to_list())

    labels = h5_load(
        {"source_path": str(PROJECT_ROOT / "labels"), "product": "Equity",
         "region": TARGET_REGION, "freq": "1mo", "source": "tradable_close_return"},
        start_date=START, end_date=END, keys=["tradable_close_return"],
    )

    jp_sectors = pl.read_csv(SYMBOL_DICT_PATH).filter(pl.col("region") == TARGET_REGION).select(
        ["SYMBOL", "gics_sector"]).filter(pl.col("gics_sector").is_not_null() & (pl.col("gics_sector") != ""))
    us_sectors = load_symbol_sectors(PEER_REGION)

    # True lnCap
    lncap = build_true_lncap(TARGET_REGION, months)
    print(f"  lnCap: {lncap.height if lncap is not None else 0} rows", flush=True)

    # Style variables (same lookback as factor)
    ind_mom = (jp_m.sort(["SYMBOL", "DATETIME"])
        .with_columns(((pl.col("close") / pl.col("close").shift(lookback)) - 1).over("SYMBOL").alias("ret"))
        .drop_nulls("ret").join(jp_sectors, on="SYMBOL", how="inner"))
    sa = ind_mom.group_by(["DATETIME", "gics_sector"]).agg(pl.col("ret").mean().alias("industry_mom"))
    ind_mom = (ind_mom.join(sa, on=["DATETIME", "gics_sector"], how="left")
        .select(["DATETIME", "SYMBOL", "industry_mom"]).drop_nulls("industry_mom")
        .filter(pl.col("DATETIME").is_in(months)))
    own_mom = (jp_m.sort(["SYMBOL", "DATETIME"])
        .with_columns(((pl.col("close") / pl.col("close").shift(lookback)) - 1).over("SYMBOL").alias("own_mom"))
        .drop_nulls("own_mom").select(["DATETIME", "SYMBOL", "own_mom"])
        .filter(pl.col("DATETIME").is_in(months)))

    # US peer sector-relative returns
    cum_ret = (us_m.sort(["SYMBOL", "DATETIME"])
        .with_columns(((pl.col("close") / pl.col("close").shift(lookback)) - 1).over("SYMBOL").alias("cum_return"))
        .select(["DATETIME", "SYMBOL", "cum_return"])
        .filter(pl.col("DATETIME").is_in(months))
        .join(us_sectors, on="SYMBOL", how="left"))
    sr = sector_relative_return_df(cum_ret, sector_col="sector", return_col="cum_return")
    peer_returns = {}
    for m in months:
        md = sr.filter(pl.col("DATETIME") == m)
        if md.height > 0:
            peer_returns[m] = (md["SYMBOL"].to_list(), md["sector_relative_return"].to_numpy())

    all_results = []

    def test_method(name, factor_df):
        """Run raw + three-style backtest for a factor."""
        # Raw
        bt_raw = run_bt(factor_df, labels, f"baseline_{name}_raw_{lookback}mo", TARGET_REGION)
        bt_raw.update({"method": name, "version": "raw", "lookback": lookback})
        all_results.append(bt_raw)

        # Three-style
        if lncap is not None and factor_df.height > 0:
            f_3s, r2 = three_style_neutralize(factor_df, lncap, jp_sectors, ind_mom, own_mom)
            if f_3s.height > 0:
                bt_3s = run_bt(f_3s, labels, f"baseline_{name}_3style_{lookback}mo", TARGET_REGION)
                bt_3s.update({"method": name, "version": "3style", "lookback": lookback, "r2": r2})
                all_results.append(bt_3s)
                return bt_raw, bt_3s, r2
        return bt_raw, {}, 0.0

    def f(v, fmt=".3f"):
        return f"{v:{fmt}}" if isinstance(v, (int, float)) and np.isfinite(v) else "  N/A"

    # ================================================================
    # 1. OpenAI Embedding (w128)
    # ================================================================
    print(f"\n[1] OpenAI Embedding (w128)", flush=True)
    cached = load_similarity_cache(TARGET_REGION, PEER_REGION, "w128", cache_dir=CACHE_SIM_DIR)
    w_emb = sigmoid_weights(cached.pct_ranks, k=K, c=C)
    f_emb = build_monthly_factor(w_emb, cached.target_symbols, cached.peer_symbols, peer_returns, months)
    bt_r, bt_3, r2 = test_method("openai_emb", f_emb)
    print(f"  Raw:    ICIR={f(bt_r.get('rank_icir'))}  Sharpe={f(bt_r.get('ls_sharpe'))}  Ret={f(bt_r.get('ls_ret'),'.2%')}")
    if bt_3:
        print(f"  3style: ICIR={f(bt_3.get('rank_icir'))}  Sharpe={f(bt_3.get('ls_sharpe'))}  Ret={f(bt_3.get('ls_ret'),'.2%')}  R²={r2:.3f}")

    # ================================================================
    # 2. Return Correlation
    # ================================================================
    print(f"\n[2] Return Correlation ({CORR_WINDOW}d)", flush=True)
    from scripts.kernels import load_daily_close_prices
    from scripts.kernels.return_correlation import compute_return_correlation_matrix
    jp_d = load_daily_close_prices(TARGET_REGION, "2014-01-01", END, source_path=MARKET_DATA_PATH)
    us_d = load_daily_close_prices(PEER_REGION, "2014-01-01", END, source_path=MARKET_DATA_PATH)
    def _rw(p):
        return (p.sort(["SYMBOL", "DATETIME"]).with_columns(
            ((pl.col("close") / pl.col("close").shift(1)) - 1).over("SYMBOL").alias("ret"))
            .drop_nulls("ret").pivot(index="DATETIME", on="SYMBOL", values="ret").sort("DATETIME"))
    jw, uw = _rw(jp_d), _rw(us_d)
    cm = jw.join(uw.select("DATETIME"), on="DATETIME", how="inner")
    uc = uw.join(jw.select("DATETIME"), on="DATETIME", how="inner")
    cm, uc = cm.tail(CORR_WINDOW), uc.tail(CORR_WINDOW)
    js = sorted([c for c in cm.columns if c != "DATETIME"])
    us = sorted([c for c in uc.columns if c != "DATETIME"])
    corr = compute_return_correlation_matrix(
        cm.select(js).to_numpy().astype(np.float64),
        uc.select(us).to_numpy().astype(np.float64), min_overlap=120)
    w_corr = sigmoid_weights(np.nan_to_num(percentile_rank_fast(corr, axis=1), nan=0.0), k=K, c=C)
    w_corr = np.nan_to_num(w_corr, nan=0.0)
    f_corr = build_monthly_factor(w_corr, js, us, peer_returns, months)
    bt_r, bt_3, r2 = test_method("return_corr", f_corr)
    print(f"  Raw:    ICIR={f(bt_r.get('rank_icir'))}  Sharpe={f(bt_r.get('ls_sharpe'))}  Ret={f(bt_r.get('ls_ret'),'.2%')}")
    if bt_3:
        print(f"  3style: ICIR={f(bt_3.get('rank_icir'))}  Sharpe={f(bt_3.get('ls_sharpe'))}  Ret={f(bt_3.get('ls_ret'),'.2%')}  R²={r2:.3f}")

    # ================================================================
    # 3. GICS Sector Equal-Weight
    # ================================================================
    print(f"\n[3] GICS Sector Equal-Weight", flush=True)
    jp_sec_map = dict(zip(jp_sectors["SYMBOL"].to_list(), jp_sectors["gics_sector"].to_list()))
    us_sec_map = dict(zip(
        pl.read_csv(SYMBOL_DICT_PATH).filter(pl.col("region") == PEER_REGION)["SYMBOL"].to_list(),
        pl.read_csv(SYMBOL_DICT_PATH).filter(pl.col("region") == PEER_REGION)["gics_sector"].to_list()))
    gics_recs = []
    for m in months:
        if m not in peer_returns:
            continue
        ps, pv = peer_returns[m]
        # Compute sector-level mean return
        sec_rets = {}
        for s, r in zip(ps, pv):
            sec = us_sec_map.get(s)
            if sec and np.isfinite(r):
                sec_rets.setdefault(sec, []).append(r)
        sec_avg = {s: np.mean(v) for s, v in sec_rets.items()}
        # Assign to JP stocks by sector
        for sym, sec in jp_sec_map.items():
            if sec in sec_avg:
                gics_recs.append({"DATETIME": m, "SYMBOL": sym, "factor_value": sec_avg[sec]})
    f_gics = pl.DataFrame(gics_recs)
    bt_r, bt_3, r2 = test_method("gics_sector_ew", f_gics)
    print(f"  Raw:    ICIR={f(bt_r.get('rank_icir'))}  Sharpe={f(bt_r.get('ls_sharpe'))}  Ret={f(bt_r.get('ls_ret'),'.2%')}")
    if bt_3:
        print(f"  3style: ICIR={f(bt_3.get('rank_icir'))}  Sharpe={f(bt_3.get('ls_sharpe'))}  Ret={f(bt_3.get('ls_ret'),'.2%')}  R²={r2:.3f}")

    # ================================================================
    # 4. Domestic (JP→JP)
    # ================================================================
    print(f"\n[4] Domestic (JP→JP)", flush=True)
    try:
        cached_dom = load_similarity_cache(TARGET_REGION, TARGET_REGION, "w128", cache_dir=CACHE_SIM_DIR)
        w_dom = sigmoid_weights(cached_dom.pct_ranks, k=K, c=C)
        # Domestic peer returns: JP sector-relative
        jp_sectors_load = load_symbol_sectors(TARGET_REGION)
        jp_cum = (jp_m.sort(["SYMBOL", "DATETIME"])
            .with_columns(((pl.col("close") / pl.col("close").shift(lookback)) - 1).over("SYMBOL").alias("cum_return"))
            .select(["DATETIME", "SYMBOL", "cum_return"])
            .filter(pl.col("DATETIME").is_in(months))
            .join(jp_sectors_load, on="SYMBOL", how="left"))
        jp_sr = sector_relative_return_df(jp_cum, sector_col="sector", return_col="cum_return")
        dom_returns = {}
        for m in months:
            md = jp_sr.filter(pl.col("DATETIME") == m)
            if md.height > 0:
                dom_returns[m] = (md["SYMBOL"].to_list(), md["sector_relative_return"].to_numpy())
        f_dom = build_monthly_factor(w_dom, cached_dom.target_symbols, cached_dom.peer_symbols, dom_returns, months)
        bt_r, bt_3, r2 = test_method("domestic", f_dom)
        print(f"  Raw:    ICIR={f(bt_r.get('rank_icir'))}  Sharpe={f(bt_r.get('ls_sharpe'))}  Ret={f(bt_r.get('ls_ret'),'.2%')}")
        if bt_3:
            print(f"  3style: ICIR={f(bt_3.get('rank_icir'))}  Sharpe={f(bt_3.get('ls_sharpe'))}  Ret={f(bt_3.get('ls_ret'),'.2%')}  R²={r2:.3f}")
    except FileNotFoundError:
        print("  SKIPPED: domestic cache not found")

    # ================================================================
    # 5. Random Shuffle (Placebo)
    # ================================================================
    print(f"\n[5] Random Shuffle (10 trials)", flush=True)
    np.random.seed(42)
    placebo_results = []
    for trial in range(10):
        perm = np.random.permutation(w_emb.shape[1])
        w_shuf = w_emb[:, perm]
        f_shuf = build_monthly_factor(w_shuf, cached.target_symbols, cached.peer_symbols, peer_returns, months)
        bt = run_bt(f_shuf, labels, f"baseline_random_{trial}_{lookback}mo", TARGET_REGION)
        placebo_results.append(bt.get("rank_icir", float("nan")))
    avg_icir = np.nanmean(placebo_results)
    std_icir = np.nanstd(placebo_results)
    print(f"  Avg ICIR={avg_icir:.3f} ± {std_icir:.3f}")
    all_results.append({"method": "random_shuffle", "version": "raw", "lookback": lookback,
                        "rank_icir": avg_icir, "icir_std": std_icir})

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 110}")
    print(f"MONTHLY BASELINE SUMMARY — {lookback}mo lookback")
    print(f"{'=' * 110}")
    print(f"{'Method':<20}{'Version':<10}{'ICIR':>7}{'Sharpe':>8}{'Ret':>8}{'MaxDD':>8}"
          f"{'LgSh':>7}{'LgRet':>8}")
    print("-" * 80)
    for r in all_results:
        print(f"{r.get('method',''):<20}{r.get('version',''):<10}"
              f"{f(r.get('rank_icir', float('nan'))):>7}"
              f"{f(r.get('ls_sharpe', float('nan'))):>8}"
              f"{f(r.get('ls_ret', float('nan')), '.2%'):>8}"
              f"{f(r.get('ls_max_dd', float('nan')), '.2%'):>8}"
              f"{f(r.get('long_sharpe', float('nan'))):>7}"
              f"{f(r.get('long_ret', float('nan')), '.2%'):>8}")

    out = PROJECT_ROOT / "backtest_results_daily" / f"baselines_monthly_{lookback}mo.json"
    json.dump(all_results, open(out, "w"), indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
