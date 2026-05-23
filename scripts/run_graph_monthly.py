#!/usr/bin/env python3
"""Monthly graph experiments + random placebo.

Tests supply chain graph boost and validates with random graph placebo.
Uses three-style neutralization with true market cap.

Usage:
    python scripts/run_graph_monthly.py
    python scripts/run_graph_monthly.py --lookback 6
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

from scripts.kernels import load_monthly_close_prices, load_symbol_sectors, sigmoid_weights
from scripts.kernels.returns import sector_relative_return_df
from scripts.kernels.ranking import percentile_rank_fast
from scripts.similarity_cache import load_similarity_cache
from scripts.h5_data.h5_utils import h5_load
from scripts.backtesting.backtest_utils import run_cross_section_backtest

DATA_ROOT = Path("${CROSSALPHA_DATA_ROOT}")
CACHE_SIM_DIR = DATA_ROOT / "cache" / "similarity"
SYMBOL_DICT_PATH = PROJECT_ROOT / "docs" / "SymbolDict.csv"
TARGET_REGION = "jp_topix_500"
PEER_REGION = "us_russell_1000"
START, END = "2015-01-01", "2025-12-31"
C, K = 0.99, 50.0
FEES = [0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.0010, 0.0015, 0.0020]
LAYERS = 5
LAG = 1
ANNUAL_DAYS = 12
MIN_STOCKS = 10


def _w(x): return np.clip(x, np.nanpercentile(x, 1), np.nanpercentile(x, 99))
def _z(x): mu, sd = np.nanmean(x), np.nanstd(x); return (x - mu) / sd if sd > 1e-12 else np.zeros_like(x)


def three_style(factor_df, lncap_df, sectors_df, ind_mom_df, own_mom_df):
    merged = (factor_df.join(lncap_df, on=["DATETIME", "SYMBOL"], how="inner")
              .join(sectors_df, on="SYMBOL", how="inner")
              .join(ind_mom_df, on=["DATETIME", "SYMBOL"], how="inner")
              .join(own_mom_df, on=["DATETIME", "SYMBOL"], how="inner"))
    recs, r2s = [], []
    for dt in merged["DATETIME"].unique().sort().to_list():
        d = merged.filter(pl.col("DATETIME") == dt)
        y, lc, im, om = [d[c].to_numpy().astype(np.float64) for c in ["factor_value", "lncap", "industry_mom", "own_mom"]]
        secs, syms = d["gics_sector"].to_list(), d["SYMBOL"].to_list()
        m = np.isfinite(y) & np.isfinite(lc) & np.isfinite(im) & np.isfinite(om)
        if m.sum() < MIN_STOCKS: continue
        y2, lc2, im2, om2 = y[m], lc[m], im[m], om[m]
        s2 = [s for s, mm in zip(syms, m) if mm]; sec2 = [s for s, mm in zip(secs, m) if mm]
        yy = y2.copy()
        for sec in set(sec2):
            sm = np.array([s == sec for s in sec2])
            if sm.any(): yy[sm] -= np.nanmean(y2[sm])
        yy, lc2, im2, om2 = _z(_w(yy)), _z(_w(lc2)), _z(_w(im2)), _z(_w(om2))
        X = np.column_stack([np.ones(len(yy)), lc2, im2, om2])
        try: b, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
        except: continue
        resid = yy - X @ b
        ss_r, ss_t = np.sum(resid**2), np.sum((yy - yy.mean())**2)
        r2s.append(1 - ss_r / ss_t if ss_t > 1e-12 else 0)
        for s, v in zip(s2, resid): recs.append({"DATETIME": dt, "SYMBOL": s, "factor_value": float(v)})
    return pl.DataFrame(recs) if recs else pl.DataFrame(), np.mean(r2s) if r2s else 0


def build_true_lncap(region, months):
    sp = PROJECT_ROOT / "market_data" / "shares_outstanding" / f"{region}.parquet"
    if not sp.exists(): return None
    shares = pl.read_parquet(sp).with_columns(pl.col("DATETIME").str.slice(0, 7).alias("_ym"))
    sm = shares.sort("DATETIME").group_by(["_ym", "SYMBOL"]).agg(pl.col("shares_outstanding").last())
    all_yms = sorted(set([m[:7] for m in months]))
    recs = []
    for sym in shares["SYMBOL"].unique().to_list():
        sd = sm.filter(pl.col("SYMBOL") == sym).sort("_ym")
        yms, vals = sd["_ym"].to_list(), sd["shares_outstanding"].to_list()
        if not yms: continue
        lv = None
        for ym in all_yms:
            for i in range(len(yms)-1,-1,-1):
                if yms[i] <= ym: lv = vals[i]; break
            if lv is not None: recs.append({"DATETIME": ym+"-01", "SYMBOL": sym, "shares_outstanding": lv})
    sf = pl.DataFrame(recs)
    prices = load_monthly_close_prices(region, "2013-01-01", END)
    return (prices.filter(pl.col("DATETIME").is_in(months))
            .join(sf, on=["DATETIME", "SYMBOL"], how="inner")
            .with_columns((pl.col("close") * pl.col("shares_outstanding")).log().alias("lncap"))
            .select(["DATETIME", "SYMBOL", "lncap"]).drop_nulls("lncap").filter(pl.col("lncap").is_finite()))


def build_factor(weights, tsyms, psyms, returns_dict, months):
    p2i = {s: i for i, s in enumerate(psyms)}; recs = []
    for m in months:
        if m not in returns_dict: continue
        ps, pv = returns_dict[m]
        a = np.full(len(psyms), np.nan)
        for s, r in zip(ps, pv):
            if s in p2i: a[p2i[s]] = r
        v = ~np.isnan(a); rm = np.where(v, a, 0.0)
        ws = weights @ rm; wd = weights @ v.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"): sc = np.where(wd > 1, ws / wd, np.nan)
        for sym, val in zip(tsyms, sc):
            if np.isfinite(val): recs.append({"DATETIME": m, "SYMBOL": sym, "factor_value": float(val)})
    return pl.DataFrame(recs)


def extract_metrics(path):
    m = json.load(open(path)).get("metrics", {})
    def g(k, c="0.0 bp"):
        v = m.get(k, {}).get("by_cost", {}).get(c); return float(v) if v is not None else float("nan")
    c2 = "2.0 bp"
    return {"rank_ic": g("rank_ic"), "rank_icir": g("rank_icir_annual"),
            "ls_sharpe": g("long_short_ret_sharpe", c2), "ls_ret": g("long_short_ret_annual", c2),
            "ls_max_dd": g("long_short_ret_max_dd", c2), "long_sharpe": g("long_ret_sharpe", c2),
            "long_ret": g("long_ret_annual", c2)}


def run_bt(fdf, labels, name):
    label_col = [c for c in labels.columns if c not in ("DATETIME", "SYMBOL")][0]
    df = fdf.join(labels, on=["DATETIME", "SYMBOL"], how="inner")
    if df.height == 0: return {}
    d = PROJECT_ROOT / "backtest_results_monthly" / TARGET_REGION / name; d.mkdir(parents=True, exist_ok=True)
    run_cross_section_backtest(df=df, factor_column_name="factor_value",
        datetime_column_name="DATETIME", symbol_column_name="SYMBOL",
        raw_label_column_name=label_col, freq="1mo", layers_use=LAYERS, fees=FEES,
        backtest_mode="long/short_layers",
        backtest_params={"long_layer_index": [1], "short_layer_index": [LAYERS]},
        lag=LAG, annual_days=ANNUAL_DAYS,
        factor_info_dict={"factor_name": name}, output_dir=d)
    return extract_metrics(d / "summary_df.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=12, choices=[3, 6, 12])
    args = parser.parse_args()
    lookback = args.lookback

    print(f"{'=' * 90}")
    print(f"MONTHLY GRAPH + PLACEBO — JP TOPIX 500, {lookback}mo")
    print(f"{'=' * 90}")

    jp_sectors = pl.read_csv(SYMBOL_DICT_PATH).filter(pl.col("region") == TARGET_REGION).select(
        ["SYMBOL", "gics_sector"]).filter(pl.col("gics_sector").is_not_null() & (pl.col("gics_sector") != ""))
    jp_m = load_monthly_close_prices(TARGET_REGION, "2012-01-01", END)
    us_m = load_monthly_close_prices(PEER_REGION, "2012-01-01", END)
    months = sorted(jp_m.filter((pl.col("DATETIME") >= START) & (pl.col("DATETIME") <= END))
                    .select("DATETIME").unique()["DATETIME"].to_list())
    labels = h5_load({"source_path": str(PROJECT_ROOT / "labels"), "product": "Equity",
        "region": TARGET_REGION, "freq": "1mo", "source": "tradable_close_return"},
        start_date=START, end_date=END, keys=["tradable_close_return"])
    lncap = build_true_lncap(TARGET_REGION, months)
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

    # Peer returns
    us_sectors = load_symbol_sectors(PEER_REGION)
    cum_ret = (us_m.sort(["SYMBOL", "DATETIME"])
        .with_columns(((pl.col("close") / pl.col("close").shift(lookback)) - 1).over("SYMBOL").alias("cum_return"))
        .select(["DATETIME", "SYMBOL", "cum_return"]).filter(pl.col("DATETIME").is_in(months))
        .join(us_sectors, on="SYMBOL", how="left"))
    sr = sector_relative_return_df(cum_ret, sector_col="sector", return_col="cum_return")
    peer_returns = {}
    for m in months:
        md = sr.filter(pl.col("DATETIME") == m)
        if md.height > 0: peer_returns[m] = (md["SYMBOL"].to_list(), md["sector_relative_return"].to_numpy())

    # Embedding weights
    cached = load_similarity_cache(TARGET_REGION, PEER_REGION, "w128", cache_dir=CACHE_SIM_DIR)
    emb_w = sigmoid_weights(cached.pct_ranks, k=K, c=C)
    target_syms, peer_syms = cached.target_symbols, cached.peer_symbols

    # Graph adjacency
    opus = json.load(open(PROJECT_ROOT / "backtest_results_daily" / "supply_chain_opus_full.json"))
    t2i = {s: i for i, s in enumerate(target_syms)}; u2j = {}
    for uc in opus:
        for j, s in enumerate(peer_syms):
            if s.startswith(uc + ".") or s == uc: u2j[uc] = j; break
    A = np.zeros((len(target_syms), len(peer_syms)))
    for uc, info in opus.items():
        j = u2j.get(uc)
        if j is None: continue
        for c in info.get("connections", []):
            i = t2i.get(c.get("jp_ticker", ""))
            if i is not None: A[i, j] = 1.0
    n_edges = int(A.sum())
    n_covered = int((A.sum(axis=1) > 0).sum())
    print(f"\nGraph: {n_edges} edges, {n_covered}/{len(target_syms)} JP covered", flush=True)

    def f(v, fmt=".3f"):
        return f"{v:{fmt}}" if isinstance(v, (int, float)) and np.isfinite(v) else "  N/A"

    results = []

    def test(name, weights):
        fdf = build_factor(weights, target_syms, peer_syms, peer_returns, months)
        bt_r = run_bt(fdf, labels, f"graph_{name}_raw_{lookback}mo")
        bt_r.update({"method": name, "version": "raw", "lookback": lookback})
        results.append(bt_r)
        print(f"  {name:<25} raw:    ICIR={f(bt_r.get('rank_icir'))}  Sh={f(bt_r.get('ls_sharpe'))}  Ret={f(bt_r.get('ls_ret'),'.2%')}  LgRet={f(bt_r.get('long_ret'),'.2%')}", flush=True)
        if lncap is not None:
            f_3s, r2 = three_style(fdf, lncap, jp_sectors, ind_mom, own_mom)
            if f_3s.height > 0:
                bt_3 = run_bt(f_3s, labels, f"graph_{name}_3style_{lookback}mo")
                bt_3.update({"method": name, "version": "3style", "lookback": lookback, "r2": r2})
                results.append(bt_3)
                print(f"  {name:<25} 3style: ICIR={f(bt_3.get('rank_icir'))}  Sh={f(bt_3.get('ls_sharpe'))}  Ret={f(bt_3.get('ls_ret'),'.2%')}  R²={r2:.3f}", flush=True)

    # 1. Embedding only
    print(f"\n[1] Embedding only", flush=True)
    test("emb_only", emb_w)

    # 2. Graph boost β=0.2
    print(f"\n[2] Graph boost (β=0.2)", flush=True)
    test("graph_boost_02", emb_w * (1.0 + 0.2 * A))

    # 3. Graph mask (binary)
    print(f"\n[3] Graph mask (binary)", flush=True)
    ng = A.sum(axis=1) == 0
    gw = emb_w * A; gw[ng, :] = emb_w[ng, :]
    test("graph_mask", gw)

    # 4. Multi-attested (≥2)
    print(f"\n[4] Multi-attested (≥2)", flush=True)
    attest = np.zeros(len(target_syms), dtype=int)
    for uc, info in opus.items():
        for c in info.get("connections", []):
            i = t2i.get(c.get("jp_ticker", ""))
            if i is not None: attest[i] += 1
    A2 = A.copy()
    for i in range(len(target_syms)):
        if attest[i] < 2: A2[i, :] = 0.0
    ng2 = A2.sum(axis=1) == 0
    gw2 = emb_w * A2; gw2[ng2, :] = emb_w[ng2, :]
    test("multi_attest_ge2", gw2)

    # 5. Random graph placebo (10 trials)
    print(f"\n[5] Random graph placebo (10 trials)", flush=True)
    np.random.seed(42)
    placebo_icirs = []
    for trial in range(10):
        A_rand = np.zeros_like(A)
        # Random edges with same density
        n_edges_per_row = (A.sum(axis=1) > 0).sum()
        for i in range(A.shape[0]):
            if A.sum(axis=1)[i] > 0:
                n = int(A[i].sum())
                cols = np.random.choice(A.shape[1], size=n, replace=False)
                A_rand[i, cols] = 1.0
        ng_r = A_rand.sum(axis=1) == 0
        gw_r = emb_w * A_rand; gw_r[ng_r, :] = emb_w[ng_r, :]
        f_r = build_factor(gw_r, target_syms, peer_syms, peer_returns, months)
        bt_r = run_bt(f_r, labels, f"graph_placebo_{trial}_{lookback}mo")
        placebo_icirs.append(bt_r.get("rank_icir", float("nan")))
    avg = np.nanmean(placebo_icirs)
    std = np.nanstd(placebo_icirs)
    print(f"  Random graph ICIR: {avg:.3f} ± {std:.3f}")
    real_icir = [r for r in results if r.get("method") == "graph_mask" and r.get("version") == "raw"]
    if real_icir:
        ri = real_icir[0].get("rank_icir", float("nan"))
        z = (ri - avg) / std if std > 0 else 0
        print(f"  Real graph ICIR: {ri:.3f}, z-score vs random: {z:.2f}")
    results.append({"method": "random_graph", "version": "raw", "lookback": lookback,
                    "rank_icir": avg, "icir_std": std})

    # Summary
    print(f"\n{'=' * 90}")
    print(f"GRAPH SUMMARY — {lookback}mo")
    print(f"{'=' * 90}")
    print(f"{'Method':<25}{'Ver':<10}{'ICIR':>7}{'Sharpe':>8}{'Ret':>8}{'LgRet':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r.get('method',''):<25}{r.get('version',''):<10}"
              f"{f(r.get('rank_icir')):>7}{f(r.get('ls_sharpe')):>8}"
              f"{f(r.get('ls_ret'),'.2%'):>8}{f(r.get('long_ret'),'.2%'):>8}")

    out = PROJECT_ROOT / "backtest_results_daily" / f"graph_monthly_{lookback}mo.json"
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
