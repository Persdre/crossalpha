#!/usr/bin/env python3
"""MVP: US EMA(n) mean-reversion projected onto JP via w128+gb20 graph.

Tests whether a fast US signal (EMA deviation, mean-reversion sign) projected
through the existing production similarity matrix produces a usable monthly
cross-sectional JP factor.

Usage:
    python scripts/run_us_ema_mr_to_jp_mvp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.kernels import (
    load_daily_close_prices,
    load_monthly_close_prices,
    sigmoid_weights,
)
from scripts.similarity_cache import load_similarity_cache
from scripts.h5_data.h5_utils import h5_load

DATA_ROOT = Path("${CROSSALPHA_DATA_ROOT}")
CACHE_SIM_DIR = DATA_ROOT / "cache" / "similarity"
SYMBOL_DICT_PATH = PROJECT_ROOT / "docs" / "SymbolDict.csv"
TARGET_REGION = "jp_topix_500"
PEER_REGION = "us_russell_1000"
START, END = "2015-01-01", "2025-12-31"
C, K = 0.99, 50.0
BETA = 0.2
LOOKBACKS = [10, 20, 60]
MIN_STOCKS = 10


def _winsorize(x, lo=0.01, hi=0.99):
    a, b = np.nanpercentile(x, lo * 100), np.nanpercentile(x, hi * 100)
    return np.clip(x, a, b)


def _zscore(x):
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / sd if sd > 1e-12 else np.zeros_like(x)


def compute_us_ema_dev(us_daily: pl.DataFrame, span: int) -> pl.DataFrame:
    """Per-symbol EMA(span) deviation: close / EMA(close) - 1, daily."""
    return (
        us_daily.sort(["SYMBOL", "DATETIME"])
        .with_columns(
            pl.col("close")
            .ewm_mean(span=span, adjust=False)
            .over("SYMBOL")
            .alias("ema")
        )
        .with_columns((pl.col("close") / pl.col("ema") - 1.0).alias("ema_dev"))
        .select(["DATETIME", "SYMBOL", "ema_dev"])
        .drop_nulls("ema_dev")
    )


def us_signal_at_jp_month_ends(
    ema_dev: pl.DataFrame, jp_months: list[str]
) -> dict[str, tuple[list[str], np.ndarray]]:
    """For each JP month-end date, take US signal from the last US trading day <= that date.

    Returns {jp_month_end: (symbols, negated_zscored_winsorized_ema_dev)}.
    MR convention: negate so +signal = undervalued = long.
    """
    us_dates = sorted(ema_dev["DATETIME"].unique().to_list())
    jp_to_us: dict[str, str] = {}
    i = 0
    for jm in sorted(jp_months):
        while i + 1 < len(us_dates) and us_dates[i + 1] <= jm:
            i += 1
        if us_dates[i] <= jm:
            jp_to_us[jm] = us_dates[i]

    out: dict[str, tuple[list[str], np.ndarray]] = {}
    for jm, ud in jp_to_us.items():
        d = ema_dev.filter(pl.col("DATETIME") == ud)
        if d.height < MIN_STOCKS:
            continue
        syms = d["SYMBOL"].to_list()
        vals = d["ema_dev"].to_numpy().astype(np.float64)
        vals = _zscore(_winsorize(vals))
        vals = -vals  # mean-reversion: +signal = undervalued
        out[jm] = (syms, vals)
    return out


def build_factor(weights, tsyms, psyms, signal_dict, months):
    p2i = {s: i for i, s in enumerate(psyms)}
    recs = []
    for m in months:
        if m not in signal_dict:
            continue
        ps, pv = signal_dict[m]
        a = np.full(len(psyms), np.nan)
        for s, r in zip(ps, pv):
            if s in p2i:
                a[p2i[s]] = r
        v = ~np.isnan(a)
        rm = np.where(v, a, 0.0)
        ws = weights @ rm
        wd = weights @ v.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            sc = np.where(wd > 1, ws / wd, np.nan)
        for sym, val in zip(tsyms, sc):
            if np.isfinite(val):
                recs.append({"DATETIME": m, "SYMBOL": sym, "factor_value": float(val)})
    return pl.DataFrame(recs)


def ms_neutralize(
    factor_df: pl.DataFrame, jp_monthly_close: pl.DataFrame, sectors: pl.DataFrame
) -> pl.DataFrame:
    """MS-style: cross-sectional OLS on gics_sector dummies + ln(close). Return residual."""
    merged = (
        factor_df.join(
            jp_monthly_close.select(["DATETIME", "SYMBOL", "close"]),
            on=["DATETIME", "SYMBOL"],
            how="inner",
        )
        .join(sectors, on="SYMBOL", how="inner")
        .with_columns(pl.col("close").log().alias("lnclose"))
    )
    recs = []
    for dt in merged["DATETIME"].unique().sort().to_list():
        d = merged.filter(pl.col("DATETIME") == dt)
        y = d["factor_value"].to_numpy().astype(np.float64)
        lc = d["lnclose"].to_numpy().astype(np.float64)
        secs = d["gics_sector"].to_list()
        syms = d["SYMBOL"].to_list()
        m = np.isfinite(y) & np.isfinite(lc)
        if m.sum() < MIN_STOCKS:
            continue
        y2, lc2 = y[m], lc[m]
        sec2 = [s for s, mm in zip(secs, m) if mm]
        sym2 = [s for s, mm in zip(syms, m) if mm]
        uniq = sorted(set(sec2))
        # one-hot sector, drop first to avoid collinearity with intercept
        D = np.zeros((len(sec2), len(uniq) - 1))
        for j, s in enumerate(uniq[1:]):
            D[:, j] = [1.0 if x == s else 0.0 for x in sec2]
        X = np.column_stack([np.ones(len(y2)), _zscore(_winsorize(lc2)), D])
        try:
            b, _, _, _ = np.linalg.lstsq(X, y2, rcond=None)
        except Exception:
            continue
        resid = y2 - X @ b
        for s, v in zip(sym2, resid):
            recs.append({"DATETIME": dt, "SYMBOL": s, "factor_value": float(v)})
    return pl.DataFrame(recs) if recs else pl.DataFrame()


def rank_ic_series(factor_df: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Lag=1 Rank IC: factor at M vs tradable_close_return at M+1."""
    label_col = [c for c in labels.columns if c not in ("DATETIME", "SYMBOL")][0]
    months = sorted(factor_df["DATETIME"].unique().to_list())
    recs = []
    for i, m in enumerate(months[:-1]):
        next_m = months[i + 1]
        f_m = factor_df.filter(pl.col("DATETIME") == m)
        l_m = labels.filter(pl.col("DATETIME") == next_m)
        j = f_m.join(l_m, on="SYMBOL", how="inner", suffix="_l")
        if j.height < MIN_STOCKS:
            continue
        f = j["factor_value"].to_numpy()
        lab_col = label_col if label_col in j.columns else f"{label_col}"
        y = j[lab_col].to_numpy()
        mask = np.isfinite(f) & np.isfinite(y)
        if mask.sum() < MIN_STOCKS:
            continue
        r, _ = spearmanr(f[mask], y[mask])
        recs.append({"DATETIME": m, "rank_ic": float(r), "n": int(mask.sum())})
    return pl.DataFrame(recs)


def summarize(ic_df: pl.DataFrame, label: str) -> dict:
    if ic_df.height == 0:
        return {"label": label, "n_months": 0}
    ic = ic_df["rank_ic"].to_numpy()
    ic = ic[np.isfinite(ic)]
    if len(ic) == 0:
        return {"label": label, "n_months": 0}
    mean_ic = float(np.mean(ic))
    std_ic = float(np.std(ic, ddof=1)) if len(ic) > 1 else float("nan")
    icir_annual = mean_ic / std_ic * np.sqrt(12) if std_ic and std_ic > 1e-12 else float("nan")
    t_stat = mean_ic / (std_ic / np.sqrt(len(ic))) if std_ic and std_ic > 1e-12 else float("nan")
    return {
        "label": label,
        "n_months": len(ic),
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir_annual": float(icir_annual),
        "t_stat": float(t_stat),
        "hit_rate": float(np.mean(ic > 0)),
    }


def build_weights(target_syms, peer_syms):
    cached = load_similarity_cache(
        TARGET_REGION, PEER_REGION, "w128", cache_dir=CACHE_SIM_DIR
    )
    emb_w = sigmoid_weights(cached.pct_ranks, k=K, c=C)
    assert cached.target_symbols == target_syms
    assert cached.peer_symbols == peer_syms

    opus = json.load(
        open(PROJECT_ROOT / "backtest_results_daily" / "supply_chain_opus_full.json")
    )
    t2i = {s: i for i, s in enumerate(target_syms)}
    u2j = {}
    for uc in opus:
        for j, s in enumerate(peer_syms):
            if s.startswith(uc + ".") or s == uc:
                u2j[uc] = j
                break
    A = np.zeros((len(target_syms), len(peer_syms)))
    for uc, info in opus.items():
        j = u2j.get(uc)
        if j is None:
            continue
        for c in info.get("connections", []):
            i = t2i.get(c.get("jp_ticker", ""))
            if i is not None:
                A[i, j] = 1.0
    return emb_w * (1.0 + BETA * A), int(A.sum())


def main():
    print("=" * 90)
    print("MVP: US EMA(n) mean-reversion → JP via w128+gb20 (monthly IC test)")
    print("=" * 90)

    # JP monthly universe + labels
    jp_m = load_monthly_close_prices(TARGET_REGION, "2012-01-01", END)
    months = sorted(
        jp_m.filter((pl.col("DATETIME") >= START) & (pl.col("DATETIME") <= END))
        .select("DATETIME")
        .unique()["DATETIME"]
        .to_list()
    )
    labels = h5_load(
        {
            "source_path": str(PROJECT_ROOT / "labels"),
            "product": "Equity",
            "region": TARGET_REGION,
            "freq": "1mo",
            "source": "tradable_close_return",
        },
        start_date=START,
        end_date=END,
        keys=["tradable_close_return"],
    )
    jp_sectors = (
        pl.read_csv(SYMBOL_DICT_PATH)
        .filter(pl.col("region") == TARGET_REGION)
        .select(["SYMBOL", "gics_sector"])
        .filter(pl.col("gics_sector").is_not_null() & (pl.col("gics_sector") != ""))
    )
    jp_monthly_close = jp_m.filter(pl.col("DATETIME").is_in(months))

    # Similarity (target/peer symbol order comes from cache)
    cached = load_similarity_cache(
        TARGET_REGION, PEER_REGION, "w128", cache_dir=CACHE_SIM_DIR
    )
    target_syms, peer_syms = cached.target_symbols, cached.peer_symbols
    W, n_edges = build_weights(target_syms, peer_syms)
    print(f"Weight matrix: {W.shape}, graph edges: {n_edges}")

    # US daily prices — once, covering enough history for EMA(60) warmup
    us_daily = load_daily_close_prices(PEER_REGION, "2014-06-01", END)
    print(f"US daily rows: {us_daily.height}, symbols: {us_daily['SYMBOL'].n_unique()}")

    all_results = []
    for span in LOOKBACKS:
        print(f"\n--- EMA({span}) ---")
        ema_dev = compute_us_ema_dev(us_daily, span)
        signal = us_signal_at_jp_month_ends(ema_dev, months)
        print(f"  month-ends with US signal: {len(signal)}/{len(months)}")

        fdf = build_factor(W, target_syms, peer_syms, signal, months)
        print(f"  raw factor rows: {fdf.height}")

        fdf_n = ms_neutralize(fdf, jp_monthly_close, jp_sectors)
        print(f"  neutralized factor rows: {fdf_n.height}")

        ic_raw = rank_ic_series(fdf, labels)
        ic_neu = rank_ic_series(fdf_n, labels) if fdf_n.height > 0 else pl.DataFrame()

        sraw = summarize(ic_raw, f"ema{span}_raw")
        sneu = summarize(ic_neu, f"ema{span}_neutralized")
        for s in (sraw, sneu):
            print(
                f"  {s['label']:<28} n={s.get('n_months',0):>3}  "
                f"IC={s.get('mean_ic', float('nan')):+.4f}  "
                f"ICIR={s.get('icir_annual', float('nan')):+.3f}  "
                f"t={s.get('t_stat', float('nan')):+.2f}  "
                f"hit={s.get('hit_rate', float('nan')):.2f}"
            )
        all_results.append(
            {
                "span": span,
                "raw": sraw,
                "neutralized": sneu,
                "ic_series_raw": ic_raw.to_dicts() if ic_raw.height else [],
                "ic_series_neu": ic_neu.to_dicts() if ic_neu.height else [],
            }
        )

    print("\n" + "=" * 90)
    print(f"{'span':<6}{'version':<14}{'n':>4}{'IC':>10}{'ICIR':>9}{'t':>7}{'hit':>7}")
    print("-" * 60)
    for r in all_results:
        for v in ("raw", "neutralized"):
            s = r[v]
            print(
                f"{r['span']:<6}{v:<14}{s.get('n_months',0):>4}"
                f"{s.get('mean_ic', float('nan')):>+10.4f}"
                f"{s.get('icir_annual', float('nan')):>+9.3f}"
                f"{s.get('t_stat', float('nan')):>+7.2f}"
                f"{s.get('hit_rate', float('nan')):>7.2f}"
            )

    out = PROJECT_ROOT / "backtest_results_daily" / "us_ema_mr_to_jp_mvp.json"
    json.dump(all_results, open(out, "w"), indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
