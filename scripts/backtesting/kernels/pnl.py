"""Numba kernels for transaction-PnL computation."""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange


@njit(cache=True, nogil=True, inline="always")
def _tpnl_step(
    pos: float,
    ref: float,
    fee: float,
    prev_shares: float,
    prev_pos: float,
) -> tuple[float, float, float]:
    old_pos = prev_shares * ref
    trade = pos - old_pos
    tpnl = old_pos - prev_pos - abs(trade) * fee
    return tpnl, pos / ref, pos


@njit(cache=True, nogil=True, parallel=True)
def tpnl_obs_layer_membership_cs(
    layer_code: np.ndarray,
    layer_pos: np.ndarray,
    ref_price: np.ndarray,
    sym_offsets: np.ndarray,
    sym_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute observation-level tpnl for per-layer equal-weight portfolios (fee=0).

    Each observation belongs to at most one layer portfolio (layer_code in 1..L) or
    to none (0). `layer_pos[i]` is the position value for that membership at row i
    (0 if layer_code[i] == 0). The resulting tpnl at row i is attributed to the
    *previous* layer membership (the one held coming into row i), matching the
    legacy tpnl definition.

    Returns
    -------
    tpnl_obs:
        Per-row tpnl contribution (aligned to input row order).
    prev_layer_code:
        Per-row previous layer code (0..L) that owns `tpnl_obs[i]`.
    """
    n_rows = ref_price.shape[0]
    n_syms = sym_offsets.shape[0] - 1

    tpnl_obs = np.empty(n_rows, dtype=np.float64)
    prev_layer_out = np.empty(n_rows, dtype=np.int32)

    for s in prange(n_syms):
        prev_shares = 0.0
        prev_pos = 0.0
        prev_layer = 0

        for k in range(sym_offsets[s], sym_offsets[s + 1]):
            i = sym_rows[k]
            ref = ref_price[i]

            tpnl_obs[i] = prev_shares * ref - prev_pos
            prev_layer_out[i] = prev_layer

            cur_pos = layer_pos[i]
            prev_shares = cur_pos / ref
            prev_pos = cur_pos
            prev_layer = int(layer_code[i])

    return tpnl_obs, prev_layer_out


@njit(cache=True, nogil=True, parallel=True)
def sum_tpnl_by_dt_and_layer(
    tpnl_obs: np.ndarray,
    prev_layer_code: np.ndarray,
    dt_offsets_rows: np.ndarray,
    n_layers: int,
) -> np.ndarray:
    """Aggregate per-row tpnl into (datetime x layer) totals."""
    n_dt = dt_offsets_rows.shape[0] - 1
    out = np.zeros((n_dt, n_layers), dtype=np.float64)

    for d in prange(n_dt):
        start = dt_offsets_rows[d]
        end = dt_offsets_rows[d + 1]
        for i in range(start, end):
            lc = int(prev_layer_code[i])
            if 1 <= lc <= n_layers:
                out[d, lc - 1] += tpnl_obs[i]

    return out


@njit(cache=True, nogil=True, parallel=True)
def tpnl_obs_cs_4(
    pos_ls: np.ndarray,
    pos_long: np.ndarray,
    pos_short: np.ndarray,
    pos_passive: np.ndarray,
    ref_price: np.ndarray,
    sym_offsets: np.ndarray,
    sym_rows: np.ndarray,
    fee: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute observation-level tpnl for 4 strategies in CS mode.

    Notes
    -----
    Matches legacy logic:
      old_pos = (pos/ref_price).shift(1).over(symbol) * ref_price
      trade   = pos - old_pos
      tpnl    = old_pos - pos.shift(1).over(symbol) - abs(trade)*fee

    The work parallelizes over symbols; each symbol path is sequential.
    """
    n_rows = ref_price.shape[0]
    n_syms = sym_offsets.shape[0] - 1

    out_ls = np.empty(n_rows, dtype=np.float64)
    out_long = np.empty(n_rows, dtype=np.float64)
    out_short = np.empty(n_rows, dtype=np.float64)
    out_passive = np.empty(n_rows, dtype=np.float64)

    wins = np.zeros((n_syms, 4), dtype=np.int64)
    active = np.zeros((n_syms, 4), dtype=np.int64)

    for s in prange(n_syms):
        prev_shares_ls = 0.0
        prev_shares_long = 0.0
        prev_shares_short = 0.0
        prev_shares_passive = 0.0

        prev_pos_ls = 0.0
        prev_pos_long = 0.0
        prev_pos_short = 0.0
        prev_pos_passive = 0.0

        wins_ls = 0
        wins_long = 0
        wins_short = 0
        wins_passive = 0

        active_ls = 0
        active_long = 0
        active_short = 0
        active_passive = 0

        for k in range(sym_offsets[s], sym_offsets[s + 1]):
            i = sym_rows[k]
            ref = ref_price[i]

            pos = pos_ls[i]
            tpnl, next_shares, next_pos = _tpnl_step(pos, ref, fee, prev_shares_ls, prev_pos_ls)
            out_ls[i] = tpnl
            if pos != 0.0:
                active_ls += 1
                if tpnl > 0.0:
                    wins_ls += 1
            prev_shares_ls = next_shares
            prev_pos_ls = next_pos

            pos = pos_long[i]
            tpnl, next_shares, next_pos = _tpnl_step(
                pos, ref, fee, prev_shares_long, prev_pos_long
            )
            out_long[i] = tpnl
            if pos != 0.0:
                active_long += 1
                if tpnl > 0.0:
                    wins_long += 1
            prev_shares_long = next_shares
            prev_pos_long = next_pos

            pos = pos_short[i]
            tpnl, next_shares, next_pos = _tpnl_step(
                pos, ref, fee, prev_shares_short, prev_pos_short
            )
            out_short[i] = tpnl
            if pos != 0.0:
                active_short += 1
                if tpnl > 0.0:
                    wins_short += 1
            prev_shares_short = next_shares
            prev_pos_short = next_pos

            pos = pos_passive[i]
            tpnl, next_shares, next_pos = _tpnl_step(
                pos, ref, fee, prev_shares_passive, prev_pos_passive
            )
            out_passive[i] = tpnl
            if pos != 0.0:
                active_passive += 1
                if tpnl > 0.0:
                    wins_passive += 1
            prev_shares_passive = next_shares
            prev_pos_passive = next_pos

        wins[s, 0] = wins_ls
        wins[s, 1] = wins_long
        wins[s, 2] = wins_short
        wins[s, 3] = wins_passive

        active[s, 0] = active_ls
        active[s, 1] = active_long
        active[s, 2] = active_short
        active[s, 3] = active_passive

    return out_ls, out_long, out_short, out_passive, wins, active


@njit(nogil=True, parallel=True)
def tpnl_dt_components_cs_4_multi_fee(
    pos_ls: np.ndarray,
    pos_long: np.ndarray,
    pos_short: np.ndarray,
    pos_passive: np.ndarray,
    ref_price: np.ndarray,
    sym_offsets: np.ndarray,
    sym_rows: np.ndarray,
    dt_offsets_rows: np.ndarray,
    fees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute datetime-level tpnl components for multiple fee scenarios in one pass.

    For each strategy, we decompose:
      tpnl = base_pnl - abs_trade * fee

    where (per row):
      base_pnl  = old_pos - prev_pos
      abs_trade = abs(pos - old_pos)
      old_pos   = prev_shares * ref_price

    This allows:
      dt_tpnl(fee) = dt_base_pnl - fee * dt_abs_trade

    Returns
    -------
    dt_base:
        float64 array, shape (n_dt, 4) with columns [ls, long, short, long_passive]
    dt_abs_trade:
        float64 array, shape (n_dt, 4) with columns [ls, long, short, long_passive]
    wins:
        int64 array, shape (n_fees, 5) counting tpnl>0 where pos!=0 for each fee,
        with columns [ls, long, short, long_passive, short_passive]
    active:
        int64 array, shape (5,) counting pos!=0 (fee-independent), with the same
        column order as `wins`
    """
    n_rows = ref_price.shape[0]
    n_syms = sym_offsets.shape[0] - 1
    n_dt = dt_offsets_rows.shape[0] - 1
    n_fees = fees.shape[0]

    dt_base = np.zeros((n_dt, 4), dtype=np.float64)
    dt_abs_trade = np.zeros((n_dt, 4), dtype=np.float64)
    wins = np.zeros((n_fees, 5), dtype=np.int64)
    active = np.zeros(5, dtype=np.int64)

    if n_rows == 0 or n_syms <= 0 or n_dt <= 0:
        return dt_base, dt_abs_trade, wins, active

    n_threads = get_num_threads()
    base_thread = np.zeros((n_threads, n_dt, 4), dtype=np.float64)
    abs_thread = np.zeros((n_threads, n_dt, 4), dtype=np.float64)
    wins_thread = np.zeros((n_threads, n_fees, 5), dtype=np.int64)
    active_thread = np.zeros((n_threads, 5), dtype=np.int64)

    for s in prange(n_syms):
        tid = get_thread_id()
        bt = base_thread[tid]
        at = abs_thread[tid]
        wt = wins_thread[tid]
        act = active_thread[tid]

        prev_shares_ls = 0.0
        prev_shares_long = 0.0
        prev_shares_short = 0.0
        prev_shares_passive = 0.0

        prev_pos_ls = 0.0
        prev_pos_long = 0.0
        prev_pos_short = 0.0
        prev_pos_passive = 0.0

        d = 0
        next_dt_end = int(dt_offsets_rows[1])

        for k in range(sym_offsets[s], sym_offsets[s + 1]):
            i = int(sym_rows[k])
            while i >= next_dt_end:
                d += 1
                next_dt_end = int(dt_offsets_rows[d + 1])

            ref = ref_price[i]
            inv_ref = 1.0 / ref

            pos = pos_ls[i]
            old_pos = prev_shares_ls * ref
            trade = pos - old_pos
            base = old_pos - prev_pos_ls
            abs_trade = abs(trade)
            bt[d, 0] += base
            at[d, 0] += abs_trade
            if pos != 0.0:
                act[0] += 1
                for f in range(n_fees):
                    if base - abs_trade * fees[f] > 0.0:
                        wt[f, 0] += 1
            prev_shares_ls = pos * inv_ref
            prev_pos_ls = pos

            pos = pos_long[i]
            old_pos = prev_shares_long * ref
            trade = pos - old_pos
            base = old_pos - prev_pos_long
            abs_trade = abs(trade)
            bt[d, 1] += base
            at[d, 1] += abs_trade
            if pos != 0.0:
                act[1] += 1
                for f in range(n_fees):
                    if base - abs_trade * fees[f] > 0.0:
                        wt[f, 1] += 1
            prev_shares_long = pos * inv_ref
            prev_pos_long = pos

            pos = pos_short[i]
            old_pos = prev_shares_short * ref
            trade = pos - old_pos
            base = old_pos - prev_pos_short
            abs_trade = abs(trade)
            bt[d, 2] += base
            at[d, 2] += abs_trade
            if pos != 0.0:
                act[2] += 1
                for f in range(n_fees):
                    if base - abs_trade * fees[f] > 0.0:
                        wt[f, 2] += 1
            prev_shares_short = pos * inv_ref
            prev_pos_short = pos

            pos = pos_passive[i]
            old_pos = prev_shares_passive * ref
            trade = pos - old_pos
            base = old_pos - prev_pos_passive
            abs_trade = abs(trade)
            bt[d, 3] += base
            at[d, 3] += abs_trade
            if pos != 0.0:
                act[3] += 1
                act[4] += 1
                for f in range(n_fees):
                    if base - abs_trade * fees[f] > 0.0:
                        wt[f, 3] += 1
                    if (-base) - abs_trade * fees[f] > 0.0:
                        wt[f, 4] += 1
            prev_shares_passive = pos * inv_ref
            prev_pos_passive = pos

    for tid in range(n_threads):
        for d in range(n_dt):
            dt_base[d, 0] += base_thread[tid, d, 0]
            dt_base[d, 1] += base_thread[tid, d, 1]
            dt_base[d, 2] += base_thread[tid, d, 2]
            dt_base[d, 3] += base_thread[tid, d, 3]

            dt_abs_trade[d, 0] += abs_thread[tid, d, 0]
            dt_abs_trade[d, 1] += abs_thread[tid, d, 1]
            dt_abs_trade[d, 2] += abs_thread[tid, d, 2]
            dt_abs_trade[d, 3] += abs_thread[tid, d, 3]

    for tid in range(n_threads):
        for f in range(n_fees):
            wins[f, 0] += wins_thread[tid, f, 0]
            wins[f, 1] += wins_thread[tid, f, 1]
            wins[f, 2] += wins_thread[tid, f, 2]
            wins[f, 3] += wins_thread[tid, f, 3]
            wins[f, 4] += wins_thread[tid, f, 4]

        active[0] += active_thread[tid, 0]
        active[1] += active_thread[tid, 1]
        active[2] += active_thread[tid, 2]
        active[3] += active_thread[tid, 3]
        active[4] += active_thread[tid, 4]

    return dt_base, dt_abs_trade, wins, active


@njit(cache=True, nogil=True, parallel=True)
def tpnl_obs_cs_single(
    pos: np.ndarray,
    ref_price: np.ndarray,
    sym_offsets: np.ndarray,
    sym_rows: np.ndarray,
    fee: float,
) -> np.ndarray:
    """Compute observation-level tpnl for a single strategy in CS mode."""
    n_rows = ref_price.shape[0]
    n_syms = sym_offsets.shape[0] - 1

    out = np.empty(n_rows, dtype=np.float64)

    for s in prange(n_syms):
        prev_shares = 0.0
        prev_pos = 0.0

        for k in range(sym_offsets[s], sym_offsets[s + 1]):
            i = sym_rows[k]
            ref = ref_price[i]
            p = pos[i]
            tpnl, prev_shares, prev_pos = _tpnl_step(p, ref, fee, prev_shares, prev_pos)
            out[i] = tpnl

    return out


@njit(cache=True, nogil=True)
def tpnl_ts_4(
    pos_ls: np.ndarray,
    pos_long: np.ndarray,
    pos_short: np.ndarray,
    pos_passive: np.ndarray,
    ref_price: np.ndarray,
    fee: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-datetime tpnl for 4 strategies in TS mode."""
    n = ref_price.shape[0]
    out_ls = np.empty(n, dtype=np.float64)
    out_long = np.empty(n, dtype=np.float64)
    out_short = np.empty(n, dtype=np.float64)
    out_passive = np.empty(n, dtype=np.float64)

    wins = np.empty(4, dtype=np.int64)
    active = np.empty(4, dtype=np.int64)

    prev_shares_ls = 0.0
    prev_shares_long = 0.0
    prev_shares_short = 0.0
    prev_shares_passive = 0.0

    prev_pos_ls = 0.0
    prev_pos_long = 0.0
    prev_pos_short = 0.0
    prev_pos_passive = 0.0

    wins_ls = 0
    wins_long = 0
    wins_short = 0
    wins_passive = 0

    active_ls = 0
    active_long = 0
    active_short = 0
    active_passive = 0

    for i in range(n):
        ref = ref_price[i]

        pos = pos_ls[i]
        tpnl, next_shares, next_pos = _tpnl_step(pos, ref, fee, prev_shares_ls, prev_pos_ls)
        out_ls[i] = tpnl
        if pos != 0.0:
            active_ls += 1
            if tpnl > 0.0:
                wins_ls += 1
        prev_shares_ls = next_shares
        prev_pos_ls = next_pos

        pos = pos_long[i]
        tpnl, next_shares, next_pos = _tpnl_step(
            pos, ref, fee, prev_shares_long, prev_pos_long
        )
        out_long[i] = tpnl
        if pos != 0.0:
            active_long += 1
            if tpnl > 0.0:
                wins_long += 1
        prev_shares_long = next_shares
        prev_pos_long = next_pos

        pos = pos_short[i]
        tpnl, next_shares, next_pos = _tpnl_step(
            pos, ref, fee, prev_shares_short, prev_pos_short
        )
        out_short[i] = tpnl
        if pos != 0.0:
            active_short += 1
            if tpnl > 0.0:
                wins_short += 1
        prev_shares_short = next_shares
        prev_pos_short = next_pos

        pos = pos_passive[i]
        tpnl, next_shares, next_pos = _tpnl_step(
            pos, ref, fee, prev_shares_passive, prev_pos_passive
        )
        out_passive[i] = tpnl
        if pos != 0.0:
            active_passive += 1
            if tpnl > 0.0:
                wins_passive += 1
        prev_shares_passive = next_shares
        prev_pos_passive = next_pos

    wins[0] = wins_ls
    wins[1] = wins_long
    wins[2] = wins_short
    wins[3] = wins_passive

    active[0] = active_ls
    active[1] = active_long
    active[2] = active_short
    active[3] = active_passive

    return out_ls, out_long, out_short, out_passive, wins, active
