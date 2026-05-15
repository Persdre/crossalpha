"""Input builders: polars/pandas objects -> numpy arrays + grouping metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from numba import njit


@dataclass(frozen=True, slots=True)
class GroupOffsets:
    """Contiguous group offsets (CSR-style) for 1D arrays."""

    offsets: np.ndarray  # shape (n_groups + 1,), int64

    @property
    def n_groups(self) -> int:
        return int(self.offsets.shape[0] - 1)


@dataclass(frozen=True, slots=True)
class CSRIndex:
    """CSR index for grouping rows by key (e.g., symbol)."""

    offsets: np.ndarray  # shape (n_keys + 1,), int64
    rows: np.ndarray  # shape (n_rows,), int64

    @property
    def n_keys(self) -> int:
        return int(self.offsets.shape[0] - 1)


def build_group_offsets_from_sorted_keys(keys: Sequence) -> Tuple[List, GroupOffsets]:
    """Build unique keys and group offsets from a sorted key sequence."""
    n = len(keys)
    if n == 0:
        return [], GroupOffsets(np.array([0], dtype=np.int64))

    arr = np.asarray(keys)
    # `keys` is sorted, so group boundaries are where consecutive values differ.
    boundaries = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    offsets = np.empty(boundaries.size + 2, dtype=np.int64)
    offsets[0] = 0
    offsets[1:-1] = boundaries.astype(np.int64, copy=False)
    offsets[-1] = n
    uniques = arr[offsets[:-1]].tolist()
    return uniques, GroupOffsets(offsets)


def build_day_offsets_from_sorted_dates(dates: Sequence) -> Tuple[List, GroupOffsets]:
    """Specialized alias for date grouping (kept for readability)."""
    return build_group_offsets_from_sorted_keys(dates)


@njit(cache=True, nogil=True)
def _stable_rows_from_codes(codes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Stable grouping of row indices by integer codes.

    Produces a concatenation of rows for code=0..K-1, where within each code the
    original row order is preserved.
    """
    n = codes.shape[0]
    out = np.empty(n, dtype=np.int64)
    cursor = offsets[:-1].copy()
    for i in range(n):
        c = int(codes[i])
        pos = cursor[c]
        out[pos] = i
        cursor[c] = pos + 1
    return out


def build_symbol_csr_from_codes(codes: np.ndarray, n_symbols: int | None = None) -> CSRIndex:
    """Build CSR index for rows grouped by integer symbol codes (0..n_symbols-1)."""
    n_rows = int(codes.shape[0])
    if n_rows == 0:
        empty = np.array([0], dtype=np.int64)
        return CSRIndex(empty, empty[:0])

    if n_symbols is None:
        n_symbols = int(codes.max()) + 1

    counts = np.bincount(codes.astype(np.int64, copy=False), minlength=n_symbols).astype(
        np.int64, copy=False
    )
    offsets = np.empty(n_symbols + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    rows = _stable_rows_from_codes(codes.astype(np.int64, copy=False), offsets)
    return CSRIndex(offsets=offsets, rows=rows)


def build_symbol_csr(symbols_in_row_order: Sequence) -> Tuple[List, CSRIndex]:
    """Build CSR index for rows grouped by symbol, preserving time order within symbol.

    `symbols_in_row_order` must be sorted by (datetime, symbol), so iterating rows
    preserves chronological order for each symbol.
    """
    if len(symbols_in_row_order) == 0:
        empty = np.array([0], dtype=np.int64)
        return [], CSRIndex(empty, empty[:0])

    sym_arr = np.asarray(symbols_in_row_order)
    unique_symbols, inv = np.unique(sym_arr, return_inverse=True)
    counts = np.bincount(inv, minlength=unique_symbols.size).astype(np.int64, copy=False)

    offsets = np.empty(unique_symbols.size + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts, dtype=np.int64)

    rows = _stable_rows_from_codes(inv.astype(np.int64, copy=False), offsets)
    return unique_symbols.tolist(), CSRIndex(offsets=offsets, rows=rows)
