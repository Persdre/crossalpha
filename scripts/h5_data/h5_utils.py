"""Fast H5 utilities with Polars-friendly APIs.

This module is designed for high-throughput daily-bar pipelines where the data
is stored as one H5 file per date with the following datasets:

- DATA:   2D float matrix (n_symbols x n_fields)
- DATETIME: scalar date string (YYYY-MM-DD)
- KEY:    1D array of field names (strings)
- SYMBOL: 1D array of symbol tickers (strings)

It provides:
- A zero-pandas, fast "copy/select/rename" path for the common case where you
  just need to subset/rename fields and (optionally) symbols while writing new
  per-date H5 files.
- Optional helpers to load/save into Polars long-form frames.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypedDict

import h5py
import numpy as np
import polars as pl


class H5Params(TypedDict):
    """Parameter dictionary for H5 directory layout."""

    source_path: str
    product: str
    region: str
    freq: str
    source: str


ProgressCallback = Callable[[int, int], None]

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def build_h5_dir(params: H5Params) -> Path:
    """Return the directory for a given (source_path, product, region, freq, source)."""

    return (
        Path(params["source_path"])
        / params["product"]
        / params["region"]
        / params["freq"]
        / params["source"]
    )


def list_h5_files(
    params: H5Params,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[Path]:
    """List date-named ``.h5`` files (sorted), optionally filtered by date range."""

    load_dir = build_h5_dir(params)
    if not load_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {load_dir}")

    h5_files = sorted(load_dir.glob("*.h5"))
    if not (start_date or end_date):
        return [f for f in h5_files if _DATE_RE.match(f.stem)]

    start = start_date or "0000-01-01"
    end = end_date or "9999-12-31"

    out: list[Path] = []
    for f in h5_files:
        stem = f.stem
        if not _DATE_RE.match(stem):
            continue
        if start <= stem <= end:
            out.append(f)
    return out


def _decode_str_array(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "S":
        return np.char.decode(arr, "utf-8")
    return arr.astype(str, copy=False)


def _decode_str_scalar(val: object) -> str:
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8")
    return str(val)


def _match_sorted_bytes(
    source: np.ndarray,
    dest: np.ndarray,
) -> np.ndarray:
    """Return indices of ``dest`` within ``source`` (or -1 for missing).

    Fast path uses ``np.searchsorted`` (requires ``source`` sorted). If some
    values don't match, falls back to a hash-map lookup for those entries.
    """

    src = source if source.dtype.kind == "S" else np.asarray(source, dtype="S")
    dst = dest if dest.dtype.kind == "S" else np.asarray(dest, dtype="S")

    if dst.size == 0:
        return np.empty(0, dtype=np.int64)

    pos = np.searchsorted(src, dst)
    matched = np.full(len(dst), -1, dtype=np.int64)

    in_range = (pos >= 0) & (pos < len(src))
    pos_clip = pos.clip(0, len(src) - 1)
    ok = in_range & (src[pos_clip] == dst)
    matched[ok] = pos[ok]

    if np.any(~ok):
        src_map = {k: i for i, k in enumerate(src.tolist())}
        for i in np.flatnonzero(~ok):
            matched[i] = src_map.get(dst[i], -1)

    return matched


def _resolve_symbol_indices(
    *,
    source_symbols: np.ndarray,
    requested_symbols: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dest_symbols, src_row_indices_or_-1).

    dest_symbols are sorted if requested_symbols is provided.
    src_row_indices has the same length as dest_symbols, with -1 for missing.
    """

    src_syms = _decode_str_array(source_symbols)

    if not requested_symbols:
        dest_syms = src_syms
        return dest_syms, np.arange(len(dest_syms), dtype=np.int64)

    # Match prior pipeline behavior: output symbols are deterministic.
    seen: set[str] = set()
    ordered: list[str] = []
    for sym in requested_symbols:
        s = str(sym)
        if s in seen:
            continue
        seen.add(s)
        ordered.append(s)
    dest_syms = np.array(ordered, dtype=str)

    # If source symbols are sorted (they are in provided datasets), use searchsorted.
    src_syms_sorted = src_syms
    pos = np.searchsorted(src_syms_sorted, dest_syms)
    in_range = (pos >= 0) & (pos < len(src_syms_sorted))
    matched = np.full(len(dest_syms), -1, dtype=np.int64)
    ok = in_range & (src_syms_sorted[pos.clip(0, len(src_syms_sorted) - 1)] == dest_syms)
    matched[ok] = pos[ok]

    if np.any(~ok):
        src_map = {k: i for i, k in enumerate(src_syms.tolist())}
        for i in np.flatnonzero(~ok):
            matched[i] = src_map.get(dest_syms[i], -1)

    return dest_syms, matched


def _resolve_key_indices(
    *,
    source_keys: np.ndarray,
    dest_to_source_keys: Sequence[tuple[str, str]],
) -> tuple[list[str], np.ndarray]:
    """Return (dest_keys, src_col_indices_or_-1) aligned to dest order."""

    src_keys = _decode_str_array(source_keys)
    src_key_to_idx = {k: i for i, k in enumerate(src_keys.tolist())}

    dest_keys: list[str] = []
    src_indices = np.empty(len(dest_to_source_keys), dtype=np.int64)

    for j, (dest_key, source_key) in enumerate(dest_to_source_keys):
        dest_keys.append(dest_key)
        src_indices[j] = src_key_to_idx.get(source_key, -1)

    return dest_keys, src_indices


def copy_h5_selected(
    *,
    src_params: H5Params,
    dst_params: H5Params,
    dest_to_source_keys: Mapping[str, str],
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: Sequence[str] | None = None,
    compression: str | None = "gzip",
    compression_opts: int = 4,
    overwrite: bool = True,
    n_workers: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> int:
    """Copy H5 files from ``src_params`` to ``dst_params`` applying key/symbol selection.

    This is the fastest path for ``data_processing``: it avoids building any
    intermediate (pandas/polars) DataFrames and operates directly on H5 matrices.

    Args:
        src_params: Source directory params.
        dst_params: Destination directory params.
        dest_to_source_keys: Mapping of output key name -> source key name.
        start_date: Optional inclusive date filter (YYYY-MM-DD).
        end_date: Optional inclusive date filter (YYYY-MM-DD).
        symbols: Optional symbol filter. Missing symbols will be included with NaNs.
        compression: HDF5 compression ("gzip", "lzf", or None).
        compression_opts: gzip level when ``compression="gzip"``.
        overwrite: Overwrite existing destination files.
        n_workers: Parallel workers (threads). Use 1 for deterministic serial IO.
        progress_callback: Optional callback(current, total).

    Returns:
        Number of files written.
    """

    src_files = list_h5_files(src_params, start_date=start_date, end_date=end_date)
    if not src_files:
        return 0

    dst_dir = build_h5_dir(dst_params)
    dst_dir.mkdir(parents=True, exist_ok=True)

    key_pairs = list(dest_to_source_keys.items())

    comp_opts = compression_opts if compression == "gzip" else None

    with h5py.File(src_files[0], "r") as f0:
        base_keys = f0["KEY"][:]
        dest_keys, base_src_col_idx = _resolve_key_indices(
            source_keys=base_keys,
            dest_to_source_keys=key_pairs,
        )
        dest_keys_bytes = np.asarray(dest_keys, dtype="S")

    base_col_pos = np.flatnonzero(base_src_col_idx >= 0)

    dest_symbols_bytes: np.ndarray | None = None
    if symbols:
        unique_symbols: list[str] = []
        seen: set[str] = set()
        for sym in symbols:
            s = str(sym)
            if s in seen:
                continue
            seen.add(s)
            unique_symbols.append(s)
        dest_symbols_bytes = np.asarray(unique_symbols, dtype="S")

    def _process_one(src_file: Path) -> bool:
        stem = src_file.stem
        dst_file = dst_dir / f"{stem}.h5"
        if not overwrite and dst_file.exists():
            return False

        with h5py.File(src_file, "r") as fin:
            src_date = _decode_str_scalar(fin["DATETIME"][()])

            data_ds = fin["DATA"]

            # Keys are typically stable across dates; only re-resolve if needed.
            src_col_idx = base_src_col_idx
            col_pos = base_col_pos
            if key_pairs:
                src_keys = fin["KEY"][:]
                if not np.array_equal(src_keys, base_keys):
                    _, src_col_idx = _resolve_key_indices(
                        source_keys=src_keys,
                        dest_to_source_keys=key_pairs,
                    )
                    col_pos = np.flatnonzero(src_col_idx >= 0)

            n_cols = len(dest_keys_bytes)

            has_missing_cols = col_pos.size != n_cols

            if not symbols:
                symbols_out = fin["SYMBOL"][:]
                n_rows = len(symbols_out)

                src_cols = src_col_idx.tolist()
                if not has_missing_cols:
                    if len(src_cols) == 1:
                        out = np.asarray(data_ds[:, src_cols[0]], dtype=np.float64)[:, None]
                    elif len(src_cols) == data_ds.shape[1] and src_cols == list(
                        range(data_ds.shape[1])
                    ):
                        out = np.asarray(data_ds[:, :], dtype=np.float64)
                    else:
                        out = np.asarray(data_ds[:, src_cols], dtype=np.float64)
                else:
                    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
                    if col_pos.size:
                        src_cols_present = src_col_idx[col_pos].tolist()
                        if len(src_cols_present) == 1:
                            data = np.asarray(
                                data_ds[:, src_cols_present[0]], dtype=np.float64
                            )[:, None]
                        else:
                            data = np.asarray(
                                data_ds[:, src_cols_present], dtype=np.float64
                            )
                        out[:, col_pos] = data
            else:
                assert dest_symbols_bytes is not None
                symbols_out = dest_symbols_bytes
                n_rows = len(symbols_out)

                src_symbols = fin["SYMBOL"][:]
                src_row_idx = _match_sorted_bytes(src_symbols, symbols_out)
                row_pos = np.flatnonzero(src_row_idx >= 0)
                has_missing_rows = row_pos.size != n_rows

                if not has_missing_rows and not has_missing_cols:
                    src_cols = src_col_idx.tolist()
                    src_rows = src_row_idx.tolist()
                    if len(src_cols) == 1:
                        out = np.asarray(
                            data_ds[src_rows, src_cols[0]], dtype=np.float64
                        )[:, None]
                    else:
                        out = np.asarray(
                            data_ds[src_rows, :][:, src_cols], dtype=np.float64
                        )
                else:
                    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
                    if col_pos.size and row_pos.size:
                        src_cols_present = src_col_idx[col_pos].tolist()
                        src_rows_present = src_row_idx[row_pos].tolist()
                        if len(src_cols_present) == 1:
                            data = np.asarray(
                                data_ds[src_rows_present, src_cols_present[0]],
                                dtype=np.float64,
                            )[:, None]
                        else:
                            data = np.asarray(
                                data_ds[src_rows_present, :][:, src_cols_present],
                                dtype=np.float64,
                            )
                        out[np.ix_(row_pos, col_pos)] = data

        with h5py.File(dst_file, "w") as fout:
            fout.create_dataset(
                "DATA",
                data=out,
                compression=compression,
                compression_opts=comp_opts,
            )
            fout.create_dataset("DATETIME", data=src_date)
            fout.create_dataset("KEY", data=dest_keys_bytes)
            fout.create_dataset("SYMBOL", data=symbols_out)

        return True

    written = 0
    total = len(src_files)

    if n_workers <= 1 or total < 32:
        for i, f in enumerate(src_files, start=1):
            if _process_one(f):
                written += 1
            if progress_callback and (i % 100 == 0 or i == total):
                progress_callback(i, total)
        return written

    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one, f): f for f in src_files}
        for fut in as_completed(futures):
            if fut.result():
                written += 1
            completed += 1
            if progress_callback and (completed % 100 == 0 or completed == total):
                progress_callback(completed, total)

    return written


def h5_load(
    params: H5Params,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: Sequence[str] | None = None,
    keys: Sequence[str] | None = None,
    n_workers: int = 1,
) -> pl.DataFrame:
    """Load H5 files into a Polars long-form DataFrame.

    The output schema is:
      - DATETIME: Utf8 (YYYY-MM-DD)
      - SYMBOL:   Utf8
      - <keys...>: Float64
    """

    files = list_h5_files(params, start_date=start_date, end_date=end_date)
    if not files:
        return pl.DataFrame()

    # Find the file with maximum symbols to use as canonical symbol list.
    # This handles cases where symbol count varies across files.
    max_symbols_file = files[0]
    max_symbols_count = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            n_sym = f["SYMBOL"].shape[0]
            if n_sym > max_symbols_count:
                max_symbols_count = n_sym
                max_symbols_file = fp

    # Resolve keys and symbol universe from the file with most symbols.
    with h5py.File(max_symbols_file, "r") as f0:
        src_keys = _decode_str_array(f0["KEY"][:]).tolist()
        src_symbols = f0["SYMBOL"][:]

    if keys:
        key_pairs = [(k, k) for k in keys]
    else:
        key_pairs = [(k, k) for k in src_keys]

    dest_keys, src_col_idx = _resolve_key_indices(
        source_keys=np.asarray(src_keys, dtype=str),
        dest_to_source_keys=key_pairs,
    )

    dest_symbols, src_row_idx_first = _resolve_symbol_indices(
        source_symbols=src_symbols,
        requested_symbols=symbols,
    )

    n_rows_per_day = len(dest_symbols)
    n_cols = len(dest_keys)
    total_rows = n_rows_per_day * len(files)

    all_dates = np.empty(total_rows, dtype="<U10")
    all_syms = np.tile(dest_symbols, len(files))
    all_data = np.full((total_rows, n_cols), np.nan, dtype=np.float64)

    def _load_one(idx: int, file_path: Path) -> tuple[int, str, np.ndarray]:
        with h5py.File(file_path, "r") as fin:
            date = _decode_str_scalar(fin["DATETIME"][()])
            data_ds = fin["DATA"]
            file_n_symbols = data_ds.shape[0]

            # Check if this file has different symbol count than first file
            # If so, we need per-file symbol resolution
            need_per_file_resolution = (
                symbols or file_n_symbols != n_rows_per_day
            )

            if need_per_file_resolution:
                src_symbols_local = fin["SYMBOL"][:]
                _, src_row_idx = _resolve_symbol_indices(
                    source_symbols=src_symbols_local,
                    requested_symbols=dest_symbols.tolist(),
                )
            else:
                # Fast path: no explicit symbol filter and stable symbol count
                src_row_idx = src_row_idx_first

            out = np.full((n_rows_per_day, n_cols), np.nan, dtype=np.float64)

            col_pos = np.flatnonzero(src_col_idx >= 0)
            row_pos = np.flatnonzero(src_row_idx >= 0)
            if col_pos.size and row_pos.size:
                src_cols = src_col_idx[col_pos].tolist()
                src_rows = src_row_idx[row_pos].tolist()
                if not need_per_file_resolution and col_pos.size == len(src_col_idx):
                    if len(src_cols) == 1:
                        data = np.asarray(data_ds[:, src_cols[0]], dtype=np.float64)[:, None]
                    elif len(src_cols) == data_ds.shape[1] and src_cols == list(
                        range(data_ds.shape[1])
                    ):
                        data = np.asarray(data_ds[:, :], dtype=np.float64)
                    else:
                        data = np.asarray(data_ds[:, src_cols], dtype=np.float64)
                    out[:, :] = data
                else:
                    if len(src_cols) == 1:
                        data = np.asarray(
                            data_ds[src_rows, src_cols[0]], dtype=np.float64
                        )[:, None]
                    else:
                        data = np.asarray(
                            data_ds[src_rows, :][:, src_cols], dtype=np.float64
                        )
                    out[np.ix_(row_pos, col_pos)] = data

        return idx, date, out

    if n_workers <= 1 or len(files) < 32:
        for i, fp in enumerate(files):
            _, date, mat = _load_one(i, fp)
            start = i * n_rows_per_day
            end = start + n_rows_per_day
            all_dates[start:end] = date
            all_data[start:end, :] = mat
    else:
        results: list[tuple[int, str, np.ndarray]] = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futs = [executor.submit(_load_one, i, fp) for i, fp in enumerate(files)]
            for fut in as_completed(futs):
                results.append(fut.result())
        results.sort(key=lambda x: x[0])
        for i, date, mat in results:
            start = i * n_rows_per_day
            end = start + n_rows_per_day
            all_dates[start:end] = date
            all_data[start:end, :] = mat

    df = pl.DataFrame(all_data, schema=dest_keys)
    df.insert_column(0, pl.Series("SYMBOL", all_syms, dtype=pl.Utf8))
    df.insert_column(0, pl.Series("DATETIME", all_dates, dtype=pl.Utf8))
    return df


def h5_save(
    df: pl.DataFrame,
    params: H5Params,
    *,
    compression: str | None = "gzip",
    compression_opts: int = 4,
    assume_sorted: bool = False,
    overwrite: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> int:
    """Save a Polars long-form frame into per-date H5 files."""

    if df.is_empty():
        return 0

    if "DATETIME" not in df.columns or "SYMBOL" not in df.columns:
        raise ValueError("Polars DataFrame must have 'DATETIME' and 'SYMBOL' columns.")

    field_cols = [c for c in df.columns if c not in {"DATETIME", "SYMBOL"}]
    if not field_cols:
        raise ValueError("No field columns found to save.")

    save_dir = build_h5_dir(params)
    save_dir.mkdir(parents=True, exist_ok=True)

    comp_opts = compression_opts if compression == "gzip" else None

    df_sorted = df if assume_sorted else df.sort(["DATETIME", "SYMBOL"])
    dates = df_sorted.get_column("DATETIME").to_numpy()
    syms = df_sorted.get_column("SYMBOL").to_numpy()
    data = df_sorted.select(field_cols).to_numpy()

    # Find group boundaries by date.
    change_idx = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(dates)]))

    total = len(starts)
    for i, (s, e) in enumerate(zip(starts, ends), start=1):
        date = str(dates[s])
        out_path = save_dir / f"{date}.h5"
        if out_path.exists() and not overwrite:
            continue

        symbols = syms[s:e]
        matrix = data[s:e, :]

        with h5py.File(out_path, "w") as f:
            f.create_dataset(
                "DATA",
                data=matrix,
                compression=compression,
                compression_opts=comp_opts,
            )
            f.create_dataset("DATETIME", data=date)
            f.create_dataset("KEY", data=np.asarray(field_cols, dtype="S"))
            f.create_dataset("SYMBOL", data=np.asarray(symbols, dtype="S"))

        if progress_callback and (i % 100 == 0 or i == total):
            progress_callback(i, total)

    return total
