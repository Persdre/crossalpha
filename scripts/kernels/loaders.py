"""Data loading utilities for momentum factor construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class EmbeddingSet(NamedTuple):
    """Container for embeddings from multiple categories."""

    symbols: list[str]
    embeddings: dict[str, np.ndarray]  # {category: (n_symbols, dim)}


def load_embeddings(
    region: str,
    categories: list[str],
    embedding_type: str = "128_dimensions",
) -> EmbeddingSet:
    """Load embeddings for a region across all categories.

    Args:
        region: 'jp_topix', 'jp_topix_500', or 'us_russell_1000'
        categories: List of category names
        embedding_type: 'raw' or '128_dimensions'

    Returns:
        EmbeddingSet with symbols and embeddings dict
    """
    embeddings_dir = PROJECT_ROOT / "embeddings" / embedding_type

    all_symbols: set[str] = set()
    category_data: dict[str, tuple[list[str], np.ndarray]] = {}

    for category in categories:
        h5_path = embeddings_dir / category / f"{region}_{category}.h5"
        if not h5_path.exists():
            continue

        with h5py.File(h5_path, "r") as f:
            emb = f["embeddings"][:].astype(np.float64)
            syms = json.loads(f["embeddings"].attrs["symbols"])

        category_data[category] = (syms, emb)
        all_symbols.update(syms)

    # Create unified symbol list (sorted for consistency)
    symbols = sorted(all_symbols)
    sym_to_idx = {s: i for i, s in enumerate(symbols)}

    # Build aligned embedding matrices
    embeddings: dict[str, np.ndarray] = {}
    for category, (cat_syms, cat_emb) in category_data.items():
        dim = cat_emb.shape[1]
        aligned = np.full((len(symbols), dim), np.nan, dtype=np.float64)
        for i, sym in enumerate(cat_syms):
            if sym in sym_to_idx:
                aligned[sym_to_idx[sym]] = cat_emb[i]
        embeddings[category] = aligned

    return EmbeddingSet(symbols=symbols, embeddings=embeddings)


def load_symbol_sectors(region: str = "us_russell_1000") -> pl.DataFrame:
    """Load symbol to sector mapping from SymbolDict.csv.

    Args:
        region: Region to filter

    Returns:
        DataFrame with columns [SYMBOL, sector]
    """
    symbol_dict_path = PROJECT_ROOT / "docs" / "SymbolDict.csv"
    df = pl.read_csv(symbol_dict_path)

    filtered = df.filter(pl.col("region") == region)
    # Prefer gics_sector if available, fall back to sector
    if "gics_sector" in filtered.columns:
        result = filtered.select([
            pl.col("SYMBOL"),
            pl.when(pl.col("gics_sector").is_not_null() & (pl.col("gics_sector") != ""))
            .then(pl.col("gics_sector"))
            .otherwise(pl.col("sector"))
            .alias("sector"),
        ])
    else:
        result = filtered.select([pl.col("SYMBOL"), pl.col("sector")])
    return result


def load_monthly_close_prices(
    region: str,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Load monthly close prices for a region.

    Args:
        region: 'jp_topix', 'jp_topix_500', or 'us_russell_1000'
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with columns [DATETIME, SYMBOL, close]
    """
    from scripts.h5_data.h5_utils import H5Params, h5_load

    params: H5Params = {
        "source_path": str(PROJECT_ROOT / "market_data"),
        "product": "Equity",
        "region": region,
        "freq": "1mo",
        "source": "yfinance",
    }

    df = h5_load(params, start_date=start_date, end_date=end_date, keys=["close"])

    return df.select([
        pl.col("DATETIME"),
        pl.col("SYMBOL"),
        pl.col("close"),
    ])


def load_daily_close_prices(
    region: str,
    start_date: str,
    end_date: str,
    source_path: str | None = None,
) -> pl.DataFrame:
    """Load daily close prices for a region.

    Args:
        region: Region identifier (e.g. 'jp_topix_500', 'us_russell_1000')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        source_path: Override market_data root path (default: PROJECT_ROOT/market_data)

    Returns:
        DataFrame with columns [DATETIME, SYMBOL, close]
    """
    from scripts.h5_data.h5_utils import H5Params, h5_load

    params: H5Params = {
        "source_path": source_path or str(PROJECT_ROOT / "market_data"),
        "product": "Equity",
        "region": region,
        "freq": "1d",
        "source": "yfinance",
    }

    df = h5_load(params, start_date=start_date, end_date=end_date, keys=["close"])

    return df.select([
        pl.col("DATETIME"),
        pl.col("SYMBOL"),
        pl.col("close"),
    ])
