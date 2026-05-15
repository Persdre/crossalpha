"""Utility functions for ar-scraper."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ar_scraper.config import detect_region_from_symbol


def get_existing_symbols(output_dir: Path) -> set[str]:
    """Get symbols that already have downloaded PDFs.

    Args:
        output_dir: Directory containing PDF files

    Returns:
        Set of symbol strings extracted from filenames
    """
    if not output_dir.exists():
        return set()

    existing = set()
    for report_file in list(output_dir.glob("*.pdf")) + list(output_dir.glob("*.zip")):
        # Filename format: {SYMBOL}_{TYPE}_{DATE}.pdf/.zip
        # Handle symbols with dots like "7974.T"
        stem = report_file.stem
        parts = stem.rsplit("_", 2)  # Split from right to handle dots in symbol
        if len(parts) >= 3:
            existing.add(parts[0])
    return existing


def save_json(data: dict, path: Path) -> None:
    """Save dict to JSON, merging with existing file.

    Args:
        data: Dictionary to save
        path: Output path
    """
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            pass

    existing.update(data)
    path.write_text(
        json.dumps(existing, indent=2, sort_keys=True, ensure_ascii=False)
    )


def load_symbols_from_csv(csv_path: Path, region_filter: str) -> list[str]:
    """Load symbols from SymbolDict.csv filtered by region.

    Args:
        csv_path: Path to SymbolDict.csv
        region_filter: Region value to filter by (e.g., "us_russell_1000")

    Returns:
        Sorted list of symbol strings
    """
    import polars as pl

    df = pl.read_csv(csv_path)
    symbols = (
        df.filter(pl.col("region") == region_filter)
        .select("SYMBOL")
        .to_series()
        .to_list()
    )
    return sorted(symbols)


def group_symbols_by_region(symbols: list[str]) -> dict[str, list[str]]:
    """Group symbols by their detected region.

    Args:
        symbols: List of stock symbols

    Returns:
        Dict mapping region code to list of symbols
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for symbol in symbols:
        region = detect_region_from_symbol(symbol)
        groups[region].append(symbol)
    return dict(groups)


def get_data_path(filename: str) -> Path:
    """Get path to bundled data file.

    Args:
        filename: Name of file in data directory

    Returns:
        Path to the data file
    """
    return Path(__file__).parent / "data" / filename
