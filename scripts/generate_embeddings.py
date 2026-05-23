#!/usr/bin/env python3
"""Generate text embeddings for parsed business reports.

Creates 3072-dimensional embeddings using OpenAI's text-embedding-3-large
for each of the 10 business categories extracted from US 10-K and JP
Securities reports.

Usage:
    python scripts/generate_embeddings.py                    # Process all
    python scripts/generate_embeddings.py --category main_business_segments
    python scripts/generate_embeddings.py --test             # Test with 5 stocks
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import TypedDict

import h5py
import numpy as np
import polars as pl
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSED_10K_DIR = PROJECT_ROOT / "parsed_reports" / "10_k_filings"
PARSED_JP_DIR = PROJECT_ROOT / "parsed_reports" / "securities_reports"
PARSED_TW_DIR = PROJECT_ROOT / "parsed_reports" / "securities_reports_tw"
PARSED_KR_DIR = PROJECT_ROOT / "parsed_reports" / "securities_reports_kr"
PARSED_HK_DIR = PROJECT_ROOT / "parsed_reports" / "securities_reports_hk"
PARSED_VN_DIR = PROJECT_ROOT / "parsed_reports" / "securities_reports_vn"
SYMBOL_DICT_PATH = PROJECT_ROOT / "docs" / "SymbolDict.csv"
OUTPUT_DIR = PROJECT_ROOT / "embeddings" / "raw"

DEFAULT_WORKERS = 100
DEFAULT_BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

# 10 Business Categories
CATEGORIES = [
    "main_business_segments",
    "core_technologies_production_methods",
    "primary_customers_markets",
    "geographic_coverage",
    "supply_chain_position",
    "strategic_focus_rd_direction",
    "revenue_model",
    "key_competitors_industry_positioning",
    "financial_scale_growth_profile",
    "value_proposition_product_differentiation",
]

# Index regions to generate filtered files for
INDEX_REGIONS = ["jp_topix", "jp_topix_500", "us_russell_1000", "tw_twse", "kr_kospi", "hk_hsci", "hk_main", "vn_hose"]


class ParsedReport(TypedDict):
    """Structure of a parsed report JSON."""

    symbol: str
    categories: dict[str, str]


def load_parsed_reports() -> pl.DataFrame:
    """Load all parsed reports from both US and JP directories.

    Returns:
        DataFrame with columns: symbol, source, filing_date, + 10 category columns
    """
    records: list[dict] = []

    # Load US 10-K reports
    for json_path in PARSED_10K_DIR.glob("*.json"):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            record = {
                "symbol": data.get("symbol", ""),
                "source": "us_10k",
                "filing_date": data.get("filing_date", ""),
            }
            categories = data.get("categories", {})
            for cat in CATEGORIES:
                record[cat] = categories.get(cat, "")
            records.append(record)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Failed to load {json_path.name}: {e}")

    # Load JP Securities reports
    for json_path in PARSED_JP_DIR.glob("*.json"):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            record = {
                "symbol": data.get("symbol", ""),
                "source": "jp_securities",
                "filing_date": data.get("filing_date", ""),
            }
            categories = data.get("categories", {})
            for cat in CATEGORIES:
                record[cat] = categories.get(cat, "")
            records.append(record)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Failed to load {json_path.name}: {e}")

    # Load TW Annual reports
    for json_path in PARSED_TW_DIR.glob("*.json"):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            record = {
                "symbol": data.get("symbol", ""),
                "source": "tw_annual",
                "filing_date": data.get("filing_date", ""),
            }
            categories = data.get("categories", {})
            for cat in CATEGORIES:
                record[cat] = categories.get(cat, "")
            records.append(record)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Failed to load {json_path.name}: {e}")

    # Load KR Annual reports
    for json_path in PARSED_KR_DIR.glob("*.json"):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            record = {
                "symbol": data.get("symbol", ""),
                "source": "kr_annual",
                "filing_date": data.get("filing_date", ""),
            }
            categories = data.get("categories", {})
            for cat in CATEGORIES:
                record[cat] = categories.get(cat, "")
            records.append(record)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Failed to load {json_path.name}: {e}")

    # Load HK Annual reports
    for json_path in PARSED_HK_DIR.glob("*.json"):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            record = {
                "symbol": data.get("symbol", ""),
                "source": "hk_annual",
                "filing_date": data.get("filing_date", ""),
            }
            categories = data.get("categories", {})
            for cat in CATEGORIES:
                record[cat] = categories.get(cat, "")
            records.append(record)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Failed to load {json_path.name}: {e}")

    # Load VN Annual reports
    for json_path in PARSED_VN_DIR.glob("*.json"):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            record = {
                "symbol": data.get("symbol", ""),
                "source": "vn_annual",
                "filing_date": data.get("filing_date", ""),
            }
            categories = data.get("categories", {})
            for cat in CATEGORIES:
                record[cat] = categories.get(cat, "")
            records.append(record)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Failed to load {json_path.name}: {e}")

    if not records:
        raise ValueError("No parsed reports found")

    df = pl.DataFrame(records)

    # Deduplicate by symbol, keeping most recent filing_date
    df = (
        df.sort("filing_date", descending=True)
        .group_by("symbol")
        .first()
        .sort("symbol")
    )

    print(f"Loaded {len(df)} unique symbols from parsed reports")
    print(f"  US 10-K: {df.filter(pl.col('source') == 'us_10k').height}")
    print(f"  JP Securities: {df.filter(pl.col('source') == 'jp_securities').height}")
    print(f"  TW Annual: {df.filter(pl.col('source') == 'tw_annual').height}")
    print(f"  KR Annual: {df.filter(pl.col('source') == 'kr_annual').height}")
    print(f"  HK Annual: {df.filter(pl.col('source') == 'hk_annual').height}")
    print(f"  VN Annual: {df.filter(pl.col('source') == 'vn_annual').height}")

    return df


def load_symbol_dict() -> pl.DataFrame:
    """Load SymbolDict.csv for index membership filtering.

    Returns:
        DataFrame with columns: symbol, region
    """
    df = pl.read_csv(SYMBOL_DICT_PATH)
    df = df.select(["SYMBOL", "region"]).rename({"SYMBOL": "symbol"})
    print(f"Loaded {len(df)} symbols from SymbolDict.csv")
    return df


async def generate_embeddings_batch(
    client: AsyncOpenAI,
    texts: list[str],
    semaphore: asyncio.Semaphore,
) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    async with semaphore:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]


async def embed_category_texts(
    texts: list[str],
    symbols: list[str],
    max_workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[str], np.ndarray]:
    """Embed all texts for a single category.

    Returns:
        Tuple of (valid_symbols, embeddings_array)
    """
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_workers)

    # Filter out empty texts
    valid_pairs = [
        (sym, txt) for sym, txt in zip(symbols, texts)
        if txt and txt.strip()
    ]

    if not valid_pairs:
        return [], np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIM)

    valid_symbols, valid_texts = zip(*valid_pairs)
    valid_symbols = list(valid_symbols)
    valid_texts = list(valid_texts)

    print(f"  Embedding {len(valid_texts)} texts ({len(texts) - len(valid_texts)} empty skipped)")

    # Process in batches
    all_embeddings: list[list[float]] = []
    n_batches = (len(valid_texts) + batch_size - 1) // batch_size

    tasks = []
    for i in range(n_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(valid_texts))
        batch_texts = valid_texts[batch_start:batch_end]
        tasks.append(generate_embeddings_batch(client, batch_texts, semaphore))

    print(f"  Processing {n_batches} batches with {max_workers} workers...")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  Error in batch {i}: {result}")
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(valid_texts))
            batch_len = batch_end - batch_start
            all_embeddings.extend([[0.0] * EMBEDDING_DIM] * batch_len)
        else:
            all_embeddings.extend(result)

    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    print(f"  Generated embeddings shape: {embeddings_array.shape}")

    return valid_symbols, embeddings_array


def save_embeddings_h5(
    symbols: list[str],
    embeddings: np.ndarray,
    category: str,
    output_path: Path,
) -> None:
    """Save embeddings to H5 file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        dset = f.create_dataset(
            "embeddings",
            data=embeddings,
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        dset.attrs["symbols"] = json.dumps(symbols)
        dset.attrs["category"] = category
        dset.attrs["model"] = EMBEDDING_MODEL
        dset.attrs["dimensions"] = EMBEDDING_DIM

    print(f"  Saved {len(symbols)} embeddings to {output_path}")


def check_h5_exists(output_path: Path, expected_count: int) -> bool:
    """Check if H5 file exists and has expected number of embeddings."""
    if not output_path.exists():
        return False

    try:
        with h5py.File(output_path, "r") as f:
            if "embeddings" not in f:
                return False
            actual_count = f["embeddings"].shape[0]
            return actual_count == expected_count
    except Exception:
        return False


async def process_category(
    category: str,
    reports_df: pl.DataFrame,
    symbol_dict_df: pl.DataFrame,
    max_workers: int,
    batch_size: int,
    force: bool = False,
) -> None:
    """Process a single category: embed texts and save H5 files."""
    print(f"\n{'='*60}")
    print(f"Processing category: {category}")
    print(f"{'='*60}")

    category_dir = OUTPUT_DIR / category
    symbols = reports_df["symbol"].to_list()
    texts = reports_df[category].to_list()

    # Check if "all" file already exists
    all_output_path = category_dir / f"all_{category}.h5"
    valid_count = sum(1 for t in texts if t and t.strip())

    if not force and check_h5_exists(all_output_path, valid_count):
        print(f"  Skipping: {all_output_path.name} already exists with {valid_count} embeddings")
        return

    # Generate embeddings
    valid_symbols, embeddings = await embed_category_texts(
        texts, symbols, max_workers, batch_size
    )

    if len(valid_symbols) == 0:
        print(f"  Warning: No valid texts for category {category}")
        return

    # Save "all" file
    save_embeddings_h5(valid_symbols, embeddings, category, all_output_path)

    # Create symbol-to-embedding mapping for filtering
    symbol_to_idx = {sym: i for i, sym in enumerate(valid_symbols)}

    # Save index-specific files
    for region in INDEX_REGIONS:
        region_symbols = (
            symbol_dict_df.filter(pl.col("region") == region)["symbol"].to_list()
        )
        filtered_symbols = [s for s in region_symbols if s in symbol_to_idx]

        if not filtered_symbols:
            print(f"  Warning: No symbols found for region {region}")
            continue

        indices = [symbol_to_idx[s] for s in filtered_symbols]
        filtered_embeddings = embeddings[indices]

        region_output_path = category_dir / f"{region}_{category}.h5"
        save_embeddings_h5(filtered_symbols, filtered_embeddings, category, region_output_path)


async def process_all_categories(
    categories: list[str],
    max_workers: int,
    batch_size: int,
    force: bool = False,
    test_mode: bool = False,
) -> None:
    """Process all categories sequentially."""
    print("Loading data...")
    reports_df = load_parsed_reports()
    symbol_dict_df = load_symbol_dict()

    if test_mode:
        print("\n*** TEST MODE: Using only 5 stocks ***")
        reports_df = reports_df.head(5)

    for category in categories:
        await process_category(
            category, reports_df, symbol_dict_df, max_workers, batch_size, force
        )

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for parsed business reports"
    )
    parser.add_argument(
        "--category", type=str, default=None, choices=CATEGORIES,
        help="Process a single category (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Max concurrent API calls (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Texts per API batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    parser.add_argument("--test", action="store_true", help="Test mode: 5 stocks only")
    args = parser.parse_args()

    categories = [args.category] if args.category else CATEGORIES

    print(f"Embedding Generation Script")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Dimensions: {EMBEDDING_DIM}")
    print(f"  Categories: {len(categories)}")
    print(f"  Workers: {args.workers}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output: {OUTPUT_DIR}")

    asyncio.run(
        process_all_categories(
            categories=categories,
            max_workers=args.workers,
            batch_size=args.batch_size,
            force=args.force,
            test_mode=args.test,
        )
    )


if __name__ == "__main__":
    main()
