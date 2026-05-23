#!/usr/bin/env python3
"""Whiten embedding vectors via PCA dimensionality reduction.

Reduces 3072-dimensional raw embeddings to 128 whitened dimensions
using SVD-based whitening per the Morgan Stanley methodology.

The whitening transformation:
    1. Mean-center: X_c = X - mean(X)
    2. SVD: X_c = U @ S @ V.T
    3. Whitened: Z = sqrt(n) * U[:, :w]  where w=128

Result: Cov(Z) = I (identity matrix)

Usage:
    python scripts/whiten_embeddings.py                    # Process all
    python scripts/whiten_embeddings.py --category main_business_segments
    python scripts/whiten_embeddings.py --verify           # Run verification
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import polars as pl

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings" / "raw"
OUTPUT_BASE_DIR = PROJECT_ROOT / "embeddings"
SYMBOL_DICT_PATH = PROJECT_ROOT / "docs" / "SymbolDict.csv"

RAW_DIM = 3072
WHITENED_DIMS = [64, 100, 128, 150, 200, 256, 300, 350, 400, 450, 512, 768, 1024]
MAX_DIM = max(WHITENED_DIMS)  # 1024
DEFAULT_DIMS = WHITENED_DIMS  # All dimensions by default

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

INDEX_REGIONS = ["jp_topix", "jp_topix_500", "us_russell_1000", "tw_twse", "kr_kospi", "hk_hsci", "hk_main", "vn_hose", "cn_csi300", "cn_star", "cn_chinext"]


class EmbeddingData(NamedTuple):
    """Container for loaded embedding data."""

    symbols: list[str]
    embeddings: np.ndarray  # shape: (n_samples, n_dims)
    category: str


def load_raw_embeddings(category: str, index_prefix: str = "all") -> EmbeddingData:
    """Load raw embeddings from H5 file.

    Args:
        category: Category name
        index_prefix: 'all', 'jp_topix', 'jp_topix_500', or 'us_russell_1000'

    Returns:
        EmbeddingData with symbols and embeddings array
    """
    h5_path = RAW_EMBEDDINGS_DIR / category / f"{index_prefix}_{category}.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"Raw embeddings not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        embeddings = f["embeddings"][:].astype(np.float64)
        symbols = json.loads(f["embeddings"].attrs["symbols"])

    return EmbeddingData(symbols=symbols, embeddings=embeddings, category=category)


def load_symbol_dict() -> pl.DataFrame:
    """Load SymbolDict.csv for index membership filtering."""
    return pl.read_csv(SYMBOL_DICT_PATH).select(
        pl.col("SYMBOL").alias("symbol"),
        pl.col("region"),
    )


class WhiteningResult(NamedTuple):
    """Container for whitening transformation results."""

    whitened: np.ndarray       # shape: (n_samples, n_components)
    symbols: list[str]
    mean_vector: np.ndarray    # shape: (RAW_DIM,)
    singular_values: np.ndarray  # shape: (n_components,)
    explained_variance_ratio: np.ndarray  # shape: (n_components,)
    n_samples: int
    n_components: int          # Number of dimensions


def whiten_embeddings_multi(
    embeddings: np.ndarray,
    symbols: list[str],
    dimensions: list[int],
) -> dict[int, WhiteningResult]:
    """Apply PCA whitening for multiple target dimensions in one SVD pass.

    Implements: Z = sqrt(n) * U_w where X = U @ S @ V.T

    Args:
        embeddings: Raw embeddings, shape (n_samples, n_dims)
        symbols: List of symbols corresponding to rows
        dimensions: List of target dimensions to generate

    Returns:
        Dict mapping dimension -> WhiteningResult
    """
    n_samples, n_dims = embeddings.shape
    max_dim = max(dimensions)

    # Step 1: Mean-center
    mean_vector = embeddings.mean(axis=0)
    X_centered = embeddings - mean_vector

    # Step 2: SVD decomposition (computed ONCE)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Compute total variance for explained variance ratio
    total_var = np.sum(S ** 2)

    # Step 3: Generate whitened embeddings for each dimension
    results = {}
    for dim in dimensions:
        U_w = U[:, :dim]
        S_w = S[:dim]

        # Whitened vectors: Z = sqrt(n) * U_w
        Z = np.sqrt(n_samples) * U_w

        # Explained variance ratio for this dimension
        explained_var = S_w ** 2
        explained_variance_ratio = explained_var / total_var

        results[dim] = WhiteningResult(
            whitened=Z.astype(np.float32),
            symbols=symbols,
            mean_vector=mean_vector.astype(np.float32),
            singular_values=S_w.astype(np.float32),
            explained_variance_ratio=explained_variance_ratio.astype(np.float32),
            n_samples=n_samples,
            n_components=dim,
        )

    return results


def verify_whitening(whitened: np.ndarray, rtol: float = 1e-4) -> bool:
    """Verify whitened vectors have identity covariance.

    Checks: Cov(Z) = (1/n) * Z.T @ Z ≈ I

    Args:
        whitened: Whitened embedding matrix, shape (n_samples, n_dims)
        rtol: Relative tolerance for identity check

    Returns:
        True if verification passes
    """
    n_samples = whitened.shape[0]
    cov = (whitened.T @ whitened) / n_samples

    # Check diagonal elements ≈ 1
    diag = np.diag(cov)
    diag_ok = np.allclose(diag, 1.0, rtol=rtol)

    # Check off-diagonal elements ≈ 0
    off_diag = cov - np.diag(diag)
    off_diag_ok = np.allclose(off_diag, 0.0, atol=rtol)

    if not diag_ok:
        print(f"  WARNING: Diagonal not ≈ 1. Range: [{diag.min():.6f}, {diag.max():.6f}]")
    if not off_diag_ok:
        max_off = np.abs(off_diag).max()
        print(f"  WARNING: Off-diagonal not ≈ 0. Max: {max_off:.6f}")

    return diag_ok and off_diag_ok


def save_whitened_h5(
    symbols: list[str],
    whitened: np.ndarray,
    category: str,
    output_path: Path,
    explained_variance_ratio: np.ndarray,
    n_components: int,
) -> None:
    """Save whitened embeddings to H5 file.

    Args:
        symbols: List of stock symbols
        whitened: Whitened embeddings, shape (n_samples, n_components)
        category: Category name
        output_path: Output file path
        explained_variance_ratio: Variance explained by each component
        n_components: Number of dimensions
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        dset = f.create_dataset(
            "embeddings",
            data=whitened,
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        dset.attrs["symbols"] = json.dumps(symbols)
        dset.attrs["category"] = category
        dset.attrs["dimensions"] = n_components
        dset.attrs["method"] = "svd_whitening"
        f.create_dataset(
            "explained_variance_ratio",
            data=explained_variance_ratio,
            dtype=np.float32,
        )

    print(f"  Saved {len(symbols)} embeddings to {output_path.name}")


def get_output_dir(dim: int) -> Path:
    """Get output directory for a specific dimension."""
    return OUTPUT_BASE_DIR / f"{dim}_dimensions"


def process_category(
    category: str,
    symbol_dict_df: pl.DataFrame,
    dimensions: list[int],
    force: bool = False,
    verify: bool = True,
) -> None:
    """Process a single category: whiten and save for all dimensions.

    Args:
        category: Category name to process
        symbol_dict_df: DataFrame with symbol -> region mapping
        dimensions: List of target dimensions
        force: If True, overwrite existing files
        verify: If True, run verification checks
    """
    print(f"\n{'=' * 60}")
    print(f"Processing: {category}")
    print(f"Dimensions: {dimensions}")
    print(f"{'=' * 60}")

    # Check which dimensions need processing
    dims_to_process = []
    for dim in dimensions:
        output_dir = get_output_dir(dim)
        all_output_path = output_dir / category / f"all_{category}.h5"
        if force or not all_output_path.exists():
            dims_to_process.append(dim)
        else:
            print(f"  Skipping {dim}-dim: {all_output_path.name} exists")

    if not dims_to_process:
        print("  All dimensions already processed (use --force to overwrite)")
        return

    # Load raw embeddings
    try:
        data = load_raw_embeddings(category, "all")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    print(f"  Loaded {len(data.symbols)} embeddings, shape {data.embeddings.shape}")

    # Compute whitening for all needed dimensions (single SVD)
    print(f"  Computing SVD and whitening for {len(dims_to_process)} dimensions...")
    results = whiten_embeddings_multi(data.embeddings, data.symbols, dims_to_process)

    # Save each dimension
    for dim in dims_to_process:
        result = results[dim]
        output_dir = get_output_dir(dim)

        print(f"\n  --- {dim} dimensions ---")
        print(f"  Whitened shape: {result.whitened.shape}")
        print(f"  Explained variance: {result.explained_variance_ratio.sum():.4f}")

        # Verify whitening
        if verify:
            ok = verify_whitening(result.whitened)
            print(f"  Verification: {'PASSED' if ok else 'FAILED'}")

        # Save "all" file
        all_output_path = output_dir / category / f"all_{category}.h5"
        save_whitened_h5(
            result.symbols,
            result.whitened,
            category,
            all_output_path,
            result.explained_variance_ratio,
            result.n_components,
        )

        # Create symbol -> row index mapping for filtering
        symbol_to_idx = {sym: i for i, sym in enumerate(result.symbols)}

        # Save index-specific files
        for region in INDEX_REGIONS:
            region_symbols = (
                symbol_dict_df.filter(pl.col("region") == region)["symbol"].to_list()
            )
            filtered_symbols = [s for s in region_symbols if s in symbol_to_idx]

            if not filtered_symbols:
                print(f"  WARNING: No symbols for {region}")
                continue

            indices = [symbol_to_idx[s] for s in filtered_symbols]
            filtered_whitened = result.whitened[indices]

            output_path = output_dir / category / f"{region}_{category}.h5"
            save_whitened_h5(
                filtered_symbols,
                filtered_whitened,
                category,
                output_path,
                result.explained_variance_ratio,
                result.n_components,
            )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Whiten embeddings via PCA dimensionality reduction"
    )
    parser.add_argument(
        "--category", type=str, default=None, choices=CATEGORIES,
        help="Process single category (default: all)",
    )
    parser.add_argument(
        "--dimensions", type=int, nargs="+", default=None,
        choices=WHITENED_DIMS,
        help=f"Dimensions to generate (default: all {len(WHITENED_DIMS)})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force regeneration of existing files",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run verification checks on output",
    )
    args = parser.parse_args()

    categories = [args.category] if args.category else CATEGORIES
    dimensions = args.dimensions if args.dimensions else DEFAULT_DIMS

    print("Embedding Whitening Script (Multi-Dimension)")
    print(f"  Input: {RAW_EMBEDDINGS_DIR}")
    print(f"  Output base: {OUTPUT_BASE_DIR}")
    print(f"  Raw dimensions: {RAW_DIM}")
    print(f"  Target dimensions: {dimensions}")
    print(f"  Categories: {len(categories)}")

    symbol_dict_df = load_symbol_dict()
    print(f"  Loaded {len(symbol_dict_df)} symbols from SymbolDict.csv")

    for category in categories:
        process_category(
            category,
            symbol_dict_df,
            dimensions=dimensions,
            force=args.force,
            verify=args.verify,
        )

    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
