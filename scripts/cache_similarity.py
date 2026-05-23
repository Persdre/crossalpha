#!/usr/bin/env python3
"""Compute and cache similarity percentile rank matrices.

This script computes weighted cosine similarity across categories,
converts to percentile ranks, and saves to disk. The cache is reused
when constructing factors with different (c, k) parameters.

Usage:
    # Cache all embedding types for all region pairs
    python scripts/cache_similarity.py

    # Cache specific configuration
    python scripts/cache_similarity.py --target-region jp_topix \
        --peer-region us_russell_1000 --embedding-type w128

    # Parallel execution
    python scripts/cache_similarity.py --parallel 4

    # Skip existing caches
    python scripts/cache_similarity.py --skip-existing
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.kernels import (
    CATEGORIES,
    CATEGORY_WEIGHTS,
    load_embeddings,
    percentile_rank_fast,
    weighted_similarity_across_categories,
)
from scripts.similarity_cache import (
    cache_exists,
    save_similarity_cache,
)

JP_REGIONS = ["jp_topix", "jp_topix_500"]
US_REGION = "us_russell_1000"

EMBEDDING_TYPES = {
    "raw": "raw",
    "w64": "64_dimensions",
    "w100": "100_dimensions",
    "w128": "128_dimensions",
    "w150": "150_dimensions",
    "w200": "200_dimensions",
    "w256": "256_dimensions",
    "w300": "300_dimensions",
    "w350": "350_dimensions",
    "w400": "400_dimensions",
    "w450": "450_dimensions",
    "w512": "512_dimensions",
    "w768": "768_dimensions",
    "w1024": "1024_dimensions",
}


def compute_and_cache_similarity(
    target_region: str,
    peer_region: str,
    embedding_type_key: str,
    skip_existing: bool = False,
) -> bool:
    """Compute similarity and cache percentile ranks.

    Args:
        target_region: Target region (e.g., 'jp_topix')
        peer_region: Peer region (e.g., 'us_russell_1000')
        embedding_type_key: Embedding type key (e.g., 'w128')
        skip_existing: Skip if cache already exists

    Returns:
        True if cache was created, False if skipped
    """
    if skip_existing and cache_exists(
        target_region, peer_region, embedding_type_key
    ):
        print(f"  SKIP: {target_region} <- {peer_region} / {embedding_type_key}")
        return False

    embedding_type = EMBEDDING_TYPES[embedding_type_key]

    print(f"  Computing: {target_region} <- {peer_region} / {embedding_type_key}")

    target_emb = load_embeddings(target_region, CATEGORIES, embedding_type)
    peer_emb = load_embeddings(peer_region, CATEGORIES, embedding_type)

    if len(target_emb.symbols) == 0 or len(peer_emb.symbols) == 0:
        print("    WARNING: Empty embeddings, skipping")
        return False

    similarity = weighted_similarity_across_categories(
        target_emb.embeddings,
        peer_emb.embeddings,
        target_emb.symbols,
        peer_emb.symbols,
        CATEGORY_WEIGHTS,
    )

    pct_ranks = percentile_rank_fast(similarity, axis=1)

    save_similarity_cache(
        pct_ranks=pct_ranks,
        target_symbols=target_emb.symbols,
        peer_symbols=peer_emb.symbols,
        target_region=target_region,
        peer_region=peer_region,
        embedding_type=embedding_type_key,
    )

    print(f"    Cached: {pct_ranks.shape[0]} x {pct_ranks.shape[1]} matrix")
    return True


def _worker(args: tuple) -> bool:
    """Worker function for parallel processing."""
    target, peer, emb_key, skip = args
    try:
        return compute_and_cache_similarity(target, peer, emb_key, skip)
    except Exception as e:
        print(f"  ERROR: {target} <- {peer} / {emb_key}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and cache similarity percentile rank matrices"
    )
    parser.add_argument(
        "--target-region",
        type=str,
        default=None,
        choices=JP_REGIONS + [US_REGION],
        help="Target region (default: all)",
    )
    parser.add_argument(
        "--peer-region",
        type=str,
        default=None,
        choices=JP_REGIONS + [US_REGION],
        help="Peer region (default: inferred from target)",
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        default=None,
        choices=list(EMBEDDING_TYPES.keys()),
        help="Embedding type (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if cache already exists",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    args = parser.parse_args()

    # Build (target, peer) pairs
    if args.target_region and args.peer_region:
        target_peer_pairs = [(args.target_region, args.peer_region)]
    elif args.target_region:
        if args.target_region in JP_REGIONS:
            target_peer_pairs = [(args.target_region, US_REGION)]
        else:
            target_peer_pairs = [(args.target_region, "jp_topix")]
    else:
        target_peer_pairs = [
            (jp_region, US_REGION) for jp_region in JP_REGIONS
        ] + [(US_REGION, "jp_topix")]

    emb_types = (
        [args.embedding_type] if args.embedding_type
        else list(EMBEDDING_TYPES.keys())
    )

    tasks = [
        (target, peer, emb_key, args.skip_existing)
        for target, peer in target_peer_pairs
        for emb_key in emb_types
    ]

    print("Similarity Cache Generation")
    print(f"  Target-Peer pairs: {target_peer_pairs}")
    print(f"  Embedding types: {len(emb_types)}")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Parallel workers: {args.parallel}")
    print(f"  Skip existing: {args.skip_existing}")
    print("=" * 60)

    if args.parallel > 1:
        n_workers = min(args.parallel, cpu_count(), len(tasks))
        with Pool(n_workers) as pool:
            results = pool.map(_worker, tasks)
        created = sum(results)
    else:
        created = 0
        for task in tasks:
            if _worker(task):
                created += 1

    print("=" * 60)
    print(f"COMPLETE: {created} cache files created")
    print(f"          {len(tasks) - created} skipped or failed")


if __name__ == "__main__":
    main()
