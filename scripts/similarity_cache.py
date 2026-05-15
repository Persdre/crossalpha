#!/usr/bin/env python3
"""Similarity cache utilities for percentile rank matrices.

This module provides functions to compute, save, and load cached
percentile rank matrices. The cache eliminates redundant similarity
computations when sweeping (c, k) parameters.

Cache file format (HDF5):
    - pct_ranks: float64 array of shape (n_target, n_peer)
    - target_symbols: JSON-encoded list of target stock symbols
    - peer_symbols: JSON-encoded list of peer stock symbols
    - metadata: JSON-encoded dict with embedding_type, target_region, etc.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CachedSimilarity(NamedTuple):
    """Container for cached percentile rank data."""

    pct_ranks: np.ndarray
    target_symbols: list[str]
    peer_symbols: list[str]
    metadata: dict


def build_cache_filename(
    target_region: str,
    peer_region: str,
    embedding_type: str,
) -> str:
    """Build standardized cache filename.

    Args:
        target_region: Target region (e.g., 'jp_topix')
        peer_region: Peer region (e.g., 'us_russell_1000')
        embedding_type: Embedding type key (e.g., 'w128')

    Returns:
        Filename like 'pct_ranks_jp_topix_from_us_russell_1000_w128.h5'
    """
    return f"pct_ranks_{target_region}_from_{peer_region}_{embedding_type}.h5"


def get_cache_path(
    target_region: str,
    peer_region: str,
    embedding_type: str,
    cache_dir: Path | None = None,
) -> Path:
    """Get full path to cache file.

    Args:
        target_region: Target region
        peer_region: Peer region
        embedding_type: Embedding type key
        cache_dir: Cache directory (default: cache/similarity/)

    Returns:
        Full path to cache file
    """
    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "cache" / "similarity"
    filename = build_cache_filename(target_region, peer_region, embedding_type)
    return cache_dir / filename


def save_similarity_cache(
    pct_ranks: np.ndarray,
    target_symbols: list[str],
    peer_symbols: list[str],
    target_region: str,
    peer_region: str,
    embedding_type: str,
    cache_dir: Path | None = None,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> Path:
    """Save percentile rank matrix to cache.

    Args:
        pct_ranks: Percentile rank matrix of shape (n_target, n_peer)
        target_symbols: List of target stock symbols
        peer_symbols: List of peer stock symbols
        target_region: Target region identifier
        peer_region: Peer region identifier
        embedding_type: Embedding type key
        cache_dir: Cache directory (default: cache/similarity/)
        compression: HDF5 compression type
        compression_opts: Compression level

    Returns:
        Path to saved cache file
    """
    cache_path = get_cache_path(
        target_region, peer_region, embedding_type, cache_dir
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "target_region": target_region,
        "peer_region": peer_region,
        "embedding_type": embedding_type,
        "n_target": len(target_symbols),
        "n_peer": len(peer_symbols),
    }

    with h5py.File(cache_path, "w") as f:
        f.create_dataset(
            "pct_ranks",
            data=pct_ranks.astype(np.float64),
            compression=compression,
            compression_opts=compression_opts,
        )
        f.attrs["target_symbols"] = json.dumps(target_symbols)
        f.attrs["peer_symbols"] = json.dumps(peer_symbols)
        f.attrs["metadata"] = json.dumps(metadata)

    return cache_path


def load_similarity_cache(
    target_region: str,
    peer_region: str,
    embedding_type: str,
    cache_dir: Path | None = None,
) -> CachedSimilarity:
    """Load percentile rank matrix from cache.

    Args:
        target_region: Target region identifier
        peer_region: Peer region identifier
        embedding_type: Embedding type key
        cache_dir: Cache directory (default: cache/similarity/)

    Returns:
        CachedSimilarity with pct_ranks, symbols, and metadata

    Raises:
        FileNotFoundError: If cache file does not exist
    """
    cache_path = get_cache_path(
        target_region, peer_region, embedding_type, cache_dir
    )

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    with h5py.File(cache_path, "r") as f:
        pct_ranks = f["pct_ranks"][:].astype(np.float64)
        target_symbols = json.loads(f.attrs["target_symbols"])
        peer_symbols = json.loads(f.attrs["peer_symbols"])
        metadata = json.loads(f.attrs["metadata"])

    return CachedSimilarity(
        pct_ranks=pct_ranks,
        target_symbols=target_symbols,
        peer_symbols=peer_symbols,
        metadata=metadata,
    )


def cache_exists(
    target_region: str,
    peer_region: str,
    embedding_type: str,
    cache_dir: Path | None = None,
) -> bool:
    """Check if cache file exists.

    Args:
        target_region: Target region identifier
        peer_region: Peer region identifier
        embedding_type: Embedding type key
        cache_dir: Cache directory

    Returns:
        True if cache file exists
    """
    cache_path = get_cache_path(
        target_region, peer_region, embedding_type, cache_dir
    )
    return cache_path.exists()


def list_cached_configs(
    cache_dir: Path | None = None,
) -> list[tuple[str, str, str]]:
    """List all cached (target_region, peer_region, embedding_type) tuples.

    Args:
        cache_dir: Cache directory

    Returns:
        List of (target_region, peer_region, embedding_type) tuples
    """
    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "cache" / "similarity"

    if not cache_dir.exists():
        return []

    configs = []
    for h5_file in cache_dir.glob("pct_ranks_*.h5"):
        stem = h5_file.stem
        if not stem.startswith("pct_ranks_"):
            continue

        rest = stem[len("pct_ranks_"):]
        if "_from_" not in rest:
            continue

        parts = rest.split("_from_")
        if len(parts) != 2:
            continue

        target_region = parts[0]
        peer_and_emb = parts[1]

        last_underscore = peer_and_emb.rfind("_")
        if last_underscore == -1:
            continue

        peer_region = peer_and_emb[:last_underscore]
        embedding_type = peer_and_emb[last_underscore + 1:]

        configs.append((target_region, peer_region, embedding_type))

    return configs


def encode_ck_params(c: float, k: float) -> str:
    """Encode (c, k) parameters into a filename-safe string.

    Uses underscore decimal notation: c=0.97 -> 'c0_97', c=0.995 -> 'c0_995'

    Args:
        c: Sigmoid center parameter
        k: Sigmoid steepness parameter

    Returns:
        String like 'c0_97_k30'

    Examples:
        >>> encode_ck_params(0.99, 50)
        'c0_99_k50'
        >>> encode_ck_params(0.97, 30)
        'c0_97_k30'
        >>> encode_ck_params(0.995, 80)
        'c0_995_k80'
    """
    c_str = str(c).replace(".", "_")
    k_str = str(int(k))
    return f"c{c_str}_k{k_str}"


def encode_threshold_param(threshold: float) -> str:
    """Encode threshold parameter into a filename-safe string.

    Uses fixed 2-decimal precision for consistent naming.

    Args:
        threshold: Percentile rank cutoff (e.g. 0.95)

    Returns:
        String like 't0_95'

    Examples:
        >>> encode_threshold_param(0.95)
        't0_95'
        >>> encode_threshold_param(0.80)
        't0_80'
    """
    t_str = f"{threshold:.2f}".replace(".", "_")
    return f"t{t_str}"


def decode_ck_params(encoded: str) -> tuple[float, float]:
    """Decode (c, k) parameters from encoded string.

    Args:
        encoded: String like 'c0_97_k30'

    Returns:
        Tuple of (c, k) as floats

    Raises:
        ValueError: If string format is invalid

    Examples:
        >>> decode_ck_params('c0_99_k50')
        (0.99, 50.0)
        >>> decode_ck_params('c0_97_k30')
        (0.97, 30.0)
    """
    match = re.match(r"c(\d+_\d+)_k(\d+)", encoded)
    if not match:
        raise ValueError(f"Invalid encoded ck params: {encoded}")

    c_str = match.group(1).replace("_", ".")
    k_str = match.group(2)

    return float(c_str), float(k_str)


def build_factor_name(
    lookback: int,
    embedding_type: str,
    c: float,
    k: float,
) -> str:
    """Build standardized factor name with (c, k) encoding.

    Args:
        lookback: Lookback period in months
        embedding_type: Embedding type key (e.g., 'w128')
        c: Sigmoid center parameter
        k: Sigmoid steepness parameter

    Returns:
        Factor name like 'momentum_12mo_w128_c0_97_k30'
    """
    ck_encoded = encode_ck_params(c, k)
    return f"momentum_{lookback}mo_{embedding_type}_{ck_encoded}"


def parse_factor_name(factor_name: str) -> dict:
    """Parse factor name into components.

    Args:
        factor_name: Factor name like 'momentum_12mo_w128_c0_97_k30'

    Returns:
        Dict with keys: lookback, embedding_type, c, k

    Raises:
        ValueError: If factor name format is invalid
    """
    pattern = r"momentum_(\d+)mo_(\w+)_c(\d+_\d+)_k(\d+)"
    match = re.match(pattern, factor_name)

    if not match:
        raise ValueError(f"Invalid factor name format: {factor_name}")

    lookback = int(match.group(1))
    embedding_type = match.group(2)
    c_str = match.group(3).replace("_", ".")
    k = float(match.group(4))

    return {
        "lookback": lookback,
        "embedding_type": embedding_type,
        "c": float(c_str),
        "k": k,
    }
