"""Cosine similarity computation kernels (BLAS-optimized)."""

from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between two sets of embeddings.

    Uses NumPy BLAS for optimized matrix multiplication.

    Args:
        embeddings_a: Shape (n_a, dim)
        embeddings_b: Shape (n_b, dim)

    Returns:
        Similarity matrix of shape (n_a, n_b)
    """
    # Normalize rows to unit vectors (vectorized)
    norm_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    # Avoid division by zero
    norm_a = np.where(norm_a == 0, 1.0, norm_a)
    norm_b = np.where(norm_b == 0, 1.0, norm_b)

    a_normalized = embeddings_a / norm_a
    b_normalized = embeddings_b / norm_b

    # BLAS-optimized matrix multiplication
    return a_normalized @ b_normalized.T


def weighted_similarity_across_categories(
    target_embeddings_dict: dict[str, np.ndarray],
    peer_embeddings_dict: dict[str, np.ndarray],
    target_symbols: list[str],
    peer_symbols: list[str],
    category_weights: dict[str, float],
) -> np.ndarray:
    """Compute weighted average similarity across categories.

    Fully vectorized implementation using NumPy broadcasting.

    Args:
        target_embeddings_dict: {category: (n_target, dim) array}
        peer_embeddings_dict: {category: (n_peer, dim) array}
        target_symbols: List of target stock symbols
        peer_symbols: List of peer stock symbols
        category_weights: {category: weight}

    Returns:
        Similarity matrix of shape (n_target, n_peer)
    """
    n_target = len(target_symbols)
    n_peer = len(peer_symbols)

    valid_cats = [
        cat for cat in category_weights
        if cat in target_embeddings_dict and cat in peer_embeddings_dict
    ]

    if not valid_cats:
        return np.zeros((n_target, n_peer), dtype=np.float64)

    n_cat = len(valid_cats)
    matrices = np.empty((n_cat, n_target, n_peer), dtype=np.float64)
    weights = np.empty(n_cat, dtype=np.float64)

    for idx, cat in enumerate(valid_cats):
        matrices[idx] = cosine_similarity_matrix(
            target_embeddings_dict[cat],
            peer_embeddings_dict[cat],
        )
        weights[idx] = category_weights[cat]

    # Vectorized weighted average with NaN handling
    # weights: (n_cat,) -> (n_cat, 1, 1) for broadcasting
    weights_3d = weights[:, np.newaxis, np.newaxis]

    # Create mask for valid (non-NaN) values
    valid_mask = ~np.isnan(matrices)

    # Weighted sum (NaN treated as 0)
    weighted_matrices = np.where(valid_mask, matrices * weights_3d, 0.0)
    weighted_sum = np.sum(weighted_matrices, axis=0)

    # Sum of weights for valid entries
    weight_sum = np.sum(np.where(valid_mask, weights_3d, 0.0), axis=0)

    # Normalize (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(weight_sum > 0, weighted_sum / weight_sum, 0.0)

    return result
