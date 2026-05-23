"""Retrieval tools for the Phase-C agentic search loop.

Exposes vector search over a pre-normalized candidate matrix. The higher-level
pipeline in run_agentic_retrieval.py builds the candidate matrix by averaging
per-category US Russell 1000 embeddings (128-d OpenAI projection).
"""

from __future__ import annotations

import numpy as np


def search_us_semantic_by_vec(query_vec: np.ndarray, candidate_vecs: np.ndarray,
                              tickers: list[str], k: int = 10) -> list[dict]:
    """Return top-k {ticker, score} by cosine similarity.

    query_vec: 1-D vector; candidate_vecs: (n, d), rows may contain NaN (missing).
    Rows with any NaN produce score=-inf so they rank last without crashing.
    The function does NOT assume candidate_vecs are pre-normalized; it
    normalizes both sides to unit length for safety.
    """
    q = np.asarray(query_vec, dtype=np.float64)
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        return [{"ticker": t, "score": 0.0} for t in tickers[:k]]
    q = q / qn

    cand = np.asarray(candidate_vecs, dtype=np.float64)
    norms = np.linalg.norm(cand, axis=1)
    bad = ~np.isfinite(norms) | (norms < 1e-12)
    norms_safe = np.where(bad, 1.0, norms)
    cand_unit = cand / norms_safe[:, None]
    scores = cand_unit @ q
    scores = np.where(bad, -np.inf, scores)

    order = np.argsort(-scores)[:k]
    return [{"ticker": tickers[i], "score": float(scores[i])} for i in order]


def build_us_embedding_matrix(categories: list[str],
                              embedding_type: str = "128_dimensions",
                              region: str = "us_russell_1000"):
    """Load + average per-category US embeddings into a single (n_stocks, d) matrix.

    Returns (matrix, tickers). Missing categories are skipped; rows for stocks
    absent from all categories will contain NaN and are treated as unrankable by
    search_us_semantic_by_vec.
    """
    from scripts.kernels.loaders import load_embeddings

    emb_set = load_embeddings(region=region, categories=categories,
                              embedding_type=embedding_type)
    stacks = []
    for cat in categories:
        arr = emb_set.embeddings.get(cat)
        if arr is None:
            continue
        stacks.append(arr)
    if not stacks:
        raise RuntimeError("No category embeddings were loaded")
    # Mean across categories, ignoring NaN per cell
    stack_arr = np.stack(stacks, axis=0)  # (n_cat, n_stocks, d)
    mean = np.nanmean(stack_arr, axis=0)  # (n_stocks, d)
    return mean, list(emb_set.symbols)
