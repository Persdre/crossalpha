"""Kernel functions for momentum factor construction."""

from scripts.kernels.similarity import (
    cosine_similarity_matrix,
    weighted_similarity_across_categories,
)
from scripts.kernels.ranking import percentile_rank, percentile_rank_fast
from scripts.kernels.returns import (
    cumulative_return,
    monthly_return,
    sector_relative_return,
    sector_relative_return_df,
    tradable_monthly_return,
)
from scripts.kernels.weighting import (
    sigmoid_weights,
    threshold_weights,
    compute_momentum_score,
    CATEGORY_WEIGHTS,
    CATEGORIES,
)
from scripts.kernels.loaders import (
    EmbeddingSet,
    load_embeddings,
    load_symbol_sectors,
    load_monthly_close_prices,
    load_daily_close_prices,
)

__all__ = [
    "cosine_similarity_matrix",
    "weighted_similarity_across_categories",
    "percentile_rank",
    "percentile_rank_fast",
    "cumulative_return",
    "monthly_return",
    "tradable_monthly_return",
    "sector_relative_return",
    "sector_relative_return_df",
    "sigmoid_weights",
    "threshold_weights",
    "compute_momentum_score",
    "CATEGORY_WEIGHTS",
    "CATEGORIES",
    "EmbeddingSet",
    "load_embeddings",
    "load_symbol_sectors",
    "load_monthly_close_prices",
    "load_daily_close_prices",
]
