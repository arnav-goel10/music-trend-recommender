"""Exact, zero-safe metrics for binary-relevance rankings."""

from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from math import log2

from music_trend_recommender.domain import ScoredCandidate


def _validate_unique_ranked_keys(ranked_keys: Sequence[str]) -> None:
    if len(set(ranked_keys)) != len(ranked_keys):
        raise ValueError("ranked_keys must contain unique keys")


def precision_at_k(ranked_keys: Sequence[str], relevant: AbstractSet[str], k: int) -> float:
    """Return relevant results in the first ``k`` positions divided by ``k``."""
    _validate_unique_ranked_keys(ranked_keys)
    if k <= 0:
        return 0.0
    hits = sum(key in relevant for key in ranked_keys[:k])
    return hits / k


def recall_at_k(ranked_keys: Sequence[str], relevant: AbstractSet[str], k: int) -> float:
    """Return the fraction of relevant keys retrieved in the first ``k`` positions."""
    _validate_unique_ranked_keys(ranked_keys)
    if k <= 0 or not relevant:
        return 0.0
    hits = sum(key in relevant for key in ranked_keys[:k])
    return hits / len(relevant)


def ndcg_at_k(ranked_keys: Sequence[str], relevant: AbstractSet[str], k: int) -> float:
    """Return binary-relevance normalized discounted cumulative gain at ``k``."""
    _validate_unique_ranked_keys(ranked_keys)
    if k <= 0 or not relevant:
        return 0.0

    discounted_gain = sum(
        1.0 / log2(position + 1)
        for position, key in enumerate(ranked_keys[:k], start=1)
        if key in relevant
    )
    ideal_length = min(len(relevant), k)
    ideal_gain = sum(1.0 / log2(position + 1) for position in range(1, ideal_length + 1))
    return discounted_gain / ideal_gain


def source_coverage_at_k(
    ranked: Sequence[ScoredCandidate],
    *,
    available_sources: AbstractSet[str],
    k: int,
) -> float:
    """Return the fraction of available sources represented in the first ``k`` results."""
    if k <= 0 or not available_sources:
        return 0.0
    represented = {
        source
        for item in ranked[:k]
        for source in item.candidate.source_ranks
        if source in available_sources
    }
    return len(represented) / len(available_sources)
