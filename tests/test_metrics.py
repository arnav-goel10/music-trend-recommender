"""Exact, independently derived tests for temporal ranking metrics."""

import importlib
from datetime import date
from types import ModuleType

import pytest

from music_trend_recommender.domain import Candidate, ScoredCandidate


def load_metrics() -> ModuleType:
    """Load metrics after collection so an absent module is an assertion RED."""
    try:
        return importlib.import_module("music_trend_recommender.evaluation.metrics")
    except ModuleNotFoundError:
        pytest.fail("evaluation metrics are not implemented", pytrace=False)


def scored(key: str, source_ranks: dict[str, int]) -> ScoredCandidate:
    candidate = Candidate(
        key=key,
        title=f"Title {key}",
        artist=f"Artist {key}",
        source_ranks=source_ranks,
        first_seen=date(2026, 1, 1),
    )
    return ScoredCandidate(
        candidate=candidate,
        raw_score=0.5,
        final_score=0.5,
        components={"source_blend": 0.5},
    )


def test_ranking_metrics_match_a_hand_calculated_example() -> None:
    metrics = load_metrics()
    ranked = ["a", "b", "c"]
    relevant = {"a", "c"}

    assert metrics.precision_at_k(ranked, relevant, 2) == pytest.approx(0.5)
    assert metrics.recall_at_k(ranked, relevant, 3) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(ranked, relevant, 3) == pytest.approx(
        (1.0 + 0.5) / (1.0 + 1.0 / 1.584962500721156),
    )


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_returns_zero_metrics(k: int) -> None:
    metrics = load_metrics()

    assert metrics.precision_at_k(["a"], {"a"}, k) == 0.0
    assert metrics.recall_at_k(["a"], {"a"}, k) == 0.0
    assert metrics.ndcg_at_k(["a"], {"a"}, k) == 0.0


def test_empty_relevance_is_zero_safe() -> None:
    metrics = load_metrics()

    assert metrics.precision_at_k(["a", "b"], set(), 2) == 0.0
    assert metrics.recall_at_k(["a", "b"], set(), 2) == 0.0
    assert metrics.ndcg_at_k(["a", "b"], set(), 2) == 0.0


def test_precision_uses_requested_cutoff_when_ranking_is_shorter() -> None:
    metrics = load_metrics()

    assert metrics.precision_at_k(["a"], {"a"}, 3) == pytest.approx(1.0 / 3.0)
    assert metrics.recall_at_k(["a"], {"a"}, 3) == 1.0
    assert metrics.ndcg_at_k(["a"], {"a"}, 3) == 1.0


@pytest.mark.parametrize("metric_name", ["precision_at_k", "recall_at_k", "ndcg_at_k"])
def test_ranking_metrics_reject_duplicate_ranked_keys(metric_name: str) -> None:
    metrics = load_metrics()
    metric = getattr(metrics, metric_name)

    with pytest.raises(ValueError, match="ranked_keys must contain unique keys"):
        metric(["a", "a"], {"a"}, 2)


def test_source_coverage_counts_distinct_represented_sources() -> None:
    metrics = load_metrics()
    ranked = [
        scored("alpha", {"youtube": 1, "shazam": 2}),
        scored("bravo", {"youtube": 3}),
        scored("charlie", {"apple_music": 4}),
    ]

    assert metrics.source_coverage_at_k(
        ranked,
        available_sources={"youtube", "shazam", "apple_music", "tiktok_breakout"},
        k=2,
    ) == pytest.approx(0.5)


def test_source_coverage_is_zero_safe() -> None:
    metrics = load_metrics()

    assert metrics.source_coverage_at_k([], available_sources=set(), k=5) == 0.0
    assert (
        metrics.source_coverage_at_k(
            [scored("alpha", {"youtube": 1})],
            available_sources={"youtube"},
            k=0,
        )
        == 0.0
    )
