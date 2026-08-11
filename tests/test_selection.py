"""Behavioral tests for deterministic and exploratory top-k selection."""

import importlib
import random
from collections.abc import Iterator
from datetime import date
from types import ModuleType

import pytest

from music_trend_recommender.domain import Candidate, ScoredCandidate


def load_selection() -> ModuleType:
    """Load selection after collection so an absent module is an assertion RED."""
    try:
        return importlib.import_module("music_trend_recommender.ranking.selection")
    except ModuleNotFoundError:
        pytest.fail("selection module is not implemented", pytrace=False)


def scored(key: str, final_score: float) -> ScoredCandidate:
    candidate = Candidate(
        key=key,
        title=f"Title {key}",
        artist=f"Artist {key}",
        source_ranks={"youtube": 1},
        first_seen=date(2026, 1, 1),
    )
    return ScoredCandidate(
        candidate=candidate,
        raw_score=final_score,
        final_score=final_score,
        components={"source_blend": final_score},
    )


def keys(items: list[ScoredCandidate]) -> list[str]:
    return [item.candidate.key for item in items]


def test_deterministic_top_k_orders_by_score_then_key() -> None:
    selection = load_selection()
    items = [scored("charlie", 0.8), scored("bravo", 0.9), scored("alpha", 0.9)]

    ranked = selection.deterministic_top_k(items, k=3)

    assert keys(ranked) == ["alpha", "bravo", "charlie"]


def test_selection_does_not_mutate_source_or_candidates() -> None:
    selection = load_selection()
    items = [scored("bravo", 0.6), scored("alpha", 0.9), scored("charlie", 0.3)]
    before_order = tuple(items)
    before_values = tuple(
        (item.raw_score, item.final_score, dict(item.components), dict(item.candidate.source_ranks))
        for item in items
    )

    deterministic = selection.deterministic_top_k(items, k=2)
    exploratory = selection.gumbel_top_k(items, k=2, temperature=0.7, seed=42)

    assert tuple(items) == before_order
    assert (
        tuple(
            (
                item.raw_score,
                item.final_score,
                dict(item.components),
                dict(item.candidate.source_ranks),
            )
            for item in items
        )
        == before_values
    )
    assert all(any(result is source for source in items) for result in deterministic + exploratory)


def test_gumbel_top_k_is_repeatable_for_a_seed_and_has_no_duplicates() -> None:
    selection = load_selection()
    items = [
        scored("alpha", 0.9),
        scored("bravo", 0.7),
        scored("charlie", 0.5),
        scored("delta", 0.3),
    ]

    first = selection.gumbel_top_k(items, k=3, temperature=0.7, seed=42)
    second = selection.gumbel_top_k(items, k=3, temperature=0.7, seed=42)

    assert keys(first) == keys(second)
    assert len(first) == 3
    assert len(set(keys(first))) == 3


def test_gumbel_priority_uses_log_weight_divided_by_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = load_selection()
    items = [scored("low", 0.1), scored("high", 0.2)]
    noises: Iterator[float] = iter((1.0, 0.0))

    def controlled_gumbel(_rng: random.Random) -> float:
        return next(noises)

    monkeypatch.setattr(selection, "_gumbel", controlled_gumbel)

    ranked = selection.gumbel_top_k(items, k=2, temperature=0.5, seed=1)

    # Correct priorities are log(0.1) / 0.5 + 1 and log(0.2) / 0.5 + 0.
    # A score / temperature implementation would incorrectly put "low" first.
    assert keys(ranked) == ["high", "low"]


@pytest.mark.parametrize("temperature", [0.0, -0.1, float("nan"), float("inf"), float("-inf")])
def test_gumbel_top_k_rejects_non_positive_or_non_finite_temperature(
    temperature: float,
) -> None:
    selection = load_selection()

    with pytest.raises(ValueError, match="^temperature must be positive and finite$"):
        selection.gumbel_top_k([scored("alpha", 0.5)], k=1, temperature=temperature, seed=1)


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_returns_an_empty_selection(k: int) -> None:
    selection = load_selection()
    items = [scored("alpha", 0.5), scored("bravo", 0.4)]

    assert selection.deterministic_top_k(items, k=k) == []
    assert selection.gumbel_top_k(items, k=k, temperature=1.0, seed=1) == []


def test_k_larger_than_input_returns_each_item_once() -> None:
    selection = load_selection()
    items = [scored("alpha", 0.8), scored("bravo", 0.2)]

    deterministic = selection.deterministic_top_k(items, k=10)
    exploratory = selection.gumbel_top_k(items, k=10, temperature=1.0, seed=7)

    assert keys(deterministic) == ["alpha", "bravo"]
    assert len(exploratory) == 2
    assert set(keys(exploratory)) == {"alpha", "bravo"}


def test_zero_scores_are_clamped_and_tie_break_by_key(monkeypatch: pytest.MonkeyPatch) -> None:
    selection = load_selection()
    items = [scored("bravo", 0.0), scored("alpha", 0.0)]
    monkeypatch.setattr(selection, "_gumbel", lambda _rng: 0.0)

    ranked = selection.gumbel_top_k(items, k=2, temperature=1.0, seed=3)

    assert keys(ranked) == ["alpha", "bravo"]


def test_gumbel_top_k_does_not_advance_module_global_random_state() -> None:
    selection = load_selection()
    items = [scored("alpha", 0.8), scored("bravo", 0.2)]
    original_state = random.getstate()
    try:
        random.seed(8675309)
        state_before = random.getstate()

        selection.gumbel_top_k(items, k=1, temperature=0.8, seed=19)

        assert random.getstate() == state_before
    finally:
        random.setstate(original_state)
