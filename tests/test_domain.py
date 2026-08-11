import importlib
from dataclasses import FrozenInstanceError
from datetime import date
from math import inf, nan
from typing import Any

import pytest


def load_domain() -> Any:
    try:
        return importlib.import_module("music_trend_recommender.domain")
    except ModuleNotFoundError:
        pytest.fail("domain module is not implemented", pytrace=False)


def candidate(domain: Any, source_ranks: dict[str, int] | None = None) -> Any:
    return domain.Candidate(
        key="artist::song",
        title="Song",
        artist="Artist",
        source_ranks={} if source_ranks is None else source_ranks,
        first_seen=date(2026, 1, 1),
    )


def test_candidate_is_frozen_and_defensively_copies_source_ranks() -> None:
    domain = load_domain()
    supplied_ranks = {"youtube": 3}
    item = candidate(domain, supplied_ranks)

    supplied_ranks["youtube"] = 1

    assert dict(item.source_ranks) == {"youtube": 3}
    with pytest.raises(TypeError):
        item.source_ranks["youtube"] = 2
    with pytest.raises(FrozenInstanceError):
        item.title = "Changed"


@pytest.mark.parametrize("invalid_rank", [0, -1, 999, True, 1.5])
def test_candidate_rejects_invalid_or_sentinel_source_ranks(invalid_rank: object) -> None:
    domain = load_domain()

    with pytest.raises(ValueError, match="source ranks"):
        candidate(domain, {"youtube": invalid_rank})


@pytest.mark.parametrize("invalid_score", [-0.1, inf, -inf, nan])
def test_scored_candidate_rejects_invalid_scores(invalid_score: float) -> None:
    domain = load_domain()

    with pytest.raises(ValueError, match="finite and non-negative"):
        domain.ScoredCandidate(
            candidate(domain),
            raw_score=invalid_score,
            final_score=0.2,
            components={},
        )


def test_scored_candidate_defensively_copies_components() -> None:
    domain = load_domain()
    supplied_components = {"source_blend": 0.8}
    scored = domain.ScoredCandidate(
        candidate(domain),
        raw_score=0.8,
        final_score=0.8,
        components=supplied_components,
    )

    supplied_components["source_blend"] = 0.1

    assert dict(scored.components) == {"source_blend": 0.8}
    with pytest.raises(TypeError):
        scored.components["source_blend"] = 0.5


@pytest.mark.parametrize("invalid_component", [-0.1, inf, -inf, nan])
def test_scored_candidate_rejects_invalid_components(invalid_component: float) -> None:
    domain = load_domain()

    with pytest.raises(ValueError, match="component values"):
        domain.ScoredCandidate(
            candidate(domain),
            raw_score=0.8,
            final_score=0.8,
            components={"source_blend": invalid_component},
        )


def test_weekly_snapshot_defensively_copies_candidates_and_labels() -> None:
    domain = load_domain()
    first = candidate(domain)
    supplied_candidates = [first]
    supplied_labels = {first.key}

    snapshot = domain.WeeklySnapshot(
        week=date(2026, 1, 5),
        candidates=supplied_candidates,
        relevant_next_week=supplied_labels,
    )
    supplied_candidates.clear()
    supplied_labels.clear()

    assert snapshot.candidates == (first,)
    assert snapshot.relevant_next_week == frozenset({first.key})


def test_ranking_result_defensively_copies_ranked_keys() -> None:
    domain = load_domain()
    supplied_keys = ["artist::song", "artist::other-song"]

    result = domain.RankingResult(
        method="deterministic",
        week=date(2026, 1, 5),
        ranked_keys=supplied_keys,
        seed=None,
    )
    supplied_keys.reverse()

    assert result.ranked_keys == ("artist::song", "artist::other-song")
