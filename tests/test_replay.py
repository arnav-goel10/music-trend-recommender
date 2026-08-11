"""Behavioral tests for leakage-safe chronological replay."""

import importlib
from datetime import date, timedelta
from types import ModuleType

import pytest

from music_trend_recommender.domain import Candidate, WeeklySnapshot

EXPECTED_ROW_KEYS = {
    "week",
    "method",
    "k",
    "seed_count",
    "ranked_keys",
    "precision_at_k",
    "precision_at_k_std",
    "recall_at_k",
    "recall_at_k_std",
    "ndcg_at_k",
    "ndcg_at_k_std",
    "source_coverage",
    "source_coverage_std",
}


def load_replay() -> ModuleType:
    """Load replay after collection so an absent module is an assertion RED."""
    try:
        return importlib.import_module("music_trend_recommender.evaluation.replay")
    except ModuleNotFoundError:
        pytest.fail("temporal replay is not implemented", pytrace=False)


def candidate(
    key: str,
    ranks: dict[str, int],
    *,
    first_seen: date = date(2026, 1, 1),
) -> Candidate:
    return Candidate(
        key=key,
        title=f"Title {key}",
        artist=f"Artist {key}",
        source_ranks=ranks,
        first_seen=first_seen,
    )


def snapshot(
    week: date,
    candidates: tuple[Candidate, ...],
    relevant: frozenset[str] = frozenset(),
) -> WeeklySnapshot:
    return WeeklySnapshot(week=week, candidates=candidates, relevant_next_week=relevant)


def metric_values(row: dict[str, float | str | int]) -> tuple[float | str | int, ...]:
    return tuple(row[key] for key in sorted(EXPECTED_ROW_KEYS - {"ranked_keys"}))


def test_future_features_and_candidates_cannot_change_an_earlier_ranking() -> None:
    replay_module = load_replay()
    week_one = snapshot(
        date(2026, 1, 5),
        (
            candidate("alpha", {"youtube": 1}),
            candidate("bravo", {"youtube": 20}),
        ),
        frozenset({"alpha"}),
    )
    ordinary_future = snapshot(
        date(2026, 1, 12),
        (candidate("future", {"shazam": 100}),),
    )
    extreme_future = snapshot(
        date(2026, 1, 12),
        (
            candidate("future", {"shazam": 1, "tiktok_popular": 1}),
            candidate("future-entry", {"youtube": 1, "apple_music": 1}),
        ),
    )

    ordinary_first = replay_module.replay(
        [week_one, ordinary_future], k=2, method="deterministic", seeds=[7, 9]
    )[0]
    extreme_first = replay_module.replay(
        [week_one, extreme_future], k=2, method="deterministic", seeds=[1]
    )[0]

    assert ordinary_first["ranked_keys"] == extreme_first["ranked_keys"] == "alpha|bravo"
    assert metric_values(ordinary_first) == metric_values(extreme_first)


def test_relevance_labels_change_metrics_but_never_the_ranking() -> None:
    replay_module = load_replay()
    week = date(2026, 1, 5)
    candidates = (
        candidate("alpha", {"youtube": 1}),
        candidate("bravo", {"youtube": 20}),
    )

    alpha_relevant = replay_module.replay(
        [snapshot(week, candidates, frozenset({"alpha"}))],
        k=1,
        method="deterministic",
        seeds=[],
    )[0]
    bravo_relevant = replay_module.replay(
        [snapshot(week, candidates, frozenset({"bravo"}))],
        k=1,
        method="deterministic",
        seeds=[999],
    )[0]

    assert alpha_relevant["ranked_keys"] == bravo_relevant["ranked_keys"] == "alpha"
    assert alpha_relevant["precision_at_k"] == 1.0
    assert bravo_relevant["precision_at_k"] == 0.0


def test_momentum_carries_the_prior_raw_score_stage_between_weeks() -> None:
    replay_module = load_replay()
    week_one = snapshot(
        date(2026, 1, 5),
        (
            candidate("z-rising", {"youtube": 100}),
            candidate("a-steady", {"youtube": 10}),
        ),
    )
    week_two = snapshot(
        date(2026, 1, 12),
        (
            candidate("z-rising", {"youtube": 10}),
            candidate("a-steady", {"youtube": 10}),
        ),
    )

    rows = replay_module.replay([week_one, week_two], k=2, method="deterministic", seeds=[])

    assert rows[0]["ranked_keys"] == "a-steady|z-rising"
    assert rows[1]["ranked_keys"] == "z-rising|a-steady"


def test_out_of_order_input_is_sorted_and_duplicate_weeks_are_rejected() -> None:
    replay_module = load_replay()
    first = snapshot(date(2026, 1, 5), (candidate("alpha", {"youtube": 1}),))
    second = snapshot(date(2026, 1, 12), (candidate("bravo", {"shazam": 1}),))

    rows = replay_module.replay([second, first], k=1, method="deterministic", seeds=[])

    assert [row["week"] for row in rows] == ["2026-01-05", "2026-01-12"]
    with pytest.raises(ValueError, match="duplicate week"):
        replay_module.replay([first, first], k=1, method="deterministic", seeds=[])


@pytest.mark.parametrize(
    ("method", "seeds"),
    [("deterministic", []), ("gumbel", [7])],
)
def test_replay_rejects_duplicate_candidate_keys_within_a_week(
    method: str,
    seeds: list[int],
) -> None:
    replay_module = load_replay()
    week = snapshot(
        date(2026, 1, 5),
        (
            candidate("duplicate", {"youtube": 1}),
            candidate("duplicate", {"shazam": 100}),
        ),
        frozenset({"duplicate"}),
    )

    with pytest.raises(ValueError, match="duplicate candidate keys.*2026-01-05"):
        replay_module.replay([week], k=2, method=method, seeds=seeds)


def test_repeated_gumbel_seeds_are_reproducible_order_independent_aggregates() -> None:
    replay_module = load_replay()
    week = snapshot(
        date(2026, 1, 5),
        tuple(
            candidate(
                key,
                {"youtube": index + 1, "shazam": 8 - index},
                first_seen=date(2026, 1, 1),
            )
            for index, key in enumerate("abcdefgh")
        ),
        frozenset({"a", "c", "e"}),
    )

    first = replay_module.replay([week], k=3, method="gumbel", seeds=[3, 1, 2])
    second = replay_module.replay([week], k=3, method="gumbel", seeds=[1, 2, 3])

    assert first == second
    assert first[0]["seed_count"] == 3
    assert first[0]["ranked_keys"].startswith("1:")
    assert all(first[0][name] >= 0.0 for name in EXPECTED_ROW_KEYS if name.endswith("_std"))


def test_deterministic_replay_has_one_trial_and_ignores_seed_values() -> None:
    replay_module = load_replay()
    week = snapshot(date(2026, 1, 5), (candidate("alpha", {"youtube": 1}),))

    without_seeds = replay_module.replay([week], k=1, method="deterministic", seeds=[])
    with_seeds = replay_module.replay([week], k=1, method="deterministic", seeds=[1, 2, 3])

    assert without_seeds == with_seeds
    assert without_seeds[0]["seed_count"] == 1
    assert without_seeds[0]["ranked_keys"] == "alpha"


def test_replay_reports_source_coverage_and_a_stable_scalar_schema() -> None:
    replay_module = load_replay()
    week = snapshot(
        date(2026, 1, 5),
        (
            candidate("alpha", {"youtube": 1}),
            candidate("bravo", {"shazam": 50}),
        ),
    )

    row = replay_module.replay([week], k=1, method="deterministic", seeds=[])[0]

    assert set(row) == EXPECTED_ROW_KEYS
    assert row["source_coverage"] == pytest.approx(0.5)
    assert row["source_coverage_std"] == 0.0
    assert all(isinstance(value, (float, str, int)) for value in row.values())


def test_replay_does_not_mutate_snapshots_candidates_or_source_rank_inputs() -> None:
    replay_module = load_replay()
    supplied_ranks = {"youtube": 2, "shazam": 4}
    track = candidate("alpha", supplied_ranks)
    week = snapshot(date(2026, 1, 5), (track,), frozenset({"alpha"}))
    before = (
        week.candidates,
        week.relevant_next_week,
        dict(track.source_ranks),
        dict(supplied_ranks),
    )

    replay_module.replay([week], k=1, method="gumbel", seeds=[3, 7])

    after = (
        week.candidates,
        week.relevant_next_week,
        dict(track.source_ranks),
        dict(supplied_ranks),
    )
    assert after == before


def test_invalid_method_and_missing_gumbel_seeds_are_rejected() -> None:
    replay_module = load_replay()
    week = snapshot(date(2026, 1, 5), (candidate("alpha", {"youtube": 1}),))

    with pytest.raises(ValueError, match="method"):
        replay_module.replay([week], k=1, method="roulette", seeds=[])
    with pytest.raises(ValueError, match="at least one seed"):
        replay_module.replay([week], k=1, method="gumbel", seeds=[])
    with pytest.raises(ValueError, match="at least one seed"):
        replay_module.replay([], k=1, method="gumbel", seeds=[])


def test_state_is_carried_across_an_absent_intermediate_week() -> None:
    replay_module = load_replay()
    first = snapshot(
        date(2026, 1, 5),
        (
            candidate("z-returning", {"youtube": 100}),
            candidate("a-peer", {"youtube": 10}),
        ),
    )
    middle = snapshot(
        date(2026, 1, 12),
        (candidate("middle-only", {"shazam": 1}),),
    )
    last = snapshot(
        date(2026, 1, 19),
        (
            candidate("z-returning", {"youtube": 10}),
            candidate("a-peer", {"youtube": 10}),
        ),
    )

    rows = replay_module.replay([first, middle, last], k=2, method="deterministic", seeds=[])

    assert rows[2]["ranked_keys"] == "z-returning|a-peer"


def test_dates_need_not_be_exactly_one_week_apart() -> None:
    replay_module = load_replay()
    start = date(2026, 1, 5)
    snapshots = [
        snapshot(start, (candidate("alpha", {"youtube": 1}),)),
        snapshot(start + timedelta(days=8), (candidate("alpha", {"youtube": 2}),)),
    ]

    rows = replay_module.replay(snapshots, k=1, method="deterministic", seeds=[])

    assert [row["week"] for row in rows] == ["2026-01-05", "2026-01-13"]
