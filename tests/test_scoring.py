import importlib
from datetime import date
from math import isfinite
from typing import Any

import pytest

from music_trend_recommender.domain import Candidate

EXPECTED_COMPONENTS = {
    "source_blend",
    "confirmation",
    "momentum",
    "freshness",
    "age_decay",
    "popularity",
}


def load_scoring() -> Any:
    try:
        return importlib.import_module("music_trend_recommender.ranking.scoring")
    except ModuleNotFoundError:
        pytest.fail("scoring module is not implemented", pytrace=False)


def make_candidate(
    source_ranks: dict[str, int],
    *,
    first_seen: date = date(2026, 1, 1),
    popularity: int | None = None,
) -> Candidate:
    return Candidate(
        key="artist::song",
        title="Song",
        artist="Artist",
        source_ranks=source_ranks,
        first_seen=first_seen,
        popularity=popularity,
    )


def test_source_weights_sum_exactly_to_one() -> None:
    scoring = load_scoring()

    assert set(scoring.SOURCE_WEIGHTS) == {
        "tiktok_popular",
        "tiktok_breakout",
        "youtube",
        "shazam",
        "apple_music",
        "competitor_playlist",
    }
    assert sum(scoring.SOURCE_WEIGHTS.values()) == 1.0


def test_rank_value_is_monotonic_and_bounded() -> None:
    scoring = load_scoring()

    assert (
        1.0
        == scoring.rank_value(1, 100)
        > scoring.rank_value(10, 100)
        > scoring.rank_value(100, 100)
        >= 0.0
    )
    assert scoring.rank_value(101, 100) == 0.0


@pytest.mark.parametrize(("rank", "list_size"), [(0, 100), (-1, 100), (1, 0)])
def test_rank_value_rejects_invalid_boundaries(rank: int, list_size: int) -> None:
    scoring = load_scoring()

    with pytest.raises(ValueError):
        scoring.rank_value(rank, list_size)


def test_missing_sources_are_excluded_from_weight_normalization() -> None:
    scoring = load_scoring()
    rank = 10
    as_of = date(2026, 1, 8)

    youtube_only = scoring.score_candidate(
        make_candidate({"youtube": rank}), as_of=as_of, previous_raw_score=None
    )
    apple_only = scoring.score_candidate(
        make_candidate({"apple_music": rank}), as_of=as_of, previous_raw_score=None
    )
    equal_rank_pair = scoring.score_candidate(
        make_candidate({"youtube": rank, "apple_music": rank}),
        as_of=as_of,
        previous_raw_score=None,
    )

    expected = scoring.rank_value(rank)
    assert youtube_only.raw_score == pytest.approx(expected)
    assert apple_only.raw_score == pytest.approx(expected)
    assert equal_rank_pair.raw_score == pytest.approx(expected)


def test_weighted_source_blend_uses_only_present_known_sources() -> None:
    scoring = load_scoring()
    candidate = make_candidate({"tiktok_popular": 1, "youtube": 100})

    result = scoring.score_candidate(candidate, as_of=date(2026, 1, 8), previous_raw_score=None)

    expected = (0.22 * scoring.rank_value(1) + 0.13 * scoring.rank_value(100)) / (0.22 + 0.13)
    assert result.raw_score == pytest.approx(expected)
    assert result.components["source_blend"] == result.raw_score


def test_momentum_compares_raw_score_stage() -> None:
    scoring = load_scoring()
    candidate = make_candidate({"youtube": 1, "shazam": 2})
    as_of = date(2026, 1, 8)

    baseline = scoring.score_candidate(candidate, as_of=as_of, previous_raw_score=None)
    rising = scoring.score_candidate(
        candidate,
        as_of=as_of,
        previous_raw_score=baseline.raw_score - 0.1,
    )
    falling = scoring.score_candidate(
        candidate,
        as_of=as_of,
        previous_raw_score=baseline.raw_score + 0.2,
    )

    assert rising.raw_score == baseline.raw_score == falling.raw_score
    assert rising.components["momentum"] == 1.20
    assert falling.components["momentum"] == 0.80
    assert rising.final_score > baseline.final_score > falling.final_score


@pytest.mark.parametrize("previous", [-0.1, float("inf"), float("-inf"), float("nan"), True])
def test_invalid_previous_raw_score_is_rejected(previous: object) -> None:
    scoring = load_scoring()

    with pytest.raises(ValueError, match="previous_raw_score"):
        scoring.score_candidate(
            make_candidate({"youtube": 2}),
            as_of=date(2026, 1, 8),
            previous_raw_score=previous,
        )


def test_scoring_does_not_mutate_candidate() -> None:
    scoring = load_scoring()
    supplied_ranks = {"youtube": 3, "shazam": 5}
    candidate = make_candidate(supplied_ranks, popularity=20)
    before = dict(candidate.source_ranks)

    result = scoring.score_candidate(candidate, as_of=date(2026, 1, 8), previous_raw_score=0.5)

    assert result.candidate is candidate
    assert dict(candidate.source_ranks) == before
    assert supplied_ranks == before


def test_scoring_is_deterministic_finite_and_decomposed() -> None:
    scoring = load_scoring()
    candidate = make_candidate(
        {"tiktok_breakout": 2, "shazam": 4, "competitor_playlist": 10},
        popularity=18,
    )

    first = scoring.score_candidate(candidate, as_of=date(2026, 1, 8), previous_raw_score=0.4)
    second = scoring.score_candidate(candidate, as_of=date(2026, 1, 8), previous_raw_score=0.4)

    assert first == second
    assert set(first.components) == EXPECTED_COMPONENTS
    assert all(isfinite(value) and value >= 0.0 for value in first.components.values())
    expected_final = first.raw_score
    for name in EXPECTED_COMPONENTS - {"source_blend"}:
        expected_final *= first.components[name]
    assert first.final_score == pytest.approx(expected_final)


def test_unknown_or_absent_sources_produce_a_zero_diagnostic_score() -> None:
    scoring = load_scoring()

    unknown = scoring.score_candidate(
        make_candidate({"unconfigured_source": 1}),
        as_of=date(2026, 1, 8),
        previous_raw_score=None,
    )
    absent = scoring.score_candidate(
        make_candidate({}), as_of=date(2026, 1, 8), previous_raw_score=None
    )

    assert unknown.raw_score == unknown.final_score == 0.0
    assert absent.raw_score == absent.final_score == 0.0
    assert set(unknown.components) == EXPECTED_COMPONENTS


def test_as_of_before_first_seen_is_rejected() -> None:
    scoring = load_scoring()

    with pytest.raises(ValueError, match="as_of"):
        scoring.score_candidate(
            make_candidate({"youtube": 1}, first_seen=date(2026, 1, 8)),
            as_of=date(2026, 1, 7),
            previous_raw_score=None,
        )


@pytest.mark.parametrize("popularity", [-1, 101, True])
def test_invalid_popularity_is_rejected(popularity: object) -> None:
    scoring = load_scoring()

    with pytest.raises(ValueError, match="popularity"):
        scoring.score_candidate(
            make_candidate({"youtube": 1}, popularity=popularity),
            as_of=date(2026, 1, 8),
            previous_raw_score=None,
        )


def test_freshness_age_decay_and_popularity_have_sensible_boundaries() -> None:
    scoring = load_scoring()
    as_of = date(2026, 1, 15)

    fresh_breakout = scoring.score_candidate(
        make_candidate(
            {"tiktok_breakout": 4, "shazam": 4},
            first_seen=date(2026, 1, 1),
            popularity=24,
        ),
        as_of=as_of,
        previous_raw_score=None,
    )
    older_mainstream = scoring.score_candidate(
        make_candidate(
            {"tiktok_breakout": 4, "shazam": 4},
            first_seen=date(2025, 11, 1),
            popularity=81,
        ),
        as_of=as_of,
        previous_raw_score=None,
    )

    assert fresh_breakout.components["freshness"] == 1.08
    assert fresh_breakout.components["popularity"] == 1.10
    assert older_mainstream.components["freshness"] == 1.0
    assert older_mainstream.components["popularity"] == 0.93
    assert fresh_breakout.components["age_decay"] > older_mainstream.components["age_decay"]
    assert fresh_breakout.final_score > older_mainstream.final_score
