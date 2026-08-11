"""Deterministic, decomposed scoring for multi-source music signals.

The score is diagnostic: its hand-set factors make ranking decisions inspectable,
but do not by themselves demonstrate recommendation-quality improvement.
"""

from collections.abc import Mapping
from datetime import date
from math import exp, isclose, isfinite, log
from types import MappingProxyType

from music_trend_recommender.domain import Candidate, ScoredCandidate

SOURCE_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "tiktok_popular": 0.22,
        "tiktok_breakout": 0.13,
        "youtube": 0.13,
        "shazam": 0.17,
        "apple_music": 0.15,
        "competitor_playlist": 0.20,
    }
)

_MOMENTUM_THRESHOLD_ABS_TOLERANCE = 1e-12


def rank_value(rank: int, list_size: int = 100) -> float:
    """Map a positive chart rank to a bounded log-linear relevance value."""
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or rank <= 0
        or not isinstance(list_size, int)
        or isinstance(list_size, bool)
        or list_size <= 0
    ):
        raise ValueError("rank and list_size must be positive integers")
    return max(0.0, 1.0 - log(rank) / log(list_size + 1))


def _source_blend(candidate: Candidate) -> tuple[float, int]:
    present = {
        source: rank for source, rank in candidate.source_ranks.items() if source in SOURCE_WEIGHTS
    }
    denominator = sum(SOURCE_WEIGHTS[source] for source in present)
    if denominator == 0.0:
        return 0.0, 0
    numerator = sum(SOURCE_WEIGHTS[source] * rank_value(rank) for source, rank in present.items())
    return numerator / denominator, len(present)


def _confirmation_factor(source_count: int) -> float:
    if source_count <= 1:
        return 1.0
    if source_count == 2:
        return 1.20
    if source_count == 3:
        return 1.35
    if source_count == 4:
        return 1.45
    return 1.50


def _validate_previous_raw_score(previous_raw_score: float | None) -> None:
    if previous_raw_score is None:
        return
    if (
        isinstance(previous_raw_score, bool)
        or not isinstance(previous_raw_score, (int, float))
        or not isfinite(previous_raw_score)
        or previous_raw_score < 0.0
    ):
        raise ValueError("previous_raw_score must be finite and non-negative")


def _momentum_factor(raw_score: float, previous_raw_score: float | None) -> float:
    if previous_raw_score is None:
        return 1.0
    delta = raw_score - previous_raw_score
    if delta > 0.05 or isclose(
        delta,
        0.05,
        rel_tol=0.0,
        abs_tol=_MOMENTUM_THRESHOLD_ABS_TOLERANCE,
    ):
        return 1.20
    if delta < -0.15 or isclose(
        delta,
        -0.15,
        rel_tol=0.0,
        abs_tol=_MOMENTUM_THRESHOLD_ABS_TOLERANCE,
    ):
        return 0.80
    return 1.0


def _age_decay_factor(age_days: int) -> float:
    """Return a stable sigmoid with a five-week midpoint."""
    exponent = age_days / 7.0 - 5.0
    if exponent >= 0.0:
        inverse = exp(-exponent)
        return inverse / (1.0 + inverse)
    return 1.0 / (1.0 + exp(exponent))


def _popularity_factor(popularity: int | None, source_count: int) -> float:
    if popularity is None:
        return 1.0
    if (
        not isinstance(popularity, int)
        or isinstance(popularity, bool)
        or not 0 <= popularity <= 100
    ):
        raise ValueError("popularity must be an integer from 0 to 100")
    if popularity > 80:
        return 0.93
    if popularity < 25 and source_count >= 2:
        return 1.10
    return 1.0


def score_candidate(
    candidate: Candidate,
    *,
    as_of: date,
    previous_raw_score: float | None,
) -> ScoredCandidate:
    """Score a candidate without mutating it or comparing incompatible score stages.

    Unknown sources are ignored. A candidate with no configured source produces a
    zero diagnostic score rather than receiving artificial evidence.
    """
    if as_of < candidate.first_seen:
        raise ValueError("as_of cannot be before candidate.first_seen")
    _validate_previous_raw_score(previous_raw_score)

    raw_score, source_count = _source_blend(candidate)
    age_days = (as_of - candidate.first_seen).days
    confirmation = _confirmation_factor(source_count)
    momentum = _momentum_factor(raw_score, previous_raw_score)
    freshness = 1.08 if "tiktok_breakout" in candidate.source_ranks and age_days <= 14 else 1.0
    age_decay = _age_decay_factor(age_days)
    popularity = _popularity_factor(candidate.popularity, source_count)

    components = {
        "source_blend": raw_score,
        "confirmation": confirmation,
        "momentum": momentum,
        "freshness": freshness,
        "age_decay": age_decay,
        "popularity": popularity,
    }
    final_score = raw_score * confirmation * momentum * freshness * age_decay * popularity
    return ScoredCandidate(
        candidate=candidate,
        raw_score=raw_score,
        final_score=final_score,
        components=components,
    )
