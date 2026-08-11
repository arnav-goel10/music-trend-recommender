"""Immutable domain contracts shared by ranking and evaluation."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from math import isfinite
from types import MappingProxyType

_MISSING_RANK_SENTINEL = 999


def _validate_source_ranks(source_ranks: Mapping[str, int]) -> Mapping[str, int]:
    copied = dict(source_ranks)
    for source, rank in copied.items():
        if (
            not isinstance(source, str)
            or not source
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank <= 0
            or rank == _MISSING_RANK_SENTINEL
        ):
            raise ValueError(
                "source ranks must map non-empty source names to positive integer ranks; "
                "omit a source when its rank is missing"
            )
    return MappingProxyType(copied)


def _validate_non_negative(value: float, field_name: str) -> float:
    converted = float(value)
    if not isfinite(converted) or converted < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return converted


@dataclass(frozen=True)
class Candidate:
    """A normalized track candidate and the ranks observed for it."""

    key: str
    title: str
    artist: str
    source_ranks: Mapping[str, int]
    first_seen: date
    spotify_uri: str | None = None
    popularity: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_ranks", _validate_source_ranks(self.source_ranks))


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate paired with its raw, decomposed, and final scores."""

    candidate: Candidate
    raw_score: float
    final_score: float
    components: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_score",
            _validate_non_negative(self.raw_score, "scores"),
        )
        object.__setattr__(
            self,
            "final_score",
            _validate_non_negative(self.final_score, "scores"),
        )
        copied_components = {
            name: _validate_non_negative(value, "component values")
            for name, value in self.components.items()
        }
        object.__setattr__(self, "components", MappingProxyType(copied_components))


@dataclass(frozen=True)
class WeeklySnapshot:
    """Candidates observed in one week with labels for the following week."""

    week: date
    candidates: tuple[Candidate, ...]
    relevant_next_week: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "relevant_next_week", frozenset(self.relevant_next_week))


@dataclass(frozen=True)
class RankingResult:
    """An ordered ranking produced for a weekly snapshot."""

    method: str
    week: date
    ranked_keys: tuple[str, ...]
    seed: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranked_keys", tuple(self.ranked_keys))
