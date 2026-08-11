"""Strict JSON file and directory adapter for credential-free snapshots."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from json import JSONDecodeError
from pathlib import Path

from music_trend_recommender.domain import Candidate, WeeklySnapshot

_SNAPSHOT_FIELDS = {"week", "candidates", "relevant_next_week"}
_CANDIDATE_REQUIRED_FIELDS = {"key", "title", "artist", "source_ranks", "first_seen"}
_CANDIDATE_OPTIONAL_FIELDS = {"spotify_uri", "popularity"}


class SnapshotFormatError(ValueError):
    """A local snapshot does not satisfy the documented JSON contract."""


def _object(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SnapshotFormatError(f"{context} must be an object")
    return value


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SnapshotFormatError(f"{context} must be a non-empty string")
    return value


def _iso_date(value: object, *, context: str) -> date:
    if not isinstance(value, str):
        raise SnapshotFormatError(f"{context} must be an ISO date (YYYY-MM-DD)")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise SnapshotFormatError(f"{context} must be an ISO date (YYYY-MM-DD)") from error
    if parsed.isoformat() != value:
        raise SnapshotFormatError(f"{context} must be an ISO date (YYYY-MM-DD)")
    return parsed


def _source_ranks(value: object, *, context: str) -> dict[str, int]:
    mapping = _object(value, context=f"{context}.source_ranks")
    ranks: dict[str, int] = {}
    for source, rank in mapping.items():
        if not source or not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise SnapshotFormatError(
                f"{context}.source_ranks must map non-empty source names to positive integer ranks"
            )
        ranks[source] = rank
    return ranks


def _optional_uri(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context=context)


def _optional_popularity(value: object, *, context: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100:
        raise SnapshotFormatError(f"{context} must be an integer from 0 to 100 or null")
    return value


def _candidate(value: object, *, index: int, week: date) -> Candidate:
    context = f"candidate[{index}]"
    payload = _object(value, context=context)
    fields = set(payload)
    missing = _CANDIDATE_REQUIRED_FIELDS - fields
    unexpected = fields - _CANDIDATE_REQUIRED_FIELDS - _CANDIDATE_OPTIONAL_FIELDS
    if missing:
        raise SnapshotFormatError(f"{context} has missing fields: {', '.join(sorted(missing))}")
    if unexpected:
        raise SnapshotFormatError(
            f"{context} has unexpected fields: {', '.join(sorted(unexpected))}"
        )

    first_seen = _iso_date(payload["first_seen"], context=f"{context}.first_seen")
    if first_seen > week:
        raise SnapshotFormatError(f"{context}.first_seen cannot be after snapshot week")
    return Candidate(
        key=_string(payload["key"], context=f"{context}.key"),
        title=_string(payload["title"], context=f"{context}.title"),
        artist=_string(payload["artist"], context=f"{context}.artist"),
        source_ranks=_source_ranks(payload["source_ranks"], context=context),
        first_seen=first_seen,
        spotify_uri=_optional_uri(payload.get("spotify_uri"), context=f"{context}.spotify_uri"),
        popularity=_optional_popularity(payload.get("popularity"), context=f"{context}.popularity"),
    )


def _relevant_keys(value: object) -> frozenset[str]:
    if not isinstance(value, list):
        raise SnapshotFormatError("relevant_next_week must be an array of non-empty strings")
    keys = [_string(key, context="relevant_next_week key") for key in value]
    if len(set(keys)) != len(keys):
        raise SnapshotFormatError("duplicate relevant_next_week key")
    return frozenset(keys)


def _parse_snapshot(value: object) -> WeeklySnapshot:
    payload = _object(value, context="snapshot")
    fields = set(payload)
    missing = _SNAPSHOT_FIELDS - fields
    unexpected = fields - _SNAPSHOT_FIELDS
    if missing:
        raise SnapshotFormatError(f"snapshot has missing fields: {', '.join(sorted(missing))}")
    if unexpected:
        raise SnapshotFormatError(
            f"snapshot has unexpected fields: {', '.join(sorted(unexpected))}"
        )

    week = _iso_date(payload["week"], context="week")
    candidate_values = payload["candidates"]
    if not isinstance(candidate_values, list):
        raise SnapshotFormatError("candidates must be an array")
    candidates = tuple(
        _candidate(value, index=index, week=week) for index, value in enumerate(candidate_values)
    )
    keys = [candidate.key for candidate in candidates]
    if len(set(keys)) != len(keys):
        raise SnapshotFormatError(f"duplicate candidate key in week {week.isoformat()}")
    return WeeklySnapshot(
        week=week,
        candidates=candidates,
        relevant_next_week=_relevant_keys(payload["relevant_next_week"]),
    )


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as error:
        raise SnapshotFormatError(f"{path}: invalid JSON ({error.msg})") from error
    except UnicodeDecodeError as error:
        raise SnapshotFormatError(f"{path}: file must be UTF-8 JSON") from error


@dataclass(frozen=True)
class FileSnapshotSource:
    """Load one JSON snapshot or a filename-ordered directory of snapshots."""

    path: Path

    def __init__(self, path: str | Path) -> None:
        object.__setattr__(self, "path", Path(path))

    def load(self) -> tuple[WeeklySnapshot, ...]:
        if not self.path.exists():
            raise SnapshotFormatError(f"input path does not exist: {self.path}")
        if self.path.is_file():
            paths = [self.path]
        elif self.path.is_dir():
            paths = sorted(self.path.glob("*.json"))
            if not paths:
                raise SnapshotFormatError(
                    f"input directory contains no JSON snapshots: {self.path}"
                )
        else:
            raise SnapshotFormatError(f"input path is not a file or directory: {self.path}")

        snapshots = tuple(_parse_snapshot(_load_json(path)) for path in paths)
        weeks = [snapshot.week for snapshot in snapshots]
        if len(set(weeks)) != len(weeks):
            raise SnapshotFormatError("duplicate week dates are not allowed")
        if any(current >= following for current, following in pairwise(weeks)):
            raise SnapshotFormatError("directory snapshot weeks must be strictly increasing")
        return snapshots
