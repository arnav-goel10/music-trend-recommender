"""Source boundary for weekly ranking snapshots."""

from typing import Protocol

from music_trend_recommender.domain import WeeklySnapshot


class SnapshotSource(Protocol):
    """Load a finite sequence of snapshots without prescribing transport."""

    def load(self) -> tuple[WeeklySnapshot, ...]:
        """Return snapshots in strictly increasing chronological order."""
        ...
