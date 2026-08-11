"""Credential-free and optional external snapshot adapters."""

from music_trend_recommender.sources.base import SnapshotSource
from music_trend_recommender.sources.file_snapshot import FileSnapshotSource

__all__ = ["FileSnapshotSource", "SnapshotSource"]
