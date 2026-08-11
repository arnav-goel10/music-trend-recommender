"""Integrity tests for redistributable synthetic weekly snapshots."""

import json
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIRECTORY = Path("data/sample/weeks")
EXPECTED_WEEKS = ("2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26")


def load_fixtures() -> list[dict[str, Any]]:
    paths = [FIXTURE_DIRECTORY / f"{week}.json" for week in EXPECTED_WEEKS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        pytest.fail(f"synthetic weekly fixtures are not implemented: {missing}", pytrace=False)
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def test_each_fixture_has_a_valid_synthetic_candidate_catalogue() -> None:
    fixtures = load_fixtures()

    for expected_week, fixture in zip(EXPECTED_WEEKS, fixtures, strict=True):
        assert set(fixture) == {"week", "candidates", "relevant_next_week"}
        assert fixture["week"] == expected_week
        assert len(fixture["candidates"]) >= 8
        keys = [candidate["key"] for candidate in fixture["candidates"]]
        assert len(keys) == len(set(keys))
        assert all(key.startswith("synthetic-artist-") for key in keys)
        assert all("::synthetic-track-" in key for key in keys)
        for candidate in fixture["candidates"]:
            assert set(candidate) == {
                "key",
                "title",
                "artist",
                "source_ranks",
                "first_seen",
            }
            assert candidate["title"].startswith("Synthetic Track ")
            assert candidate["artist"].startswith("Synthetic Artist ")
            assert date.fromisoformat(candidate["first_seen"]) <= date.fromisoformat(expected_week)
            assert len(candidate["source_ranks"]) >= 1
            assert all(
                isinstance(rank, int) and not isinstance(rank, bool) and rank > 0
                for rank in candidate["source_ranks"].values()
            )


def test_fixtures_contain_multiple_sources_entries_and_exits() -> None:
    fixtures = load_fixtures()
    catalogues = [{candidate["key"] for candidate in fixture["candidates"]} for fixture in fixtures]

    for fixture in fixtures:
        sources = {
            source for candidate in fixture["candidates"] for source in candidate["source_ranks"]
        }
        assert len(sources) >= 4
        assert any(len(candidate["source_ranks"]) >= 2 for candidate in fixture["candidates"])
    for current, following in pairwise(catalogues):
        assert following - current
        assert current - following


def test_relevance_labels_follow_the_documented_next_week_persistence_rule() -> None:
    fixtures = load_fixtures()
    catalogues = [{candidate["key"] for candidate in fixture["candidates"]} for fixture in fixtures]

    for index, fixture in enumerate(fixtures[:-1]):
        expected_relevant = catalogues[index] & catalogues[index + 1]
        assert set(fixture["relevant_next_week"]) == expected_relevant
    assert fixtures[-1]["relevant_next_week"] == []
