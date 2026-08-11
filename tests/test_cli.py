"""End-to-end contracts for credential-free snapshot loading and CLI output."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SAMPLE_DIRECTORY = Path("data/sample/weeks")
SAMPLE_WEEK = SAMPLE_DIRECTORY / "2026-01-26.json"
EXPECTED_SEEDS = list(range(20))


def load_file_source() -> ModuleType:
    """Load the adapter after collection so absence becomes an assertion RED."""
    try:
        return importlib.import_module("music_trend_recommender.sources.file_snapshot")
    except ModuleNotFoundError:
        pytest.fail("file snapshot adapter is not implemented", pytrace=False)


def run_cli(*arguments: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "music_trend_recommender.cli", *arguments]
    return subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        env=env,
    )


def valid_snapshot(*, week: str = "2026-02-02") -> dict[str, Any]:
    return {
        "week": week,
        "candidates": [
            {
                "key": "artist-a::track-a",
                "title": "Track A",
                "artist": "Artist A",
                "source_ranks": {"youtube": 1, "shazam": 4},
                "first_seen": "2026-01-26",
            },
            {
                "key": "artist-b::track-b",
                "title": "Track B",
                "artist": "Artist B",
                "source_ranks": {"apple_music": 8},
                "first_seen": "2026-02-02",
            },
        ],
        "relevant_next_week": ["artist-a::track-a"],
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_file_source_loads_one_file_and_a_chronological_directory(tmp_path: Path) -> None:
    source_module = load_file_source()
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_json(first_path, valid_snapshot(week="2026-02-02"))
    second = valid_snapshot(week="2026-02-09")
    for candidate in second["candidates"]:
        candidate["first_seen"] = "2026-02-02"
    write_json(second_path, second)

    single = source_module.FileSnapshotSource(first_path).load()
    directory = source_module.FileSnapshotSource(tmp_path).load()

    assert len(single) == 1
    assert single[0].week.isoformat() == "2026-02-02"
    assert [candidate.key for candidate in single[0].candidates] == [
        "artist-a::track-a",
        "artist-b::track-b",
    ]
    assert single[0].relevant_next_week == frozenset({"artist-a::track-a"})
    assert [snapshot.week.isoformat() for snapshot in directory] == [
        "2026-02-02",
        "2026-02-09",
    ]


def test_rank_command_emits_the_stable_public_schema() -> None:
    result = run_cli(
        "rank",
        "--input",
        str(SAMPLE_WEEK),
        "--k",
        "5",
        "--method",
        "gumbel",
        "--seed",
        "42",
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload == {
        "method": "gumbel",
        "ranked_keys": [
            "synthetic-artist-m::synthetic-track-mosaic",
            "synthetic-artist-k::synthetic-track-kite",
            "synthetic-artist-a::synthetic-track-alpha",
            "synthetic-artist-l::synthetic-track-lumen",
            "synthetic-artist-i::synthetic-track-ion",
        ],
        "seed": 42,
        "week": "2026-01-26",
    }
    assert result.stdout == json.dumps(payload, indent=2, sort_keys=True) + "\n"


def test_deterministic_rank_has_a_null_seed() -> None:
    result = run_cli(
        "rank",
        "--input",
        str(SAMPLE_WEEK),
        "--k",
        "3",
        "--method",
        "deterministic",
        "--seed",
        "999",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["method"] == "deterministic"
    assert payload["seed"] is None
    assert len(payload["ranked_keys"]) == 3


def test_evaluate_compares_methods_and_exposes_the_exact_repeated_seed_set(
    tmp_path: Path,
) -> None:
    output = tmp_path / "summary.json"
    result = run_cli(
        "evaluate",
        "--input",
        str(SAMPLE_DIRECTORY),
        "--k",
        "5",
        "--seeds",
        "20",
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert len(rows) == 8
    assert [(row["week"], row["method"]) for row in rows] == [
        (week, method)
        for week in ("2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26")
        for method in ("deterministic", "gumbel")
    ]
    for row in rows:
        assert {
            "method",
            "week",
            "k",
            "precision_at_k",
            "recall_at_k",
            "ndcg_at_k",
            "source_coverage",
        } <= set(row)
        assert row["k"] == 5
        if row["method"] == "gumbel":
            assert row["seeds"] == EXPECTED_SEEDS
            assert row["seed_count"] == 20
            assert {
                "precision_at_k_std",
                "recall_at_k_std",
                "ndcg_at_k_std",
                "source_coverage_std",
            } <= set(row)
        else:
            assert "seeds" not in row
            assert "seed_count" not in row
            assert not any(key.endswith("_std") for key in row)


def test_evaluate_is_byte_stable_across_repeated_generation(tmp_path: Path) -> None:
    first = tmp_path / "nested" / "first.json"
    second = tmp_path / "second.json"
    arguments = (
        "evaluate",
        "--input",
        str(SAMPLE_DIRECTORY),
        "--k",
        "5",
        "--seeds",
        "20",
    )

    first_result = run_cli(*arguments, "--output", str(first))
    second_result = run_cli(*arguments, "--output", str(second))

    assert first_result.returncode == second_result.returncode == 0
    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8").endswith("\n")


def test_checked_in_summary_is_generated_byte_for_byte_by_the_cli(tmp_path: Path) -> None:
    regenerated = tmp_path / "evaluation-summary.json"

    result = run_cli(
        "evaluate",
        "--input",
        str(SAMPLE_DIRECTORY),
        "--k",
        "5",
        "--seeds",
        "20",
        "--output",
        str(regenerated),
    )

    assert result.returncode == 0, result.stderr
    assert regenerated.read_bytes() == Path("examples/evaluation-summary.json").read_bytes()


def test_sample_commands_do_not_import_or_initialize_network_clients(tmp_path: Path) -> None:
    guard = tmp_path / "network_guard"
    guard.mkdir()
    for module_name in ("requests", "httpx", "spotipy"):
        (guard / f"{module_name}.py").write_text(
            f'raise RuntimeError("{module_name} must not be imported")\n',
            encoding="utf-8",
        )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(guard), *([existing_pythonpath] if existing_pythonpath else [])]
    )

    rank = run_cli("rank", "--input", str(SAMPLE_WEEK), "--k", "2", env=env)
    evaluate = run_cli(
        "evaluate",
        "--input",
        str(SAMPLE_DIRECTORY),
        "--k",
        "2",
        "--seeds",
        "2",
        env=env,
    )

    assert rank.returncode == evaluate.returncode == 0
    assert "must not be imported" not in rank.stderr + evaluate.stderr


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(extra="unexpected"), "unexpected fields"),
        (lambda payload: payload.pop("candidates"), "missing fields"),
        (lambda payload: payload.update(week="2026-99-99"), "week must be an ISO date"),
        (
            lambda payload: payload["candidates"][0].update(first_seen="not-a-date"),
            "first_seen must be an ISO date",
        ),
        (
            lambda payload: payload["candidates"].append(payload["candidates"][0].copy()),
            "duplicate candidate key",
        ),
        (
            lambda payload: payload.update(relevant_next_week=["artist-a::track-a"] * 2),
            "duplicate relevant_next_week key",
        ),
        (
            lambda payload: payload["candidates"][0].update(source_ranks=[]),
            "source_ranks must be an object",
        ),
    ],
)
def test_file_source_rejects_malformed_snapshot_shapes(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    source_module = load_file_source()
    payload = valid_snapshot()
    mutation(payload)
    path = tmp_path / "snapshot.json"
    write_json(path, payload)

    with pytest.raises(ValueError, match=message):
        source_module.FileSnapshotSource(path).load()


def test_file_source_rejects_invalid_json_with_the_path(tmp_path: Path) -> None:
    source_module = load_file_source()
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"broken\.json: invalid JSON"):
        source_module.FileSnapshotSource(path).load()


def test_file_source_rejects_duplicate_and_non_increasing_directory_weeks(
    tmp_path: Path,
) -> None:
    source_module = load_file_source()
    duplicate_directory = tmp_path / "duplicate"
    duplicate_directory.mkdir()
    write_json(duplicate_directory / "a.json", valid_snapshot(week="2026-02-02"))
    write_json(duplicate_directory / "b.json", valid_snapshot(week="2026-02-02"))

    with pytest.raises(ValueError, match="duplicate week"):
        source_module.FileSnapshotSource(duplicate_directory).load()

    reversed_directory = tmp_path / "reversed"
    reversed_directory.mkdir()
    write_json(reversed_directory / "a.json", valid_snapshot(week="2026-02-09"))
    write_json(reversed_directory / "b.json", valid_snapshot(week="2026-02-02"))

    with pytest.raises(ValueError, match="strictly increasing"):
        source_module.FileSnapshotSource(reversed_directory).load()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("rank", "--input", "missing.json"), "input path does not exist"),
        (
            ("evaluate", "--input", "data/sample/weeks", "--seeds", "0"),
            "--seeds must be positive",
        ),
        (("rank", "--input", "data/sample/weeks", "--k", "2"), "exactly one snapshot"),
    ],
)
def test_cli_reports_user_errors_without_a_traceback(
    arguments: tuple[str, ...],
    message: str,
) -> None:
    result = run_cli(*arguments)

    assert result.returncode == 2
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("arguments", [(), ("--help",), ("rank", "--help"), ("evaluate", "--help")])
def test_cli_help_is_available(arguments: tuple[str, ...]) -> None:
    result = run_cli(*arguments)

    if arguments:
        assert result.returncode == 0
        assert "usage:" in result.stdout
    else:
        assert result.returncode == 2
        assert "usage:" in result.stderr


def test_evaluate_help_documents_the_seed_expansion() -> None:
    result = run_cli("evaluate", "--help")

    assert result.returncode == 0
    assert "0 through N-1" in result.stdout
