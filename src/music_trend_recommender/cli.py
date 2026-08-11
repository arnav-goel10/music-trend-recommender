"""Command-line demo for local ranking and leakage-safe temporal evaluation."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from music_trend_recommender.domain import RankingResult, WeeklySnapshot
from music_trend_recommender.evaluation.replay import ReplayRow, replay
from music_trend_recommender.ranking.scoring import score_candidate
from music_trend_recommender.ranking.selection import deterministic_top_k, gumbel_top_k
from music_trend_recommender.sources.file_snapshot import FileSnapshotSource

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

_GUMBEL_TEMPERATURE = 1.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music-trends",
        description="Rank and replay credential-free weekly music snapshots.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    rank_parser = commands.add_parser("rank", help="rank candidates in one snapshot")
    rank_parser.add_argument("--input", type=Path, required=True, help="snapshot JSON file")
    rank_parser.add_argument("--k", type=int, default=5, help="number of candidates to return")
    rank_parser.add_argument(
        "--method",
        choices=("deterministic", "gumbel"),
        default="deterministic",
        help="selection method",
    )
    rank_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="local RNG seed used by Gumbel selection (ignored for deterministic ranking)",
    )

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="compare deterministic and repeated-seed Gumbel rankings",
    )
    evaluate_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="snapshot JSON file or filename-ordered directory",
    )
    evaluate_parser.add_argument("--k", type=int, default=5, help="evaluation cutoff")
    evaluate_parser.add_argument(
        "--seeds",
        type=int,
        default=20,
        metavar="N",
        help="number of Gumbel trials; expands deterministically to seeds 0 through N-1",
    )
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        help="write JSON to this path instead of standard output",
    )
    return parser


def _rank(snapshot: WeeklySnapshot, *, k: int, method: str, seed: int) -> RankingResult:
    scored = [
        score_candidate(candidate, as_of=snapshot.week, previous_raw_score=None)
        for candidate in snapshot.candidates
    ]
    if method == "deterministic":
        selected = deterministic_top_k(scored, k)
        result_seed: int | None = None
    else:
        selected = gumbel_top_k(scored, k, temperature=_GUMBEL_TEMPERATURE, seed=seed)
        result_seed = seed
    return RankingResult(
        method=method,
        week=snapshot.week,
        ranked_keys=tuple(item.candidate.key for item in selected),
        seed=result_seed,
    )


def _rank_payload(result: RankingResult) -> dict[str, JsonValue]:
    return {
        "method": result.method,
        "week": result.week.isoformat(),
        "ranked_keys": list(result.ranked_keys),
        "seed": result.seed,
    }


def _evaluation_row(row: ReplayRow, *, seeds: list[int]) -> dict[str, JsonValue]:
    metric_names = (
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
        "source_coverage",
    )
    output: dict[str, JsonValue] = {
        "method": row["method"],
        "week": row["week"],
        "k": row["k"],
        **{name: row[name] for name in metric_names},
    }
    if row["method"] == "gumbel":
        output["seed_count"] = len(seeds)
        seed_values: list[JsonValue] = list(seeds)
        output["seeds"] = seed_values
        for name in metric_names:
            output[f"{name}_std"] = row[f"{name}_std"]
    return output


def _evaluate(
    snapshots: tuple[WeeklySnapshot, ...],
    *,
    k: int,
    seed_count: int,
) -> list[JsonValue]:
    seeds = list(range(seed_count))
    deterministic_rows = replay(snapshots, k=k, method="deterministic", seeds=[])
    gumbel_rows = replay(snapshots, k=k, method="gumbel", seeds=seeds)
    by_method_and_week = {
        (str(row["method"]), str(row["week"])): row for row in (*deterministic_rows, *gumbel_rows)
    }
    output: list[JsonValue] = []
    for snapshot in snapshots:
        week = snapshot.week.isoformat()
        for method in ("deterministic", "gumbel"):
            output.append(_evaluation_row(by_method_and_week[(method, week)], seeds=seeds))
    return output


def _json_text(payload: JsonValue) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _emit(payload: JsonValue, *, output: Path | None) -> None:
    text = _json_text(payload)
    if output is None:
        sys.stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and translate input problems into concise exit-code-2 errors."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.k <= 0:
            raise ValueError("--k must be positive")
        snapshots = FileSnapshotSource(arguments.input).load()
        if arguments.command == "rank":
            if len(snapshots) != 1:
                raise ValueError("rank input must contain exactly one snapshot")
            result = _rank(
                snapshots[0],
                k=arguments.k,
                method=arguments.method,
                seed=arguments.seed,
            )
            _emit(_rank_payload(result), output=None)
        else:
            if arguments.seeds <= 0:
                raise ValueError("--seeds must be positive")
            _emit(
                _evaluate(snapshots, k=arguments.k, seed_count=arguments.seeds),
                output=arguments.output,
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
