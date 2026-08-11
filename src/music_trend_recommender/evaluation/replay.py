"""Chronological, leakage-safe offline replay of weekly music rankings.

Candidate features and carried raw-score state available at week ``t`` produce
that week's ranking. ``relevant_next_week`` is read only after selection to
score the ranking; it never enters scoring or state.
"""

from collections.abc import Sequence
from statistics import fmean, pstdev

from music_trend_recommender.domain import ScoredCandidate, WeeklySnapshot
from music_trend_recommender.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    source_coverage_at_k,
)
from music_trend_recommender.ranking.scoring import score_candidate
from music_trend_recommender.ranking.selection import deterministic_top_k, gumbel_top_k

MetricValue = float | str | int
ReplayRow = dict[str, MetricValue]

_GUMBEL_TEMPERATURE = 1.0


def _mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    mean = fmean(values)
    return mean, pstdev(values) if len(values) > 1 else 0.0


def _rankings_for_method(
    scored: Sequence[ScoredCandidate],
    *,
    k: int,
    method: str,
    seeds: Sequence[int],
) -> tuple[list[list[ScoredCandidate]], str]:
    if method == "deterministic":
        ranking = deterministic_top_k(scored, k)
        signature = "|".join(item.candidate.key for item in ranking)
        return [ranking], signature

    ordered_seeds = sorted(seeds)
    if not ordered_seeds:
        raise ValueError("gumbel replay requires at least one seed")
    rankings = [
        gumbel_top_k(
            scored,
            k,
            temperature=_GUMBEL_TEMPERATURE,
            seed=seed,
        )
        for seed in ordered_seeds
    ]
    signature = ";".join(
        f"{seed}:{'|'.join(item.candidate.key for item in ranking)}"
        for seed, ranking in zip(ordered_seeds, rankings, strict=True)
    )
    return rankings, signature


def _evaluate_rankings(
    rankings: Sequence[Sequence[ScoredCandidate]],
    snapshot: WeeklySnapshot,
    *,
    k: int,
) -> dict[str, float]:
    available_sources = {
        source for candidate in snapshot.candidates for source in candidate.source_ranks
    }
    metric_trials: dict[str, list[float]] = {
        "precision_at_k": [],
        "recall_at_k": [],
        "ndcg_at_k": [],
        "source_coverage": [],
    }
    for ranking in rankings:
        ranked_keys = [item.candidate.key for item in ranking]
        metric_trials["precision_at_k"].append(
            precision_at_k(ranked_keys, snapshot.relevant_next_week, k)
        )
        metric_trials["recall_at_k"].append(
            recall_at_k(ranked_keys, snapshot.relevant_next_week, k)
        )
        metric_trials["ndcg_at_k"].append(ndcg_at_k(ranked_keys, snapshot.relevant_next_week, k))
        metric_trials["source_coverage"].append(
            source_coverage_at_k(ranking, available_sources=available_sources, k=k)
        )

    aggregated: dict[str, float] = {}
    for name, values in metric_trials.items():
        mean, standard_deviation = _mean_and_std(values)
        aggregated[name] = mean
        aggregated[f"{name}_std"] = standard_deviation
    return aggregated


def replay(
    snapshots: Sequence[WeeklySnapshot],
    k: int,
    method: str,
    seeds: Sequence[int],
) -> list[ReplayRow]:
    """Evaluate snapshots chronologically without allowing future feature leakage.

    Input snapshots are sorted by date; duplicate dates are rejected. Deterministic
    replay runs exactly one seed-free trial and ignores ``seeds``. Gumbel replay
    requires at least one seed and aggregates trials in sorted seed order.
    """
    if method not in {"deterministic", "gumbel"}:
        raise ValueError("method must be 'deterministic' or 'gumbel'")
    if method == "gumbel" and not seeds:
        raise ValueError("gumbel replay requires at least one seed")

    chronological = sorted(snapshots, key=lambda snapshot: snapshot.week)
    weeks = [snapshot.week for snapshot in chronological]
    if len(set(weeks)) != len(weeks):
        raise ValueError("duplicate week dates are not allowed")
    for snapshot in chronological:
        candidate_keys = [candidate.key for candidate in snapshot.candidates]
        if len(set(candidate_keys)) != len(candidate_keys):
            raise ValueError(
                f"duplicate candidate keys are not allowed in week {snapshot.week.isoformat()}"
            )

    previous_raw_scores: dict[str, float] = {}
    rows: list[ReplayRow] = []
    for snapshot in chronological:
        scored = [
            score_candidate(
                candidate,
                as_of=snapshot.week,
                previous_raw_score=previous_raw_scores.get(candidate.key),
            )
            for candidate in snapshot.candidates
        ]
        rankings, ranking_signature = _rankings_for_method(
            scored,
            k=k,
            method=method,
            seeds=seeds,
        )
        metrics = _evaluate_rankings(rankings, snapshot, k=k)
        rows.append(
            {
                "week": snapshot.week.isoformat(),
                "method": method,
                "k": k,
                "seed_count": len(rankings),
                "ranked_keys": ranking_signature,
                **metrics,
            }
        )
        previous_raw_scores.update({item.candidate.key: item.raw_score for item in scored})

    return rows
