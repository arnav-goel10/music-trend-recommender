"""Non-mutating deterministic and seeded exploratory ranking selection."""

import random
from collections.abc import Sequence
from math import isfinite, log

from music_trend_recommender.domain import ScoredCandidate

_MIN_POSITIVE_WEIGHT = 1e-12
_MIN_UNIFORM = 1e-12
_MAX_UNIFORM = 1.0 - _MIN_UNIFORM


def deterministic_top_k(
    items: Sequence[ScoredCandidate],
    k: int,
) -> list[ScoredCandidate]:
    """Return the highest-scoring items with stable candidate-key tie breaking."""
    ranked = sorted(items, key=lambda item: (-item.final_score, item.candidate.key))
    return ranked[: max(k, 0)]


def _gumbel(rng: random.Random) -> float:
    uniform = min(max(rng.random(), _MIN_UNIFORM), _MAX_UNIFORM)
    return -log(-log(uniform))


def gumbel_top_k(
    items: Sequence[ScoredCandidate],
    k: int,
    temperature: float,
    seed: int,
) -> list[ScoredCandidate]:
    """Sample a seeded Gumbel-Top-k ranking without mutating input candidates."""
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")

    rng = random.Random(seed)
    priorities = [
        (
            log(max(item.final_score, _MIN_POSITIVE_WEIGHT)) / temperature + _gumbel(rng),
            item,
        )
        for item in items
    ]
    priorities.sort(key=lambda pair: (-pair[0], pair[1].candidate.key))
    return [item for _, item in priorities[: max(k, 0)]]
