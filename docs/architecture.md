# Architecture

## Design goal

The package separates data-source adapters from ranking and evaluation. A local JSON snapshot is enough to exercise the complete scoring, selection, and temporal-evaluation path without credentials or network access.

## Components

### Domain contracts

`domain.py` defines frozen `Candidate`, `ScoredCandidate`, `WeeklySnapshot`, and `RankingResult` dataclasses. Nested mappings are copied into read-only proxies, preventing callers or ranking functions from silently changing input evidence.

### Source boundary

`SnapshotSource` is the ingestion protocol. `FileSnapshotSource` is the included implementation and strictly validates local UTF-8 JSON. It rejects unknown/missing fields, duplicate candidate keys or weeks, invalid dates/ranks/popularity, and non-increasing directory snapshots.

Additional data sources can implement the same protocol without changing the scoring, selection, or evaluation core.

### Scoring

The scorer converts each available source rank into a bounded log-linear value. It computes a normalized weighted blend over present configured sources, so missing sources are absence, not sentinel rank values. The final score is:

```text
source_blend
× cross_source_confirmation
× week_over_week_momentum
× breakout_freshness
× age_decay
× popularity_factor
```

Every component is returned with the score for inspection. Momentum compares `raw_score` at week `t` with the same `raw_score` stage from the latest prior observation; it never compares raw and post-factor values.

The source weights and multiplicative factors are transparent diagnostic choices. They are not learned parameters and do not constitute evidence of production recommendation quality.

### Selection

`deterministic_top_k` sorts by descending final score and then candidate key. `gumbel_top_k` uses a local seeded random-number generator and the priority:

```text
log(max(final_score, 1e-12)) / temperature + Gumbel(0, 1)
```

Both selectors return new lists and leave the source sequence and immutable candidates unchanged. Gumbel temperature must be positive and finite.

### Temporal replay

Replay sorts snapshots chronologically and rejects duplicate weeks. For each week:

1. score current candidates using only current features and previously observed raw-score state;
2. select the ranking;
3. read `relevant_next_week` only for metric calculation;
4. update carried raw-score state from the current snapshot.

Future candidate features and labels cannot affect the earlier ranking. Tests mutate later snapshots with extreme evidence and verify byte-for-byte stability of earlier results.

### CLI and artifacts

The `music-trends rank` command ranks one snapshot. `music-trends evaluate` compares deterministic selection with repeated seeded Gumbel trials and emits sorted, indented JSON. CI regenerates `examples/evaluation-summary.json` and diffs it byte-for-byte.

## Important invariants

- candidate input objects are immutable and never mutated;
- candidate and ranked keys are unique within their public boundary;
- rank inputs are positive integers;
- all stored and emitted scores are finite and non-negative;
- stochastic output is deterministic for the same explicit seed;
- evaluation is chronological and next-week labels never enter scoring;
- the offline demo imports no network client and requires no credential.
