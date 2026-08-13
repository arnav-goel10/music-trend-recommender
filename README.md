# Music Trend Recommender

[![CI](https://github.com/arnav-goel10/music-trend-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/arnav-goel10/music-trend-recommender/actions/workflows/ci.yml)

A multi-source music discovery and ranking pipeline with temporal replay evaluation and seeded Gumbel-Top-k exploration.

This repository demonstrates candidate normalization, inspectable score decomposition, stochastic exploration, state carried across weekly snapshots, and offline ranking diagnostics. The checked-in demo uses synthetic data and requires no platform credentials.

> **Evidence boundary:** every metric below is a deterministic diagnostic from the repository's synthetic four-week fixture. It is not a real-user result and does not establish recommendation-quality improvement.

Developed and maintained by Arnav Goel. The included evaluation dataset is fully synthetic and reproducible.

## What it demonstrates

- **Multi-source retrieval signals:** normalizes ranks from six configurable chart/source types and reweights over the evidence actually present for each candidate.
- **Explainable ranking:** exposes source blend, cross-source confirmation, week-over-week momentum, freshness, age decay, and popularity factors for every score.
- **Exploration without mutation:** implements deterministic top-k and seeded Gumbel-Top-k using `log(weight) / temperature + Gumbel(0, 1)` with stable tie-breaking.
- **Leakage-safe replay:** ranks week `t` using only features and carried state available at `t`, then evaluates against next-week persistence labels.
- **Reproducible evaluation:** reports Precision@k, Recall@k, NDCG@k, source coverage, and repeated-seed population standard deviations.
- **Offline-first interfaces:** strict JSON adapters and CLI commands run without Spotify credentials, paid APIs, or network access.
- **Live replay demo:** a static viewer over the synthetic fixture, rebuilt by `scripts/build_demo.py`, at [arnav-goel10.github.io/music-trend-recommender](https://arnav-goel10.github.io/music-trend-recommender/).

## Architecture

```text
weekly JSON snapshots
        │
        ▼
strict file adapter ──► immutable candidates
        │
        ▼
decomposed scoring ◄── prior raw-score state
        │
        ├──► deterministic top-k
        └──► seeded Gumbel-Top-k
                  │
                  ▼
        temporal t → t+1 evaluation
                  │
                  ▼
      reproducible JSON diagnostics
```

The [architecture note](docs/architecture.md) explains the boundaries and invariants. The [evaluation note](docs/evaluation.md) documents label timing, metrics, and claim limits.

## Quick start

Requires Python 3.12 or 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

music-trends rank \
  --input data/sample/weeks/2026-01-26.json \
  --k 5 \
  --method gumbel \
  --seed 42

music-trends evaluate \
  --input data/sample/weeks \
  --k 5 \
  --seeds 20 \
  --output /tmp/evaluation-summary.json

diff -u examples/evaluation-summary.json /tmp/evaluation-summary.json
```

The final `diff` should be empty. The checked-in [evaluation artifact](examples/evaluation-summary.json) is regenerated in CI to detect behavior drift.

## Synthetic diagnostic snapshot

For the first synthetic week at `k=5`, the deterministic selector reports Precision@5 `1.00`, Recall@5 `0.833`, NDCG@5 `1.00`, and source coverage `0.833`. Across 20 seeded Gumbel trials, mean source coverage is `0.925` while mean NDCG@5 is `0.844`.

Those values make the exploration/coverage trade-off observable on a controlled fixture. They are **not** a claim that Gumbel selection improves real recommendation quality. The terminal week has no observed next week, so its relevance metrics are intentionally zero.

## Input contract

Each snapshot is a UTF-8 JSON object with:

- `week`: canonical ISO date;
- `candidates`: unique tracks with `key`, title, artist, `first_seen`, and positive integer `source_ranks`;
- `relevant_next_week`: unique candidate keys used only after ranking for evaluation.

See [DATA_CARD.md](DATA_CARD.md) for fixture construction and limitations. The adapter rejects malformed fields, duplicate keys/weeks, non-canonical dates, and invalid rank or popularity values with concise errors.

## Quality gate

```bash
ruff check .
ruff format --check .
mypy src
pytest -q
music-trends evaluate --input data/sample/weeks --k 5 --seeds 20 --output /tmp/summary.json
diff -u examples/evaluation-summary.json /tmp/summary.json
```

The test suite covers immutability, scoring boundaries, exact metric examples, seeded selection, non-mutation, duplicate rejection, future-feature leakage, repeated-seed aggregation, strict adapters, deterministic artifacts, and credential-free CLI execution.

## License and security

This project is released under the [MIT License](LICENSE). Please report vulnerabilities according to [SECURITY.md](SECURITY.md).
