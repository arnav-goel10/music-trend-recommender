# Evaluation protocol

## Question answered

The offline replay asks: given only the candidate evidence available during week `t`, how well does the resulting top-`k` ranking recover candidates that persist into the immediately following synthetic snapshot?

This is a mechanics and diagnostics exercise. Persistence is a convenient synthetic label, not a substitute for listens, saves, completions, skips, satisfaction, or an online experiment.

## Temporal label relationship

For each non-terminal snapshot, `relevant_next_week` is exactly the intersection of:

- candidate keys visible at week `t`; and
- candidate keys visible at week `t+1`.

Replay computes week `t` scores and the ranking before reading this label set. Only current features and raw-score state observed at or before `t` may enter scoring. The terminal fixture has no observed `t+1`, so its relevance set is empty and its precision, recall, and NDCG are intentionally zero.

## Metrics

All relevance metrics use binary labels.

- **Precision@k:** relevant retrieved keys in the first `k`, divided by `k`.
- **Recall@k:** relevant retrieved keys in the first `k`, divided by the number of relevant keys. It is zero when no relevance labels exist.
- **NDCG@k:** discounted cumulative gain of relevant keys, normalized by the ideal binary-relevance ordering at `k`.
- **Source coverage@k:** distinct source types represented by the first `k` candidates, divided by all source types present in that week's candidate set.

Metric functions reject duplicate ranked keys rather than allowing a duplicate to inflate recall or NDCG.

## Compared selectors

- **Deterministic top-k** runs once per snapshot. Seed arguments do not apply.
- **Gumbel-Top-k** runs over the explicit seeds `0..N-1`. The artifact uses 20 trials. The report contains the population mean and population standard deviation of each metric across those trials.

Seeds are sorted inside replay, so the aggregate is invariant to their input order. A seed controls a local RNG; it does not depend on Python's module-global random state.

## Reproduction

```bash
music-trends evaluate \
  --input data/sample/weeks \
  --k 5 \
  --seeds 20 \
  --output /tmp/summary.json
diff -u examples/evaluation-summary.json /tmp/summary.json
```

The checked-in artifact contains four synthetic weeks and both selectors. CI fails if regeneration differs.

## Interpreting the sample

The fixture makes behavior visible: deterministic selection tends to concentrate on the highest hand-scored candidates; seeded selection explores alternative candidates and may change both relevance metrics and source coverage. The reported values are only synthetic diagnostics.

Do not describe these results as:

- an improvement over a production baseline;
- real-user recommendation quality;
- performance on Spotify, TikTok, YouTube, Shazam, or Apple Music data;
- an A/B test, deployment, or business impact result.

A real evaluation would require licensed historical candidates, engagement-grounded labels, carefully selected baselines, user/item cold-start analysis, slice metrics, uncertainty estimates, and ultimately an online experiment with product guardrails.
