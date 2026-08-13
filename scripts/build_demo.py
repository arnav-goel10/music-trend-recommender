#!/usr/bin/env python3
"""Build demo/index.html: a static, self-contained viewer over the synthetic fixture.

Scores every candidate in data/sample/weeks with the real scoring code
(score_candidate, deterministic ordering, previous_raw_score=None, exactly
like `music-trends rank`), inlines the result plus examples/evaluation-summary.json
into a template, and writes a single HTML file with no external assets.

Usage: PYTHONPATH=src python3 scripts/build_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

from music_trend_recommender.ranking.scoring import score_candidate
from music_trend_recommender.ranking.selection import deterministic_top_k
from music_trend_recommender.sources.file_snapshot import FileSnapshotSource

ROOT = Path(__file__).resolve().parents[1]


def build_payload() -> dict:
    snapshots = FileSnapshotSource(ROOT / "data" / "sample" / "weeks").load()
    weeks = []
    for snapshot in snapshots:
        scored = [
            score_candidate(candidate, as_of=snapshot.week, previous_raw_score=None)
            for candidate in snapshot.candidates
        ]
        ranked = deterministic_top_k(scored, len(scored))
        weeks.append(
            {
                "week": snapshot.week.isoformat(),
                "relevant_next_week": sorted(snapshot.relevant_next_week),
                "ranked": [
                    {
                        "key": item.candidate.key,
                        "title": item.candidate.title,
                        "artist": item.candidate.artist,
                        "sources": dict(item.candidate.source_ranks),
                        "first_seen": item.candidate.first_seen.isoformat(),
                        "final_score": round(item.final_score, 4),
                        "components": {k: round(v, 4) for k, v in item.components.items()},
                    }
                    for item in ranked
                ],
            }
        )
    evaluation = json.loads((ROOT / "examples" / "evaluation-summary.json").read_text())
    return {"weeks": weeks, "evaluation": evaluation}


def main() -> None:
    payload = build_payload()
    template = (ROOT / "scripts" / "demo_template.html").read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(payload))
    out = ROOT / "demo" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
