# Data card: synthetic weekly music snapshots

## Summary

`data/sample/weeks/` contains four invented weekly snapshots dated from 2026-01-05 through 2026-01-26. Each snapshot contains eight synthetic candidates, multiple source-rank types, candidate entry/exit behavior, and next-week persistence labels.

## Intended use

- exercise strict snapshot parsing;
- demonstrate normalized multi-source scoring;
- test deterministic and seeded ranking behavior;
- verify chronological replay and future-leakage resistance;
- reproduce the checked-in diagnostic evaluation artifact without credentials.

## Dataset creation

Every artist name, track name, key, rank, date, popularity value, and relevance label is synthetic and was created for this repository. No Spotify, TikTok, YouTube, Shazam, Apple Music, competitor-playlist, or other real chart export is included.

Source names are fictionalized schema categories that resemble possible signal families. Their presence does not imply data access, endorsement, affiliation, or collection from those services.

## Label construction

For each non-terminal week, `relevant_next_week` is the set of current candidate keys that also appear in the immediately following fixture. The terminal snapshot has an empty label set because the sample contains no later observation.

## Limitations

- four weeks and eight candidates per week are too small for model or product conclusions;
- persistence is not an engagement or satisfaction label;
- hand-authored source ranks do not reproduce real platform distributions or missingness;
- no user, geography, language, creator, catalogue, licensing, or cold-start slices exist;
- synthetic metric values cannot support a recommendation-quality or improvement claim.

## Redistribution and privacy

The sample contains no personal data, account identifiers, private API responses, or copyrighted audio. It is distributed with the repository under the MIT License.
