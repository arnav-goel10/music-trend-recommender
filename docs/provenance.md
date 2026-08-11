# Provenance and publication boundary

This repository is a newly initialized, clean-room public rebuild of a private Spotify trend-ranking prototype created by Arnav Goel.

## What was retained

The rebuild retains high-level engineering lessons: combining multiple ranked sources, decomposing scores for inspection, balancing exploitation with seeded exploration, carrying week-over-week state, and evaluating rankings chronologically.

All public implementation code, tests, synthetic fixtures, documentation, and repository history were created for this standalone repository. The GitHub repository is published directly under `arnav-goel10`; it is not a fork or imported repository.

## What was excluded

- private repository files and commit history;
- real or raw chart exports and local state;
- scraped data without clear redistribution rights;
- anti-bot bypasses, request-signing logic, or hard-coded signing material;
- API credentials, tokens, cookies, or `.env` files;
- credentialed Spotify playlist writes or account mutations;
- claims based on private or real-user metrics.

The checked-in dataset is invented and redistributable. Platform-like source names describe schema categories only; they do not imply that the sample contains records from those services or that this repository is affiliated with them.

## Ownership and license

Arnav Goel owns the clean-room materials published in this repository and licenses them under MIT. No license is asserted over excluded private or third-party materials.
