# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-13

First tagged release. Not published to PyPI; install from this repository (`pip install .`) or use the Docker image built from the `Dockerfile`.

### Added
- Log-to-eval-dataset pipeline and `evalforge` CLI: JSONL, OpenTelemetry GenAI span and SQLite adapters; PII/secret redaction; embedding dedup and clustering; LLM-judge labelling; anomaly flags; nightly selection of novel and failed cases; dataset export and eval runs.
- Non-root Docker image, smoke-tested in CI; Python 3.11+.
- Docs: README onboarding guide, ARCHITECTURE, DATASET_FORMAT.
- Test suite with per-module mutation grades (testteeth) and a non-blocking weekly mutation workflow; measured before/after table in the README.
