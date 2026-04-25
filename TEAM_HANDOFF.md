# Phoenix Work Team Handoff

## Purpose

This document captures the current project state and the exact next work package for teammates taking over from this point.

## Branching Instructions

- Start from the latest `phoenix-work` copy.
- Create a new branch for continuation work.
- Suggested branch name: `feature/packaging-quality-profiles`.
- Do not continue this work on the current branch history.

## Current State Summary

### Already completed

- Frontend/backend wiring hardening and reliability updates were completed in prior sessions.
- End-to-end flow was exercised previously (upload -> processing -> chat) and several integration issues were fixed.
- Context-aware SAHI optimization has been implemented in backend CV modules and wired into pipeline flow behind runtime controls.
- Packaging migration groundwork has been completed:
  - `pyproject.toml` upgraded to PEP-621 package metadata.
  - `src/gaffers_guide/` scaffold created.
  - CLI stub implemented at `src/gaffers_guide/cli.py`.
  - migration map added at `GAFFERS_GUIDE_MIGRATION_MAP.md`.

### Packaging artifacts currently present

- `pyproject.toml`
  - `[build-system]`, `[project]`, and `[project.scripts]` added
  - script entrypoint: `gaffers-guide = "gaffers_guide.cli:main"`
  - setuptools src-layout package discovery configured
- `src/gaffers_guide/__init__.py`
- `src/gaffers_guide/cli.py`
  - typed `argparse`
  - `run` subcommand with `--video`, `--precision`, `--output`
  - logging-only behavior (no full runtime wiring yet)
- `GAFFERS_GUIDE_MIGRATION_MAP.md`
  - move map from `backend/scripts` and `backend/services/cv` into future `src/gaffers_guide/*`
  - import rewrite patterns
  - migration order

## Next Work Package (Primary Assignment)

Implement a user-facing quality/speed profile system in the package/CLI so users can explicitly choose tradeoffs between output quality and runtime speed.

### Product goal

Provide multiple execution profiles and make their tradeoffs explicit:

- `fast`: prioritize speed/latency
- `balanced`: middle ground
- `high_res`: higher quality with lower FPS
- `sahi` (or `max_quality`): highest ball recall / most expensive runtime

## Functional Requirements

### 1) CLI profile selection

Extend `gaffers-guide run` with an explicit profile selector:

- Add `--quality-profile` with enumerated allowed values.
- Keep compatibility with `--precision` (map to profile internally) or deprecate with a clear warning path.
- Update `--help` text so users understand speed/quality tradeoff intent.

Optional but recommended:

- `gaffers-guide profiles list` to show all profiles and defaults.
- `--explain-profile` to print the active resolved runtime knobs.

### 2) Single source of truth for profile config

Create one typed config module for profile definitions. Do not scatter constants across runtime modules.

Each profile should define, at minimum:

- SAHI enabled/disabled
- slice width/height
- slice overlap ratios
- temporal search region strategy (radius/base behavior)
- pitch ROI masking behavior
- confidence thresholds for ball candidate selection
- optional frame processing cadence controls
- batch-size related limits for slice inference

### 3) Runtime integration

At startup:

- resolve selected profile
- log resolved profile and effective parameters in structured logs
- pass config down pipeline entrypoints cleanly

### 4) Safe defaults and validation

- default profile should be safe for normal operation (`balanced` preferred)
- invalid profile should fail fast with useful error text + allowed values
- avoid breaking existing invocations

### 5) Documentation

Add/extend docs with:

- profile descriptions in plain language
- expected speed vs quality tradeoffs
- example commands
- recommended profile by use case (preview, full analysis, QA, debugging)

## Architecture Expectations

- Keep package architecture under `src/gaffers_guide/*`.
- Maintain typed interfaces and mypy-friendly code.
- No ad-hoc conditionals spread across many modules for profile behavior.
- Prefer one resolver that maps CLI selection -> normalized config object.

## Testing and Verification Requirements

At minimum include:

- CLI parse tests for valid profiles
- invalid profile behavior tests
- config mapping/resolution tests
- smoke test that selected profile reaches runtime path (can be via structured logs or injected config assertion)

Run and attach outputs for:

- lint/type checks for changed files
- CLI help verification (`python -m gaffers_guide.cli --help`)

If benchmarking is done, report by profile:

- FPS
- ball recall/continuity behavior
- false positive trend
- runtime cost delta

## Definition of Done

- User can run profile-specific commands such as:
  - `gaffers-guide run --video <path> --output <dir> --quality-profile fast`
  - `gaffers-guide run --video <path> --output <dir> --quality-profile balanced`
  - `gaffers-guide run --video <path> --output <dir> --quality-profile high_res`
  - `gaffers-guide run --video <path> --output <dir> --quality-profile sahi`
- Each profile maps to distinct runtime behavior via centralized config.
- CLI and docs clearly communicate tradeoffs.
- No regression in existing pipeline entry flows.
- PR includes tests and verification evidence.

## Notes and Constraints

- SAHI slicing implementation already exists; do not re-implement SAHI internals unless needed for profile parameterization.
- Focus on productization and clean control surfaces (CLI + config + runtime integration).
- Follow migration strategy in `GAFFERS_GUIDE_MIGRATION_MAP.md` for future moves; do not perform broad module moves unless explicitly scoped.
