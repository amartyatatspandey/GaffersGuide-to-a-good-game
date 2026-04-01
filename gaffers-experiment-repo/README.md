# Gaffers Experiment Repo

This repository is an isolated experimental stack split from the main product.

## Structure

- `experiment-backend/` — FastAPI experiment API (`/api/exp/*`, `/ws/exp/*`)
- `experiment-frontend/` — separate desktop/web client
- `experiment-infra/` — separate deployment manifests
- `docs/` — runbook and isolation audit

## Non-goals

- No production `v1` route wiring.
- No shared runtime endpoints.
- No shared experiment artifacts under main product output directories.
