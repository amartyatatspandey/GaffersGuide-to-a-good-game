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

## RunPod network volume (S3 upload)

To push **this repo only** to a RunPod volume via S3 API, see [docs/runpod_volume_upload.md](docs/runpod_volume_upload.md) and run `./scripts/sync_to_runpod_s3.sh` with `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` set (see `.env.runpod.example`).
