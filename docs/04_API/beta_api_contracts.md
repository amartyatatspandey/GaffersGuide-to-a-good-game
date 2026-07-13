# Beta API Contract Freeze (v1beta)

This document freezes beta contracts so frontend and backend evolve safely.

## Endpoints

### Create Job

- `POST /api/v1beta/jobs`
- Form fields:
  - `file` (mp4)
  - `cv_engine`: `local|cloud`
  - `llm_engine`: `local|cloud`
  - `idempotency_key` (optional)
- Response (`CreateJobResponse`):
  - `job_id`
  - `status`
  - `cv_engine`
  - `llm_engine`

### Job Status

- `GET /api/v1beta/jobs/{job_id}`
- Response:
  - `job_id`
  - `status`
  - `current_step`
  - `result_path`
  - `tracking_overlay_path`
  - `tracking_data_path`
  - `error`
  - `created_at`
  - `updated_at`

### Job Artifacts

- `GET /api/v1beta/jobs/{job_id}/artifacts`
- Response:
  - `job_id`
  - `status`
  - `report_path`
  - `tracking_overlay_path`
  - `tracking_data_path`

### Job Progress WebSocket

- `WS /ws/v1beta/jobs/{job_id}`
- Message shape:
  - `job_id`
  - `status`
  - `current_step`
  - `result_path`
  - `tracking_overlay_path`
  - `tracking_data_path`
  - `error`

### Metrics

- `GET /api/v1beta/metrics`
- Response:
  - `snapshot.counters`
  - `snapshot.timers` (with count, p50_ms, p95_ms, avg_ms)
  - `promotion_gate` (success rate threshold pass/fail)

## Frontend Contract Notes

- `Gaffers-Guide_ULTIMATE-frontend_dev/src/lib/api.ts` exports:
  - `uploadVideoBeta(...)`
  - `jobsWsUrlBeta(...)`
- `useWebSocketProgress` supports `useBeta` flag.

## Compatibility Rules

- v1 contracts remain stable and untouched.
- v1beta contracts can evolve, but only with explicit version increments and frontend updates in same release.
