# Beta Pipeline Architecture Specification

## Objective

Define a secondary beta pipeline that can run in production with lower blast radius and measurable performance before promotion.

## Control Plane

- API service accepts jobs at `/api/v1beta/jobs`.
- Jobs are persisted via JSON-backed durable store abstraction in `backend/services/beta_job_store.py`.
- Idempotency key support avoids duplicate jobs on client retries.
- Job status is available by REST and websocket:
  - `GET /api/v1beta/jobs/{job_id}`
  - `WS /ws/v1beta/jobs/{job_id}`

## Data Plane

- API enqueues jobs to in-process async queue (`backend/services/beta_queue.py`) instead of running heavy CV inside request handler.
- Worker dequeues one job at a time and executes CV runner through existing routing layer (`CVRouterFactory`), preserving existing analytics behavior.
- Artifacts remain compatible with existing naming:
  - `{job_id}_report.json`
  - `{job_id}_tracking_data.json`
  - `{job_id}_tracking_overlay.mp4`

## Storage Model

- Current beta store:
  - Job metadata: `backend/output/beta_jobs_store.json`
  - Artifacts: `backend/output/*.json` and `*.mp4`
- Planned promotion path:
  - Metadata -> Firestore/managed DB
  - Artifacts -> object storage (GCS)

## Rollback Strategy

- Keep v1 endpoints unchanged.
- Beta traffic explicitly calls `/api/v1beta/*`.
- Kill-switch is route-level: stop directing clients to beta endpoints.
- If queue path degrades, fallback to existing v1 async task pipeline.

## Failure Semantics

- Worker failures mark jobs as `error` with step `Error`.
- Cloud/local routing errors are surfaced in job error payload for deterministic troubleshooting.
- Queue processing remains isolated from request latency.

## Metrics Integration

- `backend/services/observability.py` captures counters and p50/p95 timings.
- `GET /api/v1beta/metrics` returns:
  - stage counters
  - latency distributions
  - promotion gate pass/fail snapshot
