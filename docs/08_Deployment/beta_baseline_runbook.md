# Beta Baseline Metrics Runbook

## Goal

Capture baseline latency/reliability metrics before promoting beta pipeline behavior.

## Steps

1. Deploy beta service with `infra/cloudbuild.beta.yaml`.
2. Run representative upload workloads against `/api/v1beta/jobs`.
3. Track job completion via `/ws/v1beta/jobs/{job_id}`.
4. Query `/api/v1beta/metrics` every N minutes and persist snapshots externally.

## Minimum Dataset

- Short clip (~30-60s)
- Medium clip (~3-5 min)
- Stress clip (longest expected duration)

## KPIs

- `beta.job.e2e.total_ms` p50/p95
- `beta.upload.write_ms` p50/p95
- `beta.jobs.succeeded / (succeeded + failed)`
- failure counts by error type (from job payload `error`)

## Exit Criteria

- Metrics are stable across at least 3 repeated runs per clip profile.
- Success rate >= 95% with no critical error class spikes.
