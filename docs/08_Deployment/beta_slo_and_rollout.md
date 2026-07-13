# Beta SLO Gates And Canary Rollout

## Baseline Metrics

Use `GET /api/v1beta/metrics` to collect baseline snapshots for:

- `beta.upload.write_ms`
- `beta.job.e2e.total_ms`
- `beta.jobs.created`
- `beta.jobs.enqueued`
- `beta.jobs.dequeued`
- `beta.jobs.succeeded`
- `beta.jobs.failed`

## Promotion Gates

Promotion from beta to primary requires all gates to pass on representative sample:

1. Job success rate >= 95%
2. p95 end-to-end latency improves or stays within agreed budget
3. No increase in hard failure classes (`CLOUD_CV_*`, queue worker failures)
4. Tactical output readability remains compliant (3-step normalized structure)

## Canary Strategy

1. Shadow mode: internal execution only, no user routing
2. Canary 5% of eligible traffic
3. Expand to 20%
4. Expand to 50%
5. Full promotion only if gates pass

## Rollback Triggers

Automatic rollback when any trigger is hit:

- Success rate below threshold for rolling window
- p95 latency regression beyond budget
- Repeated worker error spikes
- Artifact-readiness failures above tolerance

## Operational Notes

- Keep beta deployment isolated (`gaffers-guide-api-beta`) using `infra/cloudbuild.beta.yaml`.
- Keep max instances bounded (`--max-instances=1`) while storage remains local+JSON.
- Move to external stores before scaling out.
