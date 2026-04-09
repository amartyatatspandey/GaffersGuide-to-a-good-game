# Isolation Audit

## Scope

Audit target: `gaffers-experiment-repo/` only.

## Runtime Isolation Checks

- Experiment backend routes are namespaced to `/api/exp/*` and `/ws/exp/*`.
- Experiment tests explicitly assert `/api/v1/*` routes return `404`.
- Queue backend is isolated to experiment-only durable file path:
  - `experiment-backend/output/exp/exp_task_queue.json`
- Worker process is separated from API process:
  - API entrypoint: `main.py`
  - Worker entrypoint: `worker_main.py`
- Runtime tracks are isolated under experiment routing/config:
  - `runtime_target=nvidia` for cloud GPU release gates
  - `runtime_target=apple_mps` for macOS local/dev benchmarking
- Job store path is isolated:
  - `experiment-backend/output/exp/exp_jobs_store.json`
- Experiment artifacts are isolated:
  - `experiment-backend/output/exp/*`

## Coupling Findings

- No runtime imports from the main product backend/frontend code paths were found.
- One test/benchmark fixture currently reads source video from:
  - `../../backend/data/match_test.mp4`

This fixture dependency is only for validation input media and does not route requests through the main product runtime.

## Smoke Validation Result

- File: `docs/exp_smoke_result.json`
- Outcome:
  - job completed successfully
  - `/api/exp/coach/advice` returned `200`
  - `/api/exp/jobs/{job_id}/tracking` returned `200`
  - `/api/exp/reports` returned `200`
  - `/api/exp/chat` returned `200`

## Benchmark Result

- File: `experiment-backend/output/exp/decoder_benchmark_match_test.json`
- `match_test.mp4` processed by both decoders in isolated experiment backend.
