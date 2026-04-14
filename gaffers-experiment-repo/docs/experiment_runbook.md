# Experiment Runbook

## Local Startup

1. Start API backend:
   - `cd experiment-backend`
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
   - `uvicorn main:app --host 127.0.0.1 --port 8100`
2. Start worker process:
   - `cd experiment-backend`
   - `source .venv/bin/activate`
   - `python worker_main.py`
3. Start frontend:
   - `cd experiment-frontend`
   - `npm install`
   - `npm run dev`

## Runtime Defaults

- Decoder default is `opencv`; `pyav` remains available and falls back to `opencv` on decoder failure.
- Task backend default is `redis` (`EXP_TASK_BACKEND=redis`).
- For local-only operation without Redis, set `EXP_TASK_BACKEND=local`.
- If Redis is unavailable and `EXP_ALLOW_LOCAL_BACKEND_FALLBACK=1` (default), the backend falls back to local file queue.

## Homography In Experiment Backend

- Experiment backend now uses an experiment-local homography stack under:
  - `experiment-backend/services/homography_vendor/`
  - `experiment-backend/services/homography_adapter.py`
- Coordinates in `tracking_data.json` frames are emitted in pitch space:
  - `ball_xy` now represents pitch-space coordinates when homography is available.
  - `coord_space` is set to `pitch`.
  - `homography_applied` indicates whether the frame used a valid (or fallback) homography.
- Homography telemetry is included in tracking + job status:
  - `frames_with_homography`
  - `frames_without_homography`
  - `fallback_frames`
  - `calibration_latency_ms`
- Optional job input form field:
  - `homography_weights_dir` (absolute or resolvable path for calibration resources)

Environment toggles:
- `EXP_HOMOGRAPHY_ENABLED=1|0` (default `1`)
- `EXP_HOMOGRAPHY_SAMPLE_EVERY=<int>` (default `5`, run calibrator every Nth frame and reuse previous matrix between samples)

## Smoke Validation

1. Upload `match_test.mp4`.
2. Wait for websocket status `done`.
3. Load advice + tracking + reports from experiment endpoints.

## Benchmark

Run:

`python -m scripts.benchmark_decoders "/absolute/path/to/match_test.mp4" --trials 5 --output-json output/exp/decoder_benchmark_match_test.json`

Run NVIDIA matrix + release SLA gates:

`python -m scripts.benchmark_decoders "/absolute/path/to/match_test.mp4" --matrix --trials 5 --runtime-target nvidia --hardware-profile l4 --quality-regression-ok --output-json output/exp/decoder_benchmark_matrix_nvidia.json`

Run Apple MPS matrix + tracked SLA:

`python -m scripts.benchmark_decoders "/absolute/path/to/match_test.mp4" --matrix --trials 5 --runtime-target apple_mps --hardware-profile mps --quality-regression-ok --output-json output/exp/decoder_benchmark_matrix_mps.json`

Matrix and temporary benchmark artifacts are written under run-unique timestamp paths:
- `output/exp/bench_tmp/<run_id>/...`
- `output/exp/bench_matrix/<run_id>/...`

## Promotion Policy

- NVIDIA profile is release-blocking:
  - Tier A P50 <= 10 min equivalent
  - Tier B P50 <= 5 min and P95 <= 7 min equivalent
  - Success rate >= 99%
  - No major quality regressions
- Apple MPS profile is non-blocking, trend-tracked:
  - Relaxed target (default P50 <= 12 min equivalent)
  - Must not regress materially versus prior baseline

This benchmark executes only experiment code paths.
