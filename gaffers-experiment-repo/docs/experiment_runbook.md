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
- In cloud mode (`EXP_CLOUD_MODE=1`), set `EXP_TASK_BACKEND=redis` and keep `EXP_ALLOW_LOCAL_BACKEND_FALLBACK=0` to fail fast if Redis is unavailable.

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
- `output/exp/bench_runs/<run_id>/benchmark.json`
- `output/exp/bench_runs/<run_id>/manifest.json`
- `output/exp/bench_runs/<run_id>/trials/...`

Optional baseline comparison:

`python -m scripts.compare_benchmarks --current output/exp/decoder_benchmark_matrix_nvidia.json --baseline /absolute/path/baseline.json --output-json output/exp/benchmark_delta_report_nvidia.json --min-improvement-pct 0.0`

## RunPod Setup (GPU)

1. Build and run the GPU worker image target:
   - `docker build -f experiment-backend/Dockerfile --target worker-gpu -t exp-worker-gpu experiment-backend`
2. Set cloud-safe env:
   - `EXP_CLOUD_MODE=1`
   - `EXP_TASK_BACKEND=redis`
   - `EXP_ALLOW_LOCAL_BACKEND_FALLBACK=0`
   - `EXP_REDIS_URL=<your redis url>`
3. Run API/worker separately and verify `/health`.
4. Execute benchmark commands above.
5. Download artifacts from:
   - `experiment-backend/output/exp/bench_runs/<run_id>/`
   - `experiment-backend/output/exp/decoder_benchmark_matrix_*.json`
   - `experiment-backend/output/exp/benchmark_delta_report_*.json`

Study checklist:
- Keep same video fixture and trial count for baseline/current runs.
- Keep same runtime target and hardware profile.
- Compare using `scripts.compare_benchmarks.py` and archive delta report.

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

## Stability Preflight and Integrity

Run before any matrix benchmark:

- `free -h`
- `cat /sys/fs/cgroup/memory.max`
- `which ffmpeg`
- `python - <<'PY'\nfrom services.splitter import read_cgroup_memory_limit_bytes\nprint(read_cgroup_memory_limit_bytes())\nPY`

Recommended staged run order:

1. Smoke: `--trials 1` on `train_short_5m.mp4` (no `--matrix`)
2. Matrix canary: `--matrix --trials 1` on short clip
3. Full matrix: `--matrix --trials 5` across short/medium/long manifest

Artifact integrity checks:

- Tracking artifacts are written atomically (`.tmp` then rename).
- Verify checksum sidecar:
  - `sha256sum output/exp/<job_id>_tracking_data.jsonl`
  - `cat output/exp/<job_id>_tracking_data.jsonl.sha256`
- If checksum mismatch occurs, mark run invalid and rerun from step 1.
