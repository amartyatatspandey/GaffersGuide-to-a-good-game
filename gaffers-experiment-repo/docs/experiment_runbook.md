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

## Smoke Validation

1. Upload a fixture clip from `experiment-backend/data/fixtures/` (for example, `short/short_czech_england_120s.mp4`).
2. Wait for websocket status `done`.
3. Load advice + tracking + reports from experiment endpoints.

## Fixture Dataset Preparation

Build experiment-local fixtures from `backend/data/training_samples`:

`python -m scripts.prepare_fixtures --training-samples-dir "../../../backend/data/training_samples" --fixtures-dir "../data/fixtures"`

This writes:

- categorized fixture clips under `experiment-backend/data/fixtures/{short,medium,long,stress}/`
- dataset metadata manifest at `experiment-backend/data/fixtures/manifest.json`

## Benchmark

Run:

`python -m scripts.benchmark_decoders "data/fixtures/short/short_czech_england_120s.mp4" --output-json output/exp/decoder_benchmark_short.json`

Run NVIDIA matrix + release SLA gates:

`python -m scripts.benchmark_decoders "data/fixtures/long/long_barca_madrid_1200s.mp4" --matrix --runtime-target nvidia --hardware-profile l4 --output-json output/exp/decoder_benchmark_matrix_nvidia.json`

Run Apple MPS matrix + tracked SLA:

`python -m scripts.benchmark_decoders "data/fixtures/long/long_barca_madrid_1200s.mp4" --matrix --runtime-target apple_mps --hardware-profile mps --output-json output/exp/decoder_benchmark_matrix_mps.json`

## Promotion Policy

- NVIDIA profile is release-blocking:
  - P50 <= 5 min and P95 <= 7 min (30-min equivalent)
  - Success rate >= 99%
  - No major quality regressions
- Apple MPS profile is non-blocking, trend-tracked:
  - Relaxed target (default P50 <= 12 min equivalent)
  - Must not regress materially versus prior baseline

This benchmark executes only experiment code paths.
