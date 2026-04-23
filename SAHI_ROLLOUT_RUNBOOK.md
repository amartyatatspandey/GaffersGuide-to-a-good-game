# SAHI Rollout Runbook

## Baseline vs Candidate (match_test.mp4)

Source metrics:
- `backend/output/sahi_baseline_metrics.json`
- `backend/output/sahi_candidate_metrics_tuned.json`
- `backend/output/sahi_ab_summary.json`

Current A/B snapshot:
- Baseline FPS: **24.27**
- Context-SAHI FPS (tuned): **8.02**
- FPS delta: **-66.95%**
- Ball visibility ratio delta: **+207.08%** (0.254 -> 0.780)
- Max missing streak delta: **293 -> 80 frames**

Interpretation:
- Ball recall/continuity improves significantly.
- Performance regression is currently too large for default-on rollout.

## Acceptance thresholds

Use these gates before enabling by default:
- Visibility ratio improvement: **>= +35%** over baseline.
- Max missing streak reduction: **>= 30%**.
- FPS degradation: **<= 25%** from baseline.
- Pipeline stability: zero crash/fallback fatal errors over 2 full runs.

Current status:
- Recall gates: **PASS**
- FPS gate: **FAIL**
- Default-on decision: **NO-GO**

## Runtime flags

All flags are optional env vars; defaults are conservative and wrapper is off by default.

- `GAFFERS_ENABLE_CONTEXT_SAHI=1`
- `GAFFERS_SAHI_CONF` (default `0.25`)
- `GAFFERS_SAHI_HIGH_CONF_SKIP` (default `0.25`)
- `GAFFERS_SAHI_SLICE_W` / `GAFFERS_SAHI_SLICE_H` (default `256`)
- `GAFFERS_SAHI_OVERLAP_RATIO` (default `0.15`)
- `GAFFERS_SAHI_MAX_SLICES` (default `4`)
- `GAFFERS_SAHI_TEMPORAL_RADIUS` (default `112`)
- `GAFFERS_SAHI_TEMPORAL_MAX_RADIUS` (default `360`)
- `GAFFERS_SAHI_TEMPORAL_EXPAND` (default `24`)

## Recommended staged rollout

1) **Off by default (current)**
- Keep `GAFFERS_ENABLE_CONTEXT_SAHI=0`.

2) **Internal profiling mode**
- Enable SAHI only for targeted clips and collect:
  - `Context SAHI summary` log line (frames/slices/fallback),
  - end-to-end FPS,
  - visibility ratio and miss streak.

3) **Performance tuning cycle**
- Reduce SAHI invocation frequency:
  - run on strict miss-only windows,
  - enforce smaller temporal radii and lower max slices.
- Consider adding cadence control (every Nth miss frame).

4) **Candidate default-on trial**
- Enable for internal users only after passing acceptance thresholds for at least 3 representative clips.

5) **Default-on production**
- Only when FPS and stability gates pass with margin.

## Quick benchmark commands

Baseline:
`cd backend && python3.11 - <<'PY' ... run_cv_tracking_batched(video) ... PY`

Candidate:
`cd backend && python3.11 - <<'PY' ... run_cv_tracking_batched(video, enable_context_sahi=True) ... PY`

## Rollback

Immediate rollback path:
- Set `GAFFERS_ENABLE_CONTEXT_SAHI=0`
- Restart API process

No schema changes were introduced in tracking outputs, so rollback is non-breaking.

