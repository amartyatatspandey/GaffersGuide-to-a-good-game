# SAHI Optimization Plan

## Current State Diagnosis

### 1) What is currently running for ball detection
- The active CV path uses plain `ultralytics.YOLO` inference, not SAHI.
- Inference entrypoints:
  - [`backend/scripts/pipeline_core/run_e2e_cloud.py`](/Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/backend/scripts/pipeline_core/run_e2e_cloud.py)
  - [`backend/scripts/pipeline_core/e2e_shared_impl.py`](/Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/backend/scripts/pipeline_core/e2e_shared_impl.py)
- `run_e2e_cloud.py` creates a YOLO model and calls `model(batch_frames, conf=0.3, ...)` (batched full-frame inference across consecutive frames), then performs per-frame parsing of detections.
- `e2e_shared_impl.py` contains a similar non-batched-per-frame variant (`model(frame, conf=0.3, ...)`) used by shared/legacy flows.

### 2) Is SAHI implemented?
- No SAHI import or usage was found in backend pipeline scripts.
- No `sahi` usage was found in backend requirements either.
- Conclusion: **SAHI is not currently present in the detection path**.

### 3) How ball filtering currently works
- Ball class IDs are resolved by `_resolve_primary_ball_class_ids(model)` in [`backend/scripts/pipeline_core/e2e_shared_impl.py`](/Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/backend/scripts/pipeline_core/e2e_shared_impl.py):
  - It searches YOLO class names for `"ball"` strings.
  - Fallback IDs are `[32, 0]` if names do not resolve.
- In `run_e2e_cloud.py`, per frame:
  - Iterate through detections.
  - Keep only detections whose `class_id` is in `primary_ball_class_ids`.
  - Track max confidence via `best_ball_score` and retain the single best bbox (`best_ball_bbox`).
  - Map that bbox to radar coordinates and optionally fallback with optical flow projection if homography confidence is low.
- Net effect: **single best confidence ball candidate from full-frame YOLO detections**.

### 4) Important adjacent observation
- [`backend/scripts/pipeline_core/generate_analytics.py`](/Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/backend/scripts/pipeline_core/generate_analytics.py) is analytics post-processing; it does not run model inference and is not part of the ball detector execution path.

## Gap Analysis vs Target SAHI Optimizations

### Pitch ROI Masking
- Current: not present; YOLO runs on entire frame.
- Gap: no pitch-aware spatial pruning before ball search.

### Temporal Tracking Crop
- Current: not present in detector entry stage; `last_ball_radar` is used later for fallback projection logic, not for narrowing detector search area.
- Gap: no "search radius around last known ball" for primary detection.

### Batched Slice Inference
- Current: frame batching exists (`model(batch_frames, ...)`), but this is not SAHI slice batching.
- Gap: no slice generation pipeline and no batched GPU inferencing over generated slices.

## Upgrade Path (Context-Aware SAHI without FPS collapse)

### Step 0: Introduce feature flags and safe fallback
1. Add config flags for:
   - `enable_context_sahi`
   - `sahi_temporal_radius_px`
   - `sahi_slice_h`, `sahi_slice_w`, `sahi_overlap_ratio`
   - `sahi_min_confidence`
   - `sahi_max_slices_per_frame`
2. Keep full-frame YOLO path as fallback when:
   - pitch mask unavailable
   - temporal prior unavailable and slice budget exceeded
   - SAHI runtime errors

### Step 1: Build pitch ROI provider
1. Create a pitch mask provider that returns a binary mask per frame (or a reused mask with lightweight updates).
2. Generate candidate SAHI regions only where the mask indicates pitch occupancy.
3. Enforce hard cap on ROI area ratio to control worst-case cost.

### Step 2: Add temporal prior crop
1. Maintain `last_ball_xy` in image coordinates (not only radar coordinates).
2. Build a temporal search box centered at `last_ball_xy` with adaptive radius:
   - small radius when confidence/history is stable
   - expanded radius when confidence drops or misses occur
3. Intersect temporal box with pitch mask region.
4. If no temporal prior exists (cold start), use pitch ROI slicing.

### Step 3: Implement SAHI slice generation policy
1. Generate slices only over the selected context region (ROI + temporal crop intersection).
2. Prioritize central slices around temporal prior first.
3. Enforce max slice count per frame and deterministic ordering.

### Step 4: Batched slice GPU inference
1. Convert generated slices into one batched tensor call (single model forward path per frame window).
2. Avoid sequential Python for-loop inference over slices.
3. Decode and remap slice detections back to global image coordinates.

### Step 5: Detection fusion and final ball pick
1. Merge overlapping candidate boxes (NMS/WBF policy).
2. Score candidates with context-aware ranking:
   - detector confidence
   - proximity to temporal prior
   - inside-pitch-mask penalty/bonus
3. Output one `best_ball_bbox` preserving current downstream contract.

### Step 6: Runtime control and adaptive modes
1. Add adaptive mode switching:
   - if FPS drops below threshold, reduce slice count/overlap
   - if repeated ball misses, temporarily expand search radius and slice area
2. Add telemetry:
   - slices/frame
   - SAHI latency/frame
   - detector hit rate
   - fallback hit rate

### Step 7: Verification protocol
1. Baseline benchmark with current YOLO path:
   - FPS
   - ball recall / miss streaks
   - confidence distribution
2. Compare against optimized SAHI modes:
   - ROI-only
   - ROI + temporal
   - ROI + temporal + batched slices
3. Accept only if recall improves and FPS remains within target envelope.

## Proposed Class Design: `OptimizedSAHIWrapper`

## Design goals
- Keep `run_e2e_cloud.py` orchestration clean.
- Encapsulate all SAHI-specific logic.
- Strict typing and composable components.
- No deeply nested control flow in pipeline entrypoint.

### Suggested module split
- `services/cv/optimized_sahi_wrapper.py`
- `services/cv/pitch_roi_provider.py`
- `services/cv/temporal_ball_prior.py`
- `services/cv/slice_batch_inferencer.py`
- `services/cv/ball_candidate_fuser.py`

### Core types
```python
from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass(frozen=True)
class BallCandidate:
    xyxy: np.ndarray
    confidence: float
    source: str

@dataclass(frozen=True)
class DetectionContext:
    frame_idx: int
    frame_bgr: np.ndarray
    pitch_mask: np.ndarray | None
    last_ball_xy: tuple[float, float] | None

@dataclass(frozen=True)
class BallDetectionResult:
    best_ball_bbox: np.ndarray | None
    best_ball_score: float
    candidates: list[BallCandidate]
    used_fallback: bool
```

### Class contract
```python
class OptimizedSAHIWrapper:
    def detect_ball(self, ctx: DetectionContext) -> BallDetectionResult: ...
```

### Internal responsibilities
1. `_build_search_region(ctx) -> np.ndarray | None`
   - Combines pitch ROI and temporal crop.
2. `_generate_slices(search_region) -> list[np.ndarray]`
   - Deterministic, budgeted slice creation.
3. `_infer_slices_batched(slices, frame_bgr) -> list[BallCandidate]`
   - Single batched GPU inference path.
4. `_fuse_candidates(candidates, ctx) -> BallCandidate | None`
   - NMS/WBF + temporal/context ranking.
5. `_fallback_fullframe(ctx) -> BallDetectionResult`
   - Safety path on misses/errors.

### Integration point in existing pipeline
- Replace current `best_ball_score`/`best_ball_bbox` loop in:
  - [`backend/scripts/pipeline_core/run_e2e_cloud.py`](/Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/backend/scripts/pipeline_core/run_e2e_cloud.py)
- Keep downstream mapping and possession logic unchanged.

## Architecture Flow

```mermaid
flowchart TD
    frameIn["Frame + Metadata"] --> roiBuild["Build pitch ROI"]
    roiBuild --> temporalCrop["Apply temporal crop from last_ball_xy"]
    temporalCrop --> sliceGen["Generate bounded slices"]
    sliceGen --> batchInfer["Run batched slice inference on GPU"]
    batchInfer --> fuseCandidates["Fuse and rank ball candidates"]
    fuseCandidates --> bestBall["Emit best_ball_bbox + best_ball_score"]
    bestBall --> pipelineOut["Existing radar/projection/possession pipeline"]
    fuseCandidates -->| "no reliable candidate" | fallbackPath["Fallback full-frame detector"]
    fallbackPath --> pipelineOut
```

## Blunt Summary
- **Current state:** SAHI is absent; detector is standard YOLO full-frame with class-id filtering and max-confidence ball pick.
- **Target state:** Context-aware SAHI wrapper with pitch ROI + temporal crop + batched slice inference, integrated as a clean typed service to improve ball recall while preserving FPS.
