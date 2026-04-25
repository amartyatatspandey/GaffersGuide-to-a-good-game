# gaffers-guide Migration Map (No File Moves Yet)

This map defines the target package layout for moving script-oriented modules under `backend/scripts` into `src/gaffers_guide` incrementally and safely.

## Exact Scaffolding Commands

```bash
mkdir -p src/gaffers_guide
touch src/gaffers_guide/__init__.py
touch src/gaffers_guide/cli.py
```

## Target Namespace Blueprint

- `src/gaffers_guide/cli.py`: CLI entrypoint.
- `src/gaffers_guide/runtime/*`: pipeline orchestrators and run commands.
- `src/gaffers_guide/pipeline/*`: core CV/runtime pipeline modules.
- `src/gaffers_guide/calibration/*`: calibration package.
- `src/gaffers_guide/cv/*`: SAHI and CV service modules (from `backend/services/cv`).
- `src/gaffers_guide/tools/*`: non-runtime auxiliary tools.
- `src/gaffers_guide/legacy/*`: deferred legacy scripts.

## Migration Table

| Current path | Future path | Priority | Notes |
|---|---|---|---|
| `backend/scripts/pipeline_core/run_e2e_cloud.py` | `src/gaffers_guide/runtime/run_e2e_cloud.py` | Core runtime first | Main cloud runner / CV orchestration |
| `backend/scripts/pipeline_core/run_e2e.py` | `src/gaffers_guide/runtime/run_e2e.py` | Core runtime first | Local runner |
| `backend/scripts/pipeline_core/track_teams.py` | `src/gaffers_guide/pipeline/track_teams.py` | Core runtime first | Team tracking |
| `backend/scripts/pipeline_core/track_teams_reid_hybrid.py` | `src/gaffers_guide/pipeline/track_teams_reid_hybrid.py` | Core runtime first | ReID hybrid tracking |
| `backend/scripts/pipeline_core/tactical_radar.py` | `src/gaffers_guide/pipeline/tactical_radar.py` | Core runtime first | Tactical radar generation |
| `backend/scripts/pipeline_core/dynamic_homography.py` | `src/gaffers_guide/pipeline/dynamic_homography.py` | Core runtime first | Homography logic |
| `backend/scripts/pipeline_core/e2e_shared.py` | `src/gaffers_guide/pipeline/e2e_shared.py` | Core runtime first | Shared E2E utilities |
| `backend/scripts/pipeline_core/e2e_shared_impl.py` | `src/gaffers_guide/pipeline/e2e_shared_impl.py` | Core runtime first | Shared implementation |
| `backend/scripts/pipeline_core/team_classifier.py` | `src/gaffers_guide/pipeline/team_classifier.py` | Core runtime first | Team model classifier |
| `backend/scripts/pipeline_core/reid_healer.py` | `src/gaffers_guide/pipeline/reid_healer.py` | Core runtime first | ReID stabilization |
| `backend/scripts/pipeline_core/global_refiner.py` | `src/gaffers_guide/pipeline/global_refiner.py` | Core runtime first | Track refinement |
| `backend/scripts/pipeline_core/generate_analytics.py` | `src/gaffers_guide/pipeline/generate_analytics.py` | Core runtime first | Analytics generation |
| `backend/scripts/pipeline_core/advanced_pitch_calibration.py` | `src/gaffers_guide/pipeline/advanced_pitch_calibration.py` | Core runtime first | Pitch calibration flow |
| `backend/scripts/pipeline_core/run_calibrator_on_video.py` | `src/gaffers_guide/runtime/run_calibrator_on_video.py` | Core runtime first | Calibration runtime entry |
| `backend/scripts/pipeline_core/sn_calib_weights.py` | `src/gaffers_guide/pipeline/sn_calib_weights.py` | Core runtime first | Calibration weights |
| `backend/scripts/pipeline_core/track_teams_constants.py` | `src/gaffers_guide/pipeline/track_teams_constants.py` | Core runtime first | Tracking constants |
| `backend/scripts/pipeline_core/calibration/*` | `src/gaffers_guide/calibration/*` | Core runtime first | Preserve as subpackage |
| `backend/services/cv/optimized_sahi_wrapper.py` | `src/gaffers_guide/cv/optimized_sahi_wrapper.py` | Core runtime first | SAHI wrapper |
| `backend/services/cv/pitch_roi_provider.py` | `src/gaffers_guide/cv/pitch_roi_provider.py` | Core runtime first | ROI estimation |
| `backend/services/cv/temporal_ball_prior.py` | `src/gaffers_guide/cv/temporal_ball_prior.py` | Core runtime first | Temporal prior |
| `backend/services/cv/slice_batch_inferencer.py` | `src/gaffers_guide/cv/slice_batch_inferencer.py` | Core runtime first | Batch inference |
| `backend/services/cv/ball_candidate_fuser.py` | `src/gaffers_guide/cv/ball_candidate_fuser.py` | Core runtime first | Candidate fusion |
| `backend/scripts/auxiliary_tools/*` | `src/gaffers_guide/tools/*` | Supporting tools second | Utility scripts and helpers |
| `backend/scripts/pipeline_core/legacy/*` | `src/gaffers_guide/legacy/*` | Legacy paths last | Migrate last; deprecate gradually |

## Import Rewrite Patterns

Use the following rewrite rules during migration:

```python
# Runtime and pipeline modules
from backend.scripts.pipeline_core.<module> import X
# ->
from gaffers_guide.pipeline.<module> import X
```

```python
# Runtime entry modules
from backend.scripts.pipeline_core.run_e2e_cloud import main
# ->
from gaffers_guide.runtime.run_e2e_cloud import main
```

```python
# Calibration package
from backend.scripts.pipeline_core.calibration.geometry import Y
# ->
from gaffers_guide.calibration.geometry import Y
```

```python
# SAHI / CV services
from backend.services.cv.optimized_sahi_wrapper import OptimizedSAHIWrapper
# ->
from gaffers_guide.cv.optimized_sahi_wrapper import OptimizedSAHIWrapper
```

```python
# Auxiliary tools
from backend.scripts.auxiliary_tools.analytics_filter import filter_records
# ->
from gaffers_guide.tools.analytics_filter import filter_records
```

## Recommended Move Order

1. **Core runtime first**
   - Move `runtime`, `pipeline`, `calibration`, and `cv` modules.
   - Update imports to `gaffers_guide.*`.
   - Keep temporary compatibility shims only if needed.
2. **Supporting tools second**
   - Move `auxiliary_tools` to `gaffers_guide.tools`.
   - Remove direct references to script paths.
3. **Legacy paths last**
   - Move `pipeline_core/legacy` into `gaffers_guide.legacy`.
   - Mark as deprecated and isolate from hot paths.
