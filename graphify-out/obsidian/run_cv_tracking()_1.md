---
source_file: "backend/scripts/pipeline_core/legacy/run_e2e_legacy.py"
type: "code"
community: "Community 5"
location: "L805"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Community_5
---

# run_cv_tracking()

## Connections
- [[.ensure_homography_json()]] - `calls` [INFERRED]
- [[.get()]] - `calls` [INFERRED]
- [[.map_to_2d()]] - `calls` [INFERRED]
- [[.predict_frame()]] - `calls` [INFERRED]
- [[.process_and_heal()]] - `calls` [INFERRED]
- [[.update()_7]] - `calls` [EXTRACTED]
- [[.update_camera_angle()]] - `calls` [INFERRED]
- [[.write()_1]] - `calls` [INFERRED]
- [[CVTelemetry_1]] - `calls` [EXTRACTED]
- [[HybridIDHealer]] - `calls` [INFERRED]
- [[Model]] - `calls` [INFERRED]
- [[OpticalFlowCameraShiftEstimator_1]] - `calls` [EXTRACTED]
- [[Run CV tracking in-memory and return TacticalFrame timeline._1]] - `rationale_for` [EXTRACTED]
- [[TacticalFrame_1]] - `calls` [EXTRACTED]
- [[TacticalPlayer_1]] - `calls` [EXTRACTED]
- [[TacticalRadar]] - `calls` [INFERRED]
- [[TeamClassifier]] - `calls` [INFERRED]
- [[TrackingFrameArtifact_1]] - `calls` [EXTRACTED]
- [[_draw_annotated_frame()_1]] - `calls` [EXTRACTED]
- [[_fallback_project_from_camera_shift()_1]] - `calls` [EXTRACTED]
- [[_homography_confidence()_1]] - `calls` [EXTRACTED]
- [[_prediction_to_team()_1]] - `calls` [EXTRACTED]
- [[_resolve_primary_ball_class_ids()_1]] - `calls` [EXTRACTED]
- [[compute_possession_team_id()]] - `calls` [INFERRED]
- [[default_homography_provider()]] - `calls` [INFERRED]
- [[format_tracking_model_missing_reason()]] - `calls` [INFERRED]
- [[interpolate_ball_positions()]] - `calls` [INFERRED]
- [[max()]] - `calls` [INFERRED]
- [[mkdir()]] - `calls` [INFERRED]
- [[run_e2e()_2]] - `calls` [EXTRACTED]
- [[run_e2e_legacy.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/INFERRED #community/Community_5