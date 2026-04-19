---
source_file: "backend/scripts/pipeline_core/e2e_shared_impl.py"
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
- [[.update()_6]] - `calls` [EXTRACTED]
- [[.update_camera_angle()]] - `calls` [INFERRED]
- [[.write()_1]] - `calls` [INFERRED]
- [[CVTelemetry]] - `calls` [EXTRACTED]
- [[HybridIDHealer]] - `calls` [INFERRED]
- [[Model]] - `calls` [INFERRED]
- [[OpticalFlowCameraShiftEstimator]] - `calls` [EXTRACTED]
- [[Run CV tracking in-memory and return TacticalFrame timeline.]] - `rationale_for` [EXTRACTED]
- [[TacticalFrame]] - `calls` [EXTRACTED]
- [[TacticalPlayer]] - `calls` [EXTRACTED]
- [[TacticalRadar]] - `calls` [INFERRED]
- [[TeamClassifier]] - `calls` [INFERRED]
- [[TrackingFrameArtifact]] - `calls` [EXTRACTED]
- [[_draw_annotated_frame()]] - `calls` [EXTRACTED]
- [[_fallback_project_from_camera_shift()]] - `calls` [EXTRACTED]
- [[_homography_confidence()]] - `calls` [EXTRACTED]
- [[_prediction_to_team()]] - `calls` [EXTRACTED]
- [[_resolve_primary_ball_class_ids()]] - `calls` [EXTRACTED]
- [[compute_possession_team_id()]] - `calls` [INFERRED]
- [[default_homography_provider()]] - `calls` [INFERRED]
- [[e2e_shared_impl.py]] - `contains` [EXTRACTED]
- [[format_tracking_model_missing_reason()]] - `calls` [INFERRED]
- [[interpolate_ball_positions()]] - `calls` [INFERRED]
- [[max()]] - `calls` [INFERRED]
- [[mkdir()]] - `calls` [INFERRED]
- [[run_e2e()_1]] - `calls` [EXTRACTED]

#graphify/code #graphify/INFERRED #community/Community_5