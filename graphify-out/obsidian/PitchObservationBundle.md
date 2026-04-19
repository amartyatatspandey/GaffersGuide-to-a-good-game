---
source_file: "backend/scripts/pipeline_core/dynamic_homography.py"
type: "code"
community: "Community 8"
location: "L45"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Community_8
---

# PitchObservationBundle

## Connections
- [[.collect_pitch_observations()]] - `calls` [EXTRACTED]
- [[AdvancedPitchCalibrator]] - `uses` [INFERRED]
- [[Apply user-specified division-style undistortion in normalized centered coordina]] - `uses` [INFERRED]
- [[Convert sn-calibration polyline (row, col) to pixel segments (u, v).]] - `uses` [INFERRED]
- [[Delegate to underlying V1 calibrator (for tests  introspection).]] - `uses` [INFERRED]
- [[Each observation is (line_class, u_pixel, v_pixel) for one extremity.]] - `uses` [INFERRED]
- [[Image line l such that lT x = 0 for image point x (homogeneous), l ~ inv(H).T @]] - `uses` [INFERRED]
- [[Image points (N, 2) in pixel coords and list of point_dict keys for world corres]] - `uses` [INFERRED]
- [[Intermediate calibration observations at segmentation resolution (SEG_WIDTH x SE]] - `rationale_for` [EXTRACTED]
- [[Intersection of finite segments p0–p1 and q0–q1 in R2.     Returns the point if]] - `uses` [INFERRED]
- [[Longest polyline per class (same heuristic as get_line_extremities).]] - `uses` [INFERRED]
- [[Mean L2 distance in seg pixels between projections of four pitch corners.]] - `uses` [INFERRED]
- [[Orthogonal distance from (u,v) to line l=(a,b,c) with a2+b2=1.]] - `uses` [INFERRED]
- [[Pitch - image in 1280×720 pixels (TacticalRadar calibration space).]] - `uses` [INFERRED]
- [[PitchLineModel]] - `uses` [INFERRED]
- [[Return refined H (3x3, h33=1) and final cost.]] - `uses` [INFERRED]
- [[SegmentationNetwork]] - `uses` [INFERRED]
- [[Two-stage pitch calibration algebraic line homography + geometric LM refinement]] - `uses` [INFERRED]
- [[V2 calibrator coarse SVD (+ optional RANSAC seed), LM geometric refinement, out]] - `uses` [INFERRED]
- [[dynamic_homography.py]] - `contains` [EXTRACTED]
- [[p is 8-vector h11..h32; h33 fixed to 1.]] - `uses` [INFERRED]

#graphify/code #graphify/INFERRED #community/Community_8