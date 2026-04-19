---
source_file: "backend/scripts/pipeline_core/dynamic_homography.py"
type: "code"
community: "Community 8"
location: "L60"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Community_8
---

# DynamicPitchCalibrator

## Connections
- [[.__init__()_216]] - `method` [EXTRACTED]
- [[.__init__()_219]] - `calls` [INFERRED]
- [[.collect_pitch_observations()]] - `method` [EXTRACTED]
- [[.get_homography()_6]] - `method` [EXTRACTED]
- [[AdvancedPitchCalibrator]] - `uses` [INFERRED]
- [[Apply user-specified division-style undistortion in normalized centered coordina]] - `uses` [INFERRED]
- [[Convert sn-calibration polyline (row, col) to pixel segments (u, v).]] - `uses` [INFERRED]
- [[Delegate to underlying V1 calibrator (for tests  introspection).]] - `uses` [INFERRED]
- [[Each observation is (line_class, u_pixel, v_pixel) for one extremity.]] - `uses` [INFERRED]
- [[Image line l such that lT x = 0 for image point x (homogeneous), l ~ inv(H).T @]] - `uses` [INFERRED]
- [[Image points (N, 2) in pixel coords and list of point_dict keys for world corres]] - `uses` [INFERRED]
- [[Intersection of finite segments p0–p1 and q0–q1 in R2.     Returns the point if]] - `uses` [INFERRED]
- [[Longest polyline per class (same heuristic as get_line_extremities).]] - `uses` [INFERRED]
- [[Mean L2 distance in seg pixels between projections of four pitch corners.]] - `uses` [INFERRED]
- [[Orthogonal distance from (u,v) to line l=(a,b,c) with a2+b2=1.]] - `uses` [INFERRED]
- [[Pitch - image in 1280×720 pixels (TacticalRadar calibration space).]] - `uses` [INFERRED]
- [[PitchLineModel]] - `uses` [INFERRED]
- [[Resolve homography JSON for ``video_path`` (env override or per-stem under ``out]] - `uses` [INFERRED]
- [[Return refined H (3x3, h33=1) and final cost.]] - `uses` [INFERRED]
- [[Run pitch calibrator on a video file (e.g. match_test.mp4).  By default uses ``A]] - `uses` [INFERRED]
- [[SegmentationNetwork]] - `uses` [INFERRED]
- [[Standalone calibrator one video frame (BGR) in - 3x3 homography (pitch - imag]] - `rationale_for` [EXTRACTED]
- [[Two-stage pitch calibration algebraic line homography + geometric LM refinement]] - `uses` [INFERRED]
- [[V2 calibrator coarse SVD (+ optional RANSAC seed), LM geometric refinement, out]] - `uses` [INFERRED]
- [[dynamic_homography.py]] - `contains` [EXTRACTED]
- [[main()_16]] - `calls` [INFERRED]
- [[p is 8-vector h11..h32; h33 fixed to 1.]] - `uses` [INFERRED]
- [[run()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Community_8