---
type: community
cohesion: 0.05
members: 85
---

# Community 8

**Cohesion:** 0.05 - loosely connected
**Members:** 85 nodes

## Members
- [[.__init__()_219]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[.__init__()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[.__init__()_216]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[.analyse_image()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[.collect_pitch_observations()_1]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[.collect_pitch_observations()]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[.get_2d_homogeneous_line()]] - code - backend/calibration/ports.py
- [[.get_homography()_7]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[.get_homography()_6]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[.init_weight()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[AdvancedPitchCalibrator]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Apply user-specified division-style undistortion in normalized centered coordina]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Convert sn-calibration polyline (row, col) to pixel segments (u, v).]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Delegate to underlying V1 calibrator (for tests  introspection).]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Dynamic pitch calibration using SoccerNetsn-calibration reference pipeline.  Ma]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[DynamicPitchCalibrator]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[Each observation is (line_class, u_pixel, v_pixel) for one extremity.]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[FIFA pitch model used for line correspondences (read-only).]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[Fits circles on the True pixels of the mask and returns those which have enough]] - rationale - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[Given a list of points that were extracted from the blobs belonging to a same se]] - rationale - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[Given the dictionary {lines_class points}, finds plausible extremities of each]] - rationale - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[Image line l such that lT x = 0 for image point x (homogeneous), l ~ inv(H).T @]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Image points (N, 2) in pixel coords and list of point_dict keys for world corres]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Initialize the SoccerNet segmentation model and load weights.          param we]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[Intermediate calibration observations at segmentation resolution (SEG_WIDTH x SE]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[Intersection of finite segments p0–p1 and q0–q1 in R2.     Returns the point if]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Longest polyline per class (same heuristic as get_line_extremities).]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Mean L2 distance in seg pixels between projections of four pitch corners.]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Minimal surface used by line-match and homography helpers.]] - rationale - backend/calibration/ports.py
- [[Orthogonal distance from (u,v) to line l=(a,b,c) with a2+b2=1.]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Pitch - image in 1280×720 pixels (TacticalRadar calibration space).]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[PitchLineModel]] - code - backend/calibration/ports.py
- [[PitchObservationBundle]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[Process image and perform inference, returns mask of detected classes         p]] - rationale - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[Resolve homography JSON for ``video_path`` (env override or per-stem under ``out]] - rationale - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[Return ``(pth, mean_npy, std_npy)`` paths or raise ``FileNotFoundError``.]] - rationale - backend/scripts/pipeline_core/sn_calib_weights.py
- [[Return refined H (3x3, h33=1) and final cost.]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Returns the barycenter of the True pixels under the area of the mask delimited b]] - rationale - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[Run SoccerNet inference on a single frame and return the 3x3 homography.]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[Run pitch calibrator on a video file (e.g. match_test.mp4).  By default uses ``A]] - rationale - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[Run segmentation through coarse SVD homography at SEG resolution.          para]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[Segmentation inference helpers for DynamicPitchCalibrator.]] - rationale - backend/scripts/pipeline_core/calibration/inference.py
- [[SegmentationNetwork]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[SoccerNet pitch-segmentation weight file layout (shared by V1V2 calibrators).]] - rationale - backend/scripts/pipeline_core/sn_calib_weights.py
- [[Standalone calibrator one video frame (BGR) in - 3x3 homography (pitch - imag]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[This function selects for each class present in the semantic mask, a set of circ]] - rationale - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[Two-stage pitch calibration algebraic line homography + geometric LM refinement]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[V2 calibrator coarse SVD (+ optional RANSAC seed), LM geometric refinement, out]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_build_residuals_factory()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_collect_corner_image_points()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_condition_ok()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_default_output_for_video()]] - code - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[_homogeneous_line_from_two_pixels()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_intersect_lines_homogeneous()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_line_image_from_pitch()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_longest_polylines_from_skeletons()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_mean_corner_disagreement_px()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_params_to_H()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_parse_args()]] - code - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[_point_line_distance()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_polyline_to_pixel_segments()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_ransac_homography_from_corners()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_refine_homography_lm()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_segment_intersection()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_undistort_division_normalized()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_world_points_for_keys()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[advanced_pitch_calibration.py]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[detect_extremities.py]] - code - backend/calibration/sn_calibration_vendor/src/detect_extremities.py
- [[detect_extremities.py_1]] - code - backend/references/sn-calibration/src/detect_extremities.py
- [[detect_extremities.py_2]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[dynamic_homography.py]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[ensure_homography_json_for_video()]] - code - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[field()]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[generate_class_synthesis()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[get_line_extremities()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[get_support_center()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[inference.py_1]] - code - backend/scripts/pipeline_core/calibration/inference.py
- [[join_points()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[p is 8-vector h11..h32; h33 fixed to 1.]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[require_sn_calibration_weight_files()]] - code - backend/scripts/pipeline_core/sn_calib_weights.py
- [[run()]] - code - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[run_calibrator_on_video.py]] - code - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[run_segmentation_and_extremities()]] - code - backend/scripts/pipeline_core/calibration/inference.py
- [[sn_calib_weights.py]] - code - backend/scripts/pipeline_core/sn_calib_weights.py
- [[synthesize_mask()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Community_8
SORT file.name ASC
```

## Connections to other communities
- 12 edges to [[_COMMUNITY_Community 0]]
- 2 edges to [[_COMMUNITY_Community 29]]
- 2 edges to [[_COMMUNITY_Community 16]]
- 2 edges to [[_COMMUNITY_Community 6]]
- 2 edges to [[_COMMUNITY_Community 17]]
- 2 edges to [[_COMMUNITY_Community 3]]
- 1 edge to [[_COMMUNITY_Community 5]]
- 1 edge to [[_COMMUNITY_Community 1]]
- 1 edge to [[_COMMUNITY_Community 14]]

## Top bridge nodes
- [[.collect_pitch_observations()]] - degree 8, connects to 2 communities
- [[run()]] - degree 6, connects to 2 communities
- [[DynamicPitchCalibrator]] - degree 28, connects to 1 community
- [[SegmentationNetwork]] - degree 17, connects to 1 community
- [[PitchLineModel]] - degree 13, connects to 1 community