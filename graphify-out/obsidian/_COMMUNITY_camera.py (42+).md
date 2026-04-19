---
type: community
cohesion: 0.03
members: 167
---

# camera.py (42+)

**Cohesion:** 0.03 - loosely connected
**Members:** 167 nodes

## Members
- [[.__init__()_221]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[.__init__()_1]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[.__init__()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[.__init__()_216]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[.__init__()_2]] - code - backend/references/soccersegcal/sncalib/soccerpitch.py
- [[.analyse_image()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[.collect_pitch_observations()_1]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[.collect_pitch_observations()]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[.distort()]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[.draw_colorful_pitch()]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[.draw_corners()]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[.draw_pitch()]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[.estimate_calibration_matrix_from_plane_homography()]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[.from_homography()]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[.get_2d_homogeneous_line()]] - code - backend/references/soccersegcal/sncalib/soccerpitch.py
- [[.get_homography()_7]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[.get_homography()_6]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[.init_weight()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[.points()]] - code - backend/references/soccersegcal/sncalib/soccerpitch.py
- [[.project_point()]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[.refine_camera()]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[.sample_field_points()]] - code - backend/references/soccersegcal/sncalib/soccerpitch.py
- [[.solve_pnp()]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[.to_json_parameters()]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[AdvancedPitchCalibrator]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Apply user-specified division-style undistortion in normalized centered coordina]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Build list of (pitch_line_2d_homogeneous, image_line_2d_homogeneous) for homogra]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[Camera]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[Computes confusion matrices for a level of precision specified by the threshold.]] - rationale - backend/references/soccersegcal/sncalib/evaluate_camera.py
- [[Computes euclidian distance between 2D points     param point1     param point]] - rationale - backend/references/soccersegcal/sncalib/evaluate_extremities.py
- [[Computes euclidian distance between a point and a polyline.     param point 2D]] - rationale - backend/references/soccersegcal/sncalib/evaluate_camera.py
- [[Computes the similarity transform such that the list of points is centered aroun]] - rationale - backend/references/soccersegcal/sncalib/baseline_cameras.py
- [[Conversion from euler angles to orientation matrix.     param pan     param t]] - rationale - backend/references/soccersegcal/sncalib/camera.py
- [[Convert sn-calibration polyline (row, col) to pixel segments (u, v).]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Decomposes the rotation matrix into pan, tilt and roll angles. There are two sol]] - rationale - backend/references/soccersegcal/sncalib/camera.py
- [[Delegate to underlying V1 calibrator (for tests  introspection).]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Draw the corners of a standard soccer pitch in the image.         param image_1]] - rationale - backend/references/soccersegcal/sncalib/camera.py
- [[Draw the corners of a standard soccer pitch in the image.         param image]] - rationale - backend/references/sn-calibration/src/camera.py
- [[Draws all the lines of the pitch on the image, each line color is specified by t_1]] - rationale - backend/references/soccersegcal/sncalib/camera.py
- [[Draws all the lines of the pitch on the image, each line color is specified by t]] - rationale - backend/references/sn-calibration/src/camera.py
- [[Draws all the lines of the pitch on the image.         param image         par_1]] - rationale - backend/references/soccersegcal/sncalib/camera.py
- [[Draws all the lines of the pitch on the image.         param image         par]] - rationale - backend/references/sn-calibration/src/camera.py
- [[Draws points along the soccer pitch markings elements in the image based on the]] - rationale - backend/references/soccersegcal/sncalib/baseline_cameras.py
- [[Dynamic pitch calibration using SoccerNetsn-calibration reference pipeline.  Ma]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[DynamicPitchCalibrator]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[Each observation is (line_class, u_pixel, v_pixel) for one extremity.]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Estimate 3x3 homography (pitch plane - image) from line extremities.     Return]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[Evaluates the prediction of extremities. The extremities associated to a class a]] - rationale - backend/references/soccersegcal/sncalib/evaluate_extremities.py
- [[FIFA pitch model used for line correspondences (read-only).]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[Fits circles on the True pixels of the mask and returns those which have enough]] - rationale - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[For lines belonging to the pitch lawn plane returns its 2D homogenous equation c]] - rationale - backend/references/sn-calibration/src/soccerpitch.py
- [[For lines belonging to the pitch lawn plane returns its 2D homogenous equation c_4]] - rationale - backend/references/soccersegcal/sncalib/soccerpitch.py
- [[Given a list of points that were extracted from the blobs belonging to a same se]] - rationale - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[Given a point in the normalized image plane, apply distortion         param poi_1]] - rationale - backend/references/soccersegcal/sncalib/camera.py
- [[Given a point in the normalized image plane, apply distortion         param poi]] - rationale - backend/references/sn-calibration/src/camera.py
- [[Given a set of camera parameters, this function adapts the camera to the desired]] - rationale - backend/references/soccersegcal/sncalib/evaluate_camera.py
- [[Given lines correspondences, computes the homography that maps best the two set]] - rationale - backend/references/soccersegcal/sncalib/baseline_cameras.py
- [[Given the dictionary {lines_class points}, finds plausible extremities of each]] - rationale - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[Given the homography from the world plane of the pitch and the image and a point]] - rationale - backend/references/soccersegcal/sncalib/camera.py
- [[Image line l such that lT x = 0 for image point x (homogeneous), l ~ inv(H).T @]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Image points (N, 2) in pixel coords and list of point_dict keys for world corres]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Initialize 3D coordinates of all elements of the soccer pitch.         param pi]] - rationale - backend/references/sn-calibration/src/soccerpitch.py
- [[Initialize 3D coordinates of all elements of the soccer pitch.         param pi_4]] - rationale - backend/references/soccersegcal/sncalib/soccerpitch.py
- [[Initialize the SoccerNet segmentation model and load weights.          param we]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[Intermediate calibration observations at segmentation resolution (SEG_WIDTH x SE]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[Intersection of finite segments p0–p1 and q0–q1 in R2.     Returns the point if]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Longest polyline per class (same heuristic as get_line_extremities).]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Mean L2 distance in seg pixels between projections of four pitch corners.]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Once that there is a minimal set of initial camera parameters (calibration, rota_1]] - rationale - backend/references/soccersegcal/sncalib/camera.py
- [[Once that there is a minimal set of initial camera parameters (calibration, rota]] - rationale - backend/references/sn-calibration/src/camera.py
- [[Orthogonal distance from (u,v) to line l=(a,b,c) with a2+b2=1.]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Pitch - image in 1280×720 pixels (TacticalRadar calibration space).]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[PitchObservationBundle]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[Process image and perform inference, returns mask of detected classes         p]] - rationale - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[Replace each line class key of the dictionary with its opposite element accordin]] - rationale - backend/references/soccersegcal/sncalib/evaluate_extremities.py
- [[Resolve homography JSON for ``video_path`` (env override or per-stem under ``out]] - rationale - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[Return refined H (3x3, h33=1) and final cost.]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Returns the barycenter of the True pixels under the area of the mask delimited b]] - rationale - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[Run SoccerNet inference on a single frame and return the 3x3 homography.]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[Run pitch calibrator on a video file (e.g. match_test.mp4).  By default uses ``A]] - rationale - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[Run segmentation through coarse SVD homography at SEG resolution.          para]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[Samples each pitch element every dist meters, returns a dictionary associating t]] - rationale - backend/references/sn-calibration/src/soccerpitch.py
- [[Samples each pitch element every dist meters, returns a dictionary associating t_4]] - rationale - backend/references/soccersegcal/sncalib/soccerpitch.py
- [[Saves camera to a JSON serializable dictionary.         return The dictionary_1]] - rationale - backend/references/soccersegcal/sncalib/camera.py
- [[Saves camera to a JSON serializable dictionary.         return The dictionary]] - rationale - backend/references/sn-calibration/src/camera.py
- [[Scale points by s_width and s_height factors     param points_dict dictionary]] - rationale - backend/references/soccersegcal/sncalib/evaluate_extremities.py
- [[SegmentationNetwork]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[SoccerPitch]] - code - backend/references/soccersegcal/sncalib/soccerpitch.py
- [[Standalone calibrator one video frame (BGR) in - 3x3 homography (pitch - imag]] - rationale - backend/scripts/pipeline_core/dynamic_homography.py
- [[Static class variables that are specified by the rules of the game]] - rationale - backend/references/soccersegcal/sncalib/soccerpitch.py
- [[This function selects for each class present in the semantic mask, a set of circ]] - rationale - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[This method initializes the calibration matrix from the homography between the w_1]] - rationale - backend/references/soccersegcal/sncalib/camera.py
- [[This method initializes the calibration matrix from the homography between the w]] - rationale - backend/references/sn-calibration/src/camera.py
- [[This method initializes the essential camera parameters from the homography betw_1]] - rationale - backend/references/soccersegcal/sncalib/camera.py
- [[This method initializes the essential camera parameters from the homography betw]] - rationale - backend/references/sn-calibration/src/camera.py
- [[Two-stage pitch calibration algebraic line homography + geometric LM refinement]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[Uses current camera parameters to predict where a 3D point is seen by the camera_1]] - rationale - backend/references/soccersegcal/sncalib/camera.py
- [[Uses current camera parameters to predict where a 3D point is seen by the camera]] - rationale - backend/references/sn-calibration/src/camera.py
- [[V2 calibrator coarse SVD (+ optional RANSAC seed), LM geometric refinement, out]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[With a known calibration matrix, this method can be used in order to retrieve ro_1]] - rationale - backend/references/soccersegcal/sncalib/camera.py
- [[With a known calibration matrix, this method can be used in order to retrieve ro]] - rationale - backend/references/sn-calibration/src/camera.py
- [[_build_line_matches()]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[_build_residuals_factory()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_collect_corner_image_points()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_condition_ok()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_extremities_to_homography()]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[_homogeneous_line_from_two_pixels()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_intersect_lines_homogeneous()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_line_image_from_pitch()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_longest_polylines_from_skeletons()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_mean_corner_disagreement_px()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_params_to_H()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_point_line_distance()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_polyline_to_pixel_segments()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_ransac_homography_from_corners()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_refine_homography_lm()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_segment_intersection()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_undistort_division_normalized()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[_world_points_for_keys()]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[advanced_pitch_calibration.py]] - code - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[baseline_cameras.py]] - code - backend/calibration/sn_calibration_vendor/src/baseline_cameras.py
- [[baseline_cameras.py_1]] - code - backend/references/sn-calibration/src/baseline_cameras.py
- [[baseline_cameras.py_2]] - code - backend/references/soccersegcal/sncalib/baseline_cameras.py
- [[calibration()]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[camera.py]] - code - backend/calibration/sn_calibration_vendor/src/camera.py
- [[camera.py_1]] - code - backend/references/sn-calibration/src/camera.py
- [[camera.py_2]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[detect_extremities.py]] - code - backend/calibration/sn_calibration_vendor/src/detect_extremities.py
- [[detect_extremities.py_1]] - code - backend/references/sn-calibration/src/detect_extremities.py
- [[detect_extremities.py_2]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[distance()]] - code - backend/references/soccersegcal/sncalib/evaluate_extremities.py
- [[distance_to_polyline()]] - code - backend/references/soccersegcal/sncalib/evaluate_camera.py
- [[draw_pitch_homography()]] - code - backend/references/soccersegcal/sncalib/baseline_cameras.py
- [[dynamic_homography.py]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[estimate_homography_from_line_correspondences()]] - code - backend/references/soccersegcal/sncalib/baseline_cameras.py
- [[evalai_camera.py]] - code - backend/calibration/sn_calibration_vendor/src/evalai_camera.py
- [[evalai_camera.py_1]] - code - backend/references/sn-calibration/src/evalai_camera.py
- [[evalai_camera.py_2]] - code - backend/references/soccersegcal/sncalib/evalai_camera.py
- [[evaluate()_2]] - code - backend/references/tvcalib/evaluation/eval_projection.py
- [[evaluate()]] - code - backend/references/soccersegcal/sncalib/evalai_camera.py
- [[evaluate_camera.py]] - code - backend/calibration/sn_calibration_vendor/src/evaluate_camera.py
- [[evaluate_camera.py_1]] - code - backend/references/sn-calibration/src/evaluate_camera.py
- [[evaluate_camera.py_2]] - code - backend/references/soccersegcal/sncalib/evaluate_camera.py
- [[evaluate_camera_prediction()]] - code - backend/references/soccersegcal/sncalib/evaluate_camera.py
- [[evaluate_detection_prediction()]] - code - backend/references/soccersegcal/sncalib/evaluate_extremities.py
- [[evaluate_extremities.py]] - code - backend/calibration/sn_calibration_vendor/src/evaluate_extremities.py
- [[evaluate_extremities.py_1]] - code - backend/references/sn-calibration/src/evaluate_extremities.py
- [[evaluate_extremities.py_3]] - code - backend/references/soccersegcal/sncalib/evaluate_extremities.py
- [[evaluate_extremities.py_2]] - code - backend/references/tvcalib/sn_segmentation/src/evaluate_extremities.py
- [[field()]] - code - backend/scripts/pipeline_core/dynamic_homography.py
- [[generate_class_synthesis()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[get_line_extremities()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[get_polylines()]] - code - backend/references/soccersegcal/sncalib/evaluate_camera.py
- [[get_support_center()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[join_points()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[mirror_labels()]] - code - backend/references/soccersegcal/sncalib/evaluate_extremities.py
- [[move_camera()]] - code - backend/references/soccersegcal/soccersegcal/pose.py
- [[normalization_transform()]] - code - backend/references/soccersegcal/sncalib/baseline_cameras.py
- [[p is 8-vector h11..h32; h33 fixed to 1.]] - rationale - backend/scripts/pipeline_core/advanced_pitch_calibration.py
- [[pan_tilt_roll_to_orientation()]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[rotation_matrix_to_pan_tilt_roll()]] - code - backend/references/soccersegcal/sncalib/camera.py
- [[scale_points()]] - code - backend/references/soccersegcal/sncalib/evaluate_extremities.py
- [[soccerpitch.py]] - code - backend/calibration/sn_calibration_vendor/src/soccerpitch.py
- [[soccerpitch.py_1]] - code - backend/references/sn-calibration/src/soccerpitch.py
- [[soccerpitch.py_2]] - code - backend/references/soccersegcal/sncalib/soccerpitch.py
- [[synthesize_mask()]] - code - backend/references/soccersegcal/sncalib/detect_extremities.py
- [[unproject_image_point()]] - code - backend/references/soccersegcal/sncalib/camera.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/camera.py_(42+)
SORT file.name ASC
```

## Connections to other communities
- 40 edges to [[_COMMUNITY_pose.py (49+)]]
- 6 edges to [[_COMMUNITY_pipeline_paths.py (23+)]]
- 4 edges to [[_COMMUNITY_test_run_e2e_cloud_fallbacks.py (17+)]]
- 3 edges to [[_COMMUNITY_run_e2e_legacy.py (66+)]]
- 2 edges to [[_COMMUNITY_sncalib_dataset.py (18+)]]
- 1 edge to [[_COMMUNITY_reid_healer.py (37+)]]
- 1 edge to [[_COMMUNITY_cam_modules.py (43+)]]
- 1 edge to [[_COMMUNITY_main.py (50+)]]

## Top bridge nodes
- [[SoccerPitch]] - degree 107, connects to 4 communities
- [[.sample_field_points()]] - degree 10, connects to 2 communities
- [[move_camera()]] - degree 4, connects to 2 communities
- [[DynamicPitchCalibrator]] - degree 28, connects to 1 community
- [[Camera]] - degree 23, connects to 1 community