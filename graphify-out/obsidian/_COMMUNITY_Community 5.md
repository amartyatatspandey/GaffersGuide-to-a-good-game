---
type: community
cohesion: 0.04
members: 123
---

# Community 5

**Cohesion:** 0.04 - loosely connected
**Members:** 123 nodes

## Members
- [[.__init__()_230]] - code - backend/scripts/pipeline_core/run_e2e_cloud.py
- [[.__init__()_229]] - code - backend/scripts/pipeline_core/tactical_radar.py
- [[.__init__()_228]] - code - backend/scripts/pipeline_core/team_classifier.py
- [[.__init__()_218]] - code - backend/scripts/pipeline_core/track_teams_reid_hybrid.py
- [[._build_feature_mask()]] - code - backend/scripts/pipeline_core/run_e2e_cloud.py
- [[._detect_features()]] - code - backend/scripts/pipeline_core/run_e2e_cloud.py
- [[._extract_base_anchors()]] - code - backend/scripts/pipeline_core/team_classifier.py
- [[._hunt_for_gk_anchors()]] - code - backend/scripts/pipeline_core/team_classifier.py
- [[._resolve_heal_chain()]] - code - backend/scripts/pipeline_core/track_teams_reid_hybrid.py
- [[._to_gray_downscaled()]] - code - backend/scripts/pipeline_core/run_e2e_cloud.py
- [[.cleanup_ghost_ids()]] - code - backend/scripts/pipeline_core/track_teams_reid_hybrid.py
- [[.draw_blank_pitch()]] - code - backend/scripts/pipeline_core/tactical_radar.py
- [[.ensure_homography_json()]] - code - backend/services/homography_provider.py
- [[.extract_features()_3]] - code - backend/scripts/pipeline_core/reid_healer.py
- [[.extract_features()_2]] - code - backend/references/sn-reid/torchreid/engine/video/softmax.py
- [[.extract_features()_1]] - code - backend/references/sn-reid/torchreid/engine/video/triplet.py
- [[.forward()_4]] - code - backend/references/soccersegcal/soccersegcal/pose.py
- [[.get_dominant_color()]] - code - backend/scripts/pipeline_core/team_classifier.py
- [[.map_many_to_2d()]] - code - backend/scripts/pipeline_core/tactical_radar.py
- [[.map_to_2d()]] - code - backend/scripts/pipeline_core/tactical_radar.py
- [[.predict_frame()]] - code - backend/scripts/pipeline_core/team_classifier.py
- [[.process_and_heal()]] - code - backend/scripts/pipeline_core/track_teams_reid_hybrid.py
- [[.update()_5]] - code - backend/scripts/pipeline_core/run_e2e_cloud.py
- [[.update_camera_angle()]] - code - backend/scripts/pipeline_core/tactical_radar.py
- [[.write()_1]] - code - backend/references/sn-reid/torchreid/utils/loggers.py
- [[CoordsTelemetry]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[Cosine similarity between two vectors (e.g. 512-D ReID embeddings).]] - rationale - backend/scripts/pipeline_core/track_teams_constants.py
- [[Crop ``bbox`` (x1, y1, x2, y2), resize to ReID input, return 512-D vector.]] - rationale - backend/scripts/pipeline_core/reid_healer.py
- [[DownscaledOpticalFlowEstimator]] - code - backend/scripts/pipeline_core/run_e2e_cloud.py
- [[Draw bounding boxes and labels on frame from YOLO results.      Args         fr]] - rationale - backend/scripts/auxiliary_tools/validate_adit_model.py
- [[Draw boxes and labels with per-detection colors and labels.]] - rationale - backend/scripts/pipeline_core/track_teams.py
- [[Drop stale ReID  radar state for IDs not seen for 300 frames (~12 s @ 25 fps).]] - rationale - backend/scripts/pipeline_core/track_teams_reid_hybrid.py
- [[Dynamic Anchor architecture base anchors (teams + ref) at frame 200, lazy-loade]] - rationale - backend/scripts/pipeline_core/team_classifier.py
- [[Extract dominant jersey color from center-top of bbox (chest) in HSV.         Cr]] - rationale - backend/scripts/pipeline_core/team_classifier.py
- [[Extract exact color profiles for the two teams and referee at kickoff. Lock ref]] - rationale - backend/scripts/pipeline_core/team_classifier.py
- [[Fill short missing ball tracks via linear interpolation, then backfill possessio]] - rationale - backend/calculators/possession.py
- [[Firewall between ByteTrack IDs and reid_healer optional ReID + radar distance c]] - rationale - backend/scripts/pipeline_core/track_teams_reid_hybrid.py
- [[Geographical Hunt Scans Defensive Thirds with Spatial Threshold Override.]] - rationale - backend/scripts/pipeline_core/team_classifier.py
- [[Homography artifact resolution for tracking (isolates calibration policy from CV]] - rationale - backend/services/homography_provider.py
- [[HomographyProvider]] - code - backend/services/homography_provider.py
- [[Human-readable hint when ``path`` is not a readable weights file.]] - rationale - backend/services/paths/models.py
- [[HybridIDHealer]] - code - backend/scripts/pipeline_core/track_teams_reid_hybrid.py
- [[Lazy import so importing ``services`` stays light when calibration is unused.]] - rationale - backend/services/homography_provider.py
- [[Load model, process video, and generate validation output.]] - rationale - backend/scripts/auxiliary_tools/validate_adit_model.py
- [[Load model, run YOLO + ByteTrack + team classification, write output video.]] - rationale - backend/scripts/pipeline_core/track_teams.py
- [[Load the pre-trained YOLO model from the specified path.      Args         mode]] - rationale - backend/scripts/auxiliary_tools/validate_adit_model.py
- [[Loads OSNet x0.25 (MSMT17) and extracts L2-normalized 512-D embeddings from play]] - rationale - backend/scripts/pipeline_core/reid_healer.py
- [[Map Global Draft role string to annotation payload -1, (team_id, is_gk), or Non]] - rationale - backend/scripts/pipeline_core/track_teams.py
- [[Model]] - code - backend/references/soccersegcal/soccersegcal/pose.py
- [[Optional ReID + radar ID healing (extracted from track_teams for Rule 4).]] - rationale - backend/scripts/pipeline_core/track_teams_reid_hybrid.py
- [[Optionally rewrites tracker IDs on ``detections`` using ReID + radar proximity.]] - rationale - backend/scripts/pipeline_core/track_teams_reid_hybrid.py
- [[Panning-aware 2D tactical map using dynamic homographies.     Unified Architectu]] - rationale - backend/scripts/pipeline_core/tactical_radar.py
- [[Process the entire frame at once; global draft guarantees at most one tracker_id]] - rationale - backend/scripts/pipeline_core/team_classifier.py
- [[Process video, run inference, visualize detections, and save output video.]] - rationale - backend/scripts/auxiliary_tools/validate_adit_model.py
- [[Resolve and auto-generate TacticalRadar homography JSON (services-owned policy).]] - rationale - backend/services/homography_resolution.py
- [[Resolve homography JSON for ``video_path`` (env override or per-stem under ``out_1]] - rationale - backend/services/homography_resolution.py
- [[Return BGR color Team 01 for players, gray Unknown, yellow GK, default BallRe]] - rationale - backend/scripts/pipeline_core/track_teams.py
- [[Return label string; for players optionally include ID and team (e.g. ID3 T0-GK]] - rationale - backend/scripts/pipeline_core/track_teams.py
- [[Returns a validated homography JSON path for a given match video.]] - rationale - backend/services/homography_provider.py
- [[Run CV tracking without video encoding, drawing, or optical-flow fallback.]] - rationale - backend/scripts/auxiliary_tools/run_coords_only.py
- [[Shared constants and small math helpers for team tracking (Rule 4 modularization]] - rationale - backend/scripts/pipeline_core/track_teams_constants.py
- [[Sparse optical-flow estimator on downscaled grayscale frames.     Returns camera]] - rationale - backend/scripts/pipeline_core/run_e2e_cloud.py
- [[Tactical radar projection module extracted from track_teams.]] - rationale - backend/scripts/pipeline_core/tactical_radar.py
- [[TacticalFrame_2]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[TacticalPlayer_2]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[TacticalRadar]] - code - backend/scripts/pipeline_core/tactical_radar.py
- [[Team classification module extracted from track_teams.]] - rationale - backend/scripts/pipeline_core/team_classifier.py
- [[Team-aware tracking YOLO + ByteTrack + team classification with soft lock and g]] - rationale - backend/scripts/pipeline_core/track_teams.py
- [[TeamClassifier]] - code - backend/scripts/pipeline_core/team_classifier.py
- [[Validate Adit Jain's pre-trained model on match_test.mp4.  Runs inference to ver]] - rationale - backend/scripts/auxiliary_tools/validate_adit_model.py
- [[Vectorized projection for many bboxes in one frame.          Safety guarantees]] - rationale - backend/scripts/pipeline_core/tactical_radar.py
- [[VisualFingerprint]] - code - backend/scripts/pipeline_core/reid_healer.py
- [[Writes result.          Args            name (str) dataset name.            ep]] - rationale - backend/references/sn-reid/torchreid/utils/loggers.py
- [[_default_homography_dir()]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[_default_output_dir()]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[_draw_annotated_frame()]] - code - backend/scripts/pipeline_core/e2e_shared_impl.py
- [[_draw_annotated_frame()_1]] - code - backend/scripts/pipeline_core/legacy/run_e2e_legacy.py
- [[_fallback_project_from_camera_shift()]] - code - backend/scripts/pipeline_core/e2e_shared_impl.py
- [[_fallback_project_from_camera_shift()_1]] - code - backend/scripts/pipeline_core/legacy/run_e2e_legacy.py
- [[_homography_confidence()]] - code - backend/scripts/pipeline_core/e2e_shared_impl.py
- [[_homography_confidence()_2]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[_homography_confidence()_1]] - code - backend/scripts/pipeline_core/legacy/run_e2e_legacy.py
- [[_iter_frame_batches()]] - code - backend/scripts/pipeline_core/run_e2e_cloud.py
- [[_prediction_to_team()_2]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[_process_one_video()]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[_resolve_ball_classes()_2]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[_resolve_primary_ball_class_ids()_2]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[_serialize_run()]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[_validate_homography_json()]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[annotate_frame()]] - code - backend/scripts/pipeline_core/track_teams.py
- [[compute_possession_team_id()]] - code - backend/calculators/possession.py
- [[cosine_similarity()_1]] - code - backend/scripts/pipeline_core/track_teams_constants.py
- [[default_homography_provider()]] - code - backend/services/homography_provider.py
- [[draw_detections()]] - code - backend/scripts/auxiliary_tools/validate_adit_model.py
- [[ensure_homography_json_for_video()_1]] - code - backend/services/homography_resolution.py
- [[format_tracking_model_missing_reason()]] - code - backend/services/paths/models.py
- [[get_detection_color()]] - code - backend/scripts/pipeline_core/track_teams.py
- [[get_detection_label()]] - code - backend/scripts/pipeline_core/track_teams.py
- [[homography_provider.py]] - code - backend/services/homography_provider.py
- [[homography_resolution.py]] - code - backend/services/homography_resolution.py
- [[interpolate_ball_positions()]] - code - backend/calculators/possession.py
- [[load_model()]] - code - backend/scripts/auxiliary_tools/validate_adit_model.py
- [[main()_14]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[main()_11]] - code - backend/scripts/pipeline_core/run_e2e_cloud.py
- [[main()_10]] - code - backend/scripts/pipeline_core/track_teams.py
- [[main()_20]] - code - backend/scripts/auxiliary_tools/validate_adit_model.py
- [[max()]] - code - backend/references/tvcalib/sn_segmentation/src/segmentation/utils.py
- [[median()]] - code - backend/references/tvcalib/sn_segmentation/src/segmentation/utils.py
- [[parse_args()_1]] - code - backend/scripts/pipeline_core/run_e2e_cloud.py
- [[process_video()]] - code - backend/scripts/auxiliary_tools/validate_adit_model.py
- [[role_to_payload()]] - code - backend/scripts/pipeline_core/track_teams.py
- [[run_coords_only()]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[run_coords_only.py]] - code - backend/scripts/auxiliary_tools/run_coords_only.py
- [[run_cv_tracking()]] - code - backend/scripts/pipeline_core/e2e_shared_impl.py
- [[run_cv_tracking()_1]] - code - backend/scripts/pipeline_core/legacy/run_e2e_legacy.py
- [[run_cv_tracking_batched()]] - code - backend/scripts/pipeline_core/run_e2e_cloud.py
- [[run_e2e_cloud.py]] - code - backend/scripts/pipeline_core/run_e2e_cloud.py
- [[tactical_radar.py]] - code - backend/scripts/pipeline_core/tactical_radar.py
- [[team_classifier.py]] - code - backend/scripts/pipeline_core/team_classifier.py
- [[track_teams.py]] - code - backend/scripts/pipeline_core/track_teams.py
- [[track_teams_constants.py]] - code - backend/scripts/pipeline_core/track_teams_constants.py
- [[track_teams_reid_hybrid.py]] - code - backend/scripts/pipeline_core/track_teams_reid_hybrid.py
- [[validate_adit_model.py]] - code - backend/scripts/auxiliary_tools/validate_adit_model.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Community_5
SORT file.name ASC
```

## Connections to other communities
- 60 edges to [[_COMMUNITY_Community 1]]
- 15 edges to [[_COMMUNITY_Community 3]]
- 12 edges to [[_COMMUNITY_Community 0]]
- 10 edges to [[_COMMUNITY_Community 6]]
- 10 edges to [[_COMMUNITY_Community 4]]
- 6 edges to [[_COMMUNITY_Community 17]]
- 5 edges to [[_COMMUNITY_Community 15]]
- 5 edges to [[_COMMUNITY_Community 21]]
- 4 edges to [[_COMMUNITY_Community 29]]
- 2 edges to [[_COMMUNITY_Community 9]]
- 2 edges to [[_COMMUNITY_Community 16]]
- 2 edges to [[_COMMUNITY_Community 11]]
- 2 edges to [[_COMMUNITY_Community 27]]
- 2 edges to [[_COMMUNITY_Community 14]]
- 1 edge to [[_COMMUNITY_Community 8]]

## Top bridge nodes
- [[max()]] - degree 45, connects to 10 communities
- [[Model]] - degree 28, connects to 7 communities
- [[run_cv_tracking_batched()]] - degree 31, connects to 3 communities
- [[.write()_1]] - degree 10, connects to 3 communities
- [[run_cv_tracking()]] - degree 31, connects to 2 communities