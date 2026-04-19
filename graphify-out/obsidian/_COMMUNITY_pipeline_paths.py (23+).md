---
type: community
cohesion: 0.05
members: 60
---

# pipeline_paths.py (23+)

**Cohesion:** 0.05 - loosely connected
**Members:** 60 nodes

## Members
- [[.__getitem__()_13]] - code - backend/references/sn-reid/torchreid/data/datasets/dataset.py
- [[.transform()]] - code - backend/references/soccersegcal/soccersegcal/pose.py
- [[Alias for func`sn_calibration_vendor_dir` (backward-compatible name for script]] - rationale - backend/services/pipeline_paths.py
- [[Build one tidy DataFrame for a single pickle (players only).]] - rationale - backend/scripts/auxiliary_tools/prepare_viz_data.py
- [[CLI verify local CV pipeline filesystem prerequisites; exit 1 if anything is mi]] - rationale - backend/scripts/auxiliary_tools/check_pipeline_prerequisites.py
- [[Download SoccerNet ReID dataset zips (and optionally unzip them).      Args]] - rationale - backend/scripts/auxiliary_tools/download_reid_data.py
- [[Download SoccerNet ReID dataset zips (trainvalidtestchallenge) using SoccerNe]] - rationale - backend/scripts/auxiliary_tools/download_reid_data.py
- [[Entry point for CLI execution.]] - rationale - backend/scripts/auxiliary_tools/download_reid_data.py
- [[Explain why homography JSON is missing when auto-calibration cannot finish the j]] - rationale - backend/services/pipeline_paths.py
- [[Extract downloaded ReID zip files into split-specific folders.      This follows]] - rationale - backend/scripts/auxiliary_tools/download_reid_data.py
- [[Meters on pitch, speed (kmh), outlier handling, smoothing, cumulative distance.]] - rationale - backend/scripts/auxiliary_tools/prepare_viz_data.py
- [[Parse CLI arguments for calibration batch generation.]] - rationale - backend/scripts/auxiliary_tools/generate_calibration.py
- [[Parse CLI arguments for the downloader script.]] - rationale - backend/scripts/auxiliary_tools/download_reid_data.py
- [[Path to the homography JSON used by TacticalRadar for this video.      If ``GAFF]] - rationale - backend/services/pipeline_paths.py
- [[Per match_id, player_id aggregates (small table).]] - rationale - backend/scripts/auxiliary_tools/prepare_viz_data.py
- [[Raise a clear error if the SoccerNet package is not installed.]] - rationale - backend/scripts/auxiliary_tools/download_reid_data.py
- [[Return a list of blocking issues before starting an expensive local CV job.]] - rationale - backend/services/pipeline_paths.py
- [[Return a single-line gap message if ``path`` is not a valid non-empty homography]] - rationale - backend/services/pipeline_paths.py
- [[Single source of truth for local CV  coaching pipeline filesystem layout.]] - rationale - backend/services/pipeline_paths.py
- [[SoccerNet pitch-segmentation weights (``soccer_pitch_segmentation.pth``, ``mean.]] - rationale - backend/services/pipeline_paths.py
- [[Vendored sn-calibration tree contains ``src`` so ``from src.`` imports resolv]] - rationale - backend/services/pipeline_paths.py
- [[YOLO weights for ``run_e2e_cloud``  ``track_teams``.      Default ``backendmo]] - rationale - backend/services/pipeline_paths.py
- [[_add_physics_and_clean()]] - code - backend/scripts/auxiliary_tools/prepare_viz_data.py
- [[_default_output_for_video()]] - code - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[_default_weights_dir()]] - code - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[_ensure_soccernet_available()]] - code - backend/scripts/auxiliary_tools/download_reid_data.py
- [[_extract_zip_files()]] - code - backend/scripts/auxiliary_tools/download_reid_data.py
- [[_load_match_tidy()]] - code - backend/scripts/auxiliary_tools/prepare_viz_data.py
- [[_parse_args()_1]] - code - backend/scripts/auxiliary_tools/download_reid_data.py
- [[_parse_args()]] - code - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[_summaries_from_frame()]] - code - backend/scripts/auxiliary_tools/prepare_viz_data.py
- [[_team_to_id()_1]] - code - backend/scripts/auxiliary_tools/prepare_viz_data.py
- [[check_pipeline_prerequisites.py]] - code - backend/scripts/auxiliary_tools/check_pipeline_prerequisites.py
- [[collect_local_cv_pipeline_gaps()]] - code - backend/services/pipeline_paths.py
- [[download_reid_data()]] - code - backend/scripts/auxiliary_tools/download_reid_data.py
- [[download_reid_data.py]] - code - backend/scripts/auxiliary_tools/download_reid_data.py
- [[ensure_homography_json_for_video()]] - code - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[format_homography_blocker_detail()]] - code - backend/services/pipeline_paths.py
- [[format_homography_missing_error()]] - code - backend/services/pipeline_paths.py
- [[format_pipeline_prerequisite_errors()]] - code - backend/services/pipeline_paths.py
- [[main()_20]] - code - backend/scripts/auxiliary_tools/check_pipeline_prerequisites.py
- [[main()_21]] - code - backend/scripts/auxiliary_tools/download_reid_data.py
- [[main()_18]] - code - backend/scripts/auxiliary_tools/prepare_viz_data.py
- [[main()_15]] - code - backend/scripts/auxiliary_tools/verify_sn_calibration.py
- [[output_dir()]] - code - backend/services/pipeline_paths.py
- [[parse_args()_3]] - code - backend/scripts/auxiliary_tools/generate_calibration.py
- [[pipeline_paths.py]] - code - backend/services/pipeline_paths.py
- [[pipeline_prerequisites()]] - code - backend/main.py
- [[prepare_viz_data.py]] - code - backend/scripts/auxiliary_tools/prepare_viz_data.py
- [[resolve_tracking_homography_json_path()]] - code - backend/services/pipeline_paths.py
- [[run()]] - code - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[run_calibrator_on_video.py]] - code - backend/scripts/pipeline_core/run_calibrator_on_video.py
- [[sn_calibration_resources_dir()]] - code - backend/services/pipeline_paths.py
- [[sn_calibration_root_dir()]] - code - backend/services/pipeline_paths.py
- [[sn_calibration_vendor_dir()]] - code - backend/services/pipeline_paths.py
- [[tactical_library_path()]] - code - backend/services/pipeline_paths.py
- [[tracking_model_weights_path()]] - code - backend/services/pipeline_paths.py
- [[uploads_dir()]] - code - backend/services/pipeline_paths.py
- [[validate_homography_json_file()]] - code - backend/services/pipeline_paths.py
- [[verify_sn_calibration.py]] - code - backend/scripts/auxiliary_tools/verify_sn_calibration.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/pipeline_paths.py_(23+)
SORT file.name ASC
```

## Connections to other communities
- 12 edges to [[_COMMUNITY_run_e2e_legacy.py (66+)]]
- 9 edges to [[_COMMUNITY_main.py (50+)]]
- 6 edges to [[_COMMUNITY_camera.py (42+)]]
- 5 edges to [[_COMMUNITY_pose.py (49+)]]
- 2 edges to [[_COMMUNITY_model_complexity.py (25+)]]
- 1 edge to [[_COMMUNITY_sampler.py (19+)]]
- 1 edge to [[_COMMUNITY_dataset.py (36+)]]
- 1 edge to [[_COMMUNITY_extract_tactical_library_from_pdfs.py (10+)]]

## Top bridge nodes
- [[parse_args()_3]] - degree 13, connects to 5 communities
- [[run()]] - degree 7, connects to 3 communities
- [[collect_local_cv_pipeline_gaps()]] - degree 13, connects to 2 communities
- [[ensure_homography_json_for_video()]] - degree 9, connects to 2 communities
- [[download_reid_data()]] - degree 7, connects to 2 communities