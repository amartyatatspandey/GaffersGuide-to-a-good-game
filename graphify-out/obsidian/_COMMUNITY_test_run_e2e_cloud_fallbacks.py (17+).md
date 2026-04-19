---
type: community
cohesion: 0.06
members: 50
---

# test_run_e2e_cloud_fallbacks.py (17+)

**Cohesion:** 0.06 - loosely connected
**Members:** 50 nodes

## Members
- [[NOTE We intentionally avoid exporting raw extremity points for moving target]] - rationale - backend/references/process_batch.py
- [[.__init__()_5]] - code - backend/references/tvcalib/sn_segmentation/src/custom_extremities.py
- [[.__init__()_207]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.__init__()_27]] - code - backend/references/tvcalib/tvcalib/inference.py
- [[.__init__()_212]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[.__init__()_213]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[.__init__()_214]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[.__init__()_215]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[.empty_cache()]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[.empty_cache()_1]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[.forward()]] - code - backend/references/tvcalib/sn_segmentation/src/custom_extremities.py
- [[.inference()]] - code - backend/references/tvcalib/tvcalib/inference.py
- [[.is_available()]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[.is_available()_1]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[CustomNetwork]] - code - backend/references/tvcalib/sn_segmentation/src/custom_extremities.py
- [[Ensure TVCalib + its submodules are importable.]] - rationale - backend/references/process_batch.py
- [[Fits circles on the True pixels of the mask and returns those which have enough_1]] - rationale - backend/references/tvcalib/sn_segmentation/src/custom_extremities.py
- [[Given a list of points that were extracted from the blobs belonging to a same se_1]] - rationale - backend/references/tvcalib/sn_segmentation/src/custom_extremities.py
- [[Given the dictionary {lines_class points}, finds plausible extremities of each_1]] - rationale - backend/references/tvcalib/sn_segmentation/src/custom_extremities.py
- [[InferenceSegmentationModel]] - code - backend/references/tvcalib/tvcalib/inference.py
- [[Load TVCalib segmentation model.]] - rationale - backend/references/process_batch.py
- [[Returns the barycenter of the True pixels under the area of the mask delimited b_1]] - rationale - backend/references/tvcalib/sn_segmentation/src/custom_extremities.py
- [[Run TVCalib segmentation + point selection, then return semantic intersection ke]] - rationale - backend/references/process_batch.py
- [[Run keypoint extraction on `match_video.mp4` every 10th frame.]] - rationale - backend/references/process_batch.py
- [[TVCalib-based keypoint extraction for camera calibration.  Outputs `calibration_]] - rationale - backend/references/process_batch.py
- [[This function selects for each class present in the semantic mask, a set of circ_1]] - rationale - backend/references/tvcalib/sn_segmentation/src/custom_extremities.py
- [[_DummyCuda]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[_DummyMpsBackend]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[_DummyMpsDevice]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[_DummyTorch]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[_clear_device_cache()]] - code - backend/scripts/pipeline_core/run_e2e_cloud.py
- [[_extract_keypoints_from_frame()]] - code - backend/references/process_batch.py
- [[_infer_device()]] - code - backend/scripts/pipeline_core/run_e2e_cloud.py
- [[_load_segmentation_model()]] - code - backend/references/process_batch.py
- [[_setup_import_paths()]] - code - backend/references/process_batch.py
- [[custom_extremities.py]] - code - backend/references/tvcalib/sn_segmentation/src/custom_extremities.py
- [[generate_class_synthesis()_1]] - code - backend/references/tvcalib/sn_segmentation/src/custom_extremities.py
- [[get_line_extremities()_1]] - code - backend/references/tvcalib/sn_segmentation/src/custom_extremities.py
- [[get_support_center()_1]] - code - backend/references/tvcalib/sn_segmentation/src/custom_extremities.py
- [[join_points()_1]] - code - backend/references/tvcalib/sn_segmentation/src/custom_extremities.py
- [[load_checkpoint()]] - code - backend/references/sn-reid/torchreid/utils/torchtools.py
- [[main()]] - code - backend/references/process_batch.py
- [[process_batch.py]] - code - backend/references/process_batch.py
- [[rLoads checkpoint.      ``UnicodeDecodeError`` can be well handled, which mea]] - rationale - backend/references/sn-reid/torchreid/utils/torchtools.py
- [[synthesize_mask()_1]] - code - backend/references/tvcalib/sn_segmentation/src/custom_extremities.py
- [[test_clear_device_cache_safe_on_cpu()]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[test_infer_device_falls_back_to_cpu_when_no_torch()]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[test_infer_device_prefers_mps_without_cuda()]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[test_run_e2e_cloud_fallbacks.py]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py
- [[test_wrapper_run_e2e_delegates_to_cloud()]] - code - backend/tests/test_run_e2e_cloud_fallbacks.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/test_run_e2e_cloud_fallbacks.py_(17+)
SORT file.name ASC
```

## Connections to other communities
- 8 edges to [[_COMMUNITY_run_e2e_legacy.py (66+)]]
- 5 edges to [[_COMMUNITY_model_complexity.py (25+)]]
- 4 edges to [[_COMMUNITY_camera.py (42+)]]
- 3 edges to [[_COMMUNITY_pose.py (49+)]]
- 2 edges to [[_COMMUNITY_transforms.py (47+)]]
- 2 edges to [[_COMMUNITY_reid_healer.py (37+)]]
- 2 edges to [[_COMMUNITY_utils.py (36+)]]
- 1 edge to [[_COMMUNITY_cam_modules.py (43+)]]
- 1 edge to [[_COMMUNITY_sncalib_dataset.py (18+)]]
- 1 edge to [[_COMMUNITY_sampler.py (19+)]]
- 1 edge to [[_COMMUNITY_engine.py (23+)]]

## Top bridge nodes
- [[.is_available()_1]] - degree 18, connects to 7 communities
- [[_extract_keypoints_from_frame()]] - degree 9, connects to 3 communities
- [[.forward()]] - degree 4, connects to 3 communities
- [[InferenceSegmentationModel]] - degree 11, connects to 2 communities
- [[_clear_device_cache()]] - degree 5, connects to 1 community