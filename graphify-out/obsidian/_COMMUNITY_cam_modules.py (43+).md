---
type: community
cohesion: 0.04
members: 83
---

# cam_modules.py (43+)

**Cohesion:** 0.04 - loosely connected
**Members:** 83 nodes

## Members
- [[TODO T! also need indivudual lens_dist_coeff for each t in T]] - rationale - backend/references/tvcalib/tvcalib/cam_modules.py
- [[TODO modify later to dynamically cunstruct a tensor of shape (k_1,k_2,p_1,p_2]] - rationale - backend/references/tvcalib/tvcalib/cam_modules.py
- [[TODO verify]] - rationale - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.__init__()_25]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.__init__()_26]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.__init__()_31]] - code - backend/references/tvcalib/tvcalib/utils/data_distr.py
- [[.__init__()_30]] - code - backend/references/tvcalib/tvcalib/module.py
- [[.__len__()_7]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.__repr__()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.__str__()_3]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.construct_intrinsics_ndc()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.construct_intrinsics_raster()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.construct_projection_matrix()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.denormalize()]] - code - backend/references/tvcalib/tvcalib/utils/data_distr.py
- [[.distort_points()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.forward()_1]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.forward()_3]] - code - backend/references/tvcalib/tvcalib/utils/data_distr.py
- [[.forward()_2]] - code - backend/references/tvcalib/tvcalib/module.py
- [[.get_homography_raster()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.get_parameters()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.get_rays_world()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.initialize()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.project_point2ndc()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.project_point2ndc_from_P()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.project_point2pixel()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.project_point2pixel_from_P()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.rotation_from_euler_angles()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.self_optim_batch()]] - code - backend/references/tvcalib/tvcalib/module.py
- [[.str_lens_distortion_coeff()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.str_pan_tilt_roll_fl()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.undistort_images()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[.undistort_points()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[Args             z (Tensor) tensor of size (B, ) to be denormalized.]] - rationale - backend/references/tvcalib/tvcalib/utils/data_distr.py
- [[Batched version for point-pointcloud distance calculation     Args         poin]] - rationale - backend/references/tvcalib/tvcalib/utils/linalg.py
- [[CameraParameterWLensDistDictZScore]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[Compensate for lens distortion a set of 2D image points.          Wrapper for ko]] - rationale - backend/references/tvcalib/tvcalib/cam_modules.py
- [[Computes mean and std given min,max values with respect a confidence interval (s]] - rationale - backend/references/tvcalib/tvcalib/utils/data_distr.py
- [[Distortion of a set of 2D points based on the lens distortion model.          Wr]] - rationale - backend/references/tvcalib/tvcalib/cam_modules.py
- [[FeatureScalerZScore]] - code - backend/references/tvcalib/tvcalib/utils/data_distr.py
- [[Get dict of relevant camera parameters and homography matrix         return Th]] - rationale - backend/references/tvcalib/tvcalib/cam_modules.py
- [[Holds individual camera parameters including lens distortion parameters as nn.Mo]] - rationale - backend/references/tvcalib/tvcalib/cam_modules.py
- [[Initializes all camera parameters with zeros and replace specific values with pr]] - rationale - backend/references/tvcalib/tvcalib/cam_modules.py
- [[LightningModule]] - code
- [[Line to point cloud distance with arbitrary leading dimensions.      TODO. if cr]] - rationale - backend/references/tvcalib/tvcalib/utils/linalg.py
- [[Project world coordinates to pixel coordinates from the projection matrix.]] - rationale - backend/references/tvcalib/tvcalib/cam_modules.py
- [[Project world coordinates to pixel coordinates from the projection matrix._1]] - rationale - backend/references/tvcalib/tvcalib/cam_modules.py
- [[Project world coordinates to pixel coordinates.          Args             point]] - rationale - backend/references/tvcalib/tvcalib/cam_modules.py
- [[Project world coordinates to pixel coordinates.          Args             point_1]] - rationale - backend/references/tvcalib/tvcalib/cam_modules.py
- [[Projective camera defined as K @ R I-t with lens distortion module and batch]] - rationale - backend/references/tvcalib/tvcalib/cam_modules.py
- [[SNProjectiveCamera]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[TVCalibModule]] - code - backend/references/tvcalib/tvcalib/module.py
- [[_summary_          Args             x (_type_) x of shape (B, 3, N)          R]] - rationale - backend/references/tvcalib/tvcalib/cam_modules.py
- [[cam_modules.py]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[data_distr.py]] - code - backend/references/tvcalib/tvcalib/utils/data_distr.py
- [[distance_line_pointcloud_3d()]] - code - backend/references/tvcalib/tvcalib/utils/linalg.py
- [[distance_point_pointcloud()]] - code - backend/references/tvcalib/tvcalib/utils/linalg.py
- [[draw_image()]] - code - backend/references/tvcalib/tvcalib/utils/visualization_mpl_min.py
- [[draw_reprojection()]] - code - backend/references/tvcalib/tvcalib/utils/visualization_mpl_min.py
- [[draw_selected_points()]] - code - backend/references/tvcalib/tvcalib/utils/visualization_mpl_min.py
- [[frame_image()]] - code - backend/references/tvcalib/tvcalib/utils/visualization_mpl_min.py
- [[get_aov_rad()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[get_cam_distr()_3]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_behind.py
- [[get_cam_distr()_2]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_center.py
- [[get_cam_distr()_4]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_left.py
- [[get_cam_distr()]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_right.py
- [[get_cam_distr()_1]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_tribune.py
- [[get_dist_distr()_3]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_behind.py
- [[get_dist_distr()_2]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_center.py
- [[get_dist_distr()_4]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_left.py
- [[get_dist_distr()]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_right.py
- [[get_dist_distr()_1]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_tribune.py
- [[get_fl_from_aov_rad()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[init_figure()]] - code - backend/references/tvcalib/tvcalib/utils/visualization_mpl_min.py
- [[linalg.py]] - code - backend/references/tvcalib/tvcalib/utils/linalg.py
- [[mean_std_with_confidence_interval()]] - code - backend/references/tvcalib/tvcalib/utils/data_distr.py
- [[module.py]] - code - backend/references/tvcalib/tvcalib/module.py
- [[static_undistort_points()]] - code - backend/references/tvcalib/tvcalib/cam_modules.py
- [[tv_main_behind.py]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_behind.py
- [[tv_main_center.py]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_center.py
- [[tv_main_left.py]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_left.py
- [[tv_main_right.py]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_right.py
- [[tv_main_tribune.py]] - code - backend/references/tvcalib/tvcalib/cam_distr/tv_main_tribune.py
- [[visualization_mpl_min.py]] - code - backend/references/tvcalib/tvcalib/utils/visualization_mpl_min.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/cam_modules.py_(43+)
SORT file.name ASC
```

## Connections to other communities
- 3 edges to [[_COMMUNITY_sncalib_dataset.py (18+)]]
- 1 edge to [[_COMMUNITY_camera.py (42+)]]
- 1 edge to [[_COMMUNITY_test_run_e2e_cloud_fallbacks.py (17+)]]
- 1 edge to [[_COMMUNITY_model_complexity.py (25+)]]
- 1 edge to [[_COMMUNITY_engine.py (23+)]]
- 1 edge to [[_COMMUNITY_transforms.py (47+)]]
- 1 edge to [[_COMMUNITY_objects_3d.py (26+)]]

## Top bridge nodes
- [[SNProjectiveCamera]] - degree 28, connects to 2 communities
- [[FeatureScalerZScore]] - degree 22, connects to 1 community
- [[linalg.py]] - degree 3, connects to 1 community
- [[.__init__()_30]] - degree 3, connects to 1 community
- [[.self_optim_batch()]] - degree 3, connects to 1 community