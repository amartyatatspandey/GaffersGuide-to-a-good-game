# Graph Report - backend/  (2026-04-19)

## Corpus Check
- Large corpus: 286 files · ~171,199 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder, or use --no-semantic to run AST-only.

## Summary
- 2447 nodes · 5085 edges · 71 communities detected
- Extraction: 66% EXTRACTED · 34% INFERRED · 0% AMBIGUOUS · INFERRED: 1746 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]

## God Nodes (most connected - your core abstractions)
1. `SoccerPitch` - 81 edges
2. `ChunkTacticalInsight` - 68 edges
3. `GlobalRefiner` - 68 edges
4. `TacticalAnalyzer` - 66 edges
5. `GeneratedPromptRecord` - 66 edges
6. `EngineRoutingError` - 62 edges
7. `TacticalLibrary` - 59 edges
8. `max()` - 45 edges
9. `ImageDataset` - 45 edges
10. `mkdir()` - 38 edges

## Surprising Connections (you probably didn't know these)
- `This function selects for each class present in the semantic mask, a set of circ` --uses--> `SoccerPitch`  [INFERRED]
  backend/references/soccersegcal/sncalib/detect_extremities.py → backend/references/soccersegcal/sncalib/soccerpitch.py
- `Given a list of points that were extracted from the blobs belonging to a same se` --uses--> `SoccerPitch`  [INFERRED]
  backend/references/soccersegcal/sncalib/detect_extremities.py → backend/references/soccersegcal/sncalib/soccerpitch.py
- `Given the dictionary {lines_class: points}, finds plausible extremities of each` --uses--> `SoccerPitch`  [INFERRED]
  backend/references/soccersegcal/sncalib/detect_extremities.py → backend/references/soccersegcal/sncalib/soccerpitch.py
- `Returns the barycenter of the True pixels under the area of the mask delimited b` --uses--> `SoccerPitch`  [INFERRED]
  backend/references/soccersegcal/sncalib/detect_extremities.py → backend/references/soccersegcal/sncalib/soccerpitch.py
- `Fits circles on the True pixels of the mask and returns those which have enough` --uses--> `SoccerPitch`  [INFERRED]
  backend/references/soccersegcal/sncalib/detect_extremities.py → backend/references/soccersegcal/sncalib/soccerpitch.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (120): draw_pitch_homography(), estimate_homography_from_line_correspondences(), normalization_transform(), Draws points along the soccer pitch markings elements in the image based on the, Computes the similarity transform such that the list of points is centered aroun, Given lines correspondences, computes the homography that maps best the two set, Camera, pan_tilt_roll_to_orientation() (+112 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (152): in_zone14(), Return Zone 14 bounds as (x_min, x_max, y_min, y_max) for attacking direction., zone14_bounds_for_team(), apply_ball_metrics_gate(), compute_ball_visibility_ratio(), Share of frames with valid ball coordinates (including interpolated points)., Gate ball-dependent metrics based on visibility confidence., _apply_ball_metrics_gate() (+144 more)

### Community 2 - "Community 2"
Cohesion: 0.03
Nodes (79): CUHK01, CUHK01.      Reference:         Li et al. Human Reidentification with Transferre, Image name format: 0001001.png, where first four digits represent identity, CUHK02, CUHK02.      Reference:         Li and Wang. Locally Aligned Feature Transforms, CUHK03, CUHK03.      Reference:         Li et al. DeepReID: Deep Filter Pairing Neural N, CUHKSYSU (+71 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (128): BaseModel, BetaJobRecord, BetaJobStore, Thread-safe JSON-backed persistent store for beta jobs., BetaPipelineQueue, BetaQueueItem, Queue-backed execution model for beta job isolation., CVRouterFactory (+120 more)

### Community 4 - "Community 4"
Cohesion: 0.02
Nodes (78): accuracy(), Computes the accuracy over the k top predictions for     the specified values of, AverageMeter, MetricMeter, A collection of metrics.      Source: https://github.com/KaiyangZhou/Dassl.pytor, Computes and stores the average and current value.      Examples::         >>> #, CrossEntropyLoss, Args:             inputs (torch.Tensor): prediction matrix (before softmax) with (+70 more)

### Community 5 - "Community 5"
Cohesion: 0.04
Nodes (88): _draw_annotated_frame(), _fallback_project_from_camera_shift(), _homography_confidence(), run_cv_tracking(), default_homography_provider(), HomographyProvider, Homography artifact resolution for tracking (isolates calibration policy from CV, Returns a validated homography JSON path for a given match video. (+80 more)

### Community 6 - "Community 6"
Cohesion: 0.02
Nodes (75): generate_class_synthesis(), get_line_extremities(), get_support_center(), join_points(), Returns the barycenter of the True pixels under the area of the mask delimited b, Fits circles on the True pixels of the mask and returns those which have enough, This function selects for each class present in the semantic mask, a set of circ, Process image and perform inference, returns mask of detected classes         :p (+67 more)

### Community 7 - "Community 7"
Cohesion: 0.03
Nodes (51): BasicConv2d, Block17, Block35, Block8, InceptionResNetV2, Mixed_5b, Mixed_6a, Mixed_7a (+43 more)

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (65): AdvancedPitchCalibrator, _build_residuals_factory(), _collect_corner_image_points(), _condition_ok(), _homogeneous_line_from_two_pixels(), _intersect_lines_homogeneous(), _line_image_from_pitch(), _longest_polylines_from_skeletons() (+57 more)

### Community 9 - "Community 9"
Cohesion: 0.04
Nodes (42): _coco_remove_images_without_annotations(), convert_coco_poly_to_mask(), ConvertCocoPolysToMask, FilterAndRemapCocoCategories, get_coco(), CustomNetwork, generate_class_synthesis(), get_line_extremities() (+34 more)

### Community 10 - "Community 10"
Cohesion: 0.04
Nodes (34): CameraParameterWLensDistDictZScore, get_fl_from_aov_rad(), Holds individual camera parameters including lens distortion parameters as nn.Mo, Projective camera defined as K @ R [I|-t] with lens distortion module and batch, # TODO: T! also need indivudual lens_dist_coeff for each t in T, Project world coordinates to pixel coordinates.          Args:             point, Project world coordinates to pixel coordinates.          Args:             point, Project world coordinates to pixel coordinates from the projection matrix. (+26 more)

### Community 11 - "Community 11"
Cohesion: 0.05
Nodes (54): _copy_artifact(), _homography_path_for_video(), _log_skip_homography(), main(), Run E2E for one video with GAFFERS_HOMOGRAPHY_JSON set to that match's calibrati, Per-match file: {video_stem}_homographies.json under GAFFERS_HOMOGRAPHY_DIR or o, Return (ok, reason). Requires readable JSON with non-empty homographies list, _run_single_video() (+46 more)

### Community 12 - "Community 12"
Cohesion: 0.06
Nodes (26): AvgPoolPad, BranchSeparables, BranchSeparablesReduction, BranchSeparablesStem, CellStem0, CellStem1, FirstCell, init_pretrained_weights() (+18 more)

### Community 13 - "Community 13"
Cohesion: 0.06
Nodes (26): ChannelGate, Conv1x1, Conv1x1Linear, Conv3x3, ConvLayer, init_pretrained_weights(), LightConv3x3, LightConvStream (+18 more)

### Community 14 - "Community 14"
Cohesion: 0.06
Nodes (34): HFlipDataset, download_reid_data(), _ensure_soccernet_available(), _extract_zip_files(), main(), _parse_args(), Download SoccerNet ReID dataset zips (train/valid/test/challenge) using SoccerNe, Parse CLI arguments for the downloader script. (+26 more)

### Community 15 - "Community 15"
Cohesion: 0.07
Nodes (23): computeConditionNumber(), isMatrixValid(), main(), evaluate(), main(), train_one_epoch(), cat_list(), collate_fn() (+15 more)

### Community 16 - "Community 16"
Cohesion: 0.06
Nodes (23): get_camera_from_per_sample_output(), InferenceDatasetCalibration, InferenceDatasetSegmentation, InferenceSegmentationModel, load_annotated_points(), prepare_per_sample(), _extract_keypoints_from_frame(), _load_segmentation_model() (+15 more)

### Community 17 - "Community 17"
Cohesion: 0.06
Nodes (29): SoccerNet sn-calibration vendor and weight directories., Vendored sn-calibration tree: contains ``src/`` so ``from src.*`` imports resolv, Alias for :func:`sn_calibration_vendor_dir` (backward-compatible name for script, SoccerNet pitch-segmentation weights (``soccer_pitch_segmentation.pth``, ``mean., sn_calibration_resources_dir(), sn_calibration_root_dir(), sn_calibration_vendor_dir(), main() (+21 more)

### Community 18 - "Community 18"
Cohesion: 0.08
Nodes (23): ChannelGate, Conv1x1, Conv1x1Linear, Conv3x3, ConvLayer, init_pretrained_weights(), LightConv3x3, OSBlock (+15 more)

### Community 19 - "Community 19"
Cohesion: 0.08
Nodes (18): ChannelAttn, ConvBlock, HACNN, HardAttn, HarmAttn, InceptionA, InceptionB, Basic convolutional block.          convolution + batch normalization + relu. (+10 more)

### Community 20 - "Community 20"
Cohesion: 0.06
Nodes (21): main(), Compute channel-wise mean and standard deviation of a dataset.  Usage: $ python, DataManager, ImageDataManager, r"""Base data manager.      Args:         sources (str or list): source dataset(, r"""Video data manager.      Args:         root (str): root path to datasets., Returns query and gallery of a test dataset, each containing         tuples of (, Transforms a PIL image to torch tensor for testing. (+13 more)

### Community 21 - "Community 21"
Cohesion: 0.09
Nodes (16): build_osnet_x0_25(), _ChannelGate, _Conv1x1, _Conv1x1Linear, _ConvLayer, _download_to_path(), _infer_num_classes(), _LightConv3x3 (+8 more)

### Community 22 - "Community 22"
Cohesion: 0.11
Nodes (22): Bottleneck, init_pretrained_weights(), Base class for bottlenecks that implements `forward()` method., Bottleneck for SENet154., ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe     imp, ResNeXt bottleneck type C with a Squeeze-and-Excitation module, Squeeze-and-excitation network.          Reference:         Hu et al. Squeeze-an, Parameters         ----------         block (nn.Module): Bottleneck class. (+14 more)

### Community 23 - "Community 23"
Cohesion: 0.11
Nodes (13): BasicConv2d, Inception_A, Inception_B, Inception_C, InceptionV4, init_pretrained_weights(), Mixed_3a, Mixed_4a (+5 more)

### Community 24 - "Community 24"
Cohesion: 0.1
Nodes (15): LineCollection, Wrapper class to represent lines by support and direction vectors.          Args, Abstract3dModel, Meshgrid, Static class variables that are specified by the rules of the game, Initialize 3D coordinates of all elements of the soccer pitch.         :param pi, Samples each pitch element every dist meters, returns a dictionary associating t, For lines belonging to the pitch lawn plane returns its 2D homogenous equation c (+7 more)

### Community 25 - "Community 25"
Cohesion: 0.1
Nodes (14): ConvBlock, ConvLayers, Fusion, MuDeep, MultiScaleA, MultiScaleB, Basic convolutional block.          convolution + batch normalization + relu., Saliency-based learning fusion layer (Sec.3.2) (+6 more)

### Community 26 - "Community 26"
Cohesion: 0.13
Nodes (19): BasicBlock, Bottleneck, conv1x1(), conv3x3(), init_pretrained_weights(), Code source: https://github.com/pytorch/vision, Residual network.          Reference:         - He et al. Deep Residual Learning, Constructs fully connected layer          Args:             fc_dims (list or tup (+11 more)

### Community 27 - "Community 27"
Cohesion: 0.11
Nodes (21): evaluate_timeline(), _evidence_text(), main(), Evaluate tactical rules cumulatively across the full metrics timeline.      Prod, Backwards-compatible frame-by-frame evaluator.          The production pipeline, Evaluate tactical metrics and produce chunk-level aggregated insights.      Retu, RuleEngine, run_engine() (+13 more)

### Community 28 - "Community 28"
Cohesion: 0.14
Nodes (16): _DenseBlock, _DenseLayer, DenseNet, densenet121(), densenet121_fc512(), densenet161(), densenet169(), densenet201() (+8 more)

### Community 29 - "Community 29"
Cohesion: 0.1
Nodes (19): compute_advanced_ball_metrics(), FrameLike, is_defensive_third(), _nearest_player_distance(), PlayerLike, Return positive forward progression for a team along radar X., Check whether ball is in a team's defensive third on 1050x680 radar., Euclidean distance from ball to nearest player on a team for one frame. (+11 more)

### Community 30 - "Community 30"
Cohesion: 0.12
Nodes (12): BasicBlock, Bottleneck, conv3x3(), IBN, init_pretrained_weights(), Credit to https://github.com/XingangPan/IBN-Net., Residual network + IBN layer.          Reference:         - He et al. Deep Resid, 3x3 convolution with padding (+4 more)

### Community 31 - "Community 31"
Cohesion: 0.09
Nodes (3): hook_batchnormNd(), hook_groupnorm(), hook_instancenormNd()

### Community 32 - "Community 32"
Cohesion: 0.13
Nodes (11): BasicBlock, Bottleneck, conv3x3(), DimReduceLayer, init_pretrained_weights(), PCB, pcb_p4(), pcb_p6() (+3 more)

### Community 33 - "Community 33"
Cohesion: 0.14
Nodes (10): BasicBlock, Bottleneck, conv3x3(), init_pretrained_weights(), Residual network + mid-level features.          Reference:         Yu et al. The, 3x3 convolution with padding, Constructs fully connected layer          Args:             fc_dims (list or tup, Initializes model with pretrained weights.          Layers that don't match with (+2 more)

### Community 34 - "Community 34"
Cohesion: 0.2
Nodes (12): channel_shuffle(), depthwise_conv(), init_pretrained_weights(), InvertedResidual, Code source: https://github.com/pytorch/vision, ShuffleNetV2.          Reference:         Ma et al. ShuffleNet V2: Practical Gui, Initializes model with pretrained weights.          Layers that don't match with, shufflenet_v2_x0_5() (+4 more)

### Community 35 - "Community 35"
Cohesion: 0.29
Nodes (2): ConvertHomography_Chen_to_SN, ConvertSN720p2SN540p

### Community 36 - "Community 36"
Cohesion: 0.29
Nodes (2): ConvertHomography_WC14GT_to_SN, ConvertHomographyJiang_to_WC14GT

### Community 37 - "Community 37"
Cohesion: 0.4
Nodes (5): _ensure_gemini_configured(), generate_coaching_advice(), Google Gemini helpers for tactical coaching completions., Configure the Gemini client once per process., Call Gemini with the assembled RAG prompt and return plain text coaching advice.

### Community 38 - "Community 38"
Cohesion: 0.33
Nodes (1): numpy_include()

### Community 39 - "Community 39"
Cohesion: 0.4
Nodes (0): 

### Community 40 - "Community 40"
Cohesion: 0.4
Nodes (4): estimate_homography_from_line_correspondences(), normalization_transform(), Computes the similarity transform such that the list of points is centered aroun, Given lines correspondences, computes the homography that maps best the two set

### Community 41 - "Community 41"
Cohesion: 0.5
Nodes (1): ConvertHomography_Chen_to_SN

### Community 42 - "Community 42"
Cohesion: 0.5
Nodes (1): ConvertHomography_WC14GT_to_SN

### Community 43 - "Community 43"
Cohesion: 0.5
Nodes (1): TacticalPhysicsFilter

### Community 44 - "Community 44"
Cohesion: 0.67
Nodes (1): Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective

### Community 45 - "Community 45"
Cohesion: 0.67
Nodes (0): 

### Community 46 - "Community 46"
Cohesion: 0.67
Nodes (0): 

### Community 47 - "Community 47"
Cohesion: 0.67
Nodes (1): Lightweight checks for Rule 1 P2 (generate_calibration import hygiene).

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Single source of truth for local CV / coaching pipeline filesystem layout.  Impl

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (0): 

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (0): 

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (0): 

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (0): 

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (0): 

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (0): 

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (0): 

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (0): 

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (0): 

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (0): 

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (0): 

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (0): 

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (0): 

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (0): 

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (0): 

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (0): 

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (0): 

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (0): 

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (0): 

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Returns the number of training person identities.

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): Returns the number of training cameras.

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **316 isolated node(s):** `Aggregated tactical flaw over an entire video chunk (macro-trend).      frequenc`, `Job tracking for asynchronous video processing.`, `One dataset folder under DATASETS_ROOT (optional API for tooling UIs).`, `Google Gemini helpers for tactical coaching completions.`, `Configure the Gemini client once per process.` (+311 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 48`** (2 nodes): `pipeline_paths.py`, `Single source of truth for local CV / coaching pipeline filesystem layout.  Impl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `masks_pred2chen.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `optimize.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `fuse_stack.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `fuse_argmin.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `generate_table_wc14-center.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `prepare_iou_results.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `generate_table_sncalib-center.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `generate_table_lens_distortion.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `visualize_initialization_static.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `summarize_results_sncalib-test-all.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `visualize_ndc_losses_multiple_datasets.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `setup.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `evaluate_soccernetv3_reid.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `conf.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `test_cython.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `Returns the number of training person identities.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `Returns the number of training cameras.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `e2e_shared.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_model()` connect `Community 6` to `Community 2`, `Community 7`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Why does `Model` connect `Community 5` to `Community 0`, `Community 4`, `Community 6`, `Community 9`, `Community 14`, `Community 15`, `Community 16`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `mkdir()` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 8`, `Community 9`, `Community 11`, `Community 14`, `Community 15`, `Community 21`, `Community 27`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Are the 73 inferred relationships involving `SoccerPitch` (e.g. with `Adapter around vendored sn-calibration SoccerPitch and helpers.` and `Computes the similarity transform such that the list of points is centered aroun`) actually correct?**
  _`SoccerPitch` has 73 INFERRED edges - model-reasoned connections that need verification._
- **Are the 65 inferred relationships involving `ChunkTacticalInsight` (e.g. with `TacticalPlayer` and `TacticalFrame`) actually correct?**
  _`ChunkTacticalInsight` has 65 INFERRED edges - model-reasoned connections that need verification._
- **Are the 61 inferred relationships involving `GlobalRefiner` (e.g. with `DownscaledOpticalFlowEstimator` and `Sparse optical-flow estimator on downscaled grayscale frames.     Returns camera`) actually correct?**
  _`GlobalRefiner` has 61 INFERRED edges - model-reasoned connections that need verification._
- **Are the 58 inferred relationships involving `TacticalAnalyzer` (e.g. with `TacticalPlayer` and `TacticalFrame`) actually correct?**
  _`TacticalAnalyzer` has 58 INFERRED edges - model-reasoned connections that need verification._