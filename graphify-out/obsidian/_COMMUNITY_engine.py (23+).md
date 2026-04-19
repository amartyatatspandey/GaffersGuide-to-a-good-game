---
type: community
cohesion: 0.03
members: 100
---

# engine.py (23+)

**Cohesion:** 0.03 - loosely connected
**Members:** 100 nodes

## Members
- [[.__init__()_65]] - code - backend/references/sn-reid/torchreid/utils/avgmeter.py
- [[.__init__()_59]] - code - backend/references/sn-reid/torchreid/losses/cross_entropy_loss.py
- [[.__init__()_58]] - code - backend/references/sn-reid/torchreid/losses/hard_mine_triplet_loss.py
- [[.__init__()_211]] - code - backend/references/sn-reid/torchreid/engine/image/softmax.py
- [[.__init__()_209]] - code - backend/references/sn-reid/torchreid/engine/video/softmax.py
- [[.__init__()_53]] - code - backend/references/soccersegcal/soccersegcal/train.py
- [[.__init__()_210]] - code - backend/references/sn-reid/torchreid/engine/image/triplet.py
- [[.__init__()_208]] - code - backend/references/sn-reid/torchreid/engine/video/triplet.py
- [[.compute_loss()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.export_ranking_results_for_ext_eval()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.extract_features()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.forward()_10]] - code - backend/references/sn-reid/torchreid/losses/cross_entropy_loss.py
- [[.forward()_9]] - code - backend/references/sn-reid/torchreid/losses/hard_mine_triplet_loss.py
- [[.forward_backward()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.forward_backward()_2]] - code - backend/references/sn-reid/torchreid/engine/image/softmax.py
- [[.forward_backward()_1]] - code - backend/references/sn-reid/torchreid/engine/image/triplet.py
- [[.get_current_lr()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.get_model_names()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.parse_data_for_eval()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.parse_data_for_train()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.parse_data_for_train()_2]] - code - backend/references/sn-reid/torchreid/engine/video/softmax.py
- [[.parse_data_for_train()_1]] - code - backend/references/sn-reid/torchreid/engine/video/triplet.py
- [[.register_model()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.reset()_1]] - code - backend/references/sn-reid/torchreid/utils/avgmeter.py
- [[.run()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.save_model()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.set_model_mode()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.step()_3]] - code - backend/references/sn-reid/torchreid/optim/radam.py
- [[.test()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.train()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.two_stepped_transfer_learning()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[.update()_3]] - code - backend/references/sn-reid/torchreid/utils/avgmeter.py
- [[.update_lr()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[A wrapper function for computing distance matrix.      Args         input1 (tor]] - rationale - backend/references/sn-reid/torchreid/metrics/distance.py
- [[Args             inputs (torch.Tensor) feature matrix with shape (batch_size,]] - rationale - backend/references/sn-reid/torchreid/losses/hard_mine_triplet_loss.py
- [[Args             inputs (torch.Tensor) prediction matrix (before softmax) with]] - rationale - backend/references/sn-reid/torchreid/losses/cross_entropy_loss.py
- [[AverageMeter]] - code - backend/references/sn-reid/torchreid/utils/avgmeter.py
- [[Computes and stores the average and current value.      Examples]] - rationale - backend/references/sn-reid/torchreid/utils/avgmeter.py
- [[Computes cosine distance.      Args         input1 (torch.Tensor) 2-D feature]] - rationale - backend/references/sn-reid/torchreid/metrics/distance.py
- [[Computes euclidean squared distance.      Args         input1 (torch.Tensor) 2]] - rationale - backend/references/sn-reid/torchreid/metrics/distance.py
- [[Computes the accuracy over the k top predictions for     the specified values of]] - rationale - backend/references/sn-reid/torchreid/metrics/accuracy.py
- [[CrossEntropyLoss]] - code - backend/references/sn-reid/torchreid/losses/cross_entropy_loss.py
- [[DeepSupervision      Applies criterion to each element in a list.      Args]] - rationale - backend/references/sn-reid/torchreid/losses/__init__.py
- [[DeepSupervision()]] - code - backend/references/sn-reid/torchreid/losses/__init__.py
- [[Engine_1]] - code
- [[Engine]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[Evaluates CMC rank.      Args         distmat (numpy.ndarray) distance matrix]] - rationale - backend/references/sn-reid/torchreid/metrics/rank.py
- [[Evaluation with cuhk03 metric     Key one image for each gallery identity is ra]] - rationale - backend/references/sn-reid/torchreid/metrics/rank.py
- [[Evaluation with market1501 metric     Key for each query identity, its gallery_1]] - rationale - backend/references/sn-reid/torchreid/metrics/rank.py
- [[Evaluation with market1501 metric     Key for each query identity, its gallery]] - rationale - backend/references/sn-reid/torchreid/metrics/rank.py
- [[ImageSoftmaxEngine]] - code
- [[ImageSoftmaxEngine_1]] - code - backend/references/sn-reid/torchreid/engine/image/softmax.py
- [[ImageTripletEngine]] - code
- [[ImageTripletEngine_1]] - code - backend/references/sn-reid/torchreid/engine/image/triplet.py
- [[Softmax-loss engine for video-reid.      Args         datamanager (DataManager)]] - rationale - backend/references/sn-reid/torchreid/engine/video/softmax.py
- [[Triplet loss with hard positivenegative mining.          Reference         Her]] - rationale - backend/references/sn-reid/torchreid/losses/hard_mine_triplet_loss.py
- [[Triplet-loss engine for video-reid.      Args         datamanager (DataManager)]] - rationale - backend/references/sn-reid/torchreid/engine/video/triplet.py
- [[TripletLoss]] - code - backend/references/sn-reid/torchreid/losses/hard_mine_triplet_loss.py
- [[Two-stepped transfer learning.          The idea is to freeze base layers for a]] - rationale - backend/references/sn-reid/torchreid/engine/engine.py
- [[VideoSoftmaxEngine]] - code - backend/references/sn-reid/torchreid/engine/video/softmax.py
- [[VideoTripletEngine]] - code - backend/references/sn-reid/torchreid/engine/video/triplet.py
- [[__init__.py_16]] - code - backend/references/sn-reid/torchreid/engine/image/__init__.py
- [[__init__.py_14]] - code - backend/references/sn-reid/torchreid/engine/__init__.py
- [[__init__.py_15]] - code - backend/references/sn-reid/torchreid/engine/video/__init__.py
- [[__init__.py_6]] - code - backend/references/sn-reid/torchreid/losses/__init__.py
- [[__init__.py_4]] - code - backend/references/sn-reid/torchreid/metrics/__init__.py
- [[_evaluate()]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[accuracy()]] - code - backend/references/sn-reid/torchreid/metrics/accuracy.py
- [[accuracy.py]] - code - backend/references/sn-reid/torchreid/metrics/accuracy.py
- [[build_engine()]] - code - backend/references/sn-reid/benchmarks/baseline/main.py
- [[compute_distance_matrix()]] - code - backend/references/sn-reid/torchreid/metrics/distance.py
- [[cosine_distance()]] - code - backend/references/sn-reid/torchreid/metrics/distance.py
- [[criterion()]] - code - backend/references/tvcalib/sn_segmentation/src/segmentation/train.py
- [[cross_entropy_loss.py]] - code - backend/references/sn-reid/torchreid/losses/cross_entropy_loss.py
- [[distance.py]] - code - backend/references/sn-reid/torchreid/metrics/distance.py
- [[engine.py]] - code - backend/references/sn-reid/torchreid/engine/engine.py
- [[euclidean_squared_distance()]] - code - backend/references/sn-reid/torchreid/metrics/distance.py
- [[eval_cuhk03()]] - code - backend/references/sn-reid/torchreid/metrics/rank.py
- [[eval_market1501()]] - code - backend/references/sn-reid/torchreid/metrics/rank.py
- [[eval_soccernetv3()]] - code - backend/references/sn-reid/torchreid/metrics/rank.py
- [[evaluate_py()]] - code - backend/references/sn-reid/torchreid/metrics/rank.py
- [[evaluate_rank()]] - code - backend/references/sn-reid/torchreid/metrics/rank.py
- [[hard_mine_triplet_loss.py]] - code - backend/references/sn-reid/torchreid/losses/hard_mine_triplet_loss.py
- [[open_all_layers()]] - code - backend/references/sn-reid/torchreid/utils/torchtools.py
- [[open_specified_layers()]] - code - backend/references/sn-reid/torchreid/utils/torchtools.py
- [[rA generic base Engine class for both image- and video-reid.      Args]] - rationale - backend/references/sn-reid/torchreid/engine/engine.py
- [[rA unified pipeline for training and evaluating a model.          Args]] - rationale - backend/references/sn-reid/torchreid/engine/engine.py
- [[rCross entropy loss with label smoothing regularizer.          Reference]] - rationale - backend/references/sn-reid/torchreid/losses/cross_entropy_loss.py
- [[rOpens all layers in model for training.      Examples          from tor]] - rationale - backend/references/sn-reid/torchreid/utils/torchtools.py
- [[rOpens specified layers in model for training while keeping     other layers]] - rationale - backend/references/sn-reid/torchreid/utils/torchtools.py
- [[rSaves checkpoint.      Args         state (dict) dictionary.         save_]] - rationale - backend/references/sn-reid/torchreid/utils/torchtools.py
- [[rSoftmax-loss engine for image-reid.      Args         datamanager (DataMana]] - rationale - backend/references/sn-reid/torchreid/engine/image/softmax.py
- [[rTests model on target datasets.          .. note              This functio]] - rationale - backend/references/sn-reid/torchreid/engine/engine.py
- [[rTriplet-loss engine for image-reid.      Args         datamanager (DataMana]] - rationale - backend/references/sn-reid/torchreid/engine/image/triplet.py
- [[rank.py]] - code - backend/references/sn-reid/torchreid/metrics/rank.py
- [[save_checkpoint()]] - code - backend/references/sn-reid/torchreid/utils/torchtools.py
- [[softmax.py_1]] - code - backend/references/sn-reid/torchreid/engine/image/softmax.py
- [[softmax.py]] - code - backend/references/sn-reid/torchreid/engine/video/softmax.py
- [[triplet.py_1]] - code - backend/references/sn-reid/torchreid/engine/image/triplet.py
- [[triplet.py]] - code - backend/references/sn-reid/torchreid/engine/video/triplet.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/engine.py_(23+)
SORT file.name ASC
```

## Connections to other communities
- 11 edges to [[_COMMUNITY_model_complexity.py (25+)]]
- 7 edges to [[_COMMUNITY_run_e2e_legacy.py (66+)]]
- 5 edges to [[_COMMUNITY_utils.py (36+)]]
- 3 edges to [[_COMMUNITY_pose.py (49+)]]
- 2 edges to [[_COMMUNITY_transforms.py (47+)]]
- 2 edges to [[_COMMUNITY_cloud_batch_processor.py (12+)]]
- 2 edges to [[_COMMUNITY_main.py (50+)]]
- 1 edge to [[_COMMUNITY_cam_modules.py (43+)]]
- 1 edge to [[_COMMUNITY_extract_tactical_library_from_pdfs.py (10+)]]
- 1 edge to [[_COMMUNITY_dataset.py (36+)]]
- 1 edge to [[_COMMUNITY_sampler.py (19+)]]
- 1 edge to [[_COMMUNITY_test_run_e2e_cloud_fallbacks.py (17+)]]

## Top bridge nodes
- [[.step()_3]] - degree 8, connects to 4 communities
- [[.train()]] - degree 14, connects to 3 communities
- [[_evaluate()]] - degree 10, connects to 3 communities
- [[Engine]] - degree 24, connects to 2 communities
- [[AverageMeter]] - degree 8, connects to 2 communities