# Graph Report - /Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/gaffers-experiment-repo/experiment-backend  (2026-04-19)

## Corpus Check
- 39 files · ~8,047 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 200 nodes · 342 edges · 17 communities detected
- Extraction: 68% EXTRACTED · 32% INFERRED · 0% AMBIGUOUS · INFERRED: 108 edges (avg confidence: 0.74)
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

## God Nodes (most connected - your core abstractions)
1. `process_video()` - 19 edges
2. `LocalFileTaskBackend` - 13 edges
3. `ExperimentJobStore` - 12 edges
4. `ExperimentQueue` - 11 edges
5. `RedisTaskBackend` - 11 edges
6. `run()` - 10 edges
7. `AdvancedPitchCalibrator` - 10 edges
8. `create_job()` - 9 edges
9. `TaskPayload` - 9 edges
10. `main()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `ExperimentJobStore`  [INFERRED]
  /Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/gaffers-experiment-repo/experiment-backend/worker_main.py → /Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/gaffers-experiment-repo/experiment-backend/services/job_store.py
- `run()` --calls--> `MetricsRegistry`  [INFERRED]
  /Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/gaffers-experiment-repo/experiment-backend/worker_main.py → /Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/gaffers-experiment-repo/experiment-backend/services/observability.py
- `run()` --calls--> `ExperimentQueue`  [INFERRED]
  /Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/gaffers-experiment-repo/experiment-backend/worker_main.py → /Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/gaffers-experiment-repo/experiment-backend/services/queue.py
- `create_job()` --calls--> `CreateJobResponse`  [INFERRED]
  /Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/gaffers-experiment-repo/experiment-backend/main.py → /Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/gaffers-experiment-repo/experiment-backend/models.py
- `chat()` --calls--> `ChatResponse`  [INFERRED]
  /Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/gaffers-experiment-repo/experiment-backend/main.py → /Users/trickyoutlaw/Documents/Coding/PROJECTS/phoenix-work/gaffers-experiment-repo/experiment-backend/models.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (11): Protocol, build_task_backend(), LocalFileTaskBackend, RedisTaskBackend, TaskBackend, TaskPayload, TaskBackend, test_local_file_task_backend_enqueue_dequeue() (+3 more)

### Community 1 - "Community 1"
Cohesion: 0.1
Nodes (25): benchmark_decoders(), DecodedChunk, _estimate_total_frames(), _iter_frames(), _iter_pyav_frames(), PipelineArtifacts, process_video(), _read_video_dimensions() (+17 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (15): ExperimentJob, ExperimentJobStore, chat(), create_job(), ensure_runtime_directories(), get_job(), get_tracking(), preflight_check() (+7 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (13): AdvancedPitchCalibrator, Experiment-local advanced calibrator with production-compatible API.      Output, DensePassResult, _iter_window_frame_indices(), run_dense_pass(), DynamicPitchCalibrator, PitchObservationBundle, Experiment-local homography calibrator interface compatible with production call (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.18
Nodes (6): get_metrics(), MetricsRegistry, timed(), TimerStat, ExperimentQueue, QueueItem

### Community 5 - "Community 5"
Cohesion: 0.28
Nodes (12): BaseModel, coach_advice(), list_reports(), AdviceItem, AdviceResponse, ChatRequest, ChatResponse, CreateJobResponse (+4 more)

### Community 6 - "Community 6"
Cohesion: 0.4
Nodes (9): _build_manifest(), _compute_stats(), main(), parse_args(), _run_decoder_trials(), _sample_decoder_utilization(), _sha256_file(), _try_command() (+1 more)

### Community 7 - "Community 7"
Cohesion: 0.29
Nodes (7): GpuRuntimeConfig, select_gpu_runtime(), _select_mps_runtime(), ModelProfile, resolve_profile(), test_mps_runtime_selection_falls_back_when_unavailable(), test_nvidia_runtime_selection_uses_gpu_backend()

### Community 8 - "Community 8"
Cohesion: 0.46
Nodes (7): _bootstrap_ci(), _delta_pct(), main(), _matrix_by_profile(), parse_args(), _read_json(), _welch_ttest()

### Community 9 - "Community 9"
Cohesion: 0.83
Nodes (3): _has_forbidden_runtime_import(), main(), _runtime_files()

### Community 10 - "Community 10"
Cohesion: 0.5
Nodes (2): ExperimentError, Exception

### Community 11 - "Community 11"
Cohesion: 1.0
Nodes (2): FastPassResult, run_fast_pass()

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (0): 

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (0): 

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (0): 

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **2 isolated node(s):** `ModelProfile`, `Experiment-local homography calibrator interface compatible with production call`
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 12`** (2 nodes): `test_architecture_invariants()`, `test_architecture_invariants.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `paths.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ExperimentJobStore` connect `Community 2` to `Community 0`, `Community 1`, `Community 4`?**
  _High betweenness centrality (0.228) - this node is a cross-community bridge._
- **Why does `process_video()` connect `Community 1` to `Community 2`, `Community 4`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.219) - this node is a cross-community bridge._
- **Why does `run_dense_pass()` connect `Community 3` to `Community 1`?**
  _High betweenness centrality (0.208) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `process_video()` (e.g. with `test_process_video_writes_experiment_artifacts()` and `_run_decoder_trials()`) actually correct?**
  _`process_video()` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `LocalFileTaskBackend` (e.g. with `TaskBackend` and `TaskPayload`) actually correct?**
  _`LocalFileTaskBackend` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `ExperimentJobStore` (e.g. with `QueueItem` and `ExperimentQueue`) actually correct?**
  _`ExperimentJobStore` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `ExperimentQueue` (e.g. with `ExperimentJobStore` and `MetricsRegistry`) actually correct?**
  _`ExperimentQueue` has 4 INFERRED edges - model-reasoned connections that need verification._