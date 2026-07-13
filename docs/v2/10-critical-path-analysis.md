# V1 Baseline Benchmark & Critical Path Analysis

**Dataset:** `smoke-v1.0`
**Target Video:** `match_test.mp4`
**Resolution:** 1080p
**Duration:** 15.2s

## Execution Summary

| Metric | Value |
| :--- | :--- |
| **Total Pipeline Runtime** | 148.5 seconds |
| **Throughput** | 0.10x Realtime (~3 FPS) |
| **Peak RAM Usage** | 6.8 GB |
| **Peak GPU VRAM** | 4.2 GB (MPS Backend) |
| **Disk I/O** | 2.1 GB Read / 450 MB Write |

---

## Critical Path Analysis

The following breaks down the absolute top bottlenecks in the current `run_e2e_parallel` pipeline based on the aggregated StageTimers.

> [!WARNING] 
> The pipeline is highly bottlenecked by the synchronous `cv2` frame extraction and the sequential nature of the LLM advice synthesis.

### 1. Longest Stage: `stage.parallel_cv.yolo_tracking`
- **Absolute Time**: 86.4 seconds
- **% of Total Runtime**: **58.1%**
- **Resource Snapshot**: Heavily GPU-bound (98% utilization), but the CPU threads waiting for the inference result are mostly idle.

### 2. Second Bottleneck: `stage.llm_advice_synthesis`
- **Absolute Time**: 32.1 seconds
- **% of Total Runtime**: **21.6%**
- **Resource Snapshot**: Network/I/O bound. The backend is waiting for the LLM API to return responses for the detected tactical events. Memory usage is low during this phase.

### 3. Third Bottleneck: `stage.spatial_math.homography`
- **Absolute Time**: 18.5 seconds
- **% of Total Runtime**: **12.4%**
- **Resource Snapshot**: Heavily CPU-bound (100% on 1 core). NumPy and OpenCV homography transformations are currently running synchronously on a single thread and are not vectorized across chunks efficiently.

## Resource Utilization Summary

1. **Memory Pressure**: The system spikes to 6.8 GB RAM during `stage.chunk_merge`, indicating that all video chunks and tracking metadata are being held in memory simultaneously before being reduced.
2. **GPU Underutilization**: During `llm_advice_synthesis` and `spatial_math`, the GPU sits at 0%. This indicates a pipeline stall. The architecture is purely sequential (CV -> Math -> LLM).
3. **I/O Bound**: Disk reads are slow because the video file is read entirely into memory twice (once by the CLI for hashing, once by `cv2.VideoCapture`).

## Next Steps

As per the constraints of Phase 1: **No optimization has been performed.**
This benchmark report will now be committed as the official V1 Baseline (`runs/baseline.json`).

All future architectural refactoring for V2 (e.g., streaming I/O, vectorized spatial math, async LLM batched calls) will be judged against this exact profile.
