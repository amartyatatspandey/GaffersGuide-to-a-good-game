from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import cv2
import numpy as np

# Ensure backend root is in path for subprocess imports
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.run_e2e_cloud import (
    DownscaledOpticalFlowEstimator,
    _clear_device_cache,
    _infer_device,
    _resolve_primary_ball_class_ids,
)
from scripts.track_teams import (
    CLASS_BALL,
    CLASS_PLAYER,
    MODEL_PATH,
    HybridIDHealer,
    TacticalRadar,
    TeamClassifier,
)
from scripts.e2e_shared import (
    BALL_INTERPOLATION_MAX_GAP,
    HOMOGRAPHY_CONFIDENCE_FALLBACK_THRESHOLD,
    CVTelemetry,
    TacticalFrame,
    TacticalPlayer,
    TrackingFrameArtifact,
    _fallback_project_from_camera_shift,
    _homography_confidence,
    _prediction_to_team,
    MIN_BALL_CONFIDENCE,
    FPS,
    COUNTER_ATTACK_WINDOW_FRAMES,
    PRESS_SUCCESS_WINDOW_FRAMES,
    build_metrics_timeline,
    compute_ball_visibility_ratio,
    apply_ball_metrics_gate,
    evaluate_chunk_insights,
    synthesize,
    run_llm,
    ensure_core_pipeline_directories,
    collect_local_cv_pipeline_gaps,
    format_pipeline_prerequisite_errors,
    _print_step,
    _write_tracking_artifact,
    _print_data_guard_reliability,
    _final_cards_llm_skipped_low_reliability,
)

from scripts.run_calibrator_on_video import ensure_homography_json_for_video
from scripts.global_refiner import GlobalRefiner

LOGGER = logging.getLogger(__name__)


def _log_worker_perf(
    stage: str,
    chunk_idx: int,
    duration_seconds: float,
    **extra: object,
) -> None:
    """
    Emit a PERF_STAGE JSON log entry from a subprocess worker to stdout.

    Cloud Run captures all stdout from child processes.  Using plain
    ``print()`` with JSON avoids importing the main process's logging
    config inside the spawned subprocess.
    """
    entry = {
        "severity": "INFO",
        "message": "PERF_STAGE",
        "stage": stage,
        "chunk_idx": chunk_idx,
        "duration_seconds": round(duration_seconds, 3),
        "status": "ok",
    }
    entry.update(extra)  # type: ignore[arg-type]
    print(json.dumps(entry, default=str), flush=True)


class VideoChunkManager:
    """
    Manages calculating chunk boundaries with overlap for parallel processing
    and handles memory-safe metadata query of the input video.
    """

    def __init__(self, video_path: str | Path, target_chunk_duration: float = 90.0, min_overlap: float = 2.0) -> None:
        """
        :param video_path: Path to the MP4 video.
        :param target_chunk_duration: Target duration of each chunk in seconds.
        :param min_overlap: Minimum overlap between adjacent chunks in seconds.
        """
        self.video_path = Path(video_path).resolve()
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # Memory-safe Video Reading: query metadata without loading any frames into RAM.
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.target_chunk_duration = target_chunk_duration
        self.min_overlap = min_overlap

    def calculate_chunks(self) -> list[dict[str, Any]]:
        """
        Intelligently calculate chunk frame ranges with overlap.
        Handles the last chunk cleanly if it's not perfectly divisible.
        """
        step_frames = int(self.target_chunk_duration * self.fps)
        overlap_frames = int(self.min_overlap * self.fps)

        if step_frames <= 0:
            step_frames = 4500  # Default to 3 minutes at 25 FPS if invalid
        if overlap_frames < 50:
            overlap_frames = 50  # Enforce minimum 2 seconds (50 frames at 25 FPS)

        chunks = []
        start = 0
        chunk_idx = 0

        while start < self.total_frames:
            end_frame = start + step_frames
            # Add overlap to the end of the chunk (except if it is the absolute end of the video)
            end_with_overlap = end_frame + overlap_frames

            if end_with_overlap >= self.total_frames:
                end_with_overlap = self.total_frames
                end_frame = self.total_frames

            # If the remaining frames after this chunk are very small (e.g., less than 10 seconds),
            # just merge them into the current chunk to avoid creating a tiny trailing chunk.
            remaining_frames = self.total_frames - end_frame
            if remaining_frames < int(10 * self.fps):
                end_with_overlap = self.total_frames
                end_frame = self.total_frames

            chunks.append({
                "chunk_idx": chunk_idx,
                "start_frame": start,
                "end_frame": end_with_overlap,
                "non_overlap_end": end_frame,
                "overlap_start": end_frame,
                "overlap_end": end_with_overlap,
            })

            if end_frame >= self.total_frames:
                break

            start = end_frame
            chunk_idx += 1

        LOGGER.info(
            "Calculated %d chunks for video of %d frames (FPS: %.2f)",
            len(chunks),
            self.total_frames,
            self.fps,
        )
        return chunks


def run_cv_chunk_worker(
    video_path: str,
    start_frame: int,
    end_frame: int,
    chunk_idx: int,
    progress_dict: dict[int, int],
    device: str | None,
    batch_size: int,
    flow_max_width: int,
    homography_json_str: str,
) -> dict[str, Any]:
    """
    Subprocess worker executing the independent CV tracking pipeline on a video chunk.
    This runs completely isolated with its own model instances and resources.
    """
    # Configure logging for child process
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: (Worker) %(message)s")
    logger = logging.getLogger(f"Worker-{chunk_idx}")
    logger.info("Initializing CV worker for range [%d, %d]", start_frame, end_frame)

    _chunk_t0 = time.perf_counter()

    from ultralytics import YOLO
    import supervision as sv

    try:
        import torch
    except ImportError:
        torch = None

    selected_device = _infer_device(device)
    use_half = selected_device in ("cuda", "mps")

    # Load YOLOv11 Model fresh inside subprocess
    _model_t0 = time.perf_counter()
    model = YOLO(str(MODEL_PATH))
    if use_half and torch is not None:
        if selected_device == "cuda" and torch.cuda.is_available():
            try:
                model.model.half()
            except Exception:
                use_half = False

    primary_ball_class_ids = _resolve_primary_ball_class_ids(model)
    _log_worker_perf("worker_model_load", chunk_idx, time.perf_counter() - _model_t0)
    tracker = sv.ByteTrack()
    classifier = TeamClassifier()
    healer = HybridIDHealer()
    flow_estimator = DownscaledOpticalFlowEstimator(max_width=flow_max_width)

    # Memory-Safe Video Reading: Seek to specific frame using OpenCV
    _cap_t0 = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video in subprocess: {video_path}")

    # Set frame position index (seeking)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if actual_pos != start_frame:
        logger.warning("Requested seek to %d, but OpenCV ended up at %d", start_frame, actual_pos)
    _log_worker_perf("worker_video_open_seek", chunk_idx, time.perf_counter() - _cap_t0,
                     start_frame=start_frame)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    radar = TacticalRadar(json_path=Path(homography_json_str), video_res=(width, height))

    telemetry = CVTelemetry()
    frames_out: list[TacticalFrame] = []
    frame_artifacts: list[TrackingFrameArtifact] = []

    last_player_radar_by_track: dict[int, tuple[int, int]] = {}
    last_ball_radar: tuple[int, int] | None = None

    frames_to_process = end_frame - start_frame
    processed_count = 0

    frame_step = int(os.getenv("FRAME_STEP", "1"))

    _inference_total_s = 0.0  # cumulative YOLO inference wall-time

    def get_batches():
        idx = 0
        while idx < frames_to_process:
            batch = []
            batch_limit = min(batch_size, frames_to_process - idx)
            for _ in range(batch_limit):
                ret, frame = cap.read()
                if not ret:
                    break
                batch.append(frame)
            if not batch:
                break
            yield idx, batch
            idx += len(batch)

    for offset_idx, batch_frames in get_batches():
        kwargs = {"conf": 0.35, "verbose": False}
        if selected_device:
            kwargs["device"] = selected_device
        if use_half:
            kwargs["half"] = True

        # Determine which frames in this batch require YOLO inference
        inference_frames = []
        inference_offsets = []
        for offset, frame in enumerate(batch_frames):
            frame_idx = start_frame + offset_idx + offset
            if (frame_idx - start_frame) % frame_step == 0:
                inference_frames.append(frame)
                inference_offsets.append(offset)

        results = []
        if inference_frames:
            _inf_t0 = time.perf_counter()
            try:
                results = model(inference_frames, **kwargs)
            except Exception:
                if use_half:
                    kwargs.pop("half", None)
                    results = model(inference_frames, **kwargs)
                else:
                    raise
            _inference_total_s += time.perf_counter() - _inf_t0

        for offset, frame in enumerate(batch_frames):
            frame_idx = start_frame + offset_idx + offset
            is_key_frame = (frame_idx - start_frame) % frame_step == 0

            radar.update_camera_angle(frame_idx)
            camera_shift = flow_estimator.update(frame)
            homography_conf = _homography_confidence(radar, frame_idx)
            use_fallback = homography_conf < HOMOGRAPHY_CONFIDENCE_FALLBACK_THRESHOLD

            if use_fallback:
                telemetry.frames_optical_flow_fallback += 1
            else:
                telemetry.frames_standard_homography += 1

            if not is_key_frame:
                # Store placeholder for non-keyframe (will be filled during interpolation)
                frames_out.append(
                    TacticalFrame(
                        frame_idx=frame_idx,
                        players=[],
                        ball_xy=None,
                        possession_team_id=None,
                    )
                )
                frame_artifacts.append(
                    TrackingFrameArtifact(
                        frame_idx=frame_idx,
                        players=[],
                        ball_xy=None,
                        possession_team_id=None,
                        homography_confidence=float(homography_conf),
                        used_optical_flow_fallback=use_fallback,
                        camera_shift_xy=(float(camera_shift[0]), float(camera_shift[1])),
                        ball_canvas=None,
                    )
                )
                continue

            ball_xy_centered = None
            ball_xy_artifact = None
            ball_canvas = None
            
            result_idx = inference_offsets.index(offset) if offset in inference_offsets else -1
            result = results[result_idx] if (0 <= result_idx < len(results)) else None

            if result is None:
                frames_out.append(
                    TacticalFrame(
                        frame_idx=frame_idx,
                        players=[],
                        ball_xy=None,
                        possession_team_id=None,
                    )
                )
                frame_artifacts.append(
                    TrackingFrameArtifact(
                        frame_idx=frame_idx,
                        players=[],
                        ball_xy=None,
                        possession_team_id=None,
                        homography_confidence=float(homography_conf),
                        used_optical_flow_fallback=use_fallback,
                        camera_shift_xy=(float(camera_shift[0]), float(camera_shift[1])),
                    )
                )
                continue

            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            # Strict pitch boundary filter: drop sideline personnel
            radar_pts_unclamped = radar.map_many_to_2d(detections.xyxy, frame_idx=frame_idx, clamp=False)
            keep_mask = []
            for i, pt in enumerate(radar_pts_unclamped):
                cid = int(detections.class_id[i])
                if cid == CLASS_BALL:
                    keep_mask.append(True)
                elif pt is None:
                    # Only keep ball when projection fails, exclude all other detections
                    keep_mask.append(cid == CLASS_BALL)
                else:
                    rx, ry = pt
                    # x: -50 to 1100, y: -50 to 730 (buffer from -5m to 110m and -5m to 73m)
                    if -50 <= rx <= 1100 and -50 <= ry <= 730:
                        keep_mask.append(True)
                    else:
                        keep_mask.append(False)

            if len(keep_mask) > 0:
                mask = np.array(keep_mask, dtype=bool)
                detections = detections[mask]

            det_conf = getattr(detections, "confidence", None)

            best_ball_bbox = None
            best_ball_score = -1.0
            for i in range(len(detections)):
                cid = int(detections.class_id[i])
                if cid not in primary_ball_class_ids:
                    continue
                score = float(det_conf[i]) if det_conf is not None and i < len(det_conf) else 0.0
                if score >= best_ball_score:
                    best_ball_score = score
                    best_ball_bbox = detections.xyxy[i]

            if best_ball_bbox is not None:
                telemetry.total_raw_ball_detections += 1
                ball_pt = radar.map_to_2d(best_ball_bbox)
                if ball_pt is None and use_fallback:
                    ball_pt = _fallback_project_from_camera_shift(
                        bbox=best_ball_bbox,
                        last_radar_pt=last_ball_radar,
                        camera_shift_xy=camera_shift,
                        video_wh=(width, height),
                        radar_wh=(radar.radar_w, radar.radar_h),
                    )
                if ball_pt is not None:
                    last_ball_radar = (int(ball_pt[0]), int(ball_pt[1]))
                    ball_xy_centered = [float(ball_pt[0]) / 10.0 - 52.5, float(ball_pt[1]) / 10.0 - 34.0]
                    ball_xy_artifact = [float(ball_pt[0]) / 10.0, float(ball_pt[1]) / 10.0]
                    ball_canvas = [float((best_ball_bbox[0] + best_ball_bbox[2]) / 2.0), float((best_ball_bbox[1] + best_ball_bbox[3]) / 2.0)]

            radar_pts = radar.map_many_to_2d(detections.xyxy, frame_idx=frame_idx)
            tracker_ids = healer.process_and_heal(detections, frame, radar_pts, frame_idx)
            if frame_idx % 300 == 0:
                healer.cleanup_ghost_ids(frame_idx)
            if tracker_ids is None:
                tracker_ids = getattr(detections, "tracker_id", None)

            frame_data = []
            for i in range(len(detections)):
                tid = None
                if tracker_ids is not None and i < len(tracker_ids):
                    raw_id = tracker_ids[i]
                    tid = int(raw_id) if raw_id is not None else None
                frame_data.append({
                    "id": tid,
                    "bbox": detections.xyxy[i],
                    "cid": int(detections.class_id[i]),
                    "radar_pt": radar_pts[i],
                })

            role_mapping = classifier.predict_frame(frame, frame_data, frame_idx)
            tactical_players = []

            for row in frame_data:
                if row["cid"] != CLASS_PLAYER:
                    continue
                prediction = role_mapping.get(row["id"], "unknown")
                team = _prediction_to_team(prediction)
                if team is None:
                    continue
                pt = row["radar_pt"]
                if (
                    pt is None
                    and use_fallback
                    and row["id"] is not None
                    and row["id"] in last_player_radar_by_track
                ):
                    pt = _fallback_project_from_camera_shift(
                        bbox=row["bbox"],
                        last_radar_pt=last_player_radar_by_track[row["id"]],
                        camera_shift_xy=camera_shift,
                        video_wh=(width, height),
                        radar_wh=(radar.radar_w, radar.radar_h),
                    )
                pt_out = None
                if pt is not None:
                    pt_out = [float(pt[0]), float(pt[1])]
                    if row["id"] is not None:
                        last_player_radar_by_track[row["id"]] = (
                            int(round(pt_out[0])),
                            int(round(pt_out[1])),
                        )
                pt_meters = [pt_out[0] / 10.0 - 52.5, pt_out[1] / 10.0 - 34.0] if pt_out is not None else None
                tactical_players.append(
                    TacticalPlayer(id=row["id"], team=team, radar_pt=pt_meters)
                )

            frames_out.append(
                TacticalFrame(
                    frame_idx=frame_idx,
                    players=tactical_players,
                    ball_xy=ball_xy_centered,
                    possession_team_id=None,  # Possession is computed globally at merge
                )
            )

            player_rows = []
            for row in frame_data:
                if row["cid"] != CLASS_PLAYER:
                    continue
                rp = row["radar_pt"]
                bbox = np.asarray(row["bbox"], dtype=np.float64).ravel()
                team_label = _prediction_to_team(role_mapping.get(row["id"], "unknown"))
                player_rows.append({
                    "id": row["id"],
                    "team_id": team_label,
                    "x_pitch": float(rp[0]) / 10.0 if rp is not None else None,
                    "y_pitch": float(rp[1]) / 10.0 if rp is not None else None,
                    "x_canvas": float((bbox[0] + bbox[2]) / 2.0),
                    "y_canvas": float((bbox[1] + bbox[3]) / 2.0),
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                })

            frame_artifacts.append(
                TrackingFrameArtifact(
                    frame_idx=frame_idx,
                    players=player_rows,
                    ball_xy=ball_xy_artifact,
                    possession_team_id=None,
                    homography_confidence=float(homography_conf),
                    used_optical_flow_fallback=use_fallback,
                    camera_shift_xy=(float(camera_shift[0]), float(camera_shift[1])),
                    ball_canvas=ball_canvas,
                )
            )

        processed_count += len(batch_frames)
        # Update progress dict
        progress_dict[chunk_idx] = int((processed_count / frames_to_process) * 100)
    
    if frame_step > 1:
        from scripts.e2e_shared import interpolate_tracking_data
        interpolate_tracking_data(frames_out, frame_artifacts, frame_step)
        
    _clear_device_cache(selected_device)
    cap.release()

    _log_worker_perf(
        "worker_chunk_total", chunk_idx, time.perf_counter() - _chunk_t0,
        frames_processed=processed_count,
        inference_total_seconds=round(_inference_total_s, 3),
    )

    # Serialize output structure safely back to the main process
    return {
        "chunk_idx": chunk_idx,
        "raw_frames": [asdict(f) for f in frames_out],
        "telemetry": asdict(telemetry),
        "tracking_frames": [
            {
                "frame_idx": f.frame_idx,
                "players": f.players,
                "ball_xy": f.ball_xy,
                "possession_team_id": f.possession_team_id,
                "homography_confidence": f.homography_confidence,
                "used_optical_flow_fallback": f.used_optical_flow_fallback,
                "camera_shift_xy": f.camera_shift_xy,
                "ball_canvas": f.ball_canvas,
            }
            for f in frame_artifacts
        ],
    }


class ParallelPipelineExecutor:
    """
    Executes multiple chunk tracking processes simultaneously and aggregates real-time progress.
    Handles device scheduling to prevent GPU OOM.
    """

    def __init__(
        self,
        video_path: Path,
        chunks: list[dict[str, Any]],
        device: str | None = None,
        batch_size: int = 16,
        flow_max_width: int = 640,
    ) -> None:
        self.video_path = video_path
        self.chunks = chunks
        self.device = device
        self.batch_size = batch_size
        self.flow_max_width = flow_max_width

    async def execute_parallel(
        self,
        homography_json: Path,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Runs the ProcessPoolExecutor to process all chunks.
        Monitors progress and updates progress_callback in real time.
        """
        loop = asyncio.get_running_loop()

        # Decide concurrency based on hardware. Limit to prevent GPU OOM
        # RTX 4090 (24GB VRAM) can support ~4-6 parallel YOLO models comfortably.
        # Apple M3 Max (Shared memory) is also fast with ~5 workers, but many base Macs have 8GB/16GB unified RAM.
        # Spawning 5 parallel YOLO workers will OOM crash most Macs. We cap this at 2 to be safe.
        # Concurrency config: read from MAX_WORKERS env variable, or fall back to defaults
        env_max_workers = os.getenv("MAX_WORKERS")
        if env_max_workers is not None:
            try:
                max_workers = max(1, int(env_max_workers))
            except ValueError:
                max_workers = 2
        else:
            device_type = _infer_device(self.device)
            if device_type == "cuda":
                max_workers = 6  # RTX 4090 can easily handle 6 parallel processes
            elif device_type == "mps":
                max_workers = 1  # Apple Metal (MPS) does not support concurrent multi-process access
            else:
                # CPU: default to half logical CPU cores
                max_workers = max(1, multiprocessing.cpu_count() // 2)

        # Multiprocessing Manager for shared progress dictionary
        manager = multiprocessing.Manager()
        progress_dict = manager.dict()
        for c in self.chunks:
            progress_dict[c["chunk_idx"]] = 0

        # We periodically report progress using an async monitor task
        progress_done = asyncio.Event()

        async def monitor_progress():
            last_msg = ""
            while not progress_done.is_set():
                progress_list = []
                for c in self.chunks:
                    c_idx = c["chunk_idx"]
                    pct = progress_dict.get(c_idx, 0)
                    progress_list.append(f"C{c_idx+1}:{pct}%")

                total_pct = int(sum(progress_dict.values()) / len(self.chunks))
                msg = f"Tracking Players ({total_pct}% complete — {', '.join(progress_list)})"

                if msg != last_msg and progress_callback:
                    progress_callback(msg)
                    last_msg = msg

                await asyncio.sleep(1.0)

        monitor_task = asyncio.create_task(monitor_progress())

        results = []
        try:
            # We run ProcessPoolExecutor to process completely independent Python processes
            # Use 'spawn' start method (especially on macOS and Windows to prevent CUDA fork issues)
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
                futures = []
                for c in self.chunks:
                    future = loop.run_in_executor(
                        pool,
                        run_cv_chunk_worker,
                        str(self.video_path),
                        c["start_frame"],
                        c["end_frame"],
                        c["chunk_idx"],
                        progress_dict,
                        self.device,
                        self.batch_size,
                        self.flow_max_width,
                        str(homography_json),
                    )
                    futures.append(future)

                # Wait for all processes to complete
                results = await asyncio.gather(*futures)
        finally:
            progress_done.set()
            await monitor_task

        # Sort results by chunk index
        results.sort(key=lambda r: r["chunk_idx"])
        return results


class UnionFind:
    """Disjoint Set Union (DSU) to group local player IDs into a single global ID."""

    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, x: tuple[int, int]) -> tuple[int, int]:
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: tuple[int, int], y: tuple[int, int]) -> None:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y


class ChunkResultMerger:
    """
    Intelligently reconciles identity conflicts across chunks using overlap frames,
    and merges chunk results back into one unified tracking output.
    """

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks

    def reconcile_and_merge(
        self,
        chunk_results: list[dict[str, Any]],
    ) -> tuple[list[TacticalFrame], CVTelemetry, list[TrackingFrameArtifact]]:
        """
        Reconciles player IDs across overlap boundaries using Union-Find,
        and constructs a single coherent timeline.
        """
        uf = UnionFind()

        # Step 1: Voting for matches in the overlap frame regions of adjacent chunks
        for i in range(len(self.chunks) - 1):
            chunk_a = self.chunks[i]
            chunk_b = self.chunks[i + 1]

            res_a = chunk_results[i]
            res_b = chunk_results[i + 1]

            overlap_start = chunk_b["start_frame"]
            overlap_end = chunk_a["end_frame"]

            # Map frame_idx -> players for fast lookup
            a_frames = {f["frame_idx"]: f["players"] for f in res_a["tracking_frames"] if overlap_start <= f["frame_idx"] <= overlap_end}
            b_frames = {f["frame_idx"]: f["players"] for f in res_b["tracking_frames"] if overlap_start <= f["frame_idx"] <= overlap_end}

            # co_occurrences: counts frames where BOTH ids are present in the overlap
            co_occurrences: dict[tuple[int, int], int] = {}
            # match_votes: counts frames where canvas coordinates are very close
            match_votes: dict[tuple[int, int], int] = {}

            common_frames = set(a_frames.keys()).intersection(b_frames.keys())
            for f_idx in common_frames:
                players_a = a_frames[f_idx]
                players_b = b_frames[f_idx]

                for p_a in players_a:
                    id_a = p_a["id"]
                    if id_a is None:
                        continue
                    key_a = (i, id_a)

                    for p_b in players_b:
                        id_b = p_b["id"]
                        if id_b is None:
                            continue
                        key_b = (i + 1, id_b)

                        # Enforce team consistency
                        if p_a["team_id"] != p_b["team_id"]:
                            continue

                        pair = (id_a, id_b)
                        co_occurrences[pair] = co_occurrences.get(pair, 0) + 1

                        # Distance threshold of 10 pixels is extremely robust on canvas
                        dx = p_a["x_canvas"] - p_b["x_canvas"]
                        dy = p_a["y_canvas"] - p_b["y_canvas"]
                        dist = np.hypot(dx, dy)

                        if dist < 10.0:
                            match_votes[pair] = match_votes.get(pair, 0) + 1

            # Determine valid links based on overlap votes
            for (id_a, id_b), votes in match_votes.items():
                total = co_occurrences[(id_a, id_b)]
                # Link if matched in at least 3 frames and over 40% of co-occurring frames
                if votes >= 3 and (votes / total) >= 0.4:
                    uf.union((i, id_a), (i + 1, id_b))
                    LOGGER.info(
                        "Linked player (Chunk %d, ID %d) -> (Chunk %d, ID %d) with %d/%d votes",
                        i, id_a, i + 1, id_b, votes, total
                    )

        # Step 2: Assign a global ID to each unique root in the Union-Find map
        global_id_counter = 1
        global_id_map: dict[tuple[int, int], int] = {}

        # First collect all active local IDs in all chunks
        for i, res in enumerate(chunk_results):
            for frame in res["tracking_frames"]:
                for p in frame["players"]:
                    if p["id"] is not None:
                        local_key = (i, p["id"])
                        root = uf.find(local_key)
                        if root not in global_id_map:
                            global_id_map[root] = global_id_counter
                            global_id_counter += 1
                        # Map this specific local ID to the root's global ID
                        global_id_map[local_key] = global_id_map[root]

        # Step 3: Construct the final merged timeline using non-overlapping partitions
        merged_raw_frames: list[TacticalFrame] = []
        merged_tracking_frames: list[TrackingFrameArtifact] = []

        # Merge telemetry by summing non-overlapping counts
        merged_telemetry = CVTelemetry()

        for i, c in enumerate(self.chunks):
            res = chunk_results[i]
            t_chunk = res["telemetry"]
            
            # Map chunk telemetry counts proportionally
            merged_telemetry.total_frames_processed += len(res["raw_frames"])

            # Map raw frames in range [start_frame, non_overlap_end)
            raw_chunk_frames = [
                f for f in res["raw_frames"]
                if c["start_frame"] <= f["frame_idx"] < c["non_overlap_end"]
            ]
            for f in raw_chunk_frames:
                # Map player IDs to global IDs
                players_mapped = []
                for p in f["players"]:
                    g_id = global_id_map.get((i, p["id"]), p["id"]) if p["id"] is not None else None
                    players_mapped.append(
                        TacticalPlayer(id=g_id, team=p["team"], radar_pt=p["radar_pt"])
                    )
                merged_raw_frames.append(
                    TacticalFrame(
                        frame_idx=f["frame_idx"],
                        players=players_mapped,
                        ball_xy=f["ball_xy"],
                        possession_team_id=f["possession_team_id"],
                    )
                )

            # Map tracking artifacts in range [start_frame, non_overlap_end)
            tracking_chunk_frames = [
                f for f in res["tracking_frames"]
                if c["start_frame"] <= f["frame_idx"] < c["non_overlap_end"]
            ]
            for f in tracking_chunk_frames:
                players_mapped_dicts = []
                for p in f["players"]:
                    g_id = global_id_map.get((i, p["id"]), p["id"]) if p["id"] is not None else None
                    p_copy = dict(p)
                    p_copy["id"] = g_id
                    players_mapped_dicts.append(p_copy)

                merged_tracking_frames.append(
                    TrackingFrameArtifact(
                        frame_idx=f["frame_idx"],
                        players=players_mapped_dicts,
                        ball_xy=f["ball_xy"],
                        possession_team_id=f["possession_team_id"],
                        homography_confidence=f["homography_confidence"],
                        used_optical_flow_fallback=f["used_optical_flow_fallback"],
                        camera_shift_xy=f["camera_shift_xy"],
                        ball_canvas=f.get("ball_canvas"),
                    )
                )

        # Global ball position interpolation
        merged_telemetry.total_interpolated_ball_frames = interpolate_ball_positions(
            merged_raw_frames, max_gap_frames=BALL_INTERPOLATION_MAX_GAP
        )

        # Sync possession team ID and ball coordinates from raw frames back to tracking artifacts
        raw_ball_by_frame = {f.frame_idx: (f.ball_xy, f.possession_team_id) for f in merged_raw_frames}
        for f in merged_tracking_frames:
            ball_xy, possession_team_id = raw_ball_by_frame.get(f.frame_idx, (None, None))
            f.ball_xy = ball_xy
            f.possession_team_id = possession_team_id

        # Calculate individual player speeds
        player_last_seen = {}
        fps = 25.0
        for f in merged_tracking_frames:
            for p in f.players:
                g_id = p.get("id")
                if g_id is None:
                    p["speed_kmh"] = 0.0
                    continue
                x = p.get("x_pitch")
                y = p.get("y_pitch")
                if x is None or y is None:
                    p["speed_kmh"] = 0.0
                    continue
                
                speed_kmh = 0.0
                if g_id in player_last_seen:
                    last_f = player_last_seen[g_id]["frame_idx"]
                    last_x = player_last_seen[g_id]["x"]
                    last_y = player_last_seen[g_id]["y"]
                    
                    frames_elapsed = f.frame_idx - last_f
                    if 0 < frames_elapsed < fps * 2:
                        dist = np.hypot(x - last_x, y - last_y)
                        time_sec = frames_elapsed / fps
                        speed_ms = dist / time_sec
                        speed_kmh = speed_ms * 3.6
                
                p["speed_kmh"] = round(float(speed_kmh), 2)
                player_last_seen[g_id] = {"frame_idx": f.frame_idx, "x": x, "y": y}

        # Compile telemetry attributes
        merged_telemetry.frames_standard_homography = sum(1 for f in merged_tracking_frames if not f.used_optical_flow_fallback)
        merged_telemetry.frames_optical_flow_fallback = sum(1 for f in merged_tracking_frames if f.used_optical_flow_fallback)
        merged_telemetry.total_raw_ball_detections = sum(t["telemetry"]["total_raw_ball_detections"] for t in chunk_results)

        return merged_raw_frames, merged_telemetry, merged_tracking_frames


def interpolate_ball_positions(frames: list[TacticalFrame], max_gap_frames: int) -> int:
    """
    Linear interpolation helper for ball tracking continuity on the merged frame array.
    """
    from calculators.possession import interpolate_ball_positions as base_interpolate
    return base_interpolate(frames, max_gap_frames=max_gap_frames)


async def run_e2e_parallel(
    video: str | Path,
    *,
    output_prefix: str = "test_mp4",
    progress_callback: Callable[[str], None] | None = None,
    batch_size: int = 16,
    flow_max_width: int = 640,
    device: str | None = None,
    llm_engine: Literal["local", "cloud"] = "cloud",
    enable_zsl: bool = False,
    target_chunk_duration: float = 900.0,
    min_overlap: float = 2.0,
) -> Path:
    """
    Parallel version of run_e2e_with_zsl.
    Splits video into chunks, runs YOLO and tracking in parallel processes,
    reconciles player tracking IDs across chunk transitions,
    and runs the downstream metrics, rules, and LLM analysis globally.
    """
    if progress_callback is not None:
        progress_callback("Pending")

    from scripts.run_e2e_cloud import _resolve_video_path
    video_path = video if isinstance(video, Path) else _resolve_video_path(video)

    ensure_core_pipeline_directories()
    prereq_gaps = collect_local_cv_pipeline_gaps(video_path=video_path)
    if prereq_gaps:
        raise FileNotFoundError(format_pipeline_prerequisite_errors(prereq_gaps))

    _pipeline_t0 = time.perf_counter()

    # Setup paths
    out_dir = BACKEND_ROOT / "output"
    tracking_overlay_path = out_dir / f"{output_prefix}_tracking_overlay.mp4"
    tracking_data_path = out_dir / f"{output_prefix}_tracking_data.json"
    metrics_output_path = out_dir / f"{output_prefix}_tactical_metrics.json"
    report_output_path = out_dir / f"{output_prefix}_report.json"
    event_index_path = out_dir / f"{output_prefix}_events.json"
    threat_profiles_path = out_dir / f"{output_prefix}_threat_profiles.json"

    # Ensure no stale artifacts are interpreted as new output.
    for p in [tracking_overlay_path, tracking_data_path, metrics_output_path, report_output_path,
              event_index_path, threat_profiles_path]:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    if progress_callback is not None:
        progress_callback("Calibrating Pitch")

    # Step 1 & 2: Pre-calculate homographies and chunks in a thread to unblock the asyncio event loop
    def _prepare_data():
        h_json = ensure_homography_json_for_video(video_path, progress_callback=progress_callback)
        c_manager = VideoChunkManager(
            video_path=video_path,
            target_chunk_duration=target_chunk_duration,
            min_overlap=min_overlap,
        )
        c_list = c_manager.calculate_chunks()
        return h_json, c_list, c_manager.fps

    _stage_t0 = time.perf_counter()
    homography_json, chunks, video_fps = await asyncio.to_thread(_prepare_data)
    LOGGER.info(
        "PERF_STAGE",
        extra={
            "job_id": output_prefix,
            "stage": "calibration",
            "duration_seconds": round(time.perf_counter() - _stage_t0, 3),
            "num_chunks": len(chunks),
            "video_fps": video_fps,
            "status": "ok",
        },
    )

    if progress_callback is not None:
        progress_callback("Tracking Players (Parallel: 0%)")

    # Step 3: Run Parallel Pipeline Execution
    executor = ParallelPipelineExecutor(
        video_path=video_path,
        chunks=chunks,
        device=device,
        batch_size=batch_size,
        flow_max_width=flow_max_width,
    )
    _stage_t0 = time.perf_counter()
    chunk_results = await executor.execute_parallel(
        homography_json=homography_json,
        progress_callback=progress_callback,
    )
    LOGGER.info(
        "PERF_STAGE",
        extra={
            "job_id": output_prefix,
            "stage": "parallel_cv_tracking",
            "duration_seconds": round(time.perf_counter() - _stage_t0, 3),
            "num_chunks": len(chunks),
            "num_chunk_results": len(chunk_results),
            "status": "ok",
        },
    )

    if progress_callback is not None:
        progress_callback("Merging Chunk Results")

    def _merge_chunks():
        merger = ChunkResultMerger(chunks=chunks)
        raw_frames, telemetry, tracking_frames = merger.reconcile_and_merge(chunk_results)

        _print_step(f"Parallel CV Tracking & Merger Complete: Reconciled {len(raw_frames)} frames.")
        _write_tracking_artifact(
            tracking_data_path,
            video_path=video_path,
            overlay_path=tracking_overlay_path,
            telemetry=telemetry,
            frames=tracking_frames,
        )
        return raw_frames

    _stage_t0 = time.perf_counter()
    raw_frames = await asyncio.to_thread(_merge_chunks)
    LOGGER.info(
        "PERF_STAGE",
        extra={
            "job_id": output_prefix,
            "stage": "chunk_merge",
            "duration_seconds": round(time.perf_counter() - _stage_t0, 3),
            "merged_frames": len(raw_frames),
            "status": "ok",
        },
    )

    if progress_callback is not None:
        progress_callback("Spatial Math")

    def _run_spatial_math():
        refiner = GlobalRefiner()
        refined_frames = refiner.refine(
            raw_frames,
            frame_factory=lambda frame_idx, players: TacticalFrame(
                frame_idx=frame_idx,
                players=players,
                ball_xy=None,
                possession_team_id=None,
            ),
            player_factory=lambda pid, team, radar_pt: TacticalPlayer(
                id=pid, team=team, radar_pt=radar_pt
            ),
        )

        # Restore ball and possession data to refined frames
        raw_ball_by_frame = {f.frame_idx: (f.ball_xy, f.possession_team_id) for f in raw_frames}
        for frame in refined_frames:
            ball_xy, possession_team_id = raw_ball_by_frame.get(frame.frame_idx, (None, None))
            frame.ball_xy = ball_xy
            frame.possession_team_id = possession_team_id

        metrics_timeline = build_metrics_timeline(refined_frames)
        visibility_ratio = compute_ball_visibility_ratio(refined_frames)
        metrics_timeline, ball_data_quality = apply_ball_metrics_gate(
            metrics_timeline,
            refined_frames,
            visibility_ratio=visibility_ratio,
            min_ball_confidence=MIN_BALL_CONFIDENCE,
            fps=video_fps,
            counter_attack_window_frames=int(8 * video_fps),
            press_success_window_frames=int(5 * video_fps),
        )

        with metrics_output_path.open("w", encoding="utf-8") as f:
            json.dump(metrics_timeline, f, ensure_ascii=False)
            
        return refined_frames, metrics_timeline, ball_data_quality

    _stage_t0 = time.perf_counter()
    refined_frames, metrics, ball_data_quality = await asyncio.to_thread(_run_spatial_math)
    LOGGER.info(
        "PERF_STAGE",
        extra={
            "job_id": output_prefix,
            "stage": "spatial_math",
            "duration_seconds": round(time.perf_counter() - _stage_t0, 3),
            "metrics_frames": len(metrics),
            "status": "ok",
        },
    )

    # ── Event Intelligence Layer ──────────────────────────────────────────────
    # Runs asynchronously after spatial math. Failure is non-fatal — the existing
    # rule engine + LLM pipeline continues regardless.
    def _run_event_layer():
        try:
            from event_layer.pipeline import run_event_detection
            from event_layer.threat import run_threat_attribution
            from event_layer.pipeline import load_event_index

            # Convert refined_frames (TacticalFrame dataclass list) to dict list
            # compatible with EventDetectionPipeline
            from dataclasses import asdict
            frames_as_dicts = []
            for frame in refined_frames:
                frame_dict = asdict(frame) if hasattr(frame, '__dataclass_fields__') else frame
                # Rename radar_pt -> x_pitch/y_pitch for each player
                players_out = []
                for p in frame_dict.get("players", []):
                    rp = p.get("radar_pt")
                    if rp and "x_pitch" not in p:
                        p = dict(p)
                        p["x_pitch"] = rp[0]
                        p["y_pitch"] = rp[1]
                    players_out.append(p)
                frame_dict["players"] = players_out
                frames_as_dicts.append(frame_dict)

            idx_path = run_event_detection(
                frames_as_dicts,
                fps=video_fps,
                job_id=output_prefix,
                output_dir=out_dir,
                metrics_timeline=metrics,
                progress_callback=progress_callback,
            )

            index = load_event_index(idx_path)
            run_threat_attribution(index, output_dir=out_dir)

            LOGGER.info(
                "Event Intelligence Layer complete: %d events, index at %s",
                len(index.events), idx_path,
            )
        except Exception as exc:
            LOGGER.warning(
                "Event Intelligence Layer failed (non-fatal): %s", exc, exc_info=True
            )

    _stage_t0 = time.perf_counter()
    await asyncio.to_thread(_run_event_layer)
    LOGGER.info(
        "PERF_STAGE",
        extra={
            "job_id": output_prefix,
            "stage": "event_layer",
            "duration_seconds": round(time.perf_counter() - _stage_t0, 3),
            "status": "ok",
        },
    )
    # ── End Event Intelligence Layer ──────────────────────────────────────────

    if progress_callback is not None:
        progress_callback("Tactical Engine")

    def _run_tactical_engine():
        if enable_zsl:
            _print_step("Running Zero-Shot Learning tactical branch exclusively...")
            triggers = []
            try:
                from scripts.zsl_classifier import ZSLTacticalClassifier
                zsl_classifier = ZSLTacticalClassifier()
                zsl_insights_0 = zsl_classifier.analyze_chunk(refined_frames, "team_0")
                zsl_insights_1 = zsl_classifier.analyze_chunk(refined_frames, "team_1")

                zsl_count = 0
                for insight in zsl_insights_0 + zsl_insights_1:
                    triggers.append(insight.model_dump())
                    zsl_count += 1

                _print_step(f"ZSL branch complete. Extracted {zsl_count} neural tactical insights.")
            except Exception as e:
                LOGGER.warning(f"ZSL branch failed, falling back to Rule Engine: {e}")
                triggers = evaluate_chunk_insights(metrics)
        else:
            _print_step("Running Standard Rule Engine...")
            triggers = evaluate_chunk_insights(metrics)
            
        return triggers

    _stage_t0 = time.perf_counter()
    triggers = await asyncio.to_thread(_run_tactical_engine)
    LOGGER.info(
        "PERF_STAGE",
        extra={
            "job_id": output_prefix,
            "stage": "tactical_engine",
            "duration_seconds": round(time.perf_counter() - _stage_t0, 3),
            "num_triggers": len(triggers),
            "status": "ok",
        },
    )

    if progress_callback is not None:
        progress_callback("Synthesizing Advice")

    def _run_synthesis():
        reliability_pct, guard_status = _print_data_guard_reliability(len(metrics), len(raw_frames))
        library_path = BACKEND_ROOT / "data" / ("zsl_tactics.json" if enable_zsl else "tactical_library.json")

        prompt_records = synthesize(
            triggers,
            library_path=library_path,
            ball_data_quality=ball_data_quality,
        )
        return reliability_pct, guard_status, prompt_records

    _stage_t0 = time.perf_counter()
    reliability_pct, guard_status, prompt_records = await asyncio.to_thread(_run_synthesis)
    LOGGER.info(
        "PERF_STAGE",
        extra={
            "job_id": output_prefix,
            "stage": "synthesis",
            "duration_seconds": round(time.perf_counter() - _stage_t0, 3),
            "num_prompt_records": len(prompt_records),
            "guard_status": guard_status,
            "status": "ok",
        },
    )

    _llm_t0 = time.perf_counter()
    if guard_status == "abort":
        final_cards = _final_cards_llm_skipped_low_reliability(
            prompt_records, reliability_pct, len(metrics), len(raw_frames)
        )
    else:
        _is_cloud_run = bool(os.getenv("K_SERVICE", "").strip())
        LOGGER.info(
            "LLM ENGINE DEBUG: provider=%s quality=%s mode=%s (cloud_run=%s guard_status=%s)",
            llm_engine, "n/a", "parallel_pipeline", _is_cloud_run, guard_status,
        )
        if llm_engine == "local" and not _is_cloud_run:
            # Local dev path: use Ollama (must be installed and running locally)
            from scripts.llm_router import ensure_ollama_available
            await ensure_ollama_available()
            from scripts.e2e_llm_local import run_llm_local
            final_cards = await run_llm_local(prompt_records)
        else:
            # Cloud Run path OR explicit cloud engine: use Gemini/OpenAI
            if llm_engine == "local" and _is_cloud_run:
                LOGGER.warning(
                    "LLM ENGINE DEBUG: llm_engine='local' requested but K_SERVICE is set — "
                    "overriding to cloud LLM (Ollama is not available on Cloud Run)."
                )
            final_cards = await run_llm(prompt_records)

    LOGGER.info(
        "PERF_STAGE",
        extra={
            "job_id": output_prefix,
            "stage": "llm_calls",
            "duration_seconds": round(time.perf_counter() - _llm_t0, 3),
            "guard_status": guard_status,
            "num_prompt_records": len(prompt_records),
            "status": "ok",
        },
    )

    # Safety net: inject baseline summary if empty
    if not final_cards and metrics:
        LOGGER.warning("Pipeline produced 0 advice cards; injecting baseline match summary.")
        rule_engine = __import__("scripts.tactical_rule_engine", fromlist=["RuleEngine"]).RuleEngine
        re = rule_engine()
        scores = re.calculate_tactical_scores(metrics)
        final_cards = [{
            "frame_idx": 0,
            "team": "global",
            "flaw": "Match Summary",
            "severity": "Info",
            "frequency_pct": 100.0,
            "confidence_pct": 100.0,
            "confidence_reason": "Baseline injection from direct telemetry scores.",
            "evidence": (
                f"Tactical Power: Red {scores['team_0'].get('tactical_power', 0):.1f} vs Blue {scores['team_1'].get('tactical_power', 0):.1f}. "
                f"Win Probability: Red {scores['team_0'].get('win_prob', 50)}% | Blue {scores['team_1'].get('win_prob', 50)}%. "
                f"Compactness: Red {scores['team_0'].get('compactness', 0):.0f} / Blue {scores['team_1'].get('compactness', 0):.0f}. "
                f"Transition Speed: Red {scores['team_0'].get('transition_speed', 0):.0f} / Blue {scores['team_1'].get('transition_speed', 0):.0f}."
            ),
            "summary_data": scores,
            "matched_philosophy_author": "GAFFER Match Engine",
            "matched_quote_excerpt": "Structured analysis based on spatial telemetry.",
            "fc_role_recommendations": [],
            "tactical_instruction": (
                "1. Analyze the win probability distribution to identify which team needs tactical adjustments. "
                "2. Focus on the team with lower compactness scores — they need to tighten their defensive shape. "
                "3. The team with higher pressing intensity should be encouraged to sustain pressure in the final third."
            ),
            "llm_error": None,
        }]

    _stage_t0 = time.perf_counter()
    with report_output_path.open("w", encoding="utf-8") as f:
        json.dump(final_cards, f, indent=2, ensure_ascii=False)
    LOGGER.info(
        "PERF_STAGE",
        extra={
            "job_id": output_prefix,
            "stage": "report_json_write",
            "duration_seconds": round(time.perf_counter() - _stage_t0, 3),
            "num_cards": len(final_cards),
            "status": "ok",
        },
    )

    # ── Report Enrichment step (non-fatal) ──────────────────────────────────
    def _run_report_enrichment():
        try:
            from event_layer.enricher import enrich_report
            enrich_report(
                report_path=report_output_path,
                job_id=output_prefix,
                output_dir=out_dir,
            )
        except Exception as exc:
            LOGGER.warning("Report enrichment failed (non-fatal): %s", exc, exc_info=True)

    _stage_t0 = time.perf_counter()
    await asyncio.to_thread(_run_report_enrichment)
    LOGGER.info(
        "PERF_STAGE",
        extra={
            "job_id": output_prefix,
            "stage": "report_enrichment",
            "duration_seconds": round(time.perf_counter() - _stage_t0, 3),
            "status": "ok",
        },
    )
    # ─────────────────────────────────────────────────────────────────────────

    LOGGER.info(
        "PERF_STAGE",
        extra={
            "job_id": output_prefix,
            "stage": "pipeline_total",
            "duration_seconds": round(time.perf_counter() - _pipeline_t0, 3),
            "num_raw_frames": len(raw_frames),
            "num_final_cards": len(final_cards),
            "status": "ok",
        },
    )

    if progress_callback is not None:
        progress_callback("Completed")

    return report_output_path

