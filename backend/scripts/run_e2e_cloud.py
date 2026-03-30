from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT_LOCAL = SCRIPT_DIR.parent
PROJECT_ROOT_LOCAL = BACKEND_ROOT_LOCAL.parent

if str(BACKEND_ROOT_LOCAL) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT_LOCAL))

from scripts.run_e2e import (  # noqa: E402
    BALL_INTERPOLATION_MAX_GAP,
    COUNTER_ATTACK_WINDOW_FRAMES,
    FPS,
    HOMOGRAPHY_CONFIDENCE_FALLBACK_THRESHOLD,
    MIN_BALL_CONFIDENCE,
    PRESS_SUCCESS_WINDOW_FRAMES,
    TacticalFrame,
    TacticalPlayer,
    CVTelemetry,
    TrackingFrameArtifact,
    _final_cards_llm_skipped_low_reliability,
    _fallback_project_from_camera_shift,
    _homography_confidence,
    _prediction_to_team,
    _print_data_guard_reliability,
    _print_step,
    _resolve_primary_ball_class_ids,
    _resolve_video_path,
    _write_tracking_artifact,
    apply_ball_metrics_gate,
    build_metrics_timeline,
    compute_ball_visibility_ratio,
    evaluate_chunk_insights,
    run_llm,
    synthesize,
)
from scripts.global_refiner import GlobalRefiner  # noqa: E402
from scripts.track_teams import (  # noqa: E402
    BACKEND_ROOT,
    CLASS_BALL,
    CLASS_PLAYER,
    HybridIDHealer,
    MODEL_PATH,
    TacticalRadar,
    TeamClassifier,
)
from calculators.possession import (  # noqa: E402
    compute_possession_team_id,
    interpolate_ball_positions,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_BATCH_SIZE = 16
DEFAULT_FLOW_MAX_WIDTH = 640


class DownscaledOpticalFlowEstimator:
    """
    Sparse optical-flow estimator on downscaled grayscale frames.
    Returns camera shift in original-frame pixel coordinates.
    """

    def __init__(self, max_width: int = DEFAULT_FLOW_MAX_WIDTH) -> None:
        self.max_width = max(64, int(max_width))
        self.prev_gray: np.ndarray | None = None
        self.prev_features: np.ndarray | None = None
        self.lk_params: dict[str, Any] = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        }

    def _to_gray_downscaled(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        if w <= self.max_width:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return gray, 1.0
        scale = self.max_width / float(w)
        resized = cv2.resize(
            frame,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return gray, scale

    def _build_feature_mask(self, gray: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(gray)
        h, w = gray.shape
        edge_band = max(12, int(w * 0.08))
        mask[:, 0:edge_band] = 255
        mask[:, max(0, w - edge_band) : w] = 255
        if h > 0:
            top_band = max(12, int(h * 0.10))
            mask[0:top_band, :] = 255
        return mask

    def _detect_features(self, gray: np.ndarray) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=self._build_feature_mask(gray),
        )

    def update(self, frame: np.ndarray) -> tuple[float, float]:
        gray, scale = self._to_gray_downscaled(frame)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_features = self._detect_features(gray)
            return (0.0, 0.0)

        if self.prev_features is None or len(self.prev_features) == 0:
            self.prev_features = self._detect_features(self.prev_gray)
            if self.prev_features is None:
                self.prev_gray = gray
                return (0.0, 0.0)

        new_features, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_features,
            None,
            **self.lk_params,
        )
        if new_features is None or status is None:
            self.prev_gray = gray
            self.prev_features = self._detect_features(gray)
            return (0.0, 0.0)

        max_distance = 0.0
        best_dx = 0.0
        best_dy = 0.0
        for new_pt, old_pt, st in zip(new_features, self.prev_features, status, strict=False):
            if st[0] != 1:
                continue
            dx = float(new_pt[0][0] - old_pt[0][0])
            dy = float(new_pt[0][1] - old_pt[0][1])
            dist = float(np.hypot(dx, dy))
            if dist > max_distance:
                max_distance = dist
                best_dx = dx
                best_dy = dy

        self.prev_gray = gray
        self.prev_features = self._detect_features(gray)
        if scale <= 0.0:
            return (0.0, 0.0)
        inv = 1.0 / scale
        return (best_dx * inv, best_dy * inv)


def _infer_device(requested: str | None) -> str | None:
    if requested and requested != "auto":
        return requested
    if torch is None:
        return None
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _clear_device_cache(device: str | None) -> None:
    if torch is None:
        return
    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        LOGGER.debug("Device cache clear skipped", exc_info=True)


def _iter_frame_batches(
    cap: cv2.VideoCapture,
    *,
    batch_size: int,
) -> tuple[int, list[np.ndarray]]:
    idx = 0
    while True:
        batch: list[np.ndarray] = []
        start_idx = idx
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            batch.append(frame)
            idx += 1
        if not batch:
            break
        yield start_idx, batch


def run_cv_tracking_batched(
    video_path: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    flow_max_width: int = DEFAULT_FLOW_MAX_WIDTH,
    device: str | None = None,
) -> tuple[list[TacticalFrame], CVTelemetry, list[TrackingFrameArtifact]]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Tracking model not found: {MODEL_PATH}")

    selected_device = _infer_device(device)
    LOGGER.info("Cloud beta tracking device: %s", selected_device or "default")
    use_half = selected_device == "cuda"

    model: YOLO = YOLO(str(MODEL_PATH))
    if use_half and torch is not None and torch.cuda.is_available():
        try:
            model.model.half()
            LOGGER.info("Cloud beta inference precision: FP16 (CUDA)")
        except Exception:  # noqa: BLE001
            use_half = False
            LOGGER.warning("FP16 model cast failed, falling back to FP32.")
    primary_ball_class_ids = _resolve_primary_ball_class_ids(model)
    tracker = sv.ByteTrack()
    classifier = TeamClassifier()
    healer = HybridIDHealer()
    flow_estimator = DownscaledOpticalFlowEstimator(max_width=flow_max_width)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    radar = TacticalRadar(video_res=(width, height))

    telemetry = CVTelemetry()
    frames_out: list[TacticalFrame] = []
    frame_artifacts: list[TrackingFrameArtifact] = []
    last_player_radar_by_track: dict[int, tuple[int, int]] = {}
    last_ball_radar: tuple[int, int] | None = None

    try:
        for start_idx, batch_frames in _iter_frame_batches(cap, batch_size=batch_size):
            kwargs: dict[str, Any] = {"conf": 0.3, "verbose": False}
            if selected_device:
                kwargs["device"] = selected_device
            if use_half:
                kwargs["half"] = True
            try:
                results: list[Any] = model(batch_frames, **kwargs)
            except Exception:  # noqa: BLE001
                if use_half:
                    LOGGER.warning("FP16 inference failed for batch; retrying FP32.")
                    kwargs.pop("half", None)
                    results = model(batch_frames, **kwargs)
                else:
                    raise

            # Strict chronological tracking updates to preserve ID continuity.
            for offset, frame in enumerate(batch_frames):
                frame_idx = start_idx + offset
                radar.update_camera_angle(frame_idx)
                camera_shift = flow_estimator.update(frame)
                homography_conf = _homography_confidence(radar, frame_idx)
                use_fallback = homography_conf < HOMOGRAPHY_CONFIDENCE_FALLBACK_THRESHOLD
                if use_fallback:
                    telemetry.frames_optical_flow_fallback += 1
                else:
                    telemetry.frames_standard_homography += 1

                ball_xy: list[float] | None = None
                result = results[offset] if offset < len(results) else None
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
                            camera_shift_xy=(
                                float(camera_shift[0]),
                                float(camera_shift[1]),
                            ),
                        )
                    )
                    continue

                detections = sv.Detections.from_ultralytics(result)
                detections = tracker.update_with_detections(detections)
                det_conf = getattr(detections, "confidence", None)

                best_ball_bbox: np.ndarray | None = None
                best_ball_score = -1.0
                for i in range(len(detections)):
                    cid = int(detections.class_id[i])
                    if cid not in primary_ball_class_ids:
                        continue
                    score = (
                        float(det_conf[i])
                        if det_conf is not None and i < len(det_conf)
                        else 0.0
                    )
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
                        ball_xy = [float(ball_pt[0]), float(ball_pt[1])]

                # Vectorized projection with strict index-order preservation.
                radar_pts: list[tuple[int, int] | None] = radar.map_many_to_2d(
                    detections.xyxy, frame_idx=frame_idx
                )
                tracker_ids = healer.process_and_heal(
                    detections, frame, radar_pts, frame_idx
                )
                if tracker_ids is None:
                    tracker_ids = getattr(detections, "tracker_id", None)

                frame_data: list[dict[str, Any]] = []
                for i in range(len(detections)):
                    tid: int | None = None
                    if tracker_ids is not None and i < len(tracker_ids):
                        raw_id = tracker_ids[i]
                        tid = int(raw_id) if raw_id is not None else None
                    frame_data.append(
                        {
                            "id": tid,
                            "bbox": detections.xyxy[i],
                            "cid": int(detections.class_id[i]),
                            "radar_pt": radar_pts[i],
                        }
                    )

                role_mapping = classifier.predict_frame(frame, frame_data, frame_idx)
                tactical_players: list[TacticalPlayer] = []
                possession_team_id: int | None = None

                for row in frame_data:
                    if row["cid"] in (CLASS_BALL,) or row["cid"] != CLASS_PLAYER:
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
                    pt_out: list[float] | None = None
                    if pt is not None:
                        pt_out = [float(pt[0]), float(pt[1])]
                        if row["id"] is not None:
                            last_player_radar_by_track[row["id"]] = (
                                int(round(pt_out[0])),
                                int(round(pt_out[1])),
                            )
                    tactical_players.append(
                        TacticalPlayer(id=row["id"], team=team, radar_pt=pt_out)
                    )

                if ball_xy is not None:
                    possession_team_id = compute_possession_team_id(
                        TacticalFrame(
                            frame_idx=frame_idx,
                            players=tactical_players,
                            ball_xy=ball_xy,
                            possession_team_id=None,
                        )
                    )

                frames_out.append(
                    TacticalFrame(
                        frame_idx=frame_idx,
                        players=tactical_players,
                        ball_xy=ball_xy,
                        possession_team_id=possession_team_id,
                    )
                )

                player_rows: list[dict[str, Any]] = []
                for row in frame_data:
                    if row["cid"] != CLASS_PLAYER:
                        continue
                    rp = row["radar_pt"]
                    bbox = np.asarray(row["bbox"], dtype=np.float64).ravel()
                    team_label = _prediction_to_team(
                        role_mapping.get(row["id"], "unknown")
                    )
                    player_rows.append(
                        {
                            "id": row["id"],
                            "team_id": team_label,
                            "x_pitch": float(rp[0]) if rp is not None else None,
                            "y_pitch": float(rp[1]) if rp is not None else None,
                            "x_canvas": float((bbox[0] + bbox[2]) / 2.0),
                            "y_canvas": float((bbox[1] + bbox[3]) / 2.0),
                        }
                    )
                frame_artifacts.append(
                    TrackingFrameArtifact(
                        frame_idx=frame_idx,
                        players=player_rows,
                        ball_xy=ball_xy,
                        possession_team_id=possession_team_id,
                        homography_confidence=float(homography_conf),
                        used_optical_flow_fallback=use_fallback,
                        camera_shift_xy=(
                            float(camera_shift[0]),
                            float(camera_shift[1]),
                        ),
                    )
                )
                telemetry.total_frames_processed += 1

            _clear_device_cache(selected_device)
    finally:
        cap.release()

    telemetry.total_interpolated_ball_frames = interpolate_ball_positions(
        frames_out, max_gap_frames=BALL_INTERPOLATION_MAX_GAP
    )
    return frames_out, telemetry, frame_artifacts


def run_e2e_cloud(
    video: str | Path,
    *,
    output_prefix: str = "test_mp4",
    progress_callback: Callable[[str], None] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    flow_max_width: int = DEFAULT_FLOW_MAX_WIDTH,
    device: str | None = None,
) -> Path:
    if progress_callback is not None:
        progress_callback("Pending")

    video_path = video if isinstance(video, Path) else _resolve_video_path(video)

    if output_prefix == "test_mp4":
        tracking_overlay_path = BACKEND_ROOT / "output" / "test_mp4_tracking_overlay.mp4"
        tracking_data_path = BACKEND_ROOT / "output" / "test_mp4_tracking_data.json"
        metrics_output_path = BACKEND_ROOT / "output" / "tactical_metrics_e2e.json"
        report_output_path = BACKEND_ROOT / "output" / "test_mp4_report.json"
    else:
        tracking_overlay_path = BACKEND_ROOT / "output" / f"{output_prefix}_tracking_overlay.mp4"
        tracking_data_path = BACKEND_ROOT / "output" / f"{output_prefix}_tracking_data.json"
        metrics_output_path = BACKEND_ROOT / "output" / f"{output_prefix}_tactical_metrics.json"
        report_output_path = BACKEND_ROOT / "output" / f"{output_prefix}_report.json"

    # Headless cloud run: ensure no stale overlay is interpreted as new output.
    if tracking_overlay_path.exists():
        tracking_overlay_path.unlink()

    if progress_callback is not None:
        progress_callback("Tracking Players")
    raw_frames, telemetry, tracking_frames = run_cv_tracking_batched(
        video_path,
        batch_size=batch_size,
        flow_max_width=flow_max_width,
        device=device,
    )
    _print_step(f"CV Tracking Complete: Processed {len(raw_frames)} frames.")
    telemetry.total_frames_processed = len(raw_frames)
    _write_tracking_artifact(
        tracking_data_path,
        video_path=video_path,
        overlay_path=tracking_overlay_path,
        telemetry=telemetry,
        frames=tracking_frames,
    )
    _print_step(f"Tracking timeline saved to {tracking_data_path}.")

    total_cv_frames = len(raw_frames)
    metrics_before = build_metrics_timeline(raw_frames)
    refiner = GlobalRefiner()
    if progress_callback is not None:
        progress_callback("Spatial Math")
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
    raw_ball_by_frame: dict[int, tuple[list[float] | None, int | None]] = {
        frame.frame_idx: (frame.ball_xy, frame.possession_team_id) for frame in raw_frames
    }
    for frame in refined_frames:
        ball_xy, possession_team_id = raw_ball_by_frame.get(frame.frame_idx, (None, None))
        frame.ball_xy = ball_xy
        frame.possession_team_id = possession_team_id

    metrics = build_metrics_timeline(refined_frames)
    visibility_ratio = compute_ball_visibility_ratio(refined_frames)
    metrics, ball_data_quality = apply_ball_metrics_gate(
        metrics,
        refined_frames,
        visibility_ratio=visibility_ratio,
        min_ball_confidence=MIN_BALL_CONFIDENCE,
        fps=FPS,
        counter_attack_window_frames=COUNTER_ATTACK_WINDOW_FRAMES,
        press_success_window_frames=PRESS_SUCCESS_WINDOW_FRAMES,
    )
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output_path.open("w", encoding="utf-8") as f_metrics:
        json.dump(metrics, f_metrics, indent=2, ensure_ascii=False)
    _print_step(f"Metrics timeline saved to {metrics_output_path}.")

    if progress_callback is not None:
        progress_callback("Rule Engine")
    triggers = evaluate_chunk_insights(metrics)
    valid_metric_frames = len(metrics)
    reliability_pct, guard_status = _print_data_guard_reliability(
        valid_metric_frames, total_cv_frames
    )

    library_path = BACKEND_ROOT / "data" / "tactical_library.json"
    if progress_callback is not None:
        progress_callback("Synthesizing Advice")
    prompt_records = synthesize(
        triggers,
        library_path=library_path,
        ball_data_quality=ball_data_quality,
    )

    if guard_status == "abort":
        final_cards = _final_cards_llm_skipped_low_reliability(
            prompt_records,
            reliability_pct=reliability_pct,
            valid_metric_frames=valid_metric_frames,
            total_cv_frames=total_cv_frames,
        )
    else:
        final_cards = asyncio.run(run_llm(prompt_records))

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    with report_output_path.open("w", encoding="utf-8") as f_out:
        json.dump(final_cards, f_out, indent=2, ensure_ascii=False)

    if progress_callback is not None:
        progress_callback("Completed")
    return report_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cloud-optimized batched CV→Math→Rules→RAG→LLM E2E pipeline."
    )
    parser.add_argument("video", type=str, help="Video filename or path.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="test_mp4",
        help="Output artifact prefix (default: test_mp4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="YOLO inference batch size (default: 16).",
    )
    parser.add_argument(
        "--flow-max-width",
        type=int,
        default=DEFAULT_FLOW_MAX_WIDTH,
        help="Max width for downscaled optical flow frames (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="YOLO device: auto|cuda|mps|cpu (default: auto).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_e2e_cloud(
        args.video,
        output_prefix=args.output_prefix,
        progress_callback=None,
        batch_size=max(1, int(args.batch_size)),
        flow_max_width=max(64, int(args.flow_max_width)),
        device=args.device,
    )


if __name__ == "__main__":
    main()
