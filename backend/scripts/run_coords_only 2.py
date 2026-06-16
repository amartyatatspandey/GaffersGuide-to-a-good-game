#!/usr/bin/env python3
"""
High-throughput tracking export: YOLO + ByteTrack + homography radar projection.

Skips VideoWriter, composed overlays, and OpticalFlowCameraShiftEstimator (CPU-heavy).
For EDA, writes a pickle with per-frame tracks (bboxes, optional radar, team labels).

Usage (single video; homography defaults to backend/output/{stem}_homographies.json or set GAFFERS_HOMOGRAPHY_JSON):
  PYTHONPATH=/workspace python backend/scripts/run_coords_only.py --video /path/to/MATCH.mp4

Batch:
  PYTHONPATH=/workspace python backend/scripts/run_coords_only.py \\
    --input-dir /workspace/data/training_samples --output-dir /workspace/output
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from calculators.possession import (  # noqa: E402
    compute_possession_team_id,
    interpolate_ball_positions,
)
from scripts.run_calibrator_on_video import ensure_homography_json_for_video  # noqa: E402
from track_teams import (  # noqa: E402
    CLASS_BALL,
    CLASS_PLAYER,
    HybridIDHealer,
    MODEL_PATH,
    TacticalRadar,
    TeamClassifier,
    format_tracking_model_missing_reason,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

HOMOGRAPHY_CONFIDENCE_FALLBACK_THRESHOLD = 0.55
BALL_INTERPOLATION_MAX_GAP = 8


@dataclass(slots=True)
class TacticalPlayer:
    id: int | None
    team: str
    radar_pt: list[float] | None


@dataclass(slots=True)
class TacticalFrame:
    frame_idx: int
    players: list[TacticalPlayer]
    ball_xy: list[float] | None
    possession_team_id: int | None


@dataclass(slots=True)
class CoordsTelemetry:
    total_frames_read: int = 0
    total_frames_inferred: int = 0
    frames_standard_homography: int = 0
    frames_low_homography_conf: int = 0
    total_raw_ball_detections: int = 0
    total_interpolated_ball_frames: int = 0


def _prediction_to_team(prediction: str) -> str | None:
    if prediction == "team_0":
        return "team_0"
    if prediction == "team_1":
        return "team_1"
    return None


def _resolve_ball_classes(model: YOLO) -> list[int]:
    names: dict[int, str] | list[str] = model.names
    class_ids: list[int] = []
    if isinstance(names, dict):
        for class_id, class_name in names.items():
            if "ball" in class_name.lower():
                class_ids.append(int(class_id))
    else:
        for class_id, class_name in enumerate(names):
            if "ball" in class_name.lower():
                class_ids.append(int(class_id))
    return class_ids


def _resolve_primary_ball_class_ids(model: YOLO) -> list[int]:
    resolved = _resolve_ball_classes(model)
    if resolved:
        return resolved
    return [32, 0]


def _homography_confidence(radar: TacticalRadar, frame_idx: int) -> float:
    available = radar.available_frames
    if not available:
        return 0.0
    if frame_idx in radar.inv_homographies:
        return 1.0

    pos = int(np.searchsorted(np.asarray(available), frame_idx))
    before = available[pos - 1] if pos > 0 else None
    after = available[pos] if pos < len(available) else None

    if before is None and after is not None:
        dist = abs(after - frame_idx)
        return float(max(0.0, 1.0 - (dist / 20.0)))
    if after is None and before is not None:
        dist = abs(frame_idx - before)
        return float(max(0.0, 1.0 - (dist / 20.0)))
    if before is None or after is None:
        return 0.0

    gap = after - before
    if gap > 10:
        return 0.25
    return float(max(0.55, 1.0 - (gap / 12.0)))


def _default_output_dir() -> Path:
    env_out = os.getenv("GAFFERS_COORDS_OUTPUT_DIR", "").strip()
    if env_out:
        return Path(env_out).expanduser().resolve()
    ws = Path("/workspace/output")
    if ws.is_dir():
        return ws
    return BACKEND_ROOT / "output"


def _default_homography_dir() -> Path:
    env_h = os.getenv("GAFFERS_HOMOGRAPHY_DIR", "").strip()
    if env_h:
        return Path(env_h).expanduser().resolve()
    ws = Path("/workspace/output")
    if ws.is_dir():
        return ws
    return BACKEND_ROOT / "output"


def _validate_homography_json(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"not a file: {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"invalid JSON: {exc}"
    if not isinstance(data, dict):
        return False, "root must be a JSON object"
    homographies = data.get("homographies")
    if not isinstance(homographies, list):
        return False, "missing or invalid 'homographies' array"
    if len(homographies) == 0:
        return False, "'homographies' is empty"
    return True, ""


def run_coords_only(
    video_path: Path,
    *,
    frame_stride: int,
) -> tuple[list[TacticalFrame], list[list[dict[str, Any]]], CoordsTelemetry]:
    """
    Run CV tracking without video encoding, drawing, or optical-flow fallback.

    When frame_stride > 1, only runs YOLO/ByteTrack on every Nth *read* frame
    (still advances through every frame for radar timeline alignment). ByteTrack
    may be less stable across gaps; use stride=1 for best continuity.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Tracking model not found: {MODEL_PATH}. {format_tracking_model_missing_reason(MODEL_PATH)}"
        )

    homography_path = ensure_homography_json_for_video(video_path)
    ok, reason = _validate_homography_json(homography_path)
    if not ok:
        raise ValueError(f"Invalid homography file {homography_path}: {reason}")

    model: YOLO = YOLO(str(MODEL_PATH))
    primary_ball_class_ids = _resolve_primary_ball_class_ids(model)
    LOGGER.info("Primary model ball class IDs: %s", primary_ball_class_ids)

    tracker = sv.ByteTrack()
    classifier = TeamClassifier()
    healer = HybridIDHealer()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    radar = TacticalRadar(json_path=homography_path, video_res=(width, height))

    telemetry = CoordsTelemetry()
    frames_out: list[TacticalFrame] = []
    track_snapshots: list[list[dict[str, Any]]] = []

    video_fi = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            telemetry.total_frames_read += 1

            radar.update_camera_angle(video_fi)
            homography_conf = _homography_confidence(radar, video_fi)
            if homography_conf >= HOMOGRAPHY_CONFIDENCE_FALLBACK_THRESHOLD:
                telemetry.frames_standard_homography += 1
            else:
                telemetry.frames_low_homography_conf += 1

            run_inference = frame_stride <= 1 or (video_fi % frame_stride == 0)

            if not run_inference:
                video_fi += 1
                continue

            ball_xy: list[float] | None = None

            results: list[Any] = model(frame, conf=0.3, verbose=False)
            telemetry.total_frames_inferred += 1

            if not results:
                frames_out.append(
                    TacticalFrame(
                        frame_idx=video_fi,
                        players=[],
                        ball_xy=ball_xy,
                        possession_team_id=None,
                    )
                )
                track_snapshots.append([])
                video_fi += 1
                continue

            detections = sv.Detections.from_ultralytics(results[0])
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
                if ball_pt is not None:
                    ball_xy = [float(ball_pt[0]), float(ball_pt[1])]

            radar_pts: list[tuple[int, int] | None] = [
                radar.map_to_2d(detections.xyxy[i]) for i in range(len(detections))
            ]
            tracker_ids = healer.process_and_heal(
                detections, frame, radar_pts, video_fi
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

            role_mapping = classifier.predict_frame(frame, frame_data, video_fi)
            tactical_players: list[TacticalPlayer] = []
            possession_team_id: int | None = None

            tracks_row: list[dict[str, Any]] = []
            for row in frame_data:
                tid = row["id"]
                pred = role_mapping.get(tid, "unknown") if tid is not None else "unknown"
                team_cls = (
                    _prediction_to_team(pred)
                    if row["cid"] == CLASS_PLAYER
                    else None
                )
                bbox = np.asarray(row["bbox"], dtype=np.float64).ravel()
                rp = row["radar_pt"]
                radar_list: list[float] | None
                if rp is not None:
                    radar_list = [float(rp[0]), float(rp[1])]
                else:
                    radar_list = None
                tracks_row.append(
                    {
                        "track_id": tid,
                        "class_id": int(row["cid"]),
                        "bbox_xyxy": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "radar_xy": radar_list,
                        "team": team_cls,
                    }
                )

                if row["cid"] in (CLASS_BALL,):
                    continue
                if row["cid"] != CLASS_PLAYER:
                    continue
                prediction = role_mapping.get(row["id"], "unknown")
                team = _prediction_to_team(prediction)
                if team is None:
                    continue
                pt_out: list[float] | None = radar_list
                tactical_players.append(
                    TacticalPlayer(id=row["id"], team=team, radar_pt=pt_out)
                )

            if ball_xy is not None:
                possession_team_id = compute_possession_team_id(
                    TacticalFrame(
                        frame_idx=video_fi,
                        players=tactical_players,
                        ball_xy=ball_xy,
                        possession_team_id=None,
                    )
                )

            frames_out.append(
                TacticalFrame(
                    frame_idx=video_fi,
                    players=tactical_players,
                    ball_xy=ball_xy,
                    possession_team_id=possession_team_id,
                )
            )
            track_snapshots.append(tracks_row)
            video_fi += 1
    finally:
        cap.release()

    telemetry.total_interpolated_ball_frames = interpolate_ball_positions(
        frames_out, max_gap_frames=BALL_INTERPOLATION_MAX_GAP
    )
    return frames_out, track_snapshots, telemetry


def _serialize_run(
    video_path: Path,
    homography_path: Path,
    frame_stride: int,
    frames: list[TacticalFrame],
    track_snapshots: list[list[dict[str, Any]]],
    telemetry: CoordsTelemetry,
) -> dict[str, Any]:
    serialized_frames: list[dict[str, Any]] = []
    for tf, tracks in zip(frames, track_snapshots, strict=True):
        serialized_frames.append(
            {
                "frame_idx": tf.frame_idx,
                "ball_xy": tf.ball_xy,
                "possession_team_id": tf.possession_team_id,
                "players_tactical": [
                    {
                        "id": p.id,
                        "team": p.team,
                        "radar_pt": p.radar_pt,
                    }
                    for p in tf.players
                ],
                "tracks": tracks,
            }
        )
    return {
        "video_path": str(video_path.resolve()),
        "video_stem": video_path.stem,
        "homography_json": str(homography_path.resolve()),
        "frame_stride": frame_stride,
        "telemetry": {
            "total_frames_read": telemetry.total_frames_read,
            "total_frames_inferred": telemetry.total_frames_inferred,
            "frames_standard_homography": telemetry.frames_standard_homography,
            "frames_low_homography_conf": telemetry.frames_low_homography_conf,
            "total_raw_ball_detections": telemetry.total_raw_ball_detections,
            "total_interpolated_ball_frames": telemetry.total_interpolated_ball_frames,
        },
        "frames": serialized_frames,
    }


def _process_one_video(
    video_path: Path,
    homography_path: Path,
    output_dir: Path,
    frame_stride: int,
) -> Path:
    os.environ["GAFFERS_HOMOGRAPHY_JSON"] = str(homography_path.resolve())
    frames, tracks, telemetry = run_coords_only(video_path, frame_stride=frame_stride)
    payload = _serialize_run(
        video_path, homography_path, frame_stride, frames, tracks, telemetry
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}_coords.pkl"
    with out_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info(
        "Wrote %s (%d inferred frames, %d read)",
        out_path,
        telemetry.total_frames_inferred,
        telemetry.total_frames_read,
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export YOLO+ByteTrack coordinates only (no video overlays)."
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Single video file (homography: --homography, GAFFERS_HOMOGRAPHY_JSON, or default output/{stem}_homographies.json).",
    )
    parser.add_argument(
        "--homography",
        type=Path,
        help="Homography JSON for --video (overrides GAFFERS_HOMOGRAPHY_JSON).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory of videos; pairs each with {stem}_homographies.json in homography dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for *_coords.pkl (default: GAFFERS_COORDS_OUTPUT_DIR or /workspace/output).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Run detector/tracker every Nth frame (1 = all frames; larger = faster, weaker tracks).",
    )
    args = parser.parse_args()

    if args.frame_stride < 1:
        LOGGER.error("--frame-stride must be >= 1")
        return 2

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir()
    )
    homography_dir = _default_homography_dir()

    if args.input_dir is not None:
        input_dir = args.input_dir.expanduser().resolve()
        if not input_dir.is_dir():
            LOGGER.error("Not a directory: %s", input_dir)
            return 2
        videos = sorted(input_dir.glob("*.mp4")) + sorted(input_dir.glob("*.MP4"))
        if not videos:
            LOGGER.error("No MP4 files in %s", input_dir)
            return 2
        ok_count = 0
        bad_count = 0
        for vp in videos:
            hp = homography_dir / f"{vp.stem}_homographies.json"
            ok, reason = _validate_homography_json(hp)
            if not ok:
                LOGGER.error("Skip %s: homography %s — %s", vp.name, hp, reason)
                bad_count += 1
                continue
            try:
                _process_one_video(vp, hp, output_dir, args.frame_stride)
                ok_count += 1
            except Exception:
                LOGGER.exception("Failed processing %s", vp)
                bad_count += 1
        LOGGER.info(
            "Batch done: %d succeeded, %d skipped/failed (of %d)",
            ok_count,
            bad_count,
            len(videos),
        )
        return 1 if ok_count == 0 else 0

    if args.video is None:
        LOGGER.error("Provide --video or --input-dir")
        return 2

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        LOGGER.error("Video not found: %s", video_path)
        return 2

    if args.homography is not None:
        homography_path = args.homography.expanduser().resolve()
    else:
        he = os.getenv("GAFFERS_HOMOGRAPHY_JSON", "").strip()
        if he:
            homography_path = Path(he).expanduser().resolve()
        else:
            homography_path = ensure_homography_json_for_video(video_path)

    ok, reason = _validate_homography_json(homography_path)
    if not ok:
        LOGGER.error("Invalid homography: %s", reason)
        return 2

    try:
        _process_one_video(video_path, homography_path, output_dir, args.frame_stride)
    except Exception:
        LOGGER.exception("Processing failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
