from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT_LOCAL = SCRIPT_DIR.parent
PROJECT_ROOT_LOCAL = BACKEND_ROOT_LOCAL.parent

load_dotenv(PROJECT_ROOT_LOCAL / ".env")
load_dotenv(BACKEND_ROOT_LOCAL / ".env")

# Ensure backend root is importable so `from models import ...` works.
if str(BACKEND_ROOT_LOCAL) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT_LOCAL))

from llm_service import generate_coaching_advice, gemini_is_configured  # noqa: E402

from generate_analytics import TacticalAnalyzer
from global_refiner import GlobalRefiner
from rag_coach import (
    GeneratedPromptRecord,
    TacticalLibrary,
    load_json,
    process_insights,
)
from models import ChunkTacticalInsight
from tactical_rule_engine import evaluate_timeline as evaluate_chunk_insights
from track_teams import (
    BACKEND_ROOT,
    CLASS_BALL,
    CLASS_PLAYER,
    HybridIDHealer,
    MODEL_PATH,
    TacticalRadar,
    TeamClassifier,
)

from calculators.advanced_ball_metrics import (
    compute_advanced_ball_metrics,
    in_zone14,
    zone14_bounds_for_team,
)
from calculators.ball_visibility import (
    apply_ball_metrics_gate,
    compute_ball_visibility_ratio,
)
from calculators.possession import (
    compute_possession_team_id,
    interpolate_ball_positions,
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
class CVTelemetry:
    total_frames_processed: int = 0
    frames_standard_homography: int = 0
    frames_optical_flow_fallback: int = 0
    total_raw_ball_detections: int = 0
    total_interpolated_ball_frames: int = 0


class OpticalFlowCameraShiftEstimator:
    """
    Estimate frame-to-frame camera translation using sparse optical flow.
    """

    def __init__(self) -> None:
        self.minimum_distance_px = 5.0
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

    def _build_feature_mask(self, gray: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(gray)
        h, w = gray.shape
        edge_band = max(20, int(w * 0.08))
        mask[:, 0:edge_band] = 255
        mask[:, max(0, w - edge_band) : w] = 255
        if h > 0:
            top_band = max(20, int(h * 0.10))
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
        """
        Return camera shift (dx, dy) in image pixels for the current frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        for new_pt, old_pt, st in zip(
            new_features, self.prev_features, status, strict=False
        ):
            if int(st[0]) == 0:
                continue
            new_xy = new_pt.ravel()
            old_xy = old_pt.ravel()
            dx = float(new_xy[0] - old_xy[0])
            dy = float(new_xy[1] - old_xy[1])
            dist = float(np.hypot(dx, dy))
            if dist > max_distance:
                max_distance = dist
                best_dx = dx
                best_dy = dy

        self.prev_gray = gray
        if max_distance > self.minimum_distance_px:
            self.prev_features = self._detect_features(gray)
            return (best_dx, best_dy)
        return (0.0, 0.0)


def _print_step(message: str) -> None:
    print(f"[✓] {message}")


RELIABILITY_WARN_PCT = 85.0
RELIABILITY_ABORT_PCT = 70.0
# Production threshold: gate advanced ball metrics on reliable ball visibility.
MIN_BALL_CONFIDENCE = 0.85
FPS = 25
COUNTER_ATTACK_WINDOW_FRAMES = 8 * FPS
PRESS_SUCCESS_WINDOW_FRAMES = 5 * FPS


def _print_data_guard_reliability(
    valid_metric_frames: int, total_cv_frames: int
) -> tuple[float, Literal["ok", "warn", "abort"]]:
    """
    Report how many CV frames produced valid dual-team metric rows (Data Guard pass rate).

    Returns:
        (reliability_pct, status). ``abort`` means the LLM step must be skipped.
    """
    if total_cv_frames <= 0:
        reliability_pct = 0.0
    else:
        reliability_pct = (valid_metric_frames / total_cv_frames) * 100.0

    print(
        f"[🛡️] Data Guard Reliability: {reliability_pct:.1f}% "
        f"({valid_metric_frames}/{total_cv_frames})"
    )

    use_color = sys.stdout.isatty()
    yellow = "\033[33m" if use_color else ""
    red = "\033[31m" if use_color else ""
    reset = "\033[0m" if use_color else ""

    if reliability_pct < RELIABILITY_ABORT_PCT:
        print(
            f"{red}[❌] ERROR: Data quality too low for tactical analysis.{reset}"
        )
        return reliability_pct, "abort"
    if reliability_pct < RELIABILITY_WARN_PCT:
        print(
            f"{yellow}[⚠️] WARNING: Low data quality. AI advice may be hallucinated "
            f"due to tracking loss.{reset}"
        )
        return reliability_pct, "warn"
    return reliability_pct, "ok"


def _calc_reliability_pct(valid_metric_frames: int, total_cv_frames: int) -> float:
    """Return percentage of CV frames that produced valid metric frames."""
    if total_cv_frames <= 0:
        return 0.0
    return (valid_metric_frames / total_cv_frames) * 100.0


def _final_cards_llm_skipped_low_reliability(
    prompt_records: list[GeneratedPromptRecord],
    reliability_pct: float,
    valid_metric_frames: int,
    total_cv_frames: int,
) -> list[dict[str, Any]]:
    """Build report rows without calling the LLM (Data Guard abort)."""

    msg = (
        f"Skipped: Data Guard reliability {reliability_pct:.1f}% "
        f"({valid_metric_frames}/{total_cv_frames}) is below "
        f"{RELIABILITY_ABORT_PCT:.0f}% threshold."
    )
    return [
        {
            **r.model_dump(),
            "tactical_instruction": None,
            "llm_error": msg,
        }
        for r in prompt_records
    ]


def _resolve_video_path(video_name: str) -> Path:
    candidate = Path(video_name)
    if candidate.is_file():
        return candidate
    env_dir = os.getenv("GAFFERS_VIDEO_INPUT_DIR", "").strip()
    if env_dir:
        env_candidate = Path(env_dir).expanduser() / video_name
        if env_candidate.is_file():
            return env_candidate.resolve()
    backend_data_path = BACKEND_ROOT / "data" / video_name
    if backend_data_path.is_file():
        return backend_data_path
    if video_name == "test.mp4":
        alias = BACKEND_ROOT / "data" / "match_test.mp4"
        if alias.is_file():
            print("[i] test.mp4 not found, using backend/data/match_test.mp4")
            return alias
    raise FileNotFoundError(
        f"Video not found: {video_name}. Tried local path and {backend_data_path}"
    )


def _prediction_to_team(prediction: str) -> str | None:
    if prediction == "team_0":
        return "team_0"
    if prediction == "team_1":
        return "team_1"
    return None


def _team_to_id(team: str) -> int | None:
    if team == "team_0":
        return 0
    if team == "team_1":
        return 1
    return None


def _resolve_ball_classes(model: YOLO) -> list[int]:
    """
    Return class IDs that correspond to the football class.
    """
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
    """
    Resolve likely ball class IDs for the primary tracking model.
    """
    resolved = _resolve_ball_classes(model)
    if resolved:
        return resolved
    # Fallback for common COCO-like IDs if names are absent/unexpected.
    return [32, 0]


def _homography_confidence(radar: TacticalRadar, frame_idx: int) -> float:
    """
    Heuristic confidence score for frame-wise homography projection.
    """
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


def _fallback_project_from_camera_shift(
    bbox: np.ndarray,
    last_radar_pt: tuple[int, int] | None,
    camera_shift_xy: tuple[float, float],
    video_wh: tuple[int, int],
    radar_wh: tuple[int, int],
) -> tuple[int, int] | None:
    """
    Approximate radar point by compensating previous radar point with camera shift.
    """
    if last_radar_pt is None:
        return None
    video_w, video_h = max(video_wh[0], 1), max(video_wh[1], 1)
    radar_w, radar_h = radar_wh
    shift_x = camera_shift_xy[0] * (radar_w / float(video_w))
    shift_y = camera_shift_xy[1] * (radar_h / float(video_h))
    radar_x = int(round(last_radar_pt[0] - shift_x))
    radar_y = int(round(last_radar_pt[1] - shift_y))
    radar_x = max(0, min(radar_w, radar_x))
    radar_y = max(0, min(radar_h, radar_y))
    _ = bbox  # Explicitly keep bbox in signature for future scale-aware fallback.
    return (radar_x, radar_y)


def _compute_possession_team_id(frame: TacticalFrame) -> int | None:
    # Delegate to the calculator module to keep orchestration logic out of this file.
    return compute_possession_team_id(frame)


def _interpolate_ball_positions(
    frames: list[TacticalFrame], max_gap_frames: int
) -> int:
    """
    Fill short missing ball tracks via linear interpolation, then backfill possession.
    """
    # Delegate to the calculator module to keep orchestration logic out of this file.
    return interpolate_ball_positions(frames, max_gap_frames=max_gap_frames)


def _compute_ball_visibility_ratio(frames: list[TacticalFrame]) -> float:
    """
    Share of frames with valid ball coordinates (including interpolated points).
    """
    # Delegate to the calculator module to keep orchestration logic out of this file.
    return compute_ball_visibility_ratio(frames)


def _team_forward_progress(team_id: int, start_x: float, end_x: float) -> float:
    """Return positive forward progression for a team along radar X."""
    if team_id == 0:
        return end_x - start_x
    return start_x - end_x


def _is_defensive_third(team_id: int, ball_x: float) -> bool:
    """Check whether ball is in a team's defensive third on 1050x680 radar."""
    if team_id == 0:
        return ball_x <= 350.0
    return ball_x >= 700.0


def _zone14_bounds_for_team(team_id: int) -> tuple[float, float, float, float]:
    # Delegate to the calculator module.
    return zone14_bounds_for_team(team_id)


def _in_zone14(team_id: int, ball_x: float, ball_y: float) -> bool:
    # Delegate to the calculator module.
    return in_zone14(team_id, ball_x, ball_y)


def _nearest_player_distance(
    frame: TacticalFrame, *, team_id: int, ball_xy: list[float]
) -> float | None:
    """
    Euclidean distance from ball to nearest player on a team for one frame.
    """
    team_name = "team_0" if team_id == 0 else "team_1"
    bx, by = float(ball_xy[0]), float(ball_xy[1])
    dists: list[float] = []
    for player in frame.players:
        if player.team != team_name or player.radar_pt is None:
            continue
        px, py = float(player.radar_pt[0]), float(player.radar_pt[1])
        dists.append(float(np.hypot(px - bx, py - by)))
    if not dists:
        return None
    return min(dists)


def _compute_advanced_ball_metrics(
    frames: list[TacticalFrame],
) -> dict[int, dict[str, float]]:
    """
    Compute advanced ball-dependent tactical metrics from refined frame stream.
    """
    # Delegated to `backend/calculators/advanced_ball_metrics.py` to keep this
    # script focused on orchestration rather than tactical math.
    return compute_advanced_ball_metrics(
        frames,
        fps=FPS,
        counter_attack_window_frames=COUNTER_ATTACK_WINDOW_FRAMES,
        press_success_window_frames=PRESS_SUCCESS_WINDOW_FRAMES,
    )

    stats: dict[int, dict[str, float]] = {
        0: {
            "rapid_counter_attacks": 0.0,
            "high_press_success": 0.0,
            "lethargic_press_allowed": 0.0,
            "second_ball_won": 0.0,
            "second_ball_lost": 0.0,
            "zone14_penetrations": 0.0,
        },
        1: {
            "rapid_counter_attacks": 0.0,
            "high_press_success": 0.0,
            "lethargic_press_allowed": 0.0,
            "second_ball_won": 0.0,
            "second_ball_lost": 0.0,
            "zone14_penetrations": 0.0,
        },
    }

    n = len(frames)
    if n <= 1:
        return stats

    # --- Rangnick 8-second counter attacks & pressing events from turnovers.
    lethargic_state: dict[int, dict[str, float | int | None]] = {
        0: {"streak": 0, "start_x": None},
        1: {"streak": 0, "start_x": None},
    }
    for i in range(1, n):
        prev_team = frames[i - 1].possession_team_id
        curr_team = frames[i].possession_team_id
        curr_ball = frames[i].ball_xy
        if curr_ball is None:
            continue
        ball_x = float(curr_ball[0])

        if prev_team in (0, 1) and curr_team in (0, 1) and prev_team != curr_team:
            win_team = int(curr_team)
            start_ball = frames[i].ball_xy
            end_idx = min(n - 1, i + COUNTER_ATTACK_WINDOW_FRAMES)
            if start_ball is not None and frames[end_idx].ball_xy is not None:
                start_x = float(start_ball[0])
                end_x = float(frames[end_idx].ball_xy[0])  # type: ignore[index]
                if _team_forward_progress(win_team, start_x, end_x) > 35.0:
                    stats[win_team]["rapid_counter_attacks"] += 1.0

            # Press success: prior pressing team is the one not in possession.
            pressing_team = int(prev_team)
            nearest = _nearest_player_distance(
                frames[i - 1],
                team_id=pressing_team,
                ball_xy=frames[i - 1].ball_xy
                if frames[i - 1].ball_xy is not None
                else curr_ball,
            )
            if nearest is not None and nearest < 2.5:
                lookahead_end = min(n - 1, i + PRESS_SUCCESS_WINDOW_FRAMES)
                success = False
                for j in range(i, lookahead_end + 1):
                    p_team = frames[j - 1].possession_team_id if j > 0 else None
                    c_team = frames[j].possession_team_id
                    if (
                        p_team in (0, 1)
                        and c_team in (0, 1)
                        and p_team != c_team
                        and c_team == pressing_team
                    ):
                        success = True
                        break
                if success:
                    stats[pressing_team]["high_press_success"] += 1.0

        # Lethargic press allowed while opponent builds in defensive third.
        if curr_team in (0, 1):
            poss_team = int(curr_team)
            pressing_team = 1 - poss_team
            if _is_defensive_third(poss_team, ball_x):
                nearest_press = _nearest_player_distance(
                    frames[i], team_id=pressing_team, ball_xy=curr_ball
                )
                state = lethargic_state[pressing_team]
                if nearest_press is not None and nearest_press > 3.0:
                    if state["streak"] == 0:
                        state["start_x"] = ball_x
                    state["streak"] = int(state["streak"]) + 1
                else:
                    state["streak"] = 0
                    state["start_x"] = None

                if int(state["streak"]) > 100 and state["start_x"] is not None:
                    start_x = float(state["start_x"])
                    forward = _team_forward_progress(poss_team, start_x, ball_x)
                    if forward > 10.0:
                        stats[pressing_team]["lethargic_press_allowed"] += 1.0
                        state["streak"] = 0
                        state["start_x"] = None
            else:
                state = lethargic_state[pressing_team]
                state["streak"] = 0
                state["start_x"] = None

    # --- Direct play second-ball win rate.
    i = 1
    while i < n:
        curr_team = frames[i].possession_team_id
        prev_team = frames[i - 1].possession_team_id
        if curr_team not in (0, 1) or prev_team != curr_team:
            i += 1
            continue
        start_ball = frames[i - 1].ball_xy
        if start_ball is None:
            i += 1
            continue

        max_j = min(n - 1, i + FPS)
        long_pass_idx: int | None = None
        for j in range(i, max_j + 1):
            if frames[j].possession_team_id != curr_team or frames[j].ball_xy is None:
                break
            dist = float(
                np.hypot(
                    float(frames[j].ball_xy[0]) - float(start_ball[0]),
                    float(frames[j].ball_xy[1]) - float(start_ball[1]),
                )
            )
            if dist > 30.0:
                long_pass_idx = j
                break
        if long_pass_idx is None:
            i += 1
            continue

        stabilize_end = min(n - 1, long_pass_idx + 20)
        stable_idx: int | None = None
        for j in range(long_pass_idx + 1, stabilize_end + 1):
            if frames[j - 1].ball_xy is None or frames[j].ball_xy is None:
                continue
            step = float(
                np.hypot(
                    float(frames[j].ball_xy[0]) - float(frames[j - 1].ball_xy[0]),
                    float(frames[j].ball_xy[1]) - float(frames[j - 1].ball_xy[1]),
                )
            )
            if step < 5.0:
                stable_idx = j
                break
        if stable_idx is not None:
            final_team = frames[stable_idx].possession_team_id
            init_team = int(curr_team)
            if final_team == init_team:
                stats[init_team]["second_ball_won"] += 1.0
            elif final_team in (0, 1) and final_team != init_team:
                stats[init_team]["second_ball_lost"] += 1.0
        i = long_pass_idx + 1

    # --- Zone 14 penetrations by distinct possession sequences.
    for team_id in (0, 1):
        in_zone_seq = False
        for frame in frames:
            if frame.possession_team_id != team_id or frame.ball_xy is None:
                in_zone_seq = False
                continue
            bx, by = float(frame.ball_xy[0]), float(frame.ball_xy[1])
            if _in_zone14(team_id, bx, by):
                if not in_zone_seq:
                    stats[team_id]["zone14_penetrations"] += 1.0
                    in_zone_seq = True
            else:
                in_zone_seq = False

    return stats


def _apply_ball_metrics_gate(
    timeline: list[dict[str, Any]],
    frames: list[TacticalFrame],
    *,
    visibility_ratio: float,
) -> tuple[list[dict[str, Any]], Literal["sufficient", "insufficient"]]:
    """
    Gate ball-dependent metrics based on visibility confidence.
    """
    # Delegated to `backend/calculators/ball_visibility.py`.
    return apply_ball_metrics_gate(
        timeline,
        frames,
        visibility_ratio=visibility_ratio,
        min_ball_confidence=MIN_BALL_CONFIDENCE,
        fps=FPS,
        counter_attack_window_frames=COUNTER_ATTACK_WINDOW_FRAMES,
        press_success_window_frames=PRESS_SUCCESS_WINDOW_FRAMES,
    )

    quality: Literal["sufficient", "insufficient"] = (
        "sufficient" if visibility_ratio >= MIN_BALL_CONFIDENCE else "insufficient"
    )
    enabled = quality == "sufficient"

    team_ids: tuple[str, str] = ("team_0", "team_1")
    possession_counts: dict[int, int] = {0: 0, 1: 0}
    valid_possession_frames = 0
    turnovers = 0
    previous_possession: int | None = None

    for row in timeline:
        possession_team_id = row.get("possession_team_id")
        if possession_team_id in (0, 1):
            valid_possession_frames += 1
            possession_counts[int(possession_team_id)] += 1
            if previous_possession is not None and possession_team_id != previous_possession:
                turnovers += 1
            previous_possession = int(possession_team_id)

    team_possession_pct: dict[int, float] = {
        0: (possession_counts[0] / valid_possession_frames * 100.0)
        if valid_possession_frames > 0
        else 0.0,
        1: (possession_counts[1] / valid_possession_frames * 100.0)
        if valid_possession_frames > 0
        else 0.0,
    }

    advanced = _compute_advanced_ball_metrics(frames) if enabled else None

    for row in timeline:
        row["ball_visibility_ratio"] = visibility_ratio
        row["ball_data_quality"] = quality
        for team_idx, team_key in enumerate(team_ids):
            team_metrics = row.get(team_key)
            if not isinstance(team_metrics, dict):
                continue
            if enabled:
                team_metrics["possession_pct"] = float(team_possession_pct[team_idx])
                team_metrics["turnovers"] = float(turnovers)
                team_metrics["high_press_success"] = advanced[team_idx][  # type: ignore[index]
                    "high_press_success"
                ]
                team_metrics["counter_attack_velocity"] = advanced[team_idx][  # type: ignore[index]
                    "rapid_counter_attacks"
                ]
                team_metrics["rapid_counter_attacks"] = advanced[team_idx][  # type: ignore[index]
                    "rapid_counter_attacks"
                ]
                team_metrics["lethargic_press_allowed"] = advanced[team_idx][  # type: ignore[index]
                    "lethargic_press_allowed"
                ]
                team_metrics["second_ball_won"] = advanced[team_idx][  # type: ignore[index]
                    "second_ball_won"
                ]
                team_metrics["second_ball_lost"] = advanced[team_idx][  # type: ignore[index]
                    "second_ball_lost"
                ]
                team_metrics["zone14_penetrations"] = advanced[team_idx][  # type: ignore[index]
                    "zone14_penetrations"
                ]
            else:
                team_metrics["possession_pct"] = None
                team_metrics["turnovers"] = 0.0
                team_metrics["high_press_success"] = None
                team_metrics["counter_attack_velocity"] = None
                team_metrics["rapid_counter_attacks"] = 0.0
                team_metrics["lethargic_press_allowed"] = 0.0
                team_metrics["second_ball_won"] = 0.0
                team_metrics["second_ball_lost"] = 0.0
                team_metrics["zone14_penetrations"] = 0.0
    return timeline, quality


def _draw_annotated_frame(
    frame: np.ndarray,
    players: list[dict[str, Any]],
    ball_xy: list[float] | None,
    radar: TacticalRadar,
    role_mapping: dict[Any, str],
) -> np.ndarray:
    """
    Draw 2D annotations and side-by-side tactical radar overlay.
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]
    radar_img = radar.draw_blank_pitch()

    for row in players:
        bbox = row["bbox"].astype(int)
        tid = row["id"]
        cid = row["cid"]
        if cid != CLASS_PLAYER:
            continue
        role = role_mapping.get(tid, "unknown")
        color = (128, 128, 128)
        if role == "team_0":
            color = (0, 0, 255)
        elif role == "team_1":
            color = (255, 0, 0)
        label = f"ID:{tid} {role}" if tid is not None else role
        cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(
            annotated,
            label,
            (bbox[0], max(10, bbox[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        if row.get("radar_pt") is not None:
            rp = row["radar_pt"]
            cv2.circle(radar_img, (int(rp[0]), int(rp[1])), 5, color, -1)

    if ball_xy is not None:
        cv2.circle(radar_img, (int(ball_xy[0]), int(ball_xy[1])), 4, (255, 255, 0), -1)

    out_h = max(h, radar.radar_h)
    out_w = w + radar.radar_w
    composed = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    composed[0:h, 0:w] = annotated
    composed[0 : radar.radar_h, w : w + radar.radar_w] = radar_img
    return composed


def run_cv_tracking(
    video_path: Path,
) -> tuple[list[TacticalFrame], CVTelemetry]:
    """
    Run CV tracking in-memory and return TacticalFrame timeline.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Tracking model not found: {MODEL_PATH}")

    model: YOLO = YOLO(str(MODEL_PATH))
    primary_ball_class_ids = _resolve_primary_ball_class_ids(model)
    LOGGER.info("Primary model ball class IDs: %s", primary_ball_class_ids)
    tracker = sv.ByteTrack()
    classifier = TeamClassifier()
    healer = HybridIDHealer()
    flow_estimator = OpticalFlowCameraShiftEstimator()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    radar = TacticalRadar(video_res=(width, height))
    last_player_radar_by_track: dict[int, tuple[int, int]] = {}
    last_ball_radar: tuple[int, int] | None = None
    telemetry = CVTelemetry()
    annotated_output_path = BACKEND_ROOT / "output" / "test_mp4_tracking_overlay.mp4"
    annotated_output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_writer: cv2.VideoWriter | None = None

    frames_out: list[TacticalFrame] = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            radar.update_camera_angle(frame_idx)
            camera_shift = flow_estimator.update(frame)
            homography_conf = _homography_confidence(radar, frame_idx)
            use_fallback = (
                homography_conf < HOMOGRAPHY_CONFIDENCE_FALLBACK_THRESHOLD
            )
            if use_fallback:
                telemetry.frames_optical_flow_fallback += 1
            else:
                telemetry.frames_standard_homography += 1

            ball_xy: list[float] | None = None

            results: list[Any] = model(frame, conf=0.3, verbose=False)
            if not results:
                if annotated_writer is None:
                    out_h = max(height, radar.radar_h)
                    out_w = width + radar.radar_w
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    annotated_writer = cv2.VideoWriter(
                        str(annotated_output_path),
                        fourcc,
                        max(1.0, float(cap.get(cv2.CAP_PROP_FPS))),
                        (out_w, out_h),
                    )
                frames_out.append(
                    TacticalFrame(
                        frame_idx=frame_idx,
                        players=[],
                        ball_xy=ball_xy,
                        possession_team_id=None,
                    )
                )
                empty_comp = _draw_annotated_frame(
                    frame=frame,
                    players=[],
                    ball_xy=ball_xy,
                    radar=radar,
                    role_mapping={},
                )
                if annotated_writer is not None:
                    annotated_writer.write(empty_comp)
                frame_idx += 1
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

            radar_pts: list[tuple[int, int] | None] = [
                radar.map_to_2d(detections.xyxy[i]) for i in range(len(detections))
            ]
            tracker_ids = healer.process_and_heal(detections, frame, radar_pts, frame_idx)
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
                if row["cid"] in (CLASS_BALL,):
                    continue
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
            if annotated_writer is None:
                out_h = max(height, radar.radar_h)
                out_w = width + radar.radar_w
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                annotated_writer = cv2.VideoWriter(
                    str(annotated_output_path),
                    fourcc,
                    max(1.0, float(cap.get(cv2.CAP_PROP_FPS))),
                    (out_w, out_h),
                )
            composed = _draw_annotated_frame(
                frame=frame,
                players=frame_data,
                ball_xy=ball_xy,
                radar=radar,
                role_mapping=role_mapping,
            )
            if annotated_writer is not None:
                annotated_writer.write(composed)
            frame_idx += 1
            telemetry.total_frames_processed += 1
    finally:
        cap.release()
        if annotated_writer is not None:
            annotated_writer.release()

    telemetry.total_interpolated_ball_frames = interpolate_ball_positions(
        frames_out, max_gap_frames=BALL_INTERPOLATION_MAX_GAP
    )
    return frames_out, telemetry


def build_metrics_timeline(raw_frames: list[TacticalFrame]) -> list[dict[str, Any]]:
    """
    Compute tactical metrics from in-memory TacticalFrame data.
    """
    analyzer = TacticalAnalyzer()
    timeline: list[dict[str, Any]] = []

    for frame in raw_frames:
        t0_pts: list[list[float]] = []
        t1_pts: list[list[float]] = []
        t0_speeds: list[float] = []
        t1_speeds: list[float] = []

        for p in frame.players:
            if p.radar_pt is None or p.id is None:
                continue
            speed = analyzer.calc_speed(p.id, frame.frame_idx, p.radar_pt)
            if p.team == "team_0":
                t0_pts.append(p.radar_pt)
                t0_speeds.append(speed)
            elif p.team == "team_1":
                t1_pts.append(p.radar_pt)
                t1_speeds.append(speed)

        metrics_0 = analyzer.analyze_team_spatial(t0_pts)
        metrics_1 = analyzer.analyze_team_spatial(t1_pts)
        if not metrics_0 or not metrics_1:
            continue

        t0_pct, t1_pct = analyzer.calculate_pitch_control(t0_pts, t1_pts)
        t0_def_mid, t0_mid_att = analyzer.calculate_line_gaps(t0_pts)
        t1_def_mid, t1_mid_att = analyzer.calculate_line_gaps(t1_pts)

        metrics_0.update(
            {
                "pitch_control_pct": t0_pct,
                "pressure_index_m": analyzer.calculate_pressure_index(t0_pts, t1_pts),
                "line_gap_def_mid_m": t0_def_mid,
                "line_gap_mid_att_m": t0_mid_att,
                "avg_speed_kmh": float(np.mean(t0_speeds)) if t0_speeds else 0.0,
                "max_speed_kmh": float(np.max(t0_speeds)) if t0_speeds else 0.0,
            }
        )
        metrics_1.update(
            {
                "pitch_control_pct": t1_pct,
                "pressure_index_m": analyzer.calculate_pressure_index(t1_pts, t0_pts),
                "line_gap_def_mid_m": t1_def_mid,
                "line_gap_mid_att_m": t1_mid_att,
                "avg_speed_kmh": float(np.mean(t1_speeds)) if t1_speeds else 0.0,
                "max_speed_kmh": float(np.max(t1_speeds)) if t1_speeds else 0.0,
            }
        )
        timeline.append(
            {
                "frame_idx": frame.frame_idx,
                "team_0": metrics_0,
                "team_1": metrics_1,
                "ball_x": (frame.ball_xy[0] if frame.ball_xy is not None else None),
                "ball_y": (frame.ball_xy[1] if frame.ball_xy is not None else None),
                "possession_team_id": frame.possession_team_id,
            }
        )

    return timeline


def synthesize(
    triggers: list[dict[str, Any]],
    library_path: Path,
    *,
    ball_data_quality: Literal["sufficient", "insufficient"],
) -> list[GeneratedPromptRecord]:
    """
    Build RAG prompt payloads from trigger timeline and tactical library.
    """
    raw_library = load_json(library_path)
    library = TacticalLibrary.model_validate(raw_library)
    insights = [
        ChunkTacticalInsight.model_validate(
            {**row, "ball_data_quality": ball_data_quality}
        )
        for row in triggers
    ]
    return process_insights(insights=insights, library=library)


def _resolve_llm_credentials() -> tuple[str | None, str, str | None]:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
    return api_key, model, base_url


async def _complete_prompt(
    client: AsyncOpenAI, model: str, prompt: str
) -> tuple[str | None, str | None]:
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35,
            max_tokens=600,
        )
        content = response.choices[0].message.content
        return (content.strip() if content else None, None)
    except Exception as exc:  # noqa: BLE001
        return (None, str(exc))


async def _complete_gemini_prompt(prompt: str) -> tuple[str | None, str | None]:
    """Run Gemini in a worker thread (SDK is synchronous)."""

    try:
        text = await asyncio.to_thread(generate_coaching_advice, prompt)
        return (text if text else None, None)
    except Exception as exc:  # noqa: BLE001
        return (None, str(exc))


async def run_llm(records: list[GeneratedPromptRecord]) -> list[dict[str, Any]]:
    """
    Run cloud LLM completions for each generated prompt record.

    Prefers ``GEMINI_API_KEY`` (Google Gemini). Falls back to OpenAI-compatible
    APIs when ``LLM_API_KEY`` / ``OPENAI_API_KEY`` is set and Gemini is not.
    """
    if gemini_is_configured():
        tasks = [_complete_gemini_prompt(r.llm_prompt) for r in records]
        results = await asyncio.gather(*tasks) if tasks else []
        out: list[dict[str, Any]] = []
        for record, (instruction, llm_error) in zip(records, results, strict=True):
            payload = record.model_dump()
            payload["tactical_instruction"] = instruction
            payload["llm_error"] = llm_error
            out.append(payload)
        return out

    api_key, model, base_url = _resolve_llm_credentials()
    if not api_key:
        return [
            {
                **r.model_dump(),
                "tactical_instruction": None,
                "llm_error": (
                    "Missing GEMINI_API_KEY or LLM_API_KEY/OPENAI_API_KEY; "
                    "skipped cloud completion."
                ),
            }
            for r in records
        ]

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = AsyncOpenAI(**kwargs)

    tasks = [_complete_prompt(client, model, r.llm_prompt) for r in records]
    results = await asyncio.gather(*tasks) if tasks else []

    out_openai: list[dict[str, Any]] = []
    for record, (instruction, llm_error) in zip(records, results, strict=True):
        payload = record.model_dump()
        payload["tactical_instruction"] = instruction
        payload["llm_error"] = llm_error
        out_openai.append(payload)
    return out_openai


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete CV→Math→Rules→RAG→LLM E2E pipeline."
    )
    parser.add_argument(
        "video",
        type=str,
        help="Video filename or path (example: test.mp4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = _resolve_video_path(args.video)

    raw_frames, telemetry = run_cv_tracking(video_path)
    _print_step(f"CV Tracking Complete: Processed {len(raw_frames)} frames.")
    telemetry.total_frames_processed = len(raw_frames)

    total_cv_frames = len(raw_frames)
    metrics_before = build_metrics_timeline(raw_frames)
    reliability_before = _calc_reliability_pct(len(metrics_before), total_cv_frames)
    print(
        f"[🛡️] Before Refinement Reliability: {reliability_before:.1f}% "
        f"({len(metrics_before)}/{total_cv_frames})"
    )

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
    raw_ball_by_frame: dict[int, tuple[list[float] | None, int | None]] = {
        frame.frame_idx: (frame.ball_xy, frame.possession_team_id) for frame in raw_frames
    }
    for frame in refined_frames:
        ball_xy, possession_team_id = raw_ball_by_frame.get(
            frame.frame_idx, (None, None)
        )
        frame.ball_xy = ball_xy
        frame.possession_team_id = possession_team_id
    _print_step(
        f"Global Refinement Complete: Processed {len(refined_frames)} frames for trajectory healing."
    )

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
    engine_state = "ENABLED" if ball_data_quality == "sufficient" else "DISABLED"
    print(
        f"[Data Guard] Ball Visibility: {visibility_ratio * 100.0:.1f}%. "
        f"Ball Metrics Engine: [{engine_state}]"
    )
    reliability_after = _calc_reliability_pct(len(metrics), total_cv_frames)
    print(
        f"[🛡️] After Refinement Reliability: {reliability_after:.1f}% "
        f"({len(metrics)}/{total_cv_frames})"
    )
    metrics_output_path = BACKEND_ROOT / "output" / "tactical_metrics_e2e.json"
    with metrics_output_path.open("w", encoding="utf-8") as f_metrics:
        json.dump(metrics, f_metrics, indent=2, ensure_ascii=False)
    _print_step(f"Metrics timeline saved to {metrics_output_path}.")

    triggers = evaluate_chunk_insights(metrics)
    flaw_count = len(triggers)
    _print_step(
        f"Analytics Complete: Built {len(metrics)} metric frames and found {flaw_count} chunk-level tactical flaws."
    )

    valid_metric_frames = len(metrics)
    reliability_pct, guard_status = _print_data_guard_reliability(
        valid_metric_frames, total_cv_frames
    )

    library_path = BACKEND_ROOT / "data" / "tactical_library.json"
    prompt_records = synthesize(
        triggers,
        library_path=library_path,
        ball_data_quality=ball_data_quality,
    )
    _print_step(f"RAG Complete: Generated {len(prompt_records)} coaching prompts.")

    if guard_status == "abort":
        final_cards = _final_cards_llm_skipped_low_reliability(
            prompt_records,
            reliability_pct=reliability_pct,
            valid_metric_frames=valid_metric_frames,
            total_cv_frames=total_cv_frames,
        )
        _print_step("LLM Skipped: Data Guard reliability below minimum threshold.")
    else:
        final_cards = asyncio.run(run_llm(prompt_records))

    output_path = BACKEND_ROOT / "output" / "test_mp4_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_cards, f, indent=2, ensure_ascii=False)

    if guard_status != "abort":
        _print_step(f"LLM Complete: Report saved to {output_path}.")
    else:
        _print_step(f"Report saved to {output_path} (LLM not invoked).")

    print("\n" + "=" * 72)
    print("CV TELEMETRY SUMMARY")
    print("=" * 72)
    print(f"Total Frames Processed: {telemetry.total_frames_processed}")
    print(f"Frames using standard Homography: {telemetry.frames_standard_homography}")
    print(
        f"Frames using Optical Flow Fallback: {telemetry.frames_optical_flow_fallback}"
    )
    print(f"Total raw ball detections: {telemetry.total_raw_ball_detections}")
    print(
        f"Total frames where the ball was interpolated: {telemetry.total_interpolated_ball_frames}"
    )

    print("\n" + "=" * 72)
    print("TACTICAL COACHING (LLM OUTPUT)")
    print("=" * 72)
    for idx, card in enumerate(final_cards, start=1):
        team = card.get("team", "?")
        flaw = card.get("flaw", "?")
        instruction = card.get("tactical_instruction")
        err = card.get("llm_error")
        print(f"\n--- Item {idx} | {team} | {flaw} ---\n")
        if instruction:
            print(instruction)
        elif err:
            print(f"[LLM skipped/error] {err}")
        else:
            print("(no instruction text)")
    print()


if __name__ == "__main__":
    main()

