#!/usr/bin/env python3
"""
Flatten Gaffer's Guide coordinate pickles into tidy CSV for EDA / Tableau / Power BI.

Reads backend/output/training_coords/*_coords.pkl (one match per file), emits:
  - football_tracking_viz_ready.csv  (one row per player per frame; streamed per match)
  - match_summaries.csv              (distance & top speed per player per match)

Pitch XY from radar projection uses TacticalRadar scale=10 → 10 px ≈ 1 m (1050×680 ≈ 105m×68m).
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.track_teams import CLASS_PLAYER  # noqa: E402

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# TacticalRadar uses scale=10: radar_w=1050, radar_h=680 → ~10 px per meter
METERS_PER_RADAR_UNIT = 1.0 / 10.0
SPEED_OUTLIER_KMH = 40.0
ROLLING_SPEED_WINDOW = 5


def _team_to_id(team: str | None) -> float:
    if team == "team_0":
        return 0.0
    if team == "team_1":
        return 1.0
    return np.nan


def _load_match_tidy(path: Path, match_id: str) -> pd.DataFrame:
    """Build one tidy DataFrame for a single pickle (players only)."""
    with path.open("rb") as f:
        payload: dict[str, Any] = pickle.load(f)

    rows: list[dict[str, Any]] = []
    for fr in payload.get("frames", []):
        frame_idx = int(fr["frame_idx"])
        for tr in fr.get("tracks", []):
            if int(tr.get("class_id", -1)) != CLASS_PLAYER:
                continue
            tid = tr.get("track_id")
            if tid is None:
                continue
            bbox = tr.get("bbox_xyxy") or [np.nan, np.nan, np.nan, np.nan]
            if len(bbox) < 4:
                bbox = [np.nan, np.nan, np.nan, np.nan]
            x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            rp = tr.get("radar_xy")
            if rp is not None and len(rp) >= 2:
                x_pitch = float(rp[0])
                y_pitch = float(rp[1])
            else:
                x_pitch = np.nan
                y_pitch = np.nan
            rows.append(
                {
                    "match_id": match_id,
                    "frame_idx": frame_idx,
                    "player_id": int(tid),
                    "team_id": _team_to_id(tr.get("team")),
                    "x_pitch": x_pitch,
                    "y_pitch": y_pitch,
                    "x_canvas": cx,
                    "y_canvas": cy,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "match_id",
                "frame_idx",
                "player_id",
                "team_id",
                "x_pitch",
                "y_pitch",
                "x_canvas",
                "y_canvas",
            ]
        )

    df = pd.DataFrame(rows)
    df.sort_values(["frame_idx", "player_id"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    return df


def _add_physics_and_clean(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Meters on pitch, speed (km/h), outlier handling, smoothing, cumulative distance."""
    if df.empty:
        df = df.copy()
        df["speed_kmh_raw"] = pd.Series(dtype=np.float64)
        df["speed_kmh"] = pd.Series(dtype=np.float64)
        df["cumulative_distance_m"] = pd.Series(dtype=np.float64)
        return df

    df = df.copy()
    df["_x_m"] = df["x_pitch"].astype(np.float64) * METERS_PER_RADAR_UNIT
    df["_y_m"] = df["y_pitch"].astype(np.float64) * METERS_PER_RADAR_UNIT

    gcols = ["match_id", "player_id"]
    g = df.groupby(gcols, sort=False)

    df["_d_frame"] = g["frame_idx"].diff()
    df["_dt_s"] = df["_d_frame"] / fps
    df["_dx_m"] = g["_x_m"].diff()
    df["_dy_m"] = g["_y_m"].diff()
    df["_disp_m"] = np.sqrt(df["_dx_m"] ** 2 + df["_dy_m"] ** 2)
    valid = df["_d_frame"].notna() & (df["_d_frame"] > 0)
    df["_disp_m"] = df["_disp_m"].where(valid)
    df["_dt_s"] = df["_dt_s"].where(valid)

    df["speed_kmh_raw"] = (df["_disp_m"] / df["_dt_s"]) * 3.6
    df["speed_kmh_raw"] = df["speed_kmh_raw"].replace([np.inf, -np.inf], np.nan)

    outlier = df["speed_kmh_raw"] > SPEED_OUTLIER_KMH

    df["_disp_seg"] = df["_disp_m"].where(~outlier)
    df["_disp_seg"] = g["_disp_seg"].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    ).fillna(0.0)
    df["cumulative_distance_m"] = g["_disp_seg"].cumsum()

    df["speed_kmh"] = df["speed_kmh_raw"].where(~outlier)
    df["speed_kmh"] = g["speed_kmh"].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
    df["speed_kmh"] = g["speed_kmh"].transform(
        lambda s: s.rolling(ROLLING_SPEED_WINDOW, min_periods=1).mean()
    )

    df.drop(
        columns=[
            "_x_m",
            "_y_m",
            "_d_frame",
            "_dt_s",
            "_dx_m",
            "_dy_m",
            "_disp_m",
            "_disp_seg",
        ],
        inplace=True,
    )
    return df


def _summaries_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per match_id, player_id aggregates (small table)."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "match_id",
                "player_id",
                "team_id",
                "total_distance_m",
                "top_speed_kmh",
                "frame_count",
            ]
        )

    agg = (
        df.groupby(["match_id", "player_id"], sort=False)
        .agg(
            team_id=("team_id", "first"),
            total_distance_m=("cumulative_distance_m", "last"),
            top_speed_kmh=("speed_kmh", "max"),
            frame_count=("frame_idx", "count"),
        )
        .reset_index()
    )
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare tidy tracking CSV for visualization.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=BACKEND_ROOT / "output" / "training_coords",
        help="Directory containing *_coords.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BACKEND_ROOT / "output" / "viz_ready",
        help="Directory for CSV outputs",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Assumed video FPS for speed/distance (default 30).",
    )
    parser.add_argument(
        "--master-name",
        type=str,
        default="football_tracking_viz_ready.csv",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="match_summaries.csv",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / args.master_name
    summary_path = output_dir / args.summary_name

    pkl_files = sorted(input_dir.glob("*_coords.pkl"))
    if not pkl_files:
        LOGGER.error("No *_coords.pkl under %s", input_dir)
        return 1

    if master_path.exists():
        master_path.unlink()

    export_cols = [
        "match_id",
        "frame_idx",
        "player_id",
        "team_id",
        "x_pitch",
        "y_pitch",
        "x_canvas",
        "y_canvas",
        "speed_kmh_raw",
        "speed_kmh",
        "cumulative_distance_m",
    ]

    summary_parts: list[pd.DataFrame] = []
    first_write = True

    for pkl_path in tqdm(pkl_files, desc="Matches", unit="match"):
        match_id = pkl_path.stem.replace("_coords", "")
        df_raw = _load_match_tidy(pkl_path, match_id)
        if df_raw.empty:
            LOGGER.warning("No player rows in %s", pkl_path.name)
            continue
        df = _add_physics_and_clean(df_raw, fps=args.fps)
        df[export_cols].to_csv(
            master_path,
            mode="a",
            index=False,
            header=first_write,
        )
        first_write = False
        summary_parts.append(_summaries_from_frame(df))

    if first_write:
        LOGGER.error("No data written (all pickles empty?).")
        return 1

    summaries = pd.concat(summary_parts, ignore_index=True)
    summaries.to_csv(summary_path, index=False)
    LOGGER.info("Wrote %s", master_path)
    LOGGER.info("Wrote %s (%d rows)", summary_path, len(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
