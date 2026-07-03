#!/usr/bin/env python3
"""
generate_real_eval.py — Build a real-match-footage evaluation set for formation classification.
================================================================================

This script processes the raw match videos in backend/data/uploads/, extracts player coordinates 
from pre-calculated tracking data, identifies high-quality non-overlapping candidate frames for each 
video, renders them as Gaussian heatmaps using a self-contained renderer (bypassing torch/clip dependencies),
and extracts low-res preview frames using ffmpeg for manual labeling.

Usage:
------
    python3 scripts/generate_real_eval.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# ── Path Setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

# Constants
MIN_SPACING_FRAMES = 150  # Spacing between selected frames in a video (~6 seconds)
MAX_CLIPS_PER_VIDEO = 2   # Target 1-2 clips per video to yield ~30-50 total
CANVAS_W_PX = 1050
CANVAS_H_PX = 680

# ── Self-Contained Heatmap Renderer ──────────────────────────────────────────
# Replicates the exact behavior of ZSLTacticalClassifier without torch/clip imports.
_PITCH_OVERLAY_CACHE: dict[tuple[int, int], np.ndarray] = {}

def _build_pitch_overlay(width: int, height: int) -> np.ndarray:
    """Draw pitch markings on a BLACK background using dim grey lines (value=77)."""
    key = (width, height)
    if key in _PITCH_OVERLAY_CACHE:
        return _PITCH_OVERLAY_CACHE[key]

    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    line_colour = (77, 77, 77)
    line_w = 2

    px_per_m_x = width / 105.0
    px_per_m_y = height / 68.0

    def m2px(x_m: float, y_m: float) -> tuple[int, int]:
        return int(round(x_m * px_per_m_x)), int(round(y_m * px_per_m_y))

    pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil)

    W, H = width, height

    # Touchlines & goal lines
    draw.rectangle([0, 0, W - 1, H - 1], outline=line_colour, width=line_w)
    # Halfway line
    draw.line([(W // 2, 0), (W // 2, H)], fill=line_colour, width=line_w)
    # Centre circle — radius 9.15 m
    r_px_x = int(9.15 * px_per_m_x)
    r_px_y = int(9.15 * px_per_m_y)
    cx, cy = W // 2, H // 2
    draw.ellipse(
        [cx - r_px_x, cy - r_px_y, cx + r_px_x, cy + r_px_y],
        outline=line_colour, width=line_w,
    )
    # Penalty boxes — 40.32 m wide × 16.5 m deep
    draw.rectangle(
        [*m2px(0, (68 - 40.32) / 2), *m2px(16.5, (68 + 40.32) / 2)],
        outline=line_colour, width=line_w,
    )
    draw.rectangle(
        [*m2px(105 - 16.5, (68 - 40.32) / 2), *m2px(105, (68 + 40.32) / 2)],
        outline=line_colour, width=line_w,
    )

    result = np.array(pil, dtype=np.uint8)
    _PITCH_OVERLAY_CACHE[key] = result
    return result

def render_gaussian_heatmap(
    points: list[tuple[float, float]],
    width: int = 1050,
    height: int = 680,
    sigma: float = 10.0,
) -> np.ndarray:
    """Render a set of 2D points as a Gaussian positional heatmap (viridis + pitch markings)."""
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Build Gaussian kernel
    k_size = int(sigma * 3)
    y, x = np.ogrid[-k_size:k_size + 1, -k_size:k_size + 1]
    kernel = np.exp(-(x * x + y * y) / (2 * sigma ** 2))
    kernel /= kernel.max()

    for px, py in points:
        ix, iy = int(round(px)), int(round(py))
        x1, x2 = max(0, ix - k_size), min(width, ix + k_size + 1)
        y1, y2 = max(0, iy - k_size), min(height, iy + k_size + 1)
        kx1, kx2 = k_size - (ix - x1), k_size + (x2 - ix)
        ky1, ky2 = k_size - (iy - y1), k_size + (y2 - iy)
        if x2 > x1 and y2 > y1:
            heatmap[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

    # Normalise to [0, 1]
    h_max = heatmap.max()
    if h_max > 0:
        heatmap /= h_max

    # Apply viridis colormap: [0,1] float → RGB uint8
    _VIRIDIS = np.array([
        [68,  1,  84], [72,  20, 103], [67,  44, 122], [57,  67, 134],
        [47,  88, 140], [38, 107, 142], [30, 124, 142], [24, 141, 139],
        [22, 158, 132], [36, 173, 118], [65, 187,  98], [103, 198,  73],
        [143, 207,  45], [184, 214,  29], [221, 219,  30], [253, 231,  37],
    ], dtype=np.uint8)
    stops = len(_VIRIDIS)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0 * (stops - 1)
        lo, hi = int(t), min(int(t) + 1, stops - 1)
        frac = t - lo
        lut[i] = (1 - frac) * _VIRIDIS[lo] + frac * _VIRIDIS[hi]

    idx = (heatmap * 255).astype(np.uint8)
    heatmap_rgb = lut[idx]  # (H, W, 3) viridis colours

    pitch = _build_pitch_overlay(width, height)
    alpha = heatmap[..., np.newaxis].astype(np.float32)
    composite = (alpha * heatmap_rgb + (1 - alpha) * pitch).astype(np.uint8)

    return composite

# ── Video Utilities ─────────────────────────────────────────────────────────

def get_video_fps(video_path: Path) -> float:
    """Retrieve video frame rate using ffprobe, fallback to 25.0 if failure."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = res.stdout.strip()
        if "/" in out:
            num, den = out.split("/")
            return float(num) / float(den)
        return float(out)
    except Exception as e:
        LOGGER.warning("ffprobe FPS detection failed for %s, falling back to 25.0: %s", video_path.name, e)
        return 25.0

def extract_preview_frame(video_path: Path, frame_idx: int, fps: float, output_path: Path) -> bool:
    """Extract a frame at frame_idx and save it resized to 640x360 as a low-res preview using ffmpeg."""
    time_sec = frame_idx / fps
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{time_sec:.4f}",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "4",
        "-vf", "scale=640:-1",
        str(output_path)
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except Exception as e:
        LOGGER.error("ffmpeg frame extraction failed at frame %d for %s: %s", frame_idx, video_path.name, e)
        return False

# ── Tracking Data Parsers ───────────────────────────────────────────────────

def parse_tracking_file(video_stem: str) -> list[dict[str, Any]]:
    """Parse tracking coordinate telemetry from existing JSON report or coords.pkl in output directory."""
    json_path = BACKEND_ROOT / "output" / f"{video_stem}_tracking_data.json"
    pkl_path = BACKEND_ROOT / "output" / f"{video_stem}_coords.pkl"
    
    frames_data = []
    
    if json_path.is_file():
        LOGGER.info("Found tracking JSON for %s, parsing...", video_stem)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            telemetry = data.get("telemetry", {})
            coord_space = telemetry.get("coord_space")
            if not coord_space:
                raise KeyError(f"coord_space metadata field is missing in telemetry of {json_path.name}!")
                
            LOGGER.info("Detected coord_space = %s for %s", coord_space, video_stem)
            
            for frame in data.get("frames", []):
                f_idx = frame.get("frame_idx")
                homography_conf = frame.get("homography_confidence", 1.0)
                
                players_0 = []
                players_1 = []
                for p in frame.get("players", []):
                    tid = p.get("team_id")
                    xp = p.get("x_pitch")
                    yp = p.get("y_pitch")
                    if xp is not None and yp is not None:
                        xp_f, yp_f = float(xp), float(yp)
                        if coord_space == "meters_centered":
                            pt = ((xp_f + 52.5) * 10.0, (yp_f + 34.0) * 10.0)
                        elif coord_space == "meters_corner":
                            pt = (xp_f * 10.0, yp_f * 10.0)
                        elif coord_space == "pixels_corner":
                            pt = (xp_f, yp_f)
                        else:
                            raise ValueError(f"Unknown coord_space: {coord_space} in file {json_path.name}")
                            
                        if tid == "team_0":
                            players_0.append(pt)
                        elif tid == "team_1":
                            players_1.append(pt)
                
                frames_data.append({
                    "frame_idx": f_idx,
                    "homography_confidence": homography_conf,
                    "players_0": players_0,
                    "players_1": players_1
                })
        except Exception as e:
            LOGGER.error("Error reading JSON tracking for %s: %s", video_stem, e)
            
    elif pkl_path.is_file():
        LOGGER.info("Found tracking coords PKL for %s, parsing...", video_stem)
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            for frame in data.get("frames", []):
                f_idx = frame.get("frame_idx")
                homography_conf = 1.0
                
                players_0 = []
                players_1 = []
                for p in frame.get("players_tactical", []):
                    tid = p.get("team")
                    pt_list = p.get("radar_pt")
                    if pt_list is not None and len(pt_list) == 2:
                        pt = (float(pt_list[0]), float(pt_list[1]))
                        if tid == "team_0":
                            players_0.append(pt)
                        elif tid == "team_1":
                            players_1.append(pt)
                            
                frames_data.append({
                    "frame_idx": f_idx,
                    "homography_confidence": homography_conf,
                    "players_0": players_0,
                    "players_1": players_1
                })
        except Exception as e:
            LOGGER.error("Error reading PKL tracking for %s: %s", video_stem, e)
            
    return frames_data

def select_top_candidates(frames_data: list[dict[str, Any]], max_clips: int = 2, min_spacing: int = 150) -> list[dict[str, Any]]:
    """Grade frame candidates by quality (confidence & number of players) and select top non-overlapping ones."""
    graded = []
    for f in frames_data:
        f_idx = f["frame_idx"]
        h_conf = f["homography_confidence"]
        
        # We only look at frames with decent calibration confidence
        if h_conf < 0.65:
            continue
            
        for team_key in ("players_0", "players_1"):
            pts = f[team_key]
            n_players = len(pts)
            
            # Look for 9 to 11 outfield players detected (typical camera view of attacking/defending shapes)
            if 9 <= n_players <= 11:
                # Grade:
                # - Proximity to exactly 11 players (11 -> 1.0, 10 -> 0.9, 9 -> 0.8)
                # - Weighted with homography confidence
                player_score = 1.0 - (11 - n_players) * 0.1
                score = h_conf * 0.4 + player_score * 0.6
                
                graded.append({
                    "frame_idx": f_idx,
                    "team": "team_0" if team_key == "players_0" else "team_1",
                    "points": pts,
                    "score": score
                })
                
    # Sort by score descending
    graded.sort(key=lambda x: x["score"], reverse=True)
    
    # Select non-overlapping clips
    selected = []
    for item in graded:
        f_idx = item["frame_idx"]
        
        # Spacing check
        too_close = False
        for s in selected:
            if abs(s["frame_idx"] - f_idx) < min_spacing:
                too_close = True
                break
                
        if not too_close:
            selected.append(item)
            if len(selected) >= max_clips:
                break
                
    return selected

# ── Main Implementation ─────────────────────────────────────────────────────

def main() -> None:
    LOGGER.info("Starting Real-Footage Evaluation Set Generation (Self-Contained)...")
    
    # Setup directory structure
    eval_root = BACKEND_ROOT / "data" / "real_eval_set"
    heatmaps_dir = eval_root / "heatmaps"
    previews_dir = eval_root / "previews"
    
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate all raw videos
    uploads_dir = BACKEND_ROOT / "data" / "uploads"
    if not uploads_dir.is_dir():
        LOGGER.error("Uploads directory missing: %s", uploads_dir)
        sys.exit(1)
        
    videos = sorted(uploads_dir.glob("*.mp4")) + sorted(uploads_dir.glob("*.MP4"))
    
    # Load existing labels from manifest if it exists to preserve manual labels
    manifest_csv_path = eval_root / "manifest.csv"
    existing_labels = {}
    if manifest_csv_path.exists():
        try:
            with open(manifest_csv_path, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if "clip_id" in row and "label" in row:
                        existing_labels[row["clip_id"]] = row["label"]
            LOGGER.info("Loaded %d existing labels from manifest to preserve them.", len(existing_labels))
        except Exception as e:
            LOGGER.warning("Could not read existing manifest to preserve labels: %s", e)

    manifest_rows = []
    clip_count = 0
    contrib_videos = set()
    video_clip_counts = {}
    
    for video_path in videos:
        # Skip dummy video files (< 1KB size)
        if video_path.stat().st_size < 1000:
            LOGGER.info("Skipping dummy video: %s", video_path.name)
            continue
            
        video_stem = video_path.stem
        
        # 1. Parse coordinate stream (load tracking data if already present)
        frames_data = parse_tracking_file(video_stem)
        
        # If tracking data is not found, print manual intervention notice
        if not frames_data:
            LOGGER.warning(
                "No tracking data found for %s in backend/output/. "
                "To process this video, run standard coordinate tracking first:\n"
                "  python3 scripts/run_coords_only.py --video %s",
                video_path.name, video_path
            )
            continue
            
        # 2. Extract candidate clips
        candidates = select_top_candidates(frames_data, max_clips=MAX_CLIPS_PER_VIDEO, min_spacing=MIN_SPACING_FRAMES)
        if not candidates:
            LOGGER.warning("Could not find any clean tactical frame candidates in %s", video_path.name)
            continue
            
        fps = get_video_fps(video_path)
        LOGGER.info("Selected %d candidate frame(s) for %s (FPS: %.2f)", len(candidates), video_path.name, fps)
        
        # 3. Render and save evaluation assets
        for item in candidates:
            clip_count += 1
            f_idx = item["frame_idx"]
            pts = item["points"]
            
            clip_id = f"clip_{clip_count:03d}"
            
            # Render player density heatmap (viridis colormap + pitch lines overlay)
            heatmap_rgb = render_gaussian_heatmap(pts, width=CANVAS_W_PX, height=CANVAS_H_PX)
            heatmap_filename = f"{clip_id}_heatmap.png"
            heatmap_path = heatmaps_dir / heatmap_filename
            Image.fromarray(heatmap_rgb).save(heatmap_path)
            
            # Extract video frame at frame_idx for preview
            preview_filename = f"{clip_id}_preview.jpg"
            preview_path = previews_dir / preview_filename
            extract_success = extract_preview_frame(video_path, f_idx, fps, preview_path)
            
            if extract_success:
                contrib_videos.add(video_path.name)
                video_clip_counts[video_path.name] = video_clip_counts.get(video_path.name, 0) + 1
                
                # Write to manifest structure
                manifest_rows.append({
                    "clip_id": clip_id,
                    "video_source": video_path.name,
                    "frame_range": f"{f_idx}-{f_idx}",
                    "heatmap_path": f"backend/data/real_eval_set/heatmaps/{heatmap_filename}",
                    "label": existing_labels.get(clip_id, "UNLABELED")
                })
            else:
                LOGGER.error("Failed to extract preview frame for %s. Skipping clip %s.", video_path.name, clip_id)
                # Cleanup heatmap on failure
                if heatmap_path.exists():
                    heatmap_path.unlink()
                clip_count -= 1
                
    # 4. Write manifest CSV file
    manifest_csv_path = eval_root / "manifest.csv"
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["clip_id", "video_source", "frame_range", "heatmap_path", "label"])
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    # 5. Print detailed summary
    LOGGER.info("\n" + "═"*60)
    LOGGER.info("   REAL-FOOTAGE EVALUATION DATASET SUMMARY")
    LOGGER.info("═"*60)
    LOGGER.info("Total clips generated     : %d", len(manifest_rows))
    LOGGER.info("Distinct source videos    : %d", len(contrib_videos))
    LOGGER.info("Manifest CSV saved to     : %s", manifest_csv_path)
    LOGGER.info("Heatmaps saved in         : %s", heatmaps_dir)
    LOGGER.info("Previews saved in         : %s", previews_dir)
    LOGGER.info("\nClips Generated Per Source Video:")
    LOGGER.info("─"*60)
    for vid, count in sorted(video_clip_counts.items()):
        LOGGER.info("  • %-45s : %d clip(s)", vid, count)
    LOGGER.info("═"*60 + "\n")

if __name__ == "__main__":
    main()
