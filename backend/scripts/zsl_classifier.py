from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

# Setup logging
LOGGER = logging.getLogger(__name__)

# Paths
BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from models import ChunkTacticalInsight

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    LOGGER.warning("CLIP (openai-clip) not found. ZSL branch will be inactive.")

DEFAULT_ZSL_TACTICS_PATH = BACKEND_ROOT / "data" / "zsl_tactics.json"


class ZSLTacticalClassifier:
    """
    Zero-Shot Learning Tactical Classifier using CLIP.
    Renders 2D player coordinates as Gaussian heatmaps and compares them to 
    textual tactical descriptions.
    """

    def __init__(self, tactics_path: Path | None = None):
        self.tactics_path = tactics_path or DEFAULT_ZSL_TACTICS_PATH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.tactical_descriptions = []
        self.text_features = None

        if CLIP_AVAILABLE:
            try:
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                self._load_tactics()
            except Exception as e:
                LOGGER.error(f"Failed to load CLIP model: {e}")
                globals()["CLIP_AVAILABLE"] = False

    def _load_tactics(self):
        if not self.tactics_path.exists():
            LOGGER.warning(f"ZSL tactics config not found at {self.tactics_path}")
            return

        with open(self.tactics_path, "r") as f:
            data = json.load(f)
            self.tactical_patterns = data.get("tactical_patterns", [])

        if self.tactical_patterns and self.model:
            descriptions = [p["description"] for p in self.tactical_patterns]
            text_tokens = clip.tokenize(descriptions).to(self.device)
            with torch.no_grad():
                self.text_features = self.model.encode_text(text_tokens)
                self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    # Pitch line overlay — black background, dim grey lines (drawn once per canvas size).
    # Lines are kept dim so player density blobs stay visually dominant for CLIP.
    _PITCH_OVERLAY_CACHE: dict[tuple[int, int], np.ndarray] = {}

    @classmethod
    def _build_pitch_overlay(cls, width: int, height: int) -> np.ndarray:
        """
        Draw pitch markings on a BLACK background using dim grey lines (value=77).
        Cached per (width, height) — only built once per canvas size.

        Design rationale: black bg + 30%-grey lines ensures the viridis player
        blobs are the dominant visual signal CLIP embeds, not the pitch colour.
        """
        key = (width, height)
        if key in cls._PITCH_OVERLAY_CACHE:
            return cls._PITCH_OVERLAY_CACHE[key]

        # Black field — player blobs will be the brightest elements
        overlay = np.zeros((height, width, 3), dtype=np.uint8)

        # 30%-grey pitch lines (77/255 ≈ 0.30) — visible but not dominant
        line_colour = (77, 77, 77)
        line_w = 2

        px_per_m_x = width  / 105.0
        px_per_m_y = height / 68.0

        def m2px(x_m: float, y_m: float) -> tuple[int, int]:
            return int(round(x_m * px_per_m_x)), int(round(y_m * px_per_m_y))

        from PIL import ImageDraw
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
        cls._PITCH_OVERLAY_CACHE[key] = result
        return result

    def render_gaussian_heatmap(
        self,
        points: list[tuple[float, float]],
        width: int = 1050,
        height: int = 680,
        sigma: float = 18.0,
    ) -> np.ndarray:
        """
        Render a set of 2D points as a Gaussian positional heatmap.

        Returns a 3-channel uint8 RGB array with:
          • viridis-coloured player density overlaid on a pitch diagram
          • σ = 18 px (sharper than the legacy 40 px, better CLIP discrimination)
        """
        heatmap = np.zeros((height, width), dtype=np.float32)

        # Build Gaussian kernel
        k_size = int(sigma * 3)
        y, x = np.ogrid[-k_size:k_size + 1, -k_size:k_size + 1]
        kernel = np.exp(-(x * x + y * y) / (2 * sigma ** 2))
        kernel /= kernel.max()

        for px, py in points:
            ix, iy = int(round(px)), int(round(py))
            x1, x2 = max(0, ix - k_size), min(width,  ix + k_size + 1)
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
        heatmap_rgb = lut[idx]                  # (H, W, 3) viridis colours

        # Composite: viridis blobs over black-bg pitch lines.
        # alpha=1 where players exist (bright viridis), alpha=0 on empty pitch
        # (dim grey lines show through from the overlay).
        pitch = self._build_pitch_overlay(width, height)
        alpha = heatmap[..., np.newaxis].astype(np.float32)          # linear — no gamma
        composite = (alpha * heatmap_rgb + (1 - alpha) * pitch).astype(np.uint8)

        return composite

    def classify_frame_batch(self, team_points: list[list[tuple[float, float]]]) -> list[dict[str, float]]:
        """
        Classify a batch of frames (each frame is a list of player points for one team).
        Returns a list of cosine-similarity scores per tactical pattern.

        Rendering pipeline (v2):
          1. Gaussian heatmap (σ=18 px) → viridis colour + pitch overlay
          2. CLIP ViT-B/32 image encoder
          3. Cosine similarity vs. pre-encoded text features
        """
        if not CLIP_AVAILABLE or self.model is None or not self.tactical_patterns:
            return []

        # Find non-empty points and render them
        valid_indices = []
        preprocessed_tensors = []
        
        for idx, points in enumerate(team_points):
            if not points:
                continue
            valid_indices.append(idx)
            heatmap_rgb = self.render_gaussian_heatmap(points)
            image = Image.fromarray(heatmap_rgb)
            preprocessed_tensors.append(self.preprocess(image))

        results = [{} for _ in team_points]
        if not preprocessed_tensors:
            return results

        # Stack into a single batch tensor and encode
        batch_input = torch.stack(preprocessed_tensors).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(batch_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # Cosine similarity
            similarities = image_features @ self.text_features.T  # Shape: (batch_size, num_patterns)
            probs = similarities.cpu().numpy()

        for i, idx in enumerate(valid_indices):
            results[idx] = {
                self.tactical_patterns[j]["name"]: float(probs[i, j])
                for j in range(len(self.tactical_patterns))
            }

        return results

    def analyze_chunk(self, frames: list[Any], team_id: Literal["team_0", "team_1"]) -> list[ChunkTacticalInsight]:
        """
        Analyze an entire video chunk and return aggregated tactical insights.
        """
        if not CLIP_AVAILABLE or self.model is None:
            return []

        team_name = team_id
        all_points = []
        
        for frame in frames:
            points = []
            for player in frame.players:
                if player.team == team_name and player.radar_pt is not None:
                    points.append((player.radar_pt[0], player.radar_pt[1]))
            all_points.append(points)

        if not all_points:
            return []

        # Process in batches to avoid OOM or slow processing
        batch_size = 32
        all_probs = []
        for i in range(0, len(all_points), batch_size):
            batch = all_points[i:i+batch_size]
            batch_results = self.classify_frame_batch(batch)
            all_probs.extend(batch_results)

        if not all_probs:
            return []

        # Aggregate results
        tactical_stats = {p["name"]: [] for p in self.tactical_patterns}
        for res in all_probs:
            for name, score in res.items():
                tactical_stats[name].append(score)

        # Recalibrated thresholds (v2): heatmap cosine sims peak at ~0.22-0.26;
        # lower τ from 0.25 → 0.22 and frequency gate from 0.23 → 0.20
        TAU = 0.22
        FREQ_GATE = 0.20

        insights = []
        for name, scores in tactical_stats.items():
            avg_score = np.mean(scores)
            # Threshold for "detecting" a pattern in ZSL
            if avg_score > TAU:
                # Find matching pattern metadata
                pattern = next(p for p in self.tactical_patterns if p["name"] == name)

                # Get Top-3 matches for transparency
                sorted_results = sorted(res.items(), key=lambda x: x[1], reverse=True)[:3]
                top_3_str = ", ".join([f"{n} ({s:.2f})" for n, s in sorted_results])

                frequency_pct = (np.array(scores) > FREQ_GATE).mean() * 100.0

                if frequency_pct > 15.0:  # Only report if persistent
                    severity = "Medium"
                    if avg_score > 0.27: severity = "High"
                    if avg_score > 0.32: severity = "Critical"

                    insights.append(ChunkTacticalInsight(
                        team_id=team_id,
                        flaw=f"ZSL: {name}",
                        severity=severity,
                        frequency_pct=float(frequency_pct),
                        evidence=(
                            f"Zero-Shot recognition identifies a {name} archetype "
                            f"(Confidence: {avg_score*100:.1f}%). "
                            f"Top matches: {top_3_str}. "
                            f"Mapping to {pattern.get('tags', ['unassigned'])[0]} philosophy."
                        ),
                        ball_data_quality="sufficient"
                    ))

        return insights
