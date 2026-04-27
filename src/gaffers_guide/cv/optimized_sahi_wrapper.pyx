# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv
from ultralytics import YOLO

from gaffers_guide.cv.ball_candidate_fuser import FusedCandidate, rank_candidates
from gaffers_guide.cv.pitch_roi_provider import PitchROIProvider
from gaffers_guide.cv.slice_batch_inferencer import SliceBatchInferencer, SliceBox
from gaffers_guide.cv.temporal_ball_prior import TemporalBallPrior


@dataclass(frozen=True)
class ContextAwareSAHIConfig:
    enabled: bool = False
    conf: float = 0.25
    high_conf_skip_threshold: float = 0.25
    slice_w: int = 256
    slice_h: int = 256
    overlap_ratio: float = 0.15
    max_slices_per_frame: int = 4
    temporal_radius_px: int = 112
    temporal_max_radius_px: int = 360
    temporal_expand_step_px: int = 24


@dataclass(frozen=True)
class DetectionContext:
    frame_idx: int
    frame_bgr: np.ndarray


@dataclass(frozen=True)
class BallCandidate:
    xyxy: np.ndarray
    confidence: float
    source: str


@dataclass(frozen=True)
class SAHIFrameTelemetry:
    used_sahi: bool
    used_fallback: bool
    roi_area_ratio: float
    temporal_radius_px: int
    slices_generated: int
    candidate_count: int


@dataclass(frozen=True)
class BallDetectionResult:
    best_ball_bbox: np.ndarray | None
    best_ball_score: float
    telemetry: SAHIFrameTelemetry


def _bbox_center(xyxy: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy.tolist()
    return ((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5)


def _intersect(
    a: tuple[int, int, int, int] | None,
    b: tuple[int, int, int, int] | None,
    *,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int] | None:
    if a is None and b is None:
        return (0, 0, frame_w, frame_h)
    if a is None:
        x1, y1, x2, y2 = b  # type: ignore[misc]
    elif b is None:
        x1, y1, x2, y2 = a
    else:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame_w, int(x2))
    y2 = min(frame_h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


class OptimizedSAHIWrapper:
    """Context-aware batched slice inference for ball recall."""

    def __init__(
        self,
        model: YOLO,
        *,
        ball_class_ids: list[int],
        config: ContextAwareSAHIConfig,
        device: str | None,
        use_half: bool,
    ) -> None:
        self._model = model
        self._ball_ids = set(int(x) for x in ball_class_ids)
        self._cfg = config
        self._roi = PitchROIProvider()
        self._temporal = TemporalBallPrior(
            base_radius_px=config.temporal_radius_px,
            max_radius_px=config.temporal_max_radius_px,
            miss_expand_step_px=config.temporal_expand_step_px,
        )
        self._inferencer = SliceBatchInferencer(
            model,
            ball_class_ids=self._ball_ids,
            conf=config.conf,
            device=device,
            use_half=use_half,
        )

    def _base_ball_candidates(self, detections: sv.Detections) -> list[BallCandidate]:
        det_conf = getattr(detections, "confidence", None)
        out: list[BallCandidate] = []
        for i in range(len(detections)):
            cid = int(detections.class_id[i])
            if cid not in self._ball_ids:
                continue
            score = (
                float(det_conf[i]) if det_conf is not None and i < len(det_conf) else 0.0
            )
            out.append(
                BallCandidate(
                    xyxy=np.asarray(detections.xyxy[i], dtype=np.float32),
                    confidence=score,
                    source="full_frame",
                )
            )
        return out

    def _generate_slices(
        self, region: tuple[int, int, int, int]
    ) -> list[SliceBox]:
        cdef int x1
        cdef int y1
        cdef int x2
        cdef int y2
        cdef int region_w
        cdef int region_h
        cdef int slice_w
        cdef int slice_h
        cdef int step_x
        cdef int step_y
        cdef int xx
        cdef int yy
        cdef int sx1
        cdef int sy1
        cdef int sx2
        cdef int sy2

        x1, y1, x2, y2 = region
        region_w = max(1, x2 - x1)
        region_h = max(1, y2 - y1)
        slice_w = min(self._cfg.slice_w, region_w)
        slice_h = min(self._cfg.slice_h, region_h)
        step_x = max(16, int(slice_w * (1.0 - self._cfg.overlap_ratio)))
        step_y = max(16, int(slice_h * (1.0 - self._cfg.overlap_ratio)))

        slices: list[SliceBox] = []
        yy = y1
        while yy < y2:
            xx = x1
            while xx < x2:
                sx1 = xx
                sy1 = yy
                sx2 = min(x2, sx1 + slice_w)
                sy2 = min(y2, sy1 + slice_h)
                sx1 = max(x1, sx2 - slice_w)
                sy1 = max(y1, sy2 - slice_h)
                slices.append(SliceBox(sx1, sy1, sx2, sy2))
                if len(slices) >= self._cfg.max_slices_per_frame:
                    return slices
                xx += step_x
            yy += step_y
        return slices

    def detect_ball(
        self,
        ctx: DetectionContext,
        detections: sv.Detections,
    ) -> BallDetectionResult:
        base_candidates = self._base_ball_candidates(detections)
        best_base = (
            max(base_candidates, key=lambda c: c.confidence) if base_candidates else None
        )

        frame_h, frame_w = ctx.frame_bgr.shape[:2]
        roi = self._roi.estimate(ctx.frame_bgr)
        temporal_region = self._temporal.search_region(frame_w, frame_h)
        temporal_xyxy = temporal_region.xyxy if temporal_region else None
        search_region = _intersect(
            roi.bbox_xyxy,
            temporal_xyxy,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        temporal_radius = temporal_region.radius_px if temporal_region else 0

        used_sahi = False
        used_fallback = False
        all_candidates: list[FusedCandidate] = [
            FusedCandidate(
                xyxy=c.xyxy,
                confidence=c.confidence,
                score=c.confidence,
                source=c.source,
            )
            for c in base_candidates
        ]

        should_run_sahi = self._cfg.enabled and (
            best_base is None
            or (
                best_base.confidence < self._cfg.high_conf_skip_threshold
                and self._temporal.last_ball_xy is not None
            )
        )
        slices: list[SliceBox] = []
        if should_run_sahi and search_region is not None:
            slices = self._generate_slices(search_region)
            if slices:
                used_sahi = True
                slice_candidates = self._inferencer.infer(ctx.frame_bgr, slices)
                for c in slice_candidates:
                    all_candidates.append(
                        FusedCandidate(
                            xyxy=c.xyxy,
                            confidence=c.confidence,
                            score=c.confidence,
                            source=c.source,
                        )
                    )

        temporal_anchor = self._temporal.last_ball_xy
        ranked = rank_candidates(
            all_candidates,
            temporal_anchor_xy=temporal_anchor,
            search_radius_px=temporal_radius,
        )

        if ranked is None and best_base is not None:
            used_fallback = True
            ranked = FusedCandidate(
                xyxy=best_base.xyxy,
                confidence=best_base.confidence,
                score=best_base.confidence,
                source=best_base.source,
            )

        if ranked is not None:
            center_xy = _bbox_center(ranked.xyxy)
            self._temporal.on_detection(center_xy, ranked.confidence)
            return BallDetectionResult(
                best_ball_bbox=ranked.xyxy,
                best_ball_score=ranked.confidence,
                telemetry=SAHIFrameTelemetry(
                    used_sahi=used_sahi,
                    used_fallback=used_fallback,
                    roi_area_ratio=roi.area_ratio,
                    temporal_radius_px=temporal_radius,
                    slices_generated=len(slices),
                    candidate_count=len(all_candidates),
                ),
            )

        self._temporal.on_miss()
        return BallDetectionResult(
            best_ball_bbox=None,
            best_ball_score=-1.0,
            telemetry=SAHIFrameTelemetry(
                used_sahi=used_sahi,
                used_fallback=used_fallback,
                roi_area_ratio=roi.area_ratio,
                temporal_radius_px=temporal_radius,
                slices_generated=len(slices),
                candidate_count=len(all_candidates),
            ),
        )

