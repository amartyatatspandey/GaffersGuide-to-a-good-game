"""
Two-stage pitch calibration: algebraic line homography + geometric LM refinement.

Stage 1: SVD line solver (reuse sn-calibration baseline).
Optional: RANSAC on pitch-corner intersections vs SVD seed disagreement.
Stage 2: Levenberg–Marquardt on orthogonal line distances with a 1-DOF division-model κ.

Output homography is pitch -> image in fixed 1280×720 space (TacticalRadar calibration lock).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

from scripts.dynamic_homography import DynamicPitchCalibrator, PitchObservationBundle

logger = logging.getLogger(__name__)

# sn-calibration on path via DynamicPitchCalibrator import chain — add for join_points
_REF_DIR = Path(__file__).resolve().parent.parent / "references" / "sn-calibration"
import sys

if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))

from src.detect_extremities import join_points  # noqa: E402
from src.soccerpitch import SoccerPitch  # noqa: E402

SEG_W = DynamicPitchCalibrator.SEG_WIDTH
SEG_H = DynamicPitchCalibrator.SEG_HEIGHT

CALIB_W = 1280
CALIB_H = 720

# Mean pixel disagreement (640×360) between SVD and RANSAC corner projections to prefer RANSAC seed
H_SEED_DISAGREE_THRESHOLD_PX = 15.0

RANSAC_REPROJ_THRESHOLD = 6.0

# Corner landmark: (line_class_a, line_class_b, point_dict_key)
_CORNER_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("Side line top", "Side line left", "TL_PITCH_CORNER"),
    ("Side line top", "Side line right", "TR_PITCH_CORNER"),
    ("Side line bottom", "Side line left", "BL_PITCH_CORNER"),
    ("Side line bottom", "Side line right", "BR_PITCH_CORNER"),
)


def _polyline_to_pixel_segments(
    poly_rowcol: list[np.ndarray],
    width: int,
    height: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert sn-calibration polyline (row, col) to pixel segments (u, v)."""
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    if len(poly_rowcol) < 2:
        return segs
    for i in range(len(poly_rowcol) - 1):
        r0, c0 = float(poly_rowcol[i][0]), float(poly_rowcol[i][1])
        r1, c1 = float(poly_rowcol[i + 1][0]), float(poly_rowcol[i + 1][1])
        p0 = np.array([c0, r0], dtype=np.float64)
        p1 = np.array([c1, r1], dtype=np.float64)
        segs.append((p0, p1))
    return segs


def _homogeneous_line_from_two_pixels(
    u0: float, v0: float, u1: float, v1: float
) -> np.ndarray:
    p0 = np.array([u0, v0, 1.0], dtype=np.float64)
    p1 = np.array([u1, v1, 1.0], dtype=np.float64)
    return np.cross(p0, p1)


def _intersect_lines_homogeneous(
    l1: np.ndarray, l2: np.ndarray
) -> Optional[np.ndarray]:
    pt = np.cross(l1, l2)
    if abs(pt[2]) < 1e-9:
        return None
    return np.array([pt[0] / pt[2], pt[1] / pt[2]], dtype=np.float64)


def _segment_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    eps: float = 1e-9,
) -> Optional[np.ndarray]:
    """
    Intersection of finite segments p0–p1 and q0–q1 in R^2.
    Returns the point if it lies on both segments (within tolerance); else None.
    """
    r = p1 - p0
    s = q1 - q0
    rxs = r[0] * s[1] - r[1] * s[0]
    qmp = q0 - p0
    if abs(rxs) < eps:
        return None
    t = (qmp[0] * s[1] - qmp[1] * s[0]) / rxs
    u = (qmp[0] * r[1] - qmp[1] * r[0]) / rxs
    if t < -1e-4 or t > 1.0 + 1e-4 or u < -1e-4 or u > 1.0 + 1e-4:
        return None
    return p0 + t * r


def _longest_polylines_from_skeletons(
    skeletons: dict[str, Any],
    maxdist: int,
) -> dict[str, list[np.ndarray]]:
    """Longest polyline per class (same heuristic as get_line_extremities)."""
    out: dict[str, list[np.ndarray]] = {}
    for class_name, disks_list in skeletons.items():
        polyline_list = join_points(disks_list, maxdist)
        max_len = 0
        longest: list[np.ndarray] = []
        for polyline in polyline_list:
            if len(polyline) > max_len:
                max_len = len(polyline)
                longest = list(polyline)
        if longest:
            out[class_name] = longest
    return out


def _collect_corner_image_points(
    polylines: dict[str, list[np.ndarray]],
    width: int,
    height: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Image points (N, 2) in pixel coords and list of point_dict keys for world correspondence.
    """
    img_pts: list[np.ndarray] = []
    keys: list[str] = []
    for la, lb, pkey in _CORNER_PAIRS:
        if la not in polylines or lb not in polylines:
            continue
        pa = polylines[la]
        pb = polylines[lb]
        best: Optional[np.ndarray] = None
        segs_a = _polyline_to_pixel_segments(pa, width, height)
        segs_b = _polyline_to_pixel_segments(pb, width, height)
        for a0, a1 in segs_a:
            for b0, b1 in segs_b:
                inter = _segment_intersection(a0, a1, b0, b1)
                if inter is None:
                    continue
                if -5 <= inter[0] < width + 5 and -5 <= inter[1] < height + 5:
                    best = inter
                    break
            if best is not None:
                break
        if best is None and len(pa) >= 2 and len(pb) >= 2:
            la_h = _homogeneous_line_from_two_pixels(
                float(pa[0][1]), float(pa[0][0]), float(pa[-1][1]), float(pa[-1][0])
            )
            lb_h = _homogeneous_line_from_two_pixels(
                float(pb[0][1]), float(pb[0][0]), float(pb[-1][1]), float(pb[-1][0])
            )
            inter = _intersect_lines_homogeneous(la_h, lb_h)
            if inter is not None and (
                -20 <= inter[0] < width + 20 and -20 <= inter[1] < height + 20
            ):
                best = inter
        if best is not None:
            img_pts.append(best)
            keys.append(pkey)
    if len(img_pts) < 4:
        return np.zeros((0, 2), dtype=np.float64), []
    return np.stack(img_pts, axis=0), keys


def _world_points_for_keys(field: SoccerPitch, keys: list[str]) -> np.ndarray:
    pts = []
    for k in keys:
        p = field.point_dict[k]
        pts.append([float(p[0]), float(p[1])])
    return np.asarray(pts, dtype=np.float64)


def _mean_corner_disagreement_px(
    H_a: np.ndarray, H_b: np.ndarray, field: SoccerPitch
) -> float:
    """Mean L2 distance in seg pixels between projections of four pitch corners."""
    if H_a is None or H_b is None or H_a.size == 0 or H_b.size == 0:
        return 1e6
    corners_world = _world_points_for_keys(
        field,
        ["TL_PITCH_CORNER", "TR_PITCH_CORNER", "BL_PITCH_CORNER", "BR_PITCH_CORNER"],
    )
    if corners_world.shape[0] != 4:
        return 0.0
    ones = np.ones((4, 1), dtype=np.float64)
    X = np.hstack([corners_world, ones])
    errs: list[float] = []
    for i in range(4):
        wa = H_a @ X[i]
        za = wa[2] if abs(wa[2]) > 1e-12 else 1e-12
        pa = np.array([wa[0] / za, wa[1] / za], dtype=np.float64)
        wb = H_b @ X[i]
        zb = wb[2] if abs(wb[2]) > 1e-12 else 1e-12
        pb = np.array([wb[0] / zb, wb[1] / zb], dtype=np.float64)
        errs.append(float(np.linalg.norm(pa - pb)))
    return float(np.mean(errs)) if errs else 0.0


def _ransac_homography_from_corners(
    field: SoccerPitch,
    img_pts: np.ndarray,
    world_keys: list[str],
) -> Optional[np.ndarray]:
    if img_pts.shape[0] < 4 or len(world_keys) != img_pts.shape[0]:
        return None
    world = _world_points_for_keys(field, world_keys).astype(np.float32)
    image = img_pts.astype(np.float32)
    H, mask = cv2.findHomography(
        world,
        image,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
        maxIters=3000,
        confidence=0.995,
    )
    if H is None or mask is None:
        return None
    inliers = int(mask.ravel().sum())
    if inliers < 3:
        logger.info("RANSAC homography rejected: inliers=%d", inliers)
        return None
    H64 = H.astype(np.float64)
    H64 /= H64[2, 2] if abs(H64[2, 2]) > 1e-12 else 1.0
    logger.info("RANSAC homography: inliers=%d / %d", inliers, len(mask))
    return H64


def _params_to_H(p: np.ndarray) -> np.ndarray:
    """p is 8-vector h11..h32; h33 fixed to 1."""
    return np.array(
        [
            [p[0], p[1], p[2]],
            [p[3], p[4], p[5]],
            [p[6], p[7], 1.0],
        ],
        dtype=np.float64,
    )


def _undistort_division_normalized(
    u: float,
    v: float,
    width: float,
    height: float,
    kappa: float,
) -> tuple[float, float]:
    """
    Apply user-specified division-style undistortion in normalized centered coordinates.

    Let ``tilde = (x - c) / f`` with ``f = max(width, height)``; then
    ``tilde_u = tilde_d / (1 + kappa * ||tilde_d||^2)``, map back to pixels.
    """
    f = float(max(width, height))
    cx = width / 2.0
    cy = height / 2.0
    tdx = (u - cx) / f
    tdy = (v - cy) / f
    r2 = tdx * tdx + tdy * tdy
    denom = 1.0 + kappa * r2
    if abs(denom) < 1e-9:
        denom = 1e-9
    su = tdx / denom
    sv = tdy / denom
    return su * f + cx, sv * f + cy


def _line_image_from_pitch(H: np.ndarray, L_pitch: np.ndarray) -> Optional[np.ndarray]:
    """Image line l such that l^T x = 0 for image point x (homogeneous), l ~ inv(H).T @ L_pitch."""
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    l = Hinv.T @ L_pitch
    n = np.linalg.norm(l[:2])
    if n < 1e-12:
        return None
    return l / n


def _point_line_distance(u: float, v: float, l: np.ndarray) -> float:
    """Orthogonal distance from (u,v) to line l=(a,b,c) with a^2+b^2=1."""
    return float(abs(l[0] * u + l[1] * v + l[2]))


def _build_residuals_factory(
    extremities: dict[str, Any],
    field: SoccerPitch,
    seg_w: int,
    seg_h: int,
):
    """Each observation is (line_class, u_pixel, v_pixel) for one extremity."""
    observations: list[tuple[str, float, float]] = []
    for k, v in extremities.items():
        if (
            k == "Circle central"
            or "unknown" in k
            or k not in field.line_extremities_keys
        ):
            continue
        if field.get_2d_homogeneous_line(k) is None:
            continue
        u0 = float(v[0]["x"] * seg_w)
        v0 = float(v[0]["y"] * seg_h)
        u1 = float(v[1]["x"] * seg_w)
        v1 = float(v[1]["y"] * seg_h)
        observations.append((k, u0, v0))
        observations.append((k, u1, v1))

    def residuals(vec: np.ndarray) -> np.ndarray:
        H = _params_to_H(vec[:8])
        kappa = float(vec[8])
        res: list[float] = []
        for k, u, vpix in observations:
            Lp = field.get_2d_homogeneous_line(k)
            if Lp is None:
                res.append(1e3)
                continue
            l_img = _line_image_from_pitch(H, Lp)
            if l_img is None:
                res.append(1e3)
                continue
            uu, vv = _undistort_division_normalized(
                u, vpix, float(seg_w), float(seg_h), kappa
            )
            res.append(_point_line_distance(uu, vv, l_img))
        return np.asarray(res, dtype=np.float64)

    return residuals


def _refine_homography_lm(
    H_seed: np.ndarray,
    extremities: dict[str, Any],
    field: SoccerPitch,
    seg_w: int,
    seg_h: int,
) -> tuple[np.ndarray, float]:
    """Return refined H (3x3, h33=1) and final cost."""
    p0 = np.array(
        [
            H_seed[0, 0],
            H_seed[0, 1],
            H_seed[0, 2],
            H_seed[1, 0],
            H_seed[1, 1],
            H_seed[1, 2],
            H_seed[2, 0],
            H_seed[2, 1],
            0.0,
        ],
        dtype=np.float64,
    )
    p0[:8] /= H_seed[2, 2] if abs(H_seed[2, 2]) > 1e-12 else 1.0

    resid_fn = _build_residuals_factory(extremities, field, seg_w, seg_h)
    r0 = resid_fn(p0)
    if r0.size <= 9:
        logger.info(
            "LM skipped: need m > n for LM (got %d residuals, 9 params)", r0.size
        )
        return _params_to_H(p0[:8]), float(np.sum(r0**2))

    try:
        result = least_squares(
            resid_fn,
            p0,
            method="lm",
            max_nfev=200,
            verbose=0,
        )
    except Exception as e:
        logger.warning("LM least_squares failed: %s", e)
        Hf = _params_to_H(p0[:8])
        return Hf, float(np.sum(r0**2))

    logger.info(
        "LM refinement: success=%s message=%s cost=%.6f nfev=%d",
        result.success,
        result.message,
        float(result.cost),
        int(result.nfev),
    )
    pf = result.x
    Hf = _params_to_H(pf[:8])
    Hf /= Hf[2, 2] if abs(Hf[2, 2]) > 1e-12 else 1.0
    return Hf, float(result.cost)


def _condition_ok(H: np.ndarray) -> bool:
    try:
        c = np.linalg.cond(H)
        return bool(c < 1e12 and np.isfinite(c))
    except Exception:
        return False


class AdvancedPitchCalibrator:
    """
    V2 calibrator: coarse SVD (+ optional RANSAC seed), LM geometric refinement, output H in 1280×720.
    """

    def __init__(self, weights_path: str | Path) -> None:
        self._base = DynamicPitchCalibrator(weights_path)
        self._field = self._base.field

    def get_homography(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Pitch -> image in **1280×720** pixels (TacticalRadar calibration space).

        :param frame: BGR (H, W, 3).
        :return: 3×3 float64 or None.
        """
        obs = self._base.collect_pitch_observations(frame)
        if obs is None:
            return None

        H_svd = obs.H_coarse_seg.copy()
        H_seed = H_svd

        polylines = _longest_polylines_from_skeletons(obs.skeletons, maxdist=40)
        img_pts, world_keys = _collect_corner_image_points(polylines, SEG_W, SEG_H)
        H_ransac: Optional[np.ndarray] = None
        if img_pts.shape[0] >= 4 and len(world_keys) == img_pts.shape[0]:
            H_ransac = _ransac_homography_from_corners(self._field, img_pts, world_keys)

        if H_ransac is not None:
            disagree = _mean_corner_disagreement_px(H_svd, H_ransac, self._field)
            logger.info("SVD vs RANSAC mean corner disagreement: %.2f px", disagree)
            if disagree > H_SEED_DISAGREE_THRESHOLD_PX:
                H_seed = H_ransac
                logger.info(
                    "Using RANSAC homography as LM seed (disagreement > %.1f px)",
                    H_SEED_DISAGREE_THRESHOLD_PX,
                )
            else:
                logger.info("Using SVD homography as LM seed")

        if not _condition_ok(H_seed):
            logger.warning("H_seed ill-conditioned; skipping LM")
            H_ref = H_seed
        else:
            H_ref, _ = _refine_homography_lm(
                H_seed, obs.extremities, self._field, SEG_W, SEG_H
            )
            if not _condition_ok(H_ref):
                logger.warning("LM result ill-conditioned; reverting to seed")
                H_ref = H_seed

        S_calib = np.array(
            [[CALIB_W / SEG_W, 0, 0], [0, CALIB_H / SEG_H, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        H_out = S_calib @ H_ref
        H_out /= H_out[2, 2] if abs(H_out[2, 2]) > 1e-12 else 1.0
        return H_out.astype(np.float64)

    def collect_pitch_observations(
        self, frame: np.ndarray
    ) -> Optional[PitchObservationBundle]:
        """Delegate to underlying V1 calibrator (for tests / introspection)."""
        return self._base.collect_pitch_observations(frame)
