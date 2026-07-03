"""
evaluate_zstr.py — Zero-Shot Tactical Recognition (ZSTR) Classifier Evaluation
================================================================================
Gaffer's Guide · Football Analytics System

Usage
-----
    cd /path/to/backend
    python scripts/evaluate_zstr.py [--n-clips N] [--tau T] [--seed S] [--out FILE]

Arguments
---------
    --n-clips   Number of synthetic clips per formation class (default: 20)
    --tau       Cosine-similarity threshold (default: 0.25, mirrors ZSLTacticalClassifier)
    --seed      Random seed for reproducible jitter (default: 42)
    --out       Output JSON path (default: output/zstr_eval_results.json)

What It Does
------------
1. Instantiates ZSLTacticalClassifier (loads CLIP ViT-B/32 + 5 formation descriptions)
2. Generates deterministic synthetic coordinate sets per formation class
   — each clip = 11 outfield player (x, y) positions on a 105×68 m pitch
   — Gaussian jitter σ=3 m simulates real-world tracking noise
3. Converts metre-space coords → pixel-space (1050×680 px) for heatmap rendering
4. Runs the full CLIP heatmap pipeline on every clip
5. Computes:
   • Top-1 accuracy (argmax, no threshold)
   • Top-1 accuracy with τ gate (clips scoring < τ on ALL classes = "No Match")
   • Per-class Precision / Recall / F1 (sklearn)
   • 5×5 Confusion matrix
   • Most common misclassification per class
6. Prints a clean, paper-ready table to stdout
7. Saves full results to JSON for archival

Formation Geometry Reference (metres, origin = pitch centre)
-------------------------------------------------------------
Pitch: 105 m long (x: −52.5 → +52.5), 68 m wide (y: −34 → +34)
Attacking direction: left → right (increasing x)

• 4-3-3 High Press   : Back-4 @ x≈−32, Mid-3 @ x≈−10, Front-3 @ x≈+28 (wide)
• 4-2-3-1 Mid Block  : Back-4 @ x≈−22, DPivot @ x≈−8, AM-3 @ x≈+5, ST @ x≈+20
• 3-5-2 Wing Backs   : CB-3 @ x≈−28 narrow, WBs @ x≈−8 ±30 wide, Mid-3 central, 2ST
• Double Pivot       : Back-4 @ x≈−28, DM-pair @ x≈−12 narrow, rest spread above
• Inverted Full Backs: Back-4 with FBs tucked to y≈±8 (narrow), rest standard
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.zsl_classifier import ZSLTacticalClassifier, CLIP_AVAILABLE  # noqa: E402

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
# Pitch dimensions in metres (FIFA standard)
PITCH_LENGTH_M = 105.0   # x-axis
PITCH_WIDTH_M = 68.0     # y-axis

# Pixel canvas size used by ZSLTacticalClassifier.render_gaussian_heatmap
CANVAS_W_PX = 1050
CANVAS_H_PX = 680

# Pixel-per-metre scale factors
PX_PER_M_X = CANVAS_W_PX / PITCH_LENGTH_M   # 10 px/m
PX_PER_M_Y = CANVAS_H_PX / PITCH_WIDTH_M    # 10 px/m

# ── Formation canonical geometries ───────────────────────────────────────────
# All coordinates in metres; origin at pitch centre.
# 11 outfield players (no GK for cleaner tactical signal).
# Each tuple: (x_m, y_m)

FORMATION_TEMPLATES: dict[str, list[tuple[float, float]]] = {

    "4-3-3 High Press": [
        # Back 4 — high but grouped
        (-32.0, -14.0), (-32.0, -5.0), (-32.0, +5.0), (-32.0, +14.0),
        # Mid 3 — box-to-box across centre
        (-10.0, -10.0), (-10.0,  0.0), (-10.0, +10.0),
        # Front 3 — wide + pushed into opp half
        (+28.0, -20.0), (+24.0,  0.0), (+28.0, +20.0),
        # Extra pressing player (shadow striker)
        (+18.0, -6.0),
    ],

    "4-2-3-1 Mid Block": [
        # Back 4 — own half
        (-22.0, -14.0), (-22.0, -5.0), (-22.0, +5.0), (-22.0, +14.0),
        # Double pivot — narrow, deep
        ( -8.0,  -5.0), ( -8.0,  +5.0),
        # AM line — horizontal, middle third
        (  5.0, -15.0), (  5.0,   0.0), (  5.0, +15.0),
        # Lone striker
        (+20.0,   0.0),
        # 11th (floating CAM for block)
        (  0.0,   0.0),
    ],

    "3-5-2 Wing Backs": [
        # Back 3 — narrow CBs
        (-28.0, -8.0), (-28.0,  0.0), (-28.0, +8.0),
        # Wing backs — very wide + higher
        ( -8.0, -30.0), ( -8.0, +30.0),
        # Mid 3 — central cluster
        (  0.0, -8.0), (  0.0,  0.0), (  0.0, +8.0),
        # 2 Strikers — paired
        (+22.0, -6.0), (+22.0, +6.0),
        # 11th (attacking mid)
        (+10.0,  0.0),
    ],

    "Double Pivot": [
        # Back 4 — standard depth
        (-28.0, -14.0), (-28.0, -5.0), (-28.0, +5.0), (-28.0, +14.0),
        # Double pivot — very narrow pair, clearly visible
        (-12.0,  -4.0), (-12.0,  +4.0),
        # Mid / attack — spread above the pivot
        (  0.0, -12.0), (  0.0,  0.0), (  0.0, +12.0),
        # Forwards
        (+18.0, -8.0), (+18.0, +8.0),
    ],

    "Inverted Full Backs": [
        # Centre-backs — standard
        (-28.0, -8.0), (-28.0,  0.0), (-28.0, +8.0),
        # Inverted full backs — tucked narrow into midfield, not wide
        (-14.0,  -8.0), (-14.0,  +8.0),
        # Inside forwards (wide positions)
        (+15.0, -22.0), (+15.0, +22.0),
        # Central midfielders
        ( -4.0, -5.0), ( -4.0, +5.0),
        # Advanced players
        (+22.0,  0.0),
        # 11th (attacking support)
        (  8.0,  0.0),
    ],
}

FORMATION_NAMES: list[str] = list(FORMATION_TEMPLATES.keys())

# Fixed baseline similarity means from seed=999 synthetic calibration set
CALIBRATION_MEANS = {
    "4-3-3 High Press": 0.3447,
    "4-2-3-1 Mid Block": 0.3204,
    "3-5-2 Wing Backs": 0.2941,
    "Double Pivot": 0.3217,
    "Inverted Full Backs": 0.3341
}


def metres_to_pixels(
    points_m: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """
    Convert pitch-metric coordinates (origin=centre) to pixel coordinates
    (origin=top-left corner) as used by ZSLTacticalClassifier.render_gaussian_heatmap.

    x_m ∈ [−52.5, +52.5]  →  px_x ∈ [0, 1050]
    y_m ∈ [−34,   +34  ]  →  px_y ∈ [0, 680 ]
    """
    px_points: list[tuple[float, float]] = []
    for x_m, y_m in points_m:
        px_x = (x_m + PITCH_LENGTH_M / 2.0) * PX_PER_M_X
        px_y = (y_m + PITCH_WIDTH_M / 2.0) * PX_PER_M_Y
        # Clamp to canvas
        px_x = float(np.clip(px_x, 0, CANVAS_W_PX - 1))
        px_y = float(np.clip(px_y, 0, CANVAS_H_PX - 1))
        px_points.append((px_x, px_y))
    return px_points


def generate_clips(
    n_per_class: int = 20,
    jitter_sigma_m: float = 3.0,
    seed: int = 42,
) -> tuple[list[list[tuple[float, float]]], list[str]]:
    """
    Generate n_per_class synthetic clips per formation class.

    Each clip is a list of 11 (px_x, px_y) positions (already converted to
    pixel space for direct use with `classify_frame_batch`).

    Returns
    -------
    clips : list of N*5 pixel-coordinate lists
    labels: list of N*5 ground-truth formation name strings
    """
    rng = np.random.default_rng(seed)
    clips: list[list[tuple[float, float]]] = []
    labels: list[str] = []

    for formation_name, template_m in FORMATION_TEMPLATES.items():
        template_arr = np.array(template_m, dtype=np.float32)  # shape (11, 2)
        for _ in range(n_per_class):
            # Add Gaussian jitter in metric space
            jitter = rng.normal(0, jitter_sigma_m, size=template_arr.shape).astype(np.float32)
            jittered_m = template_arr + jitter
            # Convert to pixel space
            px_points = metres_to_pixels([(float(p[0]), float(p[1])) for p in jittered_m])
            clips.append(px_points)
            labels.append(formation_name)

    return clips, labels


def run_evaluation(
    n_clips: int = 20,
    tau: float = 0.25,
    seed: int = 42,
    out_path: Path | None = None,
    calibrate: bool = False,
) -> dict[str, Any]:
    """
    Full evaluation pipeline.

    Returns a results dict containing:
        - overall_top1_acc_argmax  : float
        - overall_top1_acc_tau     : float
        - per_class                : dict per formation
        - confusion_matrix         : list[list[int]]
        - formation_names          : list[str]
        - n_clips_per_class        : int
        - raw_predictions          : list[str]
        - raw_labels               : list[str]
    """
    if not CLIP_AVAILABLE:
        raise RuntimeError(
            "openai-clip is not installed. Run:\n"
            "  pip install git+https://github.com/openai/CLIP.git\n"
            "or: pip install openai-clip"
        )

    # ── 1. Load classifier ───────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════════════════════")
    print("   ZSTR CLASSIFIER EVALUATION — GAFFER'S GUIDE")
    print("══════════════════════════════════════════════════════════════════")
    print(f"\n[1/4] Loading ZSL Tactical Classifier (CLIP ViT-B/32)...")
    classifier = ZSLTacticalClassifier()

    if not classifier.tactical_patterns:
        raise RuntimeError(
            "ZSLTacticalClassifier loaded 0 tactical patterns.\n"
            "Check that backend/data/zsl_tactics.json has a 'tactical_patterns' key."
        )

    loaded_names = [p["name"] for p in classifier.tactical_patterns]
    print(f"    ✓ Loaded {len(loaded_names)} formation pattern(s):")
    for n in loaded_names:
        print(f"      • {n}")

    # Verify all 5 expected formations are present
    missing = [f for f in FORMATION_NAMES if f not in loaded_names]
    if missing:
        raise RuntimeError(
            f"Missing formation pattern(s) in zsl_tactics.json: {missing}\n"
            "Ensure all 5 entries exist in the 'tactical_patterns' array."
        )

    # ── 2. Generate synthetic clips ──────────────────────────────────────────
    print(f"\n[2/4] Generating synthetic test clips...")
    print(f"      {n_clips} clips × {len(FORMATION_NAMES)} classes = {n_clips * len(FORMATION_NAMES)} total")
    print(f"      Jitter σ=3.0 m | Random seed={seed}")
    clips, gt_labels = generate_clips(n_per_class=n_clips, jitter_sigma_m=3.0, seed=seed)
    print(f"    ✓ Generated {len(clips)} clips")

    # ── 3. Run CLIP inference ────────────────────────────────────────────────
    print(f"\n[3/4] Running ZSTR inference...")
    all_scores: list[dict[str, float]] = []

    # Process in batches of 32 to mirror production usage
    batch_size = 32
    for i in range(0, len(clips), batch_size):
        batch = clips[i : i + batch_size]
        batch_results = classifier.classify_frame_batch(batch)
        all_scores.extend(batch_results)
        done = min(i + batch_size, len(clips))
        print(f"      ... processed {done}/{len(clips)} clips", end="\r")

    print(f"    ✓ Inference complete — {len(all_scores)} clips scored        ")

    # ── 4. Compute metrics ───────────────────────────────────────────────────
    print(f"\n[4/4] Computing evaluation metrics (τ={tau})...")

    # Build prediction lists
    pred_argmax: list[str] = []   # Top-1 by argmax
    pred_tau: list[str] = []      # Top-1 with τ gate

    for score_dict in all_scores:
        if not score_dict:
            pred_argmax.append("No Match")
            pred_tau.append("No Match")
            continue
        if calibrate:
            cal_dict = {name: val - CALIBRATION_MEANS[name] for name, val in score_dict.items()}
        else:
            cal_dict = score_dict
            
        best_name = max(cal_dict, key=lambda k: cal_dict[k])
        best_score = score_dict[best_name]
        pred_argmax.append(best_name)
        pred_tau.append(best_name if best_score >= tau else "No Match")

    gt_arr = np.array(gt_labels)
    pred_argmax_arr = np.array(pred_argmax)
    pred_tau_arr = np.array(pred_tau)

    # ── Overall accuracy ─────────────────────────────────────────────────────
    top1_argmax = float((gt_arr == pred_argmax_arr).mean()) * 100.0
    # For tau accuracy: clips below threshold are counted as wrong
    top1_tau = float((gt_arr == pred_tau_arr).mean()) * 100.0

    # ── Per-class precision / recall / F1 ────────────────────────────────────
    per_class_results: dict[str, dict[str, Any]] = {}

    for formation in FORMATION_NAMES:
        gt_binary = (gt_arr == formation).astype(int)
        pred_binary = (pred_argmax_arr == formation).astype(int)

        tp = int(((gt_binary == 1) & (pred_binary == 1)).sum())
        fp = int(((gt_binary == 0) & (pred_binary == 1)).sum())
        fn = int(((gt_binary == 1) & (pred_binary == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        # Most common misclassification (when GT=formation but pred≠formation)
        mask_wrong = (gt_arr == formation) & (pred_argmax_arr != formation)
        wrong_preds = pred_argmax_arr[mask_wrong]
        if len(wrong_preds) > 0:
            most_common_error = Counter(wrong_preds).most_common(1)[0][0]
        else:
            most_common_error = "—"

        per_class_results[formation] = {
            "n_clips": n_clips,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "most_common_misclassification": most_common_error,
        }

    # ── Confusion matrix ──────────────────────────────────────────────────────
    conf_matrix: list[list[int]] = [
        [0] * len(FORMATION_NAMES) for _ in range(len(FORMATION_NAMES))
    ]
    name_to_idx = {n: i for i, n in enumerate(FORMATION_NAMES)}

    for gt, pred in zip(gt_labels, pred_argmax):
        gt_idx = name_to_idx.get(gt, -1)
        pred_idx = name_to_idx.get(pred, -1)
        if gt_idx >= 0 and pred_idx >= 0:
            conf_matrix[gt_idx][pred_idx] += 1
        elif gt_idx >= 0:
            # "No Match" prediction — doesn't map to any column; counted as FN
            pass

    # ── Collect average similarity scores per class (diagnostic) ─────────────
    avg_similarity: dict[str, dict[str, float]] = {}
    for formation in FORMATION_NAMES:
        mask = gt_arr == formation
        class_scores = [all_scores[i] for i, m in enumerate(mask) if m]
        avg_sim: dict[str, float] = {}
        for f2 in FORMATION_NAMES:
            vals = [s.get(f2, 0.0) for s in class_scores if s]
            avg_sim[f2] = round(float(np.mean(vals)) if vals else 0.0, 4)
        avg_similarity[formation] = avg_sim

    results: dict[str, Any] = {
        "overall_top1_acc_argmax": round(top1_argmax, 2),
        "overall_top1_acc_tau": round(top1_tau, 2),
        "tau": tau,
        "n_clips_per_class": n_clips,
        "n_total_clips": len(clips),
        "formation_names": FORMATION_NAMES,
        "per_class": per_class_results,
        "confusion_matrix": conf_matrix,
        "avg_similarity_scores": avg_similarity,
        "raw_labels": gt_labels,
        "raw_predictions_argmax": pred_argmax,
        "raw_predictions_tau": pred_tau,
    }

    return results


# ── Pretty-print functions ────────────────────────────────────────────────────

def _col(text: str, width: int, align: str = "left") -> str:
    """Right or left pad a string to `width` characters."""
    if align == "right":
        return str(text).rjust(width)
    return str(text).ljust(width)


def print_results_table(results: dict[str, Any]) -> None:
    """Print a clean, paper-ready results table to stdout."""
    formations = results["formation_names"]
    per_class  = results["per_class"]
    conf       = results["confusion_matrix"]
    n_total    = results["n_total_clips"]
    tau        = results["tau"]

    SEP  = "═" * 78
    SEP2 = "─" * 78

    print()
    print(SEP)
    print("   ZSTR CLASSIFIER EVALUATION — GAFFER'S GUIDE")
    print(SEP)
    print()
    print(f"  Formation Classes  : {len(formations)}")
    print(f"  Total Clips Tested : {n_total}  ({results['n_clips_per_class']} per class)")
    print(f"  CLIP Backbone      : ViT-B/32")
    print(f"  Cosine-Sim τ       : {tau}")
    print(f"  Coord Jitter σ     : 3.0 m (Gaussian)")
    print(f"  Pitch Space        : 105 × 68 m  →  1050 × 680 px canvas")
    print(f"  Heatmap σ          : 10 px (viridis + pitch lines overlay)")
    print()

    # ── Overall accuracy ──────────────────────────────────────────────────────
    print(SEP2)
    print("  OVERALL ACCURACY")
    print(SEP2)
    print(f"  Top-1 Accuracy  (argmax, no threshold)  :  {results['overall_top1_acc_argmax']:6.2f}%")
    print(f"  Top-1 Accuracy  (τ ≥ {tau}, strict gate):  {results['overall_top1_acc_tau']:6.2f}%")
    print()

    # ── Per-class table ───────────────────────────────────────────────────────
    print(SEP2)
    print("  PER-CLASS METRICS  (argmax Top-1)")
    print(SEP2)

    # Column header
    h_form   = _col("Formation",          24)
    h_clips  = _col("Clips", 6, "right")
    h_prec   = _col("Precision", 10, "right")
    h_rec    = _col("Recall",    10, "right")
    h_f1     = _col("F1",         8, "right")
    h_err    = "  Most Common Error"
    print(f"  {h_form}{h_clips}{h_prec}{h_rec}{h_f1}{h_err}")
    print("  " + "─" * 76)

    for formation in formations:
        pc   = per_class[formation]
        err  = pc["most_common_misclassification"]
        err_display = f"→ {err}" if err != "—" else "—"

        c_form  = _col(formation,            24)
        c_clips = _col(pc["n_clips"],         6, "right")
        c_prec  = _col(f"{pc['precision']:.4f}", 10, "right")
        c_rec   = _col(f"{pc['recall']:.4f}",    10, "right")
        c_f1    = _col(f"{pc['f1']:.4f}",         8, "right")
        print(f"  {c_form}{c_clips}{c_prec}{c_rec}{c_f1}  {err_display}")

    print()

    # ── Confusion matrix ──────────────────────────────────────────────────────
    print(SEP2)
    print("  CONFUSION MATRIX  (rows = Ground Truth · cols = Predicted)")
    print(SEP2)

    # Short labels for matrix columns
    short = {
        "4-3-3 High Press":      "433-HP",
        "4-2-3-1 Mid Block":     "4231-MB",
        "3-5-2 Wing Backs":      "352-WB",
        "Double Pivot":          "2Pivot",
        "Inverted Full Backs":   "IFB",
    }
    col_labels = [short.get(f, f[:7]) for f in formations]

    # Header row
    row_w = 22
    col_w = 9
    header = " " * (row_w + 4) + "".join(_col(c, col_w, "right") for c in col_labels)
    print("  " + header)
    print("  " + " " * (row_w + 4) + "─" * (col_w * len(formations)))

    for i, formation in enumerate(formations):
        row_label = _col(short.get(formation, formation[:22]), row_w)
        row_vals  = "".join(
            _col(str(conf[i][j]), col_w, "right") for j in range(len(formations))
        )
        print(f"  {row_label}  | {row_vals}")

    print()

    # ── Average CLIP similarity scores ────────────────────────────────────────
    print(SEP2)
    print("  MEAN COSINE SIMILARITY  (per GT class vs. each formation text prompt)")
    print(SEP2)

    avg_sim = results.get("avg_similarity_scores", {})
    header2 = " " * (row_w + 4) + "".join(_col(c, col_w, "right") for c in col_labels)
    print("  " + header2)
    print("  " + " " * (row_w + 4) + "─" * (col_w * len(formations)))

    for formation in formations:
        sims = avg_sim.get(formation, {})
        row_label = _col(short.get(formation, formation[:22]), row_w)
        row_vals  = "".join(
            _col(f"{sims.get(f2, 0.0):.3f}", col_w, "right") for f2 in formations
        )
        print(f"  {row_label}  | {row_vals}")

    print()
    print(SEP)
    print("  Evaluation complete.")
    print(SEP)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ZSTR Classifier Evaluation — Gaffer's Guide"
    )
    parser.add_argument(
        "--n-clips", type=int, default=20,
        help="Synthetic clips per formation class (default: 20)",
    )
    parser.add_argument(
        "--tau", type=float, default=0.25,
        help="Cosine-similarity threshold (default: 0.25)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for position jitter (default: 42)",
    )
    parser.add_argument(
        "--out", type=str,
        default=str(BACKEND_ROOT / "output" / "zstr_eval_results.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Apply logit calibration using fixed baseline means",
    )
    args = parser.parse_args()

    results = run_evaluation(
        n_clips=args.n_clips,
        tau=args.tau,
        seed=args.seed,
        calibrate=args.calibrate,
    )

    print_results_table(results)

    # Save JSON (exclude bulky raw lists for cleaner file unless debug needed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_results = {k: v for k, v in results.items()
                    if k not in ("raw_labels", "raw_predictions_argmax", "raw_predictions_tau")}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
