#!/usr/bin/env python3
"""
evaluate_real_zstr.py — Run ZSTR Classifier Evaluation on Manually Labeled Real Footage
===================================================================================

This script:
1. Loads manually labeled clips from backend/data/real_eval_set/manifest.csv.
2. Filters out UNLABELED and UNCLEAR / SKIP clips.
3. Loads the corresponding pre-rendered heatmap PNGs from disk.
4. Performs batch CLIP inference using the existing ZSLTacticalClassifier.
5. Computes Precision, Recall, and F1-score per class, along with the Macro Average.
6. Prints a formatted metrics table and confusion matrix to stdout.
7. Saves the class-level results to backend/output/real_eval_zstr_results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# ── Path Setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

# Import the existing classifier
from scripts.zsl_classifier import ZSLTacticalClassifier, CLIP_AVAILABLE

# Configuration / Constants
CLASSES = [
    "4-3-3 High Press",
    "4-2-3-1 Mid Block",
    "3-5-2 Wing Backs",
    "Double Pivot",
    "Inverted Full Backs"
]

# Fixed baseline similarity means from seed=999 synthetic calibration set
CALIBRATION_MEANS = {
    "4-3-3 High Press": 0.3447,
    "4-2-3-1 Mid Block": 0.3204,
    "3-5-2 Wing Backs": 0.2941,
    "Double Pivot": 0.3217,
    "Inverted Full Backs": 0.3341
}

def load_labeled_manifest() -> list[dict[str, str]]:
    manifest_path = BACKEND_ROOT / "data" / "real_eval_set" / "manifest.csv"
    if not manifest_path.is_file():
        LOGGER.error("Manifest not found at %s", manifest_path)
        sys.exit(1)
        
    clips = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clips.append({
                "clip_id": row["clip_id"],
                "video_source": row["video_source"],
                "frame_range": row["frame_range"],
                "heatmap_path": row["heatmap_path"],
                "label": row["label"]
            })
    return clips

def run_real_evaluation(tau: float = 0.25, calibrate: bool = False) -> dict[str, Any]:
    if not CLIP_AVAILABLE:
        raise RuntimeError("openai-clip is not installed. Please install it first.")
        
    # 1. Load classifier
    print(f"\n[1/3] Loading ZSL Tactical Classifier (CLIP ViT-B/32)...")
    classifier = ZSLTacticalClassifier()
    
    # 2. Load and filter manifest
    all_clips = load_labeled_manifest()
    eval_clips = [c for c in all_clips if c["label"] in CLASSES]
    ignored_clips = [c for c in all_clips if c["label"] not in CLASSES]
    
    print(f"      Loaded {len(all_clips)} total clips from manifest.")
    print(f"      ✓ Kept {len(eval_clips)} clips for evaluation (labeled as one of the 5 canonical classes).")
    print(f"      • Ignored {len(ignored_clips)} clips (UNLABELED or UNCLEAR / SKIP).")
    
    if not eval_clips:
        LOGGER.error("No valid labeled clips found for evaluation. Please label the dataset first.")
        sys.exit(1)
        
    # 3. Load preprocessed tensors of real heatmaps
    print(f"\n[2/3] Loading and preprocessing real heatmap PNGs...")
    preprocessed_tensors = []
    for c in eval_clips:
        # Resolve heatmap path relative to project root
        heatmap_path = BACKEND_ROOT.parent / c["heatmap_path"]
        if not heatmap_path.is_file():
            LOGGER.error("Heatmap PNG not found for clip %s at: %s", c["clip_id"], heatmap_path)
            sys.exit(1)
            
        img = Image.open(heatmap_path).convert("RGB")
        preprocessed_tensors.append(classifier.preprocess(img))
        
    # 4. Perform inference in batches of 32
    print(f"\n[3/3] Running CLIP inference on {len(preprocessed_tensors)} real heatmaps...")
    all_scores: list[dict[str, float]] = []
    
    batch_size = 32
    for i in range(0, len(preprocessed_tensors), batch_size):
        batch_tensors = preprocessed_tensors[i : i + batch_size]
        batch_input = torch.stack(batch_tensors).to(classifier.device)
        
        with torch.no_grad():
            image_features = classifier.model.encode_image(batch_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ classifier.text_features.T
            probs = similarities.cpu().numpy()
            
        for row_idx in range(len(batch_tensors)):
            score_dict = {
                classifier.tactical_patterns[j]["name"]: float(probs[row_idx, j])
                for j in range(len(classifier.tactical_patterns))
            }
            all_scores.append(score_dict)
            
    # 5. Compute predictions & metrics
    pred_argmax: list[str] = []
    pred_tau: list[str] = []
    
    for score_dict in all_scores:
        if calibrate:
            cal_dict = {name: val - CALIBRATION_MEANS[name] for name, val in score_dict.items()}
        else:
            cal_dict = score_dict
            
        best_name = max(cal_dict, key=lambda k: cal_dict[k])
        best_score = score_dict[best_name]
        pred_argmax.append(best_name)
        pred_tau.append(best_name if best_score >= tau else "No Match")
        
    gt_labels = [c["label"] for c in eval_clips]
    gt_arr = np.array(gt_labels)
    pred_argmax_arr = np.array(pred_argmax)
    pred_tau_arr = np.array(pred_tau)
    
    top1_argmax = float((gt_arr == pred_argmax_arr).mean()) * 100.0
    top1_tau = float((gt_arr == pred_tau_arr).mean()) * 100.0
    
    per_class_results = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    f1_list, prec_list, rec_list = [], [], []
    
    for formation in CLASSES:
        gt_binary = (gt_arr == formation).astype(int)
        pred_binary = (pred_argmax_arr == formation).astype(int)
        
        tp = int(((gt_binary == 1) & (pred_binary == 1)).sum())
        fp = int(((gt_binary == 0) & (pred_binary == 1)).sum())
        fn = int(((gt_binary == 1) & (pred_binary == 0)).sum())
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        
        # Track for scikit-learn style Macro Avg
        prec_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Find most common misclassification
        mask_wrong = (gt_arr == formation) & (pred_argmax_arr != formation)
        wrong_preds = pred_argmax_arr[mask_wrong]
        most_common_error = Counter(wrong_preds).most_common(1)[0][0] if len(wrong_preds) > 0 else "—"
        
        per_class_results[formation] = {
            "n_clips": int((gt_arr == formation).sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "most_common_misclassification": most_common_error
        }
        
    macro_precision = float(np.mean(prec_list))
    macro_recall = float(np.mean(rec_list))
    macro_f1 = float(np.mean(f1_list))
    
    # 6. Confusion Matrix
    conf_matrix = [[0] * len(CLASSES) for _ in range(len(CLASSES))]
    name_to_idx = {name: idx for idx, name in enumerate(CLASSES)}
    for gt, pred in zip(gt_labels, pred_argmax):
        gt_idx = name_to_idx.get(gt, -1)
        pred_idx = name_to_idx.get(pred, -1)
        if gt_idx >= 0 and pred_idx >= 0:
            conf_matrix[gt_idx][pred_idx] += 1
            
    # 7. Collect mean similarities
    avg_similarity = {}
    for formation in CLASSES:
        mask = gt_arr == formation
        class_scores = [all_scores[i] for i, m in enumerate(mask) if m]
        avg_sim = {}
        for f2 in CLASSES:
            vals = [s.get(f2, 0.0) for s in class_scores if s]
            avg_sim[f2] = round(float(np.mean(vals)) if vals else 0.0, 4)
        avg_similarity[formation] = avg_sim
        
    return {
        "overall_top1_acc_argmax": round(top1_argmax, 2),
        "overall_top1_acc_tau": round(top1_tau, 2),
        "tau": tau,
        "n_total_clips": len(eval_clips),
        "formation_names": CLASSES,
        "per_class": per_class_results,
        "confusion_matrix": conf_matrix,
        "avg_similarity_scores": avg_similarity,
        "macro_metrics": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4)
        }
    }

def print_results(results: dict[str, Any]) -> None:
    formations = results["formation_names"]
    per_class = results["per_class"]
    conf = results["confusion_matrix"]
    macro = results["macro_metrics"]
    
    SEP = "═" * 78
    SEP2 = "─" * 78
    
    print("\n" + SEP)
    print("   ZSTR CLASSIFIER EVALUATION ON REAL FOOTAGE")
    print(SEP)
    print(f"  Total Clips Evaluated : {results['n_total_clips']}")
    print(f"  CLIP Backbone          : ViT-B/32")
    print(f"  Cosine-Sim τ           : {results['tau']}")
    print()
    
    print(SEP2)
    print("  OVERALL ACCURACY")
    print(SEP2)
    print(f"  Top-1 Accuracy (argmax, no threshold)  : {results['overall_top1_acc_argmax']:.2f}%")
    print(f"  Top-1 Accuracy (τ ≥ {results['tau']}, strict gate): {results['overall_top1_acc_tau']:.2f}%")
    print()
    
    print(SEP2)
    print("  PER-CLASS METRICS (argmax Top-1)")
    print(SEP2)
    
    def _col(text: str, width: int, align: str = "left") -> str:
        return str(text).rjust(width) if align == "right" else str(text).ljust(width)
        
    print(f"  {_col('Formation', 24)}{_col('Clips', 8, 'right')}{_col('Precision', 12, 'right')}{_col('Recall', 12, 'right')}{_col('F1', 10, 'right')}  Most Common Error")
    print("  " + "─" * 76)
    
    for f in formations:
        pc = per_class[f]
        err = pc["most_common_misclassification"]
        err_disp = f"→ {err}" if err != "—" else "—"
        p_val = f"{pc['precision']:.4f}"
        r_val = f"{pc['recall']:.4f}"
        f_val = f"{pc['f1']:.4f}"
        print(f"  {_col(f, 24)}{_col(pc['n_clips'], 8, 'right')}{_col(p_val, 12, 'right')}{_col(r_val, 12, 'right')}{_col(f_val, 10, 'right')}  {err_disp}")
        
    print("  " + "─" * 76)
    p_macro = f"{macro['precision']:.4f}"
    r_macro = f"{macro['recall']:.4f}"
    f_macro = f"{macro['f1']:.4f}"
    print(f"  {_col('Macro Average', 24)}{_col(results['n_total_clips'], 8, 'right')}{_col(p_macro, 12, 'right')}{_col(r_macro, 12, 'right')}{_col(f_macro, 10, 'right')}")
    print()
    
    print(SEP2)
    print("  CONFUSION MATRIX (rows = Ground Truth | cols = Predicted)")
    print(SEP2)
    
    short = {
        "4-3-3 High Press": "433-HP",
        "4-2-3-1 Mid Block": "4231-MB",
        "3-5-2 Wing Backs": "352-WB",
        "Double Pivot": "2Pivot",
        "Inverted Full Backs": "IFB"
    }
    col_labels = [short.get(f, f[:7]) for f in formations]
    header = " " * 26 + "".join(_col(c, 9, "right") for c in col_labels)
    print("  " + header)
    print("  " + " " * 26 + "─" * 45)
    
    for i, f in enumerate(formations):
        row_label = _col(short.get(f, f[:22]), 22)
        row_vals = "".join(_col(str(conf[i][j]), 9, "right") for j in range(len(formations)))
        print(f"  {row_label}  |{row_vals}")
        
    print()
    print(SEP)
    print("  Evaluation complete.")
    print(SEP + "\n")

def save_metrics_to_csv(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Clips", "Precision", "Recall", "F1", "TP", "FP", "FN", "Most_Common_Error"])
        
        # Per-class rows
        for f_name in CLASSES:
            pc = results["per_class"][f_name]
            writer.writerow([
                f_name,
                pc["n_clips"],
                pc["precision"],
                pc["recall"],
                pc["f1"],
                pc["tp"],
                pc["fp"],
                pc["fn"],
                pc["most_common_misclassification"]
            ])
            
        # Macro average row
        macro = results["macro_metrics"]
        writer.writerow([
            "Macro Average",
            results["n_total_clips"],
            macro["precision"],
            macro["recall"],
            macro["f1"],
            "",
            "",
            "",
            ""
        ])
    LOGGER.info("Successfully saved real-match ZSTR evaluation results to: %s", output_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ZSTR model on manually labeled real evaluation clips.")
    parser.add_argument("--tau", type=float, default=0.25, help="Cosine similarity threshold (default: 0.25)")
    parser.add_argument("--calibrate", action="store_true", help="Apply logit calibration using fixed baseline means")
    parser.add_argument("--out-dir", type=str, default=str(BACKEND_ROOT / "output" / "real_eval_zstr_final"), help="Output directory for CSV results")
    args = parser.parse_args()
    
    results = run_real_evaluation(tau=args.tau, calibrate=args.calibrate)
    print_results(results)
    
    out_dir = Path(args.out_dir)
    filename = "calibrated_results.csv" if args.calibrate else "uncalibrated_results.csv"
    save_metrics_to_csv(results, out_dir / filename)

if __name__ == "__main__":
    main()
