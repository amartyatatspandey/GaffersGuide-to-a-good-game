#!/usr/bin/env python3
"""
train_cnn_baseline.py — Train Supervised CNN Baseline Model on Real Footage Heatmaps.
===================================================================================

This script:
1. Loads the labeled heatmaps from backend/data/cnn_baseline/manifest.csv.
2. Filters out UNLABELED and UNCLEAR / SKIP clips.
3. Splits the data into stratified Train/Val sets (80/20).
4. Sets up a PyTorch Dataset/DataLoader with conservative data augmentation:
   - Light random rotation ([-5, 5] degrees)
   - Light additive Gaussian noise
   - ImageNet normalization
5. Fine-tunes a pretrained ResNet-18 (5-class output).
6. Saves the best model according to validation Macro-F1 to backend/models/cnn_baseline_resnet18.pt.
7. Logs performance metrics and flags warning if any class has < 15 samples.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ── Path Setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

# Constants
CLASSES = [
    "4-3-3 High Press",
    "4-2-3-1 Mid Block",
    "3-5-2 Wing Backs",
    "Double Pivot",
    "Inverted Full Backs"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}

# Custom Gaussian Noise Transform
class AddGaussianNoise(object):
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class TacticalHeatmapDataset(Dataset):
    def __init__(self, clips: list[dict[str, Any]], transform: Any = None):
        self.clips = clips
        self.transform = transform

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        clip = self.clips[idx]
        heatmap_path = BACKEND_ROOT.parent / clip["heatmap_path"]
        
        # Open image and convert to RGB
        img = Image.open(heatmap_path).convert("RGB")
        
        # Label to idx
        label_idx = CLASS_TO_IDX[clip["label"]]
        
        if self.transform:
            img_tensor = self.transform(img)
        else:
            from torchvision import transforms
            img_tensor = transforms.ToTensor()(img)
            
        return img_tensor, label_idx

def load_labeled_data() -> list[dict[str, Any]]:
    manifest_path = BACKEND_ROOT / "data" / "cnn_baseline" / "manifest.csv"
    if not manifest_path.is_file():
        LOGGER.error("Manifest not found at %s. Please generate and label data first.", manifest_path)
        sys.exit(1)
        
    clips = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Exclude UNLABELED and UNCLEAR / SKIP
            if row["label"] in CLASSES:
                clips.append({
                    "clip_id": row["clip_id"],
                    "video_source": row["video_source"],
                    "frame_range": row["frame_range"],
                    "heatmap_path": row["heatmap_path"],
                    "label": row["label"]
                })
    return clips

def calculate_metrics(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict[str, Any]:
    per_class_results = {}
    f1_list, prec_list, rec_list = [], [], []
    
    for idx, formation in enumerate(CLASSES):
        gt_binary = (gt_arr == idx).astype(int)
        pred_binary = (pred_arr == idx).astype(int)
        
        tp = int(((gt_binary == 1) & (pred_binary == 1)).sum())
        fp = int(((gt_binary == 0) & (pred_binary == 1)).sum())
        fn = int(((gt_binary == 1) & (pred_binary == 0)).sum())
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        
        prec_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)
        
        per_class_results[formation] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
        
    return {
        "per_class": per_class_results,
        "macro": {
            "precision": float(np.mean(prec_list)),
            "recall": float(np.mean(rec_list)),
            "f1": float(np.mean(f1_list))
        }
    }

def train_model(epochs: int = 15, batch_size: int = 16, lr: float = 1e-4) -> None:
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 1. Load data
    clips = load_labeled_data()
    LOGGER.info("Loaded %d labeled training/validation clips from manifest.", len(clips))
    
    if not clips:
        LOGGER.error("No valid labeled clips for training. Make sure you labeled the dataset!")
        sys.exit(1)
        
    # Count classes and check safeguards
    class_counts = {name: 0 for name in CLASSES}
    for c in clips:
        class_counts[c["label"]] += 1
        
    LOGGER.info("Class distribution in labeled dataset:")
    low_sample_warning = False
    for name, count in class_counts.items():
        LOGGER.info("  • %-25s: %d clips", name, count)
        if count < 15:
            low_sample_warning = True
            
    if low_sample_warning:
        print("\n" + "!"*78)
        print(" WARNING: Some classes have fewer than 15 examples!")
        print(" This is smaller than the recommended minimum size of 15 examples per class.")
        print("!"*78 + "\n")
        
    # 2. Train/Val Split (Stratified 80/20)
    labels = [CLASS_TO_IDX[c["label"]] for c in clips]
    
    from sklearn.model_selection import train_test_split
    train_clips, val_clips = train_test_split(
        clips,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )
    LOGGER.info("Split dataset into %d training samples and %d validation samples.", len(train_clips), len(val_clips))
    
    # 3. Setup transforms
    import torchvision.transforms as transforms
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.01),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = TacticalHeatmapDataset(train_clips, transform=train_transform)
    val_dataset = TacticalHeatmapDataset(val_clips, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Load pretrained ResNet-18
    import torchvision.models as models
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Training on device: %s", device)
    
    # Handle older torchvision versions safely
    try:
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except ImportError:
        model = models.resnet18(pretrained=True)
        
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_macro_f1 = 0.0
    models_dir = BACKEND_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "cnn_baseline_resnet18.pt"
    
    # 5. Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_gts = []
        val_preds = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                val_gts.extend(targets.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                
        val_loss /= len(val_dataset)
        
        # Calculate metrics
        val_gts_arr = np.array(val_gts)
        val_preds_arr = np.array(val_preds)
        metrics = calculate_metrics(val_gts_arr, val_preds_arr)
        epoch_macro_f1 = metrics["macro"]["f1"]
        
        LOGGER.info(
            "Epoch %d/%d — Train Loss: %.4f | Val Loss: %.4f | Val Macro-F1: %.4f",
            epoch, epochs, train_loss, val_loss, epoch_macro_f1
        )
        
        # Save best checkpoint
        if epoch_macro_f1 > best_macro_f1:
            best_macro_f1 = epoch_macro_f1
            torch.save(model.state_dict(), best_model_path)
            LOGGER.info("  => Saved new best model to %s (Val Macro-F1: %.4f)", best_model_path.name, best_macro_f1)
            
    # Load best weights to report final metrics
    LOGGER.info("\nLoading best model checkpoint for final evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    val_gts = []
    val_preds = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_gts.extend(targets.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
            
    val_gts_arr = np.array(val_gts)
    val_preds_arr = np.array(val_preds)
    final_metrics = calculate_metrics(val_gts_arr, val_preds_arr)
    
    # Print formatted validation report
    SEP = "═" * 78
    SEP2 = "─" * 78
    print("\n" + SEP)
    print("   SUPERVISED CNN BASELINE FINAL VALIDATION REPORT")
    print(SEP)
    print(f"  Best Validation Macro-F1 : {best_macro_f1:.4f}")
    print(f"  Model Saved To            : {best_model_path}")
    print()
    print(SEP2)
    print("  PER-CLASS VALIDATION METRICS")
    print(SEP2)
    
    def _col(text: str, width: int, align: str = "left") -> str:
        return str(text).rjust(width) if align == "right" else str(text).ljust(width)
        
    print(f"  {_col('Formation', 24)}{_col('Precision', 12, 'right')}{_col('Recall', 12, 'right')}{_col('F1-Score', 12, 'right')}  Confusion (TP/FP/FN)")
    print("  " + "─" * 76)
    
    for f in CLASSES:
        pc = final_metrics["per_class"][f]
        p_val = f"{pc['precision']:.4f}"
        r_val = f"{pc['recall']:.4f}"
        f_val = f"{pc['f1']:.4f}"
        conf = f"{pc['tp']}/{pc['fp']}/{pc['fn']}"
        print(f"  {_col(f, 24)}{_col(p_val, 12, 'right')}{_col(r_val, 12, 'right')}{_col(f_val, 12, 'right')}  {conf}")
        
    print("  " + "─" * 76)
    macro = final_metrics["macro"]
    p_macro = f"{macro['precision']:.4f}"
    r_macro = f"{macro['recall']:.4f}"
    f_macro = f"{macro['f1']:.4f}"
    print(f"  {_col('Macro Average', 24)}{_col(p_macro, 12, 'right')}{_col(r_macro, 12, 'right')}{_col(f_macro, 12, 'right')}")
    print(SEP + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN Baseline on tactical heatmaps.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
