"""
Pitch calibration model: ResNet18 backbone with 9-D homography regression head.

Outputs a flattened 3x3 homography matrix for camera calibration from a single
pitch image. Compatible with SoccerNetCalibrationDataset (224x224, ImageNet norm).
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


class PitchCalibrator(nn.Module):
    """ResNet18 backbone with final FC replaced to predict 9-D homography (flattened 3x3)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights="DEFAULT")
        in_features: int = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict 9-D homography from batch of images (N, C, H, W)."""
        return self.backbone(x)
