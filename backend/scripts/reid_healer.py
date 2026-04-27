"""
Lightweight person Re-ID feature extractor (OSNet x0.25, MSMT17 weights).

Downloads ``osnet_x0_25_msmt17.pt``, loads it with plain PyTorch, and returns
512-D embeddings for player crops. No SoccerNet / torchreid dependency.

OSNet architecture is vendored from deep-person-reid (MIT, Copyright Kaiyang Zhou).
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Paths & weight mirrors (Ultralytics CDN first, then public fallback)
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_DIR = BACKEND_ROOT / "models" / "pretrained"
WEIGHT_FILENAME = "osnet_x0_25_msmt17.pt"

OSNET_WEIGHT_URLS: tuple[str, ...] = (
    "https://weights.ultralytics.com/osnet_x0_25_msmt17.pt",
    "https://weights.ultralytics.com/models/osnet_x0_25_msmt17.pt",
    "https://weights.ultralytics.com/models/weights/osnet_x0_25_msmt17.pt",
    "https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt",
)

# Standard ReID / ImageNet-style normalization (matches torchreid training pipeline)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FEATURE_DIM = 512
REID_HEIGHT = 256
REID_WIDTH = 128


# =============================================================================
# OSNet x0.25 backbone (vendored, MIT — Kaiyang Zhou, deep-person-reid)
# =============================================================================
class _ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        IN: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        self.bn = (
            nn.InstanceNorm2d(out_channels, affine=True)
            if IN
            else nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class _Conv1x1(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class _Conv1x1Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.bn(x)


class _LightConv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return self.relu(x)


class _ChannelGate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_gates: int | None = None,
        gate_activation: str = "sigmoid",
        reduction: int = 16,
    ) -> None:
        super().__init__()
        if num_gates is None:
            num_gates = in_channels
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels, in_channels // reduction, 1, bias=True, padding=0
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction, num_gates, 1, bias=True, padding=0
        )
        if gate_activation == "sigmoid":
            self.gate_activation: nn.Module | None = nn.Sigmoid()
        elif gate_activation == "relu":
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == "linear":
            self.gate_activation = None
        else:
            raise RuntimeError(f"Unknown gate activation: {gate_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        return inp * x


class _OSBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        IN: bool = False,
        bottleneck_reduction: int = 4,
    ) -> None:
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = _Conv1x1(in_channels, mid_channels)
        self.conv2a = _LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            _LightConv3x3(mid_channels, mid_channels),
            _LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            _LightConv3x3(mid_channels, mid_channels),
            _LightConv3x3(mid_channels, mid_channels),
            _LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            _LightConv3x3(mid_channels, mid_channels),
            _LightConv3x3(mid_channels, mid_channels),
            _LightConv3x3(mid_channels, mid_channels),
            _LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = _ChannelGate(mid_channels)
        self.conv3 = _Conv1x1Linear(mid_channels, out_channels)
        self.downsample = (
            _Conv1x1Linear(in_channels, out_channels)
            if in_channels != out_channels
            else None
        )
        self.IN = nn.InstanceNorm2d(out_channels, affine=True) if IN else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


class OSNet(nn.Module):
    """Omni-Scale Network (feature_dim embedding before classifier)."""

    def __init__(
        self,
        num_classes: int,
        blocks: Sequence[type[nn.Module]],
        layers: Sequence[int],
        channels: Sequence[int],
        feature_dim: int = FEATURE_DIM,
        loss: str = "softmax",
        IN: bool = False,
    ) -> None:
        super().__init__()
        if not (len(blocks) == len(layers) == len(channels) - 1):
            raise ValueError("Invalid OSNet configuration")
        self.loss = loss
        self.feature_dim = feature_dim

        self.conv1 = _ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0], layers[0], channels[0], channels[1], True, IN=IN
        )
        self.conv3 = self._make_layer(
            blocks[1], layers[1], channels[1], channels[2], True, IN=False
        )
        self.conv4 = self._make_layer(
            blocks[2], layers[2], channels[2], channels[3], False, IN=False
        )
        self.conv5 = _Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(feature_dim, channels[3])
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(
        self,
        block: type[nn.Module],
        layer: int,
        in_channels: int,
        out_channels: int,
        reduce_spatial_size: bool,
        IN: bool = False,
    ) -> nn.Sequential:
        layers_list: list[nn.Module] = [block(in_channels, out_channels, IN=IN)]
        for _ in range(1, layer):
            layers_list.append(block(out_channels, out_channels, IN=IN))
        if reduce_spatial_size:
            layers_list.append(
                nn.Sequential(
                    _Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)
                )
            )
        return nn.Sequential(*layers_list)

    def _construct_fc_layer(self, fc_dim: int, input_dim: int) -> nn.Sequential | None:
        layers_fc = nn.Sequential(
            nn.Linear(input_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = fc_dim
        return layers_fc

    def _init_params(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == "softmax":
            return y
        if self.loss == "triplet":
            return y, v
        raise KeyError(f"Unsupported loss: {self.loss}")


def build_osnet_x0_25(num_classes: int) -> OSNet:
    """OSNet x0.25 topology (width multiplier 0.25)."""
    return OSNet(
        num_classes,
        blocks=[_OSBlock, _OSBlock, _OSBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        feature_dim=FEATURE_DIM,
        loss="softmax",
        IN=False,
    )


# =============================================================================
# Checkpoint helpers
# =============================================================================
def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        out[key] = v
    return out


def _unwrap_state_dict(ckpt: Any) -> dict[str, Any]:
    """Normalize .pt / .pth contents to a flat state_dict mapping."""
    if isinstance(ckpt, nn.Module):
        return dict(ckpt.state_dict())
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "net", "ema"):
            inner = ckpt.get(key)
            if isinstance(inner, nn.Module):
                return dict(inner.state_dict())
            if (
                isinstance(inner, dict)
                and inner
                and all(hasattr(v, "shape") for v in inner.values())
            ):
                return dict(inner)  # type: ignore[arg-type]
        if ckpt and all(hasattr(v, "shape") for v in ckpt.values()):
            return dict(ckpt)  # type: ignore[arg-type]
    raise ValueError("Checkpoint is not a recognized state_dict wrapper")


def _infer_num_classes(state_dict: dict[str, Any]) -> int:
    w = state_dict.get("classifier.weight")
    if w is not None and hasattr(w, "shape"):
        return int(w.shape[0])
    # MSMT17 combined training default
    return 4101


def _download_to_path(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "GaffersGuide-ReID/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as fh:
        fh.write(resp.read())


class VisualFingerprint:
    """
    Loads OSNet x0.25 (MSMT17) and extracts L2-normalized 512-D embeddings from player crops.
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        root = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        self.model_dir = root
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / WEIGHT_FILENAME

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing ReID VisualFingerprint on %s", self.device)

        self._ensure_model_downloaded()
        self.model = self._load_model()
        self.model.eval()

    def _ensure_model_downloaded(self) -> None:
        if self.model_path.is_file():
            return
        last_err: Exception | None = None
        for url in OSNET_WEIGHT_URLS:
            try:
                logger.info("Downloading OSNet weights from %s", url)
                _download_to_path(url, self.model_path)
                logger.info("Download complete: %s", self.model_path)
                return
            except Exception as exc:  # noqa: BLE001 — try next mirror
                last_err = exc
                logger.warning("Download failed (%s): %s", url, exc)
        raise RuntimeError(
            "Could not download OSNet weights from any mirror"
        ) from last_err

    def _load_model(self) -> OSNet:
        try:
            raw = torch.load(self.model_path, map_location="cpu", weights_only=False)
        except TypeError:
            raw = torch.load(self.model_path, map_location="cpu")
        state_dict = _unwrap_state_dict(raw)
        state_dict = _strip_module_prefix(state_dict)
        num_classes = _infer_num_classes(state_dict)
        model = build_osnet_x0_25(num_classes=num_classes)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("load_state_dict missing keys (non-strict): %s", missing[:8])
        if unexpected:
            logger.warning("load_state_dict unexpected keys: %s", unexpected[:8])
        return model.to(self.device)

    def extract_features(
        self,
        frame: np.ndarray,
        bbox: tuple[float, ...] | list[float] | np.ndarray,
        *,
        normalize: bool = True,
    ) -> np.ndarray | None:
        """
        Crop ``bbox`` (x1, y1, x2, y2), resize to ReID input, return 512-D vector.

        ``frame`` is BGR (OpenCV). Returns ``None`` for empty/invalid crops.
        """
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # ReID convention: height x width = 256 x 128
        crop_resized = cv2.resize(
            crop, (REID_WIDTH, REID_HEIGHT), interpolation=cv2.INTER_LINEAR
        )
        rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(rgb, (2, 0, 1))
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device, dtype=torch.float32)

        with torch.inference_mode():
            feat = self.model(tensor)
        vec = feat.squeeze(0).detach().float().cpu().numpy()
        if normalize:
            n = float(np.linalg.norm(vec))
            if n > 1e-12:
                vec = vec / n
        return vec.astype(np.float32, copy=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rng = np.random.default_rng(0)
    demo = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    demo_bgr = cv2.cvtColor(demo, cv2.COLOR_RGB2BGR)
    fp = VisualFingerprint()
    out = fp.extract_features(demo_bgr, (100, 50, 200, 400))
    logger.info("Smoke test embedding shape: %s", None if out is None else out.shape)
