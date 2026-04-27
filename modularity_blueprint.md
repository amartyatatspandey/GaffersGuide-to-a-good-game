# Gaffers Guide SDK Modularity Blueprint

**Version:** 2.0.0  
**Author:** SDK Architecture Team  
**Date:** April 2026  
**Status:** Implementation Ready

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Philosophy](#architecture-philosophy)
3. [Directory Structure](#directory-structure)
4. [Data Contracts (Core Types)](#data-contracts-core-types)
5. [Module Specifications](#module-specifications)
6. [Lazy Loading Strategy](#lazy-loading-strategy)
7. [Developer Usage Patterns](#developer-usage-patterns)
8. [Refactoring Roadmap](#refactoring-roadmap)
9. [Testing Strategy](#testing-strategy)
10. [Migration Guide](#migration-guide)
11. [Success Criteria](#success-criteria)

---

## Executive Summary

### Current State
`gaffers-guide` is a functional end-to-end CLI tool for football match analysis, but its internal architecture is monolithic. Importing any part of the library triggers heavy dependency loading (torch, ultralytics), making it unsuitable as a developer SDK.

### Target State
Transform `gaffers-guide` into a **scikit-learn-style modular SDK** where:
- Developers can import specific modules independently
- Heavy ML dependencies load lazily (only when needed)
- Clean, typed interfaces enable easy extension
- Existing CLI and frontend integrations remain functional

### Key Benefits
- **Memory Efficiency**: `from gaffers_guide.io import parse_json` uses ~50MB, not ~2GB
- **Developer Experience**: Import only what you need, no "kitchen sink" dependencies
- **Extensibility**: Clear interfaces for custom detectors, metrics, exporters
- **Maintainability**: Single-responsibility modules with <3 dependencies each

---

## Architecture Philosophy

### The Toolkit Analogy
Think of the transformation from a **Swiss Army knife** (monolithic) to a **professional toolkit** (modular):

```
BEFORE (Monolithic):
┌─────────────────────────────┐
│  gaffers_guide              │
│  ├─ Everything coupled      │
│  ├─ Imports torch always    │
│  └─ 2GB+ memory on import   │
└─────────────────────────────┘

AFTER (Modular):
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   vision    │  │   spatial   │  │     io      │
│  (YOLO CV)  │  │ (Homography)│  │  (Parsers)  │
│   ~2GB      │  │    ~50MB    │  │    ~10MB    │
└─────────────┘  └─────────────┘  └─────────────┘
        ↓                ↓                ↓
    ┌───────────────────────────────────────┐
    │         pipeline (Orchestrator)       │
    │    Composes modules as needed         │
    └───────────────────────────────────────┘
```

### Design Principles

1. **Lazy Loading**: Heavy dependencies (torch, ultralytics) load only when classes are instantiated
2. **Interface-First**: Data contracts (dataclasses) define module boundaries
3. **Dependency Injection**: High-level code receives instantiated objects (testability)
4. **Single Responsibility**: Each module does one thing exceptionally well
5. **Type Safety**: Full mypy strict compliance with runtime type checking

---

## Directory Structure

### Complete Package Layout

```
src/gaffers_guide/
│
├── __init__.py                      # Minimal: version + Pipeline only
├── py.typed                         # PEP 561 marker for mypy support
│
├── core/                            # Shared data structures & exceptions
│   ├── __init__.py                  # Exports: all types, exceptions
│   ├── types.py                     # Core dataclasses (FrameDetections, etc.)
│   ├── exceptions.py                # Custom exception hierarchy
│   └── validators.py                # Input validation utilities
│
├── vision/                          # Computer vision detection
│   ├── __init__.py                  # Exports: BallDetector, PlayerDetector
│   ├── _models.py                   # Heavy imports isolated (lazy cache)
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract BaseDetector protocol
│   │   ├── ball.py                  # BallDetector (YOLO + SAHI)
│   │   ├── player.py                # PlayerDetector (YOLO + ByteTrack)
│   │   └── pitch.py                 # PitchDetector (keypoint detection)
│   └── tracking/
│       ├── __init__.py              # Exports: ByteTrackWrapper
│       ├── bytetrack.py             # ByteTrack integration
│       └── state.py                 # TrackingState dataclass
│
├── spatial/                         # Coordinate transformations
│   ├── __init__.py                  # Exports: HomographyEngine, PitchMapper
│   ├── homography.py                # cv2 homography computation
│   ├── geometry.py                  # Pitch zones, extremities
│   ├── projections.py               # BBox → Pitch coordinate mapping
│   └── calibration.py               # Camera calibration utilities
│
├── tactical/                        # Match analysis & metrics
│   ├── __init__.py                  # Exports: MetricsCalculator, TeamClassifier
│   ├── metrics.py                   # Distance, speed, pass completion
│   ├── classification.py            # Team assignment (clustering)
│   ├── formations.py                # Formation detection (future)
│   └── events.py                    # Event detection (pass, shot, etc.)
│
├── io/                              # Data I/O operations
│   ├── __init__.py                  # Exports: VideoReader, parse_tracking_json
│   ├── video.py                     # cv2-based video reading/writing
│   ├── parsers.py                   # JSON/CSV tracking data parsers
│   ├── exporters.py                 # Standardized output writers
│   └── schemas.py                   # JSON schema validators
│
├── pipeline/                        # High-level orchestration
│   ├── __init__.py                  # Exports: MatchAnalysisPipeline
│   ├── e2e.py                       # End-to-end match processing
│   ├── stages.py                    # Modular pipeline stages
│   └── config.py                    # Pipeline configuration dataclass
│
└── cli/                             # Command-line interface
    ├── __init__.py
    ├── main.py                      # CLI entry point (uses pipeline)
    └── commands.py                  # Click command groups
```

### Key Structural Decisions

#### 1. `core/` as the Foundation
- **Why?** Every module depends on core types, but core depends on nothing
- **Analogy**: Core is the "language" all modules speak (like JSON for APIs)

#### 2. Flat Sub-Package Layout
- **Why?** Prevents deep nesting (`gaffers_guide.vision.detectors.ball.detector.BallDetector` 😱)
- **Rule**: Max 2 levels deep (`gaffers_guide.vision.BallDetector` ✅)

#### 3. Private `_models.py` Pattern
- **Why?** Isolates heavy imports behind a barrier
- **Naming**: `_` prefix signals "implementation detail, don't import directly"

---

## Data Contracts (Core Types)

All data structures live in `gaffers_guide/core/types.py`. These are the **"shipping containers"** passed between modules.

### Complete Type Definitions

```python
"""
Core data structures for gaffers-guide SDK.

These types define the contracts between modules. All modules depend on core,
but core depends on nothing (except standard library + numpy).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal, Protocol
from enum import Enum
import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Detection Types
# ============================================================================

@dataclass(frozen=True)
class BBoxDetection:
    """
    Single bounding box detection in pixel coordinates.
    
    Immutable to ensure detections aren't accidentally modified during pipeline.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    
    def __post_init__(self):
        """Validate bounding box coordinates."""
        if not (0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"Invalid bbox: ({self.x1},{self.y1})-({self.x2},{self.y2})")
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center point (x, y) of bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        """Width in pixels."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Height in pixels."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """Area in square pixels."""
        return self.width * self.height
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'bbox': [self.x1, self.y1, self.x2, self.y2],
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'track_id': self.track_id
        }


@dataclass
class FrameDetections:
    """
    All detections for a single video frame.
    
    Mutable to allow adding detections during processing stages.
    """
    frame_idx: int
    timestamp: float  # seconds from video start
    detections: List[BBoxDetection]
    frame_shape: Tuple[int, int]  # (height, width)
    
    def __post_init__(self):
        """Validate frame metadata."""
        if self.frame_idx < 0:
            raise ValueError(f"Frame index must be >= 0, got {self.frame_idx}")
        if self.timestamp < 0:
            raise ValueError(f"Timestamp must be >= 0, got {self.timestamp}")
    
    def filter_by_class(self, class_name: str) -> 'FrameDetections':
        """Return new FrameDetections with only specified class."""
        filtered = [d for d in self.detections if d.class_name == class_name]
        return FrameDetections(
            frame_idx=self.frame_idx,
            timestamp=self.timestamp,
            detections=filtered,
            frame_shape=self.frame_shape
        )
    
    def filter_by_confidence(self, min_confidence: float) -> 'FrameDetections':
        """Return detections above confidence threshold."""
        filtered = [d for d in self.detections if d.confidence >= min_confidence]
        return FrameDetections(
            frame_idx=self.frame_idx,
            timestamp=self.timestamp,
            detections=filtered,
            frame_shape=self.frame_shape
        )
    
    @property
    def player_count(self) -> int:
        """Number of player detections."""
        return len([d for d in self.detections if d.class_name == 'player'])
    
    @property
    def has_ball(self) -> bool:
        """Whether ball was detected in this frame."""
        return any(d.class_name == 'ball' for d in self.detections)


# ============================================================================
# Spatial Types
# ============================================================================

@dataclass(frozen=True)
class PitchCoordinate:
    """
    2D coordinate on standardized pitch (0-105m x 0-68m).
    
    Uses standard football pitch dimensions per FIFA regulations.
    Origin (0,0) is bottom-left corner from camera perspective.
    """
    x: float  # meters from left touchline (0-105)
    y: float  # meters from bottom goal line (0-68)
    confidence: Optional[float] = None
    
    def __post_init__(self):
        """Validate pitch coordinates."""
        if not (0 <= self.x <= 105):
            raise ValueError(f"Pitch x must be 0-105m, got {self.x}")
        if not (0 <= self.y <= 68):
            raise ValueError(f"Pitch y must be 0-68m, got {self.y}")
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
    
    def distance_to(self, other: 'PitchCoordinate') -> float:
        """Euclidean distance to another pitch coordinate in meters."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'x': self.x,
            'y': self.y,
            'confidence': self.confidence
        }


class PitchZone(Enum):
    """Tactical zones on the pitch."""
    DEFENSIVE_THIRD = "defensive_third"
    MIDDLE_THIRD = "middle_third"
    ATTACKING_THIRD = "attacking_third"
    LEFT_WING = "left_wing"
    CENTER = "center"
    RIGHT_WING = "right_wing"


@dataclass
class SpatialMapping:
    """
    Homography transformation between pixel space and pitch space.
    
    Immutable after computation to prevent accidental modification.
    """
    homography_matrix: NDArray[np.float64]  # 3x3 transformation matrix
    pitch_corners_px: NDArray[np.float64]   # 4x2 array of pixel coordinates
    pitch_corners_meters: NDArray[np.float64]  # 4x2 standard pitch coords
    frame_shape: Tuple[int, int]  # (height, width)
    inverse_matrix: Optional[NDArray[np.float64]] = None  # For pitch → pixel
    
    def __post_init__(self):
        """Validate and compute inverse matrix."""
        if self.homography_matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be 3x3")
        if self.pitch_corners_px.shape != (4, 2):
            raise ValueError("Pitch corners must be 4x2 array")
        
        # Compute inverse for reverse mapping
        if self.inverse_matrix is None:
            object.__setattr__(
                self, 
                'inverse_matrix', 
                np.linalg.inv(self.homography_matrix)
            )
    
    def pixel_to_pitch(self, px: Tuple[float, float]) -> PitchCoordinate:
        """
        Transform pixel coordinate to pitch coordinate.
        
        Args:
            px: (x, y) pixel coordinate
            
        Returns:
            PitchCoordinate in meters
        """
        # Homogeneous coordinates
        px_homo = np.array([px[0], px[1], 1.0])
        
        # Apply transformation
        pitch_homo = self.homography_matrix @ px_homo
        pitch_homo /= pitch_homo[2]  # Normalize
        
        # Clip to pitch boundaries
        x = np.clip(pitch_homo[0], 0, 105)
        y = np.clip(pitch_homo[1], 0, 68)
        
        return PitchCoordinate(x=x, y=y)
    
    def pitch_to_pixel(self, coord: PitchCoordinate) -> Tuple[float, float]:
        """
        Transform pitch coordinate back to pixel space.
        
        Useful for visualization overlays.
        """
        pitch_homo = np.array([coord.x, coord.y, 1.0])
        px_homo = self.inverse_matrix @ pitch_homo
        px_homo /= px_homo[2]
        
        return (px_homo[0], px_homo[1])


# ============================================================================
# Player & Team Types
# ============================================================================

@dataclass
class PlayerState:
    """
    Player state at a single timestamp.
    
    Combines detection, tracking, and spatial information.
    """
    track_id: int
    position_px: Tuple[float, float]
    position_pitch: Optional[PitchCoordinate]
    team_id: Optional[Literal[0, 1]] = None  # 0=home, 1=away
    bbox: Optional[BBoxDetection] = None
    jersey_number: Optional[int] = None
    velocity_mps: Optional[float] = None  # meters per second
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'track_id': self.track_id,
            'position_px': list(self.position_px),
            'position_pitch': self.position_pitch.to_dict() if self.position_pitch else None,
            'team_id': self.team_id,
            'bbox': self.bbox.to_dict() if self.bbox else None,
            'jersey_number': self.jersey_number,
            'velocity_mps': self.velocity_mps
        }


@dataclass
class MatchState:
    """
    Complete match state at a single frame.
    
    The "master" data structure that combines all analysis outputs.
    """
    frame_idx: int
    timestamp: float
    players: List[PlayerState]
    ball_position_px: Optional[Tuple[float, float]] = None
    ball_position_pitch: Optional[PitchCoordinate] = None
    spatial_mapping: Optional[SpatialMapping] = None
    possession_team: Optional[Literal[0, 1]] = None
    
    @property
    def home_players(self) -> List[PlayerState]:
        """Filter players by home team."""
        return [p for p in self.players if p.team_id == 0]
    
    @property
    def away_players(self) -> List[PlayerState]:
        """Filter players by away team."""
        return [p for p in self.players if p.team_id == 1]
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'frame_idx': self.frame_idx,
            'timestamp': self.timestamp,
            'players': [p.to_dict() for p in self.players],
            'ball_position_px': list(self.ball_position_px) if self.ball_position_px else None,
            'ball_position_pitch': self.ball_position_pitch.to_dict() if self.ball_position_pitch else None,
            'possession_team': self.possession_team
        }


# ============================================================================
# Protocol Definitions (Abstract Interfaces)
# ============================================================================

class Detector(Protocol):
    """Protocol for all detector implementations."""
    
    def detect(self, frame: NDArray[np.uint8]) -> FrameDetections:
        """Detect objects in frame."""
        ...


class MetricsCalculator(Protocol):
    """Protocol for tactical metrics calculators."""
    
    def calculate(self, states: List[MatchState]) -> dict:
        """Calculate metrics from sequence of match states."""
        ...
```

### Type Hierarchy Visualization

```
Core Types (No Dependencies)
    ↓
┌───────────────┬───────────────┬───────────────┐
│               │               │               │
BBoxDetection   PitchCoordinate  PlayerState
    ↓               ↓               ↓
FrameDetections SpatialMapping  MatchState
                                    ↓
                            (Used by all modules)
```

---

## Module Specifications

### 1. `vision` Module

**Purpose**: Computer vision object detection and tracking

**Dependencies**:
- **Heavy**: `torch`, `ultralytics`, `sahi` (lazy-loaded)
- **Light**: `numpy`, `opencv-python`

**Key Classes**:

```python
# vision/detectors/base.py
from abc import ABC, abstractmethod
from gaffers_guide.core.types import FrameDetections
import numpy as np

class BaseDetector(ABC):
    """
    Abstract base class for all detectors.
    
    Allows users to implement custom detectors that plug into pipeline.
    """
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> FrameDetections:
        """
        Run detection on a single frame.
        
        Args:
            frame: RGB numpy array of shape (H, W, 3)
            
        Returns:
            FrameDetections containing all detected objects
        """
        pass
    
    @abstractmethod
    def warmup(self) -> None:
        """
        Warm up the model (load weights, compile, etc.).
        
        Called before processing first frame to avoid cold-start penalty.
        """
        pass
```

```python
# vision/detectors/ball.py
from typing import Optional, TYPE_CHECKING
import numpy as np
from gaffers_guide.core.types import FrameDetections, BBoxDetection
from gaffers_guide.vision.detectors.base import BaseDetector

if TYPE_CHECKING:
    from ultralytics import YOLO

class BallDetector(BaseDetector):
    """
    Detect football/soccer ball using YOLO + SAHI.
    
    Lazy-loads torch/ultralytics only when detect() is first called.
    """
    
    def __init__(
        self, 
        model_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.3,
        use_sahi: bool = True,
        sahi_slice_size: int = 640
    ):
        """
        Initialize ball detector.
        
        Args:
            model_path: Path to YOLO weights (.pt file)
            device: 'cpu', 'cuda', or 'mps'
            confidence_threshold: Minimum confidence for detections
            use_sahi: Whether to use SAHI for small object detection
            sahi_slice_size: SAHI slice size in pixels
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.use_sahi = use_sahi
        self.sahi_slice_size = sahi_slice_size
        
        # Model NOT loaded yet (lazy loading)
        self._model: Optional['YOLO'] = None
        self._is_warmed_up = False
    
    def warmup(self) -> None:
        """Load model and run a dummy inference."""
        if not self._is_warmed_up:
            self._ensure_model_loaded()
            # Run dummy inference to compile CUDA kernels
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self._model(dummy_frame, verbose=False)
            self._is_warmed_up = True
    
    def detect(self, frame: np.ndarray) -> FrameDetections:
        """
        Detect ball in frame.
        
        Args:
            frame: RGB numpy array
            
        Returns:
            FrameDetections with ball detections
        """
        self._ensure_model_loaded()
        
        if self.use_sahi:
            results = self._detect_with_sahi(frame)
        else:
            results = self._detect_standard(frame)
        
        # Convert YOLO results to FrameDetections
        detections = []
        for box in results[0].boxes:
            bbox = BBoxDetection(
                x1=float(box.xyxy[0][0]),
                y1=float(box.xyxy[0][1]),
                x2=float(box.xyxy[0][2]),
                y2=float(box.xyxy[0][3]),
                confidence=float(box.conf[0]),
                class_id=int(box.cls[0]),
                class_name='ball'
            )
            if bbox.confidence >= self.confidence_threshold:
                detections.append(bbox)
        
        return FrameDetections(
            frame_idx=-1,  # Set by caller
            timestamp=-1.0,  # Set by caller
            detections=detections,
            frame_shape=(frame.shape[0], frame.shape[1])
        )
    
    def _ensure_model_loaded(self):
        """Lazy-load YOLO model on first use."""
        if self._model is None:
            from gaffers_guide.vision._models import get_yolo_model
            self._model = get_yolo_model(self.model_path, self.device)
    
    def _detect_with_sahi(self, frame: np.ndarray):
        """Run SAHI-based detection for small objects."""
        # Implementation details...
        pass
    
    def _detect_standard(self, frame: np.ndarray):
        """Run standard YOLO detection."""
        return self._model(frame, device=self.device, verbose=False)
```

```python
# vision/_models.py
"""
Heavy dependency isolation for vision module.

All torch/ultralytics imports happen here and are lazy-loaded.
"""
from typing import TYPE_CHECKING, Optional, Dict

if TYPE_CHECKING:
    from ultralytics import YOLO

# Global model cache to avoid reloading
_MODEL_CACHE: Dict[str, 'YOLO'] = {}

def get_yolo_model(model_path: str, device: str = "cpu") -> 'YOLO':
    """
    Lazy-load YOLO model with caching.
    
    Args:
        model_path: Path to .pt weights file
        device: Device to load model on
        
    Returns:
        Loaded YOLO model
        
    Raises:
        ImportError: If ultralytics not installed
    """
    cache_key = f"{model_path}:{device}"
    
    if cache_key not in _MODEL_CACHE:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Vision module requires ultralytics. "
                "Install with: pip install 'gaffers-guide[vision]'"
            )
        
        _MODEL_CACHE[cache_key] = YOLO(model_path).to(device)
    
    return _MODEL_CACHE[cache_key]

def clear_model_cache():
    """Clear model cache to free GPU memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
```

**Usage Example**:
```python
from gaffers_guide.vision import BallDetector
import cv2

# Instantiation is fast (no model loading)
detector = BallDetector(
    model_path="models/ball_v8.pt",
    device="cuda",
    confidence_threshold=0.4
)

# First detect() triggers lazy load
frame = cv2.imread("frame.jpg")
detections = detector.detect(frame)  # Loads torch/YOLO here

print(f"Found {len(detections.detections)} balls")
```

---

### 2. `spatial` Module

**Purpose**: Coordinate transformations and pitch geometry

**Dependencies**:
- `opencv-python` (for homography)
- `numpy`

**Key Classes**:

```python
# spatial/homography.py
import numpy as np
import cv2
from typing import Optional
from gaffers_guide.core.types import SpatialMapping, PitchCoordinate

class HomographyEngine:
    """
    Compute and apply homography transformations.
    
    Transforms between pixel coordinates (image space) and pitch coordinates
    (real-world meters).
    """
    
    def __init__(self, method: str = "ransac"):
        """
        Initialize homography engine.
        
        Args:
            method: 'ransac' or 'lmeds' for robust estimation
        """
        self.method = cv2.RANSAC if method == "ransac" else cv2.LMEDS
    
    def fit(
        self,
        pitch_corners_px: np.ndarray,
        frame_shape: tuple[int, int],
        pitch_corners_meters: Optional[np.ndarray] = None
    ) -> SpatialMapping:
        """
        Compute homography from detected pitch corners.
        
        Args:
            pitch_corners_px: 4x2 array of pixel coordinates (top-left, top-right, 
                             bottom-right, bottom-left)
            frame_shape: (height, width) of video frame
            pitch_corners_meters: Optional custom pitch dimensions. If None, uses
                                 standard FIFA pitch (105m x 68m)
                                 
        Returns:
            SpatialMapping with computed homography matrix
            
        Example:
            >>> engine = HomographyEngine()
            >>> corners_px = np.array([[120, 50], [1800, 45], [1900, 1030], [80, 1035]])
            >>> mapping = engine.fit(corners_px, frame_shape=(1080, 1920))
            >>> pitch_pos = mapping.pixel_to_pitch((960, 540))  # Center of frame
            >>> print(f"Position: {pitch_pos.x:.1f}m, {pitch_pos.y:.1f}m")
        """
        if pitch_corners_meters is None:
            # Standard FIFA pitch corners (meters)
            pitch_corners_meters = np.array([
                [0, 68],      # Top-left
                [105, 68],    # Top-right
                [105, 0],     # Bottom-right
                [0, 0]        # Bottom-left
            ], dtype=np.float64)
        
        # Compute homography matrix
        H, mask = cv2.findHomography(
            pitch_corners_px,
            pitch_corners_meters,
            method=self.method,
            ransacReprojThreshold=5.0
        )
        
        return SpatialMapping(
            homography_matrix=H,
            pitch_corners_px=pitch_corners_px,
            pitch_corners_meters=pitch_corners_meters,
            frame_shape=frame_shape
        )
    
    def fit_from_detections(
        self,
        keypoint_detections: dict,
        frame_shape: tuple[int, int]
    ) -> SpatialMapping:
        """
        Compute homography from pitch keypoint detections.
        
        Args:
            keypoint_detections: Dict mapping keypoint names to pixel coords
                                Example: {'center_circle': (960, 540), ...}
            frame_shape: (height, width) of frame
            
        Returns:
            SpatialMapping with computed homography
        """
        # Implementation: map keypoints to known pitch positions
        # Then call fit() with those correspondences
        pass
```

**Usage Example**:
```python
from gaffers_guide.spatial import HomographyEngine
from gaffers_guide.core.types import BBoxDetection
import numpy as np

# Initialize engine (no heavy dependencies)
engine = HomographyEngine(method="ransac")

# Fit from manually detected pitch corners
corners_px = np.array([
    [120, 50],     # Top-left
    [1800, 45],    # Top-right  
    [1900, 1030],  # Bottom-right
    [80, 1035]     # Bottom-left
])

mapping = engine.fit(
    pitch_corners_px=corners_px,
    frame_shape=(1080, 1920)
)

# Map player bbox to pitch coordinates
player_bbox = BBoxDetection(
    x1=450, y1=300, x2=490, y2=380,
    confidence=0.95, class_id=0, class_name="player"
)

pitch_pos = mapping.pixel_to_pitch(player_bbox.center)
print(f"Player at {pitch_pos.x:.1f}m, {pitch_pos.y:.1f}m")
```

---

### 3. `tactical` Module

**Purpose**: Match analysis and tactical metrics

**Dependencies**:
- `numpy`
- `scikit-learn` (for clustering, optional)

**Key Classes**:

```python
# tactical/metrics.py
from typing import List
from gaffers_guide.core.types import MatchState, PlayerState
import numpy as np

class MetricsCalculator:
    """
    Calculate tactical metrics from match states.
    
    Pure computation - no I/O, no ML dependencies.
    """
    
    def __init__(self, fps: float = 25.0):
        """
        Initialize metrics calculator.
        
        Args:
            fps: Video frame rate (for velocity calculations)
        """
        self.fps = fps
    
    def calculate_player_distance(
        self, 
        states: List[MatchState],
        track_id: int
    ) -> float:
        """
        Calculate total distance covered by a player.
        
        Args:
            states: Sequence of match states
            track_id: Player's tracking ID
            
        Returns:
            Total distance in meters
        """
        positions = []
        for state in states:
            player = next((p for p in state.players if p.track_id == track_id), None)
            if player and player.position_pitch:
                positions.append((player.position_pitch.x, player.position_pitch.y))
        
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        return total_distance
    
    def calculate_average_speed(
        self,
        states: List[MatchState],
        track_id: int
    ) -> float:
        """
        Calculate average speed in km/h.
        
        Args:
            states: Sequence of match states
            track_id: Player's tracking ID
            
        Returns:
            Average speed in km/h
        """
        distance_m = self.calculate_player_distance(states, track_id)
        duration_s = len(states) / self.fps
        
        if duration_s == 0:
            return 0.0
        
        speed_mps = distance_m / duration_s
        speed_kmh = speed_mps * 3.6
        
        return speed_kmh
    
    def calculate_possession_percentage(
        self,
        states: List[MatchState]
    ) -> dict:
        """
        Calculate possession percentage for each team.
        
        Args:
            states: Sequence of match states
            
        Returns:
            Dict with 'home' and 'away' possession percentages
        """
        possession_counts = {0: 0, 1: 0, None: 0}
        
        for state in states:
            possession_counts[state.possession_team] += 1
        
        total_frames = len(states)
        if total_frames == 0:
            return {'home': 0.0, 'away': 0.0}
        
        return {
            'home': (possession_counts[0] / total_frames) * 100,
            'away': (possession_counts[1] / total_frames) * 100
        }
```

---

### 4. `io` Module

**Purpose**: Data input/output operations

**Dependencies**:
- `opencv-python` (video I/O only)
- Standard library (`json`, `csv`)

**Key Classes**:

```python
# io/video.py
from typing import Iterator, Optional
import cv2
import numpy as np
from pathlib import Path

class VideoReader:
    """
    Efficient video frame reader with context manager support.
    
    Example:
        >>> with VideoReader("match.mp4") as reader:
        ...     for frame_idx, frame in reader:
        ...         # Process frame
    """
    
    def __init__(self, video_path: Path | str):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count: Optional[int] = None
        self._fps: Optional[float] = None
        self._frame_shape: Optional[tuple] = None
    
    def __enter__(self):
        """Context manager entry."""
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._cap:
            self._cap.release()
    
    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """
        Iterate over video frames.
        
        Yields:
            (frame_idx, frame_array) tuples
        """
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self._frame_shape is None:
                self._frame_shape = frame_rgb.shape[:2]
            
            yield frame_idx, frame_rgb
            frame_idx += 1
    
    @property
    def fps(self) -> float:
        """Video frame rate."""
        return self._fps
    
    @property
    def frame_count(self) -> int:
        """Total number of frames."""
        return self._frame_count
    
    @property
    def frame_shape(self) -> tuple:
        """Frame shape (height, width)."""
        return self._frame_shape
```

```python
# io/parsers.py
import json
from pathlib import Path
from typing import List
from gaffers_guide.core.types import MatchState, PlayerState, PitchCoordinate

def parse_tracking_json(json_path: Path | str) -> List[MatchState]:
    """
    Parse tracking data from JSON file.
    
    Args:
        json_path: Path to JSON file with tracking data
        
    Returns:
        List of MatchState objects
        
    Example:
        >>> from gaffers_guide.io import parse_tracking_json
        >>> states = parse_tracking_json("tracking_data.json")
        >>> print(f"Loaded {len(states)} frames")
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    match_states = []
    for frame_data in data['frames']:
        players = []
        for player_data in frame_data['players']:
            player = PlayerState(
                track_id=player_data['track_id'],
                position_px=tuple(player_data['position_px']),
                position_pitch=PitchCoordinate(**player_data['position_pitch']) 
                    if player_data.get('position_pitch') else None,
                team_id=player_data.get('team_id')
            )
            players.append(player)
        
        state = MatchState(
            frame_idx=frame_data['frame_idx'],
            timestamp=frame_data['timestamp'],
            players=players,
            ball_position_px=tuple(frame_data['ball_position_px']) 
                if frame_data.get('ball_position_px') else None,
            ball_position_pitch=PitchCoordinate(**frame_data['ball_position_pitch'])
                if frame_data.get('ball_position_pitch') else None
        )
        match_states.append(state)
    
    return match_states
```

**Usage Example**:
```python
from gaffers_guide.io import VideoReader, parse_tracking_json

# Read video without loading any ML models
with VideoReader("match.mp4") as reader:
    print(f"FPS: {reader.fps}, Frames: {reader.frame_count}")
    
    for frame_idx, frame in reader:
        if frame_idx > 100:
            break
        # Process frame...

# Parse tracking data (no dependencies beyond stdlib)
states = parse_tracking_json("tracking.json")
print(f"Loaded {len(states)} match states")
```

---

### 5. `pipeline` Module

**Purpose**: High-level orchestration

**Dependencies**:
- All other modules (but imports them only in methods, not at module level)

**Key Classes**:

```python
# pipeline/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PipelineConfig:
    """Configuration for match analysis pipeline."""
    
    # Model paths
    ball_detector_path: Path
    player_detector_path: Path
    pitch_detector_path: Path | None = None
    
    # Processing options
    device: str = "cpu"
    enable_tracking: bool = True
    enable_spatial_mapping: bool = True
    enable_tactical_analysis: bool = True
    
    # Detection thresholds
    ball_confidence: float = 0.3
    player_confidence: float = 0.4
    
    # Output options
    output_dir: Path | None = None
    export_video: bool = False
    export_json: bool = True
    export_csv: bool = True
```

```python
# pipeline/e2e.py
from typing import List, Optional
from pathlib import Path
from gaffers_guide.core.types import MatchState
from gaffers_guide.pipeline.config import PipelineConfig
from gaffers_guide.pipeline.stages import (
    DetectionStage,
    SpatialMappingStage,
    TacticalAnalysisStage
)

class MatchAnalysisPipeline:
    """
    End-to-end pipeline for match analysis.
    
    Orchestrates: detection → tracking → spatial mapping → tactical analysis.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self._detection_stage: Optional[DetectionStage] = None
        self._spatial_stage: Optional[SpatialMappingStage] = None
        self._tactical_stage: Optional[TacticalAnalysisStage] = None
    
    def process_video(
        self,
        video_path: Path,
        progress_callback: Optional[callable] = None
    ) -> List[MatchState]:
        """
        Process entire video end-to-end.
        
        Args:
            video_path: Path to match video
            progress_callback: Optional callback(frame_idx, total_frames)
            
        Returns:
            List of MatchState objects
        """
        from gaffers_guide.io import VideoReader
        
        # Lazy-initialize stages
        if self._detection_stage is None:
            self._detection_stage = DetectionStage(self.config)
        
        if self.config.enable_spatial_mapping and self._spatial_stage is None:
            self._spatial_stage = SpatialMappingStage(self.config)
        
        if self.config.enable_tactical_analysis and self._tactical_stage is None:
            self._tactical_stage = TacticalAnalysisStage(self.config)
        
        # Process video
        match_states = []
        
        with VideoReader(video_path) as reader:
            for frame_idx, frame in reader:
                # Stage 1: Detection
                frame_detections = self._detection_stage.process(frame, frame_idx)
                
                # Stage 2: Spatial mapping (optional)
                if self._spatial_stage:
                    match_state = self._spatial_stage.process(frame_detections, frame)
                else:
                    match_state = MatchState(
                        frame_idx=frame_idx,
                        timestamp=frame_idx / reader.fps,
                        players=[]  # Populate from detections
                    )
                
                match_states.append(match_state)
                
                if progress_callback:
                    progress_callback(frame_idx, reader.frame_count)
        
        # Stage 3: Tactical analysis (post-processing)
        if self._tactical_stage:
            match_states = self._tactical_stage.process(match_states)
        
        return match_states
```

---

## Lazy Loading Strategy

### Core Pattern: The Three-Layer Guard

```
Layer 1: TYPE_CHECKING imports (type hints only, no runtime cost)
    ↓
Layer 2: Deferred imports in methods (only when actually needed)
    ↓
Layer 3: Global caching (avoid reloading same model)
```

### Implementation Template

```python
# MODULE_NAME/some_class.py
from typing import TYPE_CHECKING, Optional

# Layer 1: Type-checking imports (zero runtime cost)
if TYPE_CHECKING:
    from heavy_library import HeavyModel

class SomeClass:
    """Class that uses heavy dependencies."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        # Layer 2: Deferred loading - model not loaded yet!
        self._model: Optional['HeavyModel'] = None
    
    def some_method(self):
        """Method that actually needs the heavy dependency."""
        # Layer 3: Load on first use + cache
        if self._model is None:
            try:
                from heavy_library import HeavyModel
            except ImportError:
                raise ImportError(
                    "This feature requires heavy_library. "
                    "Install with: pip install 'gaffers-guide[feature]'"
                )
            self._model = HeavyModel(self.model_path)
        
        # Now use self._model
        return self._model.predict(...)
```

### `__init__.py` Strategy

**Bad** (loads everything):
```python
# DON'T DO THIS
from .detectors.ball import BallDetector  # Triggers torch import!
from .detectors.player import PlayerDetector
```

**Good** (lazy exports):
```python
# vision/__init__.py
"""
Computer vision module.

Heavy dependencies (torch, ultralytics) are NOT imported at module level.
"""

# Only import type protocols and base classes (no heavy deps)
from gaffers_guide.vision.detectors.base import BaseDetector

# Use __all__ to control what's exposed
__all__ = ['BaseDetector', 'BallDetector', 'PlayerDetector']

# Lazy imports via __getattr__ (Python 3.7+)
def __getattr__(name: str):
    """Lazy-load detector classes."""
    if name == 'BallDetector':
        from gaffers_guide.vision.detectors.ball import BallDetector
        return BallDetector
    elif name == 'PlayerDetector':
        from gaffers_guide.vision.detectors.player import PlayerDetector
        return PlayerDetector
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### Testing Lazy Loading

```python
# tests/test_lazy_loading.py
import sys
import pytest

def test_io_module_no_torch():
    """Verify io module doesn't import torch."""
    # Import io module
    from gaffers_guide.io import parse_tracking_json
    
    # Verify torch was NOT loaded
    assert 'torch' not in sys.modules
    assert 'ultralytics' not in sys.modules

def test_vision_module_deferred_loading():
    """Verify vision module only loads torch on first detect()."""
    from gaffers_guide.vision import BallDetector
    
    # Instantiation should NOT load torch
    detector = BallDetector(model_path="dummy.pt")
    assert 'torch' not in sys.modules
    
    # First detect() triggers load
    # (This test would need mocking in practice)
```

---

## Developer Usage Patterns

### Pattern 1: Single-Module Usage (Ball Detection)

```python
"""
Use Case: User has pre-extracted frames and only wants ball detection.
Expected memory: ~2GB (torch + YOLO)
"""

from gaffers_guide.vision import BallDetector
import cv2
from pathlib import Path

# Initialize detector (fast - no loading yet)
detector = BallDetector(
    model_path="models/ball_yolov8n.pt",
    device="cuda",
    confidence_threshold=0.35,
    use_sahi=True
)

# Optional: Warm up model (first inference is slow)
detector.warmup()

# Detect ball in single frame
frame = cv2.imread("frame_450.jpg")
detections = detector.detect(frame)

# Access results
if detections.has_ball:
    ball = detections.filter_by_class("ball").detections[0]
    print(f"Ball detected at ({ball.center[0]:.0f}, {ball.center[1]:.0f})")
    print(f"Confidence: {ball.confidence:.2%}")
else:
    print("No ball detected")
```

**Memory profile**:
```
Before detector.detect(): ~50MB (Python + OpenCV)
After detector.detect():  ~2.1GB (torch + YOLO model)
```

---

### Pattern 2: Spatial Analysis Only (Homography)

```python
"""
Use Case: User has detection data, only needs pitch coordinate mapping.
Expected memory: ~50MB (OpenCV only)
"""

from gaffers_guide.spatial import HomographyEngine
from gaffers_guide.core.types import BBoxDetection
import numpy as np

# Initialize homography engine (no ML dependencies)
engine = HomographyEngine(method="ransac")

# Manually annotated pitch corners (or from pitch detector)
pitch_corners_px = np.array([
    [145, 92],      # Top-left
    [1775, 88],     # Top-right
    [1850, 988],    # Bottom-right
    [95, 992]       # Bottom-left
])

# Compute transformation
mapping = engine.fit(
    pitch_corners_px=pitch_corners_px,
    frame_shape=(1080, 1920)
)

# Map player bounding boxes to pitch coordinates
player_bboxes = [
    BBoxDetection(x1=450, y1=300, x2=490, y2=380, confidence=0.95, 
                  class_id=0, class_name="player"),
    BBoxDetection(x1=850, y1=450, x2=890, y2=530, confidence=0.92,
                  class_id=0, class_name="player"),
]

for i, bbox in enumerate(player_bboxes):
    pitch_pos = mapping.pixel_to_pitch(bbox.center)
    print(f"Player {i+1}: {pitch_pos.x:.1f}m x {pitch_pos.y:.1f}m")
```

**Memory profile**:
```
Total memory: ~50MB (no torch, no ultralytics)
```

---

### Pattern 3: Full Pipeline (End-to-End)

```python
"""
Use Case: Complete match analysis from raw video.
Expected memory: ~3GB (all modules loaded)
"""

from gaffers_guide.pipeline import MatchAnalysisPipeline
from gaffers_guide.pipeline.config import PipelineConfig
from pathlib import Path

# Configure pipeline
config = PipelineConfig(
    ball_detector_path=Path("models/ball_yolov8n.pt"),
    player_detector_path=Path("models/player_yolov8m.pt"),
    pitch_detector_path=Path("models/pitch_keypoints.pt"),
    device="cuda",
    enable_tracking=True,
    enable_spatial_mapping=True,
    enable_tactical_analysis=True,
    ball_confidence=0.3,
    player_confidence=0.4,
    output_dir=Path("analysis_output/")
)

# Initialize pipeline
pipeline = MatchAnalysisPipeline(config)

# Process video with progress tracking
def on_progress(frame_idx, total_frames):
    pct = (frame_idx / total_frames) * 100
    print(f"Progress: {pct:.1f}% ({frame_idx}/{total_frames})", end='\r')

match_states = pipeline.process_video(
    video_path=Path("match_broadcast.mp4"),
    progress_callback=on_progress
)

# Export results
from gaffers_guide.io import CSVExporter, JSONExporter

csv_exporter = CSVExporter()
csv_exporter.export(match_states, Path("tracking_data.csv"))

json_exporter = JSONExporter()
json_exporter.export(match_states, Path("tactical_report.json"))

# Compute tactical metrics
from gaffers_guide.tactical import MetricsCalculator

metrics = MetricsCalculator(fps=25.0)

# Player distance analysis
player_distances = {}
unique_tracks = set(p.track_id for state in match_states for p in state.players)

for track_id in unique_tracks:
    distance = metrics.calculate_player_distance(match_states, track_id)
    speed = metrics.calculate_average_speed(match_states, track_id)
    player_distances[track_id] = {
        'distance_km': distance / 1000,
        'avg_speed_kmh': speed
    }

# Possession analysis
possession = metrics.calculate_possession_percentage(match_states)
print(f"\nPossession - Home: {possession['home']:.1f}%, Away: {possession['away']:.1f}%")

# Top runners
top_runners = sorted(
    player_distances.items(),
    key=lambda x: x[1]['distance_km'],
    reverse=True
)[:5]

print("\nTop 5 Distance Covered:")
for track_id, stats in top_runners:
    print(f"  Player {track_id}: {stats['distance_km']:.2f}km @ {stats['avg_speed_kmh']:.1f}km/h")
```

---

### Pattern 4: Custom Detector Integration

```python
"""
Use Case: User implements custom ball detector and plugs into pipeline.
"""

from gaffers_guide.vision.detectors.base import BaseDetector
from gaffers_guide.core.types import FrameDetections, BBoxDetection
import numpy as np

class CustomBallDetector(BaseDetector):
    """Custom ball detector using proprietary algorithm."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        # Load custom model...
    
    def warmup(self):
        """Initialize custom model."""
        pass
    
    def detect(self, frame: np.ndarray) -> FrameDetections:
        """Run custom detection algorithm."""
        # Custom detection logic here
        detections = []
        
        # Example: Dummy detection
        ball_bbox = BBoxDetection(
            x1=100, y1=100, x2=120, y2=120,
            confidence=0.85,
            class_id=0,
            class_name='ball'
        )
        detections.append(ball_bbox)
        
        return FrameDetections(
            frame_idx=-1,
            timestamp=-1.0,
            detections=detections,
            frame_shape=(frame.shape[0], frame.shape[1])
        )

# Use custom detector in pipeline
from gaffers_guide.pipeline import MatchAnalysisPipeline
from gaffers_guide.pipeline.stages import DetectionStage

# Create custom detection stage
custom_detector = CustomBallDetector(config_path="my_config.yaml")

# Inject into pipeline (dependency injection pattern)
class CustomDetectionStage(DetectionStage):
    def __init__(self, config):
        super().__init__(config)
        self.ball_detector = custom_detector  # Override with custom

# Use in pipeline...
```

---

## Refactoring Roadmap

### Phase 1: Foundation (Week 1)
**Goal**: Set up structure without breaking anything

#### Day 1-2: Directory Structure
```bash
# Create new module structure
mkdir -p src/gaffers_guide/{core,vision,spatial,tactical,io,pipeline}
mkdir -p src/gaffers_guide/vision/{detectors,tracking}

# Create __init__.py files
touch src/gaffers_guide/{core,vision,spatial,tactical,io,pipeline}/__init__.py
touch src/gaffers_guide/vision/{detectors,tracking}/__init__.py

# Move existing code to _legacy/ (temporary)
mkdir src/gaffers_guide/_legacy
# Copy current implementation files here
```

**Validation**:
```bash
# Existing tests should still pass against _legacy
pytest tests/ --cov=gaffers_guide._legacy
```

#### Day 3-4: Core Types
- Implement `core/types.py` with all dataclasses
- Implement `core/exceptions.py` with exception hierarchy
- Write unit tests for type validation

**Validation**:
```python
# Test data contracts
from gaffers_guide.core.types import BBoxDetection, PitchCoordinate

# Should validate inputs
bbox = BBoxDetection(x1=10, y1=20, x2=30, y2=40, confidence=0.9, 
                     class_id=0, class_name="ball")
assert bbox.center == (20, 30)

# Should raise on invalid inputs
with pytest.raises(ValueError):
    BBoxDetection(x1=30, y1=20, x2=10, y2=40, ...)  # x2 < x1
```

#### Day 5: Dependency Configuration
- Update `pyproject.toml` with optional dependencies
- Create `requirements.txt` for each module
- Set up `py.typed` marker

**Example `pyproject.toml`**:
```toml
[project]
name = "gaffers-guide"
version = "2.0.0"
dependencies = [
    "numpy>=1.24.0",
    "opencv-python>=4.8.0"
]

[project.optional-dependencies]
vision = [
    "torch>=2.0.0",
    "ultralytics>=8.0.0",
    "sahi>=0.11.0"
]
spatial = []  # Uses base dependencies
tactical = [
    "scikit-learn>=1.3.0"  # For clustering
]
io = []  # Uses base dependencies
dev = [
    "pytest>=7.4.0",
    "mypy>=1.5.0",
    "black>=23.0.0"
]
full = ["gaffers-guide[vision,tactical,dev]"]
```

**Validation**:
```bash
# Install base package (no ML)
pip install -e .
python -c "from gaffers_guide.core import types"  # Should work

# Install with vision
pip install -e ".[vision]"
python -c "from gaffers_guide.vision import BallDetector"  # Should work
```

---

### Phase 2: Module Extraction (Weeks 2-3)
**Goal**: Migrate code module-by-module, least to most dependent

#### Week 2, Day 1-2: IO Module
**Why first?** No dependencies on other modules.

1. Extract video reading logic → `io/video.py`
2. Extract JSON/CSV parsers → `io/parsers.py`
3. Implement exporters → `io/exporters.py`
4. Write tests for each

**Migration checklist**:
- [ ] `VideoReader` class migrated
- [ ] `parse_tracking_json()` migrated
- [ ] `CSVExporter` class migrated
- [ ] Unit tests passing
- [ ] Memory test: importing io doesn't load torch

**Validation**:
```bash
# Test IO module isolation
python -c "
import sys
from gaffers_guide.io import parse_tracking_json
assert 'torch' not in sys.modules, 'IO module leaked torch import'
print('✓ IO module is isolated')
"
```

#### Week 2, Day 3-5: Spatial Module
**Why second?** Only depends on OpenCV (base dependency).

1. Extract homography logic → `spatial/homography.py`
2. Extract pitch geometry → `spatial/geometry.py`
3. Implement projections → `spatial/projections.py`
4. Write comprehensive tests

**Migration checklist**:
- [ ] `HomographyEngine` class migrated
- [ ] Pitch coordinate validation working
- [ ] pixel_to_pitch() accurate (±0.5m)
- [ ] Unit tests + integration tests passing

**Validation**:
```python
# Test homography accuracy
from gaffers_guide.spatial import HomographyEngine
import numpy as np

engine = HomographyEngine()
# Use known pitch corners from test data
corners_px = np.array([[120, 50], [1800, 45], [1900, 1030], [80, 1035]])
mapping = engine.fit(corners_px, frame_shape=(1080, 1920))

# Test center of pitch
center_px = (960, 540)
center_pitch = mapping.pixel_to_pitch(center_px)

# Should be approximately (52.5m, 34m) - center of 105x68 pitch
assert abs(center_pitch.x - 52.5) < 2.0
assert abs(center_pitch.y - 34.0) < 2.0
```

#### Week 3, Day 1-3: Vision Module (CRITICAL)
**Why third?** Heavy dependencies, most complex lazy loading.

1. Implement `_models.py` with lazy loading cache
2. Migrate `BallDetector` → `vision/detectors/ball.py`
3. Migrate `PlayerDetector` → `vision/detectors/player.py`
4. Implement `BaseDetector` protocol → `vision/detectors/base.py`
5. Set up `__getattr__` lazy imports in `vision/__init__.py`

**Migration checklist**:
- [ ] `_models.py` lazy loading working
- [ ] `BallDetector` using lazy pattern
- [ ] `PlayerDetector` using lazy pattern
- [ ] Import time <0.5s (no torch)
- [ ] First detect() triggers load correctly
- [ ] Model caching prevents reload

**Validation**:
```python
import time
import sys

# Test 1: Import speed
start = time.time()
from gaffers_guide.vision import BallDetector
import_time = time.time() - start

assert import_time < 0.5, f"Import too slow: {import_time:.2f}s"
assert 'torch' not in sys.modules, "torch loaded prematurely"

# Test 2: Lazy loading on instantiation
detector = BallDetector(model_path="models/ball.pt", device="cpu")
assert 'torch' not in sys.modules, "torch loaded on __init__"

# Test 3: Loading on first detect (would need mocking in practice)
# detector.detect(dummy_frame)  # NOW torch loads
```

#### Week 3, Day 4-5: Tactical Module
**Why last?** Depends on core types being stable.

1. Migrate metrics calculations → `tactical/metrics.py`
2. Migrate team classification → `tactical/classification.py`
3. Write property-based tests (hypothesis)

**Migration checklist**:
- [ ] `MetricsCalculator` migrated
- [ ] Distance calculations accurate
- [ ] Possession percentage correct
- [ ] No heavy dependencies required

---

### Phase 3: Pipeline Integration (Week 4)
**Goal**: Wire modules together through composable stages

#### Day 1-2: Pipeline Stages
Implement modular processing stages:

```python
# pipeline/stages.py
from gaffers_guide.core.types import FrameDetections, MatchState

class DetectionStage:
    """Stage 1: Detect objects in frame."""
    
    def __init__(self, config):
        from gaffers_guide.vision import BallDetector, PlayerDetector
        
        self.ball_detector = BallDetector(
            model_path=config.ball_detector_path,
            device=config.device,
            confidence_threshold=config.ball_confidence
        )
        self.player_detector = PlayerDetector(
            model_path=config.player_detector_path,
            device=config.device,
            confidence_threshold=config.player_confidence
        )
    
    def process(self, frame, frame_idx) -> FrameDetections:
        """Run detection on frame."""
        # Detect ball
        ball_detections = self.ball_detector.detect(frame)
        
        # Detect players
        player_detections = self.player_detector.detect(frame)
        
        # Combine detections
        all_detections = FrameDetections(
            frame_idx=frame_idx,
            timestamp=-1.0,  # Set by caller
            detections=ball_detections.detections + player_detections.detections,
            frame_shape=frame.shape[:2]
        )
        
        return all_detections

class SpatialMappingStage:
    """Stage 2: Map detections to pitch coordinates."""
    
    def __init__(self, config):
        from gaffers_guide.spatial import HomographyEngine
        
        self.engine = HomographyEngine()
        self.mapping = None
    
    def process(self, detections: FrameDetections, frame) -> MatchState:
        """Map frame detections to pitch coordinates."""
        # If no mapping yet, compute it (from pitch detection or manual)
        if self.mapping is None:
            self.mapping = self._compute_mapping(frame)
        
        # Convert detections to PlayerState objects
        players = []
        for det in detections.filter_by_class("player").detections:
            pitch_pos = self.mapping.pixel_to_pitch(det.center)
            player = PlayerState(
                track_id=det.track_id or -1,
                position_px=det.center,
                position_pitch=pitch_pos,
                bbox=det
            )
            players.append(player)
        
        # Find ball position
        ball_detections = detections.filter_by_class("ball")
        ball_px = ball_detections.detections[0].center if ball_detections.has_ball else None
        ball_pitch = self.mapping.pixel_to_pitch(ball_px) if ball_px else None
        
        return MatchState(
            frame_idx=detections.frame_idx,
            timestamp=detections.timestamp,
            players=players,
            ball_position_px=ball_px,
            ball_position_pitch=ball_pitch,
            spatial_mapping=self.mapping
        )
```

#### Day 3-4: End-to-End Pipeline
Implement `MatchAnalysisPipeline` that composes stages.

**Migration checklist**:
- [ ] Pipeline can process video end-to-end
- [ ] Progress callbacks working
- [ ] Memory usage reasonable (<4GB)
- [ ] Output matches legacy implementation

#### Day 5: Backward Compatibility
Add compatibility layer for old imports:

```python
# gaffers_guide/__init__.py
"""
Gaffers Guide SDK - Modular Football Analysis
"""

__version__ = "2.0.0"

# New preferred imports
from gaffers_guide.pipeline import MatchAnalysisPipeline
from gaffers_guide.pipeline.config import PipelineConfig

# Legacy compatibility (deprecated)
def run_analysis(*args, **kwargs):
    """
    DEPRECATED: Use MatchAnalysisPipeline instead.
    
    This function is kept for backward compatibility but will be
    removed in v3.0.0. Please migrate to:
    
        from gaffers_guide.pipeline import MatchAnalysisPipeline
        pipeline = MatchAnalysisPipeline(config)
        pipeline.process_video(video_path)
    """
    import warnings
    warnings.warn(
        "run_analysis() is deprecated. Use MatchAnalysisPipeline instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Delegate to new implementation
    # ... conversion logic ...
```

---

### Phase 4: CLI & Frontend Adaptation (Week 5)
**Goal**: Update all entry points

#### Day 1-2: CLI Refactor
Update `cli/main.py` to use new pipeline:

```python
# cli/main.py
import click
from pathlib import Path
from gaffers_guide.pipeline import MatchAnalysisPipeline
from gaffers_guide.pipeline.config import PipelineConfig

@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--ball-model', required=True, type=click.Path(exists=True))
@click.option('--player-model', required=True, type=click.Path(exists=True))
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda', 'mps']))
@click.option('--output-dir', default='output/', type=click.Path())
def analyze(video_path, ball_model, player_model, device, output_dir):
    """Analyze football match video."""
    
    config = PipelineConfig(
        ball_detector_path=Path(ball_model),
        player_detector_path=Path(player_model),
        device=device,
        output_dir=Path(output_dir)
    )
    
    pipeline = MatchAnalysisPipeline(config)
    
    click.echo(f"Processing {video_path}...")
    
    match_states = pipeline.process_video(
        video_path=Path(video_path),
        progress_callback=lambda idx, total: click.echo(
            f"Frame {idx}/{total}", nl=False, err=True
        )
    )
    
    click.echo(f"\n✓ Processed {len(match_states)} frames")

if __name__ == '__main__':
    analyze()
```

**Validation**:
```bash
# Test CLI still works
gaffers-guide analyze match.mp4 \
    --ball-model models/ball.pt \
    --player-model models/player.pt \
    --device cuda \
    --output-dir output/
```

#### Day 3-4: Electron App Update
Update Electron app imports:

```javascript
// electron/main.js (Python bridge)
const { spawn } = require('child_process');

function runAnalysis(videoPath, config) {
    // Instead of calling legacy run_analysis
    // Now uses: python -m gaffers_guide.cli analyze ...
    
    const python = spawn('python', [
        '-m', 'gaffers_guide.cli',
        'analyze',
        videoPath,
        '--ball-model', config.ballModel,
        '--player-model', config.playerModel,
        '--device', config.device
    ]);
    
    python.stdout.on('data', (data) => {
        // Send progress to renderer
        mainWindow.webContents.send('analysis-progress', data.toString());
    });
}
```

#### Day 5: Documentation Update
- Update README.md with new import examples
- Create MIGRATION.md guide for existing users
- Update API documentation

---

### Phase 5: Polish & Release (Week 6)
**Goal**: Production-ready SDK

#### Day 1: Type Checking
```bash
# Full mypy strict check
mypy src/gaffers_guide --strict --show-error-codes

# Fix all type errors
# Target: 0 errors
```

#### Day 2-3: Testing
```bash
# Run full test suite
pytest tests/ -v --cov=gaffers_guide --cov-report=html

# Test coverage targets:
# - core: 100%
# - io: 95%
# - spatial: 90%
# - vision: 85% (hard to test without models)
# - tactical: 90%
# - pipeline: 80%
```

**Integration tests**:
```python
# tests/integration/test_e2e_pipeline.py
def test_full_pipeline_matches_legacy():
    """Verify new pipeline matches legacy output."""
    
    # Run legacy implementation
    legacy_output = run_legacy_analysis("test_video.mp4")
    
    # Run new pipeline
    config = PipelineConfig(...)
    pipeline = MatchAnalysisPipeline(config)
    new_output = pipeline.process_video("test_video.mp4")
    
    # Outputs should match within tolerance
    assert_outputs_equivalent(legacy_output, new_output, tolerance=0.05)
```

#### Day 4: Documentation
- [ ] API reference docs (Sphinx)
- [ ] Usage examples for each module
- [ ] Architecture diagram
- [ ] Migration guide from v1.x

#### Day 5: Release Prep
```bash
# Build distributions
python -m build

# Test install in clean environment
python -m venv test_env
source test_env/bin/activate
pip install dist/gaffers_guide-2.0.0-py3-none-any.whl

# Verify imports work
python -c "from gaffers_guide.io import parse_tracking_json"
python -c "from gaffers_guide.vision import BallDetector"
python -c "from gaffers_guide.pipeline import MatchAnalysisPipeline"

# Upload to PyPI
twine upload dist/*
```

---

## Testing Strategy

### Unit Tests (Per-Module)

```python
# tests/core/test_types.py
import pytest
from gaffers_guide.core.types import BBoxDetection, PitchCoordinate

def test_bbox_validation():
    """BBoxDetection should validate coordinates."""
    
    # Valid bbox
    bbox = BBoxDetection(
        x1=10, y1=20, x2=30, y2=40,
        confidence=0.9, class_id=0, class_name="ball"
    )
    assert bbox.center == (20, 30)
    
    # Invalid bbox (x2 < x1)
    with pytest.raises(ValueError, match="Invalid bbox"):
        BBoxDetection(
            x1=30, y1=20, x2=10, y2=40,
            confidence=0.9, class_id=0, class_name="ball"
        )

def test_pitch_coordinate_validation():
    """PitchCoordinate should enforce pitch boundaries."""
    
    # Valid coordinate
    coord = PitchCoordinate(x=52.5, y=34.0)
    assert coord.x == 52.5
    
    # Out of bounds
    with pytest.raises(ValueError, match="Pitch x must be 0-105m"):
        PitchCoordinate(x=120, y=34)
```

### Integration Tests

```python
# tests/integration/test_spatial_pipeline.py
from gaffers_guide.spatial import HomographyEngine
from gaffers_guide.io import VideoReader
import numpy as np

def test_homography_on_real_video():
    """Test homography accuracy on real broadcast footage."""
    
    # Load test video with known pitch corners
    test_data = load_test_dataset("broadcast_01")
    
    engine = HomographyEngine()
    mapping = engine.fit(
        pitch_corners_px=test_data['pitch_corners'],
        frame_shape=test_data['frame_shape']
    )
    
    # Test known player positions
    for player_px, expected_pitch in test_data['ground_truth']:
        predicted_pitch = mapping.pixel_to_pitch(player_px)
        
        # Should be within 1 meter
        error = predicted_pitch.distance_to(expected_pitch)
        assert error < 1.0, f"Homography error {error:.2f}m too high"
```

### Performance Tests

```python
# tests/performance/test_lazy_loading.py
import time
import sys

def test_import_speed():
    """Verify module imports are fast (no heavy loading)."""
    
    start = time.time()
    from gaffers_guide.vision import BallDetector
    import_time = time.time() - start
    
    assert import_time < 0.5, f"Import took {import_time:.2f}s (too slow)"
    assert 'torch' not in sys.modules, "torch loaded prematurely"

def test_detection_throughput():
    """Verify detection FPS meets targets."""
    
    from gaffers_guide.vision import BallDetector
    import numpy as np
    
    detector = BallDetector(
        model_path="models/ball_yolov8n.pt",
        device="cuda"
    )
    detector.warmup()
    
    # Generate dummy frames
    frames = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) 
              for _ in range(100)]
    
    start = time.time()
    for frame in frames:
        _ = detector.detect(frame)
    duration = time.time() - start
    
    fps = len(frames) / duration
    assert fps >= 25.0, f"Detection too slow: {fps:.1f} FPS"
```

---

## Migration Guide

### For Existing Users (v1.x → v2.0)

#### Breaking Changes

1. **Import paths changed**:
```python
# OLD (v1.x)
from gaffers_guide import run_analysis

# NEW (v2.0)
from gaffers_guide.pipeline import MatchAnalysisPipeline
from gaffers_guide.pipeline.config import PipelineConfig
```

2. **Configuration now uses dataclass**:
```python
# OLD
run_analysis(
    video_path="match.mp4",
    ball_model="ball.pt",
    player_model="player.pt",
    device="cuda"
)

# NEW
config = PipelineConfig(
    ball_detector_path=Path("ball.pt"),
    player_detector_path=Path("player.pt"),
    device="cuda"
)
pipeline = MatchAnalysisPipeline(config)
pipeline.process_video(Path("match.mp4"))
```

3. **Return types are now dataclasses**:
```python
# OLD: Returns dict
results = run_analysis(...)
ball_pos = results['frames'][0]['ball']['position']

# NEW: Returns List[MatchState]
states = pipeline.process_video(...)
ball_pos = states[0].ball_position_pitch
```

#### Migration Steps

**Step 1**: Install v2.0 with backward compatibility
```bash
pip install --upgrade gaffers-guide
```

**Step 2**: Test existing code (should still work with deprecation warnings)
```python
# Your old code will work but show warnings
from gaffers_guide import run_analysis  # DeprecationWarning
```

**Step 3**: Update imports progressively
```python
# Recommended: Update one module at a time
from gaffers_guide.pipeline import MatchAnalysisPipeline

# You can mix old and new during migration
```

**Step 4**: Update configurations
```python
# Create PipelineConfig from old dict-based config
old_config = {'ball_model': 'ball.pt', 'device': 'cuda', ...}

new_config = PipelineConfig(
    ball_detector_path=Path(old_config['ball_model']),
    device=old_config['device'],
    ...
)
```

**Step 5**: Update result processing
```python
# OLD: Dict access
for frame in results['frames']:
    ball_x = frame['ball']['x']

# NEW: Dataclass access
for state in match_states:
    ball_x = state.ball_position_pitch.x if state.ball_position_pitch else None
```

---

## Success Criteria

### Functional Requirements
- [ ] CLI still works identically to v1.x
- [ ] Electron app runs without changes (or minimal changes)
- [ ] Output JSON format matches v1.x (for frontend compatibility)
- [ ] All v1.x test cases pass with v2.0 implementation

### Non-Functional Requirements

#### Performance
- [ ] `from gaffers_guide.io import parse_json` completes in <0.5s
- [ ] First detection after model load: <2s (GPU)
- [ ] Subsequent detections: ≥25 FPS (GPU)
- [ ] Memory usage: <4GB for full pipeline (vs ~6GB in v1.x)

#### Code Quality
- [ ] `mypy --strict` passes with 0 errors
- [ ] Test coverage ≥85% overall
- [ ] No circular dependencies (`import-linter`)
- [ ] All public APIs have docstrings

#### Developer Experience
- [ ] Can install `gaffers-guide[io]` for lightweight use (<100MB)
- [ ] Clear error messages when optional deps missing
- [ ] Sphinx docs generated and hosted
- [ ] Migration guide complete with examples

#### Modularity
- [ ] Each module has <3 dependencies
- [ ] `vision` module can be excluded (for CPU-only servers)
- [ ] Custom detectors can be implemented via `BaseDetector`
- [ ] Data contracts (core types) never change without major version bump

---

## Appendix: Quick Reference

### Import Cheat Sheet

```python
# Core data structures (always available)
from gaffers_guide.core.types import (
    BBoxDetection,
    FrameDetections,
    PitchCoordinate,
    SpatialMapping,
    PlayerState,
    MatchState
)

# Vision module (requires: pip install gaffers-guide[vision])
from gaffers_guide.vision import BallDetector, PlayerDetector

# Spatial module (no extra deps)
from gaffers_guide.spatial import HomographyEngine

# Tactical module (requires: pip install gaffers-guide[tactical])
from gaffers_guide.tactical import MetricsCalculator

# IO module (no extra deps)
from gaffers_guide.io import VideoReader, parse_tracking_json, CSVExporter

# Pipeline (high-level)
from gaffers_guide.pipeline import MatchAnalysisPipeline
from gaffers_guide.pipeline.config import PipelineConfig
```

### Dependency Matrix

| Module | Required | Optional |
|--------|----------|----------|
| `core` | `numpy` | - |
| `io` | `opencv-python` | - |
| `spatial` | `opencv-python` | - |
| `vision` | `numpy`, `opencv-python` | `torch`, `ultralytics`, `sahi` |
| `tactical` | `numpy` | `scikit-learn` |
| `pipeline` | All above | - |

### Memory Footprint Guide

| Import | Memory Usage | Load Time |
|--------|--------------|-----------|
| `from gaffers_guide.io import parse_json` | ~10MB | <0.1s |
| `from gaffers_guide.spatial import HomographyEngine` | ~50MB | <0.2s |
| `from gaffers_guide.vision import BallDetector` (no detect) | ~50MB | <0.5s |
| `BallDetector.detect()` first call | ~2.1GB | ~2s |
| Full pipeline (all modules loaded) | ~3.5GB | ~5s |

---

**End of SDK Modularity Blueprint**

*For questions or clarifications, refer to the GitHub discussions or SDK documentation.*