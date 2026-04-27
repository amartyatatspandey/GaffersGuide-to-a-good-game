# Gaffer's Guide to a Good Game

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO-FF6B35?style=for-the-badge&logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**An end-to-end automated computer vision pipeline for football tactical intelligence.**  
Turn raw match footage into structured tracking data, tactical metrics, and annotated video — fully automated.

[Getting Started](#7-installation) • [Usage](#9-usage-cli) • [Quality Profiles](#6-quality-profiles) • [Tech Stack](#12-tech-stack) • [Contributing](#14-contributing)

</div>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Pipeline Flow](#4-pipeline-flow)
5. [Key Features](#5-key-features)
6. [Quality Profiles](#6-quality-profiles)
7. [Installation](#7-installation)
8. [Environment Setup](#8-environment-setup)
9. [Usage (CLI)](#9-usage-cli)
10. [Output](#10-output)
11. [Performance Notes](#11-performance-notes)
12. [Tech Stack](#12-tech-stack)
13. [Project Structure](#13-project-structure)
14. [Contributing](#14-contributing)
15. [Future Improvements](#15-future-improvements)

---

## 1. Overview

**Gaffer's Guide** is a production-grade automated computer vision pipeline and tactical intelligence platform built for football (soccer) video analysis.

The system ingests raw broadcast or single-camera match footage and outputs:
- Frame-by-frame player and ball tracking
- Tactical metrics including formations, zones, and heatmaps
- Annotated video overlays
- Structured JSON telemetry

All fully automated — no manual annotation required.

---

## 2. Problem Statement

Analyzing football matches manually is:
- Extremely time-consuming — hours of review per match
- Error-prone — human fatigue affects consistency
- Expensive — requires dedicated analysis staff

Coaches and analysts need precise, actionable insights — player positioning, team formations, movement heatmaps — from standard footage.

**Gaffer's Guide bridges the gap between raw video and structured tactical intelligence through a fully automated pipeline.**

---

## 3. System Architecture

```
+------------------------------------------------------------------+
|                        GAFFER'S GUIDE                            |
|                                                                  |
|  +----------+    +----------+    +----------+    +----------+   |
|  |  INPUT   |--->|  DETECT  |--->|  TRACK   |--->| ANALYSE  |   |
|  |  Video   |    |  YOLO    |    |ByteTrack |    | Tactics  |   |
|  |  MP4/AVI |    |  + SAHI  |    |  + IDs   |    | Metrics  |   |
|  +----------+    +----------+    +----------+    +----------+   |
|                                                       |          |
|  +---------------------------------------------------+|          |
|  |                    OUTPUT                         ||          |
|  |  Annotated Video   JSON Metrics   Tracking Data  ||<---------+
|  +---------------------------------------------------+          |
|                                                                  |
|  +---------------------------------------------------+          |
|  |           QUALITY PROFILE SYSTEM                 |          |
|  |   fast | balanced | high_res | sahi              |          |
|  +---------------------------------------------------+          |
+------------------------------------------------------------------+
```

**Components:**
- **Input Layer** — CLI-driven ingestion of MP4/AVI footage
- **Detection Layer** — YOLO-based player, referee, and ball detection with optional SAHI tiling
- **Tracking Layer** — ByteTrack multi-object tracking with persistent IDs
- **Analytics Layer** — Spatial coordinate transformation, formation detection, zone calculation
- **Output Layer** — JSON telemetry export and rendered video overlay

---

## 4. Pipeline Flow

```
User runs CLI
      |
      v
Quality Profile resolved (fast / balanced / high_res / sahi)
      |
      v
ProfileConfig → imgsz, conf_threshold, sahi_enabled, slice_size
      |
      v
Video frames extracted and batched
      |
      v
YOLO model inference (with profile imgsz + conf)
      |
      +---> [SAHI enabled?] → Slice region → Batch infer slices → Fuse candidates
      |
      v
Ball candidate ranking (temporal prior + confidence scoring)
      |
      v
Pitch ROI masking → Homography projection → 2D radar coordinates
      |
      v
Team classification → Formation detection → Tactical metric calculation
      |
      v
Output: JSON tracking + JSON metrics + annotated MP4
```

---

## 5. Key Features

| Feature | Description |
|---|---|
| SAHI Ball Detection | Slicing Aided Hyper Inference for precise small-object detection |
| Quality Profile System | 4 runtime profiles trading speed vs. accuracy |
| Temporal Ball Prior | Adaptive search windows based on last known ball position |
| Pitch ROI Masking | HSV-based green detection to focus inference on the pitch |
| ByteTrack Integration | Robust multi-object tracking with persistent player IDs |
| Homography Projection | Camera-to-2D radar coordinate transformation |
| CLI-First Design | Headless operation for automation and cloud deployment |
| Cloud-Native Ready | Docker and cloud infrastructure for scalable batch processing |
| Fully Tested | Profile resolution, CLI parsing, and pipeline integration tests |

---

## 6. Quality Profiles

The profile system is the **core innovation** of this release. One flag controls the entire runtime behavior.

```bash
--quality-profile fast        # Speed priority
--quality-profile balanced    # Default — best all-rounder
--quality-profile high_res    # Quality priority
--quality-profile sahi        # Maximum recall
```

| Profile | imgsz | conf | SAHI | Frame Skip | Best For |
|---|---|---|---|---|---|
| fast | 480 | 0.35 | No | Every 3rd frame | Quick previews, large batch jobs |
| balanced | 640 | 0.25 | No | None | Standard full-match analysis |
| high_res | 1280 | 0.20 | No | None | High-detail QA and final renders |
| sahi | 1280 | 0.20 | Yes | None | Complex scenes, maximum ball recall |

> **How profiles work:** Each profile maps directly to `imgsz` and `conf` passed into the YOLO model inference call, plus SAHI slice configuration for the detection wrapper. The entire pipeline behavior changes from a single CLI flag.

## 7. Installation

### Prerequisites
- Python 3.11+
- Git
- CUDA-enabled GPU (recommended for `high_res` and `sahi` profiles)

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/amartyatatspandey/GaffersGuide-to-a-good-game.git
cd GaffersGuide-to-a-good-game

# Create and activate virtual environment
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Install package in editable mode
pip install -e . --no-deps
```

---

## 8. Environment Setup

Configure the Python path so internal modules resolve correctly:

```bash
# macOS / Linux — add to ~/.bashrc or ~/.zshrc
export PYTHONPATH=backend:src

# Windows — add to system environment variables
set PYTHONPATH=backend;src
```

Or create a `.env` file in the project root:
```
PYTHONPATH=backend:src
```

---

## 9. Usage (CLI)

### Basic Run

```bash
python -m gaffers_guide.cli run \
  --video "path/to/match.mp4" \
  --output "path/to/output/" \
  --quality-profile balanced
```

### Profile-Specific Examples

```bash
# Quick preview (fastest)
python -m gaffers_guide.cli run --video match.mp4 --output out/ --quality-profile fast

# Standard analysis (recommended)
python -m gaffers_guide.cli run --video match.mp4 --output out/ --quality-profile balanced

# High quality render
python -m gaffers_guide.cli run --video match.mp4 --output out/ --quality-profile high_res

# Maximum ball recall (slowest, most accurate)
python -m gaffers_guide.cli run --video match.mp4 --output out/ --quality-profile sahi
```

### List All Profiles

```bash
python -m gaffers_guide.cli profiles list
```

### CLI Arguments

| Argument | Required | Description |
|---|---|---|
| --video | Yes | Path to input video file (MP4/AVI) |
| --output | Yes | Directory for output files |
| --quality-profile | Yes | One of: fast, balanced, high_res, sahi |
| --precision | Deprecated | Use --quality-profile instead |

---

## 10. Output

The pipeline generates three output artifacts:

```
output/
├── match_tracking_data.json      # Frame-by-frame tracking coordinates
├── match_tactical_metrics.json   # Formations, zones, events
└── match_annotated.mp4           # Video with bounding boxes and overlays
```

**tracking_data.json** — Per-frame spatial data:
```json
{
  "frame_idx": 42,
  "players": [{"id": 1, "team": 0, "bbox": [x1, y1, x2, y2], "radar_xy": [rx, ry]}],
  "ball": {"bbox": [x1, y1, x2, y2], "radar_xy": [rx, ry], "confidence": 0.87}
}
```

**tactical_metrics.json** — Match analytics:
```json
{
  "formation": "4-3-3",
  "possession": {"team_0": 54.2, "team_1": 45.8},
  "zones": {},
  "events": []
}
```

---

## 11. Performance Notes

### Profile Selection Guide

| Use Case | Recommended Profile |
|---|---|
| Quick sanity check | fast |
| Standard match analysis | balanced |
| Final deliverable render | high_res |
| Ball tracking failing | sahi |
| Large batch processing | fast |
| QA and debugging | high_res |

### Hardware Recommendations

| Profile | Minimum | Recommended |
|---|---|---|
| fast | CPU or any GPU | Any GPU |
| balanced | 4GB VRAM | 8GB VRAM |
| high_res | 8GB VRAM | 16GB VRAM |
| sahi | 8GB VRAM | 16GB+ VRAM or Cloud GPU |

---

## 12. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.11+ | Core pipeline orchestration |
| Detection | Ultralytics YOLO | Player, ball, referee detection |
| Tiling | SAHI | Small object detection via slicing |
| Tracking | Supervision / ByteTrack | Multi-object tracking with IDs |
| Vision | OpenCV | Frame extraction, rendering, HSV masking |
| Data | Pandas, NumPy | Metric calculation and data manipulation |
| Clustering | Scikit-learn | Team classification and formation detection |
| CLI | Argparse | Command line interface |
| Packaging | Setuptools (PEP-621) | pip install-ready package |
| Testing | Pytest | Profile, CLI, and pipeline tests |
| Infrastructure | Docker | Containerized cloud deployment |

---

## 13. Project Structure

```
GaffersGuide-to-a-good-game/
|
+-- src/
|   +-- gaffers_guide/
|       +-- __init__.py          # Package entry
|       +-- cli.py               # CLI interface
|       +-- profiles.py          # Quality profile definitions
|
+-- backend/
|   +-- scripts/
|   |   +-- pipeline_core/
|   |       +-- run_e2e_cloud.py # Main pipeline orchestrator
|   +-- services/
|       +-- cv/
|           +-- optimized_sahi_wrapper.py   # SAHI ball detection
|           +-- slice_batch_inferencer.py   # Batched slice inference
|           +-- temporal_ball_prior.py      # Adaptive search windows
|           +-- pitch_roi_provider.py       # Pitch masking
|           +-- ball_candidate_fuser.py     # Candidate ranking
|           +-- pipeline_runner.py          # Profile to SAHI config mapper
|
+-- frontend_final/              # Electron desktop application
+-- tests/                       # Test suite
+-- docs/                        # Documentation
+-- pyproject.toml               # PEP-621 package config
+-- requirements.txt             # Dependencies
+-- README.md
```

---

## 14. Contributing

### Branch Strategy

```
main (protected)
+-- feature/packaging-quality-profiles (base feature branch)
    +-- feature/profile-integration    (Person 1)
    +-- feature/cli-improvements       (Person 2)
```

### Workflow

```bash
# 1. Always branch from feature/packaging-quality-profiles
git checkout feature/packaging-quality-profiles
git pull
git checkout -b feature/your-feature-name

# 2. Make changes and commit with descriptive messages
git add .
git commit -m "feat(scope): description of change"

# 3. Push and open PR targeting feature/packaging-quality-profiles
git push origin feature/your-feature-name
```

### Commit Convention

| Prefix | Use For |
|---|---|
| feat | New feature |
| fix | Bug fix |
| chore | Maintenance |
| docs | Documentation |
| test | Tests |
| merge | Merge conflict resolution |

---

<div align="center">

**Built with care by the Gaffer's Guide Team**

[Star this repo](https://github.com/amartyatatspandey/GaffersGuide-to-a-good-game) • [Report a Bug](https://github.com/amartyatatspandey/GaffersGuide-to-a-good-game/issues) • [Request a Feature](https://github.com/amartyatatspandey/GaffersGuide-to-a-good-game/issues)

</div>
