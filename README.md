# Gaffer's Guide

A computer vision pipeline for automated video analysis, player tracking, and tactical metrics extraction. Designed for sports footage processing with configurable quality profiles to balance speed and accuracy.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [CLI Usage](#cli-usage)
- [Quality Profiles](#quality-profiles)
- [Example](#example)
- [Output](#output)
- [Performance Notes](#performance-notes)

---

## Overview

Gaffer's Guide processes sports video footage through an end-to-end cloud pipeline to produce:
- Player tracking data
- Tactical metrics
- Annotated output video

The system exposes a CLI (`gaffers-guide`) with configurable quality profiles, allowing you to trade off inference speed against detection accuracy depending on your use case.

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/amartyatatspandey/GaffersGuide-to-a-good-game.git
cd GaffersGuide-to-a-good-game
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r backend/requirements.txt
```

---

## Environment Setup

The project uses two source roots. Set `PYTHONPATH` before running any command:

```bash
export PYTHONPATH=backend:src      # macOS/Linux
set PYTHONPATH=backend;src        # Windows
```

> **Tip:** Add this to your `.env` or shell profile to avoid setting it each session.

---

## CLI Usage

### Run the pipeline

```bash
python -m gaffers_guide.cli run \
  --video "<path_to_video>" \
  --output out \
  --quality-profile <profile>
```

| Flag | Required | Description |
|------|----------|-------------|
| `--video` | ✅ | Path to the input video file |
| `--output` | ✅ | Output directory name |
| `--quality-profile` | ✅ | One of: `fast`, `balanced`, `high_res`, `sahi` |

---

### List available profiles

```bash
python -m gaffers_guide.cli profiles list
```

Prints all available quality profiles with their descriptions.

---

## Quality Profiles

| Profile | Speed | Accuracy | Best For |
|---------|-------|----------|----------|
| `fast` | ⚡⚡⚡ | ★★☆ | Quick previews, large batch jobs |
| `balanced` | ⚡⚡ | ★★★ | General-purpose processing |
| `high_res` | ⚡ | ★★★★ | High-detail analysis, final output |
| `sahi` | ⚡ | ★★★★★ | Crowded scenes, small object detection |

- **fast** — Optimised for throughput. Reduced resolution and simplified detection. Suitable when turnaround time matters more than precision.
- **balanced** — Default recommended profile. Good accuracy at a reasonable processing speed.
- **high_res** — Full-resolution inference. Best for final deliverables or tactical review where detail matters.
- **sahi** — Uses Slicing Aided Hyper Inference (SAHI) for detecting small or densely packed objects. Highest accuracy, slowest runtime.

---

## Example

```bash
PYTHONPATH=backend:src python -m gaffers_guide.cli run \
  --video "footage/match_01.mp4" \
  --output backend/output \
  --quality-profile balanced
```

---

## Output

All outputs are written to `backend/output/`:

| File | Description |
|------|-------------|
| `*_tracking_data.json` | Frame-by-frame player position and tracking data |
| `*_tactical_metrics.json` | Derived tactical metrics (formations, zones, events) |
| `*.mp4` | Annotated video with overlaid tracking visualisation |

---

## Performance Notes

- **SAHI** significantly increases processing time due to tiled inference — use only when small-object detection is critical.
- **high_res** requires more VRAM; ensure your environment has adequate GPU memory before use.
- For batch processing of multiple videos, `fast` profile is recommended to reduce queue time.
- All profiles use the same underlying pipeline (`scripts.pipeline_core.run_e2e_cloud`) — only inference parameters differ.
