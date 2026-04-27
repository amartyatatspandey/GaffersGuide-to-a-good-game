# Gaffer's Guide

> A computer vision pipeline for automated video analysis, player tracking, and tactical metrics extraction. Designed for sports footage processing with configurable quality profiles to balance speed and accuracy.

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

- **Player tracking data** — frame-by-frame position and movement
- **Tactical metrics** — formations, zones, and key events
- **Annotated output video** — overlaid visualisations for review

The system exposes a CLI (`gaffers-guide`) with configurable quality profiles, letting you trade off inference speed against detection accuracy depending on your use case.

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
  --video "" \
  --output  \
  --quality-profile 
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--video` | ✅ | — | Path to the input video file |
| `--output` | ✅ | — | Output directory name |
| `--quality-profile` | ❌ | `balanced` | One of: `fast`, `balanced`, `high_res`, `sahi` |

> **Note:** `--quality-profile` is optional. If omitted, the pipeline defaults to `balanced`.
>
> **Deprecated:** `--precision` still works but is hidden and will be removed in a future release. Use `--quality-profile` instead.

### List available profiles

```bash
python -m gaffers_guide.cli profiles list
```

Prints a formatted table of all available profiles and their key parameters:
Available quality profiles (default: balanced)
Profile      SAHI   imgsz  Conf   Description
fast           False  640    0.25   Quick preview, reduced accuracy
balanced  *    False  1280   0.30   General-purpose processing
high_res       False  1920   0.35   Full-resolution, high detail
sahi           True   1280   0.30   Tiled inference for crowded scenes

= default profile


---

## Quality Profiles

| Profile | Speed | Accuracy | SAHI | Best For |
|---|---|---|---|---|
| `fast` | ⚡⚡⚡ | ★★☆ | ❌ | Quick previews, large batch jobs |
| `balanced` | ⚡⚡ | ★★★ | ❌ | General-purpose processing *(default)* |
| `high_res` | ⚡ | ★★★★ | ❌ | High-detail analysis, final output |
| `sahi` | ⚡ | ★★★★★ | ✅ | Crowded scenes, small object detection |

- **`fast`** — Optimised for throughput. Reduced resolution and simplified detection. Use when turnaround time matters more than precision.
- **`balanced`** — Default recommended profile. Good accuracy at a reasonable processing speed.
- **`high_res`** — Full-resolution inference. Best for final deliverables or tactical review where detail matters.
- **`sahi`** — Uses [Slicing Aided Hyper Inference (SAHI)](https://github.com/obss/sahi) for detecting small or densely packed objects. Highest accuracy, slowest runtime.

Profiles are defined in `gaffers_guide/profiles.py` and expose the following parameters:

| Parameter | Description |
|---|---|
| `sahi_enabled` | Whether tiled SAHI inference is active |
| `imgsz` | Inference image resolution |
| `confidence_threshold` | Minimum detection confidence score |
| `slice_width` / `slice_height` | SAHI tile dimensions *(SAHI only)* |
| `slice_overlap_ratio` | Overlap between SAHI tiles *(SAHI only)* |

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

All outputs are written to the specified `--output` directory:

| File | Description |
|---|---|
| `*_tracking_data.json` | Frame-by-frame player position and tracking data |
| `*_tactical_metrics.json` | Derived tactical metrics (formations, zones, events) |
| `*.mp4` | Annotated video with overlaid tracking visualisation |

---

## Performance Notes

- **SAHI** significantly increases processing time due to tiled inference — use only when small-object detection is critical.
- **`high_res`** requires more VRAM; ensure your environment has adequate GPU memory before use.
- For batch processing of multiple videos, the **`fast`** profile is recommended to reduce queue time.
- All profiles use the same underlying pipeline (`scripts.pipeline_core.run_e2e_cloud`) — only inference parameters differ.
