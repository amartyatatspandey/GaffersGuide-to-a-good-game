# Gaffers Guide

<!-- Badges -->
![PyPI Version](https://img.shields.io/pypi/v/gaffers-guide)
![Python Versions](https://img.shields.io/pypi/pyversions/gaffers-guide)
![License](https://img.shields.io/github/license/your-org/gaffers-guide)
![GitHub Stars](https://img.shields.io/github/stars/your-org/gaffers-guide?style=social)

A highly optimized, open-source computer vision pipeline for football tactical analysis. Track the ball, map player positions to a tactical radar, and generate coaching reports using YOLO, SAHI, and custom homography — all running locally.

---

## Two Ways to Use It

### For Coaches and Analysts — The Desktop App

Download the desktop app (built on React and Electron). On first launch, the app will detect whether the `gaffers-guide` engine is installed and walk you through setup automatically. No terminal required.

Once the engine is installed, point the app at a match video and select a quality profile. The app streams live pipeline output to a built-in log console and writes the final tactical report to disk.

> The desktop app is a thin, secure wrapper. It does not bundle the CV engine. It spawns the locally installed PyPI engine as a subprocess and streams its output over a safe IPC bridge.

---

### For Developers and Data Scientists — The Python SDK and CLI

Install the base package (no ML dependencies):

```bash
pip install gaffers-guide
```

Install the full vision stack (YOLO + SAHI + PyTorch):

```bash
pip install "gaffers-guide[vision]"
```

Install everything including dev tooling:

```bash
pip install "gaffers-guide[full]"
```

---

## CLI Usage

Run the full tactical analysis pipeline on a video file:

```bash
gaffers-guide run \
  --video match.mp4 \
  --output ./output \
  --quality-profile balanced
```

List all available quality profiles:

```bash
gaffers-guide profiles list
```

Full run options:

```bash
gaffers-guide run --help
```

### Quality Profiles

The `--quality-profile` flag controls the tradeoff between processing speed and detection accuracy, specifically ball recovery using SAHI (Slicing Aided Hyper Inference).

| Profile | Image Size | SAHI | Frame Skip | Use Case |
|---|---|---|---|---|
| `fast` | 480px | Disabled | 3 | Live preview, fast iteration |
| `balanced` | 640px | Disabled | 2 | Default for most matches |
| `high_res` | 1280px | Disabled | 1 | High-quality post-match review |
| `sahi` | 1280px | Enabled | 1 | Maximum ball recovery, slowest |

SAHI slices each frame into overlapping crops and re-runs inference on each crop, significantly improving detection of small or partially occluded balls at the cost of processing time.

---

## Python SDK

```python
from gaffers_guide.spatial import HomographyEngine
import numpy as np

corners_px = np.array(
    [[120.0, 50.0], [1800.0, 45.0], [1900.0, 1030.0], [80.0, 1035.0]],
    dtype=np.float64,
)
engine = HomographyEngine()
mapping = engine.fit(corners_px, frame_shape=(1080, 1920))
pitch_point = mapping.pixel_to_pitch((960.0, 540.0))
print(pitch_point.to_dict())
```

```python
from pathlib import Path
from gaffers_guide.io import parse_tracking_json

tracking = parse_tracking_json(Path("output/fast_tracking_data.json"))
print(list(tracking.keys()))
```

---

## Architecture

```
┌─────────────────────────────┐       ┌─────────────────────────────┐
│   React / Electron App       │       │   gaffers-guide PyPI Engine  │
│                             │       │                             │
│  nodeIntegration: false     │  IPC  │  gaffers-guide run ...      │
│  contextIsolation: true     │──────▶│  stdout/stderr streamed     │
│  contextBridge exposed API  │       │  back to UI console         │
└─────────────────────────────┘       └─────────────────────────────┘
```

The desktop app runs with `nodeIntegration: false` and `contextIsolation: true`. A preload script exposes a narrow, typed API via Electron's `contextBridge`. The main process uses `child_process.spawn` to launch the locally installed `gaffers-guide` CLI as a subprocess, streaming its output back to the renderer over IPC. The engine never runs inside the renderer process.

The Python engine itself is modular:

| Module | Purpose |
|---|---|
| `gaffers_guide.vision` | YOLO detection, SAHI wrapper, ByteTrack |
| `gaffers_guide.spatial` | Homography fitting, pitch-to-pixel projection |
| `gaffers_guide.tactical` | Team classification, metrics, event detection |
| `gaffers_guide.pipeline` | End-to-end orchestration |
| `gaffers_guide.io` | Tracking JSON parsing and artifact export |
| `gaffers_guide.cli` | CLI entrypoint |

---

## Output Artifacts

Each pipeline run writes three artifacts to the output directory:

- `*_tracking_data.json` — per-frame tracking timeline with player positions, ball coordinates, homography confidence, and telemetry.
- `*_tactical_metrics.json` — per-frame tactical metrics timeline (possession, pressure, spatial coverage).
- `*_report.json` — final coaching report as a list of structured insight cards.

---

## Requirements

- Python 3.11 or higher
- For vision inference: a CUDA, MPS, or CPU-capable environment (PyTorch auto-detected)
- For the desktop app: macOS, Windows, or Linux with Node.js 18+

---

## License

MIT
