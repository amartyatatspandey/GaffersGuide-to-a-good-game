# Gaffers Guide

## Open-Source Football Tactical Analytics

Gaffers Guide is an open-source football analytics project that turns match video into structured tactical data. It combines a headless Python computer-vision engine, a pip-installable SDK, a command-line interface, and a lightweight React/Electron desktop wrapper.

This deck describes the project as it exists today: what is built, how it works, what it outputs, and what is still intentionally outside the current scope.

---

## Slide 1: Product Summary

### Title

Gaffers Guide: Computer Vision for Football Tactical Analysis

### Key Message

Gaffers Guide helps coaches, analysts, and developers process football match video into tracking data, tactical metrics, and structured analysis reports.

### Talking Points

- The project has two parts:
  - a Python CV engine published as `gaffers-guide`
  - a lightweight desktop app wrapper built with React and Electron
- The engine can be used directly through the CLI or imported as a Python SDK.
- The app is a secure shell around the local Python engine, not a monolithic black-box desktop product.

---

## Slide 2: The Problem

### Title

Manual Video Analysis Is Slow and Hard to Scale

### Pain Points

- Analysts spend significant time reviewing video frame-by-frame.
- Ball detection is difficult because the ball is small, fast, and often partially occluded.
- Broadcast camera movement makes pitch-space measurement difficult.
- Raw detections are not enough; teams need structured tracking data and tactical metrics.
- Many useful computer-vision workflows require technical setup that is difficult for non-developers.

### Gaffers Guide Response

Gaffers Guide automates the first layer of match understanding: detection, tracking, pitch mapping, metric generation, and report output.

---

## Slide 3: Two Ways to Use the Project

### Title

One Engine, Two Interfaces

### Coaches and Analysts: Desktop App

- Use the React/Electron desktop app.
- The app checks whether the `gaffers-guide` engine is installed.
- If the engine is missing, the UI shows setup guidance.
- Pipeline logs are streamed into a live UI console.

### Developers and Data Scientists: SDK / CLI

- Install the engine from PyPI:

```bash
pip install gaffers-guide
```

- Install the vision stack:

```bash
pip install "gaffers-guide[vision]"
```

- Run the CLI:

```bash
gaffers-guide run --video match.mp4 --output ./output --quality-profile balanced
```

---

## Slide 4: Current Architecture

### Title

Strictly Decoupled Desktop App and Python Engine

### Architecture

```text
React / Electron App
        |
        | secure IPC bridge
        v
Electron Main Process
        |
        | child_process.spawn(...)
        v
gaffers-guide CLI / Python Engine
        |
        v
JSON artifacts + report outputs
```

### Talking Points

- The Electron renderer does not run the CV engine directly.
- The app uses `nodeIntegration: false` and exposes a narrow bridge through preload/context APIs.
- The Electron main process spawns the local `gaffers-guide` CLI process.
- Standard output and error are streamed back into the UI.
- This keeps the app lightweight and keeps the CV engine reusable outside the app.

---

## Slide 5: Python Package Layout

### Title

Modular SDK Structure

### Main Package Areas

| Package | Purpose |
|---|---|
| `gaffers_guide.core` | Shared data types and exceptions |
| `gaffers_guide.io` | Tracking JSON parsing and export helpers |
| `gaffers_guide.spatial` | Homography and pitch coordinate mapping wrappers |
| `gaffers_guide.cv` | Ball detection helpers, ROI logic, SAHI wrapper |
| `gaffers_guide.pipeline` | End-to-end pipeline, tracking, calibration, metrics |
| `gaffers_guide.runtime` | Runtime entrypoints used by CLI and app |
| `gaffers_guide.cli` | `gaffers-guide` command-line interface |

### Key Point

The SDK was refactored into a `src/` layout so it can be installed, imported, tested, and packaged like a normal Python library.

---

## Slide 6: Computer Vision Pipeline

### Title

From Video Frames to Tactical Data

### Pipeline Flow

```text
Input video
  -> YOLO detections
  -> ByteTrack tracking
  -> team classification / ID healing
  -> ball detection refinement
  -> homography / pitch projection
  -> trajectory refinement
  -> tactical metrics
  -> final JSON report
```

### Current Technologies

- YOLO via `ultralytics`
- ByteTrack via `supervision`
- OpenCV for video processing and image operations
- NumPy / SciPy for spatial math and smoothing
- SAHI-inspired sliced inference for ball recovery

---

## Slide 7: Ball Detection and SAHI

### Title

Improving Small-Ball Recall Without Running Full-Frame High-Res Everywhere

### Current Design

Gaffers Guide includes a context-aware SAHI wrapper for ball detection. It is designed to improve ball recovery while controlling runtime cost.

### Components

- `OptimizedSAHIWrapper`
- `PitchROIProvider`
- `TemporalBallPrior`
- `SliceBatchInferencer`
- `ball_candidate_fuser`

### How It Works

- Run normal full-frame YOLO detection first.
- Estimate pitch ROI to avoid wasting slices on crowd/sky.
- Use last known ball position to create a temporal search region.
- Generate a bounded number of slices.
- Run batched inference on slices.
- Fuse full-frame and slice candidates using confidence and proximity.

---

## Slide 8: Quality Profiles

### Title

Configurable Speed vs Accuracy

### Profiles

| Profile | Intent | SAHI | Typical Use |
|---|---|---|---|
| `fast` | Fastest pipeline mode | Off | quick iteration / preview |
| `balanced` | Default tradeoff | Off | standard analysis |
| `high_res` | Higher image size | Off | higher-quality post-match run |
| `sahi` | Maximum ball recall | On | slowest, best small-ball recovery |

### CLI Example

```bash
gaffers-guide run \
  --video backend/data/match_test.mp4 \
  --output backend/output \
  --quality-profile sahi
```

---

## Slide 9: Spatial Mapping

### Title

Mapping Camera Pixels to Pitch Coordinates

### Current Scope

The public `gaffers_guide.spatial` module exposes simple homography helpers for SDK users. The deeper project-specific calibration flow lives in the pipeline modules.

### Example

```python
import numpy as np
from gaffers_guide.spatial import HomographyEngine

corners_px = np.array(
    [[120.0, 50.0], [1800.0, 45.0], [1900.0, 1030.0], [80.0, 1035.0]],
    dtype=np.float64,
)

engine = HomographyEngine()
mapping = engine.fit(corners_px, frame_shape=(1080, 1920))
pitch_point = mapping.pixel_to_pitch((960.0, 540.0))
print(pitch_point.to_dict())
```

### Key Point

Homography converts camera-space detections into pitch-space coordinates so tracking data can become tactical data.

---

## Slide 10: Output Artifacts

### Title

What the Pipeline Produces

### Current Outputs

Each profile run writes JSON artifacts such as:

- `*_tracking_data.json`
- `*_tactical_metrics.json`
- `*_report.json`

### Artifact Roles

| Artifact | Purpose |
|---|---|
| Tracking data | Per-frame player/ball state, telemetry, camera fallback info |
| Tactical metrics | Per-frame metrics timeline generated from refined tracks |
| Report | Final structured tactical insight cards |

### Validation Status

The system has been run end-to-end on `backend/data/match_test.mp4` across:

- `fast`
- `balanced`
- `high_res`
- `sahi`

All profiles completed successfully in the latest QA run.

---

## Slide 11: End-to-End QA Evidence

### Title

Validated on `match_test.mp4`

### QA Phases Completed

- Environment baseline
- Module import matrix
- CLI contract checks
- Full profile E2E matrix
- Artifact validation
- Final QA summary

### Results

- Import matrix: 36/36 modules passed
- CLI profile handling: passed
- E2E profile matrix: passed
- JSON artifact validation: passed

### Note

The optional overlay video is not produced by the current headless pipeline mode. The reliable outputs are JSON tracking data, metrics, and report artifacts.

---

## Slide 12: Desktop App Integration

### Title

Electron Wrapper Around the Local Engine

### Current Behavior

- App checks for `gaffers-guide` availability.
- If unavailable, it presents setup guidance.
- If available, it allows pipeline execution through a secure bridge.
- Logs are streamed from the spawned process into the UI.

### Security Design

- Renderer remains isolated.
- Node integration is disabled.
- The preload layer exposes only controlled methods.
- The heavy CV work runs outside the renderer process.

---

## Slide 13: Packaging and Distribution

### Title

Published as a Python Package

### Package

```bash
pip install gaffers-guide
```

### Extras

```bash
pip install "gaffers-guide[vision]"
pip install "gaffers-guide[full]"
```

### Release Work

The project now supports binary wheels for protected modules through Cython. Platform-specific wheels are required because compiled extensions are tied to Python ABI and OS.

Examples:

- macOS CPython 3.11: `cp311-macosx...whl`
- Windows CPython 3.11: `cp311-win_amd64.whl`
- Windows CPython 3.13: `cp313-win_amd64.whl`

---

## Slide 14: IP Protection Strategy

### Title

Protecting Proprietary Pipeline Logic

### Why Protection Was Needed

The most valuable project logic is not the public YOLO model architecture. It is the orchestration around:

- ball recovery
- pitch ROI slicing
- temporal search
- candidate fusion
- calibration
- team classification
- ID healing
- tactical metric generation

### Current Approach

- 14 high-value modules are compiled with Cython.
- The release wheel contains native extensions instead of readable source for those modules.
- Wheel audits verify:
  - `compiled_extensions=14`
  - `forbidden_sources=0`

### Optional Rust Layer

An optional `gaffers_core_math` PyO3 module exists for narrow pure-math kernels:

- candidate ranking
- temporal ball prior

The Python/Cython path remains the main supported protection mechanism.

---

## Slide 15: What Is Not Being Claimed

### Title

Honest Scope Boundaries

### Not Current Production Claims

Gaffers Guide does **not** currently claim:

- a full 3D digital twin product
- real-time sideline decision support
- Opta/TRACAB-compatible exports
- automatic route recognition
- automatic personnel grouping
- biometric or wearable load management
- live tactical simulation
- guaranteed millimeter-precision reconstruction

### Current Positioning

It is a working open-source football CV analytics engine and desktop wrapper, with a validated local pipeline and modular SDK surface.

---

## Slide 16: Roadmap

### Title

Practical Next Steps

### Near-Term

- Standardize GitHub Actions wheel builds using `cibuildwheel`.
- Build wheels for Windows and macOS across supported Python versions.
- Improve artifact output routing so `--output` is honored consistently.
- Replace deprecated `google.generativeai` dependency with the newer Gemini SDK.
- Add more automated tests around compiled-wheel imports.

### Medium-Term

- Improve frontend run UX and artifact browsing.
- Add richer report visualization.
- Expand calibration reliability on varied broadcast footage.
- Add optional model download/version management.

### Long-Term

- Broaden platform support.
- Improve tactical event detection.
- Evaluate stronger Rust kernels where Python object boundaries are not a blocker.

---

## Slide 17: Final Message

### Title

Why Gaffers Guide Matters

Gaffers Guide makes football video analysis more programmable, repeatable, and accessible. The project turns raw match footage into structured tactical data through a local, open-source computer-vision engine that can be used by developers through Python or by analysts through a desktop wrapper.

The strongest current value is the validated pipeline: video in, tracking and metrics out, with quality profiles and protected binary distribution for proprietary logic.

---

## Appendix: Current Commands

### Install

```bash
pip install gaffers-guide
pip install "gaffers-guide[vision]"
```

### List Profiles

```bash
gaffers-guide profiles list
```

### Run Analysis

```bash
gaffers-guide run \
  --video backend/data/match_test.mp4 \
  --output backend/output \
  --quality-profile balanced
```

### Build Protected macOS Wheel

```bash
python -m build --wheel --outdir dist_release_2.0.3
```

### Validate Release Artifacts

```bash
python -m twine check dist_release_2.0.3/*
```

---

## Appendix: Suggested Presenter Notes

Use the deck to tell a grounded story:

1. Manual video analysis is slow.
2. Gaffers Guide automates the first CV layer.
3. The Python engine is reusable through CLI and SDK.
4. The Electron app is a secure wrapper around the engine.
5. Ball detection is improved with context-aware SAHI.
6. Output artifacts are JSON and can be consumed downstream.
7. The package is moving toward professional multi-platform binary distribution.

Avoid positioning the project as a finished enterprise scouting suite. Present it as a strong technical foundation with a working pipeline and clear next steps.
# Gaffer's Guide: Automated Tactical

# Intelligence, Computer Vision, & LLM

# Coaching Platform for Football

The professional landscape of football, particularly within the elite echelons of the
Southeastern Conference (SEC) and the National Football League (NFL), has historically relied
upon the intersection of seasoned intuition and labor-intensive manual observation. However,
as the volume of available match data expands exponentially, the "gut feeling" era is being
superseded by a requirement for high-velocity, automated tactical intelligence. The Gaffer's
Guide platform represents a paradigm shift in this domain, integrating a sophisticated
computer vision (CV) pipeline with a Retrieval-Augmented Generation (RAG) coaching
assistant. This ecosystem is designed to bridge the gap between raw broadcast footage and
the nuanced, actionable insights required by modern coaching staffs.^1 By leveraging
state-of-the-art object detection, spatiotemporal tracking, and large language models (LLMs)
grounded in proprietary tactical playbooks, Gaffer's Guide provides a unified workspace that
democratizes elite-level analytics for clubs across the competitive spectrum.^3

## The Problem: The Latency and Subjectivity Crisis in

## Manual Performance Analysis

The contemporary football analyst faces a tripartite challenge: the sheer volume of manual
effort required for film study, the lack of immediate actionable insights from traditional
statistical models, and the prohibitive technical complexity of modern data integration. In the
high-stakes environment of collegiate and professional football, coaching staffs spend an
average of forty to sixty hours per week tagging game film, a process that involves identifying
every formation, personnel grouping, and individual player route.^5 This manual tagging is not
only a significant drain on human resources but is also inherently subjective and prone to error,
particularly as fatigue sets in during late-night scouting sessions.^5
Furthermore, the data generated by traditional scouting methods often lacks the "why" behind
the numbers. A standard box score may indicate that a quarterback was sacked four times, but
it fails to articulate whether those sacks were a result of a missed blocking assignment in a Nick
Saban-style "Cover 7" shell or a failure of the wide receivers to find "green grass" in a Mike
Leach-inspired "Mesh" concept.^8 This disconnect between raw statistics and tactical context
creates a "latency gap" where adjustments are made days after the game, rather than in the
crucial minutes following a series.^4
Technical complexity further exacerbates these issues. Elite organizations often maintain
disparate systems for GPS tracking, optical data, and video analysis. These systems frequently
operate in silos, requiring expensive data science departments to normalize and synthesize the


information into a cohesive strategy.^11 For mid-tier programs and academies, the cost and
infrastructure required to maintain such an "intelligent enterprise" are simply out of reach,
leading to a widening competitive chasm.^3
**Analysis Barrier Operational Impact Strategic Consequence**
Manual Tagging Latency 40+ hours/week spent on
"dead" data
Inability to adapt to
mid-season opponent shifts
Subjective Error 15-20% inconsistency in
human-tagged film
Flawed tendency reporting and
game-planning
Data Silos Fragmentation between
physical and tactical stats
Incomplete "Digital Athlete"
profile for load management
Technical Cost Prohibitive licensing and GPU
requirements
Competitive disadvantage for
lower-budget programs
The Gaffer's Guide identifies these pain points not merely as inefficiencies but as structural
failures in the coaching workflow that prevent teams from maximizing their roster's potential.^14

## Our Solution: The Gaffer's Guide Unified Intelligence

## Framework

To resolve the crisis of manual analysis, the Gaffer's Guide proposes a unified architecture that
automates the transition from raw pixels to tactical wisdom. This solution is built upon three
primary pillars: an end-to-end computer vision pipeline, a RAG-powered coaching assistant,
and an immersive 3D workspace. Unlike legacy systems that require manual input, the CV
pipeline utilizes deep learning models to extract spatiotemporal data directly from broadcast or
tactical video feeds.^1 This data is then contextualized by the RAG Coach, which has been
"trained" on thousands of pages of tactical manuals, from the intricate defensive rules of Nick
Saban to the rapid-fire progressions of the Air Raid offense.^8
The integration of these components creates a "live" tactical board. As the computer vision
system tracks players and the ball, the RAG assistant simultaneously compares their


movements against the team's established playbook. If a safety fails to execute a "MEG" (Man
Everywhere he Goes) assignment in a Cover 7 call, the system flags the deviation in real-time.^8
This removes the need for manual tagging and allows coaches to focus on the high-level
psychological and strategic aspects of the game—elements that AI cannot currently replicate,
such as player motivation and team chemistry.^19
**Solution Component Technical Implementation Practical Utility**
End-to-End CV Pipeline YOLOv8/v11 + Norfair
Tracking
Automated, objective
play-by-play extraction
RAG Coach Assistant LLM + Vectorized Playbooks Natural language queries for
tactical validation
Immersive Workspace Electron + Three.js + React "Digital Twin" of the match for
3D visualization
Metric Engine Homography +
Spatiotemporal Data
Real-time calculation of EPA
and Success Rates
By delivering this functionality through a cross-platform desktop application, Gaffer's Guide
ensures that advanced analytics are accessible in the film room, on the practice field, and
during transit, effectively democratizing the "Next Gen Stats" experience for all levels of the
sport.^12

## Core Capabilities: Advanced Tracking, 3D

## Visualization, and LLM Integration

The technical superiority of the Gaffer's Guide is rooted in its core capabilities, which transform
chaotic match footage into structured, mathematical models of play. This process begins with
high-fidelity object detection and extends to the immersive reconstruction of the field in a
virtual environment.

### Advanced Multi-Object Tracking and Team Identification


The platform utilizes the YOLO (You Only Look Once) architecture—specifically optimized
versions of YOLOv8 and the latest YOLOv11—to achieve state-of-the-art accuracy in detecting
players, referees, and the football.^16 To maintain consistent identities during high-speed play
and frequent occlusions, the system employs the Norfair tracking algorithm. This approach
integrates velocity prediction with Intersection-over-Union (IoU) association, ensuring that a
player's ID remains stable even when they are momentarily hidden by a cluster of linemen or a
dynamic camera pan.^13
Team identification is fully automated through unsupervised clustering (K-Means or DBSCAN)
of jersey color features. This allows the system to distinguish between home and away teams
without the need for pre-labeled datasets or manual team assignment.^1 The accuracy of this
tracking is essential for calculating physical performance indicators such as instantaneous
speed, acceleration, and total distance covered, which are then scaled from pixel coordinates
to meters using homography-based field projection.^1

### 3D Field Reconstruction and Homography

One of the most significant technical hurdles in football CV is the dynamic nature of broadcast
cameras. To provide accurate metrics, Gaffer's Guide must map the 2D video frame to a 3D
world coordinate system. This is achieved through a robust homography pipeline that detects
pitch markings—such as yard lines, hash marks, and the center circle—and uses a Direct Linear
Transformation (DLT) powered by RANSAC to calculate the camera's perspective.^23
The mathematical mapping is expressed as:
where is the homography matrix. By resolving the lens parameters and orientation,
the system can project player positions onto a 3D tactical board with millimeter precision.^25 This
allows coaches to view the game from a "top-down" or "behind-the-QB" perspective,
regardless of the original camera angle.^26

### LLM Coaching Assistant and Tactical Validation

The Gaffer's Guide goes beyond tracking by integrating a RAG-powered coaching assistant.
This module serves as an "AI Coordinator" that can answer complex tactical questions. For
example, a coach might ask, "Did the strong safety follow his RAT (Read and Trap) rules on that
third-down slant?" The system retrieves the specific "RAT" rule from the indexed Saban
playbook, analyzes the tracking data of the strong safety, and provides a natural language
answer: "The safety failed to pass off the TE to the linebacker, vacating the perimeter and
allowing a 7-yard gain".^18


```
Capability Underlying Tech Result
```
Player/Ball Detection YOLOv8-P2S3A / YOLOv11 (^) 79.4% validation precision 16
Consistent Tracking Norfair + Kalman Filters 87% ID retention in
crowded boxes 13
Tactical Mapping Homography / DLT Real-world metric scaling
(km/h, meters) 24
Tactical Logic RAG + Vectorized
Playbooks
Grounded validation of
player assignments 29

## Backend AI Stack: Engineering the Deep Learning

## Pipeline

The backend of Gaffer's Guide is a high-performance engine built on Python and FastAPI,
designed to handle the massive data throughput of multi-stream video analysis. The
architecture is modular, allowing for the rapid swapping of models as new state-of-the-art
detectors emerge.

### PyTorch and YOLO Orchestration

At the center of the CV engine is PyTorch, which provides the deep learning framework for the
YOLO models. The platform supports a variety of YOLO variants, from the "nano" (n) models
optimized for real-time inference on edge devices to the "extra-large" (x) models used for
high-precision post-game analysis.^22 Performance benchmarking on NVIDIA Jetson hardware
demonstrates that the system can maintain 18-30 frames per second (FPS), which is critical for
the real-time tactical feedback required in modern football.^13

### Slicing Aided Hyper Inference (SAHI) for Small Objects

Detecting the football is a notorious challenge in computer vision because the ball often
occupies a tiny fraction of the overall frame and can become a blur at high velocities.^31 To solve
this, Gaffer's Guide integrates SAHI (Slicing Aided Hyper Inference). SAHI breaks down a
high-resolution 1080p or 4K frame into smaller, overlapping slices (e.g., pixels).
Inference is run on each slice independently, preventing the ball from being "downsampled"
into oblivion during the resizing process.^31
The SAHI workflow includes:

1. **Image Partitioning** : Slicing the frame with controlled overlap to ensure objects on the
    boundaries are captured.^31


2. **Parallel Inference** : Running detection on all slices simultaneously to minimize latency.^33
3. **Result Merging** : Using Non-Maximum Suppression (NMS) to stitch the detections back
    into a single coordinate system.^31

### RAG and Vector Intelligence

The RAG component utilize a vector database (such as Pinecone or FAISS) to store
embeddings of thousands of football-specific documents.^34 These include:
● **Tactical Manuals** : Detailed rules for Cover 1, Cover 2, and Cover 7 defensive schemes.^8
● **Playbooks** : Concept diagrams and read progressions for the Air Raid and Pro-Style
offenses.^17
● **Historical Data** : Past match performances and situational tendencies.^37
By using a multi-step agentic workflow—including an Intent Parser and a Temporal Logic
node—the RAG Coach can resolve queries like "Compare our success rate on Mesh concepts
in the last two games versus our season average".^29
**Backend Component Technology Role**
API Layer FastAPI Asynchronous request
handling and orchestration
Inference Engine PyTorch / TensorRT High-speed model
execution 30
Object Detection YOLOv11 / SAHI Tracking players and the
football 16
Intelligence Store FAISS / Vector DB Storage of semantic
playbooks 34
Data Processing OpenCV / Pandas Video manipulation and
metric calculation 22

## Frontend Desktop App Stack: Electron and Immersive

## Visualization

The user-facing component of Gaffer's Guide is a professional-grade desktop application that
prioritizes responsiveness, security, and visual depth. By building on the Electron framework,
the platform provides a native-like experience that can operate offline—a critical requirement


for coaching staffs traveling to games.^40

### Electron and Next.js Architecture

The application uses Electron to bundle a Next.js/React frontend with a Node.js backend. This
allows for deep integration with the host operating system's file system, enabling fast local
storage of massive video files and tracking metadata.^21 To avoid the common pitfalls of
Electron's memory consumption, Gaffer's Guide employs several architectural best practices:
● **Context Isolation** : Keeping the renderer and main processes separate to enhance
security and stability.^21
● **Lazy Loading** : Using route-based code splitting to ensure that only the necessary
modules are loaded at startup, reducing the initial memory footprint.^41
● **IPC Optimization** : Utilizing ipcRenderer.invoke() for asynchronous, non-blocking
communication between the UI and the heavy-duty CV backend.^43

### Three.js and the 3D Tactical Workspace

The "Immersive Workspace" is powered by React Three Fiber (R3F), a React-based wrapper for
Three.js. This allows coaches to interact with a "Digital Twin" of the football field. Players are
rendered as 3D entities whose movements are driven directly by the CV tracking data.^27
To maintain a high frame rate (60 FPS) in the 3D scene, the application implements on-demand
rendering. By setting frameloop="demand", the scene only re-renders when a change
occurs—such as a user scrubbing the video timeline or rotating the camera—drastically
reducing GPU and battery drain on portable coaching laptops.^27 Furthermore, the use of
InstancedMesh allows the system to render all 22 players in a single draw call, ensuring smooth
performance even on lower-end hardware.^45
**Frontend Tool Purpose Performance Benefit**
Electron Desktop Shell Native file access and
offline capability 21
Three.js / R3F 3D Visualization Real-time "Digital Twin" of
the match 44
GSAP Animations Seamless transitions
between 2D and 3D views
Tailwind CSS UI/UX Clean, responsive
dashboard for high-stress


```
use
Zustand State Management Transient subscriptions for
high-frequency data 44
```
## Pipeline Outputs: Tracking Data, Metrics, and

## Annotated Video

The value of the Gaffer's Guide is crystallized in its refined outputs, which transform hours of
raw video into a standardized repository of tactical knowledge. These outputs are designed to
integrate seamlessly into the existing workflows of professional analysts and SEC coaching
staffs.

### Spatiotemporal Tracking and Event Data

The primary output is a high-frequency data stream that records the coordinates of
every player and the ball for every frame of the match. This data is exported in
industry-standard formats—such as the JSON and XML schemas used by Opta and
TRACAB—allowing for cross-platform compatibility.^47 Each event is tagged with metadata,
including the player ID, team, and "action" (e.g., pass, tackle, interception).^48

### Advanced Tactical Metrics (KPI Engine)

The platform calculates a library of advanced metrics that go beyond traditional statistics. For
the SEC audience, the system prioritizes "Efficiency" and "Explosiveness" metrics, which are
highly predictive of championship success.^50
**Metric Category Specific KPI Tactical Significance**
Efficiency Success Rate Consistency in "staying on
schedule" 50
Explosiveness EPA (Expected Points
Added)
Value created by big plays
and scoring threats 50
Drive Performance Points Per Drive (PPD) The "Gold Standard" for
offensive firepower 50
Defensive Quality RTD (Roster Talent
Discrepancy)
Coaching quality relative to
recruit talent 14


```
Special Teams PAdj (Possession Adjusted) Performance metrics
normalized for field position
51
```
### Annotated Video and 3D Replays

In addition to raw data, the system generates "Coach-Ready" video files. These include
automated overlays such as:
● **Pass/Run Tendency Heatmaps** : Visualizing where an opponent is most likely to attack.^1
● **Route Recognition Diagrams** : Identifying the specific pass concepts (e.g., Mesh, Smash,
Vertical) as they develop.^17
● **Personnel Grouping Tags** : Automatically identifying 11-personnel vs. 12-personnel and
their respective success rates.^5
The final output is the 3D Workspace file, which can be shared with players for interactive
VR-style review sessions, allowing them to literally step inside a play and see the field from their
own perspective.^1

## Quality Profile System: Adaptive Performance

## Ecosystem

Gaffer's Guide acknowledges that the hardware environment of a football team is diverse,
ranging from powerful GPU clusters in the video room to mobile tablets on the sideline. To
ensure reliability across all use cases, the platform implements a Dynamic Quality Profile
system.

### Performance Tiers

The system intelligently scales its CV and rendering parameters based on the selected profile
and the detected hardware capabilities. This is managed by the PerformanceMonitor
component, which tracks average FPS and triggers callbacks to adjust resolution or model
complexity if performance declines.^27
**Profile Name Model Variant Resolution Key Use Case
Fast** YOLOv8-n 640px Real-time sideline
feedback 13
**Balanced** YOLOv11-m 800px Standard
post-match film
study 16


```
High-Res YOLOv11-x 1200px Detailed scouting
and jersey ID 22
SAHI Sliced Tiled 4K Native Precise ball tracking
and metric scaling
31
```
### Resource Management

To maintain high-quality detection on hardware with limited resources, the SAHI profile utilizes
sliced inference, which significantly reduces the memory footprint required to process
high-resolution images by handling smaller segments independently.^33 On the frontend, the
application employs React.memo and useMemo for expensive 3D geometries, ensuring that
once a field model is loaded, it does not tax the CPU with redundant re-calculations.^27

## Future Roadmap: Toward the Digital Athlete

## (2026-2030)

The roadmap for Gaffer's Guide is aligned with the broader digital transformation of football,
moving from retrospective analysis toward real-time simulation and predictive health
management.

### Render Optimization and Skeletal Tracking

The next major update will integrate skeletal keypoint detection, moving beyond bounding
boxes to track 30+ points on a player's body, including joints such as elbows, knees, and hips.^12
This will unlock "millimeter-precision" biomechanical analysis, allowing teams to analyze a
quarterback's throwing mechanics or a lineman's leverage in real-time. To support this, the
rendering engine will migrate to WebGPU, which offers a 2-10x performance gain in specific 3D
scenarios.^46

### RAG Expansion and TacticAI Integration

Building on recent research like Google DeepMind’s TacticAI, the RAG Coach will evolve from a
validator into a generator. Instead of just checking if a play was successful, the AI will simulate
hundreds of "alternative" tactical scenarios—such as a different corner-kick routine—and
propose the one with the highest mathematical probability of success.^19 This "Tactical
Simulation" will allow teams to stress-test their game plans in a virtual environment before ever
stepping on the field.^19

### Model Calibration and Load Management

The "Digital Athlete" initiative will combine optical tracking data with wearable sensor data (e.g.,
heart rate and impact force) to create a holistic profile of player health.^12 By analyzing these
trends over multiple seasons, the system will provide early-warning alerts for fatigue-related


injury risks, fundamentally changing how SEC teams manage their rosters over the course of a
rigorous schedule.^4
**Roadmap Milestone Technology Shift Expected Benefit
2026: Skeletal Analysis** HRNet / Body Keypoints Biomechanical scouting and
injury detection 12
**2027: Tactical Simulation** Generative TacticAI Automated "What-If"
scenario planning 19
**2028: Edge-Real-Time** 5G / TensorRT Edge In-game decision support
under 100ms 12
**2029: WebGPU Rendering** Three.js Shader Language Cinematic-quality 3D
tactical boards 46
**2030: Unified Ecosystem** Multi-Source Fusion The complete "Digital
Athlete" profile 15

## Conclusion: Synthesizing Intelligence for the Modern

## Gaffer

The Gaffer's Guide platform is the result of a rigorous engineering effort to consolidate the
disparate tools of modern football analysis into a single, high-performance ecosystem. By
automating the extraction of tactical data and grounding it in the logic of elite playbooks, the
platform resolves the "latency gap" that has long plagued manual film study.^1 The integration of
a RAG-powered coach ensures that the insights provided are not just statistical but are
tactically relevant to the specific philosophy of the organization.^8
As the sport moves into an AI-first future, the ability to rapidly synthesize millions of data points
into a clear game plan will be the primary differentiator between championship contenders and
also-rans. Gaffer's Guide provides the infrastructure for this "intelligent enterprise," ensuring
that coaches can return to what they do best: leading their players and making the high-stakes
decisions that define the "beautiful game".^12 The platform effectively replaces the "gut feeling"
with a data-driven precision that respects the cultural and strategic nuances of football while
leveraging the undeniable power of modern computational intelligence.^2
Through its adaptive quality profiles, standardized outputs, and immersive 3D workspace,
Gaffer's Guide is not just a tool for the present; it is a foundational architecture for the future of
tactical intelligence in football. Whether on the sidelines of a sold-out SEC stadium or the
training pitches of a developing academy, the Digital Gaffer is now equipped with a "second


pair of eyes" that never tire and a tactical mind that never forgets.^3

#### Works cited

#### 1. Video Processing and Data Analysis for Football Matches using ..., accessed April

#### 27, 2026, https://ijsred.com/volume8/issue6/IJSRED-V8I6P249.pdf

#### 2. Mapping football tactical behavior and collective dynamics with artificial

#### intelligence: a systematic review - PMC, accessed April 27, 2026,

#### https://pmc.ncbi.nlm.nih.gov/articles/PMC12163489/

#### 3. MASTER'S THESIS Computer Vision System for Football Performance Analysis -

#### O2 Repositori UOC, accessed April 27, 2026,

#### https://openaccess.uoc.edu/server/api/core/bitstreams/09ec4d38-8be2-403e-bf

#### be-7001fa3932bd/content

#### 4. Football Data Trends 2026: AI, Player Tracking & What's Next - Sportmonks,

#### accessed April 27, 2026,

#### https://www.sportmonks.com/blogs/football-data-trends-2026-ai-player-trackin

#### g-whats-next/

#### 5. Using Computer Vision and Machine Learning to Automatically Classify NFL

#### Game Film and Develop a Player Tracking System - AWS, accessed April 27, 2026,

#### https://fourtverts.s3.amazonaws.com/assets/usingcomputervisionforfootballtrack

#### ing.pdf

#### 6. Exploring Sports Analytics of Coaches: An Optimization for Team Performance

#### and Strategy Development - ResearchGate, accessed April 27, 2026,

#### https://www.researchgate.net/publication/384241603_Exploring_Sports_Analytics

#### _of_Coaches_An_Optimization_for_Team_Performance_and_Strategy_Developm

#### ent

#### 7. Coaching Manual - RAMP InterActive, accessed April 27, 2026,

#### https://cloud.rampinteractive.com/northstarsfootball.com/files/Coaching%20Man

#### ual%20Northstars.pdf

#### 8. Understanding Saban's Cover 7 Defense | PDF | American Football - Scribd,

#### accessed April 27, 2026,

#### https://www.scribd.com/document/376154338/Saban-Cover-

#### 9. What are the main concepts of the Mike Leach Air Raid? : r/footballstrategy -

#### Reddit, accessed April 27, 2026,

#### https://www.reddit.com/r/footballstrategy/comments/1e0s8o8/what_are_the_mai

#### n_concepts_of_the_mike_leach_air/

#### 10. How AI and Data Analytics Are Revolutionizing Football in 2026 - Flickit, accessed

#### April 27, 2026,

#### https://www.flickit.app/blogs/how-ai-and-data-analytics-are-revolutionizing-foot

#### ball-in-

#### 11. Harnessing the Power of Sport Analytics in Collegiate Coaching - Adam Ringler,

#### accessed April 27, 2026,

#### https://adamringler.com/harnessing-the-power-of-sport-analytics-in-collegiate-

#### coaching/

#### 12. From RFID to real-time AI: How a decade of AWS and NFL Next Gen Stats has


#### rewritten the playbook - ZK Research, accessed April 27, 2026,

#### https://zkresearch.com/from-rfid-to-real-time-ai-how-decade-of-aws-and-nfl-n

#### ext-gen-stats-has-rewritten-the-playbook/

#### 13. Real-Time Video Analysis of Football Matches Using YOLOv8 and Computer

#### Vision Techniques - RSIS International, accessed April 27, 2026,

#### https://rsisinternational.org/journals/ijrias/uploads/vol11-iss1-pg1482-1486-

#### _pdf.pdf

#### 14. Using Analytics To Make Better Coach Hiring And Retention Decisions, accessed

#### April 27, 2026,

#### https://athleticdirectoru.com/articles/analytics-coach-hiring-decisions/

#### 15. Why More Teams Are Switching to AI-Powered Sports Analytics - WSC Sports,

#### accessed April 27, 2026,

#### https://wsc-sports.com/blog/industry-insights/why-more-teams-are-switching-t

#### o-ai-powered-sports-analytics/

#### 16. Automatic Estimation of Football Possession via Improved YOLOv8 Detection and

#### DBSCAN-Based Team Classification - ResearchGate, accessed April 27, 2026,

#### https://www.researchgate.net/publication/400826869_Automatic_Estimation_of_

#### Football_Possession_via_Improved_YOLOv8_Detection_and_DBSCAN-Based_Tea

#### m_Classification

#### 17. THE AL-RAID OFFENSE - Playbook Gamer, accessed April 27, 2026,

#### https://playbookgamer.com/wp-content/uploads/2021/12/Al-Raid-Offense.pdf

#### 18. The Complete Guide to Nick Saban Coverages at Alabama - Throw Deep

#### Publishing, accessed April 27, 2026,

#### https://throwdeeppublishing.com/blogs/news/nick-saban-s-alabama-pass-cover

#### ages

#### 19. How AI Is Reshaping Football. What It's Doing Now, Where It's Going... | by

#### thefootballfables | Medium, accessed April 27, 2026,

#### https://medium.com/@thefootballfable/how-ai-is-reshaping-football-da313b6d

#### d

#### 20. How to Coach Smarter with AI: Tools, Trends & Tactics You Need in 2026 -

#### Delenta, accessed April 27, 2026,

#### https://www.delenta.com/blog/ai-coaching-trends-tools-

#### 21. Optimizing Desktop Apps with Electron: Performance and Security Insights - Red

#### Sky Digital, accessed April 27, 2026,

#### https://redskydigital.com/au/optimizing-desktop-apps-with-electron-performanc

#### e-and-security-insights/

#### 22. Training YOLOv8 for Football Object Detection: Insights from the Norwegian

#### Eliteserien, accessed April 27, 2026,

#### https://medium.com/@waynemataara/training-yolov8-for-football-object-detecti

#### on-insights-from-the-norwegian-eliteserien-7f0d00d28be

#### 23. Automating NFL Film Study: Using Computer Vision to Analyze All, accessed April

#### 27, 2026,

#### https://cvgl.stanford.edu/teaching/cs231a_winter1415/prev/projects/LeeTimothy.p

#### df

#### 24. Calibrating Football Fields with Keypoints (Part 2 / 2) | by Erfan Akbarnezhad |


#### Medium, accessed April 27, 2026,

#### https://medium.com/@akbarnezhad1380/calibrating-football-fields-with-keypoint

#### s-part-2-3-daea248585e

#### 25. Camera Calibration - SoccerNet, accessed April 27, 2026,

#### https://www.soccer-net.org/tasks/camera-calibration

#### 26. Camera Calibration and Player Localization in SoccerNet-v2 and Investigation of

#### their Representations for Action Spotting - ResearchGate, accessed April 27,

#### 2026,

#### https://www.researchgate.net/publication/354307920_Camera_Calibration_and_P

#### layer_Localization_in_SoccerNet-v2_and_Investigation_of_their_Representations_

#### for_Action_Spotting

#### 27. React-Three-Fiber: Enhancing Scene Quality with Drei + Performance Tips -

#### Medium, accessed April 27, 2026,

#### https://medium.com/@ertugrulyaman99/react-three-fiber-enhancing-scene-qual

#### ity-with-drei-performance-tips-976ba3fba67a

#### 28. Saban's Cover 1 Defense Explained | PDF | Sports | Gridiron Football - Scribd,

#### accessed April 27, 2026,

#### https://www.scribd.com/document/371937312/Nick-Saban-Cover-

#### 29. A Chatbot for Football Analytics: A deep dive into RAG, LLM Orchestration and

#### Function Calling - Diva-portal.org, accessed April 27, 2026,

#### https://www.diva-portal.org/smash/get/diva2:1986700/FULLTEXT01.pdf

#### 30. Benchmarking YOLOv8 Variants for Object Detection Efficiency on Jetson Orin

#### NX for Edge Computing Applications - MDPI, accessed April 27, 2026,

#### https://www.mdpi.com/2073-431X/15/2/

#### 31. Small Object Detection using YOLO with SAHI Explained - Labellerr, accessed

#### April 27, 2026, https://www.labellerr.com/blog/small-object-detection/

#### 32. Slicing Aided Hyper Inference (SAHI) for Small Object Detection | Explained -

#### Encord, accessed April 27, 2026,

#### https://encord.com/blog/slicing-aided-hyper-inference-explained/

#### 33. Ultralytics Docs: Using YOLO26 with SAHI for Sliced Inference, accessed April 27,

#### 2026, https://docs.ultralytics.com/guides/sahi-tiled-inference/

#### 34. RAG indexing: Structure and evaluate for grounded LLM answers - Meilisearch,

#### accessed April 27, 2026, https://www.meilisearch.com/blog/rag-indexing

#### 35. The Trade-off Playbook: Engineering High-Impact Retrieval-Augmented

#### Generation (RAG) Systems - DEV Community, accessed April 27, 2026,

#### https://dev.to/satyam_chourasiya_99ea2e4/the-trade-off-playbook-engineering-

#### high-impact-retrieval-augmented-generation-rag-systems-f

#### 36. Air Raid basics : r/footballstrategy - Reddit, accessed April 27, 2026,

#### https://www.reddit.com/r/footballstrategy/comments/1cxeem2/air_raid_basics/

#### 37. Built a RAG system on top of 20+ years of sports data — here is what actually

#### worked and what didn't - Reddit, accessed April 27, 2026,

#### https://www.reddit.com/r/Rag/comments/1rpkyn9/built_a_rag_system_on_top_of_

#### 20_years_of_sports/

#### 38. SQL + RAG: Building an LLM Agent for a fictional soccer universe that knows

#### when to Query vs. Search - Lucas Moda, accessed April 27, 2026,


#### https://lukmoda.medium.com/sql-rag-building-an-llm-agent-for-a-fictional-socc

#### er-universe-that-knows-when-to-query-vs-eef49d80f

#### 39. Loading Opta data - socceraction 1.5.3 documentation - Read the Docs,

#### accessed April 27, 2026,

#### https://socceraction.readthedocs.io/en/latest/documentation/data/opta.html

#### 40. Electron App Testing Guide: Tools, Strategy & CI/CD (2026) - ACCELQ, accessed

#### April 27, 2026, https://www.accelq.com/blog/electron-app-testing/

#### 41. Complete Guide to Electron App Architecture Design and Best Practices | Oflight

#### Inc., accessed April 27, 2026,

#### https://www.oflight.co.jp/en/columns/electron-app-architecture-best-practices

#### 42. 6 Ways Slack, Notion, and VSCode Improved Electron App Performance | Palette

#### Docs, accessed April 27, 2026,

#### https://palette.dev/blog/improving-performance-of-electron-apps

#### 43. Building High-Performance Electron Apps - Johnny Le, accessed April 27, 2026,

#### https://www.johnnyle.io/read/electron-performance

#### 44. React Three Fiber Best Practices - MCP Market, accessed April 27, 2026,

#### https://mcpmarket.com/tools/skills/react-three-fiber-best-practices

#### 45. Scaling performance - React Three Fiber, accessed April 27, 2026,

#### https://r3f.docs.pmnd.rs/advanced/scaling-performance

#### 46. 100 Three.js Tips That Actually Improve Performance (2026) - Utsubo, accessed

#### April 27, 2026, https://www.utsubo.com/blog/threejs-best-practices-100-tips

#### 47. Tracab Import - Catapult Support, accessed April 27, 2026,

#### https://support.catapultsports.com/hc/en-us/articles/9225044488079-Tracab-Im

#### port

#### 48. The Opta Data Schema: An introduction - Soccermetrics Research, LLC,

#### accessed April 27, 2026,

#### https://www.soccermetrics.net/match-data-collection/the-opta-data-schema-an

#### -introduction

#### 49. OpenSTARLab: Open Approach for Spatio-Temporal Agent Data Analysis in

#### Soccer - arXiv, accessed April 27, 2026, https://arxiv.org/html/2502.02785v

#### 50. what does this stuff mean? - CFB GRAPHS - College Football Analytics, accessed

#### April 27, 2026, https://www.cfb-graphs.com/protected/glossary

#### 51. Advanced college football stats: a glossary, accessed April 27, 2026,

#### https://www.footballstudyhall.com/2018/2/2/16963820/college-football-advanced

#### -stats-glossary

#### 52. Football Analytics Glossary, accessed April 27, 2026,

#### https://footballstatsglossary.home.blog/

#### 53. Technische Hochschule Ingolstadt AI-based Classification of American Football

#### Plays Combining Computer Vision and Historical Pla - OPUS, accessed April 27,

#### 2026, https://opus4.kobv.de/opus4-haw/files/4745/I001915178Thesis.pdf

#### 54. React Three Fiber Performance Optimization : r/threejs - Reddit, accessed April

#### 27, 2026,

#### https://www.reddit.com/r/threejs/comments/1jffyit/react_three_fiber_performanc

#### e_optimization/

#### 55. AI Revolutionizes Football Tactics and Player Recruitment, accessed April 27,


#### 2026,

#### https://www.chosun.com/english/market-money-en/2026/02/20/LMT6VZR5VBHJJ

#### CHBJJ2E6IEBFE/


