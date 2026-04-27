# Gaffer's Guide to a Good Game

## 1. Overview
Gaffer's Guide is an automated computer vision pipeline and tactical intelligence platform designed for sports video analysis. The system processes raw football (soccer) footage to automatically extract player tracking data, calculate tactical metrics, and generate fully annotated visualizations. 

## 2. Problem Statement
Analyzing football matches manually is a time-consuming, repetitive process that is highly susceptible to human error. Coaches and sports analysts require precise, actionable insights—such as player positioning, team formations, and movement heatmaps—from standard broadcast or single-camera footage. This project bridges the gap between raw video feeds and structured tactical data through an automated, end-to-end computer vision pipeline.

## 3. System Architecture
The system is built as a robust, multi-stage data processing pipeline:
- **Input:** Ingestion of raw sports video footage (MP4, AVI) via the Command Line Interface (CLI).
- **Processing:** Video frames are extracted, resized, and optionally sliced based on the selected quality profile.
- **Detection:** State-of-the-art computer vision models (YOLO) detect players, referees, and the ball in each frame.
- **Tracking:** Detected entities are assigned unique IDs and tracked continuously across frames using advanced tracking algorithms.
- **Analysis:** Spatial data is transformed into tactical metrics, calculating team formations, player zones, and match events.
- **Output:** The pipeline produces structured tracking data, tactical metrics in JSON format, and a final rendered video overlay.

## 4. Pipeline Flow
The step-by-step execution flow of the system operates as follows:
1. **Initialization:** The user submits a video file via the CLI, specifying an output directory and a quality profile.
2. **Configuration Resolution:** The system configures the runtime parameters (e.g., resolution, batch size, SAHI activation) based on the chosen profile.
3. **Inference Execution:** The video is processed sequentially or batched frame-by-frame. Players and the ball are detected and tracked.
4. **Metric Generation:** The raw bounding box coordinates are passed to the analytics engine to derive higher-level tactical insights.
5. **Rendering & Export:** The system overlays the tracking data onto the original video and exports the JSON telemetry files.

## 5. Key Features
- **Modular Pipeline:** Independent stages for detection, tracking, and analytics, allowing for easy updates and maintenance.
- **CLI-Based Execution:** A robust Command Line Interface for headless operation and automation.
- **Configurable Quality Profiles:** Dynamic runtime profiles that trade off inference speed against detection precision.
- **Scalable Processing:** Capable of running on local hardware or being scaled in a cloud-native environment.

## 6. Quality Profiles
The system introduces a dynamic profile configuration to meet diverse processing needs:
- `fast`: Optimized for maximum throughput and large batch jobs. It utilizes reduced resolutions for quick previews.
- `balanced`: The default profile. It offers an optimal middle ground, delivering strong accuracy at a reasonable processing speed.
- `high_res`: Operates at full resolution. Best suited for high-detail analysis and final deliverable rendering where precision is paramount.
- `sahi`: Utilizes Slicing Aided Hyper Inference (SAHI). It is designed for crowded scenes and small object detection (like the ball), offering the highest accuracy at the cost of the slowest runtime.

## 7. Installation
Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/amartyatatspandey/GaffersGuide-to-a-good-game.git
cd GaffersGuide-to-a-good-game

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install pipeline dependencies
pip install -r backend/requirements.txt
```

## 8. Environment Setup
To ensure the pipeline correctly resolves its internal modules, configure the Python path before execution. It is recommended to add this to your `.env` or shell profile:

```bash
export PYTHONPATH=backend:src      # macOS/Linux
# set PYTHONPATH=backend;src       # Windows
```

## 9. Usage (CLI)
The primary entry point for the pipeline is the CLI. Run the pipeline with the following command:

```bash
python -m gaffers_guide.cli run \
  --video "<path_to_video_file>" \
  --output "<output_directory>" \
  --quality-profile <profile_name>
```

**Arguments:**
- `--video` (Required): The relative or absolute path to the input video file.
- `--output` (Required): The directory where the final video and JSON files will be saved.
- `--quality-profile` (Required): Select one of `fast`, `balanced`, `high_res`, or `sahi`.

## 10. Output
Upon completion, the system generates the following files in the specified output directory:
- `*_tracking_data.json`: Frame-by-frame spatial coordinates and tracking IDs for every detected entity.
- `*_tactical_metrics.json`: Derived analytical insights, including calculated formations, zones, and events.
- `*.mp4`: The original video overlaid with high-quality bounding boxes and tracking visualizations.

## 11. Performance Notes
- **When to use `fast` vs `sahi`:** Use the `fast` profile for quick sanity checks or large batch processing. Reserve `sahi` strictly for complex, densely packed footage where ball tracking is failing on lower profiles.
- **Hardware Considerations:** Profiles like `high_res` and `sahi` require significant GPU VRAM. For optimal performance, execution on a dedicated CUDA-enabled GPU or high-tier cloud instance is highly recommended.

## 12. Tech Stack
The project is built on a modern, high-performance ecosystem:
- **Python (3.11+)**: Core language for pipeline orchestration.
- **OpenCV**: Video frame extraction, resizing, and rendering.
- **Ultralytics (YOLO)**: State-of-the-art object detection models.
- **Supervision**: Advanced tracking algorithms and visualization utilities.
- **SAHI**: Slicing Aided Hyper Inference for detecting highly compact or small objects.
- **Scikit-learn & Pandas**: Data manipulation and clustering for tactical metric calculation.

## 13. Future Improvements
- **Real-Time Processing:** Optimizing the pipeline to process live camera feeds with sub-second latency for in-match coaching.
- **Scalability Enhancements:** Expanding cloud-native deployment via Docker to orchestrate massively parallel batch processing across Kubernetes clusters.
- **Advanced UI/Dashboard:** Fully integrating the standalone Electron desktop application to ingest and visualize the generated 3D tracking data interactively.
