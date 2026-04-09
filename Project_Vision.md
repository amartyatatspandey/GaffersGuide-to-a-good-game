# Gaffer's Guide – Project Vision

**Version:** 1.0  
**Status:** Living document

## Why

Gaffer's Guide is a **tactical intelligence platform** for football: it uses computer vision and analytics to support coaches and analysts with automated insights from broadcast and single-camera footage.

## What (Core Capabilities)

1. **Camera calibration** – Estimate camera parameters (homography / intrinsics) from pitch views for spatial analysis.
2. **Tracking** – Detect and track players, referees, and the ball (YOLO + supervision) for heatmaps, runs, and tactical metrics.
3. **API-first** – All logic exposed via FastAPI so the React frontend and external tools can consume the same services.
4. **Cloud-native** – Designed for Google Cloud Run; data and secrets via env and Cloud Storage.

## Tech Stack (Summary)

- **Backend:** Python 3.11+, FastAPI (async), Pydantic, PyTorch for CV models.
- **Data:** SoccerNet (calibration + tracking), YOLO, supervision.
- **Frontend:** React 18 + Vite + Tailwind (per Development_Protocol).
- **Infra:** Google Cloud Run, Firestore / BigQuery, no hardcoded paths or keys.

## Roadmap (High Level)

1. Data pipelines: SoccerNet calibration + tracking loaders and PyTorch datasets.
2. FastAPI app: health, dataset info, and inference endpoints.
3. Calibration and tracking models integrated behind the API.
4. Frontend dashboards consuming the API.
5. Deployment via Docker and Cloud Build.

See **Development_Protocol.md** for coding standards, security, and directory layout.
