# ⚽ Gaffer's Guide to a Good Game

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO-FF6B35?style=for-the-badge&logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**An end-to-end automated computer vision pipeline for football tactical intelligence.**  
Turn raw match footage into structured tracking data, tactical metrics, and annotated video — fully automated.

[Getting Started](#7-installation) • [Usage](#9-usage-cli) • [Quality Profiles](#6-quality-profiles) • [Tech Stack](#12-tech-stack) • [Contributing](#contributing)

</div>

---

## 📌 Table of Contents

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
- 🎯 **Frame-by-frame player & ball tracking**
- 📊 **Tactical metrics** (formations, zones, heatmaps)
- 🎬 **Annotated video overlays**
- 📁 **Structured JSON telemetry**

All fully automated — no manual annotation required.

---

## 2. Problem Statement

Analyzing football matches manually is:
- ⏱️ **Extremely time-consuming** — hours of review per match
- ❌ **Error-prone** — human fatigue affects consistency
- 💰 **Expensive** — requires dedicated analysis staff

Coaches and analysts need precise, actionable insights — player positioning, team formations, movement heatmaps — from standard footage.

**Gaffer's Guide bridges the gap between raw video and structured tactical intelligence through a fully automated pipeline.**

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GAFFER'S GUIDE                           │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  INPUT   │───▶│  DETECT  │───▶│  TRACK   │───▶│ ANALYSE  │  │
│  │  Video   │    │  YOLO    │    │ByteTrack │    │ Tactics  │  │
│  │  MP4/AVI │    │  + SAHI  │    │  + IDs   │    │ Metrics  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                       │         │
│  ┌──────────────────────────────────────────────────┐│         │
│  │                    OUTPUT                        ││         │
│  │  📹 Annotated Video  📊 JSON Metrics  📁 Tracking││◀────────┘
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐          │
│  │           QUALITY PROFILE SYSTEM                 │          │
│  │   fast │ balanced │ high_res │ sahi               │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

**Components:**
- **Input Layer** — CLI-driven ingestion of MP4/AVI footage
- **Detection Layer** — YOLO-based player, referee, and ball detection with optional SAHI tiling
- **Tracking Layer** — ByteTrack multi-object tracking with persistent IDs
- **Analytics Layer** — Spatial coordinate transformation, formation detection, zone calculation
- **Output Layer** — JSON telemetry export + rendered video overlay

---

## 4. Pipeline Flow

```
User runs CLI
      │
      ▼
Quality Profile resolved (fast / balanced / high_res / sahi)
      │
      ▼
ProfileConfig → imgsz, conf_threshold, sahi_enabled, slice_size
      │
      ▼
Video frames extracted and batched
      │
      ▼
YOLO model inference (with profile imgsz + conf)
      │
      ├──▶ [SAHI enabled?] → Slice region → Batch infer slices → Fuse candidates
      │
      ▼
Ball candidate ranking (temporal prior + confidence scoring)
      │
      ▼
Pitch ROI masking → Homography projection → 2D radar coordinates
      │
      ▼
Team classification → Formation detection → Tactical metric calculation
      │
      ▼
Output: JSON tracking + JSON metrics + annotated MP4
```

---

## 5. Key Features

| Feature | Description |
|---|---|
| 🔍 **SAHI Ball Detection** | Slicing Aided Hyper Inference for precise small-object detection |
| ⚙️ **Quality Profile System** | 4 runtime profiles trading speed vs. accuracy |
| 🎯 **Temporal Ball Prior** | Adaptive search windows based on last known ball position |
| 🟩 **Pitch ROI Masking** | HSV-based green detection to focus inference on the pitch |
| 🏃 **ByteTrack Integration** | Robust multi-object tracking with persistent player IDs |
| 🗺️ **Homography Projection** | Camera-to-2D radar coordinate transformation |
| 📦 **CLI-First Design** | Headless operation for automation and cloud deployment |
| ☁️ **Cloud-Native Ready** | Docker + cloud infrastructure for scalable batch processing |
| 🧪 **Fully Tested** | Profile resolution, CLI parsing, and pipeline integration tests |

---

## 6. Quality Profiles

The profile system is the **core innovation** of this release. One flag controls the entire runtime behavior.

```bash
--quality-profile fast        # Speed priority
--quality-profile balanced    # Default — best all-rounder
--quality-profile high_res    # Quality priority
--quality-profile sahi        # Maximum recall
```

| Profile | `imgsz` | `conf` | SAHI | Frame Skip | Best For |
|---|---|---|---|---|---|
| `fast` | 480 | 0.35 | ❌ | Every 3rd | Quick previews, large batch jobs |
| `balanced` | 640 | 0.25 | ❌ | None | Standard full-match analysis |
| `high_res` | 1280 | 0.20 | ❌ | None | High-detail QA and final renders |
| `sahi` | 1280 | 0.20 | ✅ | None | Complex scenes, maximum ball recall |

> **How profiles work:** Each profile maps directly to `imgsz` and `conf` passed into the YOLO model inference call, plus SAHI slice configuration for the detection wrapper. The entire pipeline behavior changes from a single CLI flag.