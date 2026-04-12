# Gaffer's Guide: Desktop Tactical Workspace

Welcome to the internal software workspace for **Gaffer's Guide**. While the root directory powers our public-facing cinematic marketing website, this directory (`/src/app/workspace`) houses the actual high-performance tactical analysis software.

## Architecture & Features

This workspace is designed to be an immersive, "Dark Room" environment (`bg-[#0a0f0a]`) for football coaches and analysts.

*   **Dynamic Layout Architecture**: Features a state-driven collapse system for the Main Sidebar and Match Timeline to maximize screen real estate during film study.
*   **60/40 Split Dashboards**: The bottom Dashboard area splits the workspace horizontally, placing detailed tactical analysis cards on the left (60%) and a professional, Gemini-style conversational AI interface on the right (40%).
*   **Synchronized Video HUD**: The `VideoHUD` component renders side-by-side visualizers: raw source match feed alongside a synchronized 2D telemetry radar.
*   **Hybrid Engine Configuration**: The `EngineSettingsModal` implements a 4-mode toggle system allowing users to process models via Cloud or Local hardware, offline or online.

## Technical Implementation

*   **Framework**: Next.js / React (TypeScript)
*   **Styling**: Tailwind CSS (Strict dark mode aesthetics with `emerald-400` accents)
*   **Target Compilation**: This workspace is currently built to run inside a Chromium browser shell, with native Electron/Tauri IPC integration planned for the final desktop `.exe` build.

## Developer Quickstart

To run the software workspace locally, start the development server from the root of the project:

```bash
npm run dev
```

Navigate to `http://localhost:3000/workspace` to interact with the dashboard.
