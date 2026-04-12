# Gaffer's Guide - Frontend Package

This is the fully isolated, shippable Next.js frontend package for the Gaffer's Guide tactical analysis engine. It features the cinematic marketing website and the native-feeling desktop tactical workspace.

## Setup & Running

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Environment Variables**
   Copy `.env.example` to `.env.local` and set the connection string to the backend tracking engine (FastAPI).
   ```bash
   # Example for local development
   NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
   ```

3. **Start Development Server**
   ```bash
   npm run dev
   ```

## Architecture Notes
- This uses the **Next.js 16 App Router** (`app/`).
- The `workspace/` route connects to the external backend API (`/api/v1/jobs`) and a WebSocket endpoint to stream the telemetry generation pipeline visually.
- `lib/api/jobs.ts` handles the incoming match footage `FormData` ingestion and fetching the resulting `TrackingPayload` JSON artifacts.
- The 2D Telemetry Radar and Video UI sync via a highly optimized native `requestAnimationFrame` loop (`hooks/useVideoFrameSync.ts`).
