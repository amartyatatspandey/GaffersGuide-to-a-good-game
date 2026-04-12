# Desktop Integration Runbook

This runbook documents the integrated desktop product setup on branch `integration_testing`.

## Components

- Backend API: `backend/main.py` (FastAPI, non-experiment pipeline)
- Desktop UI: `desktop-app` (Next.js workspace from `software_dev`)
- Desktop shell: Electron (`desktop-app/electron/main.cjs`, `desktop-app/electron/preload.cjs`)

## One-command local startup

From repository root:

```bash
./scripts/run_integration_desktop.sh
```

This starts:

1. `uvicorn main:app --host 127.0.0.1 --port 8000`
2. Next.js + Electron via `desktop-app` `npm run dev:desktop`

## Manual startup

Backend:

```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

Desktop app:

```bash
cd desktop-app
npm install
cp .env.example .env.local
npm run dev:desktop
```

## Environment variables

`desktop-app/.env.example`:

- `NEXT_PUBLIC_BACKEND_URL` default `http://127.0.0.1:8000`
- `NEXT_PORT` default `3000`
- `NEXT_HOST` default `127.0.0.1`

## Packaging smoke command

From `desktop-app`:

```bash
npm run build:web
npm run dist
```

## Integration flow checklist

1. Open app route `/workspace` in Electron.
2. Upload an `.mp4` from Hopper panel.
3. Verify websocket progress advances from pending to completed.
4. Confirm timeline cards populate from `/api/v1/coach/advice`.
5. Confirm Reports view loads records from `/api/v1/reports`.
6. Confirm chat replies through `/api/v1/chat`.
