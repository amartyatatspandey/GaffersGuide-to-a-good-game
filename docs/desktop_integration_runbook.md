# Desktop Integration Runbook

> Gaffer's Guide — `desktop-app` + `backend` integration guide.

---

## Environment Matrix

| Mode | Frontend | Backend | Overlay/Media | Engine selection |
|------|----------|---------|---------------|-----------------|
| **Local dev** | `npm run dev:desktop` (Next.js + Electron) | `uvicorn main:app` on port 8000 | `http://127.0.0.1:8000/api/v1beta/…` | Controlled via Engine Settings modal in UI |
| **Integration script** | `bash scripts/run_integration_desktop.sh` | Auto-started by script | Same as local dev | Same as local dev |
| **Packaged Electron** | Electron starts `npm run start:web` | Must be running separately or embedded | `NEXT_PUBLIC_BACKEND_URL` env var | Same modal |
| **Deployed cloud backend** | `NEXT_PUBLIC_BACKEND_URL=https://…` build | Cloud Run deployment | `https://…/api/v1beta/…` | Same modal; cloud engine required for Cloud CV |

---

## Required Environment Variables

### Backend (`backend/`)

| Variable | Required for | Description |
|----------|-------------|-------------|
| `MODAL_WEBHOOK_URL` | Cloud CV engine | URL to Modal inference webhook |
| `MODAL_API_KEY` | Cloud CV engine | Bearer token for Modal webhook |
| `GEMINI_API_KEY` | Cloud LLM | Google Gemini API key |
| `LLM_API_KEY` / `OPENAI_API_KEY` | Cloud LLM (OpenAI-compatible) | API key |
| `OLLAMA_BASE_URL` | Local LLM | Ollama server URL (default: `http://localhost:11434`) |
| `CORS_ALLOW_ORIGINS` | Production | Comma-separated allowed origins |

### Desktop (`desktop-app/`)

| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_BACKEND_URL` | All modes | Backend FastAPI base URL. Default: `http://127.0.0.1:8000` |
| `NEXT_PORT` | Dev | Next.js bind port. Default: auto-selected 3000–3010 |
| `NEXT_HOST` | Dev | Next.js bind host. Default: `127.0.0.1` |

> Copy `desktop-app/.env.example` to `desktop-app/.env.local` and fill in for local dev.

---

## Startup Sequence (Local Dev)

```
bash scripts/run_integration_desktop.sh
```

The script will:
1. Check that `BACKEND_PORT` (default 8000) is free; exit if not.
2. Auto-select a free `NEXT_PORT` between 3000–3010.
3. Start `uvicorn main:app` in the background.
4. Export `NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:${BACKEND_PORT}`.
5. Wait up to 30 s for the backend to accept connections.
6. Start `npm run dev:desktop` (Next.js + Electron via `concurrently`).

### Manual startup

```bash
# Terminal 1 — backend
cd backend
source .venv/bin/activate
BACKEND_PORT=8000 uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 — desktop
cd desktop-app
NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000 \
NEXT_PORT=3000 \
npm run dev:desktop
```

---

## Engine Selection Flow

```
EngineSettingsModal (UI)
       │ sets
       ▼
EngineContext (React context — shared across workspace)
  cvEngine: "local" | "cloud"
  llmEngine: "local" | "cloud"
       │ passed via options
       ▼
Hopper.handleFileChange → startProcessing(file, { cvEngine, llmEngine })
       │ calls
       ▼
useWebSocketProgress.startProcessing(file, options)
       │ calls
       ▼
createBetaJob(file, { cvEngine, llmEngine })
       │ POST /api/v1beta/jobs
       │   cv_engine=local|cloud
       │   llm_engine=local|cloud
       ▼
backend CVRouter → LocalCVRunner | CloudCVRunner
```

**Default engine: `local / local`** — safe to use offline. No cloud credentials needed.

---

## API Contract Surface

### Job lifecycle (beta path — primary desktop flow)

| Route | Method | Description |
|-------|--------|-------------|
| `/api/v1beta/jobs` | POST | Create job; returns `{ job_id, status, cv_engine, llm_engine }` |
| `/api/v1beta/jobs/{id}` | GET | Job record with status, current_step, paths, error |
| `/api/v1beta/jobs/{id}/artifacts` | GET | Typed artifact paths (report, overlay, tracking) |
| `/api/v1beta/jobs/{id}/overlay` | GET | Streams tracking overlay `.mp4` |
| `/api/v1beta/jobs/{id}/tracking` | GET | Tracking data JSON |
| `/ws/v1beta/jobs/{id}` | WS | Progress stream; status: `pending|processing|done|error` |
| `/api/v1/llm/local/preflight` | GET | Local Ollama readiness check (`daemon`, `model`, `generation`) |

### Coaching (resolves both v1 and v1beta job IDs)

| Route | Method | Description |
|-------|--------|-------------|
| `/api/v1/coach/advice` | GET | Advice items for a job (`?job_id=`, `?skip_llm=true`) |
| `/api/v1/chat` | POST | Follow-up chat; `{ job_id, message, llm_engine? }` |

---

## Known-Failure Triage

### `CLOUD_CV_NOT_CONFIGURED` error

**Cause:** `cv_engine=cloud` was sent but `MODAL_WEBHOOK_URL` is not set on the backend.

**Fix:** Either:
- Switch Engine Settings modal to **Local Engine** (default), or
- Set `MODAL_WEBHOOK_URL` and `MODAL_API_KEY` in the backend environment.

### Backend unreachable banner in Hopper

**Cause:** The frontend cannot reach `NEXT_PUBLIC_BACKEND_URL/health`.

**Fix:**
1. Ensure backend is running: `ps aux | grep uvicorn`
2. Verify `NEXT_PUBLIC_BACKEND_URL` matches the actual backend port.
3. Check CORS if running against a remote backend.

### `TypeError: BetaJobRecord.__init__()` on backend startup

**Cause:** `beta_jobs_store.json` contains keys from an older schema version.

**Fix:** The store now auto-skips malformed records. If it starts with an empty store, old jobs are lost — this is by design for schema drift. To migrate, delete `backend/output/beta_jobs_store.json`.

### Port `EADDRINUSE` on startup

**Cause:** Another process is already using port 3000 or 8000.

**Fix:** `run_integration_desktop.sh` auto-selects a free Next.js port (3000–3010). For the backend, set `BACKEND_PORT=<free port>` before running the script.

---

## Verification Checklist

Run before any PR that touches `api.ts`, workspace hooks/components, backend routing, queue/store, or startup scripts:

- [ ] `pytest backend/tests/ -q` — all tests pass
- [ ] `ruff check backend/` — no errors
- [ ] `cd desktop-app && npx tsc --noEmit` — no TypeScript errors
- [ ] `bash scripts/run_integration_desktop.sh` — app starts without errors
- [ ] Upload a short `.mp4` — progress bar moves, overlay appears, advice populates
- [ ] Open Engine Settings, switch to **Local Engine** — upload triggers local pipeline, no cloud error
- [ ] Open Engine Settings, switch to **Cloud Engine** without configuring credentials — see pre-flight warning (or cloud error on submit, not crash)
- [ ] Chat input with a completed job ID — response comes back
- [ ] Kill backend while app is open — Hopper shows "Backend unreachable" banner with Retry button
- [ ] Restart backend, click Retry — banner clears

---

## Release Checklist for Integration-Sensitive Changes

Changes to any of the following files require running the full verification checklist above before merging:

- `desktop-app/src/lib/api.ts`
- `desktop-app/src/context/EngineContext.tsx`
- `desktop-app/src/hooks/useWebSocketProgress.ts`
- `desktop-app/src/app/workspace/components/Hopper.tsx`
- `desktop-app/src/app/workspace/components/EngineSettingsModal.tsx`
- `desktop-app/src/app/workspace/components/VideoHUD.tsx`
- `desktop-app/src/app/workspace/components/TacticalDashboard.tsx`
- `backend/main.py` (route changes)
- `backend/models.py`
- `backend/services/beta_job_store.py`
- `backend/services/beta_queue.py`
- `scripts/run_integration_desktop.sh`
- `desktop-app/electron/main.cjs`
- `desktop-app/electron/preload.cjs`
