# Gaffer's Guide – Infra

## Cloud Run (backend)

- **Build:** From repo root: `docker build -f backend/Dockerfile -t gcr.io/PROJECT_ID/gaffers-guide-api .`
- **Deploy:** Use Cloud Build: `gcloud builds submit --config=infra/cloudbuild.yaml .` (set `PROJECT_ID` and optionally `_REGION` in substitutions).
- **Env:** Configure env vars (e.g. `SOCCERNET_PASSWORD`) in Cloud Run → Edit & deploy → Variables. Use Secret Manager for secrets and reference in Cloud Run.
- No API keys or secrets in this repo.

## Cloud Run (secondary beta pipeline)

- **Build + Deploy:** `gcloud builds submit --config=infra/cloudbuild.beta.yaml .`
- **Default service:** `gaffers-guide-api-beta`
- **Purpose:** queue-backed beta path (`/api/v1beta/*`, `/ws/v1beta/*`) for staged rollout and SLO validation before primary adoption.
- **Scaling defaults:** bounded (`--max-instances=1`) to avoid inconsistent local-disk state while beta uses JSON + local artifacts.

## Frontend (optional)

Build static bundle from `frontend/` and serve via Cloud Storage + CDN or a second Cloud Run service (e.g. nginx serving `frontend/dist`). Set backend API URL via env at build time (`VITE_API_URL`).
