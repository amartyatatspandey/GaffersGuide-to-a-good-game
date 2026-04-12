Gaffer's Guide: Development Constitution & AI Protocol
Version: 2.0 (Industry Grade)
Status: MANDATORY for all AI-Generated Code

1. The "Golden Context" Rule
Before generating any code, the AI Agent must be aware of:
Project_Vision.md: The product roadmap and core logic.
Development_Protocol.md: This document.
Prompt to start every session:
"You are the Lead Engineer for Gaffer's Guide. Read Project_Vision.md and Development_Protocol.md. Confirm you understand the architecture, tech stack, and security constraints before we begin."

2. The Tech Stack (Non-Negotiable)
To ensure compatibility between Amartya's backend code and Manasi's frontend code, we enforce strict stack rules.
Language: Python 3.11+ (Backend), TypeScript (Frontend).
Frameworks:
API: FastAPI (Async is mandatory). Do not use Flask.
Frontend: React 18 + Vite + Tailwind CSS.
Data: Pydantic Models (Strict typing).
Infrastructure: Google Cloud Run (Serverless).
Database: Firestore (No SQL) or BigQuery (Analytics).

3. Security & "Vibe" Constraints
AI loves to hardcode secrets for convenience. This is forbidden in production.
Rule 1: NEVER hardcode API Keys (Gemini, SkillCorner, GCP).
Bad: api_key = "AIzaSy..."
Good: api_key = os.getenv("GEMINI_API_KEY")
Rule 2: All data validation must happen before processing. Use Pydantic schemas.
Rule 3: No local file paths (e.g., C:/Users/Amartya/...). Use relative paths or Cloud Storage buckets.

4. Coding Standards for AI
When asking the AI to write code, explicitly demand:
Type Hinting: All Python functions must have type hints. def calculate_speed(dist: float, time: float) -> float:
Docstrings: Google-style docstrings for every function explaining Why, not just What.
Statelessness: Functions should not rely on global variables. They must accept inputs and return outputs.

5. The "Review" Step
Vibecoding generates code fast, but often creates hallucinations.
Human Check: Does this code import libraries we haven't installed?
Docker Check: Will this run in a Linux container? (Avoid Windows-specific OS calls).

6. Directory Structure (Enforced)
Ensure the AI places files in this structure:
/gaffers-guide-root
├── /backend
│   ├── /app
│   │   ├── main.py          # FastAPI Entry Point
│   │   ├── /routers         # API Endpoints
│   │   ├── /services        # Core Logic (Model 1, 2, 3)
│   │   └── /schemas         # Pydantic Models
│   ├── Dockerfile
│   └── requirements.txt
├── /frontend
│   ├── /src
│   │   ├── /components      # React Components
│   │   └── /hooks           # Custom Logic
│   └── Dockerfile
├── /infra                   # Terraform / Cloud Build configs
├── .env.example             # Template for secrets
└── Development_Protocol.md  # THIS FILE


7. Local LLM (Ollama)

- Backend uses `OLLAMA_BASE_URL` and `OLLAMA_MODEL` (see `backend/.env.example`).
- Request-time auto-start: if the `ollama` binary is on PATH and `/api/tags` fails, the API spawns `ollama serve` and retries when **not** on Cloud Run and `OLLAMA_AUTO_START` is unset (default on laptops). Set `OLLAMA_AUTO_START=0` to disable. On Cloud Run (`K_SERVICE`), default is off unless `OLLAMA_AUTO_START_IN_CLOUD=1`.
- Optional `OLLAMA_MANAGED_LIFECYCLE=1` or `GAFFERS_DEFAULT_LLM_ENGINE=local`: on FastAPI startup the backend starts `ollama serve` if nothing is listening yet, and on shutdown it stops **only** that child process (not an Ollama you started yourself). Optional `OLLAMA_PULL_ON_START=1` runs `ollama pull` for `OLLAMA_MODEL` after the daemon is up (can be slow).

8. CV pipeline homography (SoccerNet sn-calibration)

Local CV (`run_e2e_cloud`, API jobs) expects per-upload homography at
`backend/output/{video_stem}_homographies.json`, or `GAFFERS_HOMOGRAPHY_JSON` pointing at a JSON file.

Auto-generation during a job requires:

- Clone https://github.com/SoccerNet/sn-calibration into `backend/references/sn-calibration`.
- Populate `backend/references/sn-calibration/resources/` per the upstream README (calibration weights).
- Leave `GAFFERS_HOMOGRAPHY_JSON` unset for normal per-upload files; set it only to override with an existing file.

Verify layout from `backend/`: `python scripts/verify_sn_calibration.py` (add `--try-import` to test `DynamicPitchCalibrator`).

Docker / Cloud Run: the default image copies the repo root; if `references/sn-calibration` is not in the build context, mount it or bake it into the image, otherwise homography auto-generation will fail the same way as on a bare laptop.

