---
source_file: "backend/services/errors.py"
type: "code"
community: "Community 3"
location: "L7"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Community_3
---

# EngineRoutingError

## Connections
- [[.__str__()_5]] - `method` [EXTRACTED]
- [[.run()_3]] - `calls` [INFERRED]
- [[.to_detail()]] - `method` [EXTRACTED]
- [[Asyncio entry point used by run_e2e_cloud.]] - `uses` [INFERRED]
- [[Auto-spawn ``ollama serve`` when the daemon is down and forking is allowed.]] - `uses` [INFERRED]
- [[Backward-compatible alias for existing callers.]] - `uses` [INFERRED]
- [[BetaPipelineQueue]] - `uses` [INFERRED]
- [[BetaQueueItem]] - `uses` [INFERRED]
- [[CVRouterFactory]] - `uses` [INFERRED]
- [[CVRunner]] - `uses` [INFERRED]
- [[CloudCVRunner]] - `uses` [INFERRED]
- [[CoachAdviceResponse]] - `uses` [INFERRED]
- [[CoachingAdviceItem]] - `uses` [INFERRED]
- [[Count regular files under ``root`` without walking unbounded trees.]] - `uses` [INFERRED]
- [[Create a new analytics job by uploading a match video.      The heavy CV→Math→Ru]] - `uses` [INFERRED]
- [[Exception]] - `inherits` [EXTRACTED]
- [[Factory-style LLM routing for tactical advice generation.]] - `uses` [INFERRED]
- [[Forking a local daemon is blocked on Cloud Run unless explicitly allowed.]] - `uses` [INFERRED]
- [[Frontend-ready payload after running the full pipeline.]] - `uses` [INFERRED]
- [[Generate coaching text using Gemini or OpenAI-compatible cloud APIs.]] - `uses` [INFERRED]
- [[Generate coaching text using local Ollama daemon.]] - `uses` [INFERRED]
- [[Generate follow-up coaching text.      If `job_id` is provided, include the job']] - `uses` [INFERRED]
- [[Generate tactical advice from local Ollama using a fixed analyst persona.]] - `uses` [INFERRED]
- [[If ``OLLAMA_MANAGED_LIFECYCLE=1``, ensure Ollama is up when the API boots.]] - `uses` [INFERRED]
- [[JobRecord]] - `uses` [INFERRED]
- [[List available job reports produced by the pipeline.      Scans `backendoutput]] - `uses` [INFERRED]
- [[List dataset folders (optional; used by some frontends).      Scans ``DATASETS_R]] - `uses` [INFERRED]
- [[Local Ollama completions for the CV→…→RAG E2E pipeline (job-time LLM).]] - `uses` [INFERRED]
- [[LocalCVRunner]] - `uses` [INFERRED]
- [[Poll apitags after spawning ``ollama serve``.]] - `uses` [INFERRED]
- [[Preflight-check that local Ollama daemon is reachable.      If connection fails]] - `uses` [INFERRED]
- [[Queue-backed beta job creation endpoint with optional idempotency key.]] - `uses` [INFERRED]
- [[Queue-backed execution model for beta job isolation.]] - `uses` [INFERRED]
- [[Re-run LLM for job report rows using Ollama when the pipeline skipped cloud keys]] - `uses` [INFERRED]
- [[Report whether local CV prerequisites are satisfied (weights + RAG library).]] - `uses` [INFERRED]
- [[Return (api_key, model, base_url) for OpenAI-compatible APIs.]] - `uses` [INFERRED]
- [[Return (content, error_message) using Google Gemini.]] - `uses` [INFERRED]
- [[Return (content, error_message).]] - `uses` [INFERRED]
- [[Return beta pipeline metrics snapshot for baseline and promotion gates.]] - `uses` [INFERRED]
- [[Return path to ``ollama`` CLI if on PATH, else None.]] - `uses` [INFERRED]
- [[Run Ollama completions for each synthesized prompt (mirrors cloud ``run_llm`` sh]] - `uses` [INFERRED]
- [[Run the tactical pipeline metrics → triggers → RAG prompts → optional LLM compl]] - `uses` [INFERRED]
- [[Single coaching recommendation for one flaw at one frame.]] - `uses` [INFERRED]
- [[Start ``ollama serve`` detached (request-time auto-start). Caller must verify th]] - `uses` [INFERRED]
- [[Start ``ollama serve`` on API startup and stop it on shutdown (this process only]] - `uses` [INFERRED]
- [[Stop Ollama only if this process started it via func`start_ollama_for_app_life]] - `uses` [INFERRED]
- [[Stream job progress updates over WebSockets.      Sends JSON messages shaped lik]] - `uses` [INFERRED]
- [[Structured service error to map into API error responses.]] - `rationale_for` [EXTRACTED]
- [[Tests for Ollama preflight, auto-start policy, and error codes.]] - `uses` [INFERRED]
- [[Tests for job-time local LLM completion helper (Ollama path).]] - `uses` [INFERRED]
- [[True when the card has a prompt but no successful coaching text yet.]] - `uses` [INFERRED]
- [[Understanding Image Retrieval Re-Ranking A Graph Neural Network Perspective]] - `uses` [INFERRED]
- [[Unset OLLAMA_AUTO_START on a non-Cloud host should still attempt spawn on connec]] - `uses` [INFERRED]
- [[_generate_cloud()]] - `calls` [INFERRED]
- [[_not_installed_error()]] - `calls` [INFERRED]
- [[_offline_error()]] - `calls` [INFERRED]
- [[_start_failed_error()]] - `calls` [INFERRED]
- [[errors.py]] - `contains` [EXTRACTED]
- [[generate_local_advice()]] - `calls` [INFERRED]
- [[get()]] - `calls` [INFERRED]
- [[get_tactical_advice()]] - `calls` [INFERRED]
- [[test_run_llm_local_engine_routing_error_on_card()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Community_3