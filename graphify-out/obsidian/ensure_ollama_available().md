---
source_file: "backend/services/ollama_client.py"
type: "code"
community: "Community 3"
location: "L275"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Community_3
---

# ensure_ollama_available()

## Connections
- [[Preflight-check that local Ollama daemon is reachable.      If connection fails]] - `rationale_for` [EXTRACTED]
- [[_base_url()]] - `calls` [EXTRACTED]
- [[_not_installed_error()]] - `calls` [EXTRACTED]
- [[_offline_error()]] - `calls` [EXTRACTED]
- [[_ollama_executable()]] - `calls` [EXTRACTED]
- [[_probe_tags()]] - `calls` [EXTRACTED]
- [[_refresh_job_report_cards_with_local_llm()]] - `calls` [INFERRED]
- [[_should_attempt_auto_start()]] - `calls` [EXTRACTED]
- [[_spawn_ollama_serve()]] - `calls` [EXTRACTED]
- [[_timeout_seconds()]] - `calls` [EXTRACTED]
- [[_wait_for_ollama_after_spawn()]] - `calls` [EXTRACTED]
- [[generate_local_advice()]] - `calls` [EXTRACTED]
- [[get_coach_advice()]] - `calls` [INFERRED]
- [[ollama_client.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Community_3