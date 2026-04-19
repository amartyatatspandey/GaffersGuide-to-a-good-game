---
source_file: "backend/services/ollama_client.py"
type: "code"
community: "Community 3"
location: "L132"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Community_3
---

# start_ollama_for_app_lifecycle()

## Connections
- [[If ``OLLAMA_MANAGED_LIFECYCLE=1``, ensure Ollama is up when the API boots.]] - `rationale_for` [EXTRACTED]
- [[_base_url()]] - `calls` [EXTRACTED]
- [[_env_truthy()]] - `calls` [EXTRACTED]
- [[_model_name()]] - `calls` [EXTRACTED]
- [[_ollama_executable()]] - `calls` [EXTRACTED]
- [[_popen_ollama_serve()]] - `calls` [EXTRACTED]
- [[_probe_tags()]] - `calls` [EXTRACTED]
- [[_should_manage_lifecycle()]] - `calls` [EXTRACTED]
- [[_startup_beta_queue()]] - `calls` [INFERRED]
- [[_terminate_managed_ollama_process()]] - `calls` [EXTRACTED]
- [[_timeout_seconds()]] - `calls` [EXTRACTED]
- [[_wait_for_ollama_after_spawn()]] - `calls` [EXTRACTED]
- [[ollama_client.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Community_3