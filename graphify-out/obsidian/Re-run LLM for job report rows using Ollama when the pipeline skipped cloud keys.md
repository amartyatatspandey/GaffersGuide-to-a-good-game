---
source_file: "backend/main.py"
type: "rationale"
community: "Community 3"
location: "L870"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_3
---

# Re-run LLM for job report rows using Ollama when the pipeline skipped cloud keys

## Connections
- [[BetaJobRecord]] - `uses` [INFERRED]
- [[BetaJobStore]] - `uses` [INFERRED]
- [[BetaPipelineQueue]] - `uses` [INFERRED]
- [[BetaQueueItem]] - `uses` [INFERRED]
- [[CVRouterFactory]] - `uses` [INFERRED]
- [[ChatRequest]] - `uses` [INFERRED]
- [[ChatResponse]] - `uses` [INFERRED]
- [[CreateJobResponse]] - `uses` [INFERRED]
- [[DatasetInfo]] - `uses` [INFERRED]
- [[DatasetsListResponse]] - `uses` [INFERRED]
- [[EngineRoutingError]] - `uses` [INFERRED]
- [[PipelineMetricsRegistry]] - `uses` [INFERRED]
- [[ReportEntry]] - `uses` [INFERRED]
- [[ReportsResponse]] - `uses` [INFERRED]
- [[_refresh_job_report_cards_with_local_llm()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_3