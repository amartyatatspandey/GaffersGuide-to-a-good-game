---
source_file: "backend/main.py"
type: "rationale"
community: "Community 3"
location: "L721"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_3
---

# Stream job progress updates over WebSockets.      Sends JSON messages shaped lik

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
- [[job_progress_ws()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_3