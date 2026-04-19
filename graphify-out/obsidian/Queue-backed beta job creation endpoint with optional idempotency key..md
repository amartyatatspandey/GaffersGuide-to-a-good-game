---
source_file: "backend/main.py"
type: "rationale"
community: "Community 3"
location: "L438"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_3
---

# Queue-backed beta job creation endpoint with optional idempotency key.

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
- [[create_beta_job()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_3