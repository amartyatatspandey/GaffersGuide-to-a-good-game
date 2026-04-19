---
source_file: "backend/main.py"
type: "rationale"
community: "Community 3"
location: "L938"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_3
---

# Run the tactical pipeline: metrics → triggers → RAG prompts → optional LLM compl

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
- [[get_coach_advice()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_3