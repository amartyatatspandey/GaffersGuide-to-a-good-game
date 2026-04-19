---
source_file: "backend/services/beta_queue.py"
type: "rationale"
community: "Community 3"
location: "L25"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_3
---

# Queue-backed execution model for beta job isolation.

## Connections
- [[BetaJobStore]] - `uses` [INFERRED]
- [[BetaPipelineQueue]] - `rationale_for` [EXTRACTED]
- [[CVRouterFactory]] - `uses` [INFERRED]
- [[EngineRoutingError]] - `uses` [INFERRED]
- [[PipelineMetricsRegistry]] - `uses` [INFERRED]

#graphify/rationale #graphify/INFERRED #community/Community_3