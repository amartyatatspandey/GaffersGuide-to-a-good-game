---
source_file: "backend/scripts/pipeline_core/run_e2e_cloud.py"
type: "rationale"
community: "Community 5"
location: "L90"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Community_5
---

# Sparse optical-flow estimator on downscaled grayscale frames.     Returns camera

## Connections
- [[DownscaledOpticalFlowEstimator]] - `rationale_for` [EXTRACTED]
- [[GlobalRefiner]] - `uses` [INFERRED]

#graphify/rationale #graphify/EXTRACTED #community/Community_5