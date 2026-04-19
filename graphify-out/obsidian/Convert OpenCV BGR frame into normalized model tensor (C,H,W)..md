---
source_file: "backend/scripts/auxiliary_tools/generate_calibration.py"
type: "rationale"
community: "Community 14"
location: "L116"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Community_14
---

# Convert OpenCV BGR frame into normalized model tensor (C,H,W).

## Connections
- [[LitSoccerFieldSegmentation]] - `uses` [INFERRED]
- [[frame_to_model_tensor()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Community_14