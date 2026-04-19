---
source_file: "backend/references/sn-reid/torchreid/data/transforms.py"
type: "rationale"
community: "Community 9"
location: "L241"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_9
---

# Builds train and test transform functions.      Args:         height (int): targ

## Connections
- [[Compose]] - `uses` [INFERRED]
- [[Normalize]] - `uses` [INFERRED]
- [[RandomHorizontalFlip]] - `uses` [INFERRED]
- [[build_transforms()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_9