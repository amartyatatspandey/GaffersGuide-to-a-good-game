---
source_file: "backend/references/sn-reid/torchreid/data/transforms.py"
type: "rationale"
community: "Community 9"
location: "L110"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_9
---

# Randomly alters the intensities of RGB channels.      Reference:         Krizhev

## Connections
- [[ColorAugmentation]] - `rationale_for` [EXTRACTED]
- [[Compose]] - `uses` [INFERRED]
- [[Normalize]] - `uses` [INFERRED]
- [[RandomHorizontalFlip]] - `uses` [INFERRED]

#graphify/rationale #graphify/INFERRED #community/Community_9