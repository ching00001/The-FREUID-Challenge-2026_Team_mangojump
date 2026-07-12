---
license: mit
tags:
  - image-classification
  - document-fraud-detection
  - dora
  - freuid-2026
---

# FREUID 2026 — team mangojump frozen inference system

Frozen weights for our FREUID Challenge 2026 (IJCAI-ECAI) submission:
feature-level fusion of DoRA-adapted foundation backbones with an
OOD-routed presentation-attack (PAD) head. Code, Docker contract, and the
technical report live in the public repository (see the competition reply);
this repo holds only the weight artifacts consumed by `src/predict_docker.py`.

## Contents

| File | What it is |
|---|---|
| `adapters_slim/<member>.pt` | DoRA adapter deltas (rank 16, α 32) + EMA copies per member; base backbone weights are NOT included |
| `heads.pt` | fusion head, capture head, PAD head (linear, LayerNorm→Dropout→Linear) |
| `fisher_idx.npz` | frozen FGTS top-64 token indices for DINOv3 members |
| `knn_ref.npz` | block-normalized digital-train reference matrix (fp32) for the kNN router |
| `config.json` | member specs, routing thresholds (capture 0.5, distance floor 0.246778), variant map |

## Base models (fetch from original sources; only deltas are hosted here)

- DINOv3 ViT-L/16 & ViT-H+/16 (`timm/vit_{large,huge_plus}_patch16_dinov3.lvd1689m`) — Meta AI, DINOv3 license (gated)
- SigLIP-2 SO400M/16 @512 (`timm/vit_so400m_patch16_siglip_512.v2_webli`) — Apache-2.0
- DFN5B CLIP ViT-H/14 @378 (`timm/vit_huge_patch14_clip_378.dfn5b`) — Apple ASCL

## Training data of the adapters

FREUID 2026 competition data; two members additionally saw small fractions of
[DLC-2021](https://zenodo.org/) (CC BY-SA 2.5) and
[SIDTD](https://doi.org/10.34810/data1815) (CC BY-SA 4.0). Neither dataset is
redistributed here. Adapter deltas are released under MIT; base-model and
dataset licenses continue to apply to their respective artifacts.
