# Reproducing the final submissions (team mangojump)

One frozen system, one Docker image, two selected final picks that differ only
in an inference-time flag (confirmed acceptable by the organizers).

## Final pick mapping

| Kaggle submission | Command | Public LB | Output checksum* |
|---|---|---|---|
| `final_routed.csv` (Pick 1, expected ranking pick) | `docker run --network none -v <images>:/data:ro -v <out>:/submissions -e VARIANT=routed freuid-mangojump` | 0.00207 | `<md5, filled at freeze>` |
| `final_plain.csv` (Pick 2, router-off ablation) | same with `-e VARIANT=plain` | 0.00207 | `<md5, filled at freeze>` |

Add `--gpus all` if the host has `nvidia-container-toolkit` configured (large
speedup; the image auto-detects CUDA vs. CPU — see Hardware/Throughput below).

\* Checksums are from our canonical run (RTX 5060 Ti 16GB, torch
2.11.0.dev20251223+cu128). GPU inference is not bit-portable across
hardware/kernel versions; expected reproduction tolerance on other GPUs:
per-row mean |Δ| ≈ 3e-4, decision flips ≤ 0.04 % of rows, leaderboard impact
< ±0.0002. Same-machine reruns are deterministic at ~1e-7.

## Build (network required once)

```bash
# no HF account/token needed: all four backbones are ungated timm mirrors
docker build -t freuid-mangojump .
```

## Run (no network)

Input: a flat directory of images (`.jpeg/.jpg/.png/.webp/.bmp/.tif/.tiff`),
row id = filename without extension. Output: `/submissions/submission.csv`
with columns `id,label` (fraud score in [0,1]).

Throughput: ≈ 8 min / 1k images on an RTX 5060 Ti (16 GB); the 134,997-image
private set takes ≈ 19 h. `VARIANT=plain` skips one backbone (~15 % faster).

## Frozen weights

Dual-hosted, identical bytes (sha256-verified):

- **Hugging Face** — what `docker build` actually fetches, via
  `download_weights.py`, at the pinned revision:
  https://huggingface.co/ching0206/freuid-2026-mangojump
  @ `a36f036aba49ede6890761c927fae8f1951922c9`.
- **This repository**, via Git LFS: `artifacts/system/` — a convenience
  mirror for browsing or offline use; not read by the Docker build itself
  (excluded from the build context by `.dockerignore`).

Verify the two match:
`python docker/verify_hf_upload.py ching0206/freuid-2026-mangojump`.

## What is frozen where

| Artifact | Path | Notes |
|---|---|---|
| DoRA adapters (7 members) | `artifacts/system/adapters_slim/*.pt` | EMA weights; ~55–115 MB each |
| Fusion / capture / PAD heads | `artifacts/system/heads.pt` | linear heads, trained pre-freeze |
| FGTS token indices | `artifacts/system/fisher_idx.npz` | frozen; never recomputed at inference |
| kNN router reference + thresholds | `artifacts/system/knn_ref.npz`, `config.json` | distance floor 0.246778, capture threshold 0.5 |
| Base backbones | HF cache baked into the image | DINOv3-L/H+ (Meta), SigLIP-2 SO400M (Apache-2.0), DFN5B (Apple ASCL) |

Training code for every member is under `src/` with per-run configs in
`experiments/<run_id>/config.json`; external data (DLC-2021, CC BY-SA 2.5;
SIDTD, CC BY-SA 4.0) is fetched by `src/data/` scripts and is not
redistributed in this repository or the image.
