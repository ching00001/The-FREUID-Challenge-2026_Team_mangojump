# Reproducing the final submissions (team mangojump)

One frozen system, one Docker image, two selected final picks that differ only
in an inference-time flag (confirmed acceptable by the organizers).

## Final pick mapping

| Kaggle submission | Frozen weight combination | Command | Output SHA-256* |
|---|---|---|---|
| Pick 1: `final_routed.csv` | 5 base adapters + `dino_hplus_ds` PAD adapter | `docker run --network none -v <images>:/data:ro -v <out-routed>:/submissions -e VARIANT=routed freuid-mangojump` | `d20e1bc65848f880580cebeb33a6426bc380ae728ce3dc25fb32ba386893386b` |
| Pick 2: `final_plain.csv` | 5 base adapters only | `docker run --network none -v <images>:/data:ro -v <out-plain>:/submissions -e VARIANT=plain freuid-mangojump` | `5e63417d13873a79a0c697b250fb9e7be257948d4ce446ecdc766d692e7aaeb3` |

Add `--gpus all` if the host has `nvidia-container-toolkit` configured (large
speedup; the image auto-detects CUDA vs. CPU — see Hardware/Throughput below).

\* Record the checksum of each selected Kaggle CSV after the canonical run
with `sha256sum <out>/submission.csv`. In PowerShell use
`Get-FileHash <out>\submission.csv -Algorithm SHA256`. The canonical hardware was RTX 5060 Ti 16GB, torch
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
private set takes ≈ 19 h. `VARIANT=plain` skips the sixth PAD adapter
(~15 % faster). Run each pick into a separate host output directory so each
has its own `submission.csv` checksum.

## Frozen weights

[`weights/`](weights/) is versioned in this repository with Git LFS and copied
into `/app/weights` during `docker build`. The image performs no external
weight download; its only online build-time step is caching the public base
backbones used by `timm`.

## What is frozen where

| Artifact | Path | Notes |
|---|---|---|
| DoRA adapters (6 members) | `weights/<member>.pt` | EMA weights; ~55–115 MB each |
| Fusion / capture / PAD heads | `weights/heads.pt` | linear heads, trained pre-freeze |
| FGTS token indices | `weights/fisher_idx.npz` | frozen; never recomputed at inference |
| kNN router reference + thresholds | `weights/knn_ref.npz`, `weights/config.json` | distance floor 0.246778, capture threshold 0.5 |
| Base backbones | HF cache baked into the image | DINOv3-L/H+ (Meta), SigLIP-2 SO400M (Apache-2.0), DFN5B (Apple ASCL) |

Training code for every member is under `src/` with per-run configs in
`experiments/<run_id>/config.json`; external data (DLC-2021, CC BY-SA 2.5;
SIDTD, CC BY-SA 4.0) is fetched by `src/data/` scripts and is not
redistributed in this repository or the image.
