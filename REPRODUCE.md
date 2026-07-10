# Reproducing the final submissions (team mangojump)

One frozen system, one Docker image, two selected final picks that differ only
in an inference-time flag (confirmed acceptable by the organizers).

## Final pick mapping

| Kaggle submission | Command | Public LB | Output checksum* |
|---|---|---|---|
| `final_routed.csv` (Pick 1, expected ranking pick) | `docker run --network none --gpus all -v <images>:/data:ro -v <out>:/submissions -e VARIANT=routed freuid-mangojump` | 0.00207 | `<md5, filled at freeze>` |
| `final_plain.csv` (Pick 2, router-off ablation) | same with `-e VARIANT=plain` | 0.00207 | `<md5, filled at freeze>` |

\* Checksums are from our canonical run (RTX 5060 Ti 16GB, torch
2.11.0.dev20251223+cu128). GPU inference is not bit-portable across
hardware/kernel versions; expected reproduction tolerance on other GPUs:
per-row mean |Δ| ≈ 3e-4, decision flips ≤ 0.04 % of rows, leaderboard impact
< ±0.0002. Same-machine reruns are deterministic at ~1e-7.

## Build (network required once)

```bash
export HF_TOKEN=<token with access to gated facebook/dinov3-* repos>
python docker/prepare_hf_cache.py        # ~11 GB of backbone checkpoints
docker build -f docker/Dockerfile -t freuid-mangojump .
```

## Run (no network)

Input: a flat directory of images (`.jpeg/.jpg/.png/.webp/.bmp/.tif/.tiff`),
row id = filename without extension. Output: `/submissions/submission.csv`
with columns `id,label` (fraud score in [0,1]).

Throughput: ≈ 8 min / 1k images on an RTX 5060 Ti (16 GB); the 134,997-image
private set takes ≈ 19 h. `VARIANT=plain` skips one backbone (~15 % faster).

## Frozen weights

All inference artifacts are mirrored at
https://huggingface.co/ching0206/freuid-2026-mangojump
(revision `fbe08e1b74631f5fb8cf9ef73e5dc1b01230d401`, per-file sha256 matches
this repo's `artifacts/system/` bit-for-bit; verify with
`python docker/verify_hf_upload.py ching0206/freuid-2026-mangojump`).

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
