# The FREUID Challenge 2026 — team mangojump

Fraud detection on identity documents (IJCAI-ECAI 2026 challenge). Final system:
feature-level fusion of DoRA-finetuned backbones (DINOv3-L/H+, SigLIP-2, DFN5B)
with an OOD **router** that switches far-out-of-distribution captures from the
clean fusion head to a presentation-attack (PAD) head trained on real
recaptures (DLC-2021) and content-forged physical documents (SIDTD clips).

- Public LB best: `fusion_C1p_dlc5` = 0.00198; private-hedged: `fusion_hybrid_routed`.
- Reproducibility package (Docker, report, commands): see `REPRODUCE.md` (WIP).

## External data (fetched by scripts under `src/data/`, not redistributed here)

| Dataset | License | Use |
|---|---|---|
| [DLC-2021](https://zenodo.org/record/DLC2021) | CC BY-SA 2.5 | real original/reprint captures (training mix + arbitration) |
| [SIDTD](https://doi.org/10.34810/data1815) | CC BY-SA 4.0 (CORA DOI record) | content-forged printed documents, video frames (training mix + arbitration) |

Pretrained backbones via `timm`: DINOv3 (Meta), SigLIP-2 (Google), DFN5B (Apple).

## Layout

```
src/                training, fusion, router, evaluation
src/data/           dataset indexing + selective external-data fetchers
experiments/        run configs + metrics (checkpoints excluded from git)
reports/            working notes, technical report, runbooks
docker/             offline inference image (see REPRODUCE.md)
artifacts/system/   frozen inference artifacts (mirrored on Hugging Face)
```

## Reproducing the final submissions (Docker)

One offline image, one flag — full details, checksums, and the floating-point
tolerance statement live in [REPRODUCE.md](REPRODUCE.md):

```bash
# no HF account/token needed: all four backbones are ungated timm mirrors
docker build -t freuid-mangojump .

docker run --rm --network none \
  -v /path/to/flat/test/images:/data:ro -v "$(pwd)/out:/submissions" \
  -e VARIANT=routed freuid-mangojump               # final pick 1
# -e VARIANT=plain                                 # final pick 2 (router off)
# add --gpus all if the host has nvidia-container-toolkit (CPU fallback otherwise)
```

Model weights: frozen in `artifacts/system/` (Git LFS, this repo) and
[ching0206/freuid-2026-mangojump](https://huggingface.co/ching0206/freuid-2026-mangojump)
(what the Docker build actually fetches), revision
`a36f036aba49ede6890761c927fae8f1951922c9`, sha256-verified identical.

**Hardware**: everything (training and inference) ran on a single NVIDIA RTX
5060 Ti 16 GB, Windows 11, torch 2.11 nightly cu128. Inference ≈ 8 min / 1k
images; member training 10–19 h per backbone.

## Data setup

**Inference only (reproducing the final submissions): no data setup needed.**
The Docker image / `src.predict_docker` takes any flat directory of images —
see REPRODUCE.md.

**Training reproduction** expects this layout under the repo root (or under
`$FREUID_DATA` if you set that environment variable; see `src/data/paths.py`):

```
train_labels.csv                  competition CSV (id, label, is_digital, type)
sample_submission.csv             competition CSV (defines the test id list)
train/train/<id>.jpeg             training images (nested dir, as Kaggle unzips)
public_test/public_test/<id>.jpeg public test images
external/dlc2021/...              DLC-2021 frames  -> fetch: reports/dlc2021_setup.md,
                                  then `python -m src.data.index_dlc2021`
                                  (writes artifacts/dlc2021_index.csv)
external/sidtd/clips_cropped/...  SIDTD frames     -> `python -m src.data.fetch_sidtd_clips`
                                  (HTTP-Range selective fetch, ~700 MB;
                                  writes artifacts/sidtd_clips_index.csv)
```

Competition data comes from the Kaggle competition page (not redistributed
here). Derived index/split CSVs land in `artifacts/`:
`python -m src.dlc_split` writes the doctype-disjoint DLC train/holdout split;
`python -m src.data.build_extra_mix` writes
`artifacts/extra_train_dlc_sidtd.csv` / `extra_val_dlc_sidtd.csv` — the exact
external-mix files passed to training via `--extra_data` / `--extra_val`.

## License

MIT (code). Competition and external data remain under their own licenses.
