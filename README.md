# The FREUID Challenge 2026 — team lilwu

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
reports/            working notes
```

## License

MIT (code). Competition and external data remain under their own licenses.
