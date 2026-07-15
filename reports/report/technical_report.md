# Fusion of DoRA-Adapted Foundation Models with OOD-Routed PAD Heads for Identity-Document Fraud Detection

**Team mangojump — The FREUID Challenge 2026 (IJCAI-ECAI)**
*Hardware budget: one RTX 5060 Ti (16 GB), zero cloud spend.*

## 1. Introduction

FREUID asks for a fraud score per identity-document image, evaluated by
FREUID = 1 − harmonic(1−AuDET, 1−APCER@1%BPCER) — an *operating-point*
metric: a handful of confidently-wrong genuines can dominate the score even
when ranking (AUC) is nearly perfect.

The central difficulty is distribution shift. The training set is 99.97 %
digitally generated documents of five layouts; the test data mixes digital
and physically captured documents (about 5 % of the public split is physical
capture by our measurement), and the private set emphasizes real captures
plus two document types never seen in training, with an explicit penalty for
relying on generator artifacts.

Our thesis: with a 16 GB single-GPU budget, the strongest defensible system is
(i) a feature-level **fusion** of frozen, DoRA-finetuned foundation backbones
for the in-distribution mass, plus (ii) an **out-of-distribution router** that
hands far-OOD physical captures to a small **presentation-attack (PAD) head**
trained on ~1.3 k real captured documents from two public datasets. The two
components are trained once and frozen; at inference the router decides,
per image, which head scores it.

## 2. Method

### 2.1 Members: DoRA-adapted backbones

All members share one recipe (`src/train_DINOV3L_512.py`): frozen backbone +
DoRA (rank 16, α 32) on attention **and** MLP linears, input 512², bf16, EMA
0.9995, 3 epochs, type×class-balanced sampling, RandomResizedCrop/ColorJitter/
h-flip. Members and their roles:

| Member | Backbone | Extra data | Role |
|---|---|---|---|
| `dino` | DINOv3 ViT-L/16 | — | fusion base |
| `dino_hplus` | DINOv3 ViT-H+/16 | — | fusion base |
| `siglip512` | SigLIP-2 SO400M/16 | — | fusion base |
| `dino_hplus_dlc` | DINOv3 ViT-H+/16 | DLC-2021 (5 %) | fusion base + capture axis |
| `dino_hplus_ds` | DINOv3 ViT-H+/16 | DLC+SIDTD (8 %) | PAD features only |

A sixth member, DFN5B CLIP ViT-H/14 @378 (Apple), was part of the system as
submitted at the July 13 freeze. We subsequently found its `apple-amlr`
license is non-commercial-only and, with explicit organizer approval, removed
it post-freeze by re-serializing the fusion/capture/PAD heads from the
pre-freeze feature cache of the remaining 5 members only (train-split
features, computed before the private set existed — no backbone retraining,
no private-test involvement). This reproduced a combination already
validated pre-freeze (2026-07-08, public LB 0.00198, identical to the
DFN5B-inclusive score); post-remediation public LB is 0.00208. All results
below reflect the DFN5B-inclusive system as it stood at freeze, since that is
the development history; the shipped, prize-eligible system is the
5-member/4-base variant described in this note and in REPRODUCE.md.

For DINOv3 members we pool the top-64 patch tokens ranked by a Fisher
criterion computed on train (FGTS); VL members use global patch pooling. The
token indices are computed once and frozen (§5).

**External-data mixing is done at the *feature* level** — a small, sampler-
mass-controlled fraction (5.3 % DLC-2021 + 2.7 % SIDTD; the type-balanced
sampler would otherwise give the external pseudo-types 2/7 of all draws) mixed
into DoRA training. Mixing the same data into a fusion *head* instead is
toxic: the dose that fixes the external axis destroys in-domain recaptures
(0.905→0.77) and the public operating point.

### 2.2 Fusion

Member features are concatenated (4,736-d in the shipped 4-base-member system;
6,016-d with DFN5B, as trained pre-freeze) and a linear head (LayerNorm →
Dropout → Linear) is trained on the digital training set. Feature-level fusion
beat every weight/rank blend we tried and every single model: our best single
model scores 0.00297 public, the shipped fusion 0.00208 (0.00198 with DFN5B
included, pre-remediation).

### 2.3 The private axis and the router

The organizers state the private set emphasizes real captures and unseen
document types. We measured what a clean-trained fusion does there using two
public real-capture datasets as arbitration sets, split by document type
(train / holdout doctypes never overlap):

* **DLC-2021** (genuine originals vs printed copies, 540 frames, 1 per clip):
  the clean fusion *inverts* — AUC 0.23, genuine documents scored 0.85 fraud
  ("default-to-fraud" on unfamiliar captures — exactly the behavior the
  organizers penalize).
* **SIDTD clips** (content-forged documents, printed and recaptured, 791
  frames): the clean fusion is blind — AUC 0.37, everything scored genuine
  ("default-to-genuine": content-forged physical documents sail through).

Two structural findings shaped the final design:

1. **A frozen fusion head dilutes member-level OOD fixes.** After adding the
   DLC-mixed member, the member itself scores unseen-doctype DLC at AUC 1.0,
   but the clean-trained fusion head on top of it still fails (0.36): a head
   trained only on clean digital data has no incentive to use the reprint
   direction. Delivery therefore needs routing, not just better members.
2. **One linear head cannot host two capture domains** (and, symmetrically, a
   strong member's features can even *poison* a frozen fusion head — our
   SigLIP+external member was excellent standalone yet flipped 4/6 genuine
   recaptures to ~1.0 when swapped into the fusion base; it is used on the
   PAD side only... in the shipped system it is not used at all, see §2.4).

**Router.** Two conditions, both computed from the same cached features:
a *capture head* (digital vs physical, trained without fraud labels — separates
at cleanref p50 0.002 vs holdout p50 0.9996) AND a *kNN distance* to the
digital training features above a floor frozen at the public test's 99th
percentile. Distance alone is too timid (the floor is pinned by the public
tail), capture alone is too aggressive (it would hand ~5 % of the public set
to the PAD head); their intersection routes "OOD physical captures" only —
0.7 % of the public set, 98 % of DLC holdout, 62 % of SIDTD holdout.

**PAD head.** A linear head on the 7,296-d PAD feature set (base + the
DLC+SIDTD member), trained on both external datasets (or/real = genuine,
cg/fake = fraud).

### 2.4 Final system

```
image ─→ 5 base member features ──→ fusion head ────────────→ score
   │                    │  (capture head > 0.5) ∧ (kNN dist > floor)
   └→ +1 PAD member ────┴──────────→ PAD head  ──(if routed)─→ score
```

Routed results on unseen-doctype holdouts: DLC 0.23 → **1.00**, SIDTD 0.37 →
**0.84**, known-type recaptures 0.905, clean digital untouched (1.0), public
cost ≈ +0.0001.

### 2.5 What did not work

Negative results that shaped the system: leave-one-type-out head ensembling
(no effect — the failure is in features, not heads); simulated recapture
augmentation (helped local proxies, hurt LB); mixing external data into the
fusion head (toxic at any effective dose); full fine-tuning (3.5× worse than
DoRA); frozen linear probes (operating-point collapse); rule-based text/date
consistency (0 % recall on generator output); per-type expert heads (worse
than one pooled head); forensic-CNN fusion members (hesitant scores destroy
the operating point); standalone externally-mixed backbones (trade in-domain
sharpness for robustness — only viable as fusion/PAD members).

## 3. Data

| Source | License | Use |
|---|---|---|
| FREUID competition data (69,352 train / 5 doc types) | competition terms | member training, heads, router reference |
| **DLC-2021** [Polevoy et al.] — 540 frames (1 per clip), genuine originals vs printed copies, 10 EU doctypes | CC BY-SA 2.5 (Zenodo) | 5 doctypes → training mix; 5 held-out doctypes → arbitration |
| **SIDTD clips_cropped** [Boned et al., Sci. Data 2024] — 791 dewarped frames (median frame per clip), bona-fide vs content-forged printed documents | CC BY-SA 4.0 (CORA, DOI 10.34810/data1815) | same doctype-disjoint split |

SIDTD frames were fetched selectively over HTTP-Range from the official host
(datasets.cvc.uab.es) — 791 frames out of a 26.6 GB archive. Neither dataset
is redistributed in our repository or Docker image; fetch scripts are under
`src/data/`. Pretrained backbones in the shipped system: DINOv3 (Meta AI,
DINOv3 license), SigLIP-2 (Google, Apache-2.0) — both permit commercial use,
obtained via `timm`; we redistribute only DoRA adapter deltas. (An earlier,
pre-freeze version also used DFN5B (Apple), whose `apple-amlr` license is
non-commercial-only; see §2.1.)

## 4. Inference procedure

One Docker image (`docker/Dockerfile`), no network, no runtime training:

```
docker run --network none --gpus all -v <images>:/data:ro \
  -v <out>:/submissions -e VARIANT=routed freuid-mangojump
```

`VARIANT=routed` reproduces final pick 1 (full system); `VARIANT=plain`
reproduces final pick 2 (router disabled — the pure fusion, an ablation).
Everything the entrypoint touches is frozen in `weights/`: adapter deltas,
three linear heads, FGTS token indices, the kNN reference matrix and both
thresholds. Throughput ≈ 8 min / 1k images on our GPU. See REPRODUCE.md
for the pick↔command↔checksum mapping and the cross-hardware floating-point
tolerance statement (per-row mean |Δ| ≈ 3e-4, ≤ 0.04 % decision flips).

## 5. Results

Public leaderboard progression (all runs single-GPU local):

| Milestone | Public LB |
|---|---|
| CLIP-L/14 + LoRA @224 (baseline) | 0.191 |
| SigLIP-2 @512 + DoRA (single) | 0.02667 |
| DINOv3-L @512 + DoRA (single) | 0.01134 |
| DINOv3-H+ @512 + DoRA (single) | 0.00297 |
| 4-member fusion | 0.00237 |
| + DLC-mixed member (5-member fusion) | 0.00198 |
| + OOD router (far-OOD variant) | 0.00227→0.00208 across iterations |
| Final frozen pair as submitted at freeze (`routed` / `plain`, DFN5B-inclusive) | 0.00207 / 0.00207 |
| **Shipped, prize-eligible pair (DFN5B removed post-freeze, org.-approved)** | **`routed`/`plain` public 0.00208; see REPRODUCE.md for the canonical private+public checksums** |

Private-axis arbitration (unseen-doctype holdouts, never trained on):

| Slice | clean fusion | + router |
|---|---|---|
| DLC-2021 holdout (reprint axis) | 0.23 (genuine_p 0.85) | **1.00** |
| SIDTD holdout (content-forgery axis) | 0.37 (fraud missed) | **0.84** |
| known-type real recaptures (n=20) | 0.93 | 0.91 |
| clean digital reference (n=2,000) | 1.00 | 1.00 |

(Arbitration numbers above are from the DFN5B-inclusive system as it stood at
freeze; we have not independently re-run this table on the post-remediation
5-member system. Given the public-LB shift from the removal was small
(0.00198→0.00208), we expect these to be directionally representative, but
they are not re-verified.)

We kept hidden-row predictions constant (0.5) across all submissions, so our
public comparisons are unaffected by the disclosed metric-leak issue.

## 6. Reproducibility

* Repository: https://github.com/ching00001/The-FREUID-Challenge-2026_Team_mangojump — frozen commit `<SHA, filled after the DFN5B-removal commit lands>`. Per the organizers' clarification, only backbone/training-code changes are restricted post-freeze; the fusion head re-serialization in this commit uses only pre-freeze cached train-split features (see the licensing-remediation note in §2.1 and the Kaggle reproducibility thread for organizer approval).
* Weights: [`weights/`](../../weights/) is versioned in this repository via Git LFS and copied
  into the Docker image at build time (no external weight download at build or run time).
* Docker: build and run commands in REPRODUCE.md; canonical output checksums
  and tolerance statement included.
* Hardware: single NVIDIA RTX 5060 Ti 16 GB, Windows 11, torch 2.11 nightly
  cu128. Member training 10–19 h each; head/router training is seconds on
  cached features; full-system inference on the 134,997-image private set
  ≈ 16 h (5 members, post-remediation).
* Training commands and hyperparameters for every member are given in full in
  §2.1; raw per-run logs are not shipped in the repository (kept lean for the
  frozen submission) but every run is reproducible from `src/train_DINOV3L_512.py`
  with the documented flags (`--backbone`, `--extra_data`, `--extra_frac`,
  `--extra_val`).

## References

[1] S.-Y. Liu, C.-Y. Wang, H. Yin, P. Molchanov, Y.-C. F. Wang, K.-T. Cheng,
M.-H. Chen. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML 2024.
arXiv:2402.09353.

[2] O. Siméoni et al. *DINOv3.* Meta AI, 2025. arXiv:2508.10104.

[3] M. Tschannen et al. *SigLIP 2: Multilingual Vision-Language Encoders with
Improved Semantic Understanding, Localization, and Dense Features.* 2025.
arXiv:2502.14786.

[4] A. Fang, A. M. Jose, A. Jain, L. Schmidt, A. Toshev, V. Shankar. *Data
Filtering Networks.* 2023. arXiv:2309.17425. (DFN5B CLIP ViT-H/14.)

[5] Z. Huang, J. Li, H. Wen, T. Li, X. Yang, L. Qi, B. Peng, X. Huang,
M.-H. Yang, G. Cheng. *Rethinking Cross-Generator Image Forgery Detection
through DINOv3.* 2025. arXiv:2511.22471. (Basis of our Fisher-guided token
selection for DINOv3 members.)

[6] D. V. Polevoy, I. V. Sigareva, D. M. Ershova, V. V. Arlazarov,
D. P. Nikolaev, Z. Ming, M. M. Luqman, J.-C. Burie. *Document Liveness
Challenge Dataset (DLC-2021).* Journal of Imaging 8(7):181, 2022.
DOI 10.3390/jimaging8070181. Data: Zenodo (CC BY-SA 2.5).

[7] C. Boned, M. Talarmain, N. Ghanmi, G. Chiron, S. Biswas, A. M. Awal,
O. Ramos Terrades. *Synthetic dataset of ID and Travel Documents.* Scientific
Data 11, 2024. DOI 10.1038/s41597-024-04160-9. Dataset: CORA,
DOI 10.34810/data1815 (CC BY-SA 4.0).

[8] K. Bulatov et al. *MIDV-2020: A Comprehensive Benchmark Dataset for
Identity Document Analysis.* Computer Optics 46(2), 2022. arXiv:2107.00396.
(Upstream corpus of SIDTD.)

[9] R. Wightman. *PyTorch Image Models (timm).* GitHub, 2019.
DOI 10.5281/zenodo.4414861.
