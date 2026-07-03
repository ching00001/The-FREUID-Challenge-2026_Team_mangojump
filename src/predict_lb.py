"""Estimate a submission's PUBLIC LB without spending a submission.

We have no public-test labels, but we DO have 5 (submission, real LB) pairs.
Method: take the best known model (SigLIP-512, LB 0.02667, ~0.99 public AUC) as
pseudo-labels on the 7,821 public ids, compute every model's FREUID vs those
pseudo-labels, then fit a 1-D calibration (real LB ~ a*pseudo_freuid + b) on the
known pairs and apply it to the target.

⚠️ HARD LIMITATION: disagreement-with-SigLIP is counted as error, so this can
estimate "≈ or worse than SigLIP" but CANNOT detect a model that is genuinely
BETTER than SigLIP on the true labels. Read the estimate as "no better than ~X".
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .data.paths import REPO_ROOT, load_test
from .metric import freuid_score

# known (submission file, real public LB)
KNOWN = {
    "subs/siglip512_dora.csv": 0.02667,
    "subs/siglip512_rcaug.csv": 0.04101,
    "subs/ens_siglip_mean.csv": 0.03480,
    "subs/ens_siglip.csv": 0.06608,
    "subs/siglip378_dora.csv": 0.08104,
}
PSEUDO_SRC = "subs/siglip512_dora.csv"   # best model = pseudo ground truth


def load(path, ids):
    return pd.read_csv(REPO_ROOT / path, dtype={"id": str}).set_index("id").loc[ids, "label"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True, help="submission csvs to estimate")
    args = ap.parse_args()
    ids = load_test().df["id"].astype(str).tolist()

    pseudo = (load(PSEUDO_SRC, ids) > 0.5).astype(int)
    print(f"pseudo-labels from {PSEUDO_SRC}: {pseudo.sum()} fraud / {len(pseudo)}")

    # calibration points: pseudo_freuid vs real LB
    xs, ys = [], []
    print(f"\n{'submission':28s} {'pseudoFREUID':>12s} {'realLB':>8s}")
    for f, lb in KNOWN.items():
        pf = freuid_score(pseudo, load(f, ids)).freuid
        xs.append(pf); ys.append(lb)
        print(f"{f.split('/')[-1]:28s} {pf:12.4f} {lb:8.4f}")
    xs, ys = np.array(xs), np.array(ys)
    a, b = np.polyfit(xs, ys, 1)
    pred_known = a * xs + b
    mae = np.abs(pred_known - ys).mean()
    print(f"\ncalibration: realLB ~= {a:.3f}*pseudoFREUID + {b:.4f}  (fit MAE={mae:.4f})")

    print(f"\n{'target':28s} {'pseudoFREUID':>12s} {'est.LB':>8s}")
    for t in args.targets:
        pf = freuid_score(pseudo, load(t, ids)).freuid
        est = a * pf + b
        print(f"{t.split('/')[-1]:28s} {pf:12.4f} {est:8.4f}")
    print("\n⚠️ 'est.LB' = lower bound-ish: cannot detect better-than-SigLIP "
          "(disagreement counts as error).")


if __name__ == "__main__":
    main()
