"""P0 follow-up: is the DLC collapse fixable at the HEAD level at all?

LOTO ensembling was a clean negative (dlc AUC 0.30 -> 0.297): every head, even
ones that practiced unseen types, maps DLC genuines to fraud. Two remaining
head-level questions, both cache-only (features already extracted):

  1. PROBE: does the frozen 4-way feature space contain the or/cg signal AT ALL?
     5-fold linear probe on DLC alone. ~0.5 => feature-level dead end, only
     backbone retraining (P0.5) or scale (7B) can help. High => head direction
     is wrong but fixable with real-recapture supervision.
  2. MIX: train the head on clean-train + half of DLC (or=0, cg=1, oversampled),
     eval on the held-out half + cleanref + recap20 + pseudo-FREUID gate.
     Fixes holdout without hurting public proxy => cheap defensive sub exists,
     and P0.5 (mixing real recaptures into DoRA training) gets a green light.

  python -m src.dlc_head
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .data.paths import REPO_ROOT, load_test
from . import fgts
from .fusion import CACHE
from .metric import freuid_score
from .train_DINOV3L_512 import make_splits

CHAMPION_SUB = "subs/fusion_C1_dfn5b.csv"


def load_eval(members, split):
    Xs, y = [], None
    for m in members:
        d = np.load(CACHE / f"eval_{m}__{split}.npz")
        Xs.append(d["X"]); y = d["y"]
    return np.concatenate(Xs, 1), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="dino,dino_hplus,siglip512,dfn5b")
    ap.add_argument("--repeats", default="1,25,100", help="DLC oversampling factors to sweep")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    members = args.members.split(",")

    Xd, yd = load_eval(members, "dlc2021")
    Xc, yc = load_eval(members, "cleanref")
    Xr, yr = load_eval(members, "recap20")
    print(f"dlc {Xd.shape} ({int(yd.sum())} cg / {int((yd == 0).sum())} or)")

    # 1) PROBE: linear separability of or/cg inside the frozen feature space
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xd, yd):
        n1, n0 = max(1, int(yd[tr].sum())), max(1, int((yd[tr] == 0).sum()))
        h = fgts.train_head(Xd[tr], yd[tr], device, epochs=150, pw=n0 / n1)
        aucs.append(roc_auc_score(yd[te], fgts.predict(h, Xd[te], device)))
    print(f"\nPROBE dlc-only 5-fold AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  {[f'{a:.3f}' for a in aucs]}")

    # 2) MIX: clean-train + half DLC -> eval held-out half + public proxies
    tr_df, _ = make_splits(42, 0)
    tr_df = tr_df[tr_df["is_digital"]].reset_index(drop=True)
    Ftr, Fte, ytr = [], [], None
    for m in members:
        d = np.load(CACHE / f"{m}.npz")
        Ftr.append(d["Xtr"]); Fte.append(d["Xte"]); ytr = d["ytr"]
    Xtr = np.concatenate(Ftr, 1); Xte = np.concatenate(Fte, 1)

    te_ids = load_test().df["id"].astype(str).tolist()
    champ = (pd.read_csv(REPO_ROOT / CHAMPION_SUB, dtype={"id": str})
             .set_index("id").loc[te_ids, "label"].to_numpy())
    pseudo = (champ > .5).astype(int)

    rng = np.random.default_rng(0)
    half = rng.permutation(len(Xd))
    dtr, dte = half[:len(Xd) // 2], half[len(Xd) // 2:]
    print(f"mix: dlc-train n={len(dtr)}, dlc-holdout n={len(dte)}")

    for rep in [int(r) for r in args.repeats.split(",")]:
        Xm = np.concatenate([Xtr] + [Xd[dtr]] * rep)
        ym = np.concatenate([ytr] + [yd[dtr]] * rep)
        n1, n0 = max(1, int(ym.sum())), max(1, int((ym == 0).sum()))
        h = fgts.train_head(Xm, ym, device, epochs=150, pw=n0 / n1)
        pd_ho = fgts.predict(h, Xd[dte], device)
        pc = fgts.predict(h, Xc, device)
        pr = fgts.predict(h, Xr, device)
        pt = fgts.predict(h, Xte, device)
        r = freuid_score(pseudo, pt)
        mid = int(((pt >= .01) & (pt <= .99)).sum())
        print(f"rep={rep:3d} | dlc-ho AUC={roc_auc_score(yd[dte], pd_ho):.4f} "
              f"genuine_p={pd_ho[yd[dte] == 0].mean():.3f} fraud_p={pd_ho[yd[dte] == 1].mean():.3f} | "
              f"cleanref AUC={roc_auc_score(yc, pc):.4f} | recap20 AUC={roc_auc_score(yr, pr):.4f} | "
              f"gate pseudoFREUID={r.freuid:.5f} mid={mid}/{len(pt)}")
        if rep == 25:
            fgts.write_sub(te_ids, pt, "subs/fusion_C1_dlcmix25.csv")


if __name__ == "__main__":
    main()
