"""Hybrid routed sub: 5-way champion base + 6-way dual-axis PAD.

LB facts: the ds member (dlcsidtd run) DILUTES the public fusion (6-way 0.00267
vs 5-way 0.00198) but its features are what makes the dual-axis PAD work
(sidtd-holdout 0.92 on 6-way feats vs 0.68 on 5-way feats — one linear head
cannot host both the reprint and the content-forgery direction unless a member
encodes them). Router architecture separates the concerns, so use each where
it wins:

  champion head : 5-way feats (public-proven 0.00198)
  PAD head      : 6-way feats (dlc 0.9994 / sidtd 0.9219 holdout, ds6 run)
  routing dist  : 5-way space (same t0 as C1p_dlc5_routed -> public cost ~+0.0001)

  python -m src.hybrid_routed
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .data.paths import REPO_ROOT, load_test
from . import fgts
from .fusion import CACHE
from .metric import freuid_score
from .router_head import blocknorm, knn_dist, pct

BASE = ["dino", "dino_hplus", "siglip512", "dfn5b", "dino_hplus_dlc"]
PADM = BASE + ["dino_hplus_ds"]
GATE_REF = "subs/fusion_C1p_dlc5.csv"
OUT = "subs/fusion_hybrid_routed.csv"
PAD_SPLITS = ["dlc2021", "sidtdclips"]


def load_eval(members, split):
    Xs, y = [], None
    for m in members:
        d = np.load(CACHE / f"eval_{m}__{split}.npz")
        Xs.append(d["X"]); y = d["y"]
    return np.concatenate(Xs, 1), y


def load_train_test(members):
    Ftr, Fte, ytr, dims = [], [], None, []
    for m in members:
        d = np.load(CACHE / f"{m}.npz")
        Ftr.append(d["Xtr"]); Fte.append(d["Xte"]); ytr = d["ytr"]
        dims.append(d["Xtr"].shape[1])
    return np.concatenate(Ftr, 1), np.concatenate(Fte, 1), ytr, dims


def main():
    device = "cuda"
    torch.manual_seed(0)

    # champion side (5-way)
    Xtr, Xte, ytr, dims = load_train_test(BASE)
    n1, n0 = max(1, int(ytr.sum())), max(1, int((ytr == 0).sum()))
    champ = fgts.train_head(Xtr, ytr, device, epochs=150, pw=n0 / n1)

    # routing distances in 5-way space
    Ntr = blocknorm(Xtr, dims)
    dist = {}
    dist["test"] = knn_dist(blocknorm(Xte, dims), Ntr, device, n_blocks=len(dims))
    for s in ["cleanref", "recap20"] + PAD_SPLITS:
        X, _ = load_eval(BASE, s)
        dist[s] = knn_dist(blocknorm(X, dims), Ntr, device, n_blocks=len(dims))
    t0 = max(np.percentile(dist["test"], 99.9), dist["recap20"].max())
    t1 = np.percentile(np.concatenate([dist[s] for s in PAD_SPLITS]), 10)
    print(f"ramp t0={t0:.4f} t1={t1:.4f} (5-way space)")

    def w(d):
        return np.clip((d - t0) / max(t1 - t0, 1e-6), 0, 1) if t1 > t0 \
            else (d > t0).astype(float)

    # PAD side (6-way feats): halves for honest holdout, full for deployment
    rng = np.random.default_rng(0)
    pads, halves = {}, {}
    for s in PAD_SPLITS:
        pads[s] = load_eval(PADM, s)
        perm = rng.permutation(len(pads[s][0]))
        halves[s] = (perm[:len(perm) // 2], perm[len(perm) // 2:])
    Xp = np.concatenate([pads[s][0][halves[s][0]] for s in PAD_SPLITS])
    yp = np.concatenate([pads[s][1][halves[s][0]] for s in PAD_SPLITS])
    n1p, n0p = max(1, int(yp.sum())), max(1, int((yp == 0).sum()))
    pad_half = fgts.train_head(Xp, yp, device, epochs=150, pw=n0p / n1p)
    Xpa = np.concatenate([pads[s][0] for s in PAD_SPLITS])
    ypa = np.concatenate([pads[s][1] for s in PAD_SPLITS])
    n1a, n0a = max(1, int(ypa.sum())), max(1, int((ypa == 0).sum()))
    pad_full = fgts.train_head(Xpa, ypa, device, epochs=150, pw=n0a / n1a)

    # holdout evals: champ on 5-way feats, PAD(half) on 6-way feats
    for s in PAD_SPLITS:
        hoi = halves[s][1]
        Xb, y = load_eval(BASE, s)
        pc = fgts.predict(champ, Xb[hoi], device)
        pp = fgts.predict(pad_half, pads[s][0][hoi], device)
        ww = w(dist[s][hoi])
        p = (1 - ww) * pc + ww * pp
        y = y[hoi]
        print(f"[{s}-holdout] hybrid AUC={roc_auc_score(y, p):.4f} "
              f"(champ {roc_auc_score(y, pc):.4f}) genuine_p={p[y == 0].mean():.3f} "
              f"fraud_p={p[y == 1].mean():.3f} routed_frac={float((ww > .01).mean()):.3f}")
    for s in ["cleanref", "recap20"]:
        Xb, y = load_eval(BASE, s)
        ww = w(dist[s])
        print(f"[{s}] routed_frac={float((ww > .01).mean()):.4f} (must be 0)")

    # test: champ(5-way) + pad_full(6-way)
    _, Xte6, _, _ = load_train_test(PADM)
    ww = w(dist["test"])
    pt = (1 - ww) * fgts.predict(champ, Xte, device) + \
        ww * fgts.predict(pad_full, Xte6, device)
    te_ids = load_test().df["id"].astype(str).tolist()
    ref = (pd.read_csv(REPO_ROOT / GATE_REF, dtype={"id": str})
           .set_index("id").loc[te_ids, "label"].to_numpy())
    r = freuid_score((ref > .5).astype(int), pt)
    mid = int(((pt >= .01) & (pt <= .99)).sum())
    print(f"\n[test] routed_frac={float((ww > .01).mean()):.5f} "
          f"gate pseudoFREUID={r.freuid:.5f} mid={mid}/{len(pt)}")
    fgts.write_sub(te_ids, pt, OUT)


if __name__ == "__main__":
    main()
