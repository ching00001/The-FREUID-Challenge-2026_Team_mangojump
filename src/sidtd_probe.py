"""Probe-first arbitration of SIDTD clips_cropped BEFORE any 19h retrain.

Answers, all cache-only after one feature-extraction pass:
  1. default-to-fraud?   5-way clean head on SIDTD reals/fakes (like DLC was).
  2. signal present?     linear probe within SIDTD (doctype-LOO = transfer).
  3. router zone?        kNN distance to digital train -> in/mid/far-OOD split.
  4. PAD blind spot?     DLC-trained PAD head on SIDTD: its 'reprint=fraud'
                         direction should call SIDTD reals ok; if it also calls
                         content-forged PHYSICAL fakes ok, our routed defense
                         has a hole that only SIDTD-style data can close.

  python -m src.sidtd_probe
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .data.paths import REPO_ROOT
from . import fgts
from .fusion import CACHE
from .loto_head import member_eval_feats
from .router_head import blocknorm, knn_dist

MEMBERS5 = ["dino", "dino_hplus", "siglip512", "dfn5b", "dino_hplus_dlc"]


def load_eval(members, split):
    Xs, y = [], None
    for m in members:
        d = np.load(CACHE / f"eval_{m}__{split}.npz")
        Xs.append(d["X"]); y = d["y"]
    return np.concatenate(Xs, 1), y


def main():
    device = "cuda"
    torch.manual_seed(0)
    sdf = pd.read_csv(REPO_ROOT / "artifacts/sidtd_clips_index.csv")
    sdf["doctype"] = sdf["type"].str.split("/").str[1]
    member_eval_feats(MEMBERS5, [("sidtdclips", sdf)], device)

    # features + train cache
    Ftr, ytr, dims = [], None, []
    for m in MEMBERS5:
        d = np.load(CACHE / f"{m}.npz")
        Ftr.append(d["Xtr"]); ytr = d["ytr"]; dims.append(d["Xtr"].shape[1])
    Xtr = np.concatenate(Ftr, 1)
    Xs, ys = load_eval(MEMBERS5, "sidtdclips")
    Xd, yd = load_eval(MEMBERS5, "dlc2021")
    print(f"sidtd clips {Xs.shape} ({int(ys.sum())} fake / {int((ys == 0).sum())} real)")

    # 1) clean 5-way head (the plain fusion) on SIDTD
    n1, n0 = max(1, int(ytr.sum())), max(1, int((ytr == 0).sum()))
    clean_head = fgts.train_head(Xtr, ytr, device, epochs=150, pw=n0 / n1)
    p = fgts.predict(clean_head, Xs, device)
    print(f"\n[1 clean 5-way head] AUC={roc_auc_score(ys, p):.4f} "
          f"real_p={p[ys == 0].mean():.3f} fake_p={p[ys == 1].mean():.3f} "
          f"mid={int(((p >= .01) & (p <= .99)).sum())}/{len(p)}")

    # 2) linear separability within SIDTD, doctype-LOO (transfer across layouts)
    aucs = []
    for dt in sorted(sdf["doctype"].unique()):
        te = (sdf["doctype"] == dt).values
        if len(set(ys[te])) < 2 or len(set(ys[~te])) < 2:
            continue
        n1t = max(1, int(ys[~te].sum())); n0t = max(1, int((ys[~te] == 0).sum()))
        h = fgts.train_head(Xs[~te], ys[~te], device, epochs=150, pw=n0t / n1t)
        aucs.append((dt, roc_auc_score(ys[te], fgts.predict(h, Xs[te], device))))
    print("\n[2 probe doctype-LOO] " + " ".join(f"{d}:{a:.3f}" for d, a in aucs)
          + f"  mean={np.mean([a for _, a in aucs]):.4f}")

    # 3) router zone: kNN distance vs the deployed ramp
    Ntr = blocknorm(Xtr, dims)
    ds = knn_dist(blocknorm(Xs, dims), Ntr, device, k=10, n_blocks=len(dims))
    q = np.percentile(ds, [1, 50, 99])
    print(f"\n[3 router zone] p1={q[0]:.4f} p50={q[1]:.4f} p99={q[2]:.4f} "
          f"(5-way ramp t0=0.3697; DLC p50=0.4312)")
    for name, mask in [("reals", ys == 0), ("fakes", ys == 1)]:
        frac = float((ds[mask] > 0.3697).mean())
        print(f"  {name}: frac beyond t0 (would be routed) = {frac:.3f}")

    # 4) DLC-trained PAD head scored on SIDTD (blind-spot test)
    n1d = max(1, int(yd.sum())); n0d = max(1, int((yd == 0).sum()))
    pad = fgts.train_head(Xd, yd, device, epochs=150, pw=n0d / n1d)
    pp = fgts.predict(pad, Xs, device)
    print(f"\n[4 DLC-PAD on SIDTD] AUC={roc_auc_score(ys, pp):.4f} "
          f"real_p={pp[ys == 0].mean():.3f} fake_p={pp[ys == 1].mean():.3f}")
    print("   (fake_p LOW here = PAD calls content-forged physical docs 'original'"
          " = the blind spot SIDTD mixing would close)")


if __name__ == "__main__":
    main()
