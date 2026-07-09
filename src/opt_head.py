"""Cheap experiment: does an operating-point-aware loss beat plain BCE for the
fusion head, on the SAME cached backbone features (no backbone retrain)?

Hypothesis: every fusion-stage failure so far (forensic, per-type routing,
rank-mean, multi-crop) died from the same mechanism -- something blurred the
score distribution near the APCER@1%BPCER operating point. All fixes so far
were architectural (which members, which data). Nobody has changed the loss
that shapes the fusion head itself, which still only optimizes plain BCE.

freuid_loss = BCE + lam * hinge(margin + tau - fraud_logit)
  tau = 99th-percentile genuine logit in the current batch (detached)
  Directly penalises fraud scores that fall below the live 1%-BPCER threshold
  -> a direct surrogate for APCER@1%BPCER itself, not just a proxy for it.

Trains on the 5 already-cached C1p_dlc5 members (dino, dino_hplus,
dino_hplus_dlc, siglip512, dfn5b); evaluates both heads with the real
metric.freuid_score on cleanref/dlc2021/sidtdclips/recap20 caches. No
submission is written -- this is proxy-only triage before spending a real LB
slot.

  python -m src.opt_head --lam 0.5,1,2,5
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import metric

CACHE = "artifacts/fusion_cache"
MEMBERS = ["dino", "dino_hplus", "dino_hplus_dlc", "siglip512", "dfn5b"]
EVAL_SETS = ["cleanref", "dlc2021", "sidtdclips", "recap20"]


def load_concat(kind: str):
    """kind: 'train' or one of EVAL_SETS."""
    Xs, y = [], None
    for m in MEMBERS:
        if kind == "train":
            d = np.load(f"{CACHE}/{m}.npz")
            Xs.append(d["Xtr"]); y = d["ytr"]
        else:
            d = np.load(f"{CACHE}/eval_{m}__{kind}.npz")
            Xs.append(d["X"]); y = d["y"]
    return np.concatenate(Xs, 1), y


def make_head(D, device):
    return nn.Sequential(nn.LayerNorm(D), nn.Dropout(0.2), nn.Linear(D, 1)).to(device)


def train_bce(Xtr, ytr, device, epochs=150):
    D = Xtr.shape[1]
    head = make_head(D, device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    n1, n0 = max(1, int((ytr == 1).sum())), max(1, int((ytr == 0).sum()))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n0 / n1, device=device))
    X = torch.tensor(Xtr, dtype=torch.float32, device=device)
    y = torch.tensor(ytr, dtype=torch.float32, device=device)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        crit(head(X).squeeze(-1), y).backward()
        opt.step()
    return head.eval()


def train_freuid(Xtr, ytr, device, lam, margin=2.0, bpcer_target=0.01, epochs=150):
    D = Xtr.shape[1]
    head = make_head(D, device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    n1, n0 = max(1, int((ytr == 1).sum())), max(1, int((ytr == 0).sum()))
    pos_weight = torch.tensor(n0 / n1, device=device)
    X = torch.tensor(Xtr, dtype=torch.float32, device=device)
    y = torch.tensor(ytr, dtype=torch.float32, device=device)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        logit = head(X).squeeze(-1)
        bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
        genuine, fraud = logit[y == 0], logit[y == 1]
        tau = torch.quantile(genuine.detach(), 1.0 - bpcer_target)
        op = F.relu(margin + tau - fraud).mean()
        (bce + lam * op).backward()
        opt.step()
    return head.eval()


@torch.no_grad()
def score(head, X, device):
    return torch.sigmoid(head(torch.tensor(X, dtype=torch.float32, device=device)).squeeze(-1)).cpu().numpy()


def eval_all(head, device, tag):
    print(f"\n--- {tag} ---")
    for kind in EVAL_SETS:
        X, y = load_concat(kind)
        p = score(head, X, device)
        if len(np.unique(y)) < 2:
            print(f"  {kind:12s} single-class, skip"); continue
        r = metric.freuid_score(y, p)
        print(f"  {kind:12s} n={len(y):4d} FREUID={r.freuid:.4f} AuDET={r.audet:.4f} "
              f"APCER@1%BPCER={r.apcer_at_1pct_bpcer:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", default="0.5,1,2,5")
    ap.add_argument("--margin", type=float, default=2.0)
    args = ap.parse_args()
    device = "cuda"

    print(f"members: {MEMBERS}")
    Xtr, ytr = load_concat("train")
    print(f"train: {Xtr.shape}, pos={int((ytr==1).sum())} neg={int((ytr==0).sum())}")

    head_bce = train_bce(Xtr, ytr, device)
    eval_all(head_bce, device, "BASELINE (plain BCE, = current fusion.py head)")

    for lam in [float(x) for x in args.lam.split(",")]:
        head_op = train_freuid(Xtr, ytr, device, lam=lam, margin=args.margin)
        eval_all(head_op, device, f"OPERATING-POINT LOSS (lam={lam}, margin={args.margin})")


if __name__ == "__main__":
    main()
