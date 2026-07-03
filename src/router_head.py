"""P0 final piece: OOD-routed dual head — champion in-domain, PAD head far-OOD.

Facts this builds on (loto_head.py, dlc_head.py, all cache-only):
  - champion head DEFAULTS TO FRAUD on far-OOD real captures (DLC AUC 0.30);
  - the or/cg (original vs reprint) signal IS in the frozen features (probe 0.96);
  - one linear head cannot serve both domains (mixing poisons recap20 + gate).
So: route. w(x) = how far x sits from the digital-train feature distribution
(kNN distance, per-member L2-normalized concat). p = (1-w)*champion + w*PAD.
Threshold chosen so the public test set is untouched (w≈0) -> public LB
provably ≈ champion (gate must read ~0.000); only far-OOD private images
switch to the physically-motivated reprint direction instead of "unknown=fraud".

Hygiene: PAD head trained on half of DLC; routing evaluated on the other half.

  python -m src.router_head
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .data.paths import REPO_ROOT, load_test
from . import fgts
from .fusion import CACHE
from .metric import freuid_score
from .train_DINOV3L_512 import make_splits

CHAMPION_SUB = "subs/fusion_C1p_dlc5.csv"   # current public best = gate reference


def load_eval(members, split):
    Xs, y = [], None
    for m in members:
        d = np.load(CACHE / f"eval_{m}__{split}.npz")
        Xs.append(d["X"]); y = d["y"]
    return np.concatenate(Xs, 1), y


def blocknorm(X, dims):
    """L2-normalize each member's block so no member dominates the distance."""
    out, i = [], 0
    for d in dims:
        B = X[:, i:i + d]
        out.append(B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8))
        i += d
    return np.concatenate(out, 1).astype(np.float32)


@torch.no_grad()
def knn_dist(Q, R, device, k=10, bs=2048, n_blocks=1):
    """Mean distance to the k nearest reference rows. Rows are concats of
    n_blocks unit blocks, so <q,r> in [-n_blocks, n_blocks]; normalize to a
    [0, 2] cosine-like distance for readability (ranking unaffected)."""
    Rt = torch.tensor(R, device=device)
    ds = []
    for i in range(0, len(Q), bs):
        q = torch.tensor(Q[i:i + bs], device=device)
        sim = q @ Rt.T / n_blocks                        # [b, n_ref] in [-1, 1]
        top = sim.topk(k, dim=1).values.mean(1)
        ds.append((1 - top).cpu())
    return torch.cat(ds).numpy()


def pct(a):
    q = np.percentile(a, [1, 50, 99, 100])
    return f"p1={q[0]:.4f} p50={q[1]:.4f} p99={q[2]:.4f} max={q[3]:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="dino,dino_hplus,siglip512,dfn5b")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="subs/fusion_C1_routed.csv")
    ap.add_argument("--pad_splits", default="dlc2021",
                    help="comma-sep eval splits the PAD head trains on "
                         "(e.g. dlc2021,sidtdclips = reprint + content-forgery axes)")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    members = args.members.split(",")

    # cached features
    tr_df, _ = make_splits(42, 0)
    tr_df = tr_df[tr_df["is_digital"]].reset_index(drop=True)
    Ftr, Fte, ytr, dims = [], [], None, []
    for m in members:
        d = np.load(CACHE / f"{m}.npz")
        Ftr.append(d["Xtr"]); Fte.append(d["Xte"]); ytr = d["ytr"]
        dims.append(d["Xtr"].shape[1])
    Xtr = np.concatenate(Ftr, 1); Xte = np.concatenate(Fte, 1)
    Xc, yc = load_eval(members, "cleanref")
    Xr, yr = load_eval(members, "recap20")
    pads = {s: load_eval(members, s) for s in args.pad_splits.split(",")}
    Xp_all = np.concatenate([X for X, _ in pads.values()])
    yp_all = np.concatenate([y for _, y in pads.values()])

    # 1) router: kNN distance to digital train, per-member-normalized space
    Ntr = blocknorm(Xtr, dims)
    D = {name: knn_dist(blocknorm(X, dims), Ntr, device, k=args.k, n_blocks=len(dims))
         for name, X in [("cleanref", Xc), ("recap20", Xr), ("test", Xte),
                         ("pad", Xp_all)]}
    print("kNN distance to digital train (routing signal):")
    for name, d in D.items():
        print(f"  {name:9s} {pct(d)}")
    sep = roc_auc_score(np.r_[np.zeros(len(D['test'])), np.ones(len(D['pad']))],
                        np.r_[D['test'], D['pad']])
    print(f"router separation test-vs-pad AUC = {sep:.4f}")

    # threshold: ramp starts above everything public-like, saturates below pad mass
    t0 = max(np.percentile(D["test"], 99.9), D["recap20"].max())
    t1 = np.percentile(D["pad"], 10)
    print(f"ramp t0={t0:.4f} (public p99.9 / recap20 max) -> t1={t1:.4f} (pad p10)")
    if t1 <= t0:
        print("WARNING: ramp inverted — domains overlap, routing unsafe as-is")

    def w(d):
        return np.clip((d - t0) / max(t1 - t0, 1e-6), 0, 1)

    # 2) heads: champion (full clean train) + PAD (half of each pad source;
    #    other halves stay out as honest holdouts)
    n1, n0 = max(1, int(ytr.sum())), max(1, int((ytr == 0).sum()))
    champ_head = fgts.train_head(Xtr, ytr, device, epochs=150, pw=n0 / n1)
    rng = np.random.default_rng(0)
    halves = {}
    for s, (X, y) in pads.items():
        perm = rng.permutation(len(X))
        halves[s] = (perm[:len(X) // 2], perm[len(X) // 2:])
    Xp_tr = np.concatenate([X[halves[s][0]] for s, (X, _) in pads.items()])
    yp_tr = np.concatenate([y[halves[s][0]] for s, (_, y) in pads.items()])
    n1d = max(1, int(yp_tr.sum())); n0d = max(1, int((yp_tr == 0).sum()))
    pad_head = fgts.train_head(Xp_tr, yp_tr, device, epochs=150, pw=n0d / n1d)

    def routed(X, d):
        pc_ = fgts.predict(champ_head, X, device)
        pp_ = fgts.predict(pad_head, X, device)
        ww = w(d)
        return (1 - ww) * pc_ + ww * pp_, ww

    # 3) eval: per-source holdout halves (the real test) + cleanref/recap20
    #    (must not move). Pad distances are per-source slices of D["pad"].
    off, dstart = {}, 0
    for s, (X, _) in pads.items():
        off[s] = dstart; dstart += len(X)
    evals = [(f"{s}-holdout", X[halves[s][1]], y[halves[s][1]],
              D["pad"][off[s] + halves[s][1]]) for s, (X, y) in pads.items()]
    evals += [("cleanref", Xc, yc, D["cleanref"]), ("recap20", Xr, yr, D["recap20"])]
    for name, X, y, d in evals:
        p, ww = routed(X, d)
        pc_ = fgts.predict(champ_head, X, device)
        print(f"[{name:18s}] routed AUC={roc_auc_score(y, p):.4f} "
              f"(champ {roc_auc_score(y, pc_):.4f}) genuine_p={p[y == 0].mean():.3f} "
              f"fraud_p={p[y == 1].mean() if (y == 1).any() else float('nan'):.3f} "
              f"| routed_frac(w>.01)={float((ww > .01).mean()):.4f}")

    # 4) test: gate + sub (deployment PAD head retrained on ALL pad data —
    #    the half-splits above were only for honest holdout numbers)
    n1a = max(1, int(yp_all.sum())); n0a = max(1, int((yp_all == 0).sum()))
    pad_full = fgts.train_head(Xp_all, yp_all, device, epochs=150, pw=n0a / n1a)
    te_ids = load_test().df["id"].astype(str).tolist()
    champ = (pd.read_csv(REPO_ROOT / CHAMPION_SUB, dtype={"id": str})
             .set_index("id").loc[te_ids, "label"].to_numpy())
    pseudo = (champ > .5).astype(int)
    wt = w(D["test"])
    pt = (1 - wt) * fgts.predict(champ_head, Xte, device) + \
        wt * fgts.predict(pad_full, Xte, device)
    r = freuid_score(pseudo, pt)
    mid = int(((pt >= .01) & (pt <= .99)).sum())
    print(f"\n[test] routed_frac(w>.01)={float((wt > .01).mean()):.5f} "
          f"gate pseudoFREUID={r.freuid:.5f} mid={mid}/{len(pt)}")
    fgts.write_sub(te_ids, pt, args.out)


if __name__ == "__main__":
    main()
