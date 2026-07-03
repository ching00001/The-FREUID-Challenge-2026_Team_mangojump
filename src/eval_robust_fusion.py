"""Is the CHAMPION 2-way fusion (dino-FGTS ⊕ siglip512, LB 0.00426) recapture-robust?

The fusion head trains on CLEAN digital -> it has no incentive to prefer
recapture-robust features. DINOv3 member is recapture-fragile (real-recap ho 0.738),
SigLIP512 is robust (0.917). Which did the learned fusion inherit on the dominant
PRIVATE axis (recapture)?

We avoid the simulated recapture profiles (recapture.py) — memory: that sim is
untrustworthy (helped local proxy, HURT public LB). We score only the GOLD slice:
the 20 REAL recaptured train samples (is_digital==False), with a same-type clean
digital reference for contrast. Head reconstructed identically from the cached
clean-train features (fusion_cache/*.npz) so it == the champion's head.

  python -m src.eval_robust_fusion
  python -m src.eval_robust_fusion --arbiter artifacts/dlc2021_index.csv   # real DLC-2021 recaptures
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .data.paths import REPO_ROOT, load_train
from . import fgts
from .fusion import MEMBERS, CACHE, forensic_feats
from .metric import freuid_score
from .train_DINOV3L_512 import make_splits


def feats_for_splits(name, spec, dfs, device):
    """Extract one member's pooled features for MANY dfs, loading the backbone
    (and computing the Fisher token selection) exactly ONCE. Returns a list of
    feature matrices, one per input df. Loading each backbone once — instead of
    once per eval split — is both faster and avoids the 16GB-card OOM caused by
    memory fragmenting across repeated load/free cycles."""
    if spec["kind"] == "forensic":
        return [forensic_feats(spec["run"], d, d, device, spec.get("img", 512))[0]
                for d in dfs]
    bb, img, mean, std, npfx = fgts.load_backbone(spec["run"], device)
    if spec["kind"] == "fgts":
        tr_df, _ = make_splits(42, 0)
        tr_df = tr_df[tr_df["is_digital"]].reset_index(drop=True)
        rank, _ = fgts.fisher_ranking(bb, tr_df, npfx, img, mean, std, device)
        idx = {name: rank[:spec["k"]].sort().values}
    else:
        T = (img // bb.patch_embed.patch_size[0]) ** 2
        idx = {name: torch.arange(T)}
    outs = []
    for d in dfs:
        X, _ = fgts.pooled(bb, d, npfx, img, mean, std, device, idx, lab=False)
        outs.append(X[name])
        torch.cuda.empty_cache()
    del bb; torch.cuda.empty_cache()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arbiter", default="",
                    help="csv (id,abspath,label,is_digital,type) of REAL recaptures, e.g. "
                         "artifacts/dlc2021_index.csv; replaces the n=20 in-train recaptures")
    ap.add_argument("--members", default="dino,siglip512",
                    help="fusion members to reconstruct + evaluate")
    args = ap.parse_args()
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    members = args.members.split(",")

    # 1) reconstruct champion head from cached CLEAN-train features
    Ftr, ytr = [], None
    for m in members:
        d = np.load(CACHE / f"{m}.npz")
        Ftr.append(d["Xtr"]); ytr = d["ytr"]
    Xtr = np.concatenate(Ftr, 1)
    n1, n0 = max(1, int((ytr == 1).sum())), max(1, int((ytr == 0).sum()))
    head = fgts.train_head(Xtr, ytr, device, epochs=150, pw=n0 / n1)
    print(f"head rebuilt on {Xtr.shape} (champion {len(members)}-way: {','.join(members)})")

    # 2) eval sets: 20 REAL recaptured + same-type CLEAN digital reference
    df = load_train().df
    rec = df[~df["is_digital"]].reset_index(drop=True)
    types = set(rec["type"])
    clean = (df[df["is_digital"] & df["type"].isin(types)]
             .groupby(["type", "label"], group_keys=False)
             .apply(lambda g: g.sample(min(len(g), 200), random_state=0))
             .reset_index(drop=True))
    print(f"real recaptured n={len(rec)} ({int((rec.label==1).sum())} fraud / "
          f"{int((rec.label==0).sum())} genuine) | clean ref n={len(clean)}")

    # 3) assemble eval splits, then extract features member-by-member (each
    #    backbone loaded once across ALL splits) to fit the 16GB card.
    splits = [("clean digital (same types)", clean),
              ("REAL recaptured in-train (n=20 gold)", rec)]
    if args.arbiter:
        adf = pd.read_csv(REPO_ROOT / args.arbiter)
        adf = adf[adf["abspath"].map(lambda x: __import__("os").path.exists(str(x)))].reset_index(drop=True)
        print(f"\n=== ARBITER {args.arbiter}: {len(adf)} real images "
              f"({int((adf.label==0).sum())} genuine / {int((adf.label==1).sum())} reproduction) ===")
        splits.append((f"DLC-2021 arbiter [{','.join(sorted(set(adf['type'])))}]", adf))

    dfs = [d for _, d in splits]
    per_split_feats = [[] for _ in splits]
    for m in members:
        print(f"  extracting member [{m}] across {len(dfs)} splits ...")
        outs = feats_for_splits(m, MEMBERS[m], dfs, device)
        for i, X in enumerate(outs):
            per_split_feats[i].append(X)

    for (tag, eval_df), F in zip(splits, per_split_feats):
        X = np.concatenate(F, 1)
        p = fgts.predict(head, X, device)
        y = eval_df["label"].astype(float).values
        auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
        gap = p[y == 1].mean() - p[y == 0].mean()
        mid = ((p >= .01) & (p <= .99)).sum()
        extra = ""
        if len(set(y)) > 1:
            r = freuid_score(y, p)
            extra = f" FREUID={r.freuid:.4f} APCER@1%={r.apcer_at_1pct_bpcer:.4f}"
        print(f"\n[{tag}] AUC={auc:.4f} gap={gap:+.4f} mid={mid}/{len(p)} "
              f"fraud_p={p[y==1].mean():.3f} genuine_p={p[y==0].mean():.3f}{extra}")


if __name__ == "__main__":
    main()
