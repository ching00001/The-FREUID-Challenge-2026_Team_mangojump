"""DLC-2021 leakage audit + doctype-level train/holdout split for P0.5.

DLC-2021 is VIDEO FRAMES: external/dlc2021/{or,cg}/clips/images/{doctype}/{clip}/{frame}.jpg
Every earlier DLC number (probe 0.964, dlc_head holdouts, router PAD head) used
frame-level random splits -> same clip on both sides. Before spending 19h on the
P0.5 DoRA retrain, re-verify the load-bearing claim (or/cg signal is learnable)
without leakage, then write the actual split used for training/arbitration:

  probe A  StratifiedKFold rows        (= old, leaky, for reference)
  probe B  GroupKFold by clip          (honest same-doctype generalization)
  probe C  leave-one-DOCTYPE-out       (honest unseen-type generalization = private axis)

Split: holdout = half the DOCTYPES entirely (both or+cg present per side) ->
arbitration measures unseen-doctype x real-capture, matching the private threat.
Writes artifacts/dlc2021_train_index.csv + artifacts/dlc2021_holdout_index.csv.

  python -m src.dlc_split
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .data.paths import REPO_ROOT
from . import fgts

MEMBERS4 = ["dino", "dino_hplus", "siglip512", "dfn5b"]
CACHE = REPO_ROOT / "artifacts" / "fusion_cache"


def load_index():
    df = pd.read_csv(REPO_ROOT / "artifacts/dlc2021_index.csv")
    parts = df["abspath"].str.replace("\\", "/", regex=False).str.split("/")
    df["doctype"] = parts.str[-3]
    df["clip"] = parts.str[-3] + "/" + parts.str[-2]
    return df


def probe(X, y, splits, device="cuda"):
    aucs = []
    for tr, te in splits:
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        n1, n0 = max(1, int(y[tr].sum())), max(1, int((y[tr] == 0).sum()))
        h = fgts.train_head(X[tr], y[tr], device, epochs=150, pw=n0 / n1)
        aucs.append(roc_auc_score(y[te], fgts.predict(h, X[te], device)))
    return np.array(aucs)


def main():
    df = load_index()
    print(df.groupby(["doctype", "type"]).size().unstack(fill_value=0).to_string())
    print(f"rows={len(df)} clips={df['clip'].nunique()} doctypes={df['doctype'].nunique()}")

    X = np.concatenate([np.load(CACHE / f"eval_{m}__dlc2021.npz")["X"] for m in MEMBERS4], 1)
    y = df["label"].astype(float).values
    assert len(X) == len(df)

    a = probe(X, y, StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    print(f"\nprobe A rows-KFold (leaky, old)   AUC={a.mean():.4f} ± {a.std():.4f}")
    b = probe(X, y, GroupKFold(5).split(X, y, groups=df["clip"]))
    print(f"probe B clip-GroupKFold (honest)  AUC={b.mean():.4f} ± {b.std():.4f}")
    dts = sorted(df["doctype"].unique())
    c_splits, kept = [], []
    for dt in dts:
        te = np.where(df["doctype"] == dt)[0]
        tr = np.where(df["doctype"] != dt)[0]
        if len(set(y[te])) < 2:
            continue
        c_splits.append((tr, te)); kept.append(dt)
    c = probe(X, y, c_splits)
    print("probe C doctype-LOO (private axis) AUC="
          + " ".join(f"{dt}:{v:.3f}" for dt, v in zip(kept, c))
          + f"  mean={c.mean():.4f}")

    # doctype-level split: alternate sorted doctypes -> both sides get or+cg mix
    tr_dts, ho_dts = dts[0::2], dts[1::2]
    tr = df[df["doctype"].isin(tr_dts)].drop(columns=["doctype", "clip"])
    ho = df[df["doctype"].isin(ho_dts)].drop(columns=["doctype", "clip"])
    print(f"\nsplit: train doctypes={tr_dts} n={len(tr)} "
          f"({int(tr['label'].sum())} cg / {int((tr['label'] == 0).sum())} or)")
    print(f"       holdout doctypes={ho_dts} n={len(ho)} "
          f"({int(ho['label'].sum())} cg / {int((ho['label'] == 0).sum())} or)")
    tr.to_csv(REPO_ROOT / "artifacts/dlc2021_train_index.csv", index=False)
    ho.to_csv(REPO_ROOT / "artifacts/dlc2021_holdout_index.csv", index=False)
    print("wrote artifacts/dlc2021_{train,holdout}_index.csv")


if __name__ == "__main__":
    main()
