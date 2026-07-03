"""P0.5 arbitration: does swapping H+ -> H+dlcmix fix the fusion's private axis?

All cache-only (fusion_cache/*.npz + eval_{m}__{split}.npz). Rebuilds the 4-way
head for BOTH member sets (old champion vs dlcmix swap), scores:
  cleanref      public proxy (must stay perfect)
  recap20       known-type real captures
  dlc-holdout   UNSEEN doctypes x real capture (the private-axis verdict);
                the dlc train-half is also shown but the swap member SAW it.
Then pseudo-FREUID-gates the candidate subs vs the routed champion.

  python -m src.p05_eval
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data.paths import REPO_ROOT, load_test
from . import fgts
from .fusion import CACHE
from .metric import freuid_score

OLD = ["dino", "dino_hplus", "siglip512", "dfn5b", "dino_hplus_dlc"]
NEW = OLD + ["dino_hplus_ds"]
GATE_REF = "subs/fusion_C1p_dlc5.csv"
GATE_TARGETS = ["subs/fusion_C1p_ds6.csv", "subs/fusion_C1p_ds6_routed.csv"]
SIDTD_HO_DT = {"aze_passport", "est_id", "grc_passport",
               "rus_internalpassport", "svk_id"}


def load_eval(members, split):
    Xs, y = [], None
    for m in members:
        d = np.load(CACHE / f"eval_{m}__{split}.npz")
        Xs.append(d["X"]); y = d["y"]
    return np.concatenate(Xs, 1), y


def main():
    device = "cuda"
    import torch; torch.manual_seed(0)
    idx = pd.read_csv(REPO_ROOT / "artifacts/dlc2021_index.csv")
    ho_ids = set(pd.read_csv(REPO_ROOT / "artifacts/dlc2021_holdout_index.csv")["id"])
    ho = idx["id"].isin(ho_ids).values
    sidx = pd.read_csv(REPO_ROOT / "artifacts/sidtd_clips_index.csv")
    sho = sidx["type"].str.split("/").str[1].isin(SIDTD_HO_DT).values

    for tag, members in [("OLD 5-way (dlc)", OLD), ("NEW 6-way (+ds)", NEW)]:
        Ftr, ytr = [], None
        for m in members:
            d = np.load(CACHE / f"{m}.npz")
            Ftr.append(d["Xtr"]); ytr = d["ytr"]
        Xtr = np.concatenate(Ftr, 1)
        n1, n0 = max(1, int(ytr.sum())), max(1, int((ytr == 0).sum()))
        head = fgts.train_head(Xtr, ytr, device, epochs=150, pw=n0 / n1)

        Xd, yd = load_eval(members, "dlc2021")
        Xs, ys = load_eval(members, "sidtdclips")
        Xc, yc = load_eval(members, "cleanref")
        Xr, yr = load_eval(members, "recap20")
        print(f"\n=== {tag}: {','.join(members)} ===")
        for name, X, y in [("cleanref", Xc, yc), ("recap20", Xr, yr),
                           ("dlc-HOLDOUT (unseen doctypes)", Xd[ho], yd[ho]),
                           ("sidtd-HOLDOUT (unseen doctypes)", Xs[sho], ys[sho])]:
            p = fgts.predict(head, X, device)
            mid = int(((p >= .01) & (p <= .99)).sum())
            print(f"  [{name:36s}] AUC={roc_auc_score(y, p):.4f} "
                  f"genuine_p={p[y == 0].mean():.3f} "
                  f"fraud_p={p[y == 1].mean() if (y == 1).any() else float('nan'):.3f} "
                  f"mid={mid}/{len(p)}")

    ids = load_test().df["id"].astype(str).tolist()

    def load_sub(p):
        return (pd.read_csv(REPO_ROOT / p, dtype={"id": str})
                .set_index("id").loc[ids, "label"].to_numpy())

    pseudo = (load_sub(GATE_REF) > .5).astype(int)
    print(f"\npseudo-FREUID gate vs {GATE_REF}:")
    for t in GATE_TARGETS:
        if not (REPO_ROOT / t).exists():
            print(f"  {t:34s} (missing, skipped)"); continue
        p = load_sub(t)
        r = freuid_score(pseudo, p)
        mid = int(((p >= .01) & (p <= .99)).sum())
        print(f"  {t:34s} pseudoFREUID={r.freuid:.5f} mid={mid}/{len(p)}")


if __name__ == "__main__":
    main()
