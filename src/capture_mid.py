"""Two-condition router: PAD iff (physical capture) AND (distance > public p99).

capture_hard closed the mid-band (holdouts 1.0) but rerouted ALL public
captures (~5%) to the external-data PAD head -> gate 0.0149 / mid 5.7% = too
aggressive in-domain. The distance router was too timid (t0 at p99.9 pinned by
recap20 max). Combine: capture head says WHAT (physical), a lower distance
floor (test p99) says OOD-ISH — together they route OOD captures (sidtd p50
0.27 > test p99 ~0.245) while touching only the top ~1% of public, and those
switch to a head that TIES champ on known-type recaptures (0.93 vs 0.93).

  python -m src.capture_mid
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .data.paths import REPO_ROOT, load_test
from . import fgts
from .metric import freuid_score
from .hybrid_routed import load_eval, load_train_test, BASE, PADM, GATE_REF
from .router_head import blocknorm, knn_dist

PAD_SPLITS = ["dlc2021", "sidtdclips"]
OUT = "subs/fusion_capture_mid.csv"


def main():
    global BASE, PADM, GATE_REF, OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=",".join(BASE))
    ap.add_argument("--padm", default=",".join(PADM))
    ap.add_argument("--gate_ref", default=GATE_REF)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    BASE, PADM = args.base.split(","), args.padm.split(",")
    GATE_REF, OUT = args.gate_ref, args.out

    device = "cuda"
    torch.manual_seed(0)
    Xtr, Xte, ytr, dims = load_train_test(BASE)
    n1, n0 = max(1, int(ytr.sum())), max(1, int((ytr == 0).sum()))
    champ = fgts.train_head(Xtr, ytr, device, epochs=150, pw=n0 / n1)

    rng = np.random.default_rng(0)
    pads5 = {s: load_eval(BASE, s) for s in PAD_SPLITS}
    pads6 = {s: load_eval(PADM, s) for s in PAD_SPLITS}
    halves = {}
    for s, (X, _) in pads5.items():
        perm = rng.permutation(len(X))
        halves[s] = (perm[:len(X) // 2], perm[len(X) // 2:])

    # capture head (5-way feats)
    dig = rng.choice(len(Xtr), 20000, replace=False)
    Xcap = np.concatenate([Xtr[dig]] + [pads5[s][0][halves[s][0]] for s in PAD_SPLITS])
    ycap = np.concatenate([np.zeros(len(dig))]
                          + [np.ones(len(halves[s][0])) for s in PAD_SPLITS])
    cap = fgts.train_head(Xcap, ycap, device, epochs=150,
                          pw=max(1, int((ycap == 0).sum())) / max(1, int(ycap.sum())))

    # PAD heads (6-way feats)
    Xp = np.concatenate([pads6[s][0][halves[s][0]] for s in PAD_SPLITS])
    yp = np.concatenate([pads6[s][1][halves[s][0]] for s in PAD_SPLITS])
    pad_half = fgts.train_head(Xp, yp, device, epochs=150,
                               pw=max(1, int((yp == 0).sum())) / max(1, int(yp.sum())))
    Xpa = np.concatenate([pads6[s][0] for s in PAD_SPLITS])
    ypa = np.concatenate([pads6[s][1] for s in PAD_SPLITS])
    pad_full = fgts.train_head(Xpa, ypa, device, epochs=150,
                               pw=max(1, int((ypa == 0).sum())) / max(1, int(ypa.sum())))

    # distances (5-way space) + floor at public p99
    Ntr = blocknorm(Xtr, dims)

    def dist(X):
        return knn_dist(blocknorm(X, dims), Ntr, device, n_blocks=len(dims))

    d_te = dist(Xte)
    floor = np.percentile(d_te, 99)
    print(f"distance floor = test p99 = {floor:.4f}")

    def route_mask(X5, d):
        return (fgts.predict(cap, X5, device) > .5) & (d > floor)

    # evals
    for s in PAD_SPLITS:
        ho = halves[s][1]
        X5, y = pads5[s][0][ho], pads5[s][1][ho]
        X6 = pads6[s][0][ho]
        m = route_mask(X5, dist(X5))
        p = np.where(m, fgts.predict(pad_half, X6, device), fgts.predict(champ, X5, device))
        print(f"[{s}-holdout] AUC={roc_auc_score(y, p):.4f} "
              f"genuine_p={p[y == 0].mean():.3f} fraud_p={p[y == 1].mean():.3f} "
              f"routed={float(m.mean()):.3f}")
    for s in ["cleanref", "recap20"]:
        X5, y = load_eval(BASE, s)
        X6, _ = load_eval(PADM, s)
        m = route_mask(X5, dist(X5))
        p = np.where(m, fgts.predict(pad_half, X6, device), fgts.predict(champ, X5, device))
        print(f"[{s}] AUC={roc_auc_score(y, p):.4f} genuine_p={p[y == 0].mean():.3f} "
              f"routed={float(m.mean()):.3f}")

    # test + gate
    _, Xte6, _, _ = load_train_test(PADM)
    m = route_mask(Xte, d_te)
    pt = np.where(m, fgts.predict(pad_full, Xte6, device), fgts.predict(champ, Xte, device))
    te_ids = load_test().df["id"].astype(str).tolist()
    ref = (pd.read_csv(REPO_ROOT / GATE_REF, dtype={"id": str})
           .set_index("id").loc[te_ids, "label"].to_numpy())
    r = freuid_score((ref > .5).astype(int), pt)
    mid = int(((pt >= .01) & (pt <= .99)).sum())
    print(f"\n[test] routed={float(m.mean()):.5f} gate pseudoFREUID={r.freuid:.5f} "
          f"mid={mid}/{len(pt)}")
    fgts.write_sub(te_ids, pt, OUT)


if __name__ == "__main__":
    main()
