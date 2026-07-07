"""Capture-classifier routing: attack the MID-BAND OOD gap.

Known weakness (P0.6): kNN-distance routing only fires on the far tail
(t0 pinned by the public test's own capture tail), so 78% of SIDTD sits in a
mid-band served by the clean head, which defaults-to-genuine on content-forged
physical documents. Distance conflates "how far" with "what kind".

Idea: route on WHAT the image is, not how far it sits — a binary
digital-vs-physical-capture head (fraud-label-free) on cached fusion features.
If it cleanly separates, physical captures (incl. mid-band) go to the dual-axis
PAD head; digital stays with the clean head (perfect there: cleanref 1.0,
unseen-type digital 1.0).

Decisive unknowns this measures:
  - PAD(6-way, DLC+SIDTD) quality on recap20 (known-type recaptures) — routing
    sends them to PAD now, so PAD must not lose to champ's 0.93 there;
  - fraction of PUBLIC test flagged as capture (public contains real captures;
    they'd all switch heads -> gate + upload is the only final judge);
  - mid-band recovery on sidtd-holdout vs the distance router's 0.79.

  python -m src.capture_router
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
from .hybrid_routed import load_eval, load_train_test, BASE, PADM, GATE_REF

PAD_SPLITS = ["dlc2021", "sidtdclips"]
OUT = "subs/fusion_capture_routed.csv"


def main():
    device = "cuda"
    torch.manual_seed(0)

    Xtr, Xte, ytr, dims = load_train_test(BASE)
    n1, n0 = max(1, int(ytr.sum())), max(1, int((ytr == 0).sum()))
    champ = fgts.train_head(Xtr, ytr, device, epochs=150, pw=n0 / n1)

    # ---- capture head (digital=0 vs physical-capture=1, fraud-label-free) ----
    rng = np.random.default_rng(0)
    pads5 = {s: load_eval(BASE, s) for s in PAD_SPLITS}
    halves = {}
    for s, (X, _) in pads5.items():
        perm = rng.permutation(len(X))
        halves[s] = (perm[:len(X) // 2], perm[len(X) // 2:])
    dig_idx = rng.choice(len(Xtr), 20000, replace=False)
    Xcap = np.concatenate([Xtr[dig_idx]]
                          + [pads5[s][0][halves[s][0]] for s in PAD_SPLITS])
    ycap = np.concatenate([np.zeros(len(dig_idx))]
                          + [np.ones(len(halves[s][0])) for s in PAD_SPLITS])
    ncp, ncn = max(1, int(ycap.sum())), max(1, int((ycap == 0).sum()))
    cap = fgts.train_head(Xcap, ycap, device, epochs=150, pw=ncn / ncp)

    def w_cap(X):
        return fgts.predict(cap, X, device)

    # ---- PAD head on 6-way feats (train halves; full for deployment) --------
    pads6 = {s: load_eval(PADM, s) for s in PAD_SPLITS}
    Xp = np.concatenate([pads6[s][0][halves[s][0]] for s in PAD_SPLITS])
    yp = np.concatenate([pads6[s][1][halves[s][0]] for s in PAD_SPLITS])
    n1p, n0p = max(1, int(yp.sum())), max(1, int((yp == 0).sum()))
    pad_half = fgts.train_head(Xp, yp, device, epochs=150, pw=n0p / n1p)
    Xpa = np.concatenate([pads6[s][0] for s in PAD_SPLITS])
    ypa = np.concatenate([pads6[s][1] for s in PAD_SPLITS])
    pad_full = fgts.train_head(Xpa, ypa, device, epochs=150,
                               pw=max(1, int((ypa == 0).sum())) / max(1, int(ypa.sum())))

    # ---- diagnostics ---------------------------------------------------------
    print("capture head p(physical) distributions:")
    for s in ["cleanref", "recap20"]:
        X, _ = load_eval(BASE, s)
        print(f"  {s:9s} p50={np.median(w_cap(X)):.4f} p99={np.percentile(w_cap(X), 99):.4f}")
    wte = w_cap(Xte)
    print(f"  test      p50={np.median(wte):.4f} p99={np.percentile(wte, 99):.4f} "
          f"frac>.5={float((wte > .5).mean()):.4f}")
    for s in PAD_SPLITS:
        ho = halves[s][1]
        print(f"  {s}-ho p50={np.median(w_cap(pads5[s][0][ho])):.4f}")

    # PAD alone on recap20 (6-way feats) — the make-or-break number
    Xr6, yr = load_eval(PADM, "recap20")
    ppr = fgts.predict(pad_half, Xr6, device)
    Xr5, _ = load_eval(BASE, "recap20")
    pcr = fgts.predict(champ, Xr5, device)
    print(f"\nrecap20: PAD-alone AUC={roc_auc_score(yr, ppr):.4f} "
          f"genuine_p={ppr[yr == 0].mean():.3f} | champ AUC={roc_auc_score(yr, pcr):.4f}")

    # ---- routed eval ---------------------------------------------------------
    def routed(X5, X6):
        ww = w_cap(X5)
        return ((1 - ww) * fgts.predict(champ, X5, device)
                + ww * fgts.predict(pad_half, X6, device), ww)

    for s in PAD_SPLITS:
        ho = halves[s][1]
        p, ww = routed(pads5[s][0][ho], pads6[s][0][ho])
        y = pads5[s][1][ho]
        print(f"[{s}-holdout] cap-routed AUC={roc_auc_score(y, p):.4f} "
              f"genuine_p={p[y == 0].mean():.3f} fraud_p={p[y == 1].mean():.3f} "
              f"routed_frac={float((ww > .5).mean()):.3f}")
    for s in ["cleanref", "recap20"]:
        X5, y = load_eval(BASE, s)
        X6, _ = load_eval(PADM, s)
        p, ww = routed(X5, X6)
        print(f"[{s}] cap-routed AUC={roc_auc_score(y, p):.4f} "
              f"genuine_p={p[y == 0].mean():.3f} routed_frac={float((ww > .5).mean()):.3f}")

    # ---- test + gate: soft blend AND hard switch (mid-band control) ----------
    _, Xte6, _, _ = load_train_test(PADM)
    te_ids = load_test().df["id"].astype(str).tolist()
    ref = (pd.read_csv(REPO_ROOT / GATE_REF, dtype={"id": str})
           .set_index("id").loc[te_ids, "label"].to_numpy())
    pc_te = fgts.predict(champ, Xte, device)
    pp_te = fgts.predict(pad_full, Xte6, device)
    hard = (wte > .5).astype(float)
    for tag, pt, out in [("soft", (1 - wte) * pc_te + wte * pp_te, OUT),
                         ("hard", np.where(hard > 0, pp_te, pc_te),
                          "subs/fusion_capture_hard.csv")]:
        r = freuid_score((ref > .5).astype(int), pt)
        mid = int(((pt >= .01) & (pt <= .99)).sum())
        print(f"[test/{tag}] routed_frac(w>.5)={float((wte > .5).mean()):.5f} "
              f"gate pseudoFREUID={r.freuid:.5f} mid={mid}/{len(pt)}")
        fgts.write_sub(te_ids, pt, out)

    # hard-routed holdout/eval numbers (soft printed above)
    for s in PAD_SPLITS:
        ho = halves[s][1]
        ww = w_cap(pads5[s][0][ho]) > .5
        p = np.where(ww, fgts.predict(pad_half, pads6[s][0][ho], device),
                     fgts.predict(champ, pads5[s][0][ho], device))
        y = pads5[s][1][ho]
        print(f"[{s}-holdout/hard] AUC={roc_auc_score(y, p):.4f} "
              f"genuine_p={p[y == 0].mean():.3f} fraud_p={p[y == 1].mean():.3f}")
    X5, y = load_eval(BASE, "recap20"); X6, _ = load_eval(PADM, "recap20")
    ww = w_cap(X5) > .5
    p = np.where(ww, fgts.predict(pad_half, X6, device), fgts.predict(champ, X5, device))
    print(f"[recap20/hard] AUC={roc_auc_score(y, p):.4f} genuine_p={p[y == 0].mean():.3f}")


if __name__ == "__main__":
    main()
