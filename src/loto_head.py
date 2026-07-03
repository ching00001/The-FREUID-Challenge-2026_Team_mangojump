"""P0: type-robust fusion head via leave-one-type-out (LOTO) ensembling.

DLC-2021 arbitration showed the champion collapses on UNSEEN-TYPE x REAL-CAPTURE
(genuine_p 0.81, AUC reversal) — it defaults to fraud on document types it never
saw, which is exactly the private axis (2 unseen types, real-capture emphasis,
generator-trace penalty). The backbones are frozen and shared; the failure lives
in the HEAD's extrapolation. Zero-backbone-cost fix attempt: train one head per
"leave this type out" split and ensemble them — every head has practiced scoring
a type it never saw, so type-identity novelty should stop mapping onto fraud.

Eval slices (features disk-cached per member so re-runs are CPU-cheap):
  cleanref   clean digital, 200/type/label      -> public proxy (must NOT regress)
  recap20    the 20 real in-train recaptures    -> known-type x real-capture gold
  dlc2021    real recaptures, unseen EU types   -> pessimistic private proxy
Plus a direct diagnostic: head-without-type-t scored on type-t clean digital
(= unseen-type behavior measured per type, no DLC domain gap involved).

Gate: pseudo-FREUID vs the champion sub — ONE-WAY brake (catches LB-1.0-level
collapse only; never veto an upload on correlation grounds).

  python -m src.loto_head                      # champion 4-way members
  python -m src.loto_head --members dino,siglip512
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .data.paths import REPO_ROOT, load_train, load_test
from . import fgts
from .fusion import MEMBERS, CACHE
from .eval_robust_fusion import feats_for_splits
from .metric import freuid_score
from .train_DINOV3L_512 import make_splits

CHAMPION_SUB = "subs/fusion_C1_dfn5b.csv"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@torch.no_grad()
def head_logits(head, X, device, bs=65536):
    zs = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i + bs], dtype=torch.float32, device=device)
        zs.append(head(x).squeeze(-1).cpu())
    return torch.cat(zs).numpy()


def eval_splits():
    """Same constructions as eval_robust_fusion (deterministic, cache-safe)."""
    df = load_train().df
    rec = df[~df["is_digital"]].reset_index(drop=True)
    types = set(rec["type"])
    clean = (df[df["is_digital"] & df["type"].isin(types)]
             .groupby(["type", "label"], group_keys=False)
             .apply(lambda g: g.sample(min(len(g), 200), random_state=0))
             .reset_index(drop=True))
    adf = pd.read_csv(REPO_ROOT / "artifacts/dlc2021_index.csv")
    adf = adf[adf["abspath"].map(lambda p: os.path.exists(str(p)))].reset_index(drop=True)
    return [("cleanref", clean), ("recap20", rec), ("dlc2021", adf)]


def member_eval_feats(members, splits, device):
    """{split: concat feature matrix} with a per-(member, split) disk cache, so
    the GPU extraction pass happens once and every later head experiment is
    pure CPU/seconds."""
    out = {s: [] for s, _ in splits}
    for m in members:
        missing = [(s, d) for s, d in splits if not (CACHE / f"eval_{m}__{s}.npz").exists()]
        if missing:
            print(f"  [{m}] extracting {[s for s, _ in missing]} ...")
            Xs = feats_for_splits(m, MEMBERS[m], [d for _, d in missing], device)
            for (s, d), X in zip(missing, Xs):
                np.savez(CACHE / f"eval_{m}__{s}.npz", X=X, y=d["label"].astype(float).values)
        for s, _ in splits:
            out[s].append(np.load(CACHE / f"eval_{m}__{s}.npz")["X"])
    return {s: np.concatenate(v, 1) for s, v in out.items()}


def report(tag, y, scores):
    for name, p in scores.items():
        auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
        mid = int(((p >= .01) & (p <= .99)).sum())
        extra = ""
        if len(set(y)) > 1:
            r = freuid_score(y, p)
            extra = f" FREUID={r.freuid:.4f} APCER@1%={r.apcer_at_1pct_bpcer:.4f}"
        print(f"[{tag}] {name:14s} AUC={auc:.4f} mid={mid}/{len(p)} "
              f"fraud_p={p[y == 1].mean() if (y == 1).any() else float('nan'):.3f} "
              f"genuine_p={p[y == 0].mean() if (y == 0).any() else float('nan'):.3f}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="dino,dino_hplus,siglip512,dfn5b")
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    members = args.members.split(",")

    # 1) cached train/test features + row-aligned types (alignment asserted)
    tr_df, _ = make_splits(42, 0)
    tr_df = tr_df[tr_df["is_digital"]].reset_index(drop=True)
    Ftr, Fte, ytr = [], [], None
    for m in members:
        d = np.load(CACHE / f"{m}.npz")
        Ftr.append(d["Xtr"]); Fte.append(d["Xte"]); ytr = d["ytr"]
    Xtr = np.concatenate(Ftr, 1); Xte = np.concatenate(Fte, 1)
    assert len(tr_df) == len(Xtr) and (tr_df["label"].values == ytr).all(), \
        "fusion_cache rows != make_splits(42,0) digital train — cannot recover types"
    types = tr_df["type"].values
    type_list = sorted(set(types))
    print(f"train {Xtr.shape} test {Xte.shape} | types={type_list}")

    # 2) heads: champion full + one per left-out type
    n1, n0 = max(1, int(ytr.sum())), max(1, int((ytr == 0).sum()))
    heads = {"full": fgts.train_head(Xtr, ytr, device, epochs=args.epochs, pw=n0 / n1)}
    for t in type_list:
        m = types != t
        n1t, n0t = max(1, int(ytr[m].sum())), max(1, int((ytr[m] == 0).sum()))
        heads[t] = fgts.train_head(Xtr[m], ytr[m], device, epochs=args.epochs, pw=n0t / n1t)
        print(f"  head w/o {t:14s} trained on n={int(m.sum())}")

    def strat_scores(X):
        z = {k: head_logits(h, X, device) for k, h in heads.items()}
        L = np.stack([z[t] for t in type_list])
        return {"full(champ)": sigmoid(z["full"]),
                "loto_prob": sigmoid(L).mean(0),
                "loto_logit": sigmoid(L.mean(0))}, z

    # 3) eval slices
    splits = eval_splits()
    F = member_eval_feats(members, splits, device)
    split_dfs = dict(splits)
    print()
    for s, _ in splits:
        y = split_dfs[s]["label"].astype(float).values
        scores, _ = strat_scores(F[s])
        report(s, y, scores)
        print()

    # 4) direct unseen-type diagnostic: head w/o type t on type-t CLEAN digital
    cdf = split_dfs["cleanref"]
    yc, tc = cdf["label"].astype(float).values, cdf["type"].values
    _, zc = strat_scores(F["cleanref"])
    print("unseen-type diagnostic (clean digital slice of the held-out type):")
    for t in type_list:
        m = tc == t
        for name in ("full", t):
            p = sigmoid(zc[name][m])
            print(f"  type={t:14s} head={'full' if name == 'full' else 'w/o-type':8s} "
                  f"AUC={roc_auc_score(yc[m], p):.4f} genuine_p={p[yc[m] == 0].mean():.3f} "
                  f"fraud_p={p[yc[m] == 1].mean():.3f}")

    # 5) test predictions + pseudo-FREUID gate vs champion + write subs
    te_ids = load_test().df["id"].astype(str).tolist()
    champ = (pd.read_csv(REPO_ROOT / CHAMPION_SUB, dtype={"id": str})
             .set_index("id").loc[te_ids, "label"].to_numpy())
    pseudo = (champ > .5).astype(int)
    te_scores, _ = strat_scores(Xte)
    print(f"\npseudo-FREUID gate vs {CHAMPION_SUB} (one-way brake only):")
    for name, p in te_scores.items():
        r = freuid_score(pseudo, p)
        agree = float(((p > .5) == (pseudo == 1)).mean())
        mid = int(((p >= .01) & (p <= .99)).sum())
        print(f"  {name:14s} pseudoFREUID={r.freuid:.5f} agree={agree:.4f} mid={mid}/{len(p)}")
        if name != "full(champ)":
            fgts.write_sub(te_ids, p, f"subs/fusion_C1_loto_{name.split('_')[1]}.csv")


if __name__ == "__main__":
    main()
