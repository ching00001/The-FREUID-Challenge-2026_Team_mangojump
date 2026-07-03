"""Combine multiple submission CSVs into one.

Only the ids whose test image exists on disk (the public 7,821) carry real
predictions; dummy-filled rows stay at --fill.

Methods (--method):
  rank   average of per-member rank (uniform [0,1]). Robust to scale, BUT maps
         everything to a smooth uniform -> DESTROYS confident bimodality. On
         FREUID this spiked APCER@1%BPCER (512+378 rank-mean = 0.066 vs single
         512 = 0.027). Avoid for this operating-point-sensitive metric.
  mean   average of probabilities. Preserves confidence: agreed-confident stays
         confident, only disagreements move to the middle. Default.
  logit  average in logit space (most confidence-preserving at the extremes).

Usage:
  python -m src.combine_subs --method mean --subs subs/a.csv subs/b.csv --out subs/ens.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data.paths import REPO_ROOT, load_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["rank", "mean", "logit"], default="mean")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="per-member weights (default equal)")
    ap.add_argument("--fill", type=float, default=0.5)
    args = ap.parse_args()

    present = set(load_test().df["id"].astype(str))
    base = pd.read_csv(REPO_ROOT / args.subs[0], dtype={"id": str})
    mask = base["id"].isin(present).values
    w = np.array(args.weights) if args.weights else np.ones(len(args.subs))
    w = w / w.sum()
    assert len(w) == len(args.subs), "weights must match number of members"
    print(f"rows={len(base)}  real preds={mask.sum()}  members={len(args.subs)}"
          f"  method={args.method}  weights={w.round(3).tolist()}")

    cols = []
    for s in args.subs:
        d = pd.read_csv(REPO_ROOT / s, dtype={"id": str})
        assert (d["id"].values == base["id"].values).all(), f"id order mismatch: {s}"
        p = d.loc[mask, "label"].to_numpy(dtype=np.float64)
        cols.append(p)
        print(f"  {s}: mean={p.mean():.4f} std={p.std():.4f}")

    P = np.vstack(cols)                       # (members, N)
    if args.method == "rank":
        vals = np.average([rankdata(p) / len(p) for p in cols], axis=0, weights=w)
    elif args.method == "logit":
        eps = 1e-6
        z = np.clip(P, eps, 1 - eps)
        z = np.log(z / (1 - z))
        zbar = np.average(z, axis=0, weights=w)
        vals = 1.0 / (1.0 + np.exp(-zbar))
    else:                                     # mean
        vals = np.average(P, axis=0, weights=w)

    out = base.copy()
    out["label"] = args.fill
    out.loc[mask, "label"] = np.clip(vals, 0, 1)
    path = Path(REPO_ROOT / args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"ensemble: mean={vals.mean():.4f} std={vals.std():.4f} -> wrote {path}")


if __name__ == "__main__":
    main()
