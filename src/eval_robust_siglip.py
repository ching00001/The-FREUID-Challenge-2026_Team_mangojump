"""Private-axis proxy: does a SigLIP-DoRA run survive recapture?

The public LB (digital) saturates; the PRIVATE test emphasises captured/recapture
examples + unseen document types and penalises reliance on generator-specific
traces. This measures the one thing we can locally: how much a trained model's
fraud separation degrades under simulated print-and-capture (recapture.py) on the
held-out canary digital, plus its score on the 20 real recaptured samples.

A model that stays separable under recapture is a better private-axis bet than
one that only wins clean/public. Inference-only; loads EMA adapters over the
frozen backbone (same as infer_multicrop). No retraining, no submission.

Usage:
  python -m src.eval_robust_siglip --run 20260612_134841_siglip512_dora_full
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .aug.recapture import recapture
from .data.paths import load_train
from .experiment import EXP_ROOT
from .infer_multicrop import load_run
from .metric import freuid_score

PROFILES = ["clean", "rc0.3", "rc0.5", "rc0.7", "rc0.9"]


def make_canary(seed=42, loto_types=""):
    """Default: canary = all recaptured + 0.5% stratified digital.

    loto_types="A,B": digital eval set = a sample of ONLY those held-out types,
    so the recapture curve measures the COMPOUND private worst case:
    unseen document type AND captured."""
    df = load_train().df[["id", "abspath", "is_digital", "label", "type"]].copy()
    if loto_types:
        held = {t.strip() for t in loto_types.split(",")}
        hd = df[df["is_digital"] & df["type"].isin(held)]
        val_idx = hd.groupby(["type", "label"], group_keys=False).apply(
            lambda g: g.sample(min(len(g), 600), random_state=seed)).index
    else:
        val_idx = (df[df["is_digital"]].groupby(["type", "label"], group_keys=False)
                   .apply(lambda g: g.sample(frac=0.005, random_state=seed)).index)
    val_mask = (~df["is_digital"]) | df.index.isin(val_idx)
    return df[val_mask].reset_index(drop=True)


class ProfileDS(Dataset):
    def __init__(self, df, profile, img, mean, std):
        self.paths = df["abspath"].tolist()
        self.labels = df["label"].astype(float).tolist()
        self.profile, self.img = profile, img
        self.mean = np.array(mean, np.float32); self.std = np.array(std, np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB").resize((self.img, self.img))
        a = np.asarray(im)
        if self.profile.startswith("rc"):
            a = recapture(a, strength=float(self.profile[2:]), p=1.0,
                          rng=np.random.default_rng(hash(self.paths[i]) % (2**32)))
        x = (a.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), self.labels[i]


@torch.no_grad()
def predict(model, ds, device, batch=32):
    ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    ps, ys = [], []
    for x, y in ld:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ps.append(torch.sigmoid(model(x.to(device)).float()).cpu().numpy())
        ys.append(np.asarray(y, dtype=float))
    return np.concatenate(ps), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--profiles", nargs="*", default=PROFILES)
    ap.add_argument("--loto_types", default="",
                    help="eval digital on ONLY these held-out types (compound OOD proxy)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    model, img, mean, std = load_run(args.run, device)
    va = make_canary(loto_types=args.loto_types)
    if args.loto_types:
        print(f"** COMPOUND OOD eval: held-out types {args.loto_types} under recapture **")
    dig = va["is_digital"].values
    rec = va[~va["is_digital"]].reset_index(drop=True)   # 20 real recaptured
    print(f"run={args.run} img={img} | canary digital={int(dig.sum())} "
          f"| real recaptured={len(rec)}")
    print(f"\n{'profile':10s} {'FREUID':>8s} {'AuDET':>7s} {'APCER@1%':>9s} {'AUC':>7s}  (held-out digital under recapture)")

    rows = []
    for p in args.profiles:
        ps, ys = predict(model, ProfileDS(va[dig].reset_index(drop=True), p, img, mean, std), device)
        r = freuid_score(ys, ps)
        print(f"{p:10s} {r.freuid:8.4f} {r.audet:7.4f} {r.apcer_at_1pct_bpcer:9.4f} {r.roc_auc:7.4f}")
        rows.append({"profile": p, **{k: getattr(r, k) for k in
                     ("freuid", "audet", "apcer_at_1pct_bpcer", "roc_auc")}})

    # the 20 real recaptured (clean — they are already physically captured)
    rp, ry = predict(model, ProfileDS(rec, "clean", img, mean, std), device)
    try:
        ho = roc_auc_score(ry, rp)
    except ValueError:
        ho = float("nan")
    print(f"\nreal recaptured (n={len(rec)}): AUC={ho:.4f} "
          f"gap={rp[ry==1].mean()-rp[ry==0].mean():+.4f}")
    pd.DataFrame(rows).to_csv(EXP_ROOT / args.run / "robust_siglip.csv", index=False)
    print(f"wrote {EXP_ROOT/args.run/'robust_siglip.csv'}")


if __name__ == "__main__":
    main()
