"""Standalone inference for a FULL-FT run -> submission CSV.

The trainer does inference inline at the end, but if a full-FT run is interrupted
(e.g. reboot) the per-epoch `model.pt` (full backbone+head state, EMA preferred)
on disk is still usable. This loads it and writes a hflip-TTA submission so no
training is wasted.

  python -m src.infer_fullft --run 20260627_173814_dinov3_l512_ft
  python -m src.infer_fullft --run <run> --out subs/dinov3_l512_ft.csv --no_ema
"""
from __future__ import annotations

import argparse
import types

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .data.paths import REPO_ROOT, load_test
from .experiment import EXP_ROOT
from .train_DINOV3L_512 import build_model, DS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--no_ema", action="store_true", help="use raw model weights, not EMA")
    args = ap.parse_args()
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    ck = torch.load(EXP_ROOT / args.run / "model.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    bargs = types.SimpleNamespace(backbone=a["backbone"], img_size=a.get("img_size", 0),
                                  full_ft=True, rank=0, alpha=0, attn_only=False)
    model, _n, img, mean, std = build_model(bargs)
    sd = ck["model"] if (args.no_ema or "ema" not in ck) else ck["ema"]
    model.load_state_dict(sd)
    model.eval().to(device)
    out = args.out or a.get("out", f"subs/{args.run}.csv")
    print(f"run={args.run} epoch={ck.get('epoch')} img={img} ema={'ema' in ck and not args.no_ema} -> {out}")

    ev = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(mean, std)])
    fl = T.Compose([T.Resize((img, img)), T.RandomHorizontalFlip(1.0), T.ToTensor(), T.Normalize(mean, std)])
    te = load_test().df.copy()
    probs = np.zeros(len(te))
    for tf in (ev, fl):
        ld = DataLoader(DS(te, tf, has_label=False), batch_size=16, shuffle=False,
                        num_workers=2, pin_memory=True)
        pp = []
        with torch.no_grad():
            for x, _ in ld:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pp.append(torch.sigmoid(model(x.to(device)).float()).cpu().numpy())
        probs += np.concatenate(pp)
    probs /= 2

    sub = pd.read_csv(REPO_ROOT / "sample_submission.csv", dtype={"id": str})
    smap = dict(zip(te["id"].astype(str), probs))
    sub["label"] = sub["id"].astype(str).map(smap).fillna(0.5).clip(0, 1)
    (REPO_ROOT / out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(REPO_ROOT / out, index=False)
    r = sub.loc[sub["id"].astype(str).isin(smap), "label"].to_numpy()
    print(f"wrote {out}  std={r.std():.3f} mid={((r>=.01)&(r<=.99)).sum()} n={len(r)}")


if __name__ == "__main__":
    main()
