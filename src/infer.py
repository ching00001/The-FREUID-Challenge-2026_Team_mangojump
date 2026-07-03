"""Inference -> submission CSV for one or more trained runs.

Supports ensembling multiple runs and multi-scale test-time augmentation.
Predicts on whatever test images are on disk; ids whose image is missing get a
neutral fill so the submission has every required row.

  --runs A B          ensemble runs A and B
  --scales 448x704,512x800   eval each run at these resolutions and average
  --combine rank      combine runs by average rank (robust to differing
                      score calibrations) instead of mean probability
  --tta hflip         (off by default; ID layout is not L/R symmetric)

Usage:
  python -m src.infer --runs <run> --out subs/x.csv
  python -m src.infer --runs A B --scales 320x512,448x704 --combine rank --out subs/ens.csv
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from .data.paths import REPO_ROOT, load_test
from .data.dataset import FreuidDataset
from .experiment import EXP_ROOT, ExperimentConfig
from .models.factory import FraudNet


def _load_run(run_id: str):
    d = EXP_ROOT / run_id
    cfg = ExperimentConfig(**json.loads((d / "config.json").read_text()))
    cfg.val_recapture = 0.0          # never augment test images
    cfg.recapture_p = 0.0
    ckpt = torch.load(d / "best.pt", map_location="cpu")
    model = FraudNet(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().cuda().to(memory_format=torch.channels_last)
    return model, cfg


@torch.no_grad()
def _predict_one_scale(model, cfg, df, tta, amp=torch.bfloat16) -> np.ndarray:
    ds = FreuidDataset(df, cfg, train=False)
    ld = DataLoader(ds, batch_size=cfg.batch_size * 2, shuffle=False,
                    num_workers=min(4, cfg.num_workers), pin_memory=True)
    out = []
    for x, _, _ in ld:
        x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
        views = [x] + ([torch.flip(x, dims=[3])] if tta in ("hflip", "all") else [])
        ps = [torch.sigmoid(model(v).float()) for v in views]
        out.append(torch.stack(ps).mean(0).cpu().numpy())
    return np.concatenate(out)


def _predict_run(model, cfg, df, tta, scales) -> np.ndarray:
    """Average a run's prediction over the requested scales."""
    sizes = scales or [(cfg.img_h, cfg.img_w)]
    acc = np.zeros(len(df), dtype=np.float64)
    for (h, w) in sizes:
        c = dataclasses.replace(cfg, img_h=h, img_w=w)
        acc += _predict_one_scale(model, c, df, tta)
    return acc / len(sizes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tta", default="none")
    ap.add_argument("--scales", default="", help="comma list HxW, e.g. 320x512,448x704")
    ap.add_argument("--combine", default="mean", choices=["mean", "rank"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fill", type=float, default=0.5)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True

    scales = None
    if args.scales:
        scales = [tuple(int(v) for v in s.split("x")) for s in args.scales.split(",")]

    te = load_test()
    present = te.df.copy()
    print(f"test present: {len(present)} ; missing: {len(te.missing)} ; "
          f"scales={scales or 'native'} combine={args.combine}")

    run_preds = []
    for run in args.runs:
        model, cfg = _load_run(run)
        p = _predict_run(model, cfg, present, args.tta, scales)
        run_preds.append(p)
        print(f"  {run}: mean={p.mean():.4f} std={p.std():.4f}")
        del model
        torch.cuda.empty_cache()

    if args.combine == "rank" and len(run_preds) > 1:
        ranks = [rankdata(p) / len(p) for p in run_preds]
        preds = np.mean(ranks, axis=0)
    else:
        preds = np.mean(run_preds, axis=0)

    sub = pd.read_csv(REPO_ROOT / "sample_submission.csv", dtype={"id": str})
    smap = dict(zip(present["id"].astype(str), preds))
    sub["label"] = sub["id"].astype(str).map(smap).fillna(args.fill)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    n_fill = sub["id"].astype(str).map(lambda i: i not in smap).sum()
    print(f"wrote {out}  ({len(sub)} rows; {n_fill} filled {args.fill})")


if __name__ == "__main__":
    main()
