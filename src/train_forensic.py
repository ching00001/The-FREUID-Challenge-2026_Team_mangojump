"""Forensic member: Bayar-constrained-conv + ConvNeXt — the 3rd paradigm.

We have SSL (DINOv3) + VL (SigLIP). Missing = low-level NOISE forensics, which
catches splicing/inpaint/JPEG inconsistencies the semantic models can't see (the
teammate's forensic_SRM_convnext was the lever behind their gated 0.00395).

Bayar constrained conv (Bayar & Stamm) is a learnable high-pass that suppresses
image content and exposes manipulation noise residuals; a ConvNeXt then classifies.
Trained full (small CNN, no DoRA) on the FREUID digital train, hflip-TTA inference.
Saves a submission (for gating the fusion) + the model (for feature fusion later).

  python -m src.train_forensic                 # train + write subs/forensic.csv
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from .data.paths import REPO_ROOT, load_test
from .train_DINOV3L_512 import make_splits
from .metric import freuid_score

EXP_ROOT = REPO_ROOT / "experiments"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class BayarConv2d(nn.Module):
    """Constrained high-pass conv: center tap = -1, other taps sum to +1 ->
    output is a learned noise residual (content suppressed)."""
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(out_ch, in_ch, k, k) * 1e-3)

    def _constrain(self):
        with torch.no_grad():
            c = self.k // 2
            self.w.data[:, :, c, c] = 0
            self.w.data /= self.w.data.sum(dim=(2, 3), keepdim=True) + 1e-8
            self.w.data[:, :, c, c] = -1.0

    def forward(self, x):
        self._constrain()
        return F.conv2d(x, self.w, padding=self.k // 2)


class ForensicNet(nn.Module):
    def __init__(self, backbone="convnextv2_tiny.fcmae_ft_in22k_in1k"):
        super().__init__()
        self.bayar = BayarConv2d(3, 3, 5)
        self.net = timm.create_model(backbone, pretrained=True, num_classes=1)

    def forward(self, x):
        return self.net(self.bayar(x)).squeeze(-1)


class DS(Dataset):
    def __init__(self, df, tfm, lab=True):
        self.paths = df["abspath"].tolist()
        self.ids = df["id"].astype(str).tolist()
        self.y = df["label"].astype(float).tolist() if lab else [0.0] * len(df)
        self.tfm, self.lab = tfm, lab

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        im = self.tfm(Image.open(self.paths[i]).convert("RGB"))
        return (im, self.y[i]) if self.lab else (im, self.ids[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="subs/forensic.csv")
    args = ap.parse_args()
    dev = "cuda"; torch.manual_seed(42); np.random.seed(42)
    torch.backends.cudnn.benchmark = True
    rid = f"{time.strftime('%Y%m%d_%H%M%S')}_forensic"; rdir = EXP_ROOT / rid
    rdir.mkdir(parents=True, exist_ok=True); t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True); open(rdir/"log.txt","a").write(m+"\n")

    # light aug only — heavy resampling/JPEG would destroy the noise residual
    tr_tf = T.Compose([T.Resize((args.img, args.img)), T.RandomHorizontalFlip(),
                       T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    ev_tf = T.Compose([T.Resize((args.img, args.img)), T.ToTensor(),
                       T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    fl_tf = T.Compose([T.Resize((args.img, args.img)), T.RandomHorizontalFlip(1.0),
                       T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    tr_df, va_df = make_splits(42, 0)
    tr_df = tr_df[tr_df["is_digital"]].reset_index(drop=True)
    va_df = va_df[va_df["is_digital"]].reset_index(drop=True)
    log(f"train={len(tr_df)} canary={len(va_df)} img={args.img}")

    tw = 1.0 / tr_df["type"].value_counts()
    n1 = max(1, int((tr_df.label == 1).sum())); n0 = max(1, int((tr_df.label == 0).sum()))
    cw = {0: len(tr_df)/(2*n0), 1: len(tr_df)/(2*n1)}
    sw = (tr_df["type"].map(tw)*tr_df["label"].map(cw)).values.astype(np.float32)
    sampler = WeightedRandomSampler(torch.from_numpy(sw), len(tr_df), replacement=True)
    tl = DataLoader(DS(tr_df, tr_tf), batch_size=args.batch, sampler=sampler,
                    num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(DS(va_df, ev_tf), batch_size=args.batch*2, shuffle=False, num_workers=2)

    model = ForensicNet().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cw[0]/cw[1]).to(dev))
    from timm.utils import ModelEmaV3
    ema = ModelEmaV3(model, decay=0.9998)
    total = len(tl)*args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)

    step = 0
    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True).float()
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ema.update(model); step += 1
            if step % 200 == 0: log(f"  e{ep} {step}/{total} loss={loss.item():.4f}")
        # canary
        ema.module.eval(); ps, ys = [], []
        with torch.no_grad():
            for x, y in vl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ps.append(torch.sigmoid(ema.module(x.to(dev)).float()).cpu().numpy())
                ys.append(np.asarray(y, float))
        ps, ys = np.concatenate(ps), np.concatenate(ys)
        log(f"epoch {ep+1} canary_freuid={freuid_score(ys,ps).freuid:.4f} auc={freuid_score(ys,ps).roc_auc:.4f}")
        torch.save({"model": ema.module.state_dict(), "args": vars(args)}, rdir/"model.pt")

    # inference: hflip TTA
    te = load_test().df.copy()
    probs = np.zeros(len(te)); ema.module.eval()
    for tf in (ev_tf, fl_tf):
        ld = DataLoader(DS(te, tf, lab=False), batch_size=args.batch*2, shuffle=False, num_workers=4)
        pp = []
        with torch.no_grad():
            for x, _ in ld:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pp.append(torch.sigmoid(ema.module(x.to(dev)).float()).cpu().numpy())
        probs += np.concatenate(pp)
    probs /= 2
    sub = pd.read_csv(REPO_ROOT/"sample_submission.csv", dtype={"id": str})
    sub["label"] = sub["id"].astype(str).map(dict(zip(te["id"].astype(str), probs))).fillna(0.5).clip(0, 1)
    Path(REPO_ROOT/args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(REPO_ROOT/args.out, index=False)
    log(f"wrote {args.out} | DONE {(time.time()-t0)/60:.0f}min")


if __name__ == "__main__":
    main()
