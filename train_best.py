"""FREUID 2026 — best single model (public LB 0.02667).

SigLIP-2 SO400M @512 + DoRA, full-train, hflip-TTA. Self-contained: depends only
on torch/timm/torchvision/pandas/numpy/sklearn/PIL, no project scaffolding, so it
runs unchanged locally or on Kaggle. This is the canonical reproduction script
for the 0.02667 submission (subs/siglip512_dora.csv).

Recipe (frozen — these constants ARE the winning config):
  backbone vit_so400m_patch16_siglip_512.v2_webli (native 512px, all frozen)
  DoRA r16 a32 into every block's attn.{qkv,proj} + mlp.{fc1,fc2} (108 layers,
    ~8M trainable); head = LayerNorm + Dropout(0.2) + Linear(.,1)
  EMA 0.9995 (eval + inference use EMA weights); bf16 autocast; grad checkpointing
  3 epochs, eff batch 16 (batch 8 x accum 2), AdamW lr 2e-4 wd 1e-2,
    cosine schedule + 1000 warmup steps, grad clip 1.0
  full-train on ALL types; canary val = 0.5% stratified digital + all recaptured
    (clean val saturates -> used only as a health check, NOT for selection)
  type x class WeightedRandomSampler + pos_weight BCE
  aug: RandomResizedCrop(512, 0.7-1.0) + hflip + ColorJitter(.15,.15,.1)
  inference: hflip TTA (mean of 2 views), missing test ids filled 0.5

Run:
  python train_best.py                      # train + write submission.csv
  python train_best.py --out subs/x.csv     # custom output path
  FREUID_DATA=/path/to/data python train_best.py
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

# ---- frozen winning config -------------------------------------------------
BACKBONE = "vit_so400m_patch16_siglip_512.v2_webli"
RANK, ALPHA = 16, 32
EPOCHS, BATCH, ACCUM = 3, 8, 2
LR, WD, WARMUP = 2e-4, 1e-2, 1000
EMA_DECAY = 0.9995
SEED = 42


# ---- data location (local nested layout OR Kaggle) -------------------------
def find_data_root() -> Path:
    cands = []
    if os.environ.get("FREUID_DATA"):
        cands.append(Path(os.environ["FREUID_DATA"]))
    cands.append(Path(__file__).resolve().parent)            # repo root
    cands += [Path(h).parent for h in
              glob.glob("/kaggle/input/**/train_labels.csv", recursive=True)]
    for c in cands:
        if (c / "train_labels.csv").exists():
            return c
    raise FileNotFoundError("train_labels.csv not found (set FREUID_DATA)")


def resolve_dir(root: Path, names: list[str]) -> Path:
    for n in names:                       # handle the train/train, public_test/public_test nesting
        for d in (root / n / n, root / n):
            if d.is_dir():
                return d
    raise FileNotFoundError(f"image dir not found under {root}: {names}")


# ---- DoRA (Weight-Decomposed Low-Rank Adaptation, arXiv 2402.09353) --------
class DoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        self.bias = (nn.Parameter(base.bias.data.clone(), requires_grad=False)
                     if base.bias is not None else None)
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.m = nn.Parameter(self.weight.norm(p=2, dim=1))      # init = row norms -> no-op at start

    def forward(self, x):
        W = self.weight + self.scaling * (self.B @ self.A)
        norm = W.norm(p=2, dim=1, keepdim=True).detach()         # detached norm (memory-efficient variant)
        return F.linear(x, (self.m.unsqueeze(1) / norm) * W, self.bias)


def build_model():
    bb = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
    cfg = bb.pretrained_cfg
    mean, std, img = cfg["mean"], cfg["std"], cfg["input_size"][1]
    for p in bb.parameters():
        p.requires_grad_(False)
    n = 0
    for blk in bb.blocks:
        blk.attn.qkv = DoRALinear(blk.attn.qkv, RANK, ALPHA); n += 1
        blk.attn.proj = DoRALinear(blk.attn.proj, RANK, ALPHA); n += 1
        blk.mlp.fc1 = DoRALinear(blk.mlp.fc1, RANK, ALPHA); n += 1
        blk.mlp.fc2 = DoRALinear(blk.mlp.fc2, RANK, ALPHA); n += 1
    if hasattr(bb, "set_grad_checkpointing"):
        bb.set_grad_checkpointing(True)
    head = nn.Sequential(nn.LayerNorm(bb.num_features), nn.Dropout(0.2),
                         nn.Linear(bb.num_features, 1))

    class M(nn.Module):
        def __init__(s):
            super().__init__(); s.bb = bb; s.head = head

        def forward(s, x):
            return s.head(s.bb(x)).squeeze(1)

    print(f"backbone {BACKBONE} img={img} | DoRA layers={n} | "
          f"trainable={sum(p.numel() for p in M().parameters() if p.requires_grad)/1e6:.2f}M")
    return M(), img, mean, std


# ---- dataset ---------------------------------------------------------------
class DS(Dataset):
    def __init__(self, df, img_dir, tfm, has_label=True):
        self.ids = df["id"].astype(str).tolist()
        self.labels = (df["label"].astype(float).tolist() if has_label
                       else [0.0] * len(df))
        self.dir, self.tfm, self.has_label = img_dir, tfm, has_label

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = self.tfm(Image.open(self.dir / f"{self.ids[i]}.jpeg").convert("RGB"))
        return (img, self.labels[i]) if self.has_label else (img, self.ids[i])


# ---- FREUID metric (canary health check only) ------------------------------
def apcer_at_bpcer(y, s, t=0.01):
    bona, atk = y == 0, y == 1
    if bona.sum() == 0 or atk.sum() == 0:
        return float("nan")
    best = 1.0
    for thr in np.sort(np.unique(s))[::-1]:
        pred = s >= thr
        if pred[bona].mean() <= t:
            best = 1.0 - pred[atk].mean()
        else:
            break
    return best


def freuid(y, s):
    au = 1.0 - roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")
    return au + apcer_at_bpcer(y, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="cap train rows (smoke/debug; 0=all)")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    root = find_data_root()
    train_dir = resolve_dir(root, ["train"])
    test_dir = resolve_dir(root, ["public_test", "test"])
    print("data root:", root)

    model, img, mean, std = build_model()
    model = model.to(device)
    from timm.utils import ModelEmaV3
    ema = ModelEmaV3(model, decay=EMA_DECAY)

    train_tfm = T.Compose([
        T.RandomResizedCrop(img, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(), T.ColorJitter(0.15, 0.15, 0.1),
        T.ToTensor(), T.Normalize(mean, std)])
    eval_tfm = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(mean, std)])
    flip_tfm = T.Compose([T.Resize((img, img)), T.RandomHorizontalFlip(1.0),
                          T.ToTensor(), T.Normalize(mean, std)])

    # split: full-train + tiny canary (all recaptured + 0.5% stratified digital)
    df = pd.read_csv(root / "train_labels.csv")
    df["is_digital"] = df["is_digital"].astype(str).str.strip().str.lower().isin(
        ["true", "1"])
    val_idx = (df[df["is_digital"]].groupby(["type", "label"], group_keys=False)
               .apply(lambda g: g.sample(frac=0.005, random_state=SEED)).index)
    val_mask = (~df["is_digital"]) | df.index.isin(val_idx)
    tr = df[~val_mask].sample(frac=1, random_state=SEED).reset_index(drop=True)
    if args.limit:
        tr = tr.iloc[:args.limit].reset_index(drop=True)
    va = df[val_mask].reset_index(drop=True)
    print(f"train={len(tr)} | canary={len(va)} (recaptured {int((~va['is_digital']).sum())})")

    tw = 1.0 / tr["type"].value_counts()
    n1, n0 = max(1, int((tr.label == 1).sum())), max(1, int((tr.label == 0).sum()))
    cw = {0: len(tr) / (2 * n0), 1: len(tr) / (2 * n1)}
    sw = (tr["type"].map(tw) * tr["label"].map(cw)).values.astype(np.float32)
    sampler = WeightedRandomSampler(torch.from_numpy(sw), len(tr), replacement=True)

    tr_ld = DataLoader(DS(tr, train_dir, train_tfm), batch_size=BATCH, sampler=sampler,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va_ld = DataLoader(DS(va, train_dir, eval_tfm), batch_size=BATCH * 2,
                       shuffle=False, num_workers=2, pin_memory=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cw[0] / cw[1]).to(device))
    total = (len(tr_ld) // ACCUM) * args.epochs

    def lr_at(s):
        if s < WARMUP:
            return s / max(1, WARMUP)
        p = (s - WARMUP) / max(1, total - WARMUP)
        return 0.5 * (1 + np.cos(np.pi * p))

    step = 0
    for ep in range(args.epochs):
        model.train(); opt.zero_grad(set_to_none=True); run = 0.0
        for it, (x, y) in enumerate(tr_ld):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x), y) / ACCUM
            loss.backward(); run += loss.item() * ACCUM
            if (it + 1) % ACCUM == 0:
                for g in opt.param_groups:
                    g["lr"] = LR * lr_at(step)
                nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); ema.update(model)
                step += 1
                if step % 200 == 0:
                    print(f"  e{ep} step {step}/{total} loss={run/(it+1):.4f}", flush=True)
        # canary health check (saturates -> monitoring only)
        ema.module.eval(); ps, ys, dg = [], [], []
        with torch.no_grad():
            for x, y in va_ld:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ps.append(torch.sigmoid(ema.module(x.to(device)).float()).cpu().numpy())
                ys.append(np.asarray(y, dtype=float))
        ps, ys = np.concatenate(ps), np.concatenate(ys)
        dg = va["is_digital"].values
        print(f"epoch {ep+1}/{args.epochs} loss={run/len(tr_ld):.4f} "
              f"canary_freuid={freuid(ys[dg], ps[dg]):.4f}", flush=True)

    # ---- inference: hflip TTA, EMA weights -------------------------------
    ss = pd.read_csv(root / "sample_submission.csv", dtype={"id": str})
    present = ss[ss["id"].map(lambda x: (test_dir / f"{x}.jpeg").exists())].reset_index(drop=True)
    print(f"public test on disk: {len(present)} / {len(ss)}")
    probs = np.zeros(len(present))
    ema.module.eval()
    for tfm in (eval_tfm, flip_tfm):
        ld = DataLoader(DS(present, test_dir, tfm, has_label=False),
                        batch_size=BATCH * 2, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
        pp = []
        with torch.no_grad():
            for x, _ in ld:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pp.append(torch.sigmoid(ema.module(x.to(device)).float()).cpu().numpy())
        probs += np.concatenate(pp)
    probs /= 2

    smap = dict(zip(present["id"].astype(str), probs))
    ss["label"] = ss["id"].astype(str).map(smap).fillna(0.5).clip(0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ss.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(ss)} rows) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
