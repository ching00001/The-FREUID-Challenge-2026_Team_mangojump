# %% [code]
"""FREUID 2026 — SigLIP-2 SO400M @512 backbone interpolated to 768 + DoRA.

Kaggle-ready, self-contained port of the local experiment:
    python -m src.train_siglip512 --name siglip768_dora --img_size 768 \
        --batch 4 --accum 4 --epochs 3

Differences from the local script (Blackwell 5060 Ti) needed for Kaggle T4/P100:
  * fp16 + GradScaler  (Turing/Pascal have no bf16; local used bf16)
  * pip install -U timm (Kaggle's preinstalled timm predates SigLIP-2 v2 tags)
  * writes /kaggle/working/submission.csv AFTER EVERY EPOCH, so a 12h-limit
    timeout still leaves a usable submission from the last finished epoch.

Recipe is otherwise identical to the 0.02667 best (@512), only the input is
interpolated to 768 (pos-embed 32x32 -> 48x48). Reality check: a T4/P100 is
several times slower than the 5060 Ti, and 3 epochs @768 will likely NOT finish
in Kaggle's 12h GPU window — drop EPOCHS to 1-2, or run the @512 config instead.
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "timm"], check=False)

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

# ---- config (this run = the 768-interp experiment) -------------------------
BACKBONE = "vit_so400m_patch16_siglip_512.v2_webli"
IMG_SIZE = 768            # interpolate pos-embed from native 512 (set 0 for native)
RANK, ALPHA = 16, 32
EPOCHS, BATCH, ACCUM = 3, 4, 4     # eff batch 16; lower EPOCHS if 12h limit bites
LR, WD, WARMUP = 2e-4, 1e-2, 1000
EMA_DECAY = 0.9995
SEED = 42
OUT = "/kaggle/working/submission.csv"

torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print("timm", timm.__version__, "| torch", torch.__version__, "| device", device)


# ---- data location (Kaggle /kaggle/input OR local repo) --------------------
def find_data_root() -> Path:
    cands = [Path(h).parent for h in
             glob.glob("/kaggle/input/**/train_labels.csv", recursive=True)]
    if os.environ.get("FREUID_DATA"):
        cands.append(Path(os.environ["FREUID_DATA"]))
    cands.append(Path.cwd())
    for c in cands:
        if (c / "train_labels.csv").exists():
            return c
    raise FileNotFoundError("train_labels.csv not found")


def resolve_dir(root: Path, names) -> Path:
    for n in names:
        for d in (root / n / n, root / n):
            if d.is_dir():
                return d
    raise FileNotFoundError(f"image dir not found under {root}: {names}")


# ---- DoRA ------------------------------------------------------------------
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
        self.m = nn.Parameter(self.weight.norm(p=2, dim=1))

    def forward(self, x):
        W = self.weight + self.scaling * (self.B @ self.A)
        norm = W.norm(p=2, dim=1, keepdim=True).detach()
        return F.linear(x, (self.m.unsqueeze(1) / norm) * W, self.bias)


def build_model():
    kw = dict(pretrained=True, num_classes=0)
    if IMG_SIZE:
        kw["img_size"] = IMG_SIZE
    bb = timm.create_model(BACKBONE, **kw)
    cfg = bb.pretrained_cfg
    mean, std = cfg["mean"], cfg["std"]
    img = IMG_SIZE or cfg["input_size"][1]
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

    m = M()
    print(f"{BACKBONE} img={img} | DoRA layers={n} | "
          f"trainable={sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6:.2f}M")
    return m, img, mean, std


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


# ---- main ------------------------------------------------------------------
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

df = pd.read_csv(root / "train_labels.csv")
df["is_digital"] = df["is_digital"].astype(str).str.strip().str.lower().isin(["true", "1"])
val_idx = (df[df["is_digital"]].groupby(["type", "label"], group_keys=False)
           .apply(lambda g: g.sample(frac=0.005, random_state=SEED)).index)
val_mask = (~df["is_digital"]) | df.index.isin(val_idx)
tr = df[~val_mask].sample(frac=1, random_state=SEED).reset_index(drop=True)
va = df[val_mask].reset_index(drop=True)
print(f"train={len(tr)} | canary={len(va)} (recaptured {int((~va['is_digital']).sum())})")

tw = 1.0 / tr["type"].value_counts()
n1, n0 = max(1, int((tr.label == 1).sum())), max(1, int((tr.label == 0).sum()))
cw = {0: len(tr) / (2 * n0), 1: len(tr) / (2 * n1)}
sw = (tr["type"].map(tw) * tr["label"].map(cw)).values.astype(np.float32)
sampler = WeightedRandomSampler(torch.from_numpy(sw), len(tr), replacement=True)

tr_ld = DataLoader(DS(tr, train_dir, train_tfm), batch_size=BATCH, sampler=sampler,
                   num_workers=4, pin_memory=True, drop_last=True)
va_ld = DataLoader(DS(va, train_dir, eval_tfm), batch_size=BATCH * 2,
                   shuffle=False, num_workers=4, pin_memory=True)

trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cw[0] / cw[1]).to(device))
scaler = torch.cuda.amp.GradScaler()                     # fp16 needs a scaler
total = (len(tr_ld) // ACCUM) * EPOCHS


def lr_at(s):
    if s < WARMUP:
        return s / max(1, WARMUP)
    p = (s - WARMUP) / max(1, total - WARMUP)
    return 0.5 * (1 + np.cos(np.pi * p))


def write_submission(net):
    """hflip-TTA predict the public test (EMA weights) and write OUT."""
    ss = pd.read_csv(root / "sample_submission.csv", dtype={"id": str})
    present = ss[ss["id"].map(lambda x: (test_dir / f"{x}.jpeg").exists())].reset_index(drop=True)
    probs = np.zeros(len(present))
    net.eval()
    for tfm in (eval_tfm, flip_tfm):
        ld = DataLoader(DS(present, test_dir, tfm, has_label=False),
                        batch_size=BATCH * 2, shuffle=False, num_workers=4, pin_memory=True)
        pp = []
        with torch.no_grad():
            for x, _ in ld:
                with torch.autocast("cuda", dtype=torch.float16):
                    pp.append(torch.sigmoid(net(x.to(device)).float()).cpu().numpy())
        probs += np.concatenate(pp)
    probs /= 2
    smap = dict(zip(present["id"].astype(str), probs))
    ss["label"] = ss["id"].astype(str).map(smap).fillna(0.5).clip(0, 1)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    ss.to_csv(OUT, index=False)
    print(f"  wrote {OUT} ({len(present)}/{len(ss)} real)")


t0 = time.time()
step = 0
for ep in range(EPOCHS):
    model.train(); opt.zero_grad(set_to_none=True); run = 0.0
    for it, (x, y) in enumerate(tr_ld):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = crit(model(x), y) / ACCUM
        scaler.scale(loss).backward(); run += loss.item() * ACCUM
        if (it + 1) % ACCUM == 0:
            for g in opt.param_groups:
                g["lr"] = LR * lr_at(step)
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True); ema.update(model)
            step += 1
            if step % 200 == 0:
                print(f"  e{ep} step {step}/{total} loss={run/(it+1):.4f} "
                      f"[{(time.time()-t0)/60:.0f}m]", flush=True)
    # canary health check (saturates -> monitoring only)
    ema.module.eval(); ps, ys = [], []
    with torch.no_grad():
        for x, y in va_ld:
            with torch.autocast("cuda", dtype=torch.float16):
                ps.append(torch.sigmoid(ema.module(x.to(device)).float()).cpu().numpy())
            ys.append(np.asarray(y, dtype=float))
    ps, ys = np.concatenate(ps), np.concatenate(ys)
    dg = va["is_digital"].values
    print(f"epoch {ep+1}/{EPOCHS} loss={run/len(tr_ld):.4f} "
          f"canary_freuid={freuid(ys[dg], ps[dg]):.4f} [{(time.time()-t0)/60:.0f}m]", flush=True)
    write_submission(ema.module)        # checkpoint a submission every epoch

print(f"DONE in {(time.time()-t0)/60:.1f} min")
