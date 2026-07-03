# %% [code]
"""
FREUID — SigLIP-2 SO400M + DoRA fine-tuning (Kaggle GPU kernel).

Follow-up to the CLIP ViT-L/14 + LoRA breakthrough (public 0.13024):
  - Backbone upgrade: SigLIP-2 SO400M @378px (the leading CLIP-family variant
    for forgery/deepfake detection in 2025; finer 378px detail for forensic cues).
  - Adapter upgrade: DoRA (ICML'24 oral) — decomposes weights into magnitude +
    direction, LoRA handles only direction; consistently beats LoRA on ViTs.

Same proven recipe otherwise: full-train on all types, canary val for
temperature calibration only, hflip TTA, writes /kaggle/working/submission.csv.
"""

import subprocess
import sys

# Kaggle's preinstalled timm may predate SigLIP-2 weight tags
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "timm"], check=False)

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from tqdm import tqdm

SEED = 42
BATCH = 16
EPOCHS = 3
LR = 2e-4
RANK = 16
ALPHA = 32
WARMUP_STEPS = 1000

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("timm", timm.__version__, "| torch", torch.__version__, "| device", device)


# ---------------------------------------------------------------------------
# Locate competition input (mounts as the hark99 dataset mirror)
# ---------------------------------------------------------------------------

def find_root() -> Path:
    hits = glob.glob("/kaggle/input/**/train_labels.csv", recursive=True)
    if hits:
        return Path(hits[0]).parent
    print("!!! train_labels.csv not found. /kaggle/input tree:")
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        depth = dirpath.replace("/kaggle/input", "").count("/")
        if depth <= 2:
            print("  " * depth, dirpath, "->", filenames[:6])
    raise FileNotFoundError("competition input not found")


ROOT = find_root()
print("INPUT ROOT:", ROOT)


def resolve_dir(split):
    nested = ROOT / split / split
    return nested if nested.is_dir() else ROOT / split


TRAIN_DIR = resolve_dir("train")
TEST_DIR = resolve_dir("public_test")


# ---------------------------------------------------------------------------
# Metric (identical to local src/metrics.py)
# ---------------------------------------------------------------------------

def apcer_at_bpcer(y_true, scores, bpcer_target=0.01):
    bona = y_true == 0
    atk = y_true == 1
    if bona.sum() == 0 or atk.sum() == 0:
        return float("nan")
    best = 1.0
    for thr in np.sort(np.unique(scores))[::-1]:
        preds = (scores >= thr).astype(int)
        if preds[bona].mean() <= bpcer_target:
            best = 1.0 - preds[atk].mean()
        else:
            break
    return best


def freuid_score(y_true, scores):
    au = 1.0 - roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    ap = apcer_at_bpcer(y_true, scores)
    return {"freuid_score": au + ap, "audet": au, "apcer@1pct_bpcer": ap}


# ---------------------------------------------------------------------------
# DoRA (Weight-Decomposed Low-Rank Adaptation, arXiv 2402.09353)
# ---------------------------------------------------------------------------

class DoRALinear(nn.Module):
    """W' = m * (W0 + scaling*B@A) / ||W0 + scaling*B@A||_row.

    Frozen base weight; trainable: A, B (direction, B zero-init) and the
    per-output magnitude vector m (init = row norms of W0, so the layer starts
    exactly equal to the pretrained linear). Row norm is detached in backprop
    per the paper's memory-efficient variant.
    """

    def __init__(self, base: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.m = nn.Parameter(self.weight.norm(p=2, dim=1))  # (out,)

    def forward(self, x):
        W = self.weight + self.scaling * (self.B @ self.A)
        norm = W.norm(p=2, dim=1, keepdim=True).detach()
        Wn = (self.m.unsqueeze(1) / norm) * W
        return F.linear(x, Wn, self.bias)


def build_model():
    last_err = None
    for tag in ["vit_so400m_patch14_siglip_378.v2_webli",   # SigLIP-2 SO400M
                "vit_so400m_patch14_siglip_378.webli",      # SigLIP-1 fallback
                "vit_large_patch14_clip_336.openai",
                "vit_large_patch14_clip_224.openai"]:
        try:
            bb = timm.create_model(tag, pretrained=True, num_classes=0)
            print("backbone:", tag)
            break
        except Exception as e:
            last_err = e
            print("failed", tag, repr(e)[:200])
    else:
        raise last_err
    cfg = bb.pretrained_cfg
    mean, std = cfg["mean"], cfg["std"]
    img_size = cfg["input_size"][1]
    for p in bb.parameters():
        p.requires_grad_(False)
    n = 0
    for blk in bb.blocks:
        blk.attn.qkv = DoRALinear(blk.attn.qkv, RANK, ALPHA); n += 1
        blk.attn.proj = DoRALinear(blk.attn.proj, RANK, ALPHA); n += 1
    print(f"DoRA injected into {n} layers | img_size {img_size} | mean {mean}")
    feat = bb.num_features
    head = nn.Sequential(nn.LayerNorm(feat), nn.Dropout(0.2), nn.Linear(feat, 1))

    class M(nn.Module):
        def __init__(s):
            super().__init__(); s.bb = bb; s.head = head
        def forward(s, x):
            return s.head(s.bb(x)).squeeze(1)
    return M(), img_size, mean, std


model, IMG_SIZE, MEAN, STD = build_model()
model = model.to(device)
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"trainable {sum(p.numel() for p in trainable)/1e6:.2f}M / "
      f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M total")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def to_bool(x):
    return x if isinstance(x, bool) else str(x).strip().lower() == "true"


class DS(Dataset):
    def __init__(self, df, img_dir, tfm, has_label=True):
        self.df = df.reset_index(drop=True)
        self.dir = img_dir
        self.tfm = tfm
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.dir / f"{r['id']}.jpeg").convert("RGB")
        img = self.tfm(img)
        if self.has_label:
            return img, float(r["label"])
        return img, str(r["id"])


train_tfm = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.15, 0.15, 0.1),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
eval_tfm = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
flip_tfm = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(1.0),
                      T.ToTensor(), T.Normalize(MEAN, STD)])

df = pd.read_csv(ROOT / "train_labels.csv")
df["is_digital"] = df["is_digital"].map(to_bool)
rec = ~df["is_digital"]
val_idx = (df[df["is_digital"]].groupby(["type", "label"], group_keys=False)
           .apply(lambda g: g.sample(frac=0.005, random_state=SEED)).index)
val_mask = rec | df.index.isin(val_idx)
train_df = df[~val_mask].sample(frac=1, random_state=SEED).reset_index(drop=True)
val_df = df[val_mask].reset_index(drop=True)
print(f"train {len(train_df)} | canary val {len(val_df)}")

tw = 1.0 / train_df["type"].value_counts()
n1 = max(1, int((train_df["label"] == 1).sum())); n0 = max(1, int((train_df["label"] == 0).sum()))
cw = {0: len(train_df) / (2 * n0), 1: len(train_df) / (2 * n1)}
sw = (train_df["type"].map(tw) * train_df["label"].map(cw)).values.astype(np.float32)
sampler = WeightedRandomSampler(torch.from_numpy(sw), len(train_df), replacement=True)

train_loader = DataLoader(DS(train_df, TRAIN_DIR, train_tfm), batch_size=BATCH,
                          sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(DS(val_df, TRAIN_DIR, eval_tfm), batch_size=BATCH,
                        shuffle=False, num_workers=4, pin_memory=True)

opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-2)
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cw[0] / cw[1]).to(device))
steps = len(train_loader) * EPOCHS
def lr_lambda(s):
    if s < WARMUP_STEPS:
        return s / max(1, WARMUP_STEPS)
    prog = (s - WARMUP_STEPS) / max(1, steps - WARMUP_STEPS)
    return 0.5 * (1 + np.cos(np.pi * prog))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
scaler = torch.cuda.amp.GradScaler()


@torch.no_grad()
def collect(loader, with_logits=False):
    model.eval(); outs, ys = [], []
    for imgs, labels in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            lo = model(imgs.to(device))
        v = lo.float().cpu().numpy() if with_logits else torch.sigmoid(lo).float().cpu().numpy()
        outs.extend(v.tolist()); ys.extend(np.asarray(labels, dtype=float).tolist())
    return np.array(outs), np.array(ys)


for epoch in range(EPOCHS):
    model.train(); tot = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"e{epoch+1}", mininterval=30):
        imgs, labels = imgs.to(device), labels.to(device).float()
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = crit(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
    ps, ys = collect(val_loader)
    m = freuid_score(ys, ps)
    print(f"Epoch {epoch+1}/{EPOCHS} loss={tot/len(train_loader):.4f} "
          f"canary_freuid={m['freuid_score']:.4f} audet={m['audet']:.4f}", flush=True)
    torch.save({k: v for k, v in model.state_dict().items()
                if any(t in k for t in ['.A', '.B', '.m', 'head'])},
               "/kaggle/working/siglip2_dora_adapters.pt")

lg, ys = collect(val_loader, with_logits=True)
def nll(t):
    p = 1 / (1 + np.exp(-lg / t)); p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(ys * np.log(p) + (1 - ys) * np.log(1 - p))
T_opt = float(minimize_scalar(nll, bounds=(0.1, 10), method="bounded").x)
print("Temperature:", round(T_opt, 4))

ss = pd.read_csv(ROOT / "sample_submission.csv")
present = ss["id"].astype(str).map(lambda x: (TEST_DIR / f"{x}.jpeg").exists())
avail_df = ss[present].reset_index(drop=True)
print(f"public test available: {len(avail_df)} / {len(ss)}")

probs = np.zeros(len(avail_df))
for tfm in [eval_tfm, flip_tfm]:
    loader = DataLoader(DS(avail_df, TEST_DIR, tfm, has_label=False), batch_size=BATCH,
                        shuffle=False, num_workers=4, pin_memory=True)
    pp = []
    model.eval()
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="predict", mininterval=30):
            with torch.autocast("cuda", dtype=torch.float16):
                pp.extend(torch.sigmoid(model(imgs.to(device))).float().cpu().numpy().tolist())
    probs += np.array(pp)
probs /= 2
probs = np.clip(probs, 1e-7, 1 - 1e-7)
cal = 1 / (1 + np.exp(-np.log(probs / (1 - probs)) / T_opt))

scores = np.full(len(ss), 0.5)
scores[present.values] = cal
out = pd.DataFrame({"id": ss["id"], "label": np.clip(scores, 0, 1)})
out.to_csv("/kaggle/working/submission.csv", index=False)
print(f"saved submission.csv | real>0.5={(cal>0.5).mean():.2%} | real median={np.median(cal):.4f}")