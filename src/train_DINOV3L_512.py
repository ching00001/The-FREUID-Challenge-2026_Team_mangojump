from __future__ import annotations

import argparse
import json
import time
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

from .data.paths import REPO_ROOT, load_test, load_train
from .metric import freuid_score

EXP_ROOT = REPO_ROOT / "experiments"


# ---------------------------------------------------------------------------
# DoRA (Weight-Decomposed Low-Rank Adaptation, arXiv 2402.09353)
# ---------------------------------------------------------------------------
class DoRALinear(nn.Module):
    """W' = m * (W0 + scaling*B@A) / ||row||; A,B,m trainable, W0 frozen.

    B zero-init and m init to W0 row norms -> identical to the pretrained
    linear at step 0. Row norm detached (paper's memory-efficient variant).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: int):
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
        return F.linear(x, (self.m.unsqueeze(1) / norm) * W, self.bias)


def build_model(args):
    kwargs = dict(pretrained=True, num_classes=0)
    override = getattr(args, "img_size", 0)
    if override:                       # interpolate pos-embed to a new input size
        kwargs["img_size"] = override
    bb = timm.create_model(args.backbone, **kwargs)
    cfg = bb.pretrained_cfg
    mean, std = cfg["mean"], cfg["std"]
    img_size = override or cfg["input_size"][1]
    if getattr(args, "full_ft", False):
        # full fine-tune: unfreeze the whole backbone, NO adapters. Biggest
        # capacity lever — fits the public distribution far tighter than DoRA.
        for p in bb.parameters():
            p.requires_grad_(True)
        if hasattr(bb, "set_grad_checkpointing"):
            bb.set_grad_checkpointing(True)
        feat = bb.num_features
        head = nn.Sequential(nn.LayerNorm(feat), nn.Dropout(0.2), nn.Linear(feat, 1))

        class M(nn.Module):
            def __init__(s):
                super().__init__(); s.bb = bb; s.head = head

            def forward(s, x):
                return s.head(s.bb(x)).squeeze(1)

        return M(), 0, img_size, mean, std
    for p in bb.parameters():
        p.requires_grad_(False)
    # inject DoRA into whichever target Linears EXIST per block (handles both
    # SigLIP mlp.fc1/fc2 and AIMv2 SwiGLU mlp.fc1_g/fc1_x/fc2)
    attn_t = ("qkv", "proj")
    mlp_t = () if args.attn_only else ("fc1", "fc2", "fc1_g", "fc1_x")
    n = 0
    for blk in bb.blocks:
        for parent, name in ([(blk.attn, t) for t in attn_t]
                             + [(blk.mlp, t) for t in mlp_t]):
            lin = getattr(parent, name, None)
            if isinstance(lin, nn.Linear):
                setattr(parent, name, DoRALinear(lin, args.rank, args.alpha)); n += 1
    if hasattr(bb, "set_grad_checkpointing"):  # [ckpt]
        bb.set_grad_checkpointing(True)
    feat = bb.num_features
    head = nn.Sequential(nn.LayerNorm(feat), nn.Dropout(0.2), nn.Linear(feat, 1))

    class M(nn.Module):
        def __init__(s):
            super().__init__(); s.bb = bb; s.head = head

        def forward(s, x):
            return s.head(s.bb(x)).squeeze(1)

    return M(), n, img_size, mean, std


# ---------------------------------------------------------------------------
# Data — full-train + tiny canary (classmate's split, on our path loader)
# ---------------------------------------------------------------------------
class RecaptureT:
    """torchvision-style transform: with prob p, apply the print-and-capture
    channel (src.aug.recapture) at strength U[mn,mx], to BOTH classes so
    recapture presence is not a class cue. Picklable (module-level) for workers.
    Bridges the 99.97%-digital train -> captured/recapture private test."""
    def __init__(self, p, mn, mx):
        self.p, self.mn, self.mx = p, mn, mx

    def __call__(self, pil):
        from .aug.recapture import recapture as _rc
        rng = np.random.default_rng()
        if rng.random() > self.p:
            return pil
        a = _rc(np.asarray(pil.convert("RGB")),
                strength=float(rng.uniform(self.mn, self.mx)), p=1.0, rng=rng)
        return Image.fromarray(a)


class DS(Dataset):
    def __init__(self, df, tfm, has_label=True):
        self.paths = df["abspath"].tolist()
        self.ids = df["id"].astype(str).tolist()
        self.labels = (df["label"].astype(float).tolist()
                       if has_label else [0.0] * len(df))
        self.tfm = tfm
        self.has_label = has_label

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = self.tfm(img)
        if self.has_label:
            return img, self.labels[i]
        return img, self.ids[i]


def make_splits(seed: int, subset: int, loto_types: str = "", extra_data: str = "",
                extra_val: str = ""):
    """Default: full-train + 0.5% canary + 20 recaptured.

    type-LOO (loto_types="A,B"): HOLD OUT those document types entirely from
    train and use them as validation -> the canary_freuid then measures
    UNSEEN-TYPE generalization (proxy for the private test's 2 unseen types).
    The 20 real recaptured stay in val too (capture axis)."""
    df = load_train().df[["id", "abspath", "is_digital", "label", "type"]].copy()
    rec = ~df["is_digital"]
    if loto_types:
        held = {t.strip() for t in loto_types.split(",")}
        in_held = df["type"].isin(held)
        # val = a capped sample of held-out types (digital) + all recaptured
        held_dig = df[in_held & df["is_digital"]]
        val_held = held_dig.groupby(["type", "label"], group_keys=False).apply(
            lambda g: g.sample(min(len(g), 600), random_state=seed))
        val_mask = rec | df.index.isin(val_held.index)
        train_df = df[~val_mask & ~in_held].sample(frac=1, random_state=seed
                                                   ).reset_index(drop=True)
    else:
        val_idx = (df[df["is_digital"]].groupby(["type", "label"], group_keys=False)
                   .apply(lambda g: g.sample(frac=0.005, random_state=seed)).index)
        val_mask = rec | df.index.isin(val_idx)
        train_df = df[~val_mask].sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    if subset > 0:
        train_df = train_df.iloc[:subset].reset_index(drop=True)
    train_df["is_extra"] = False
    val_df["is_extra"] = False
    cols = ["id", "abspath", "is_digital", "label", "type"]
    if extra_data:  # append external rows to TRAIN only (e.g. SIDTD, DLC-2021)
        ex = pd.read_csv(REPO_ROOT / extra_data)[cols].assign(is_extra=True)
        train_df = pd.concat([train_df, ex], ignore_index=True).sample(
            frac=1, random_state=seed).reset_index(drop=True)
    if extra_val:   # external eval rows (e.g. DLC holdout) -> separate metric
        ev = pd.read_csv(REPO_ROOT / extra_val)[cols].assign(is_extra=True)
        val_df = pd.concat([val_df, ev], ignore_index=True)
    return train_df, val_df


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    ps, ys = [], []
    for imgs, labels in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(imgs.to(device, non_blocking=True))
        ps.append(torch.sigmoid(lo.float()).cpu().numpy())
        ys.append(np.asarray(labels, dtype=float))
    return np.concatenate(ps), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    # defaults = the best public model (DINOv3 ViT-L @512 = LB 0.01134);
    # `python -m src.train_siglip512` reproduces subs/dinov3_l512.csv with no args.
    ap.add_argument("--name", default="dinov3_l512")
    ap.add_argument("--backbone", default="vit_large_patch16_dinov3.lvd1689m")
    ap.add_argument("--img_size", type=int, default=512,
                    help="input size (DINOv3 uses RoPE; 0=native checkpoint size)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--attn_only", action="store_true")
    ap.add_argument("--full_ft", action="store_true",
                    help="full fine-tune backbone (no DoRA); use low --lr + --head_lr_mult")
    ap.add_argument("--head_lr_mult", type=float, default=1.0,
                    help="head LR = lr * mult (full-FT: head is random-init, use ~50-100)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)   # eff batch 16, as theirs
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--ema_decay", type=float, default=0.9995)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--skip_infer", action="store_true")
    ap.add_argument("--recapture_p", type=float, default=0.0,
                    help="prob of applying print-and-capture aug to a train image")
    ap.add_argument("--rc_min", type=float, default=0.2)
    ap.add_argument("--rc_max", type=float, default=0.6)
    ap.add_argument("--loto_types", default="",
                    help="comma-sep doc types to HOLD OUT (type-LOO unseen-type proxy)")
    ap.add_argument("--extra_data", default="",
                    help="csv (id,abspath,is_digital,label,type) appended to TRAIN, e.g. SIDTD")
    ap.add_argument("--extra_frac", type=float, default=0.0,
                    help="target fraction of SAMPLER mass for extra_data rows; 0=raw "
                         "type-balanced weights (DLC's 2 pseudo-types would grab 2/7!)")
    ap.add_argument("--extra_val", default="",
                    help="csv appended to VAL for a separate per-epoch metric (e.g. DLC holdout)")
    ap.add_argument("--ckpt_every", type=int, default=1000,
                    help="also save adapters every N opt steps (power-cut insurance); 0=off")
    ap.add_argument("--num_workers", type=int, default=2)  # Windows commit limit
    ap.add_argument("--out", default="subs/dinov3_l512.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{args.name}"
    rdir = EXP_ROOT / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "config.json").write_text(json.dumps(vars(args), indent=2),
                                      encoding="utf-8")
    t0 = time.time()

    def log(msg):
        line = f"[{time.time()-t0:8.1f}s] {msg}"
        print(line, flush=True)
        with (rdir / "log.txt").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"=== {run_id} ===")
    log("config: " + json.dumps(vars(args)))

    model, n_dora, img_size, mean, std = build_model(args)
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"backbone={args.backbone} img={img_size} mean={mean} | "
        f"DoRA layers={n_dora} | trainable="
        f"{sum(p.numel() for p in trainable)/1e6:.2f}M")

    _tf = [T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
           T.RandomHorizontalFlip(), T.ColorJitter(0.15, 0.15, 0.1)]
    if args.recapture_p > 0:                 # private-axis: simulate capture pipeline
        _tf.append(RecaptureT(args.recapture_p, args.rc_min, args.rc_max))
        log(f"recapture aug ON: p={args.recapture_p} strength U[{args.rc_min},{args.rc_max}]")
    _tf += [T.ToTensor(), T.Normalize(mean, std)]
    train_tfm = T.Compose(_tf)
    eval_tfm = T.Compose([T.Resize((img_size, img_size)),
                          T.ToTensor(), T.Normalize(mean, std)])
    flip_tfm = T.Compose([T.Resize((img_size, img_size)),
                          T.RandomHorizontalFlip(1.0),
                          T.ToTensor(), T.Normalize(mean, std)])

    train_df, val_df = make_splits(args.seed, args.subset, args.loto_types,
                                   args.extra_data, args.extra_val)
    if args.extra_data:
        ex = train_df[train_df["is_extra"]]
        log(f"extra_data: +{len(ex)} rows in train "
            f"({int(ex['label'].sum())} fraud / {int((ex['label'] == 0).sum())} genuine; "
            f"types={sorted(ex['type'].unique())})")
    n_rec = int((~val_df["is_digital"] & ~val_df["is_extra"]).sum())
    n_ev = int(val_df["is_extra"].sum())
    if n_ev:
        log(f"extra_val: +{n_ev} rows in val (separate metric)")
    if args.loto_types:
        log(f"type-LOO: held-out types = {args.loto_types}")
        log(f"train types = {sorted(train_df['type'].unique())}")
    log(f"train={len(train_df)} | canary val={len(val_df)} "
        f"(digital {len(val_df)-n_rec-n_ev} + recaptured {n_rec} + extra {n_ev})")

    # type x class weighted sampler + pos_weight (as the source script)
    tw = 1.0 / train_df["type"].value_counts()
    n1 = max(1, int((train_df["label"] == 1).sum()))
    n0 = max(1, int((train_df["label"] == 0).sum()))
    cw = {0: len(train_df) / (2 * n0), 1: len(train_df) / (2 * n1)}
    sw = (train_df["type"].map(tw) * train_df["label"].map(cw)).values.astype(np.float32)
    if args.extra_frac > 0 and train_df["is_extra"].any():
        # rescale extra rows' sampler mass to the requested fraction (their
        # pseudo-types would otherwise each get a full type's share)
        xm = train_df["is_extra"].values
        f = args.extra_frac
        sw[xm] *= (f / (1 - f)) * (sw[~xm].sum() / max(sw[xm].sum(), 1e-9))
        log(f"extra sampler mass -> {sw[xm].sum() / sw.sum():.4f} "
            f"(~{sw[xm].sum() / sw.sum() * len(train_df):.0f} draws/epoch over {int(xm.sum())} imgs)")
    sampler = WeightedRandomSampler(torch.from_numpy(sw), len(train_df),
                                    replacement=True)

    train_loader = DataLoader(DS(train_df, train_tfm), batch_size=args.batch,
                              sampler=sampler, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=False)
    val_loader = DataLoader(DS(val_df, eval_tfm), batch_size=args.batch * 2,
                            shuffle=False, num_workers=2, pin_memory=True)

    if args.full_ft:                     # two groups: backbone (low) + head (high)
        bb_p = [p for p in model.bb.parameters() if p.requires_grad]
        hd_p = [p for p in model.head.parameters() if p.requires_grad]
        groups = [{"params": bb_p, "base_lr": args.lr},
                  {"params": hd_p, "base_lr": args.lr * args.head_lr_mult}]
        log(f"full-FT param groups: backbone lr={args.lr:.1e} ({sum(p.numel() for p in bb_p)/1e6:.0f}M) "
            f"head lr={args.lr*args.head_lr_mult:.1e}")
    else:
        groups = [{"params": trainable, "base_lr": args.lr}]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cw[0] / cw[1]).to(device))
    total_steps = (len(train_loader) // args.accum) * args.epochs

    def lr_at(s):
        if s < args.warmup_steps:
            return s / max(1, args.warmup_steps)
        prog = (s - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * prog))

    ema = None
    if args.ema_decay > 0:               # [EMA] ema_decay<=0 disables EMA (ablation)
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=args.ema_decay)

    def infer_model():
        return ema.module if ema is not None else model

    def adapters(sd):
        return {k: v for k, v in sd.items()
                if any(t in k for t in (".A", ".B", ".m", "head"))}

    def save_ckpt(epoch, step=None):
        if args.full_ft:
            obj = {"model": model.state_dict(), "args": vars(args), "epoch": epoch}
            if ema is not None:
                obj["ema"] = ema.module.state_dict()
            path = rdir / "model.pt"
        else:
            obj = {"adapters": adapters(model.state_dict()),
                   "args": vars(args), "epoch": epoch}
            if ema is not None:
                obj["ema_adapters"] = adapters(ema.module.state_dict())
            path = rdir / "adapters.pt"
        if step is not None:
            obj["opt_step"] = step
        tmp = path.with_suffix(".tmp")     # atomic-ish: never corrupt the last
        torch.save(obj, tmp)               # good checkpoint on power cut
        tmp.replace(path)

    opt_step = 0
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        opt.zero_grad(set_to_none=True)
        for it, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):  # [bf16]
                loss = crit(model(imgs), labels) / args.accum
            loss.backward()
            tot += loss.item() * args.accum
            if (it + 1) % args.accum == 0:
                sc = lr_at(opt_step)
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * sc
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                opt_step += 1
                if opt_step % 50 == 0:
                    log(f"  e{epoch} step {opt_step}/{total_steps} "
                        f"loss={tot/(it+1):.4f} lr={args.lr*lr_at(opt_step):.2e}")
                if args.ckpt_every > 0 and opt_step % args.ckpt_every == 0:
                    save_ckpt(epoch, opt_step)

        ps, ys = collect(infer_model(), val_loader, device)
        xtr = val_df["is_extra"].values
        dig = val_df["is_digital"].values & ~xtr
        rec = ~val_df["is_digital"].values & ~xtr
        r = freuid_score(ys[dig], ps[dig])
        m = {"epoch": epoch, "train_loss": round(tot / len(train_loader), 5),
             "canary_freuid": round(r.freuid, 5), "audet": round(r.audet, 5),
             "roc_auc": round(r.roc_auc, 5)}
        if n_rec >= 4:
            from sklearn.metrics import roc_auc_score
            ry, rp = ys[rec], ps[rec]
            try:
                m["ho_auc"] = round(float(roc_auc_score(ry, rp)), 4)
            except ValueError:
                m["ho_auc"] = None
            m["ho_gap"] = round(float(rp[ry == 1].mean() - rp[ry == 0].mean()), 4)
        if n_ev:
            from sklearn.metrics import roc_auc_score
            src = val_df["type"].str.split("/").str[0].values
            for s in sorted(set(src[xtr])):
                sm = xtr & (src == s)
                ey, ep_ = ys[sm], ps[sm]
                if len(set(ey)) > 1:
                    m[f"ev_{s}_auc"] = round(float(roc_auc_score(ey, ep_)), 4)
                m[f"ev_{s}_gp"] = round(float(ep_[ey == 0].mean()), 4)
                m[f"ev_{s}_fp"] = round(float(ep_[ey == 1].mean()), 4)
        with (rdir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(m) + "\n")
        log("metrics " + json.dumps(m))
        save_ckpt(epoch)

    # --- inference: hflip TTA on public test (EMA weights if enabled) ------
    if args.skip_infer:
        log("skip_infer set — done (training only)")
        return
    net = infer_model()
    te = load_test()
    present = te.df.copy()
    log(f"test present={len(present)} missing={len(te.missing)}")
    probs = np.zeros(len(present))
    for tfm in (eval_tfm, flip_tfm):
        ld = DataLoader(DS(present, tfm, has_label=False),
                        batch_size=args.batch * 2, shuffle=False,
                        num_workers=2, pin_memory=True)
        pp = []
        net.eval()
        with torch.no_grad():
            for imgs, _ in ld:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = net(imgs.to(device, non_blocking=True))
                pp.append(torch.sigmoid(lo.float()).cpu().numpy())
        probs += np.concatenate(pp)
    probs /= 2

    sub = pd.read_csv(REPO_ROOT / "sample_submission.csv", dtype={"id": str})
    smap = dict(zip(present["id"].astype(str), probs))
    sub["label"] = sub["id"].astype(str).map(smap).fillna(0.5).clip(0, 1)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    log(f"wrote {out} ({len(sub)} rows; "
        f"{int(sub['id'].astype(str).map(lambda i: i not in smap).sum())} filled 0.5)")
    log(f"DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
