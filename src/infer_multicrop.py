"""Multi-crop TTA inference on a trained SigLIP-DoRA run — zero extra training.

The card is ~1.585 (landscape); squashing the whole image to 512x512 halves
horizontal text resolution. Predicting on left/right (or quadrant) crops, each
resized to 512, restores horizontal pixels for text fields, then we average the
per-view probabilities (mean, NOT rank — rank smears the confident operating
point that APCER@1%BPCER rewards).

Loads a run's EMA adapters over the frozen pretrained backbone, so it reuses the
exact 0.02667 weights.

Usage:
  python -m src.infer_multicrop --run 20260612_134841_siglip512_dora_full \
      --crops halves --hflip --out subs/siglip512_mc.csv
  python -m src.infer_multicrop --run <id> --crops full --hflip --out subs/base.csv  # sanity = reproduce 0.02667
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .data.paths import REPO_ROOT, load_test
from .experiment import EXP_ROOT
from .train_DINOV3L_512 import build_model


class _Args:                       # minimal args for build_model
    def __init__(self, backbone, rank, alpha, attn_only):
        self.backbone, self.rank, self.alpha = backbone, rank, alpha
        self.attn_only, self.img_size = attn_only, 0


def load_run(run_id: str, device: str):
    ck = torch.load(EXP_ROOT / run_id / "adapters.pt", map_location="cpu",
                    weights_only=False)
    a = ck["args"]
    args = _Args(a["backbone"], a["rank"], a["alpha"], a.get("attn_only", False))
    args.img_size = a.get("img_size", 0)            # honor trained resolution (e.g. DINOv3 @512)
    model, _n, img, mean, std = build_model(args)
    sd = ck.get("ema_adapters", ck["adapters"])     # prefer EMA weights
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    # only frozen base weights should be "missing" (they come from pretrained)
    assert all(("lora" not in m and ".A" not in m and ".B" not in m
                and ".m" not in m and "head" not in m) for m in missing), \
        "a trained adapter/head tensor failed to load"
    model.eval().to(device)
    return model, img, mean, std


# view = (name, fractional crop box (l,t,r,b) or None for full, hflip bool)
def build_views(crops: str, hflip: bool):
    o = 0.05                                        # crop overlap
    views = [("full", None)]
    if crops in ("halves", "both"):
        views += [("left", (0.0, 0.0, 0.5 + o, 1.0)),
                  ("right", (0.5 - o, 0.0, 1.0, 1.0))]
    if crops in ("quad", "both"):
        views += [("tl", (0, 0, 0.5 + o, 0.5 + o)), ("tr", (0.5 - o, 0, 1, 0.5 + o)),
                  ("bl", (0, 0.5 - o, 0.5 + o, 1)), ("br", (0.5 - o, 0.5 - o, 1, 1))]
    out = []
    for name, box in views:
        out.append((name, box, False))
        if hflip:
            out.append((name + "_f", box, True))
    return out


class ViewDS(Dataset):
    def __init__(self, paths, box, flip, img, mean, std):
        self.paths, self.box, self.flip = paths, box, flip
        self.tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        if self.box is not None:
            W, H = im.size
            l, t, r, b = self.box
            im = im.crop((int(l * W), int(t * H), int(r * W), int(b * H)))
        if self.flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return self.tf(im), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--crops", default="halves", choices=["full", "halves", "quad", "both"])
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--agg", default="mean", choices=["mean", "max"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    model, img, mean, std = load_run(args.run, device)
    present = load_test().df.copy()
    paths = present["abspath"].tolist()
    views = build_views(args.crops, args.hflip)
    print(f"run={args.run} img={img} | views={[v[0] for v in views]} "
          f"| agg={args.agg} | test={len(paths)}")

    per_view = []
    for name, box, flip in views:
        ld = DataLoader(ViewDS(paths, box, flip, img, mean, std),
                        batch_size=args.batch * 2, shuffle=False,
                        num_workers=4, pin_memory=True)
        pp = np.zeros(len(paths))
        with torch.no_grad():
            for x, idx in ld:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = torch.sigmoid(model(x.to(device)).float()).cpu().numpy()
                pp[idx.numpy()] = p
        per_view.append(pp)
        print(f"  view {name}: mean={pp.mean():.4f}")

    P = np.vstack(per_view)
    probs = P.max(0) if args.agg == "max" else P.mean(0)

    sub = pd.read_csv(REPO_ROOT / "sample_submission.csv", dtype={"id": str})
    smap = dict(zip(present["id"].astype(str), probs))
    sub["label"] = sub["id"].astype(str).map(smap).fillna(0.5).clip(0, 1)
    Path(REPO_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(REPO_ROOT / args.out, index=False)
    r = probs
    print(f"ensemble: mean={r.mean():.4f} std={r.std():.4f} "
          f"<0.01:{(r<0.01).sum()} >0.99:{(r>0.99).sum()} mid:{((r>=0.01)&(r<=0.99)).sum()}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
