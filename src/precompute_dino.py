"""Precompute frozen-DINOv2 grid-patch embeddings and cache them.

Because the DINOv2 backbone is FROZEN, its patch embeddings never change during
head training. Computing them once (here) lets the attention-MIL head train in
seconds/epoch instead of ~68 min, enabling rapid iteration.

For each image: decode at half-res (IMREAD_REDUCED_COLOR_2, ~4x faster), extract
a fixed cols x rows grid of patches (no jitter -> deterministic), resize to
patch_px, run frozen DINOv2, and store the per-patch feature vectors.

Output: artifacts/dino_emb_<split>_<tag>.npz with
  ids   (N,)            str
  emb   (N, K, D)       float16   K = cols*rows patch embeddings
  label (N,)            float32   (train only; test = zeros)

Usage:
  python -m src.precompute_dino --split train --cols 4 --rows 3
  python -m src.precompute_dino --split test  --cols 4 --rows 3
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import timm
import torch
from torch.utils.data import DataLoader, Dataset

from .data.paths import REPO_ROOT, load_train, load_test
from .data.dataset import IMAGENET_MEAN, IMAGENET_STD

cv2.setNumThreads(0)
ART = REPO_ROOT / "artifacts"


class _GridDS(Dataset):
    def __init__(self, df, cols, rows, px):
        self.ids = df["id"].astype(str).tolist()
        self.paths = df["abspath"].tolist()
        self.labels = (df["label"].astype(np.float32).tolist()
                       if "label" in df.columns else [0.0] * len(df))
        self.cols, self.rows, self.px = cols, rows, px

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_REDUCED_COLOR_2)
        if img is None:                                   # fallback full res
            img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        ch, cw = H // self.rows, W // self.cols
        K = self.rows * self.cols
        out = np.empty((K, self.px, self.px, 3), dtype=np.uint8)
        k = 0
        for r in range(self.rows):
            for c in range(self.cols):
                patch = img[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
                out[k] = cv2.resize(patch, (self.px, self.px),
                                    interpolation=cv2.INTER_AREA)
                k += 1
        x = out.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
        return x, np.float32(self.labels[i]), self.ids[i]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--backbone", default="vit_small_patch14_reg4_dinov2")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--px", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16, help="images per batch")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True

    sp = load_train() if args.split == "train" else load_test()
    df = sp.df
    print(f"{args.split}: {len(df)} images, grid {args.cols}x{args.rows}")

    model = timm.create_model(args.backbone, pretrained=True, num_classes=0,
                              img_size=args.px).cuda().eval()
    D = model.num_features
    K = args.cols * args.rows

    ds = _GridDS(df, args.cols, args.rows, args.px)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    embs = np.empty((len(df), K, D), dtype=np.float16)
    labels = np.empty(len(df), dtype=np.float32)
    ids = []
    pos = 0
    t0 = time.time()
    for x, y, idb in ld:
        b = x.shape[0]
        x = x.cuda(non_blocking=True).flatten(0, 1)        # (b*K,3,P,P)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model(x).float()                           # (b*K, D)
        embs[pos:pos + b] = f.view(b, K, D).cpu().numpy().astype(np.float16)
        labels[pos:pos + b] = y.numpy()
        ids.extend(idb)
        pos += b
        if pos % 4000 < args.batch:
            r = pos / (time.time() - t0)
            print(f"  {pos}/{len(df)}  {r:.0f} img/s  ETA {(len(df)-pos)/r:.0f}s",
                  flush=True)

    tag = f"{args.backbone.split('_')[1]}_{args.cols}x{args.rows}"
    out = ART / f"dino_emb_{args.split}_{tag}.npz"
    np.savez(out, ids=np.array(ids), emb=embs, label=labels)
    print(f"wrote {out}  emb shape {embs.shape}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
