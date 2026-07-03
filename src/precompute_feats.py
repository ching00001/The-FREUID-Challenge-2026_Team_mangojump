"""Precompute FROZEN backbone features and cache them to disk.

The decisive "is the gap the backbone?" experiment (DINOv3-7B + frozen probe)
needs the backbone run only ONCE per image — a frozen backbone's features never
change while a light probe trains. This extracts and caches them so the probe
(src.train_probe) trains in seconds and many backbones/probes can be compared
without re-running the (expensive) 7B forward.

Pooling: concat(CLS token, mean of patch tokens) -> a (2*D,) vector. This is the
standard strong linear-probe feature for DINO-family ViTs (CLS = global semantics,
mean-patch = fine-grained local artifact signal the DINOv3 forgery papers credit).

Memory (16GB card): the 7B weights are ~13.4GB in bf16, leaving little headroom.
inference_mode + bf16 + a small batch keeps activations tiny (no autograd graph).
If you OOM, lower --img_size (384 -> 320 -> 256) or --batch.

Output: artifacts/feat_<tag>_<split>_<img>.npz  with
  ids   (N,)      str
  emb   (N, 2D)   float16
  label (N,)      float32   (train only; test = zeros)

Usage:
  # smoke (small, fast backbone) to validate the pipeline:
  python -m src.precompute_feats --split train --backbone vit_large_patch16_dinov3.lvd1689m --img_size 512 --subset 200
  # the real decisive run (slow; background it):
  python -m src.precompute_feats --split train --backbone vit_7b_patch16_dinov3.lvd1689m --img_size 384 --batch 2
  python -m src.precompute_feats --split test  --backbone vit_7b_patch16_dinov3.lvd1689m --img_size 384 --batch 2
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .data.paths import REPO_ROOT, load_test, load_train

ART = REPO_ROOT / "artifacts"


def backbone_tag(backbone: str) -> str:
    """Short, filesystem-safe id, e.g. vit_7b_patch16_dinov3.lvd1689m -> vit_7b_dinov3."""
    head = backbone.split(".")[0]
    parts = head.split("_")
    keep = [p for p in parts if p in ("vit", "convnext") or p.startswith(("7b", "large", "huge", "base", "small", "giant"))]
    fam = "dinov3" if "dinov3" in backbone else ("dinov2" if "dinov2" in backbone
          else ("siglip" if "siglip" in backbone else parts[-1]))
    return "_".join(keep + [fam]) if keep else head.replace(".", "_")


class _ImgDS(Dataset):
    def __init__(self, df, tfm, has_label: bool):
        self.paths = df["abspath"].tolist()
        self.ids = df["id"].astype(str).tolist()
        self.labels = (df["label"].astype(np.float32).tolist()
                       if has_label else [0.0] * len(df))
        self.tfm = tfm

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(img), np.float32(self.labels[i]), self.ids[i]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--backbone", default="vit_7b_patch16_dinov3.lvd1689m")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)  # Windows commit limit
    ap.add_argument("--subset", type=int, default=0, help="first N images (smoke)")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    t0 = time.time()

    sp = load_train() if args.split == "train" else load_test()
    df = sp.df
    has_label = "label" in df.columns and args.split == "train"
    if args.subset > 0:
        df = df.iloc[:args.subset].reset_index(drop=True)
    print(f"{args.split}: {len(df)} images | backbone={args.backbone} @ {args.img_size}",
          flush=True)

    model = timm.create_model(args.backbone, pretrained=True, num_classes=0,
                              img_size=args.img_size)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    cfg = model.pretrained_cfg
    mean, std = cfg["mean"], cfg["std"]
    D = model.num_features
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  weights on GPU: {vram:.1f}GB | num_features={D} | prefix_tokens={n_prefix}",
          flush=True)

    tfm = T.Compose([T.Resize((args.img_size, args.img_size)),
                     T.ToTensor(), T.Normalize(mean, std)])
    ds = _ImgDS(df, tfm, has_label)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    embs = np.empty((len(df), 2 * D), dtype=np.float16)
    labels = np.empty(len(df), dtype=np.float32)
    ids: list[str] = []
    pos = 0
    for x, y, idb in ld:
        x = x.to(device, dtype=torch.bfloat16, non_blocking=True)
        tokens = model.forward_features(x)              # (B, N, D)
        cls = tokens[:, 0]
        patch = tokens[:, n_prefix:].mean(dim=1)        # mean over patch tokens
        feat = torch.cat([cls, patch], dim=-1).float()  # (B, 2D)
        b = feat.shape[0]
        embs[pos:pos + b] = feat.cpu().numpy().astype(np.float16)
        labels[pos:pos + b] = y.numpy()
        ids.extend(idb)
        pos += b
        if pos % 2000 < args.batch:
            r = pos / (time.time() - t0)
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  {pos}/{len(df)}  {r:.1f} img/s  peakVRAM {peak:.1f}GB  "
                  f"ETA {(len(df)-pos)/max(r,1e-6)/60:.1f} min", flush=True)

    ART.mkdir(parents=True, exist_ok=True)
    tag = backbone_tag(args.backbone)
    out = ART / f"feat_{tag}_{args.split}_{args.img_size}.npz"
    np.savez(out, ids=np.array(ids), emb=embs, label=labels)
    print(f"wrote {out}  emb {embs.shape}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
