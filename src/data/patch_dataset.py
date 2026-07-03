"""Grid-patch dataset for the patch-DINOv2 detector.

Extracts a grid of cols x rows patches from the FULL-resolution document (so
each patch keeps real local detail), resizes each to patch_px, and returns a
(K,3,P,P) tensor per image. Augmentation is intentionally LIGHT (the Phase-2
negative result showed heavy aug hurts) — only mild photometric jitter on the
full image plus per-cell crop jitter; val is deterministic (no jitter).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import _load_rgb, IMAGENET_MEAN, IMAGENET_STD

cv2.setNumThreads(0)


def _photometric(img, rng):
    b = 1.0 + rng.uniform(-0.1, 0.1)
    c = 1.0 + rng.uniform(-0.1, 0.1)
    m = img.mean()
    out = np.clip((img.astype(np.float32) - m) * c + m * b, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:                      # mild re-encode
        q = int(rng.integers(55, 96))
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out


def extract_grid(img, cols, rows, px, jitter, rng):
    H, W = img.shape[:2]
    ch, cw = H // rows, W // cols
    out = np.empty((rows * cols, px, px, 3), dtype=np.uint8)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * ch, c * cw
            if jitter > 0 and rng is not None:
                y0 = int(np.clip(y0 + rng.uniform(-jitter, jitter) * ch, 0, H - ch))
                x0 = int(np.clip(x0 + rng.uniform(-jitter, jitter) * cw, 0, W - cw))
            patch = img[y0:y0 + ch, x0:x0 + cw]
            out[idx] = cv2.resize(patch, (px, px), interpolation=cv2.INTER_AREA)
            idx += 1
    return out                                   # (K, px, px, 3)


def _to_tensor(patches: np.ndarray) -> torch.Tensor:
    x = patches.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD       # (K,px,px,3)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))  # (K,3,px,px)


class PatchDataset(Dataset):
    def __init__(self, df, cfg, train: bool):
        self.ids = df["id"].astype(str).tolist()
        self.paths = df["abspath"].tolist()
        self.labels = (df["label"].astype(np.float32).tolist()
                       if "label" in df.columns else [0.0] * len(df))
        self.cfg = cfg
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cfg = self.cfg
        img = _load_rgb(self.paths[i])
        if self.train:
            rng = np.random.default_rng()
            img = _photometric(img, rng)
            patches = extract_grid(img, cfg.patch_grid_cols, cfg.patch_grid_rows,
                                   cfg.patch_px, cfg.patch_jitter, rng)
        else:
            patches = extract_grid(img, cfg.patch_grid_cols, cfg.patch_grid_rows,
                                   cfg.patch_px, 0.0, None)
        x = _to_tensor(patches)
        y = torch.tensor(self.labels[i], dtype=torch.float32)
        return x, y, self.ids[i]
