"""Dataset + transforms for FREUID.

Phase 1 keeps augmentation LIGHT (the heavy print-and-capture / recapture
simulation that bridges the digital->physical domain gap is Phase 2, kept in a
separate module). All cards share ~1.585 aspect, so we resize to a fixed
(img_h, img_w) with NO distortion.

We deliberately avoid horizontal flips: ID layouts are not left/right symmetric
and flipping would teach an invariance that does not exist in the data.

Implemented with cv2 + numpy (no albumentations dependency yet).
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..aug.recapture import recapture

cv2.setNumThreads(0)  # avoid oversubscription with DataLoader workers

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _aug_train(img: np.ndarray, cfg, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]

    # affine: small rotation + scale jitter (keeps fine detail, no big warps)
    if cfg.rotate_deg > 0 or cfg.scale_jitter > 0:
        ang = rng.uniform(-cfg.rotate_deg, cfg.rotate_deg)
        sc = 1.0 + rng.uniform(-cfg.scale_jitter, cfg.scale_jitter)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT101)

    # brightness / contrast (keep math in float32 to avoid float64 temporaries)
    if cfg.brightness > 0 or cfg.contrast > 0:
        b = np.float32(1.0 + rng.uniform(-cfg.brightness, cfg.brightness))
        c = np.float32(1.0 + rng.uniform(-cfg.contrast, cfg.contrast))
        mean = np.float32(img.mean())
        img = np.clip((img.astype(np.float32) - mean) * c + mean * b, 0, 255
                      ).astype(np.uint8)

    # mild JPEG re-encode (even Phase 1: cheap nod to recapture robustness)
    if cfg.jpeg_p > 0 and rng.random() < cfg.jpeg_p:
        q = int(rng.integers(cfg.jpeg_quality_min, 96))
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR),
                               cv2.COLOR_BGR2RGB)

    if cfg.hflip_p > 0 and rng.random() < cfg.hflip_p:
        img = img[:, ::-1]
    return img


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class FreuidDataset(Dataset):
    def __init__(self, df, cfg, train: bool):
        self.ids = df["id"].astype(str).tolist()
        self.paths = df["abspath"].tolist()
        self.labels = (df["label"].astype(np.float32).tolist()
                       if "label" in df.columns else [0.0] * len(df))
        self.cfg = cfg
        self.train = train

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        cfg = self.cfg
        img = _load_rgb(self.paths[i])
        img = _resize(img, cfg.img_h, cfg.img_w)
        if self.train:
            rng = np.random.default_rng()
            img = _aug_train(img, cfg, rng)
            # Phase-2 recapture aug: random strength, applied to BOTH classes
            if cfg.recapture_p > 0:
                s = rng.uniform(cfg.recapture_strength_min,
                                cfg.recapture_strength_max)
                img = recapture(img, strength=s, p=cfg.recapture_p, rng=rng)
        elif cfg.val_recapture > 0:
            # deterministic per-sample recapture so the rc-val metric (our
            # public-LB proxy) is stable across epochs and runs.
            seed = zlib.crc32(self.ids[i].encode())
            img = recapture(img, strength=cfg.val_recapture, p=1.0,
                            rng=np.random.default_rng(seed))
        x = _to_tensor(np.ascontiguousarray(img))
        y = torch.tensor(self.labels[i], dtype=torch.float32)
        return x, y, self.ids[i]


def make_tta_views(img: np.ndarray, mode: str):
    """Return a list of (numpy) views for test-time augmentation."""
    views = [img]
    if mode in ("hflip", "all"):
        views.append(img[:, ::-1])
    return views
