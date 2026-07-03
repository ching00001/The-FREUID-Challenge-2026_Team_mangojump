"""Robustness diagnostic: does a trained model's score survive recapture?

Loads a run's best.pt and evaluates its VAL fold under several augmentation
profiles that simulate the digital->physical (print-and-capture) domain shift.
If FREUID is great on `clean` but collapses under `recapture_*`, the model is
riding fragile digital-manipulation artifacts that won't survive the private
OOD test — which motivates Phase-2 recapture augmentation.

Inference-only; no retraining. Reuses the model/metric/splits.

Usage:
    python -m src.eval_robustness --run <run_id>
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data.paths import REPO_ROOT, load_train
from .data.dataset import _load_rgb, _resize, _to_tensor
from .experiment import EXP_ROOT, ExperimentConfig
from .metric import freuid_score
from .models.factory import FraudNet

cv2.setNumThreads(0)


# --- recapture simulation profiles (applied to the already-resized RGB) ------
def _jpeg(img, q):
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _downup(img, scale):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _blur(img, k):
    k = int(k) | 1
    return cv2.GaussianBlur(img, (k, k), 0)


def _noise(img, sigma):
    n = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


PROFILE_NAMES = ["clean", "jpeg_q40", "jpeg_q25", "downup_0.6", "blur3",
                 "recapture_mild", "recapture_med", "recapture_heavy"]


def apply_profile(img: np.ndarray, name: str) -> np.ndarray:
    """Top-level (picklable) recapture-simulation dispatch for DataLoader workers."""
    if name.startswith("rc"):  # realistic recapture pipeline at given strength
        from .aug.recapture import recapture
        return recapture(img, strength=float(name[2:]), p=1.0)
    if name == "clean":
        return img
    if name == "jpeg_q40":
        return _jpeg(img, 40)
    if name == "jpeg_q25":
        return _jpeg(img, 25)
    if name == "downup_0.6":
        return _downup(img, 0.6)
    if name == "blur3":
        return _blur(img, 3)
    if name == "recapture_mild":
        return _jpeg(_downup(img, 0.7), 50)
    if name == "recapture_med":
        return _noise(_jpeg(_blur(_downup(img, 0.55), 3), 35), 4)
    if name == "recapture_heavy":
        return _noise(_jpeg(_blur(_downup(img, 0.45), 5), 25), 7)
    raise ValueError(f"unknown profile {name}")


class _EvalDS(Dataset):
    def __init__(self, df, cfg, profile):
        self.paths = df["abspath"].tolist()
        self.labels = df["label"].astype(np.float32).tolist()
        self.cfg = cfg
        self.profile = profile

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = _resize(_load_rgb(self.paths[i]), self.cfg.img_h, self.cfg.img_w)
        img = np.ascontiguousarray(apply_profile(img, self.profile))
        return _to_tensor(img), torch.tensor(self.labels[i])


@torch.no_grad()
def _predict(model, ds, cfg, amp=torch.bfloat16):
    ld = DataLoader(ds, batch_size=cfg.batch_size * 2, shuffle=False,
                    num_workers=2, pin_memory=True)
    ys, ps = [], []
    for x, y in ld:
        x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=amp):
            ps.append(torch.sigmoid(model(x).float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--profiles", nargs="*", default=PROFILE_NAMES)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True

    d = EXP_ROOT / args.run
    cfg = ExperimentConfig(**json.loads((d / "config.json").read_text()))
    ckpt = torch.load(d / "best.pt", map_location="cpu")
    model = FraudNet(cfg); model.load_state_dict(ckpt["model"])
    model.eval().cuda().to(memory_format=torch.channels_last)

    # rebuild the same val fold
    import pandas as pd
    folds = pd.read_csv(REPO_ROOT / cfg.folds_csv, dtype={"id": str})
    df = load_train().df[["id", "abspath"]].merge(folds, on="id")
    va = (df[df["type"] != cfg.loto_type] if cfg.cv_scheme == "loto"
          else df[df["skf_fold"] == cfg.val_fold]).reset_index(drop=True)

    print(f"run={args.run}  val={len(va)}  (best epoch {ckpt.get('epoch')})")
    print(f"{'profile':16s} {'FREUID':>8s} {'AuDET':>7s} {'APCER@1%':>9s} {'AUC':>7s}")
    rows = []
    for p in args.profiles:
        ys, ps = _predict(model, _EvalDS(va, cfg, p), cfg)
        r = freuid_score(ys, ps)
        print(f"{p:16s} {r.freuid:8.4f} {r.audet:7.4f} "
              f"{r.apcer_at_1pct_bpcer:9.4f} {r.roc_auc:7.4f}")
        rows.append({"profile": p, "freuid": r.freuid, "audet": r.audet,
                     "apcer_at_1pct_bpcer": r.apcer_at_1pct_bpcer,
                     "roc_auc": r.roc_auc})
    pd.DataFrame(rows).to_csv(d / "robustness.csv", index=False)
    print(f"wrote {d/'robustness.csv'}")


if __name__ == "__main__":
    main()
