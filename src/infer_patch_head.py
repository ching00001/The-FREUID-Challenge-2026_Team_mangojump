"""Inference for a trained attention-MIL head on cached test embeddings.

Usage:
  python -m src.infer_patch_head --run <head_run_id> --out subs/patchdino_f0.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from .data.paths import REPO_ROOT
from .experiment import EXP_ROOT, ExperimentConfig
from .train_patch_head import HeadModel, emb_tag, ART


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fill", type=float, default=0.5)
    args = ap.parse_args()

    d = EXP_ROOT / args.run
    ckpt = torch.load(d / "best.pt", map_location="cpu")
    cfg = ExperimentConfig(**json.loads((d / "config.json").read_text()))
    tag = ckpt.get("tag", emb_tag(cfg))

    z = np.load(ART / f"dino_emb_test_{tag}.npz", allow_pickle=True)
    ids = z["ids"].astype(str)
    emb = torch.from_numpy(z["emb"].astype(np.float32)).cuda()

    model = HeadModel(emb.shape[-1], cfg.drop_rate).cuda().eval()
    model.load_state_dict(ckpt["model"])
    with torch.no_grad():
        # chunk to be safe on memory
        out = []
        for s in range(0, len(emb), 4096):
            logit, _ = model(emb[s:s + 4096])
            out.append(torch.sigmoid(logit).cpu().numpy())
    preds = np.concatenate(out)
    print(f"test preds: n={len(preds)} mean={preds.mean():.4f} std={preds.std():.4f}")

    sub = pd.read_csv(REPO_ROOT / "sample_submission.csv", dtype={"id": str})
    smap = dict(zip(ids, preds))
    sub["label"] = sub["id"].astype(str).map(smap).fillna(args.fill)
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    n_fill = sub["id"].astype(str).map(lambda i: i not in smap).sum()
    print(f"wrote {args.out} ({len(sub)} rows; {n_fill} filled {args.fill})")


if __name__ == "__main__":
    main()
