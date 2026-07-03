"""Train the attention-MIL head on cached frozen-DINOv2 embeddings (fast).

Loads artifacts/dino_emb_train_<tag>.npz into GPU memory and trains only the
AttnMIL pooling + linear head -> seconds/epoch, so we can iterate rapidly.
Reuses ExperimentConfig/Logger so runs land in experiments/registry.csv.

The frozen-backbone clean-val FREUID this produces is itself diagnostic: if it
does NOT saturate to ~0 like the CNN baseline, DINOv2 features are not encoding
the fragile source-artifact.

Usage:
  python -m src.train_patch_head --name phead_f0 --val_fold 0 --epochs 40 --lr 1e-3
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data.paths import REPO_ROOT, load_train
from .experiment import ExperimentLogger
from .metric import freuid_score
from .models.patch_dino import AttnMILPool
from .train import parse_cfg, set_seed

ART = REPO_ROOT / "artifacts"


class HeadModel(nn.Module):
    def __init__(self, dim: int, drop: float):
        super().__init__()
        self.pool = AttnMILPool(dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(drop),
                                  nn.Linear(dim, 1))

    def forward(self, f):                       # f: (B,K,D)
        pooled, attn = self.pool(f)
        return self.head(pooled).squeeze(-1), attn


def emb_tag(cfg):
    return f"{cfg.patch_backbone.split('_')[1]}_{cfg.patch_grid_cols}x{cfg.patch_grid_rows}"


def main():
    cfg = parse_cfg()
    cfg.arch = "patch_head"
    set_seed(cfg.seed)
    device = "cuda"
    logger = ExperimentLogger(cfg)

    tag = emb_tag(cfg)
    z = np.load(ART / f"dino_emb_train_{tag}.npz", allow_pickle=True)
    emb_ids = z["ids"].astype(str)
    emb = torch.from_numpy(z["emb"].astype(np.float32))      # (N,K,D)
    lab = torch.from_numpy(z["label"].astype(np.float32))
    id2row = {i: r for r, i in enumerate(emb_ids)}

    folds = pd.read_csv(REPO_ROOT / cfg.folds_csv, dtype={"id": str})
    df = load_train().df[["id"]].merge(folds, on="id")
    if cfg.cv_scheme == "loto":
        tr_ids = df[df["type"] != cfg.loto_type]["id"].tolist()
        va_ids = df[df["type"] == cfg.loto_type]["id"].tolist()
    else:
        tr_ids = df[df["skf_fold"] != cfg.val_fold]["id"].tolist()
        va_ids = df[df["skf_fold"] == cfg.val_fold]["id"].tolist()
    tr_idx = torch.tensor([id2row[i] for i in tr_ids if i in id2row])
    va_idx = torch.tensor([id2row[i] for i in va_ids if i in id2row])

    Xtr, ytr = emb[tr_idx].to(device), lab[tr_idx].to(device)
    Xva, yva = emb[va_idx].to(device), lab[va_idx].to(device)
    va_id_arr = emb_ids[va_idx.numpy()]
    D = emb.shape[-1]
    logger.log(f"emb tag={tag} D={D}  train={len(Xtr)} val={len(Xva)} "
               f"val_fraud={ytr.float().mean():.3f}/{yva.float().mean():.3f}")

    model = HeadModel(D, cfg.drop_rate).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs,
                                                       eta_min=cfg.lr * cfg.min_lr_ratio)
    bce = nn.BCEWithLogitsLoss()
    bs = cfg.batch_size
    noise = 0.05            # feature-space gaussian noise (regularisation)
    pdrop = 0.1             # patch dropout prob (robustness to missing regions)

    best = {"freuid": 1e9, "epoch": -1}
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        tot = 0.0
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            x = Xtr[idx]
            if noise > 0:
                x = x + noise * torch.randn_like(x)
            if pdrop > 0:                         # randomly mask whole patches
                keep = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > pdrop)
                x = x * keep
            logit, _ = model(x)
            loss = bce(logit, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            vlogit, _ = model(Xva)
            vp = torch.sigmoid(vlogit).cpu().numpy()
        r = freuid_score(yva.cpu().numpy(), vp)
        if epoch % 5 == 0 or epoch == cfg.epochs - 1 or r.freuid < best["freuid"]:
            logger.log_metrics(epoch, {"epoch": epoch,
                "train_loss": round(tot / len(Xtr), 5),
                "freuid": round(r.freuid, 5), "audet": round(r.audet, 5),
                "apcer@1bpcer": round(r.apcer_at_1pct_bpcer, 5),
                "roc_auc": round(r.roc_auc, 5)})
        if r.freuid < best["freuid"]:
            best = {"freuid": r.freuid, "audet": r.audet,
                    "apcer_at_1pct_bpcer": r.apcer_at_1pct_bpcer,
                    "roc_auc": r.roc_auc, "epoch": epoch}
            torch.save({"model": model.state_dict(), "tag": tag,
                        "cfg": dataclasses.asdict(cfg)}, logger.save_path("best.pt"))
            pd.DataFrame({"id": va_id_arr, "label": yva.cpu().numpy(),
                          "score": vp}).to_csv(logger.save_path("oof.csv"), index=False)

    logger.finalize({"best_epoch": best["epoch"], **{k: round(v, 5)
                     for k, v in best.items() if k != "epoch"}})


if __name__ == "__main__":
    main()
