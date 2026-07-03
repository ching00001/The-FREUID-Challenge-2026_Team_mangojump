"""Train a light probe on FROZEN cached features (src.precompute_feats).

This is the model-side half of the decisive "is the 14x gap the backbone?" test.
The literature (Simplicity Prevails, arXiv 2602.01738) finds frozen features + a
linear probe beat LoRA/DoRA fine-tuning for OOD generalization. We hold the split
IDENTICAL to the DoRA runs (reuse make_splits: full-train + 0.5% canary + 20 real
recaptured) so the only changed variables are {backbone, frozen-vs-DoRA}:

  DINOv3-L  DoRA   = 0.01134   (have it; subs/dinov3_l512.csv)
  DINOv3-L  frozen = ?         (isolates the REGIME: frozen vs DoRA)
  DINOv3-7B frozen = ?         (REGIME + SCALE: the real "bigger backbone" lever)

Decision rule:
  * 7B-frozen closes most of the gap  -> it WAS the backbone (scale).
  * L-frozen ~ L-DoRA but 7B-frozen better -> pure scale.
  * frozen ~ DoRA and neither nears the leaders -> the gap is NOT the backbone
    (it's specialized hard-fraud components / external data); stop chasing it.

Usage:
  python -m src.train_probe --tag vit_large_dinov3 --img 512 --name probe_L_frozen
  python -m src.train_probe --tag vit_7b_dinov3 --img 384 --name probe_7b_frozen \
      --out subs/dinov3_7b_frozen.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .data.paths import REPO_ROOT, load_test
from .metric import freuid_score
from .train_DINOV3L_512 import make_splits

ART = REPO_ROOT / "artifacts"
EXP_ROOT = REPO_ROOT / "experiments"


def load_feats(tag: str, split: str, img: int):
    f = ART / f"feat_{tag}_{split}_{img}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} — run src.precompute_feats for {split} first")
    z = np.load(f, allow_pickle=True)
    ids = z["ids"].astype(str)
    return {i: k for k, i in enumerate(ids)}, z["emb"], z["label"]


class Probe(nn.Module):
    """LayerNorm -> (optional MLP) -> 1 logit. hidden=0 => pure linear probe."""
    def __init__(self, d_in: int, hidden: int, p_drop: float):
        super().__init__()
        layers = [nn.LayerNorm(d_in), nn.Dropout(p_drop)]
        if hidden > 0:
            layers += [nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(p_drop),
                       nn.Linear(hidden, 1)]
        else:
            layers += [nn.Linear(d_in, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="backbone tag from precompute_feats")
    ap.add_argument("--img", type=int, required=True)
    ap.add_argument("--name", default="probe")
    ap.add_argument("--hidden", type=int, default=0, help="0=linear probe; >0=MLP")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loto_types", default="")
    ap.add_argument("--out", default="subs/probe_frozen.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda"
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{args.name}"
    rdir = EXP_ROOT / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    t0 = time.time()

    def log(msg):
        line = f"[{time.time()-t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (rdir / "log.txt").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"=== {run_id} ===  config: {json.dumps(vars(args))}")

    # --- cached features + identical split ---------------------------------
    idx, emb_tr, _ = load_feats(args.tag, "train", args.img)
    emb_tr = emb_tr.astype(np.float32)
    train_df, val_df = make_splits(args.seed, 0, args.loto_types)

    def gather(df):
        keep = df[df["id"].astype(str).isin(idx)].copy()
        rows = [idx[i] for i in keep["id"].astype(str)]
        return keep.reset_index(drop=True), emb_tr[rows]

    train_df, X_tr = gather(train_df)
    val_df, X_val = gather(val_df)
    y_tr = train_df["label"].values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.float32)
    dig = val_df["is_digital"].values
    n_rec = int((~val_df["is_digital"]).sum())
    log(f"train={len(train_df)} val={len(val_df)} (digital {len(val_df)-n_rec} "
        f"+ recaptured {n_rec}) | feat_dim={X_tr.shape[1]}")

    # standardize with TRAIN stats (helps the linear probe; LayerNorm also does)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_val = (X_val - mu) / sd
    Xtr = torch.from_numpy(X_tr).to(device)
    ytr = torch.from_numpy(y_tr).to(device)
    Xva = torch.from_numpy(X_val).to(device)

    # type x class weighted sampler + pos_weight (mirrors train_siglip512)
    tw = 1.0 / train_df["type"].value_counts()
    n1 = max(1, int((train_df["label"] == 1).sum()))
    n0 = max(1, int((train_df["label"] == 0).sum()))
    cw = {0: len(train_df) / (2 * n0), 1: len(train_df) / (2 * n1)}
    sw = (train_df["type"].map(tw) * train_df["label"].map(cw)).values.astype(np.float32)
    sampler = WeightedRandomSampler(torch.from_numpy(sw), len(train_df), replacement=True)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch, sampler=sampler)

    probe = Probe(X_tr.shape[1], args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cw[0] / cw[1], device=device))

    best = {"freuid": 1e9}
    for epoch in range(args.epochs):
        probe.train()
        tot = 0.0
        for xb, yb in loader:
            loss = crit(probe(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item()
        probe.eval()
        with torch.no_grad():
            pv = torch.sigmoid(probe(Xva)).cpu().numpy()
        r = freuid_score(y_val[dig], pv[dig])
        m = {"epoch": epoch, "loss": round(tot / len(loader), 5),
             "canary_freuid": round(r.freuid, 5), "roc_auc": round(r.roc_auc, 5)}
        if n_rec >= 4:
            ry, rp = y_val[~dig], pv[~dig]
            try:
                m["ho_auc"] = round(float(roc_auc_score(ry, rp)), 4)
            except ValueError:
                m["ho_auc"] = None
            m["ho_gap"] = round(float(rp[ry == 1].mean() - rp[ry == 0].mean()), 4)
        with (rdir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(m) + "\n")
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            log("metrics " + json.dumps(m))
        if r.freuid <= best["freuid"]:
            best = {"freuid": r.freuid, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}}
    log(f"best canary_freuid={best['freuid']:.5f} @ epoch {best['epoch']}")
    probe.load_state_dict(best["state"])
    torch.save({"state": best["state"], "mu": mu, "sd": sd, "args": vars(args)},
               rdir / "probe.pt")

    # --- inference on public test ------------------------------------------
    idx_te, emb_te, _ = load_feats(args.tag, "test", args.img)
    te = load_test()
    present = te.df.copy()
    present = present[present["id"].astype(str).isin(idx_te)].reset_index(drop=True)
    rows = [idx_te[i] for i in present["id"].astype(str)]
    Xte = (emb_te[rows].astype(np.float32) - mu) / sd
    probe.eval()
    with torch.no_grad():
        pt = torch.sigmoid(probe(torch.from_numpy(Xte).to(device))).cpu().numpy()

    sub = pd.read_csv(REPO_ROOT / "sample_submission.csv", dtype={"id": str})
    smap = dict(zip(present["id"].astype(str), pt))
    n_fill = int(sub["id"].astype(str).map(lambda i: i not in smap).sum())
    sub["label"] = sub["id"].astype(str).map(smap).fillna(0.5).clip(0, 1)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    log(f"wrote {out} ({len(sub)} rows; {n_fill} filled 0.5) | DONE {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
