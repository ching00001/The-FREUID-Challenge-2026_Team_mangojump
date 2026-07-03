"""FGTS — Fisher-Guided Token Selection on our DINOv3+DoRA backbone.

Hypothesis (arXiv 2511.22471): forgery cues are LOCAL; global mean-pool over all
1024 patch tokens dilutes them. Score each token POSITION by Fisher ratio
(inter-class / intra-class distance over the train set), keep the top-k most
discriminative positions, mean-pool ONLY those, train a light head.

Reuses the trained DINOv3+DoRA backbone (run 20260616..dinov3_l512, LB 0.01134),
FROZEN — we only add token selection + a new head. Built-in control: also pools
ALL tokens (same frozen features) so we cleanly isolate Fisher-selection's effect.

  python -m src.fgts                       # default run, top_k=128 -> 2 submissions
  python -m src.fgts --top_k 64
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .data.paths import REPO_ROOT, load_test
from .experiment import EXP_ROOT
from .train_DINOV3L_512 import build_model, make_splits


class _A:
    def __init__(s, a):
        s.backbone, s.rank, s.alpha = a["backbone"], a["rank"], a["alpha"]
        s.attn_only, s.img_size = a.get("attn_only", False), a.get("img_size", 0)


def load_backbone(run_id, device):
    ck = torch.load(EXP_ROOT / run_id / "adapters.pt", map_location="cpu", weights_only=False)
    model, _n, img, mean, std = build_model(_A(ck["args"]))
    model.load_state_dict(ck.get("ema_adapters", ck["adapters"]), strict=False)
    model.eval().to(device)
    return model.bb, img, mean, std, model.bb.num_prefix_tokens


class IMG(Dataset):
    def __init__(s, df, img, mean, std, lab=True):
        s.paths = df["abspath"].tolist()
        s.y = df["label"].astype(float).tolist() if lab else [0.0] * len(df)
        s.tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(mean, std)])

    def __len__(s): return len(s.paths)
    def __getitem__(s, i): return s.tf(Image.open(s.paths[i]).convert("RGB")), s.y[i]


@torch.no_grad()
def tokens(bb, x, npfx):                              # -> [B, npatch, D] fp32
    with torch.autocast("cuda", dtype=torch.bfloat16):  # [bf16] ~2x faster extraction
        f = bb.forward_features(x)[:, npfx:, :]
    return f.float()


@torch.no_grad()
def fisher_ranking(bb, df, npfx, img, mean, std, device, n_max=3000):
    # balanced subset
    sub = pd.concat([df[df.label == 0].sample(min(n_max // 2, int((df.label == 0).sum())), random_state=0),
                     df[df.label == 1].sample(min(n_max // 2, int((df.label == 1).sum())), random_state=0)])
    ld = DataLoader(IMG(sub, img, mean, std), batch_size=16, num_workers=4, pin_memory=True)
    # streaming per-token per-class sufficient stats
    s = {0: None, 1: None}; ss = {0: None, 1: None}; c = {0: 0, 1: 0}
    for x, y in ld:
        f = tokens(bb, x.to(device), npfx).cpu()      # [B,T,D]
        y = y.numpy()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
            pass
        for cls in (0, 1):
            m = y == cls
            if m.sum() == 0: continue
            fc = f[m]                                  # [n,T,D]
            su = fc.sum(0); sq = (fc ** 2).sum(0).sum(-1)  # [T,D], [T]
            s[cls] = su if s[cls] is None else s[cls] + su
            ss[cls] = sq if ss[cls] is None else ss[cls] + sq
            c[cls] += int(m.sum())
    mu = {cls: s[cls] / c[cls] for cls in (0, 1)}                       # [T,D]
    # RMS intra (streamable proxy): sqrt(E||f||^2 - ||mu||^2) per token
    intra = {cls: torch.sqrt(torch.clamp(ss[cls] / c[cls] - (mu[cls] ** 2).sum(-1), min=0)) for cls in (0, 1)}
    inter = (mu[0] - mu[1]).norm(dim=-1)                                # [T]
    fisher = inter / (0.5 * (intra[0] + intra[1]) + 1e-8)               # [T]
    ranking = torch.argsort(fisher, descending=True)                   # [T] best->worst
    print(f"[fisher] tokens={len(fisher)} median={fisher.median():.3f} "
          f"top1={fisher[ranking[0]]:.3f}")
    return ranking, fisher


@torch.no_grad()
def pooled(bb, df, npfx, img, mean, std, device, idx_sets, lab=True):
    """idx_sets: {name: LongTensor of token indices}. One forward pass, pool each."""
    ld = DataLoader(IMG(df, img, mean, std, lab), batch_size=16, num_workers=4, pin_memory=True)
    out = {n: [] for n in idx_sets}; Y = []
    for x, y in ld:
        f = tokens(bb, x.to(device), npfx)            # [B,T,D]
        for n, idx in idx_sets.items():
            out[n].append(f[:, idx, :].mean(1).cpu())
        Y.append(np.asarray(y, dtype=float))
    return {n: torch.cat(v).numpy() for n, v in out.items()}, np.concatenate(Y)


def train_head(Xtr, ytr, device, epochs=120, pw=1.0):
    D = Xtr.shape[1]
    head = nn.Sequential(nn.LayerNorm(D), nn.Dropout(0.2), nn.Linear(D, 1)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    X = torch.tensor(Xtr, dtype=torch.float32, device=device)
    y = torch.tensor(ytr, dtype=torch.float32, device=device)
    head.train()
    for e in range(epochs):
        opt.zero_grad()
        loss = crit(head(X).squeeze(-1), y)
        loss.backward(); opt.step()
    head.eval()
    return head


@torch.no_grad()
def predict(head, X, device):
    return torch.sigmoid(head(torch.tensor(X, dtype=torch.float32, device=device)).squeeze(-1)).cpu().numpy()


def write_sub(ids, probs, out):
    sub = pd.read_csv(REPO_ROOT / "sample_submission.csv", dtype={"id": str})
    sub["label"] = sub["id"].astype(str).map(dict(zip(ids, probs))).fillna(0.5).clip(0, 1)
    (REPO_ROOT / out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(REPO_ROOT / out, index=False)
    r = sub.loc[sub["id"].astype(str).isin(set(ids)), "label"].to_numpy()
    print(f"wrote {out}  std={r.std():.3f} mid={((r>=.01)&(r<=.99)).sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20260616_134434_dinov3_l512")
    ap.add_argument("--top_k", default="64,128,256,512", help="comma-sep k values to sweep")
    ap.add_argument("--tag", default="dinov3", help="output prefix: subs/<tag>_fgts_<k>.csv")
    args = ap.parse_args()
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    bb, img, mean, std, npfx = load_backbone(args.run, device)
    tr_df, _va = make_splits(42, 0)
    tr_df = tr_df[tr_df["is_digital"]].reset_index(drop=True)   # train heads on digital train
    print(f"backbone={args.run} img={img} prefix={npfx} | train={len(tr_df)}")

    ranking, _f = fisher_ranking(bb, tr_df, npfx, img, mean, std, device)
    ks = [int(k) for k in args.top_k.split(",")]
    idx_sets = {f"k{k}": ranking[:k].sort().values for k in ks}
    idx_sets["all"] = torch.arange(len(ranking))               # control: all tokens

    n1, n0 = max(1, int((tr_df.label == 1).sum())), max(1, int((tr_df.label == 0).sum()))
    pw = n0 / n1
    print("extracting train pooled features (one pass, all k)...")
    Xtr, ytr = pooled(bb, tr_df, npfx, img, mean, std, device, idx_sets)
    print("extracting test pooled features...")
    te = load_test().df.copy()
    Xte, _ = pooled(bb, te, npfx, img, mean, std, device, idx_sets, lab=False)
    ids = te["id"].astype(str).tolist()

    for name in idx_sets:
        head = train_head(Xtr[name], ytr, device, pw=pw)
        out = f"subs/{args.tag}_fgts_{name}.csv" if name != "all" else f"subs/{args.tag}_allpool.csv"
        write_sub(ids, predict(head, Xte[name], device), out)
        print(f"  [{name}] done")


if __name__ == "__main__":
    main()
