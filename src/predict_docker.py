"""Offline inference entrypoint (Docker contract) for the frozen system.

Contract: flat image dir mounted at /data (ids = filenames w/o extension),
output /submissions/submission.csv with columns id,label. No network, no
training — everything comes from artifacts/system/ (see export_system.py).

  VARIANT=routed  champ head + (capture AND dist>floor) -> dual-axis PAD  (A)
  VARIANT=plain   champ head only                                         (B)

  python -m src.predict_docker --data public_test --out submission.csv \
      --variant routed          # local rehearsal
  docker run --network none -v .../images:/data:ro -v .../out:/submissions \
      -e VARIANT=routed freuid-lilwu
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

REPO_ROOT = Path(__file__).resolve().parents[1]
SYS = REPO_ROOT / "artifacts" / "system"
EXTS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class _A:                                       # attr-dict for saved args
    def __init__(s, d): s.__dict__.update(d)
    def __getattr__(s, k): return s.__dict__.get(k, 0)


def load_member(m, spec, device):
    """Backbone + slim DoRA adapters + preprocessing, fully offline."""
    from .train_DINOV3L_512 import build_model
    ck = torch.load(SYS / "adapters_slim" / f"{m}.pt", map_location="cpu",
                    weights_only=False)
    model, _n, img, mean, std = build_model(_A(ck["args"]))
    model.load_state_dict(ck.get("ema_adapters", ck["adapters"]), strict=False)
    model.eval().to(device)
    return model.bb, img, mean, std, model.bb.num_prefix_tokens


class FlatDir(Dataset):
    def __init__(s, paths, img, mean, std):
        s.paths = paths
        s.tf = T.Compose([T.Resize((img, img)), T.ToTensor(),
                          T.Normalize(mean, std)])

    def __len__(s): return len(s.paths)

    def __getitem__(s, i):
        return s.tf(Image.open(s.paths[i]).convert("RGB")), i


@torch.no_grad()
def member_features(m, spec, paths, idxs, device, bs=16, workers=2):
    bb, img, mean, std, npfx = load_member(m, spec, device)
    if spec["kind"] == "fgts":
        idx = torch.tensor(idxs[m])
    else:
        idx = torch.arange((img // bb.patch_embed.patch_size[0]) ** 2)
    ld = DataLoader(FlatDir(paths, img, mean, std), batch_size=bs,
                    num_workers=workers, pin_memory=True)
    out = np.zeros((len(paths), bb.num_features), dtype=np.float32)
    done = 0
    for x, i in ld:
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            f = bb.forward_features(x.to(device))[:, npfx:, :].float()
        out[i.numpy()] = f[:, idx, :].mean(1).cpu().numpy()
        done += len(i)
        if done % 4096 < bs:
            print(f"  [{m}] {done}/{len(paths)}", flush=True)
    del bb
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def make_head(dim, sd, device):
    import torch.nn as nn
    h = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.2), nn.Linear(dim, 1))
    h.load_state_dict(sd)
    return h.eval().to(device)


@torch.no_grad()
def head_prob(h, X, device, bs=16384):
    ps = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i + bs], dtype=torch.float32, device=device)
        ps.append(torch.sigmoid(h(x).squeeze(-1)).cpu().numpy())
    return np.concatenate(ps)


def blocknorm(X, dims):
    out, i = [], 0
    for d in dims:
        B = X[:, i:i + d].astype(np.float32)
        out.append(B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8))
        i += d
    return np.concatenate(out, 1)


@torch.no_grad()
def knn_dist(Q, R, device, k, bs=1024):
    if device == "cuda":
        torch.cuda.empty_cache()
        try:
            Rt = torch.tensor(R, dtype=torch.float32, device=device)
        except torch.cuda.OutOfMemoryError:
            print("  knn: GPU OOM on reference matrix -> CPU fallback")
            device = "cpu"
            Rt = torch.tensor(R, dtype=torch.float32)
    else:
        Rt = torch.tensor(R, dtype=torch.float32)
    ds = []
    for i in range(0, len(Q), bs):
        q = torch.tensor(Q[i:i + bs], dtype=torch.float32, device=device)
        sim = q @ Rt.T
        top = sim.topk(k, dim=1).values.mean(1)
        ds.append(top.cpu())
    return torch.cat(ds).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/submissions/submission.csv")
    ap.add_argument("--variant", default=os.environ.get("VARIANT", "routed"),
                    choices=["routed", "plain"])
    ap.add_argument("--emit_both", action="store_true",
                    help="with --variant routed, also write <out>_plain.csv "
                         "(one feature pass, both final variants; local use)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    cfg = json.loads((SYS / "config.json").read_text(encoding="utf-8"))
    heads = torch.load(SYS / "heads.pt", map_location="cpu", weights_only=False)
    idxs = dict(np.load(SYS / "fisher_idx.npz"))
    ref = np.load(SYS / "knn_ref.npz")
    Ntr, ref_dims = ref["Ntr"], ref["dims"].tolist()

    paths = sorted(p for p in Path(args.data).iterdir()
                   if p.suffix.lower() in EXTS)
    ids = [p.stem for p in paths]
    print(f"variant={args.variant} device={device} images={len(paths)}")
    assert paths, f"no images under {args.data}"

    # base members are a prefix of pad members -> extract straight into ONE
    # array; the base view costs no extra memory (135k imgs = ~4GB fp32).
    members = (cfg["base_members"] if args.variant == "plain"
               else cfg["pad_members"])
    assert cfg["pad_members"][:len(cfg["base_members"])] == cfg["base_members"]
    dcum, dims = [0], []
    Xp = None
    for m in members:
        print(f"extracting [{m}] ...", flush=True)
        F = member_features(m, cfg["member_specs"][m], paths, idxs,
                            device, args.batch, args.workers)
        if Xp is None:
            total = heads["dims"]["pad" if args.variant == "routed" else "base"]
            Xp = np.zeros((len(paths), total), dtype=np.float32)
        Xp[:, dcum[-1]:dcum[-1] + F.shape[1]] = F
        dcum.append(dcum[-1] + F.shape[1])
        dims.append(F.shape[1])
        del F

    if device == "cuda":
        torch.cuda.empty_cache()
    Xb = Xp[:, :heads["dims"]["base"]]
    champ = make_head(heads["dims"]["base"], heads["champ"], device)
    p_plain = head_prob(champ, Xb, device)
    outputs = {"plain": p_plain}

    if args.variant == "routed":
        cap = make_head(heads["dims"]["base"], heads["cap"], device)
        pcap = head_prob(cap, Xb, device)
        base_dims = dims[:len(cfg["base_members"])]
        assert base_dims == ref_dims, f"dim mismatch {base_dims} vs {ref_dims}"
        sim = knn_dist(blocknorm(Xb, base_dims), Ntr, device, cfg["knn_k"])
        dist = 1.0 - sim / len(base_dims)
        mask = (pcap > cfg["thresholds"]["cap"]) & \
               (dist > cfg["thresholds"]["dist_floor"])
        p_routed = p_plain.copy()
        if mask.any():
            pad = make_head(heads["dims"]["pad"], heads["pad"], device)
            pp = head_prob(pad, Xp, device)
            p_routed = np.where(mask, pp, p_plain)
        print(f"routed {int(mask.sum())}/{len(p_routed)} images")
        outputs = {"routed": p_routed, "plain": p_plain}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    emit = [args.variant] + (["plain"] if args.emit_both
                             and args.variant == "routed" else [])
    for v in emit:
        path = out if v == args.variant else \
            out.with_name(out.stem + "_plain" + out.suffix)
        pd.DataFrame({"id": ids, "label": np.clip(outputs[v], 0, 1)}
                     ).to_csv(path, index=False)
        print(f"wrote {path} rows={len(ids)} variant={v}")


if __name__ == "__main__":
    main()
