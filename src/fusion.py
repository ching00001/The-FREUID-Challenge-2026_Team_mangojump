"""Feature-level fusion of diverse backbones — deeper than weight-blend.

Late weight-averaging mixes scalars with a global ratio. Here we concatenate each
backbone's pooled FEATURE VECTOR and train one joint head -> learned per-dimension,
per-sample combination. 2-way (DINOv3-FGTS ⊕ SigLIP) already beat the best weight
blend (0.00426 vs 0.00595). This generalises to N members with a feature cache so
adding a member (e.g. a forensic model) is cheap.

Members (frozen, reuse trained DoRA weights):
  dino       DINOv3-L @512  -> FGTS top-64 token pool   (1024-d)
  siglip512  SigLIP2 SO400M @512 -> global token pool   (1152-d)
  siglip378  SigLIP2 SO400M @378 -> global token pool   (1152-d)

  python -m src.fusion --members dino,siglip512,siglip378
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data.paths import REPO_ROOT, load_test
from .experiment import EXP_ROOT
from . import fgts
from .train_DINOV3L_512 import make_splits

CACHE = REPO_ROOT / "artifacts" / "fusion_cache"
MEMBERS = {
    "dino":      dict(run="20260616_134434_dinov3_l512", kind="fgts", k=64),
    "dino_hplus": dict(run="20260630_032659_dinov3_hplus_512", kind="fgts", k=64),
    "dino_hplus_dlc": dict(run="20260702_064234_dinov3_hplus_dlcmix", kind="fgts", k=64),
    "dino_hplus_ds": dict(run="20260703_053144_dinov3_hplus_dlcsidtd", kind="fgts", k=64),
    "dino_hplus_global": dict(run="20260630_032659_dinov3_hplus_512", kind="global"),
    "siglip512": dict(run="20260612_134841_siglip512_dora_full", kind="global"),
    "siglip378": dict(run="20260613_052227_siglip378_dora_full", kind="global"),
    "forensic":  dict(run="20260623_191812_forensic", kind="forensic", img=512),
    "dfn5b":     dict(run="20260625_163727_dfn5b_h378", kind="global"),
}


@torch.no_grad()
def forensic_feats(run, tr_df, te_df, device, img=512):
    """Extract Bayar-ConvNeXt pre-logits (768-d noise-forensic feature)."""
    from .train_forensic import ForensicNet, IMAGENET_MEAN, IMAGENET_STD
    from torchvision import transforms as T
    from torch.utils.data import DataLoader
    m = ForensicNet().to(device).eval()
    m.load_state_dict(torch.load(EXP_ROOT / run / "model.pt", map_location="cpu",
                                 weights_only=False)["model"])
    tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def run_df(df, lab):
        ld = DataLoader(fgts.IMG(df, img, IMAGENET_MEAN, IMAGENET_STD, lab),
                        batch_size=32, num_workers=4, pin_memory=True)
        out, Y = [], []
        for x, y in ld:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = m.net.forward_features(m.bayar(x.to(device)))
                pl = m.net.forward_head(f, pre_logits=True)
            out.append(pl.float().cpu()); Y.append(np.asarray(y, dtype=float))
        return torch.cat(out).numpy(), np.concatenate(Y)
    Xtr, ytr = run_df(tr_df, True); Xte, _ = run_df(te_df, False)
    del m; torch.cuda.empty_cache()
    return Xtr, Xte, ytr


def member_feats(name, spec, tr_df, te_df, device):
    """Return (Xtr, Xte, ytr) for one member; cache pooled features to disk."""
    cf = CACHE / f"{name}.npz"
    if cf.exists():
        d = np.load(cf)
        print(f"  [{name}] cached {d['Xtr'].shape}")
        return d["Xtr"], d["Xte"], d["ytr"]
    if spec["kind"] == "forensic":
        Xtr, Xte, ytr = forensic_feats(spec["run"], tr_df, te_df, device, spec.get("img", 512))
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(cf, Xtr=Xtr, Xte=Xte, ytr=ytr)
        print(f"  [{name}] extracted+cached {Xtr.shape}")
        return Xtr, Xte, ytr
    bb, img, mean, std, npfx = fgts.load_backbone(spec["run"], device)
    if spec["kind"] == "fgts":
        rank, _ = fgts.fisher_ranking(bb, tr_df, npfx, img, mean, std, device)
        idx = {name: rank[:spec["k"]].sort().values}
    else:                                              # global mean over all patch tokens
        T = (img // bb.patch_embed.patch_size[0]) ** 2
        idx = {name: torch.arange(T)}
    Xtr, ytr = fgts.pooled(bb, tr_df, npfx, img, mean, std, device, idx)
    Xte, _ = fgts.pooled(bb, te_df, npfx, img, mean, std, device, idx, lab=False)
    del bb; torch.cuda.empty_cache()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cf, Xtr=Xtr[name], Xte=Xte[name], ytr=ytr)
    print(f"  [{name}] extracted+cached {Xtr[name].shape}")
    return Xtr[name], Xte[name], ytr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="dino,siglip512,siglip378")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    members = args.members.split(",")
    out = args.out or f"subs/fusion_{'_'.join(members)}.csv"

    tr_df, _ = make_splits(42, 0)
    tr_df = tr_df[tr_df["is_digital"]].reset_index(drop=True)
    te = load_test().df.copy()
    ids = te["id"].astype(str).tolist()

    Ftr, Fte, ytr = [], [], None
    for m in members:
        Xtr, Xte, y = member_feats(m, MEMBERS[m], tr_df, te, device)
        Ftr.append(Xtr); Fte.append(Xte); ytr = y
    Xtr = np.concatenate(Ftr, 1); Xte = np.concatenate(Fte, 1)
    print(f"fused {members} -> {Xtr.shape[1]}d | train {len(Xtr)}")

    n1, n0 = max(1, int((ytr == 1).sum())), max(1, int((ytr == 0).sum()))
    head = fgts.train_head(Xtr, ytr, device, epochs=150, pw=n0 / n1)
    fgts.write_sub(ids, fgts.predict(head, Xte, device), out)
    print("DONE", out)


if __name__ == "__main__":
    main()
