"""Rebuild ALL fgts-member caches from the FROZEN fisher indices.

Root cause (2026-07-09 rehearsal): fisher_ranking is recomputed per run and
drifts near the k=64 cutoff (bf16 forward), so the fusion_cache features
(built July 1-5) and the frozen system indices (July 9) disagree ->
deterministic 84-row divergence between the cache path and the Docker path.

Fix: re-extract train/test/eval-split features for every fgts member using
weights/fisher_idx.npz, overwriting fusion_cache. After this,
export_system (rerun) trains heads on features IDENTICAL to what
predict_docker computes -> the two paths agree by construction.

  python -m src.rebuild_cache_frozen          # ~10h on the 5060 Ti
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data.paths import REPO_ROOT, load_test
from . import fgts
from .fusion import MEMBERS, CACHE
from .loto_head import eval_splits
from .train_DINOV3L_512 import make_splits

FGTS_MEMBERS = ["dino", "dino_hplus", "dino_hplus_dlc", "dino_hplus_ds"]
SPLITS = None  # filled in main


def main():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    idxs = dict(np.load(REPO_ROOT / "weights/fisher_idx.npz"))

    tr_df, _ = make_splits(42, 0)
    tr_df = tr_df[tr_df["is_digital"]].reset_index(drop=True)
    te_df = load_test().df.copy()
    sdf = pd.read_csv(REPO_ROOT / "artifacts/sidtd_clips_index.csv")
    splits = eval_splits() + [("sidtdclips", sdf)]

    for m in FGTS_MEMBERS:
        spec = MEMBERS[m]
        idx = {m: torch.tensor(idxs[m])}
        bb, img, mean, std, npfx = fgts.load_backbone(spec["run"], device)
        print(f"[{m}] rebuilding train ({len(tr_df)}) ...", flush=True)
        Xtr, ytr = fgts.pooled(bb, tr_df, npfx, img, mean, std, device, idx)
        print(f"[{m}] rebuilding test ({len(te_df)}) ...", flush=True)
        Xte, _ = fgts.pooled(bb, te_df, npfx, img, mean, std, device, idx, lab=False)
        np.savez(CACHE / f"{m}.npz", Xtr=Xtr[m], Xte=Xte[m], ytr=ytr)
        print(f"[{m}] fusion_cache updated {Xtr[m].shape}", flush=True)
        for s, d in splits:
            X, _ = fgts.pooled(bb, d, npfx, img, mean, std, device, idx, lab=False)
            np.savez(CACHE / f"eval_{m}__{s}.npz", X=X[m],
                     y=d["label"].astype(float).values)
            print(f"[{m}] eval cache updated: {s} {X[m].shape}", flush=True)
        del bb
        torch.cuda.empty_cache()
    print("DONE — now rerun: python -m src.export_system")


if __name__ == "__main__":
    main()
