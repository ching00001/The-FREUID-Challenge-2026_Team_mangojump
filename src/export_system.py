"""Freeze the final inference system into weights/ (pre-code-freeze).

Everything the Docker entrypoint needs, serialized once so that post-freeze
inference involves NO training of any kind:

  config.json    members, thresholds, variant map
  heads.pt       champ / capture / PAD linear heads (state_dicts)
  fisher_idx.npz frozen FGTS token indices per DINOv3 member
  knn_ref.npz    block-normalized digital-train reference (fp16) for kNN
  <m>.pt         DoRA adapters w/o the frozen base weights the ".m"
                 substring bug used to drag in (4.9GB -> ~120MB each)

Head-training RNG mirrors src/capture_mid.py exactly (manual_seed(0); champ ->
cap -> pad_half -> pad_full) so the frozen heads match that lineage; the FINAL
Kaggle submissions must then be re-generated through this frozen system so the
Docker output is bit-identical to what sits on the leaderboard.

  python -m src.export_system
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import torch

from .data.paths import REPO_ROOT, load_test
from . import fgts
from .fusion import MEMBERS, CACHE
from .hybrid_routed import load_eval, load_train_test
from .router_head import blocknorm, knn_dist

BASE = ["dino", "dino_hplus", "siglip512", "dino_hplus_dlc"]
PADM = BASE + ["dino_hplus_ds"]
PAD_SPLITS = ["dlc2021", "sidtdclips"]
SYS = REPO_ROOT / "weights"

ADAPTER_KEY = re.compile(r"(\.A|\.B|\.m)$|(^|\.)head\.")


def slim_adapters(run_id):
    """Keep only true DoRA params (.A/.B/.m as FINAL key component) + head."""
    from .experiment import EXP_ROOT
    ck = torch.load(EXP_ROOT / run_id / "adapters.pt", map_location="cpu",
                    weights_only=False)
    out = {"args": ck["args"], "epoch": ck.get("epoch")}
    for part in ("adapters", "ema_adapters"):
        if part in ck:
            out[part] = {k: v for k, v in ck[part].items() if ADAPTER_KEY.search(k)}
    return out


def main():
    device = "cuda"
    torch.manual_seed(0)                      # == capture_mid.py lineage
    SYS.mkdir(parents=True, exist_ok=True)

    # ---- heads, mirroring capture_mid.py call order exactly -----------------
    Xtr, Xte, ytr, dims = load_train_test(BASE)
    n1, n0 = max(1, int(ytr.sum())), max(1, int((ytr == 0).sum()))
    champ = fgts.train_head(Xtr, ytr, device, epochs=150, pw=n0 / n1)

    rng = np.random.default_rng(0)
    pads5 = {s: load_eval(BASE, s) for s in PAD_SPLITS}
    pads6 = {s: load_eval(PADM, s) for s in PAD_SPLITS}
    halves = {}
    for s, (X, _) in pads5.items():
        perm = rng.permutation(len(X))
        halves[s] = (perm[:len(X) // 2], perm[len(X) // 2:])
    dig = rng.choice(len(Xtr), 20000, replace=False)
    Xcap = np.concatenate([Xtr[dig]] + [pads5[s][0][halves[s][0]] for s in PAD_SPLITS])
    ycap = np.concatenate([np.zeros(len(dig))]
                          + [np.ones(len(halves[s][0])) for s in PAD_SPLITS])
    cap = fgts.train_head(Xcap, ycap, device, epochs=150,
                          pw=max(1, int((ycap == 0).sum())) / max(1, int(ycap.sum())))
    Xp = np.concatenate([pads6[s][0][halves[s][0]] for s in PAD_SPLITS])
    yp = np.concatenate([pads6[s][1][halves[s][0]] for s in PAD_SPLITS])
    _pad_half = fgts.train_head(Xp, yp, device, epochs=150,
                                pw=max(1, int((yp == 0).sum())) / max(1, int(yp.sum())))
    Xpa = np.concatenate([pads6[s][0] for s in PAD_SPLITS])
    ypa = np.concatenate([pads6[s][1] for s in PAD_SPLITS])
    pad_full = fgts.train_head(Xpa, ypa, device, epochs=150,
                               pw=max(1, int((ypa == 0).sum())) / max(1, int(ypa.sum())))

    # ---- routing constants (frozen numbers, NOT recomputed on private) ------
    Ntr = blocknorm(Xtr, dims)
    d_te = knn_dist(blocknorm(Xte, dims), Ntr, device, n_blocks=len(dims))
    floor = float(np.percentile(d_te, 99))
    print(f"frozen distance floor (public p99) = {floor:.6f}")

    torch.save({"champ": champ.state_dict(), "cap": cap.state_dict(),
                "pad": pad_full.state_dict(),
                "dims": {"base": int(Xtr.shape[1]),
                         "pad": int(Xpa.shape[1])}}, SYS / "heads.pt")
    np.savez_compressed(SYS / "knn_ref.npz", Ntr=Ntr.astype(np.float32),
                        dims=np.array(dims))

    # ---- frozen fisher indices for fgts members ------------------------------
    idxs = {}
    for m in set(PADM):
        spec = MEMBERS[m]
        if spec["kind"] != "fgts":
            continue
        bb, img, mean, std, npfx = fgts.load_backbone(spec["run"], device)
        from .train_DINOV3L_512 import make_splits
        tr_df, _ = make_splits(42, 0)
        tr_df = tr_df[tr_df["is_digital"]].reset_index(drop=True)
        rank, _ = fgts.fisher_ranking(bb, tr_df, npfx, img, mean, std, device)
        idxs[m] = rank[: spec["k"]].sort().values.numpy()
        del bb; torch.cuda.empty_cache()
        print(f"  fisher idx frozen: {m} k={spec['k']}")
    np.savez(SYS / "fisher_idx.npz", **{k: v for k, v in idxs.items()})

    # ---- slim adapters -------------------------------------------------------
    for m in set(PADM):
        sl = slim_adapters(MEMBERS[m]["run"])
        torch.save(sl, SYS / f"{m}.pt")
        n = sum(v.numel() for v in sl.get("ema_adapters", sl["adapters"]).values())
        print(f"  adapters slimmed: {m} params={n/1e6:.1f}M")

    cfg = {"base_members": BASE, "pad_members": PADM,
           "member_specs": {m: MEMBERS[m] for m in set(PADM)},
           "thresholds": {"cap": 0.5, "dist_floor": floor},
           "knn_k": 10,
           "variants": {"plain": "champ head only",
                        "routed": "champ + (capture AND dist>floor) -> PAD"}}
    (SYS / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # ---- verify: frozen heads reproduce the priced predictions --------------
    te_ids = load_test().df["id"].astype(str).tolist()
    _, Xte6, _, _ = load_train_test(PADM)
    pc = fgts.predict(champ, Xte, device)
    mask = (fgts.predict(cap, Xte, device) > .5) & (d_te > floor)
    pt = np.where(mask, fgts.predict(pad_full, Xte6, device), pc)
    for name, p, ref in [("plain", pc, "subs/fusion_C1p_dlc5.csv"),
                         ("routed", pt, "subs/fusion_capture_mid.csv")]:
        r = (pd.read_csv(REPO_ROOT / ref, dtype={"id": str})
             .set_index("id").loc[te_ids, "label"].to_numpy())
        d = np.abs(p - r)
        print(f"[verify {name}] vs {ref}: max|diff|={d.max():.2e} "
              f"flips(>0.5)={int(((p > .5) != (r > .5)).sum())}")
        fgts.write_sub(te_ids, p, f"subs/final_{name}.csv")
    print("DONE — weights/ frozen; upload subs/final_plain.csv + "
          "subs/final_routed.csv as the definitive pair")


if __name__ == "__main__":
    main()
