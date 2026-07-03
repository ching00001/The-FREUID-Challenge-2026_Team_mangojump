"""Face-based fraud cues via InsightFace (buffalo_l, runs on onnxruntime).

A specialized HIGH-PRECISION head to complement the global DINOv3-DoRA logit on
the documented hard tail. Train characterization (src probes, 2026-06-18):

  portrait<->ghost identity (cosine of the two highest-det faces' embeddings):
    sim<0.2  =>  genuine 0.0% FP,  fraud 23.5%   (clean, near-100% precision)
  coverage (>=2 faces): EGYPT 97%, MAURITIUS 13%, others ~0%.

The operating-point metric (APCER@1%BPCER) punishes ANY genuine false-positive
catastrophically, so only a near-zero-FP head is safe to fuse. "portrait != ghost"
is exactly that: when the photo and the ghost image are different people it is
almost certainly a swapped-photo forgery, and genuine docs never trigger it.

Also records the primary face's predicted gender (for an optional, OCR-dependent
gender-field head later — noisier, deferred).

Output: artifacts/face_<split>.csv  with columns
  id, nfaces, det1, det2, ghost_sim, prim_gender, prim_bbox

Usage:
  python -m src.face_consistency --split train
  python -m src.face_consistency --split test
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pandas as pd

from .data.paths import REPO_ROOT, load_test, load_train

ART = REPO_ROOT / "artifacts"
cv2.setNumThreads(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--det_size", type=int, default=640)
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                       allowed_modules=["detection", "genderage", "recognition"])
    app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))

    sp = load_train() if args.split == "train" else load_test()
    df = sp.df
    if args.subset > 0:
        df = df.iloc[:args.subset].reset_index(drop=True)
    print(f"{args.split}: {len(df)} images", flush=True)

    rows = []
    t0 = time.time()
    for i, r in enumerate(df.itertuples()):
        img = cv2.imread(r.abspath)
        rec = {"id": str(r.id), "nfaces": 0, "det1": 0.0, "det2": 0.0,
               "ghost_sim": np.nan, "prim_gender": "", "prim_bbox": ""}
        if img is not None:
            faces = app.get(img)
            rec["nfaces"] = len(faces)
            if faces:
                fs = sorted(faces, key=lambda x: x.det_score, reverse=True)
                f0 = fs[0]
                rec["det1"] = round(float(f0.det_score), 3)
                rec["prim_gender"] = "M" if f0.sex == "M" else "F"
                rec["prim_bbox"] = ",".join(map(str, f0.bbox.astype(int).tolist()))
                if len(fs) >= 2:
                    rec["det2"] = round(float(fs[1].det_score), 3)
                    rec["ghost_sim"] = round(float(np.dot(
                        f0.normed_embedding, fs[1].normed_embedding)), 4)
        rows.append(rec)
        if (i + 1) % 1000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(df)}  {rate:.1f} img/s  "
                  f"ETA {(len(df)-i-1)/rate/60:.1f} min", flush=True)

    out = ART / f"face_{args.split}.csv"
    ART.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    n2 = int((pd.DataFrame(rows)["nfaces"] >= 2).sum())
    print(f"wrote {out} ({len(rows)} rows; {n2} with >=2 faces) "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
