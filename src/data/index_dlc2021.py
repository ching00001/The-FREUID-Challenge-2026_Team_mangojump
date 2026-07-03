"""Index the extracted DLC-2021 dataset -> artifacts/dlc2021_index.csv.

Walks external/dlc2021/ for media, derives the liveness category from the DLC
clip-name convention `<template>/<NN>.<cat><NNNN>` where cat in {or,cg,cc,re}, and
emits the same schema the rest of the pipeline consumes (matches sidtd_index.csv):

  id, abspath, label, is_digital, type

Label mapping for FRAUD detection (recapture/reproduction == fraud here):
  or (original laminated)          -> label 0  (GENUINE)
  cg/cc (print copy) , re (screen) -> label 1  (reproduction / recapture spoof)
All rows is_digital=False (every DLC clip is physically captured); type=DLC/<cat>.

Handles BOTH layouts the zips may use: pre-extracted frame images (jpg/png) and
raw video clips (mp4/mov/avi -> sample N frames via OpenCV). To keep the arbiter
balanced and fast, cap frames/clip and rows/category.

  python -m src.data.index_dlc2021                          # default: 1 frame/clip
  python -m src.data.index_dlc2021 --frames_per_clip 3 --max_per_cat 1500
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .paths import REPO_ROOT

SRC = REPO_ROOT / "external" / "dlc2021"
OUT = REPO_ROOT / "artifacts" / "dlc2021_index.csv"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv"}
CAT_RE = re.compile(r"\.(or|cg|cc|re)\d+", re.IGNORECASE)   # ".cc0001" -> cc
GENUINE = "or"


def _cat(path: Path):
    """Derive DLC category from any path token like 'alb_id/00.cc0001'."""
    m = CAT_RE.search(path.as_posix())
    if m:
        return m.group(1).lower()
    # fallback: a top-level folder named exactly or/cg/cc/re
    for part in path.parts:
        if part.lower() in RECORDS_CATS:
            return part.lower()
    return None


RECORDS_CATS = {"or", "cg", "cc", "re"}


def _frames_from_video(vp: Path, n: int):
    """Sample up to n evenly-spaced RGB frames -> saved jpgs next to the video."""
    import cv2
    cap = cv2.VideoCapture(str(vp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release(); return []
    idxs = [int(total * (i + 0.5) / n) for i in range(n)]
    out = []
    cache = vp.parent / "_frames"
    cache.mkdir(exist_ok=True)
    for j, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        fp = cache / f"{vp.stem}_f{j}.jpg"
        if not fp.exists():
            cv2.imwrite(str(fp), frame)
        out.append(fp)
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_per_clip", type=int, default=1,
                    help="frames to sample from each VIDEO clip (ignored for image frames)")
    ap.add_argument("--max_per_cat", type=int, default=0, help="0 = no cap")
    ap.add_argument("--src", default=str(SRC))
    args = ap.parse_args()
    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"not found: {src}\n  run `python -m src.data.fetch_dlc2021` and extract the zips first.")

    # collect media, grouped by (category, clip) so frame sampling is per-clip
    rows, per_cat = [], {}
    media = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT | VID_EXT
             and "_frames" not in p.parts]            # don't re-index our own dumps
    print(f"scanning {src}: {len(media)} media files")

    for p in sorted(media):
        cat = _cat(p)
        if cat is None:
            continue
        if args.max_per_cat and per_cat.get(cat, 0) >= args.max_per_cat:
            continue
        frames = [p] if p.suffix.lower() in IMG_EXT else _frames_from_video(p, args.frames_per_clip)
        for fp in frames:
            rows.append({
                "id": f"dlc_{cat}_{len(rows):06d}",
                "abspath": str(fp),
                "label": 0 if cat == GENUINE else 1,
                "is_digital": False,
                "type": f"DLC/{cat}",
            })
            per_cat[cat] = per_cat.get(cat, 0) + 1

    if not rows:
        raise SystemExit("no DLC media matched the or/cg/cc/re naming — check extraction layout.")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(df)} rows)")
    print("by type:", df["type"].value_counts().to_dict())
    print("by label:", df["label"].value_counts().to_dict(),
          "  (0=genuine `or`, 1=reproduction)")


if __name__ == "__main__":
    main()
