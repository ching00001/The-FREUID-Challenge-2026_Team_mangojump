"""Selective fetch of SIDTD clips_cropped via HTTP Range (remote zip).

SIDTD (CC-BY-4.0, MIDV-2020-derived) clips_cropped = dewarped document crops
from video clips of PHYSICAL documents: reals/ = bona-fide content, fakes/ =
forged content (crop&replace / inpaint). This fills the one cell our training
data lacks: FRAUD CONTENT x REAL CAPTURE (DLC's cg is genuine-content reprint).

The full zip is 26.6 GB; we sample 1 frame per clip-group (median frame), cap
per doctype x label, and pull only those members (~hundreds of MB) via Range
requests. Writes external/sidtd/clips_cropped/... + artifacts/sidtd_clips_index.csv
with the standard columns (id, abspath, is_digital, label, type).

  python -m src.data.fetch_sidtd_clips --cap 60
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip

from .paths import REPO_ROOT

URL = "http://datasets.cvc.uab.es/SIDTD/clips_cropped.zip"
OUT = REPO_ROOT / "external" / "sidtd" / "clips_cropped"
IDX = REPO_ROOT / "artifacts" / "sidtd_clips_index.csv"
DOCTYPES = ("alb_id", "aze_passport", "esp_id", "est_id", "fin_id",
            "grc_passport", "lva_passport", "rus_internalpassport",
            "srb_passport", "svk_id")


def parse(name):
    """-> (doctype, label, group, frame_no) or None."""
    base = name.rsplit("/", 1)[-1]
    m = re.match(r"^(" + "|".join(DOCTYPES) + r")_(.+)_frame_(\d+)", base)
    if not m:
        return None
    dt, mid, fr = m.group(1), m.group(2), int(m.group(3))
    label = 1 if "/fakes/" in name else 0
    return dt, label, f"{dt}_{mid}", fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=60,
                    help="max clip-groups per doctype x label")
    args = ap.parse_args()

    with RemoteZip(URL) as z:
        names = [n for n in z.namelist()
                 if "/Images/" in n and n.lower().endswith(".jpg")]
        groups = defaultdict(list)
        for n in names:
            p = parse(n)
            if p:
                groups[(p[0], p[1], p[2])].append((p[3], n))
        # 1 frame per group: median frame index (mid-clip = stable pose)
        per_dl = defaultdict(list)
        for (dt, lab, g), frames in groups.items():
            frames.sort()
            per_dl[(dt, lab)].append(frames[len(frames) // 2][1])
        picks = []
        for (dt, lab), items in sorted(per_dl.items()):
            items.sort()
            step = max(1, len(items) // args.cap)
            picks += items[::step][:args.cap]
        print(f"groups={len(groups)} -> picked {len(picks)} frames "
              f"({sum('/fakes/' in p for p in picks)} fake / "
              f"{sum('/reals/' in p for p in picks)} real)")

        rows = []
        for i, n in enumerate(sorted(picks)):
            dest = OUT / n.split("Images/", 1)[1]
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(n))
            p = parse(n)
            rows.append(dict(id=f"sidtdclip_{i:05d}", abspath=str(dest),
                             is_digital=False, label=p[1], type=f"SIDTDC/{p[0]}"))
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(picks)} fetched")

    df = pd.DataFrame(rows)[["id", "abspath", "label", "is_digital", "type"]]
    df.to_csv(IDX, index=False)
    print(f"wrote {IDX} n={len(df)} "
          f"({int(df.label.sum())} fake / {int((df.label == 0).sum())} real)")
    print(df.groupby(["type", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
