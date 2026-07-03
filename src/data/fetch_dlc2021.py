"""Download the DLC-2021 (Document Liveness Challenge) dataset from Zenodo.

DLC-2021 = real GENUINE id-document captures (`or`, color laminated originals)
vs physical REPRODUCTIONS / RECAPTURES (`cg` grayscale copy, `cc` color copy,
`re` screen recapture). This is the trustworthy real-recapture signal we lack:
our local recapture.py SIMULATION helped the local proxy but HURT public LB, and
the in-train real recaptures are only n=20. DLC-2021 gives ~1400 real clips to
ARBITRATE whether a model keeps genuines clean while catching reproductions.

License: CC BY-SA 2.5  (attribution + share-alike; research/competition OK — cite
  Polevoy et al., "Document Liveness Challenge dataset (DLC-2021)", J.Imaging 2022).

Three Zenodo records (verified 2026-06-24):
  part1 7467028: or.zip 18.8GB (GENUINE) + cg.zip 15GB (gray copy) + dlc-2021.csv  [single zips]
  part2 6792396: re.zip + re.z01..z08  ~38GB  (screen recapture)   [SPLIT archive]
  part3 7467000: cc.zip + cc.z01..z07  ~33GB  (color copy)         [SPLIT archive]

Default download = part1 minimal {or.zip, cg.zip, dlc-2021.csv, README, license}
(~34GB, no split-archive hassle) = clean genuine-vs-copy arbiter. Add the screen
recapture (closest analog to phone-photographing a screen) with --parts re.

  python -m src.data.fetch_dlc2021                 # part1: or + cg (recommended)
  python -m src.data.fetch_dlc2021 --parts or      # genuine only (18.8GB)
  python -m src.data.fetch_dlc2021 --parts re      # + screen recapture (38GB split)
  python -m src.data.fetch_dlc2021 --list          # just print files+sizes, download nothing
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from .paths import REPO_ROOT

DEST = REPO_ROOT / "external" / "dlc2021"
RECORDS = {"or": "7467028", "cg": "7467028", "re": "6792396", "cc": "7467000"}
ALWAYS = {"dlc-2021.csv", "README.md", "license.txt"}   # tiny, grab with any part


def _record(rid):
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{rid}", timeout=60) as r:
        return json.load(r)


def _files_for(parts):
    """Map requested category parts -> {filename: (url, size)} across records."""
    want = {}
    rids = {RECORDS[p] for p in parts}
    for rid in rids:
        rec = _record(rid)
        for f in rec["files"]:
            key = f["key"]
            # a split archive of category X = X.zip + X.z01.. ; match by prefix
            cat_hit = any(key == f"{p}.zip" or key.startswith(f"{p}.z") for p in parts)
            if cat_hit or key in ALWAYS:
                want[key] = (f["links"]["self"], f["size"])
    return want


def _download(url, dst: Path, size: int):
    """Streaming download with resume (HTTP Range) + progress."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    have = dst.stat().st_size if dst.exists() else 0
    if have == size:
        print(f"  ✓ {dst.name} already complete ({size/1e9:.2f}GB)")
        return
    req = urllib.request.Request(url)
    mode = "wb"
    if 0 < have < size:
        req.add_header("Range", f"bytes={have}-")
        mode = "ab"
        print(f"  ↻ resuming {dst.name} from {have/1e9:.2f}/{size/1e9:.2f}GB")
    else:
        print(f"  ↓ {dst.name} ({size/1e9:.2f}GB)")
    done = have
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, mode) as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk); done += len(chunk)
            pct = 100 * done / size
            sys.stdout.write(f"\r    {pct:5.1f}%  {done/1e9:6.2f}/{size/1e9:.2f}GB")
            sys.stdout.flush()
    sys.stdout.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=["or", "cg"],
                    choices=["or", "cg", "re", "cc"],
                    help="categories to fetch (or=genuine, cg/cc=copy, re=screen recapture)")
    ap.add_argument("--list", action="store_true", help="print files + sizes, download nothing")
    args = ap.parse_args()

    files = _files_for(args.parts)
    total = sum(s for _, s in files.values())
    print(f"DLC-2021 parts={args.parts} -> {len(files)} files, {total/1e9:.1f}GB total")
    print(f"dest: {DEST}\n")
    for k, (_, s) in sorted(files.items()):
        print(f"  {k:18s} {s/1e9:7.2f}GB")
    if args.list:
        if any(p in ("re", "cc") for p in args.parts):
            print("\n⚠ re/cc are SPLIT archives (.zip + .z01..). After download, reassemble "
                  "with 7-Zip (it reads the .zip + volumes directly) or:\n"
                  "    zip -s 0 re.zip --out re_full.zip && unzip re_full.zip")
        return
    print()
    for k, (url, s) in sorted(files.items()):
        _download(url, DEST / k, s)
    print(f"\nDONE. Next: extract the zips into {DEST}, then "
          f"`python -m src.data.index_dlc2021`")


if __name__ == "__main__":
    main()
