"""Phase-0 EDA: CSV-level stats + perceptual-hash near-duplicate detection.

Subcommands
-----------
  stats   : instant tabular stats from the label CSV (fraud rate by type,
            is_digital breakdown, etc.). No image reads.
  phash   : compute a 64-bit DCT perceptual hash for every train/test image and
            cache to artifacts/phash_{split}.npz. Heavy (reads all images once);
            resumable via the cache. Use --limit to smoke-test.
  dedup   : load cached hashes and report (a) exact-pHash collisions within
            train, (b) train<->test near-duplicates by Hamming distance. This
            is the leakage check: train rows that nearly equal public_test rows
            would make the public LB partly memorizable.

Why this matters: it tells us whether near-dups can leak across CV folds (then
we switch to group-aware splitting) and whether the public LB is trustworthy.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.paths import REPO_ROOT, load_train, load_test  # noqa: E402

ART = REPO_ROOT / "artifacts"
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# --- perceptual hash ---------------------------------------------------------
def dct_phash(path: str, hash_size: int = 8, img_size: int = 32) -> np.uint64 | None:
    """64-bit DCT pHash. Robust to mild resampling/compression -> good for the
    'recapture / re-encode' near-dup case, not just byte-identical files."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    d = cv2.dct(img.astype(np.float32))
    low = d[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])          # exclude DC term
    bits = low > med
    bits[0] = low[0] > med            # keep length 64 deterministically
    val = np.uint64(0)
    for b in bits:
        val = (val << np.uint64(1)) | np.uint64(bool(b))
    return val


def compute_hashes(split: str, limit: int | None = None) -> None:
    sp = load_train() if split == "train" else load_test()
    df = sp.df
    if limit:
        df = df.head(limit)
    ids, hashes = [], []
    t0 = time.time()
    for i, (rid, path) in enumerate(zip(df["id"].astype(str), df["abspath"])):
        h = dct_phash(path)
        if h is not None:
            ids.append(rid)
            hashes.append(h)
        if (i + 1) % 2000 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(df) - i - 1) / rate
            print(f"  {split}: {i+1}/{len(df)}  {rate:.0f} img/s  ETA {eta:.0f}s",
                  flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / f"phash_{split}.npz"
    np.savez(out, ids=np.array(ids), hashes=np.array(hashes, dtype=np.uint64))
    print(f"wrote {out}  ({len(ids)} hashes, {time.time()-t0:.0f}s)")


def _load_hashes(split: str):
    z = np.load(ART / f"phash_{split}.npz", allow_pickle=True)
    return z["ids"].astype(str), z["hashes"].astype(np.uint64)


def _hamming_to_all(query: np.uint64, ref: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(ref, query).view(np.uint8).reshape(-1, 8)
    return _POPCOUNT[x].sum(axis=1)


# --- subcommands -------------------------------------------------------------
def cmd_stats(_args) -> None:
    df = load_train().df
    print("=== fraud rate & is_digital by type ===")
    g = df.groupby("type").agg(
        n=("label", "size"),
        fraud_rate=("label", "mean"),
        n_recaptured=("is_digital", lambda s: int((~s).sum())),
    ).round(4)
    print(g.to_string())
    print("\n=== is_digital x label (recaptured=False) ===")
    print(pd.crosstab(df["is_digital"], df["label"]))
    print("\n=== the 20 recaptured (is_digital=False) rows — precious OOD signal ===")
    rc = df[~df["is_digital"]]
    print(rc[["id", "type", "label"]].to_string(index=False))
    print("\nrecaptured label balance:", dict(rc["label"].value_counts()))


def cmd_phash(args) -> None:
    compute_hashes(args.split, args.limit)


def cmd_dedup(args) -> None:
    tr_ids, tr_h = _load_hashes("train")
    print(f"train hashes: {len(tr_h)}")

    # (a) exact pHash collisions within train
    uniq, counts = np.unique(tr_h, return_counts=True)
    coll = uniq[counts > 1]
    n_coll_rows = int(counts[counts > 1].sum())
    print(f"\n[within-train] exact-pHash collision groups: {len(coll)} "
          f"covering {n_coll_rows} rows "
          f"({100*n_coll_rows/len(tr_h):.2f}% of train)")
    if len(coll):
        biggest = coll[np.argmax(counts[counts > 1])]
        members = tr_ids[tr_h == biggest][:8]
        print(f"  largest group size {counts.max()}, e.g. {list(members)}")

    # (b) train <-> test near duplicates
    try:
        te_ids, te_h = _load_hashes("test")
    except FileNotFoundError:
        print("\n[train<->test] skipped: run `phash --split test` first.")
        return
    print(f"\ntest hashes: {len(te_h)}  -> scanning train<->test Hamming distance")
    thr = args.threshold
    hits = []
    t0 = time.time()
    for j, (tid, th) in enumerate(zip(te_ids, te_h)):
        d = _hamming_to_all(th, tr_h)
        k = int(d.argmin())
        if d[k] <= thr:
            hits.append((tid, tr_ids[k], int(d[k])))
        if (j + 1) % 1000 == 0:
            print(f"  scanned {j+1}/{len(te_h)} ({(j+1)/(time.time()-t0):.0f}/s)",
                  flush=True)
    print(f"\n[train<->test] test images with a train near-dup (Hamming<= {thr}): "
          f"{len(hits)} / {len(te_h)} ({100*len(hits)/len(te_h):.2f}%)")
    for tid, trid, d in hits[:15]:
        print(f"  test {tid}  ~  train {trid}  (d={d})")
    if hits:
        out = ART / "train_test_neardups.csv"
        pd.DataFrame(hits, columns=["test_id", "train_id", "hamming"]).to_csv(
            out, index=False)
        print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats").set_defaults(func=cmd_stats)
    p = sub.add_parser("phash")
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_phash)
    p = sub.add_parser("dedup")
    p.add_argument("--threshold", type=int, default=6,
                   help="max Hamming distance to call a near-duplicate")
    p.set_defaults(func=cmd_dedup)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
