"""Data loading + path resolution for the FREUID dataset.

Handles two known quirks of the local download:
  1. CSV `image_path` says `train/xxx.jpeg` but the real file is nested at
     `train/train/xxx.jpeg`; test images live at `public_test/public_test/`.
  2. The full `test/` split (142,818 ids) may not be fully downloaded yet — we
     resolve whatever exists and report what is missing rather than crashing.

Everything here is pure I/O + bookkeeping; no modelling.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

# --- repo / data roots -------------------------------------------------------
# src/data/paths.py -> repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("FREUID_DATA", REPO_ROOT))

TRAIN_LABELS_CSV = DATA_ROOT / "train_labels.csv"
SAMPLE_SUB_CSV = DATA_ROOT / "sample_submission.csv"

# Candidate directories where images may physically live (handles nesting).
_TRAIN_IMG_DIRS = [DATA_ROOT / "train" / "train", DATA_ROOT / "train"]
_TEST_IMG_DIRS = [
    DATA_ROOT / "public_test" / "public_test",
    DATA_ROOT / "test" / "test",
    DATA_ROOT / "test",
    DATA_ROOT / "public_test",
]


@dataclass
class Split:
    """A resolved data split: dataframe of rows that have an on-disk image."""
    df: pd.DataFrame          # rows with a resolvable image path (+ `abspath` col)
    missing: pd.DataFrame     # rows whose image is not on disk
    img_dir: Path | None      # the directory images were found in


def _resolve_dir(candidates: list[Path]) -> Path | None:
    for d in candidates:
        if d.is_dir():
            return d
    return None


def _attach_paths(df: pd.DataFrame, img_dir: Path | None) -> Split:
    """Add an `abspath` column; split rows into present / missing on disk."""
    if img_dir is None:
        df = df.copy()
        df["abspath"] = None
        return Split(df.iloc[0:0].copy(), df, None)

    def to_path(row_id: str) -> str:
        return str(img_dir / f"{row_id}.jpeg")

    df = df.copy()
    df["abspath"] = df["id"].astype(str).map(to_path)
    exists = df["abspath"].map(os.path.exists)
    return Split(df[exists].reset_index(drop=True),
                 df[~exists].reset_index(drop=True),
                 img_dir)


@lru_cache(maxsize=1)
def load_train() -> Split:
    """Load train_labels.csv and resolve image paths.

    Normalises columns: `label` int, `is_digital` bool, adds `country`/`doc_type`
    parsed from the `type` field (`<country>/<doc-type>`).
    """
    df = pd.read_csv(TRAIN_LABELS_CSV)
    df["label"] = df["label"].astype(int)
    # is_digital is stored as True/False strings or bools.
    df["is_digital"] = df["is_digital"].astype(str).str.strip().str.lower().isin(
        ["true", "1"])
    parts = df["type"].astype(str).str.split("/", n=1, expand=True)
    df["country"] = parts[0]
    df["doc_type"] = parts[1].fillna("")
    return _attach_paths(df, _resolve_dir(_TRAIN_IMG_DIRS))


@lru_cache(maxsize=1)
def load_test() -> Split:
    """Load the submission id list and resolve whatever test images exist."""
    df = pd.read_csv(SAMPLE_SUB_CSV)[["id"]].copy()
    return _attach_paths(df, _resolve_dir(_TEST_IMG_DIRS))


def summary() -> str:
    tr = load_train()
    te = load_test()
    lines = [
        "=== FREUID data summary ===",
        f"DATA_ROOT: {DATA_ROOT}",
        f"train rows (csv):      {len(tr.df) + len(tr.missing)}",
        f"  on disk:             {len(tr.df)}  ({tr.img_dir})",
        f"  missing on disk:     {len(tr.missing)}",
        f"test rows (sample_sub):{len(te.df) + len(te.missing)}",
        f"  on disk:             {len(te.df)}  ({te.img_dir})",
        f"  missing on disk:     {len(te.missing)}",
    ]
    if len(tr.df):
        lines.append(f"label balance: {dict(tr.df['label'].value_counts())}")
        lines.append(f"is_digital:    {dict(tr.df['is_digital'].value_counts())}")
        lines.append(f"types:         {dict(tr.df['type'].value_counts())}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
