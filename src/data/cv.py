"""Cross-validation splits for FREUID.

Two complementary schemes, both written to a single `folds.csv`:

  * skf_fold (0..K-1): Stratified K-fold on (type x label). Estimates the
    in-distribution / public-LB-correlated performance.

  * type_loo (str): the document `type` itself. Leave-One-Type-Out — train on
    the other 4 types, validate on this one. This is our primary proxy for the
    private set, which the organizers say contains TWO unseen document types.

Near-duplicate-aware grouping is intentionally deferred: it requires the
dedup clustering from EDA. Once `dup_group` ids exist we switch the stratified
split to a StratifiedGroupKFold so near-dups never straddle folds. The hook
(`group_col`) is already wired below.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from .paths import REPO_ROOT, load_train

FOLDS_CSV = REPO_ROOT / "artifacts" / "folds.csv"


def build_folds(n_splits: int = 5, seed: int = 42,
                group_col: str | None = None) -> pd.DataFrame:
    tr = load_train().df.copy()
    # combined stratification label so every (type,label) cell is balanced
    strat = tr["type"].astype(str) + "|" + tr["label"].astype(str)

    tr["skf_fold"] = -1
    if group_col and group_col in tr.columns:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed)
        splitter = sgkf.split(tr, strat, groups=tr[group_col])
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splitter = skf.split(tr, strat)
    for fold, (_, val_idx) in enumerate(splitter):
        tr.iloc[val_idx, tr.columns.get_loc("skf_fold")] = fold

    # Leave-One-Type-Out: the held-out (validation) type for that fold == its type
    tr["type_loo"] = tr["type"].astype(str)

    return tr[["id", "type", "country", "doc_type", "label", "is_digital",
               "skf_fold", "type_loo"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group_col", default=None,
                    help="column to keep within a single fold (e.g. dup_group)")
    args = ap.parse_args()

    folds = build_folds(args.n_splits, args.seed, args.group_col)
    FOLDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLDS_CSV, index=False)

    print(f"wrote {FOLDS_CSV}  ({len(folds)} rows)")
    print("\n=== stratified folds: label balance per fold ===")
    print(pd.crosstab(folds["skf_fold"], folds["label"]))
    print("\n=== stratified folds: type balance per fold ===")
    print(pd.crosstab(folds["skf_fold"], folds["type"]))
    print("\n=== type-LOO: rows held out per type (= val set size when LOO) ===")
    print(folds["type_loo"].value_counts())
    # sanity: each fold roughly equal size and label ratio preserved
    g = folds.groupby("skf_fold")["label"].mean().round(4)
    print("\nfraud-rate per stratified fold (should be ~constant):")
    print(g.to_string())


if __name__ == "__main__":
    main()
