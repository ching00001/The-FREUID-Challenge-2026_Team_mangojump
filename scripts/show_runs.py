"""Pretty-print the experiment registry, sorted by FREUID (best first).

Usage:
    python scripts/show_runs.py            # all runs
    python scripts/show_runs.py --sort freuid --top 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import REGISTRY_CSV  # noqa: E402

COLS = ["run_id", "name", "backbone", "img", "cv", "val_fold", "loto_type",
        "epochs", "eff_bs", "lr", "loss", "aug", "ema", "tta",
        "freuid", "audet", "apcer_at_1pct_bpcer", "roc_auc", "best_epoch",
        "elapsed_min", "notes"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", default="freuid")
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args()

    if not REGISTRY_CSV.exists():
        print("no experiments yet:", REGISTRY_CSV)
        return
    df = pd.read_csv(REGISTRY_CSV)
    if args.sort in df.columns:
        df = df.sort_values(args.sort, ascending=True, na_position="last")
    if args.top:
        df = df.head(args.top)
    show = [c for c in COLS if c in df.columns]
    pd.set_option("display.max_columns", None, "display.width", 240,
                  "display.max_colwidth", 28)
    print(df[show].to_string(index=False))
    print(f"\n{len(df)} run(s).  registry: {REGISTRY_CSV}")


if __name__ == "__main__":
    main()
