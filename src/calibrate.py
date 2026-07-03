"""Score calibration from OOF predictions.

NOTE: the FREUID metric is invariant to monotonic score transforms (AuDET is
rank-based; APCER@1%BPCER uses a quantile threshold). So calibration does NOT
change a single model's FREUID — its purpose here is:
  (1) produce well-formed [0,1] probabilities (submission requirement), and
  (2) put multiple models on a common probability scale so that *averaging*
      them in an ensemble is meaningful.

Fits isotonic regression (monotonic, flexible) on OOF (label vs score) and
saves the calibrator. We report FREUID before/after to confirm invariance.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .experiment import EXP_ROOT
from .metric import freuid_score


def fit_isotonic(label: np.ndarray, score: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(score, label)
    return iso


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_id whose oof.csv to use")
    args = ap.parse_args()

    d = EXP_ROOT / args.run
    oof = pd.read_csv(d / "oof.csv")
    y, s = oof["label"].values, oof["score"].values

    before = freuid_score(y, s)
    iso = fit_isotonic(y, s)
    s_cal = iso.predict(s)
    after = freuid_score(y, s_cal)

    print(f"FREUID before={before.freuid:.5f}  after={after.freuid:.5f} "
          f"(expected ~equal; metric is rank-invariant)")
    print(f"  AuDET {before.audet:.5f} | APCER@1%BPCER {before.apcer_at_1pct_bpcer:.5f}")
    print(f"  Brier before={np.mean((s-y)**2):.5f}  after={np.mean((s_cal-y)**2):.5f}")

    with (d / "calibrator.pkl").open("wb") as f:
        pickle.dump(iso, f)
    print(f"saved {d/'calibrator.pkl'}")


if __name__ == "__main__":
    main()
