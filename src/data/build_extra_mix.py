"""Build the external-data mix CSVs used by P0.5/P0.6 training.

Combines the doctype-disjoint halves of DLC-2021 (src.dlc_split) and SIDTD
clips (src.data.fetch_sidtd_clips) into the exact --extra_data / --extra_val
files used for the final members. SIDTD rows are collapsed to a single
pseudo-type ("SIDTDC") so the type-balanced sampler gives DLC its proven
share of the extra sampling mass (see the technical report, §2.1).

  python -m src.data.build_extra_mix
"""
from __future__ import annotations

import pandas as pd

from .paths import REPO_ROOT

TRAIN_DOCTYPES = {"alb_id", "esp_id", "fin_id", "lva_passport", "srb_passport"}
ART = REPO_ROOT / "artifacts"


def main():
    s = pd.read_csv(ART / "sidtd_clips_index.csv")
    dt = s["type"].str.split("/").str[1]
    s["type"] = "SIDTDC"
    s_tr, s_ho = s[dt.isin(TRAIN_DOCTYPES)], s[~dt.isin(TRAIN_DOCTYPES)]
    d_tr = pd.read_csv(ART / "dlc2021_train_index.csv")
    d_ho = pd.read_csv(ART / "dlc2021_holdout_index.csv")
    ex_tr = pd.concat([d_tr, s_tr], ignore_index=True)
    ex_ho = pd.concat([d_ho, s_ho], ignore_index=True)
    ex_tr.to_csv(ART / "extra_train_dlc_sidtd.csv", index=False)
    ex_ho.to_csv(ART / "extra_val_dlc_sidtd.csv", index=False)
    for name, d in [("extra_train", ex_tr), ("extra_val", ex_ho)]:
        print(name, len(d), d.groupby(["type", "label"]).size().to_dict())


if __name__ == "__main__":
    main()
