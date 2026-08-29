"""
VIVEKA — build benign reference statistics for factor attribution.

Factor attribution needs to know what NORMAL looks like for each
feature. Ranking by raw magnitude makes large-valued features always
win regardless of relevance (BUGLOG).

Computed on the TRAIN split, benign rows only. Using validation or
holdout would leak evaluation data into an artifact the deployed
system reads.
"""

import json
from pathlib import Path

import numpy as np

from src.context import Context
from src.features import FEATURE_NAMES
from src.model import build_matrix

OUT = Path("config/feature_reference.json")


def main():
    ctx = Context.load()
    df = build_matrix(ctx)
    benign_train = df[(df.split == "train") & (df.y == 0)]
    print(f"reference from {len(benign_train):,} benign TRAIN rows")

    ref = {}
    for name in FEATURE_NAMES:
        col = benign_train[name].values
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        q1, med, q3 = np.percentile(col, [25, 50, 75])
        ref[name] = {"median": float(med), "iqr": float(q3 - q1),
                     "n": int(len(col))}

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(ref, f, indent=2)

    print(f"\n{'feature':<34}{'median':>12}{'iqr':>12}")
    for name, r in ref.items():
        print(f"{name:<34}{r['median']:>12.2f}{r['iqr']:>12.2f}")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()