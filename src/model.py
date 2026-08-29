"""
VIVEKA — model training and evaluation on validation.

Trains BOTH candidates from Section 11.3 and applies the decision rule
that was written BEFORE any results were seen:

  - GB beats LR by > 0.05 PR-AUC  -> ship GB, add SHAP for attribution
  - GB beats LR marginally        -> ship LR (exact interpretability is
                                     worth more, given DPDP's algorithmic
                                     due diligence direction)
  - LR beats GB                   -> ship LR, report the finding

Deciding the rule in advance removes the temptation to rationalise
whichever number looks best on the day.

METRIC CHOICE
  PR-AUC, not accuracy and not ROC-AUC. At a 2% base rate, predicting
  "never fraud" scores 98% accuracy and catches nothing. ROC-AUC looks
  deceptively good under heavy imbalance because true negatives dominate.

IMBALANCE
  class_weight='balanced'. NOT SMOTE: it interpolates new minority rows,
  which on already-synthetic data means synthesising from synthetic, and
  applied before splitting it leaks across the boundary (Section 11.4.2).

CALIBRATION
  Platt scaling (sigmoid). Isotonic needs far more positives than the
  ~500 we have in train and would overfit (Section 11.5.2).

WHAT THIS FILE DOES NOT DO
  It never touches the holdout split. Holdout is used ONCE, in
  evaluate.py, after everything is frozen (Section 11.8 step 9).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.context import Context
from src.features import extract_all, FEATURE_NAMES

MODEL_DIR = Path("models")
SEED = 42
GB_MARGIN = 0.05   # PR-AUC gap required for GB to win (Section 11.3.2)


def build_matrix(ctx):
    """Extract features and align with labels and splits.

    Returns a DataFrame with features, label, route, variant, split.
    """
    ids, X = extract_all(ctx)
    labels = pd.read_csv("data/labels.csv").set_index("event_id").loc[ids]

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["event_id"] = ids
    df["y"] = labels["is_attack"].astype(int).values
    df["route"] = labels["attack_route"].fillna("benign").values
    df["variant"] = labels["attack_variant"].fillna("none").values
    df["principal_id"] = [e.principal_id for e in ctx.events]
    df["split"] = [ctx.splits[p] for p in df["principal_id"]]
    return df


def make_lr():
    """Logistic regression baseline.

    Median imputation: LR cannot consume NaN. The median is the least
    assuming filler, and feature 29 (principal_history_days) lets the
    model learn when baseline-derived features are unreliable anyway.

    Scaling matters here: LR coefficients are only comparable across
    features when inputs share a scale. GB does not need it.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )),
    ])


def make_gb():
    """Gradient boosting candidate.

    HistGradientBoostingClassifier handles NaN natively -- it learns a
    default direction per split -- so no imputation, which is strictly
    better than filling with a median it would have to unlearn.
    """
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=SEED,
    )


def calibrate(estimator, X, y):
    """Platt scaling with internal cross-validation.

    cv=5 means calibration is fitted on folds the estimator did not
    train on. Fitting it on training predictions would calibrate
    against outputs the model has already memorised, producing
    confident nonsense (Section 11.5.2).
    """
    cal = CalibratedClassifierCV(estimator, method="sigmoid", cv=5)
    cal.fit(X, y)
    return cal


def expected_calibration_error(y, p, bins=10):
    """Mean gap between predicted confidence and observed frequency."""
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return ece


def evaluate_on(df_split, probs, name):
    """Metrics for one model on one split, overall and per route."""
    y = df_split["y"].values
    pr_auc = average_precision_score(y, probs)
    roc = roc_auc_score(y, probs)
    brier = brier_score_loss(y, probs)
    ece = expected_calibration_error(y, probs)

    print(f"\n=== {name} ===")
    print(f"  PR-AUC     : {pr_auc:.4f}   <- the metric that matters")
    print(f"  ROC-AUC    : {roc:.4f}   (looks good under imbalance; ignore)")
    print(f"  Brier      : {brier:.4f}")
    print(f"  ECE        : {ece:.4f}")
    print(f"  base rate  : {y.mean():.2%}")

    # Recall per route at a fixed alert budget of 5% of traffic.
    k = max(int(len(probs) * 0.05), 1)
    top_idx = np.argsort(probs)[-k:]
    flagged = np.zeros(len(probs), dtype=bool)
    flagged[top_idx] = True

    print(f"  recall at a 5% alert budget ({k:,} alerts):")
    for route in ["A", "B", "C"]:
        m = (df_split["route"] == route).values
        if m.sum() == 0:
            continue
        print(f"    route {route}: {flagged[m].sum():4d}/{m.sum():4d} "
              f"= {flagged[m].mean():.1%}")
    for variant in ["burst_drain", "stale_credential", "subtle", "blatant"]:
        m = (df_split["variant"] == variant).values
        if m.sum() == 0:
            continue
        print(f"    {variant:<17}: {flagged[m].sum():4d}/{m.sum():4d} "
              f"= {flagged[m].mean():.1%}")

    overall_fp = flagged[y == 0].sum()
    print(f"    false alarms    : {overall_fp:,} of {(y == 0).sum():,} benign "
          f"= {flagged[y == 0].mean():.2%}")

    return pr_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    t0 = time.perf_counter()
    ctx = Context.load()
    df = build_matrix(ctx)
    print(f"features extracted in {time.perf_counter() - t0:.1f}s")

    train = df[df.split == "train"]
    val = df[df.split == "val"]

    Xtr, ytr = train[FEATURE_NAMES].values, train["y"].values
    Xva, yva = val[FEATURE_NAMES].values, val["y"].values

    print(f"\ntrain: {len(train):,} events, {ytr.sum()} attacks "
          f"({ytr.mean():.2%})")
    print(f"val  : {len(val):,} events, {yva.sum()} attacks "
          f"({yva.mean():.2%})")
    print(f"holdout is NOT touched here (Section 11.8)")

    # ---- train both candidates ----
    print("\ntraining logistic regression...")
    lr = calibrate(make_lr(), Xtr, ytr)
    p_lr = lr.predict_proba(Xva)[:, 1]

    print("training gradient boosting...")
    gb = calibrate(make_gb(), Xtr, ytr)
    p_gb = gb.predict_proba(Xva)[:, 1]

    pr_lr = evaluate_on(val, p_lr, "LOGISTIC REGRESSION (validation)")
    pr_gb = evaluate_on(val, p_gb, "GRADIENT BOOSTING (validation)")

    # ---- apply the pre-written decision rule ----
    gap = pr_gb - pr_lr
    print(f"\n--- model selection (rule written before training) ---")
    print(f"  PR-AUC gap (GB - LR) : {gap:+.4f}   threshold: {GB_MARGIN}")
    if gap > GB_MARGIN:
        chosen, model, probs = "gradient_boosting", gb, p_gb
        print(f"  -> GB wins by a material margin. Ship GB, add SHAP.")
    else:
        chosen, model, probs = "logistic_regression", lr, p_lr
        if gap > 0:
            print(f"  -> GB wins only marginally. Ship LR: exact "
                  f"interpretability outweighs {gap:.4f} PR-AUC.")
        else:
            print(f"  -> LR wins outright. Ship LR and report the finding.")

    # ---- sanity check ----
    best = max(pr_lr, pr_gb)
    if best > 0.98:
        print("\n  WARNING: PR-AUC above 0.98. Assume leakage, not success.")
        print("  Re-run the Section 9.9 checklist before trusting this.")

    # ---- persist ----
    MODEL_DIR.mkdir(exist_ok=True)
    import pickle
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": FEATURE_NAMES,
                     "chosen": chosen, "seed": args.seed}, f)
    with open(MODEL_DIR / "metrics_val.json", "w") as f:
        json.dump({"pr_auc_lr": pr_lr, "pr_auc_gb": pr_gb,
                   "gap": gap, "chosen": chosen}, f, indent=2)
    print(f"\nsaved to {MODEL_DIR.resolve()}/model.pkl  (chosen: {chosen})")


if __name__ == "__main__":
    main()