"""
VIVEKA — ablation study (Section 21).

Answers: did every part of this system earn its place?

TWO KINDS OF ABLATION, ON TWO DIFFERENT SPLITS. THIS MATTERS.

  1. LAYER ablation -> HOLDOUT is fine.
     Comparing rules-only vs model-only vs both re-reports predictions
     that were already frozen. No retraining, no tuning, no new
     information extracted from holdout.

  2. FEATURE GROUP ablation -> VALIDATION only.
     Dropping a feature group requires TRAINING A NEW MODEL. Evaluating
     seven different models on holdout would be using the holdout set
     seven times, which is exactly what "used once" forbids. So every
     retrained variant is measured on validation.

  Getting this backwards is the single most common way an ablation study
  quietly invalidates a headline result.

WHY CALIBRATION IS SKIPPED IN THE FEATURE ABLATION
  PR-AUC depends only on the RANKING of scores. Platt scaling is a
  monotonic transformation, so it cannot change the ranking, so it
  cannot change PR-AUC. Skipping it makes the ablation ~5x faster and
  costs nothing. (It would matter for Brier or ECE; those are not
  reported here.)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from src import rules as rules_mod
from src.context import Context
from src.features import FEATURE_NAMES
from src.fusion import Action, decide, load_thresholds
from src.model import build_matrix, make_gb, make_lr
from eval.cost_model import (COSTS, decisions_for, total_cost)

OUT = Path("eval/results_ablation.json")

# Feature groups from Section 10.4
GROUPS = {
    "A_mandate_state": FEATURE_NAMES[0:6],
    "B_velocity": FEATURE_NAMES[6:13],
    "C_amount": FEATURE_NAMES[13:16],
    "D_temporal": FEATURE_NAMES[16:21],
    "E_merchant_rel": FEATURE_NAMES[21:26],
    "F_instruction": FEATURE_NAMES[26:29],
}


# ---------------------------------------------------------------
# 1. Layer ablation -- holdout, frozen predictions only
# ---------------------------------------------------------------

def layer_ablation(y, probs, rule_results, hist, amounts, thresholds, costs):
    """Rules only vs model only vs both, in recall and rupees.

    Legitimate on holdout: no model is retrained and no threshold is
    chosen. These are the same frozen predictions, combined differently.
    """
    t1, t2 = thresholds["t_step"], thresholds["t_block"]
    n_benign = max(int((y == 0).sum()), 1)

    configs = {}

    # both -- the deployed system
    configs["both"] = decisions_for(rule_results, probs, hist, t1, t2)

    # rules only -- thresholds above 1.0 make the score unreachable
    configs["rules_only"] = decisions_for(rule_results, probs, hist,
                                          1.01, 1.01)

    # model only -- neutralise the rule floor by passing empty verdicts
    class _NoRule:
        severity = rules_mod.Severity.NONE
        fired = []
        reasons = []
    no_rules = [_NoRule() for _ in rule_results]
    configs["model_only"] = decisions_for(no_rules, probs, hist, t1, t2)

    print(f"\n{'config':<14}{'recall':>9}{'precision':>11}"
          f"{'benign FP':>11}{'cost (Rs)':>13}")
    out = {}
    for name, actions in configs.items():
        a = np.asarray(actions)
        flagged = a != Action.ALLOW
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        fn = int((~flagged & (y == 1)).sum())
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        cost, _, _, _ = total_cost(y, amounts, a, costs)
        out[name] = {"recall": rec, "precision": prec,
                     "benign_fp_rate": fp / n_benign, "cost_paise": cost}
        print(f"{name:<14}{rec:>9.1%}{prec:>11.1%}"
              f"{fp / n_benign:>11.2%}{cost / 100:>13,.0f}")

    both = out["both"]["cost_paise"]
    ro = out["rules_only"]["cost_paise"]
    mo = out["model_only"]["cost_paise"]
    print(f"\n  model adds over rules alone : "
          f"{(ro - both) / ro:+.1%} cost reduction")
    print(f"  rules add over model alone  : "
          f"{(mo - both) / mo:+.1%} cost reduction")
    print(f"  -> both layers earn their place if both numbers are positive")
    return out


# ---------------------------------------------------------------
# 2. Feature group ablation -- VALIDATION only (retrains)
# ---------------------------------------------------------------

def feature_ablation(train, val):
    """Drop each group, retrain, measure the PR-AUC loss on validation.

    Uses the uncalibrated estimator: PR-AUC is rank-based and Platt
    scaling is monotonic, so calibration cannot change the result.
    """
    ytr, yva = train["y"].values, val["y"].values

    def fit_eval(cols):
        m = make_gb()
        m.fit(train[cols].values, ytr)
        p = m.predict_proba(val[cols].values)[:, 1]
        return average_precision_score(yva, p)

    print("\ntraining full model...")
    full = fit_eval(FEATURE_NAMES)
    print(f"  full model PR-AUC (val): {full:.4f}")

    print(f"\n{'dropped group':<20}{'PR-AUC':>10}{'loss':>10}{'n feats':>9}")
    print(f"{'(none)':<20}{full:>10.4f}{0.0:>10.4f}{len(FEATURE_NAMES):>9}")

    out = {"full": float(full), "groups": {}}
    for name, cols in GROUPS.items():
        remaining = [c for c in FEATURE_NAMES if c not in cols]
        pr = fit_eval(remaining)
        loss = full - pr
        out["groups"][name] = {"pr_auc": float(pr), "loss": float(loss),
                               "n_dropped": len(cols)}
        print(f"{name:<20}{pr:>10.4f}{loss:>10.4f}{len(remaining):>9}")

    # Also: each group ALONE.
    print(f"\n{'group alone':<20}{'PR-AUC':>10}{'n feats':>9}")
    for name, cols in GROUPS.items():
        pr = fit_eval(cols)
        out["groups"][name]["alone"] = float(pr)
        print(f"{name:<20}{pr:>10.4f}{len(cols):>9}")

    return out


# ---------------------------------------------------------------
# 3. Permutation importance -- validation
# ---------------------------------------------------------------

def perm_importance(model, val):
    """Which individual features matter?

    Permutation importance, NOT the built-in impurity importance.
    Impurity-based importance is biased toward high-cardinality features
    and will mislead you about what actually matters (Section 11.7).
    """
    X, y = val[FEATURE_NAMES].values, val["y"].values
    r = permutation_importance(
        model, X, y, n_repeats=5, random_state=42,
        scoring="average_precision", n_jobs=-1)

    order = np.argsort(r.importances_mean)[::-1]
    print(f"\n{'feature':<34}{'drop in PR-AUC':>16}{'std':>9}")
    out = {}
    for i in order:
        name = FEATURE_NAMES[i]
        mean, sd = r.importances_mean[i], r.importances_std[i]
        out[name] = {"mean": float(mean), "std": float(sd)}
        if abs(mean) > 1e-4:
            print(f"{name:<34}{mean:>16.4f}{sd:>9.4f}")
    dead = [n for n, v in out.items() if abs(v["mean"]) <= 1e-4]
    print(f"\n  {len(dead)} features with no measurable contribution:")
    for n in dead:
        print(f"    {n}")
    return out


def main():
    print("=" * 66)
    print("ABLATION STUDY")
    print("=" * 66)

    ctx = Context.load()
    df = build_matrix(ctx)
    train = df[df.split == "train"]
    val = df[df.split == "val"]
    hold = df[df.split == "holdout"].copy().reset_index(drop=True)

    with open("models/model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    thresholds = load_thresholds()

    results = {}

    # ---- 1. layer ablation, holdout ----
    print("\n" + "-" * 66)
    print("1. LAYER ABLATION (holdout -- frozen predictions, no retraining)")
    print("-" * 66)

    hold_events = [e for e in ctx.events
                   if ctx.splits[e.principal_id] == "holdout"]
    probs = model.predict_proba(hold[FEATURE_NAMES].values)[:, 1]
    y = hold["y"].values
    rule_results = [rules_mod.evaluate(e, ctx) for e in hold_events]
    hist = np.nan_to_num(hold["principal_history_days"].values, nan=0.0)
    amounts = np.array([e.amount_paise for e in hold_events], dtype=float)

    costs = dict(COSTS)
    costs["avg_fraud_value_paise"] = float(amounts[y == 1].mean())

    results["layers"] = layer_ablation(y, probs, rule_results, hist,
                                       amounts, thresholds, costs)

    # ---- 2. feature group ablation, validation ----
    print("\n" + "-" * 66)
    print("2. FEATURE GROUP ABLATION (validation -- retrains a model each)")
    print("-" * 66)
    print("  On validation, NOT holdout: each variant is a new model, and")
    print("  evaluating seven models on holdout would spend it seven times.")
    results["features"] = feature_ablation(train, val)

    # ---- 3. permutation importance, validation ----
    print("\n" + "-" * 66)
    print("3. PERMUTATION IMPORTANCE (validation)")
    print("-" * 66)
    results["permutation"] = perm_importance(model, val)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()