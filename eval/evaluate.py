"""
VIVEKA — holdout evaluation (Sections 21, 22, 14.4).

THE HOLDOUT SET IS USED ONCE. Everything -- model, thresholds, feature
reference -- was frozen before this ran. If a number disappoints, that
is the number reported (Section 11.8 step 9).

BASE RATE WARNING
  Route C is holdout-only by design (Section 9.10), which raises the
  holdout base rate to ~3.7% against validation's 1.3%. PR-AUC rises
  MECHANICALLY with base rate. A higher holdout PR-AUC would therefore
  NOT mean the model improved. Metrics are reported twice -- with and
  without Route C -- so the comparison to validation is honest.

RULE-CAUGHT vs MODEL-CAUGHT
  Per-route recall on the full decision path MIXES two layers. Routes A
  and C always use an out-of-scope merchant, so hard rule H4 fires on
  every one of them and they show 100% recall regardless of what the
  model thought. That is policy doing policy's work, not evidence the
  model generalises.
  Section 4 below isolates the MODEL by scoring with rules switched off.
  Route B is the only route that is policy-compliant by design, so it is
  the only route where the headline recall already reflects the model.

EVASION TEST (Section 14.4)
  Measures recall against an attacker who knows the thresholds and
  stays under them. Reporting the degradation is more credible than
  reporting only the headline number.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)

from src import rules as rules_mod
from src.context import Context
from src.features import FEATURE_NAMES
from src.fusion import Action, decide, load_thresholds
from src.model import build_matrix, expected_calibration_error

OUT = Path("eval/results_holdout.json")


# ---------------------------------------------------------------
# metrics
# ---------------------------------------------------------------

def score_metrics(y, probs, label):
    """Model-level metrics. PR-AUC is the one that matters."""
    pr = average_precision_score(y, probs)
    roc = roc_auc_score(y, probs)
    brier = brier_score_loss(y, probs)
    ece = expected_calibration_error(y, probs)
    print(f"\n  {label}")
    print(f"    events     : {len(y):,}   attacks: {int(y.sum()):,} "
          f"({y.mean():.2%} base rate)")
    print(f"    PR-AUC     : {pr:.4f}")
    print(f"    ROC-AUC    : {roc:.4f}   (inflated under imbalance)")
    print(f"    Brier      : {brier:.4f}")
    print(f"    ECE        : {ece:.4f}")
    return {"n": int(len(y)), "n_attacks": int(y.sum()),
            "base_rate": float(y.mean()), "pr_auc": float(pr),
            "roc_auc": float(roc), "brier": float(brier), "ece": float(ece)}


def decision_metrics(y, actions, routes, variants):
    """System-level metrics: what the deployed decision path does."""
    actions = np.asarray(actions)
    flagged = actions != Action.ALLOW          # step_up or block
    blocked = actions == Action.BLOCK

    tp = int((flagged & (y == 1)).sum())
    fn = int((~flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    tn = int((~flagged & (y == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    n_benign = max(int((y == 0).sum()), 1)
    print(f"\n  decision path (flagged = step_up or block)")
    print(f"    precision  : {precision:.4f}   of what I flagged, "
          f"how much was real")
    print(f"    recall     : {recall:.4f}   of the fraud, how much I caught")
    print(f"    F1         : {f1:.4f}")
    print(f"    confusion  : TP {tp}  FP {fp}  FN {fn}  TN {tn:,}")
    print(f"    benign stepped up : "
          f"{int((flagged & ~blocked & (y == 0)).sum()):,} "
          f"({(flagged & ~blocked & (y == 0)).sum() / n_benign:.2%})")
    print(f"    benign blocked    : {int((blocked & (y == 0)).sum()):,} "
          f"({(blocked & (y == 0)).sum() / n_benign:.2%})")
    print(f"    NOTE: {int((flagged & ~blocked & (y == 0)).sum())} of "
          f"{fp} false positives are QUESTIONS (step-up), not declines.")

    print(f"\n  recall by route (FULL decision path -- rules + model):")
    per_route = {}
    for r in ["A", "B", "C"]:
        m = routes == r
        if m.sum() == 0:
            continue
        rec = flagged[m].mean()
        per_route[r] = float(rec)
        note = "  <- H4 fires on all of these" if r in ("A", "C") else ""
        print(f"    route {r} : {int(flagged[m].sum()):4d}/{int(m.sum()):4d} "
              f"= {rec:.1%}{note}")

    print(f"\n  recall by variant:")
    per_variant = {}
    for v in sorted(set(variants) - {"none"}):
        m = variants == v
        if m.sum() == 0:
            continue
        rec = flagged[m].mean()
        per_variant[v] = float(rec)
        print(f"    {v:<18}: {int(flagged[m].sum()):4d}/{int(m.sum()):4d} "
              f"= {rec:.1%}")

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall_by_route": per_route, "recall_by_variant": per_variant}


def model_only_metrics(y, probs, routes, variants, t_step):
    """Recall from the MODEL SCORE ALONE, with hard rules switched off.

    This is the honest generalisation measurement. On the full decision
    path, Routes A and C show 100% because H4 catches every out-of-scope
    merchant -- which says nothing about the model. Here a transaction
    is flagged only if the model scored it above the step-up threshold.

    No tuning happens: these are the same frozen predictions, re-reported
    with the rule layer removed.
    """
    flagged = probs >= t_step
    print(f"\n  recall from MODEL SCORE ONLY (score >= {t_step:.2f}, "
          f"rules ignored):")
    out = {}
    for r in ["A", "B", "C"]:
        m = routes == r
        if m.sum() == 0:
            continue
        rec = flagged[m].mean()
        out[r] = float(rec)
        print(f"    route {r} : {int(flagged[m].sum()):4d}/{int(m.sum()):4d} "
              f"= {rec:.1%}")
    print(f"    benign  : {int(flagged[y == 0].sum()):4d}/"
          f"{int((y == 0).sum()):4d} = {flagged[y == 0].mean():.1%} "
          f"(model-only false positive rate)")

    for v in sorted(set(variants) - {"none"}):
        m = variants == v
        if m.sum() == 0:
            continue
        out[f"variant_{v}"] = float(flagged[m].mean())
    return out


# ---------------------------------------------------------------
# evasion (Section 14.4)
# ---------------------------------------------------------------

def evasion_test(df_hold, model, thresholds, probs):
    """Recall against an attacker who knows the thresholds.

    Method: take Route B burst_drain events and shift the features an
    attacker CONTROLS -- velocity and amount concentration -- down by
    25%, simulating a slower, smaller drain that stays under the
    detector while still stealing.

    WHAT THIS DOES NOT DO: it perturbs features, not the underlying
    transactions. A real evasive attacker would regenerate the whole
    burst. This is an approximation and is labelled as one.
    """
    controllable = ["txn_count_1h", "txn_count_24h", "velocity_ratio_1h",
                    "velocity_ratio_24h", "amount_sum_1h_ratio",
                    "amount_vs_median_ratio", "amount_log_zscore"]

    mask = (df_hold["variant"] == "burst_drain").values
    if mask.sum() == 0:
        return None

    X_ev = df_hold[FEATURE_NAMES].values.copy()
    for c in controllable:
        i = FEATURE_NAMES.index(c)
        col = X_ev[mask, i]
        X_ev[mask, i] = np.where(np.isnan(col), col, col * 0.75)

    p_ev = model.predict_proba(X_ev)[:, 1]

    t1 = thresholds["t_step"]
    base_flag = probs[mask] >= t1
    ev_flag = p_ev[mask] >= t1

    print(f"\n--- evasion test (Section 14.4) ---")
    print(f"  target        : {int(mask.sum())} burst_drain events")
    print(f"  attacker shift: attacker-controlled features x0.75")
    print(f"  naive recall  : {base_flag.mean():.1%}")
    print(f"  evasive recall: {ev_flag.mean():.1%}")
    print(f"  degradation   : {base_flag.mean() - ev_flag.mean():+.1%}")
    print(f"  NOTE: perturbs features, not transactions. An approximation.")

    return {"n": int(mask.sum()), "naive_recall": float(base_flag.mean()),
            "evasive_recall": float(ev_flag.mean()),
            "degradation": float(base_flag.mean() - ev_flag.mean())}


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------

def main():
    print("=" * 66)
    print("HOLDOUT EVALUATION -- this split is used ONCE")
    print("=" * 66)

    ctx = Context.load()
    df = build_matrix(ctx)
    hold = df[df.split == "holdout"].copy().reset_index(drop=True)

    with open("models/model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    thresholds = load_thresholds()

    print(f"\nmodel      : {bundle['chosen']} (seed {bundle['seed']})")
    print(f"thresholds : t1={thresholds['t_step']:.2f} "
          f"t2={thresholds['t_block']:.2f} [{thresholds['source']}]")
    print(f"frozen     : model, thresholds, features, reference")

    hold_events = [e for e in ctx.events
                   if ctx.splits[e.principal_id] == "holdout"]
    assert len(hold_events) == len(hold), "event/row misalignment"

    probs = model.predict_proba(hold[FEATURE_NAMES].values)[:, 1]
    y = hold["y"].values
    routes = hold["route"].values
    variants = hold["variant"].values

    print("\nevaluating hard rules...")
    rule_results = [rules_mod.evaluate(e, ctx) for e in hold_events]

    hist = np.nan_to_num(hold["principal_history_days"].values, nan=0.0)
    actions = [decide(rr, float(p), float(h), thresholds).recommended_action
               for rr, p, h in zip(rule_results, probs, hist)]

    results = {"model": bundle["chosen"], "thresholds": thresholds}

    # ---- 1. with Route C ----
    print("\n" + "-" * 66)
    print("1. WITH Route C (the real holdout, base rate inflated by design)")
    print("-" * 66)
    results["with_route_c"] = score_metrics(y, probs, "model metrics")
    results["with_route_c"].update(
        decision_metrics(y, actions, routes, variants))

    # ---- 2. without Route C: comparable to validation ----
    keep = routes != "C"
    print("\n" + "-" * 66)
    print("2. WITHOUT Route C (comparable to validation's base rate)")
    print("-" * 66)
    results["without_route_c"] = score_metrics(
        y[keep], probs[keep], "model metrics")
    results["without_route_c"].update(
        decision_metrics(y[keep], np.asarray(actions)[keep],
                         routes[keep], variants[keep]))

    # ---- 3. validation comparison ----
    with open("models/metrics_val.json") as f:
        val = json.load(f)
    print("\n" + "-" * 66)
    print("3. VALIDATION vs HOLDOUT (like for like)")
    print("-" * 66)
    print(f"  validation PR-AUC        : {val['pr_auc_gb']:.4f}")
    print(f"  holdout PR-AUC (no C)    : "
          f"{results['without_route_c']['pr_auc']:.4f}")
    print(f"  holdout PR-AUC (with C)  : "
          f"{results['with_route_c']['pr_auc']:.4f}  <- base rate inflated")
    gap = results["without_route_c"]["pr_auc"] - val["pr_auc_gb"]
    print(f"  like-for-like gap        : {gap:+.4f}")
    if gap < -0.10:
        print("  -> material drop. Some overfitting to validation.")
    elif abs(gap) <= 0.10:
        print("  -> holds up. No significant overfitting.")
    else:
        print("  -> holdout better. Likely split variance, not improvement.")

    # ---- 4. model-only recall: the real generalisation test ----
    print("\n" + "-" * 66)
    print("4. MODEL-ONLY RECALL -- what the MODEL did, rules removed")
    print("-" * 66)
    print("  Routes A and C show 100% on the full path because H4 fires on")
    print("  every out-of-scope merchant. That is the rule layer, not the")
    print("  model. These numbers isolate the model.")
    results["model_only"] = model_only_metrics(
        y, probs, routes, variants, thresholds["t_step"])

    c_model = results["model_only"].get("C")
    if c_model is not None:
        print(f"\n  ROUTE C GENERALISATION: {c_model:.1%}")
        print(f"  No feature targets Route C and it never appeared in")
        print(f"  training, so this is pure transfer from Route A features.")
        print(f"  What is NOT caught: underpricing. Route C's signature is")
        print(f"  amount_log_zscore ~ -1.9, and no feature treats a LOW")
        print(f"  amount as suspicious. That gap is by design and reported.")

    # ---- 5. evasion ----
    results["evasion"] = evasion_test(hold, model, thresholds, probs)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwritten to {OUT}")
    print("\nHOLDOUT IS NOW SPENT. No further tuning.")


if __name__ == "__main__":
    main()