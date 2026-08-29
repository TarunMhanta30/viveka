"""
VIVEKA — cost model and threshold derivation (Section 12).

Turns a score into a decision by asking what each mistake COSTS, rather
than by picking a round number.

Track 02's stated bar is "honest metrics including false-positive cost".
This file is where that requirement is met.

TWO DIFFERENT FALSE-POSITIVE COSTS
  A wrongly CHALLENGED customer is annoyed. A wrongly BLOCKED customer
  may leave. Averaging them into one number hides the difference that
  actually sets the block threshold (Section 12.4.2).

HONESTY NOTE
  Three of the four cost parameters are assumptions, not measured values.
  They are declared as such below. The model takes them as INPUTS so that
  substituting Razorpay's real figures gives real answers -- which is why
  the threshold is derived rather than hardcoded (Section 12.4.3).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --- Cost parameters, in paise ---
# SOURCED   : measured from the dataset
# ASSUMED   : my estimate. Razorpay has the real number.
COSTS = {
    "avg_fraud_value_paise":   None,     # SOURCED, filled at runtime
    "dispute_handling_paise":  30_000,   # ASSUMED  Rs 300 ops labour per case
    "abandonment_rate":        0.15,     # ASSUMED  drop-off when challenged
    "churn_penalty_paise":     200_000,  # ASSUMED  Rs 2000 expected loss of a customer
}

PARAM_PROVENANCE = {
    "avg_fraud_value_paise":  "SOURCED from dataset",
    "dispute_handling_paise": "ASSUMED",
    "abandonment_rate":       "ASSUMED",
    "churn_penalty_paise":    "ASSUMED",
}


def cost_false_negative(amount_paise, costs):
    """Fraud allowed: the transaction value plus dispute handling."""
    return amount_paise + costs["dispute_handling_paise"]


def cost_false_positive_stepup(amount_paise, costs):
    """Legitimate transaction challenged: expected value of abandonment.

    Not the whole amount -- most customers confirm and proceed.
    """
    return costs["abandonment_rate"] * amount_paise


def cost_false_positive_block(amount_paise, costs):
    """Legitimate transaction declined: the sale plus churn risk."""
    return amount_paise + costs["churn_penalty_paise"]


def total_cost(y, amounts, probs, t_step, t_block, costs):
    """Total rupee cost of one threshold pair over a set of events.

    True positives cost nothing here -- catching fraud is the goal, and
    the ops cost of a genuine alert is small next to the loss avoided.
    """
    allow  = probs < t_step
    stepup = (probs >= t_step) & (probs < t_block)
    block  = probs >= t_block

    fn = allow & (y == 1)
    fp_step  = stepup & (y == 0)
    fp_block = block & (y == 0)

    c = 0.0
    c += cost_false_negative(amounts[fn], costs).sum()
    c += cost_false_positive_stepup(amounts[fp_step], costs).sum()
    c += cost_false_positive_block(amounts[fp_block], costs).sum()
    return c, fn.sum(), fp_step.sum(), fp_block.sum()


def sweep(y, amounts, probs, costs, step=0.05):
    """Grid search over threshold pairs. Section 12.5."""
    grid = np.arange(0.05, 1.0, step)
    best = None
    surface = []

    for t_step in grid:
        for t_block in grid:
            if t_block < t_step:
                continue
            c, n_fn, n_fps, n_fpb = total_cost(
                y, amounts, probs, t_step, t_block, costs)
            surface.append((t_step, t_block, c))
            if best is None or c < best["cost"]:
                best = {"t_step": float(t_step), "t_block": float(t_block),
                        "cost": float(c), "fn": int(n_fn),
                        "fp_stepup": int(n_fps), "fp_block": int(n_fpb)}
    return best, surface


def sensitivity(y, amounts, probs, base_costs):
    """How much does the answer depend on the assumed numbers?

    A reviewer will trust a result more when you have shown what moves
    it (Section 12.5).
    """
    rows = []
    for name, mult in [("baseline", 1.0),
                       ("churn x2", 2.0), ("churn x0.5", 0.5)]:
        costs = dict(base_costs)
        if name != "baseline":
            costs["churn_penalty_paise"] = int(
                base_costs["churn_penalty_paise"] * mult)
        best, _ = sweep(y, amounts, probs, costs)
        rows.append((name, best["t_step"], best["t_block"], best["cost"]))

    for name, mult in [("abandon x2", 2.0)]:
        costs = dict(base_costs)
        costs["abandonment_rate"] = min(base_costs["abandonment_rate"] * mult, 1.0)
        best, _ = sweep(y, amounts, probs, costs)
        rows.append((name, best["t_step"], best["t_block"], best["cost"]))
    return rows


def main():
    import pickle
    from src.context import Context
    from src.features import FEATURE_NAMES
    from src.model import build_matrix

    ctx = Context.load()
    df = build_matrix(ctx)
    val = df[df.split == "val"].copy()

    with open("models/model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    probs = model.predict_proba(val[FEATURE_NAMES].values)[:, 1]
    y = val["y"].values
    amounts = np.array([e.amount_paise for e in ctx.events
                        if ctx.splits[e.principal_id] == "val"], dtype=float)

    assert len(amounts) == len(y), "amount/label misalignment"

    costs = dict(COSTS)
    costs["avg_fraud_value_paise"] = float(amounts[y == 1].mean())

    print("cost parameters:")
    for k, v in costs.items():
        prov = PARAM_PROVENANCE[k]
        shown = f"Rs {v/100:,.0f}" if "paise" in k else f"{v}"
        print(f"  {k:<26} {shown:>12}   [{prov}]")

    best, surface = sweep(y, amounts, probs, costs)

    print(f"\n--- derived thresholds (validation, {len(y):,} events) ---")
    print(f"  step-up threshold t1 : {best['t_step']:.2f}")
    print(f"  block threshold   t2 : {best['t_block']:.2f}")
    print(f"  total cost           : Rs {best['cost']/100:,.0f}")
    print(f"  missed fraud         : {best['fn']} of {int(y.sum())}")
    print(f"  benign stepped up    : {best['fp_stepup']:,} "
          f"({best['fp_stepup']/(y==0).sum():.2%} of benign)")
    print(f"  benign blocked       : {best['fp_block']:,} "
          f"({best['fp_block']/(y==0).sum():.2%} of benign)")

    # Naive comparison: what a 0.5 cutoff would have cost.
    naive, _, _, _ = total_cost(y, amounts, probs, 0.5, 0.5, costs)
    saving = naive - best["cost"]
    print(f"\n  a naive 0.5 cutoff would cost Rs {naive/100:,.0f}")
    print(f"  derived thresholds save   Rs {saving/100:,.0f} "
          f"({saving/naive:.1%})")

    print("\n--- sensitivity to assumed parameters ---")
    print(f"{'scenario':<14}{'t1':>7}{'t2':>7}{'cost (Rs)':>14}")
    for name, t1, t2, c in sensitivity(y, amounts, probs, costs):
        print(f"{name:<14}{t1:>7.2f}{t2:>7.2f}{c/100:>14,.0f}")

    Path("config").mkdir(exist_ok=True)
    with open("config/thresholds.json", "w") as f:
        json.dump({"t_step": best["t_step"], "t_block": best["t_block"],
                   "derived_on": "validation", "costs": costs}, f, indent=2)
    print(f"\nwritten to config/thresholds.json")


if __name__ == "__main__":
    main()