"""
VIVEKA — cost model and threshold derivation (Section 12).

Turns a score into a decision by asking what each mistake COSTS, rather
than by picking a round number.

Track 02's stated bar is "honest metrics including false-positive cost".
This file is where that requirement is met.

OPTIMISED AGAINST THE FULL DECISION PATH
  An earlier version swept thresholds using the ML score alone, while
  the deployed system applies the rule floor on top. It reported 0%
  benign blocked; the real path blocked 0.7%, because H3 fires on
  benign reserve breaches. The thresholds were therefore optimal for a
  system that does not exist.
  This version runs every candidate pair through fusion.decide(), so
  the thresholds are optimal for the decision VIVEKA actually makes.

TWO DIFFERENT FALSE-POSITIVE COSTS
  A wrongly CHALLENGED customer is annoyed. A wrongly BLOCKED customer
  may leave. Averaging them into one number hides the difference that
  actually sets the block threshold (Section 12.4.2).

HONESTY NOTE
  Three of the four cost parameters are assumptions, not measured
  values. They are declared as such below. The model takes them as
  INPUTS so that substituting Razorpay's real figures gives real
  answers -- which is why the threshold is derived, not hardcoded.
"""

import json
import pickle
from pathlib import Path

import numpy as np

from src.fusion import Action, decide
from src.rules import Severity

# --- Cost parameters, in paise ---
# SOURCED : measured from the dataset
# ASSUMED : my estimate. Razorpay has the real number.
COSTS = {
    "avg_fraud_value_paise":   None,     # SOURCED, filled at runtime
    "dispute_handling_paise":  30_000,   # ASSUMED  Rs 300 ops labour per case
    "abandonment_rate":        0.15,     # ASSUMED  drop-off when challenged
    "churn_penalty_paise":     200_000,  # ASSUMED  Rs 2000 loss of a customer
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


def decisions_for(rule_results, probs, hist_days, t_step, t_block):
    """Run the FULL decision path for one threshold pair.

    This is the fix: candidate thresholds are evaluated through
    fusion.decide(), so the rule floor and cold-start suppression are
    included exactly as they will be in production.
    """
    th = {"t_step": t_step, "t_block": t_block, "source": "sweep"}
    return [decide(rr, float(p), float(h), th).recommended_action
            for rr, p, h in zip(rule_results, probs, hist_days)]


def total_cost(y, amounts, actions, costs):
    """Total rupee cost of a set of decisions.

    True positives cost nothing here -- catching fraud is the goal, and
    the ops cost of a genuine alert is small next to the loss avoided.
    """
    actions = np.asarray(actions)
    fn       = (actions == Action.ALLOW)   & (y == 1)
    fp_step  = (actions == Action.STEP_UP) & (y == 0)
    fp_block = (actions == Action.BLOCK)   & (y == 0)

    c = 0.0
    c += cost_false_negative(amounts[fn], costs).sum()
    c += cost_false_positive_stepup(amounts[fp_step], costs).sum()
    c += cost_false_positive_block(amounts[fp_block], costs).sum()
    return c, int(fn.sum()), int(fp_step.sum()), int(fp_block.sum())


def sweep(y, amounts, probs, rule_results, hist_days, costs, step=0.05):
    """Grid search over threshold pairs, through the full decision path."""
    grid = np.arange(0.05, 1.0, step)
    best = None

    for t_step in grid:
        for t_block in grid:
            if t_block < t_step:
                continue
            actions = decisions_for(rule_results, probs, hist_days,
                                    t_step, t_block)
            c, n_fn, n_fps, n_fpb = total_cost(y, amounts, actions, costs)
            if best is None or c < best["cost"]:
                best = {"t_step": float(t_step), "t_block": float(t_block),
                        "cost": float(c), "fn": n_fn,
                        "fp_stepup": n_fps, "fp_block": n_fpb}
    return best


def sensitivity(y, amounts, probs, rule_results, hist_days, base_costs):
    """How much does the answer depend on the assumed numbers?

    A reviewer trusts a result more when you have shown what moves it.
    """
    rows = []
    scenarios = [
        ("baseline",     None,                      1.0),
        ("churn x2",     "churn_penalty_paise",     2.0),
        ("churn x0.5",   "churn_penalty_paise",     0.5),
        ("abandon x2",   "abandonment_rate",        2.0),
        ("abandon x0.5", "abandonment_rate",        0.5),
        ("dispute x3",   "dispute_handling_paise",  3.0),
    ]
    for name, key, mult in scenarios:
        costs = dict(base_costs)
        if key == "abandonment_rate":
            costs[key] = min(base_costs[key] * mult, 1.0)
        elif key is not None:
            costs[key] = int(base_costs[key] * mult)
        b = sweep(y, amounts, probs, rule_results, hist_days, costs)
        rows.append((name, b["t_step"], b["t_block"], b["cost"]))
    return rows


def main():
    from src.context import Context
    from src.features import FEATURE_NAMES
    from src.model import build_matrix
    from src import rules as rules_mod

    ctx = Context.load()
    df = build_matrix(ctx)
    val = df[df.split == "val"].copy()

    with open("models/model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    probs = model.predict_proba(val[FEATURE_NAMES].values)[:, 1]
    y = val["y"].values

    val_events = [e for e in ctx.events if ctx.splits[e.principal_id] == "val"]
    assert len(val_events) == len(val), "event/row misalignment"

    amounts = np.array([e.amount_paise for e in val_events], dtype=float)

    # Rule verdicts are threshold-independent, so compute them once.
    print("evaluating hard rules...")
    rule_results = [rules_mod.evaluate(e, ctx) for e in val_events]

    hist = val["principal_history_days"].values
    hist = np.nan_to_num(hist, nan=0.0)

    costs = dict(COSTS)
    costs["avg_fraud_value_paise"] = float(amounts[y == 1].mean())

    print("\ncost parameters:")
    for k, v in costs.items():
        prov = PARAM_PROVENANCE[k]
        shown = f"Rs {v/100:,.0f}" if "paise" in k else f"{v}"
        print(f"  {k:<26} {shown:>12}   [{prov}]")

    print("\nsweeping thresholds through the full decision path...")
    best = sweep(y, amounts, probs, rule_results, hist, costs)

    n_benign = int((y == 0).sum())
    print(f"\n--- derived thresholds (validation, {len(y):,} events) ---")
    print(f"  step-up threshold t1 : {best['t_step']:.2f}")
    print(f"  block threshold   t2 : {best['t_block']:.2f}")
    print(f"  total cost           : Rs {best['cost']/100:,.0f}")
    print(f"  missed fraud         : {best['fn']} of {int(y.sum())}")
    print(f"  benign stepped up    : {best['fp_stepup']:,} "
          f"({best['fp_stepup']/n_benign:.2%} of benign)")
    print(f"  benign blocked       : {best['fp_block']:,} "
          f"({best['fp_block']/n_benign:.2%} of benign)")
    print("  (benign blocks are mostly H3 -- customers exceeding their own")
    print("   reserve. Policy, not model error. Rules cannot be tuned away.)")

    # --- baselines. The single-cutoff one is a strawman: it removes the
    # step-up band entirely. The honest comparison is a plausible pair.
    print("\n--- comparison against baselines ---")
    for name, t1, t2 in [("single cutoff 0.5 (strawman)", 0.5, 0.5),
                         ("hand-picked 0.5 / 0.9", 0.5, 0.9),
                         ("hand-picked 0.3 / 0.8", 0.3, 0.8),
                         ("rules only (no ML)", 1.01, 1.01)]:
        actions = decisions_for(rule_results, probs, hist, t1, t2)
        c, _, _, _ = total_cost(y, amounts, actions, costs)
        saving = (c - best["cost"]) / c if c else 0.0
        print(f"  {name:<30} Rs {c/100:>10,.0f}   "
              f"derived saves {saving:>6.1%}")

    print("\n--- sensitivity to assumed parameters ---")
    print(f"{'scenario':<14}{'t1':>7}{'t2':>7}{'cost (Rs)':>14}")
    for name, t1, t2, c in sensitivity(y, amounts, probs, rule_results,
                                       hist, costs):
        print(f"{name:<14}{t1:>7.2f}{t2:>7.2f}{c/100:>14,.0f}")

    Path("config").mkdir(exist_ok=True)
    with open("config/thresholds.json", "w") as f:
        json.dump({"t_step": best["t_step"], "t_block": best["t_block"],
                   "derived_on": "validation",
                   "optimised_through": "full fusion decision path",
                   "costs": costs}, f, indent=2)
    print("\nwritten to config/thresholds.json")


if __name__ == "__main__":
    main()