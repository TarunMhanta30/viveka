"""
VIVEKA — decision gate (Section 12).

Combines the Layer 1 rule verdict with the Layer 2 ML score into a
single recommended action.

PRECEDENCE (Section 11.6)
  Rules can ESCALATE a decision. They can never DE-ESCALATE one.
  A low model score cannot excuse an expired mandate. Policy is a
  floor, not an input to be averaged away.

    CRITICAL violation (H1/H2/H3) -> BLOCK,   regardless of score
    ELEVATED violation (H4)       -> STEP_UP minimum, score may raise it
    no violation                  -> thresholds decide

NO FAKE SCORE FOR A VIOLATION
  A policy breach does not get a manufactured probability. It is
  certainty about policy combined with uncertainty about intent.
  The two travel separately all the way into the audit record
  (Section 11.2.3).

RECOMMEND, DO NOT BLOCK
  The output field is `recommended_action`. Under RBI's Authentication
  Directions the integrity of authentication is the ISSUER's
  responsibility. VIVEKA emits a signal; the issuer decides. Claiming
  block authority would misstate where responsibility sits (rule R7).

COLD START (Section 10.7)
  Under 7 days of history the ML score is suppressed and only rules
  apply. Baseline-derived features are unreliable on thin history, and
  a confident score built on three transactions is worse than no score.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.rules import RuleResult, Severity

THRESHOLD_FILE = Path("config/thresholds.json")

# Fallbacks if thresholds.json is missing. Real values are DERIVED by
# eval/cost_model.py; these exist so the module never silently uses a
# made-up number without saying so.
DEFAULT_T_STEP = 0.35
DEFAULT_T_BLOCK = 0.85

MIN_HISTORY_DAYS = 7.0
LOW_CONFIDENCE_DAYS = 30.0


class Action:
    ALLOW = "allow"
    STEP_UP = "step_up"
    BLOCK = "block"


_ORDER = {Action.ALLOW: 0, Action.STEP_UP: 1, Action.BLOCK: 2}


@dataclass
class Decision:
    """The complete output for one transaction."""
    recommended_action: str
    ml_score: float | None          # None when suppressed or not run
    rule_severity: str
    rules_fired: list
    reasons: list
    thresholds: dict
    score_suppressed: bool = False
    factor_codes: list = field(default_factory=list)


def load_thresholds(path: Path = THRESHOLD_FILE) -> dict:
    """Load derived thresholds, or fall back loudly.

    Thresholds live in config, not code: they change with business
    conditions, and a threshold in versioned config gets reviewed while
    one buried in Python gets changed silently (Section 12.8).
    """
    if path.exists():
        with open(path) as f:
            d = json.load(f)
        return {"t_step": d["t_step"], "t_block": d["t_block"],
                "source": "derived"}
    return {"t_step": DEFAULT_T_STEP, "t_block": DEFAULT_T_BLOCK,
            "source": "DEFAULT_NOT_DERIVED"}


def decide(rule_result: RuleResult,
           ml_score: float | None,
           history_days: float,
           thresholds: dict) -> Decision:
    """Map (rule verdict, score) to a recommended action.

    ml_score may be None: either cold start, or the caller chose not to
    run the model because a critical rule already fired.
    """
    t_step = thresholds["t_step"]
    t_block = thresholds["t_block"]
    reasons = list(rule_result.reasons)

    # --- cold start: suppress the score, rules only ---
    suppressed = False
    if history_days < MIN_HISTORY_DAYS:
        suppressed = True
        ml_score = None
        reasons.append(
            f"ML score suppressed: only {history_days:.1f} days of history")

    # --- score-based action ---
    if ml_score is None:
        score_action = Action.ALLOW
    elif ml_score >= t_block:
        score_action = Action.BLOCK
        reasons.append(f"Risk score {ml_score:.2f} at or above block "
                       f"threshold {t_block:.2f}")
    elif ml_score >= t_step:
        score_action = Action.STEP_UP
        reasons.append(f"Risk score {ml_score:.2f} at or above step-up "
                       f"threshold {t_step:.2f}")
    else:
        score_action = Action.ALLOW

    # --- rule-based floor ---
    if rule_result.severity == Severity.CRITICAL:
        rule_action = Action.BLOCK
    elif rule_result.severity == Severity.ELEVATED:
        rule_action = Action.STEP_UP
    else:
        rule_action = Action.ALLOW

    # Escalate only. max() over the ordering enforces that a low score
    # can never soften a policy violation.
    final = rule_action if _ORDER[rule_action] >= _ORDER[score_action] \
        else score_action

    if (history_days < LOW_CONFIDENCE_DAYS and not suppressed
            and ml_score is not None):
        reasons.append(
            f"Low confidence: {history_days:.1f} days of history")

    return Decision(
        recommended_action=final,
        ml_score=ml_score,
        rule_severity=rule_result.severity.value,
        rules_fired=list(rule_result.fired),
        reasons=reasons,
        thresholds={"t_step": t_step, "t_block": t_block,
                    "source": thresholds.get("source", "unknown")},
        score_suppressed=suppressed,
    )


def top_factors(feature_values: dict, k: int = 3) -> list:
    """Rank contributing features for the audit record (rule R9).

    Placeholder ranking by absolute deviation. Replaced by SHAP in
    explain.py now that gradient boosting was selected (Section 11.7).
    """
    ranked = sorted(
        ((n, v) for n, v in feature_values.items()
         if v is not None and not (isinstance(v, float) and np.isnan(v))),
        key=lambda kv: abs(kv[1]), reverse=True)
    return [n for n, _ in ranked[:k]]


def main():
    """Self-test: apply the gate across the validation split."""
    import pickle
    import pandas as pd
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

    thresholds = load_thresholds()
    print(f"thresholds: t1={thresholds['t_step']:.2f} "
          f"t2={thresholds['t_block']:.2f}  [{thresholds['source']}]")
    if thresholds["source"] != "derived":
        print("  WARNING: using defaults. Run eval/cost_model.py first.")

    probs = model.predict_proba(val[FEATURE_NAMES].values)[:, 1]
    val_events = [e for e in ctx.events if ctx.splits[e.principal_id] == "val"]
    assert len(val_events) == len(val), "event/row misalignment"

    counts = {}
    by_route = {}
    escalations = 0
    suppressed = 0

    for ev, prob, (_, row) in zip(val_events, probs, val.iterrows()):
        rr = rules_mod.evaluate(ev, ctx)
        hist = row["principal_history_days"]
        hist = 0.0 if np.isnan(hist) else float(hist)
        d = decide(rr, float(prob), hist, thresholds)

        counts[d.recommended_action] = counts.get(d.recommended_action, 0) + 1
        by_route[(row["route"], d.recommended_action)] = \
            by_route.get((row["route"], d.recommended_action), 0) + 1
        if d.score_suppressed:
            suppressed += 1
        # did the rule floor raise the action above what the score alone gave?
        score_only = (Action.BLOCK if prob >= thresholds["t_block"]
                      else Action.STEP_UP if prob >= thresholds["t_step"]
                      else Action.ALLOW)
        if _ORDER[d.recommended_action] > _ORDER[score_only]:
            escalations += 1

    total = len(val_events)
    print(f"\nevents: {total:,}")
    print("action distribution:")
    for a in [Action.ALLOW, Action.STEP_UP, Action.BLOCK]:
        n = counts.get(a, 0)
        print(f"  {a:<9}: {n:6,}  ({n / total:.2%})")
    print(f"\nrule escalations (rule raised the action): {escalations:,}")
    print(f"scores suppressed (cold start)          : {suppressed:,}")

    print("\naction by route (row % of that route):")
    print(f"{'route':<10}{'allow':>10}{'step_up':>10}{'block':>10}")
    for route in ["benign", "A", "B"]:
        rt = sum(v for (r, _), v in by_route.items() if r == route)
        if rt == 0:
            continue
        parts = [f"{by_route.get((route, a), 0) / rt:>9.1%}"
                 for a in [Action.ALLOW, Action.STEP_UP, Action.BLOCK]]
        print(f"{route:<10}{''.join(parts)}")


if __name__ == "__main__":
    main()