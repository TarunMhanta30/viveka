"""
VIVEKA — adversarial red team (Section 14, expanded).

Attacks the FROZEN model. No retraining, no tuning, no threshold
changes. Findings are REPORTED, not fixed -- fixing after measurement
would make the measurement meaningless.

GROUNDED IN A PUBLISHED TAXONOMY
  OWASP Top 10 for Agentic Applications 2026 (published 9 Dec 2025,
  peer-reviewed with 100+ industry experts). Attacks below map to
  named threat IDs rather than categories I invented.

WHAT THIS IS
  Feature-space attacks: the attacker's controllable inputs are shifted
  to simulate a smarter adversary. This is standard adversarial
  evaluation and it is cheap.

WHAT THIS IS NOT
  It does not regenerate transactions. A real evasive attacker would
  restructure the whole burst, not perturb a feature vector. Every
  number here is an APPROXIMATION and is labelled as one.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src import rules as rules_mod
from src.context import Context
from src.features import FEATURE_NAMES
from src.fusion import Action, decide, load_thresholds
from src.model import build_matrix

OUT = Path("eval/results_redteam.json")


# ---------------------------------------------------------------
# OWASP mapping
# ---------------------------------------------------------------

OWASP_MAP = [
    ("T1", "Memory & Context Poisoning",
     "PARTIAL", "Median-based baselines resist gradual pull; no drift-rate "
                "limiting. Attack RT2 measures the gap."),
    ("T2", "Tool Misuse",
     "OUT OF SCOPE", "VIVEKA scores payments, not tool invocations."),
    ("T3", "Privilege Compromise",
     "COVERED", "Route B delegation abuse. Mandate state features 1-8."),
    ("T4", "Resource Overload",
     "PARTIAL", "Velocity features detect bursts; no rate limiting of "
                "VIVEKA itself."),
    ("T5", "Cascading Hallucination",
     "OUT OF SCOPE", "No LLM in the scoring path (Section 8.5)."),
    ("T6", "Intent Breaking & Goal Manipulation",
     "COVERED", "The core thesis. Intent divergence is what VIVEKA "
                "measures. Routes A and B."),
    ("T7", "Misaligned & Deceptive Behaviors",
     "PARTIAL", "Detected only where it changes spend pattern."),
    ("T8", "Repudiation & Untraceability",
     "COVERED", "Hash-chained replayable audit trail (Section 13)."),
    ("T9", "Identity Spoofing",
     "OUT OF SCOPE", "UAP's job. VIVEKA assumes identity verified "
                     "upstream (Section 4.6.2)."),
    ("T10", "Supply Chain",
     "OUT OF SCOPE", "7 pinned dependencies; not a detection concern."),
]


def print_mapping():
    print("\n" + "=" * 74)
    print("OWASP Top 10 for Agentic Applications 2026 -- coverage")
    print("=" * 74)
    covered = partial = 0
    for tid, name, status, note in OWASP_MAP:
        print(f"\n  {tid:<4} {name}")
        print(f"       status: {status}")
        print(f"       {note}")
        if status == "COVERED":
            covered += 1
        elif status == "PARTIAL":
            partial += 1
    print(f"\n  {covered} covered, {partial} partial, "
          f"{len(OWASP_MAP) - covered - partial} out of scope.")
    print("  Out-of-scope entries are boundary decisions, not gaps.")


# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------

def _idx(names):
    return [FEATURE_NAMES.index(n) for n in names]


def flag_rate(probs, t_step):
    return (probs >= t_step).mean()


# ---------------------------------------------------------------
# RT1 -- adaptive evader (OWASP T6)
# ---------------------------------------------------------------

ATTACKER_CONTROLS = [
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "velocity_ratio_1h", "velocity_ratio_24h",
    "amount_sum_1h_ratio", "amount_vs_median_ratio",
    "amount_log_zscore", "amount_vs_reserve_ratio",
    "utilisation_rate", "utilisation_velocity",
]


def rt1_adaptive_evader(X, mask, model, t_step):
    """An attacker who knows the thresholds and dials the attack down
    until it passes.

    Reports the RESIDUAL VALUE: how much of the original attack
    magnitude survives while staying undetected. A high residual means
    evasion is cheap for the attacker.
    """
    idx = _idx(ATTACKER_CONTROLS)
    print("\n" + "-" * 74)
    print("RT1  Adaptive evader  (OWASP T6: Intent Breaking)")
    print("-" * 74)
    print("  Attacker scales down every feature they control until the")
    print("  score falls below the step-up threshold.")

    base = model.predict_proba(X)[:, 1]
    print(f"\n  {'scale':>7}{'detected':>11}{'value kept':>13}")
    print(f"  {'1.00':>7}{flag_rate(base[mask], t_step):>10.1%}"
          f"{'100%':>13}")

    results = {}
    for scale in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        Xa = X.copy()
        for i in idx:
            col = Xa[mask, i]
            Xa[mask, i] = np.where(np.isnan(col), col, col * scale)
        p = model.predict_proba(Xa)[:, 1]
        det = flag_rate(p[mask], t_step)
        results[scale] = float(det)
        print(f"  {scale:>7.2f}{det:>10.1%}{scale:>12.0%}")

    # The attacker's best trade: lowest scale that still keeps most value
    evade_at = next((s for s in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
                     if results[s] < 0.5), None)
    if evade_at:
        print(f"\n  FINDING: detection falls below 50% at scale "
              f"{evade_at:.2f} --")
        print(f"  an attacker keeping {evade_at:.0%} of the attack "
              f"magnitude evades half the time.")
    else:
        print(f"\n  FINDING: detection stays above 50% even at 30% "
              f"magnitude.")
        print(f"  Evading costs the attacker more than 70% of their take.")
    return results


# ---------------------------------------------------------------
# RT2 -- memory poisoning (OWASP T1)
# ---------------------------------------------------------------

POISONABLE = [
    "velocity_ratio_1h", "velocity_ratio_24h", "amount_sum_1h_ratio",
    "amount_vs_median_ratio", "amount_log_zscore", "interarrival_zscore",
    "hour_deviation",
]


def rt2_memory_poisoning(X, mask, model, t_step):
    """A patient attacker who inflates the victim's baseline for weeks,
    then strikes.

    Simulated by shrinking every baseline-RELATIVE feature: if 'normal'
    has been taught to be larger, the same attack looks less unusual.
    """
    idx = _idx(POISONABLE)
    print("\n" + "-" * 74)
    print("RT2  Memory poisoning  (OWASP T1: Memory & Context Poisoning)")
    print("-" * 74)
    print("  Attacker slowly raises the victim's 'normal' before striking,")
    print("  so baseline-relative features read closer to typical.")

    base = model.predict_proba(X)[:, 1]
    print(f"\n  {'baseline inflated by':>22}{'detected':>11}")
    print(f"  {'0% (no poisoning)':>22}"
          f"{flag_rate(base[mask], t_step):>10.1%}")

    results = {}
    for infl in [1.5, 2.0, 3.0, 5.0]:
        Xa = X.copy()
        for i in idx:
            col = Xa[mask, i]
            Xa[mask, i] = np.where(np.isnan(col), col, col / infl)
        p = model.predict_proba(Xa)[:, 1]
        det = flag_rate(p[mask], t_step)
        results[infl] = float(det)
        print(f"  {f'{(infl-1)*100:.0f}%':>22}{det:>10.1%}")

    print("\n  NOTE: this simulates the EFFECT of a poisoned baseline on")
    print("  features. It does not simulate the weeks of drift needed to")
    print("  achieve it -- during which days_since_confirmation rises and")
    print("  the mandate ages, both of which the model also sees.")
    return results


# ---------------------------------------------------------------
# RT3 -- mimicry (OWASP T7)
# ---------------------------------------------------------------

def rt3_mimicry(X, mask, model, t_step):
    """An attacker who first observes the victim, then attacks using
    only their familiar merchants, hours and patterns.

    The drain still happens -- only the CONTEXT is made to look normal.
    """
    print("\n" + "-" * 74)
    print("RT3  Mimicry  (OWASP T7: Misaligned & Deceptive Behaviors)")
    print("-" * 74)
    print("  Attacker copies the victim's familiar context: known")
    print("  merchant, in scope, normal hours, familiar category.")

    base = model.predict_proba(X)[:, 1]
    print(f"\n  detection before mimicry : "
          f"{flag_rate(base[mask], t_step):.1%}")

    Xa = X.copy()
    disguise = {
        "merchant_in_mandate_scope": 1.0,
        "merchant_new_to_principal": 0.0,
        "category_new_to_principal": 0.0,
        "category_matches_agent_type": 1.0,
        "outside_active_hours": 0.0,
        "is_external_content": 0.0,
        "hour_deviation": 1.0,
        "txn_count_with_merchant": 10.0,
    }
    for name, val in disguise.items():
        Xa[mask, FEATURE_NAMES.index(name)] = val

    p = model.predict_proba(Xa)[:, 1]
    after = flag_rate(p[mask], t_step)
    print(f"  detection after mimicry  : {after:.1%}")
    drop = flag_rate(base[mask], t_step) - after
    print(f"  degradation              : {drop:+.1%}")
    print("\n  This is the hardest attack for VIVEKA: the drain is real")
    print("  but every contextual signal has been made to look ordinary.")
    print("  What survives are the velocity and utilisation features --")
    print("  the ones an attacker cannot fake while still stealing.")
    return {"before": float(flag_rate(base[mask], t_step)),
            "after": float(after), "degradation": float(drop)}


# ---------------------------------------------------------------
# RT4 -- threshold probing (OWASP T6)
# ---------------------------------------------------------------

def rt4_threshold_probing(X, model, t_step, y):
    """How cheaply can an attacker locate the decision boundary?

    VIVEKA does not return the raw score externally (Section 14.3.4),
    only the action. So the attacker sees a binary signal and must
    binary-search. This measures how many queries that takes.
    """
    print("\n" + "-" * 74)
    print("RT4  Threshold probing  (OWASP T6)")
    print("-" * 74)
    print("  External callers receive only the ACTION, never the score.")
    print("  An attacker must therefore binary-search the boundary.")

    benign = X[y == 0]
    if len(benign) < 10:
        return None
    template = np.nanmedian(benign, axis=0)

    vidx = FEATURE_NAMES.index("velocity_ratio_1h")
    lo, hi, queries = 0.0, 100.0, 0
    while hi - lo > 0.5 and queries < 60:
        mid = (lo + hi) / 2
        probe = template.copy()
        probe[vidx] = mid
        score = model.predict_proba(probe.reshape(1, -1))[0, 1]
        action = "flag" if score >= t_step else "allow"   # what they see
        if action == "flag":
            hi = mid
        else:
            lo = mid
        queries += 1

    print(f"\n  queries to locate the boundary : {queries}")
    print(f"  boundary found at velocity_ratio_1h ~ {hi:.1f}")
    print("\n  MITIGATION NOT IMPLEMENTED: rate limiting. With unlimited")
    print("  queries the boundary is cheap to find. Withholding the score")
    print("  raises the cost from 1 query to ~"
          f"{queries}, not to infinity.")
    return {"queries": queries, "boundary": float(hi)}


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------

def main():
    print("=" * 74)
    print("VIVEKA RED TEAM -- attacking the frozen model")
    print("=" * 74)
    print("\nNo retraining. No threshold changes. Findings are reported,")
    print("not fixed: fixing after measurement invalidates the measurement.")

    print_mapping()

    ctx = Context.load()
    df = build_matrix(ctx)
    hold = df[df.split == "holdout"].reset_index(drop=True)

    with open("models/model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    thresholds = load_thresholds()
    t_step = thresholds["t_step"]

    X = hold[FEATURE_NAMES].values
    y = hold["y"].values
    variants = hold["variant"].values
    mask = variants == "burst_drain"

    print("\n" + "=" * 74)
    print(f"TARGET: {int(mask.sum())} burst_drain events (holdout)")
    print(f"MODEL : {bundle['chosen']}   step-up threshold {t_step:.2f}")
    print("=" * 74)

    results = {"owasp_map": [
        {"id": t, "name": n, "status": s, "note": note}
        for t, n, s, note in OWASP_MAP]}

    results["rt1_adaptive_evader"] = rt1_adaptive_evader(
        X, mask, model, t_step)
    results["rt2_memory_poisoning"] = rt2_memory_poisoning(
        X, mask, model, t_step)
    results["rt3_mimicry"] = rt3_mimicry(X, mask, model, t_step)
    results["rt4_threshold_probing"] = rt4_threshold_probing(
        X, model, t_step, y)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nwritten to {OUT}")
    print("\nAll findings above are REPORTED, not remediated. Section 23.")


if __name__ == "__main__":
    main()