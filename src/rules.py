"""
VIVEKA — hard policy rules (Layer 1).

THREE deterministic checks, evaluated BEFORE the ML layer.
Design: Section 11.2, revised after the ablation (Section 21).

WHY THESE ARE NOT ML FEATURES
  A mandate breach is a FACT, not a probability. Passing it to a model
  invites the model to overrule policy, which is indefensible to a
  merchant and to a regulator.

H4 WAS REMOVED, AND WHY THAT IS PRINCIPLED
  Section 11 draws the line as: policy violations are FACTS, ML is for
  JUDGMENT. H1, H2 and H3 are facts -- the mandate is dead, or the
  amount exceeds what was ever granted. There is nothing to interpret.

  H4 (merchant outside the approved list) was never a fact of that
  kind. Authority EXISTS; only intent is unclear. It was miscategorised
  from the start, and the ablation made the cost of that visible: H4
  fired on 5.85% of all traffic -- mostly legitimate customers trying a
  new shop -- and forced a step-up regardless of any other signal. The
  layer ablation showed rules+model cost MORE than model alone, and H4
  was the reason.

  It now lives in features.py as merchant_in_mandate_scope, where the
  model weighs it against 30 other signals instead of acting alone.

  This is not the cost model overriding policy. H1/H2/H3 remain
  absolute and are NOT tuned on cost -- a PSP cannot let a model
  approve a payment on withdrawn consent at any price. H4 was always a
  judgment, and judgment belongs in the model.

ALL RULES ARE CRITICAL NOW
  Every remaining rule means no valid authority exists. The ELEVATED
  severity level is retained in the enum because fusion.py's escalation
  ordering depends on it and a future policy rule may need it.

TIME-AWARENESS
  Every rule is evaluated AS AT the event timestamp. An earlier version
  of H1 checked mandate.status alone and wrongly flagged 682 legitimate
  transactions that occurred before revocation took effect (BUGLOG).
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.context import Context, EventRow


class Severity(str, Enum):
    NONE = "none"
    ELEVATED = "elevated"
    CRITICAL = "critical"


# Ordering for "most severe wins"
_RANK = {Severity.NONE: 0, Severity.ELEVATED: 1, Severity.CRITICAL: 2}


@dataclass
class RuleResult:
    """Outcome of evaluating all hard rules for one event."""
    fired: list          # rule codes that triggered, e.g. ["H3"]
    severity: Severity
    reasons: list        # human-readable, one per fired rule

    @property
    def is_violation(self) -> bool:
        return self.severity != Severity.NONE


# ---------------------------------------------------------------
# The three rules
# ---------------------------------------------------------------

def h1_mandate_revoked(ev, mandate, ctx):
    """H1 (CRITICAL). Consent was withdrawn. No authority exists.

    Time-aware: fires only for transactions AFTER revocation took effect.
    Checking status alone would flag every transaction on that mandate,
    including ones that were entirely valid when they happened.
    """
    if mandate.revoked_at is not None and ev.timestamp > mandate.revoked_at:
        days = (ev.timestamp - mandate.revoked_at).total_seconds() / 86400
        return "H1", Severity.CRITICAL, f"Mandate revoked {days:.0f} days ago"
    return None


def h2_mandate_expired(ev, mandate, ctx):
    """H2 (CRITICAL). Consent has lapsed. No authority exists."""
    if ev.timestamp > mandate.expires_at:
        days = (ev.timestamp - mandate.expires_at).total_seconds() / 86400
        return "H2", Severity.CRITICAL, f"Mandate expired {days:.0f} days ago"
    return None


def h3_reserve_exceeded(ev, mandate, ctx):
    """H3 (CRITICAL). This transaction would exceed the monthly reserve.

    Uses consumed_before(), so only spend strictly earlier in the same
    calendar month counts -- leakage rule L5.
    """
    consumed = ctx.consumed_before(mandate.mandate_id, ev.timestamp)
    if consumed + ev.amount_paise > mandate.reserve_limit_paise:
        over = (consumed + ev.amount_paise - mandate.reserve_limit_paise) / 100
        return "H3", Severity.CRITICAL, f"Exceeds monthly reserve by Rs {over:,.0f}"
    return None


_RULES = [h1_mandate_revoked, h2_mandate_expired, h3_reserve_exceeded]

# Recorded in every audit entry so a reader knows which rules ran.
RULE_NAMES = ["H1", "H2", "H3"]


def evaluate(ev: EventRow, ctx: Context) -> RuleResult:
    """Run all rules. Most severe result wins."""
    mandate = ctx.mandates[ev.mandate_id]
    fired, reasons = [], []
    severity = Severity.NONE

    for rule in _RULES:
        out = rule(ev, mandate, ctx)
        if out is None:
            continue
        code, sev, reason = out
        fired.append(code)
        reasons.append(reason)
        if _RANK[sev] > _RANK[severity]:
            severity = sev

    return RuleResult(fired=fired, severity=severity, reasons=reasons)


def main():
    """Self-test: fire rate per rule, and per attack route."""
    import pandas as pd

    ctx = Context.load()
    labels = pd.read_csv("data/labels.csv").set_index("event_id")

    counts = {}          # rule code -> count
    fp_counts = {}       # rule code -> count on BENIGN events
    by_route = {}        # (route, severity) -> count
    n_violation = 0

    for ev in ctx.events:
        res = evaluate(ev, ctx)
        route = labels.at[ev.event_id, "attack_route"]
        route = "benign" if pd.isna(route) else route

        for code in res.fired:
            counts[code] = counts.get(code, 0) + 1
            if route == "benign":
                fp_counts[code] = fp_counts.get(code, 0) + 1
        if res.is_violation:
            n_violation += 1
        by_route[(route, res.severity.value)] = \
            by_route.get((route, res.severity.value), 0) + 1

    total = len(ctx.events)
    print(f"events evaluated : {total:,}")
    print(f"any violation    : {n_violation:,} ({n_violation / total:.2%})")
    print(f"  (was 6.49% with H4 -- most of that was H4 firing on")
    print(f"   legitimate customers trying a new merchant)")

    print("\nfire rate per rule (and how many were on benign traffic):")
    for code in RULE_NAMES:
        n = counts.get(code, 0)
        fp = fp_counts.get(code, 0)
        print(f"  {code}: {n:6,}  ({n / total:.2%})   on benign: {fp:5,}")

    print("\nseverity by route (row % of that route):")
    print(f"{'route':<10}{'none':>10}{'elevated':>12}{'critical':>12}")
    for route in ["benign", "A", "B", "C"]:
        row_total = sum(v for (r, _), v in by_route.items() if r == route)
        if row_total == 0:
            continue
        parts = []
        for sev in ["none", "elevated", "critical"]:
            n = by_route.get((route, sev), 0)
            parts.append(f"{n / row_total:>11.1%}")
        print(f"{route:<10}{''.join(parts)}")
    print("\n  Routes A and C should now be ~100% 'none': the rules no")
    print("  longer catch them. Whether they are detected is now entirely")
    print("  the model's job -- which is what makes the recall numbers")
    print("  a genuine measure of the model.")


if __name__ == "__main__":
    main()