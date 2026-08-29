"""
VIVEKA — hard policy rules (Layer 1).

Four deterministic checks, evaluated BEFORE the ML layer.
Design: Section 11.2.

WHY THESE ARE NOT ML FEATURES
  A mandate breach is a fact, not a probability. Passing it to a model
  invites the model to overrule policy, which is indefensible to a
  merchant and to a regulator (Section 11.2.1).

SEVERITY
  H1/H2/H3 -> CRITICAL. No valid authority exists. Recommend block.
  H4       -> ELEVATED. Authority exists but intent is unclear. The
              principal may genuinely want a new merchant, so this is a
              question, not a decline. Recommend step-up.

NO FAKE SCORE
  A violation does NOT get a probability. A policy breach is certainty
  about policy combined with uncertainty about intent; expressing it as
  0.99 would make the audit record claim a judgment the model never
  made (Section 11.2.3). The decision gate receives the flag and the
  score separately.

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
    fired: list          # rule codes that triggered, e.g. ["H3", "H4"]
    severity: Severity
    reasons: list        # human-readable, one per fired rule

    @property
    def is_violation(self) -> bool:
        return self.severity != Severity.NONE


# ---------------------------------------------------------------
# The four rules
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


def h4_scope_violation(ev, mandate, ctx):
    """H4 (ELEVATED). Merchant outside the mandate's approved list.

    Deliberately NOT critical. The principal may genuinely want to try a
    new merchant. Blocking that is the over-blocking that destroys
    agentic commerce (Section 3.4). Ask, don't decline.
    """
    if ev.merchant_id not in mandate.merchant_scope:
        return "H4", Severity.ELEVATED, "Merchant outside approved list"
    return None


_RULES = [h1_mandate_revoked, h2_mandate_expired,
          h3_reserve_exceeded, h4_scope_violation]


def evaluate(ev: EventRow, ctx: Context) -> RuleResult:
    """Run all four rules. Most severe result wins."""
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

    print("\nfire rate per rule (and how many were on benign traffic):")
    for code in ["H1", "H2", "H3", "H4"]:
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


if __name__ == "__main__":
    main()