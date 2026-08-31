"""
Hard rule unit tests (Section 18).

The rules are the layer with legal consequences: they enforce withdrawn,
expired and exceeded consent. A bug here means either processing a
payment you had no right to process, or blocking a customer who did
nothing wrong.

The most important test in this file is
test_h1_does_not_fire_before_revocation. That exact bug shipped once and
flagged 682 legitimate transactions (BUGLOG).
"""

from datetime import datetime, timedelta

import pytest

from src.context import (Context, EventRow, MandateRow, PrincipalRow,
                         AgentRow, MerchantRow)
from src import rules as R


def _world(mandate_kwargs=None, events=None):
    """Build a minimal Context with one principal and one mandate."""
    base = dict(
        mandate_id="M0001", principal_id="P0001", agent_id="A0001",
        merchant_scope={"MER0001"},
        reserve_limit_paise=1_000_00,          # Rs 1000
        created_at=datetime(2026, 3, 1),
        last_confirmed_at=datetime(2026, 3, 1),
        expires_at=datetime(2026, 12, 1),
        status="active",
        revoked_at=None,
        confirmation_times=[],
    )
    base.update(mandate_kwargs or {})
    m = MandateRow(**base)
    p = PrincipalRow("P0001", datetime(2026, 1, 1), "medium", "weekly", 8, 20)
    a = AgentRow("A0001", "P0001", "grocery", datetime(2026, 1, 1))
    merchants = [
        MerchantRow("MER0001", "grocery", datetime(2025, 1, 1), 0.9, "large"),
        MerchantRow("MER0009", "travel", datetime(2026, 3, 1), 0.1, "small"),
    ]
    return Context(events or [], [p], [a], [m], merchants, {"P0001": "train"})


def _ev(day, amount=100_00, merchant="MER0001", eid=None):
    return EventRow(
        event_id=eid or f"E{day}", timestamp=datetime(2026, 6, day, 12),
        principal_id="P0001", agent_id="A0001", mandate_id="M0001",
        merchant_id=merchant, amount_paise=amount,
        merchant_category="grocery", item_count=1,
        channel="agentic", instruction_source="scheduled")


# ---------------------------------------------------------------
# H1 -- revoked
# ---------------------------------------------------------------

def test_h1_fires_after_revocation():
    ctx = _world({"status": "revoked",
                  "revoked_at": datetime(2026, 5, 1)})
    res = R.evaluate(_ev(10), ctx)
    assert "H1" in res.fired
    assert res.severity == R.Severity.CRITICAL


def test_h1_does_not_fire_before_revocation():
    """THE REGRESSION TEST.

    An earlier version checked mandate.status alone, so a mandate revoked
    in July flagged every transaction back to March -- 682 legitimate
    transactions recommended for block. Consent existed when those
    payments happened.
    """
    ctx = _world({"status": "revoked",
                  "revoked_at": datetime(2026, 7, 1)})
    res = R.evaluate(_ev(10), ctx)           # 10 June, before revocation
    assert "H1" not in res.fired
    assert res.severity == R.Severity.NONE


def test_h1_silent_when_never_revoked():
    ctx = _world()
    assert "H1" not in R.evaluate(_ev(10), ctx).fired


# ---------------------------------------------------------------
# H2 -- expired
# ---------------------------------------------------------------

def test_h2_fires_after_expiry():
    ctx = _world({"expires_at": datetime(2026, 5, 1)})
    res = R.evaluate(_ev(10), ctx)
    assert "H2" in res.fired
    assert res.severity == R.Severity.CRITICAL


def test_h2_does_not_fire_before_expiry():
    ctx = _world({"expires_at": datetime(2026, 12, 1)})
    assert "H2" not in R.evaluate(_ev(10), ctx).fired


def test_h2_boundary_exact_expiry_moment():
    """At exactly the expiry instant the mandate is still valid.
    The rule uses > not >=, so equality is not a violation."""
    exp = datetime(2026, 6, 10, 12)
    ctx = _world({"expires_at": exp})
    ev = _ev(10)                             # timestamp == exp
    assert ev.timestamp == exp
    assert "H2" not in R.evaluate(ev, ctx).fired


# ---------------------------------------------------------------
# H3 -- reserve exceeded
# ---------------------------------------------------------------

def test_h3_fires_when_over_reserve():
    """Rs 1000 reserve, single Rs 1500 transaction."""
    ctx = _world(events=[])
    res = R.evaluate(_ev(10, amount=1_500_00), ctx)
    assert "H3" in res.fired
    assert res.severity == R.Severity.CRITICAL


def test_h3_counts_only_the_same_month():
    """Prior spend in MAY must not count against JUNE's reserve.

    The reserve is monthly. Carrying spend across the boundary would
    make every user breach as the year progressed."""
    may = EventRow(**{**_ev(10).__dict__,
                      "event_id": "MAY",
                      "timestamp": datetime(2026, 5, 20, 12),
                      "amount_paise": 900_00})
    june = _ev(10, amount=900_00, eid="JUN")
    ctx = _world(events=[may, june])
    assert "H3" not in R.evaluate(june, ctx).fired


def test_h3_accumulates_within_a_month():
    """Rs 600 + Rs 600 in the same month exceeds a Rs 1000 reserve."""
    first = EventRow(**{**_ev(5).__dict__,
                        "event_id": "J1", "amount_paise": 600_00})
    second = _ev(10, amount=600_00, eid="J2")
    ctx = _world(events=[first, second])
    assert "H3" in R.evaluate(second, ctx).fired


def test_h3_excludes_the_event_being_scored():
    """L5 at the rule layer: consumed_before must not include this
    transaction, or every transaction would be double-counted."""
    ev = _ev(10, amount=600_00)
    ctx = _world(events=[ev])
    # 600 prior would double to 1200 and breach; correctly it is 600.
    assert "H3" not in R.evaluate(ev, ctx).fired


# ---------------------------------------------------------------
# H4 must be GONE
# ---------------------------------------------------------------

def test_h4_no_longer_exists():
    """H4 was moved into the model as merchant_in_mandate_scope after the
    ablation (Section 21). If it reappears here, the layer split has
    regressed and the cost results no longer describe the system."""
    assert R.RULE_NAMES == ["H1", "H2", "H3"]
    assert len(R._RULES) == 3


def test_out_of_scope_merchant_no_longer_violates():
    ctx = _world()
    res = R.evaluate(_ev(10, merchant="MER0009"), ctx)
    assert res.severity == R.Severity.NONE
    assert res.fired == []


# ---------------------------------------------------------------
# severity precedence
# ---------------------------------------------------------------

def test_most_severe_wins_when_multiple_fire():
    ctx = _world({"expires_at": datetime(2026, 5, 1)})
    res = R.evaluate(_ev(10, amount=1_500_00), ctx)   # H2 and H3
    assert set(res.fired) == {"H2", "H3"}
    assert res.severity == R.Severity.CRITICAL


def test_clean_transaction_produces_no_violation():
    ctx = _world()
    res = R.evaluate(_ev(10), ctx)
    assert res.severity == R.Severity.NONE
    assert res.fired == []
    assert res.is_violation is False


def test_every_fired_rule_has_a_reason():
    """An audit record with a fired rule and no explanation is useless
    to whoever reads it six months later."""
    ctx = _world({"status": "revoked", "revoked_at": datetime(2026, 5, 1)})
    res = R.evaluate(_ev(10, amount=1_500_00), ctx)
    assert len(res.reasons) == len(res.fired)
    assert all(r and isinstance(r, str) for r in res.reasons)