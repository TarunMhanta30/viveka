"""
Feature unit tests (Section 18).

Each test uses HAND-COMPUTED expected values. A test that just calls the
function and checks it returns a number proves nothing -- it would pass
on a function that returns 42 for everything.

Focus: the features where a bug would be silent. Circular hour distance,
the leakage boundary, and missing-vs-zero.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.context import (Baseline, Context, EventRow, MandateRow,
                         PrincipalRow, AgentRow, MerchantRow,
                         circular_hour_distance, _circular_mean_hour)
from src import features as F


# ---------------------------------------------------------------
# fixtures: a tiny hand-built world
# ---------------------------------------------------------------

@pytest.fixture
def ctx():
    """Three transactions for one principal, hand-specified."""
    p = PrincipalRow("P0001", datetime(2026, 1, 1), "medium", "weekly", 8, 20)
    a = AgentRow("A0001", "P0001", "grocery", datetime(2026, 1, 1))
    m = MandateRow(
        mandate_id="M0001", principal_id="P0001", agent_id="A0001",
        merchant_scope={"MER0001", "MER0002"},
        reserve_limit_paise=1_000_00,          # Rs 1000
        created_at=datetime(2026, 3, 1),
        last_confirmed_at=datetime(2026, 3, 1),
        expires_at=datetime(2026, 12, 1),
        status="active",
        revoked_at=None,
        confirmation_times=[datetime(2026, 3, 20)],
    )
    merchants = [
        MerchantRow("MER0001", "grocery", datetime(2025, 1, 1), 0.9, "large"),
        MerchantRow("MER0002", "grocery", datetime(2025, 1, 1), 0.9, "large"),
        MerchantRow("MER0009", "travel", datetime(2026, 3, 1), 0.1, "small"),
    ]

    def ev(day, hour, amount, merchant="MER0001", cat="grocery",
           src="scheduled"):
        return EventRow(
            event_id=f"E{day}{hour}", timestamp=datetime(2026, 3, day, hour),
            principal_id="P0001", agent_id="A0001", mandate_id="M0001",
            merchant_id=merchant, amount_paise=amount,
            merchant_category=cat, item_count=1,
            channel="agentic", instruction_source=src)

    events = [
        ev(5, 10, 100_00),      # Rs 100
        ev(12, 11, 100_00),
        ev(19, 10, 100_00),
    ]
    return Context(events, [p], [a], [m], merchants,
                   {"P0001": "train"})


# ---------------------------------------------------------------
# circular hour arithmetic -- the silent-bug candidate
# ---------------------------------------------------------------

def test_circular_distance_wraps_midnight():
    """23:00 to 01:00 is 2 hours, not 22. A naive subtraction gives 22
    and the bug is invisible in aggregate statistics."""
    assert circular_hour_distance(23, 1) == 2
    assert circular_hour_distance(1, 23) == 2


def test_circular_distance_maximum_is_12():
    """Opposite sides of the clock. Distance can never exceed 12."""
    assert circular_hour_distance(0, 12) == 12
    assert circular_hour_distance(6, 18) == 12


def test_circular_distance_same_hour_is_zero():
    assert circular_hour_distance(14, 14) == 0


def test_circular_mean_handles_wrap():
    """Mean of 23:00 and 01:00 is midnight, not midday.
    A plain arithmetic mean gives 12 -- the opposite answer."""
    assert _circular_mean_hour([23, 1]) == pytest.approx(0, abs=0.01)
    assert np.mean([23, 1]) == 12          # what the WRONG answer is


# ---------------------------------------------------------------
# leakage boundary -- L5
# ---------------------------------------------------------------

def test_history_excludes_the_event_itself(ctx):
    """The transaction being scored must never be in its own history."""
    ts = datetime(2026, 3, 12, 11)          # exact time of event 2
    prior = ctx.history_before("P0001", ts)
    assert len(prior) == 1                   # only the 5 March event
    assert all(e.timestamp < ts for e in prior)


def test_history_excludes_simultaneous_events(ctx):
    """bisect_left, not bisect_right. An event at exactly ts is not
    information available when scoring ts."""
    ts = datetime(2026, 3, 5, 10)            # exact time of event 1
    assert ctx.history_before("P0001", ts) == []


def test_history_before_first_event_is_empty(ctx):
    assert ctx.history_before("P0001", datetime(2026, 3, 1)) == []


def test_confirmation_lookup_respects_boundary(ctx):
    """A confirmation on 20 March must not affect a 19 March transaction."""
    before = ctx.last_confirmation_before("M0001", datetime(2026, 3, 19))
    after = ctx.last_confirmation_before("M0001", datetime(2026, 3, 25))
    assert before == datetime(2026, 3, 1)    # falls back to created_at
    assert after == datetime(2026, 3, 20)    # sees the confirmation


# ---------------------------------------------------------------
# hand-computed feature values
# ---------------------------------------------------------------

def test_mandate_age_days_exact(ctx):
    """31 days from 1 March to 1 April."""
    ev = ctx.events[0]
    ev2 = EventRow(**{**ev.__dict__, "timestamp": datetime(2026, 4, 1)})
    m = ctx.mandates["M0001"]
    assert F.f_mandate_age_days(ev2, m, ctx, None, None) == pytest.approx(31.0)


def test_days_since_confirmation_differs_from_age(ctx):
    """After the 20 March confirmation, age is 31 days but time since
    confirmation is only 12. If these are equal, the confirmation
    lookup is broken."""
    ev = ctx.events[0]
    ev2 = EventRow(**{**ev.__dict__, "timestamp": datetime(2026, 4, 1)})
    m = ctx.mandates["M0001"]
    age = F.f_mandate_age_days(ev2, m, ctx, None, None)
    since = F.f_days_since_confirmation(ev2, m, ctx, None, None)
    assert age == pytest.approx(31.0)
    assert since == pytest.approx(12.0)
    assert age != since


def test_merchant_in_scope_binary(ctx):
    m = ctx.mandates["M0001"]
    in_scope = ctx.events[0]                 # MER0001, in scope
    assert F.f_merchant_in_mandate_scope(in_scope, m, ctx, None, None) == 1.0

    out = EventRow(**{**in_scope.__dict__, "merchant_id": "MER0009"})
    assert F.f_merchant_in_mandate_scope(out, m, ctx, None, None) == 0.0


def test_utilisation_rate_exact(ctx):
    """Two prior Rs 100 purchases in March, plus this Rs 100, against a
    Rs 1000 reserve = 0.30 exactly."""
    ev = ctx.events[2]                       # 19 March, 2 priors in March
    m = ctx.mandates["M0001"]
    assert F.f_utilisation_rate(ev, m, ctx, None, None) == pytest.approx(0.30)


def test_amount_vs_reserve_ratio_exact(ctx):
    """Rs 100 against a Rs 1000 reserve = 0.10."""
    ev = ctx.events[0]
    m = ctx.mandates["M0001"]
    assert F.f_amount_vs_reserve_ratio(ev, m, ctx, None, None) \
        == pytest.approx(0.10)


def test_outside_active_hours(ctx):
    """Principal's window is 08:00-20:00."""
    m = ctx.mandates["M0001"]
    inside = ctx.events[0]                   # 10:00
    assert F.f_outside_active_hours(inside, m, ctx, None, None) == 0.0

    night = EventRow(**{**inside.__dict__,
                        "timestamp": datetime(2026, 3, 5, 3)})
    assert F.f_outside_active_hours(night, m, ctx, None, None) == 1.0


# ---------------------------------------------------------------
# missing is NOT zero
# ---------------------------------------------------------------

def test_cold_start_returns_missing_not_zero(ctx):
    """With no history, a z-score must be NaN, not 0.

    Zero means 'exactly typical'. NaN means 'unknown'. Returning 0 would
    tell the model an unknown transaction is perfectly normal -- the
    opposite of true."""
    empty = Baseline(n_events=0, history_days=0.0)
    ev = ctx.events[0]
    m = ctx.mandates["M0001"]

    for fn in [F.f_amount_log_zscore, F.f_velocity_ratio_1h,
               F.f_merchant_new_to_principal, F.f_hour_deviation]:
        val = fn(ev, m, ctx, empty, [])
        assert np.isnan(val), f"{fn.__name__} returned {val}, expected NaN"


# ---------------------------------------------------------------
# structural guarantees
# ---------------------------------------------------------------

def test_feature_count_matches_function_count():
    """Column drift between train and predict is silent and catastrophic."""
    assert len(F._FUNCS) == len(F.FEATURE_NAMES) == 31


def test_feature_names_unique():
    assert len(set(F.FEATURE_NAMES)) == len(F.FEATURE_NAMES)


def test_vector_order_is_fixed(ctx):
    """to_vector must follow FEATURE_NAMES, never dict insertion order."""
    feats = {n: float(i) for i, n in enumerate(F.FEATURE_NAMES)}
    vec = F.to_vector(feats)
    assert list(vec) == [float(i) for i in range(len(F.FEATURE_NAMES))]


def test_route_c_features_are_not_read():
    """The Route C control depends on features.py never reading merchant
    provenance. If any of these strings appear, the control is broken."""
    import inspect
    source = inspect.getsource(F)
    body = source.split('"""', 2)[2]         # skip the module docstring
    for banned in ["reputation_score", "volume_band",
                   "merchant.registered_at"]:
        assert banned not in body, \
            f"features.py reads {banned} -- Route C control is invalid"