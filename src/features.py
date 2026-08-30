"""
VIVEKA — feature extraction (31 features).

Group A — mandate state          (1-8)    : Routes A, B
Group B — velocity               (9-15)   : Route B
Group C — amount                 (16-18)  : Routes A, B
Group D — temporal               (19-23)  : Routes A, B
Group E — merchant relationship  (24-28)  : Route A
Group F — instruction & maturity (29-31)  : Route A, cold start

Design: Section 10. Every feature has a stated purpose there.

TWO CHANGES AFTER THE ABLATION (Section 21)
  1. merchant_in_mandate_scope moved here from the rule layer.
     H4 was miscategorised. Section 11 says "policy violations are
     FACTS; ML is for JUDGMENT". H1/H2/H3 are facts -- no authority
     exists. H4 is a judgment -- authority exists but intent is
     unclear. As a hard rule it forced a step-up on 5.85% of traffic
     regardless of anything else, and the ablation showed that cost
     more than it saved. As a feature the model weighs it against
     30 other signals.
  2. days_since_confirmation now reads real confirmation events.
     Previously identical to mandate_age_days (permutation importance
     0.0000) because no confirmations existed in the simulation.

THREE RULES ENFORCED HERE
  1. All history comes from context.history_before() -> leakage rule L5.
  2. Missing is NOT zero. A zero z-score means "exactly typical"; a
     missing z-score means "unknown" (Section 10.8).
  3. Feature order is fixed by FEATURE_NAMES. Column drift between
     train and predict is a silent, catastrophic failure.

NOT COMPUTED HERE, DELIBERATELY
  merchant_age_days, merchant_reputation_score, price_vs_category_median,
  merchant_volume_band -- Route C signals, and Route C is the
  methodological control (Section 10.6.1).
  No demographics, location, or device -- fairness by construction
  (Section 10.6.2). The model cannot discriminate on attributes it
  never receives.
"""

import calendar
import time
from datetime import timedelta

import numpy as np

from src.context import Context, EventRow, Baseline, circular_hour_distance

MISSING = np.nan

MAX_VELOCITY_RATIO = 100.0
MAX_ABS_ZSCORE = 20.0
MIN_GAP_SD_DAYS = 0.01
MIN_LOG_SD = 0.05

AGENT_EXPECTED_CATEGORIES = {
    "grocery":      {"grocery", "food"},
    "travel":       {"travel"},
    "subscription": {"other", "electronics"},
}

# Fixed order. Never reorder; append only. (Section 10.8)
FEATURE_NAMES = [
    # --- Group A: mandate state ---
    "mandate_age_days",
    "days_since_confirmation",
    "confirmation_count",
    "merchant_in_mandate_scope",
    "utilisation_rate",
    "utilisation_velocity",
    "amount_vs_reserve_ratio",
    "days_to_expiry",
    # --- Group B: velocity ---
    "txn_count_1h",
    "txn_count_24h",
    "txn_count_7d",
    "velocity_ratio_1h",
    "velocity_ratio_24h",
    "interarrival_zscore",
    "amount_sum_1h_ratio",
    # --- Group C: amount ---
    "amount_log_zscore",
    "amount_percentile",
    "amount_vs_median_ratio",
    # --- Group D: temporal ---
    "hour_of_day",
    "outside_active_hours",
    "hour_deviation",
    "is_weekend",
    "days_since_last_txn",
    # --- Group E: merchant relationship ---
    "merchant_new_to_principal",
    "txn_count_with_merchant",
    "category_new_to_principal",
    "category_matches_agent_type",
    "merchant_concentration",
    # --- Group F: instruction source & maturity ---
    "is_external_content",
    "external_content_rate_deviation",
    "principal_history_days",
]


# ---------------------------------------------------------------
# Group A — mandate state (Routes A, B)
# ---------------------------------------------------------------

def f_mandate_age_days(ev, mandate, ctx, bl, prior) -> float:
    """Feature 1. Stale consent is higher risk: intent drifts from the
    context in which it was granted."""
    return (ev.timestamp - mandate.created_at).total_seconds() / 86400


def f_days_since_confirmation(ev, mandate, ctx, bl, prior) -> float:
    """Feature 2. Time since the principal last ACTIVELY affirmed the
    mandate -- through a step-up approval, or the original consent if
    none has occurred yet.

    Distinct from feature 1: a mandate created 90 days ago but confirmed
    3 days ago carries far less staleness risk than one never
    re-confirmed. Poisoning takes time, so this rises independently of
    the poisoned behaviour (Section 14.3.2).

    Reads only confirmations strictly before this event (L5).
    """
    last = ctx.last_confirmation_before(mandate.mandate_id, ev.timestamp)
    return (ev.timestamp - last).total_seconds() / 86400


def f_confirmation_count(ev, mandate, ctx, bl, prior) -> float:
    """Feature 3. How many times the principal has affirmed this mandate.

    A mandate re-confirmed several times reflects an engaged principal
    who has repeatedly agreed to how their agent is spending. Zero
    confirmations means the consent has never been revisited since the
    day it was granted.
    """
    return float(ctx.confirmation_count_before(mandate.mandate_id,
                                               ev.timestamp))


def f_merchant_in_mandate_scope(ev, mandate, ctx, bl, prior) -> float:
    """Feature 4. Is this merchant on the mandate's approved list?

    Moved here from hard rule H4 after the ablation (Section 21). As a
    rule it forced a step-up on every out-of-scope transaction, which
    is 5.85% of traffic and mostly legitimate customers trying a new
    shop. As a feature the model can weigh it against velocity, amount
    deviation and merchant familiarity instead of acting alone.

    1.0 = in scope, 0.0 = outside scope.
    """
    return 1.0 if ev.merchant_id in mandate.merchant_scope else 0.0


def f_utilisation_rate(ev, mandate, ctx, bl, prior) -> float:
    """Feature 5. Fraction of the monthly reserve consumed, including
    this transaction."""
    if mandate.reserve_limit_paise <= 0:
        return MISSING
    consumed = ctx.consumed_before(mandate.mandate_id, ev.timestamp)
    return (consumed + ev.amount_paise) / mandate.reserve_limit_paise


def f_utilisation_velocity(ev, mandate, ctx, bl, prior) -> float:
    """Feature 6. THE KEY FEATURE for Route B (Section 10.4).

    Consumption rate relative to the month's elapsed pace.

    MEASURED: benign median ~0.42, Route B ~0.94. Section 10 originally
    claimed ~1.0 for normal spending; that was wrong. Reserves carry
    2-3x headroom by design (Section 9), so a normal full month reaches
    only ~0.4. The signal is the ratio to the principal's own normal,
    not the absolute value.

    Floor on elapsed time prevents a blow-up on the 1st of the month.
    """
    if mandate.reserve_limit_paise <= 0:
        return MISSING
    ts = ev.timestamp
    days_in_month = calendar.monthrange(ts.year, ts.month)[1]
    elapsed_days = (ts.day - 1) + ts.hour / 24 + ts.minute / 1440
    elapsed_frac = max(elapsed_days, 0.5) / days_in_month

    util = f_utilisation_rate(ev, mandate, ctx, bl, prior)
    if np.isnan(util):
        return MISSING
    return util / elapsed_frac


def f_amount_vs_reserve_ratio(ev, mandate, ctx, bl, prior) -> float:
    """Feature 7. Share of total granted authority consumed in one act."""
    if mandate.reserve_limit_paise <= 0:
        return MISSING
    return ev.amount_paise / mandate.reserve_limit_paise


def f_days_to_expiry(ev, mandate, ctx, bl, prior) -> float:
    """Feature 8. End-of-mandate drain behaviour. Negative means already
    expired (hard rule H2 territory)."""
    return (mandate.expires_at - ev.timestamp).total_seconds() / 86400


# ---------------------------------------------------------------
# Group B — velocity (Route B)
# ---------------------------------------------------------------

def _count_within(prior, ts, hours: float) -> int:
    cutoff = ts - timedelta(hours=hours)
    return sum(1 for e in prior if e.timestamp >= cutoff)


def f_txn_count_1h(ev, mandate, ctx, bl, prior) -> float:
    """Feature 9. Raw burst count."""
    return float(_count_within(prior, ev.timestamp, 1))


def f_txn_count_24h(ev, mandate, ctx, bl, prior) -> float:
    """Feature 10."""
    return float(_count_within(prior, ev.timestamp, 24))


def f_txn_count_7d(ev, mandate, ctx, bl, prior) -> float:
    """Feature 11.

    NOTE: permutation importance 0.0000 in the first evaluation -- the
    1h and 24h windows already capture bursts. Kept and reported rather
    than silently dropped (Section 23).
    """
    return float(_count_within(prior, ev.timestamp, 24 * 7))


def f_velocity_ratio_1h(ev, mandate, ctx, bl, prior) -> float:
    """Feature 12. Burst RELATIVE to this principal (Section 10.2, F2).

    MEASURED: benign median 0.00, Route B median at the clip ceiling of
    100. The clip compresses genuine signal here, not just numerical
    blow-ups -- a stated design choice, recorded in BUGLOG.
    """
    if bl.n_events == 0 or bl.hourly_rate <= 0:
        return MISSING
    ratio = _count_within(prior, ev.timestamp, 1) / bl.hourly_rate
    return float(min(ratio, MAX_VELOCITY_RATIO))


def f_velocity_ratio_24h(ev, mandate, ctx, bl, prior) -> float:
    """Feature 13. Same, daily horizon."""
    if bl.n_events == 0 or bl.daily_rate <= 0:
        return MISSING
    ratio = _count_within(prior, ev.timestamp, 24) / bl.daily_rate
    return float(min(ratio, MAX_VELOCITY_RATIO))


def f_interarrival_zscore(ev, mandate, ctx, bl, prior) -> float:
    """Feature 14. Rhythm break. Negative means faster than usual.

    Denominator floored: near-simultaneous prior events give sd ~ 0,
    which produced z-scores over 30,000.
    """
    if bl.n_events < 3 or bl.last_timestamp is None:
        return MISSING
    sd = max(bl.sd_gap_days, MIN_GAP_SD_DAYS)
    gap = (ev.timestamp - bl.last_timestamp).total_seconds() / 86400
    z = (gap - bl.mean_gap_days) / sd
    return float(np.clip(z, -MAX_ABS_ZSCORE, MAX_ABS_ZSCORE))


def f_amount_sum_1h_ratio(ev, mandate, ctx, bl, prior) -> float:
    """Feature 15. Value concentration: trailing-hour spend against a
    normal week's spend."""
    if bl.n_events == 0 or bl.weekly_spend <= 0:
        return MISSING
    cutoff = ev.timestamp - timedelta(hours=1)
    recent = sum(e.amount_paise for e in prior if e.timestamp >= cutoff)
    return (recent + ev.amount_paise) / bl.weekly_spend


# ---------------------------------------------------------------
# Group C — amount (Routes A, B)
# ---------------------------------------------------------------

def f_amount_log_zscore(ev, mandate, ctx, bl, prior) -> float:
    """Feature 16. Deviation from typical spend, in log space.

    Log because spend is log-normal (Section 9.5): a raw z-score would
    be dominated by the long right tail.
    """
    if bl.n_events < 3 or bl.sd_log_amount <= 0:
        return MISSING
    sd = max(bl.sd_log_amount, MIN_LOG_SD)
    z = (np.log(ev.amount_paise) - bl.mean_log_amount) / sd
    return float(np.clip(z, -MAX_ABS_ZSCORE, MAX_ABS_ZSCORE))


def f_amount_percentile(ev, mandate, ctx, bl, prior) -> float:
    """Feature 17. Where this amount sits in the principal's own history.

    Robust where a z-score is not: percentile is unaffected by how heavy
    the tail is.
    """
    if bl.n_events == 0:
        return MISSING
    below = sum(1 for a in bl.amounts if a <= ev.amount_paise)
    return below / bl.n_events


def f_amount_vs_median_ratio(ev, mandate, ctx, bl, prior) -> float:
    """Feature 18. Interpretable magnitude: 3.0 means three times their
    usual purchase."""
    if bl.n_events == 0 or bl.median_amount <= 0:
        return MISSING
    return float(min(ev.amount_paise / bl.median_amount, 100.0))


# ---------------------------------------------------------------
# Group D — temporal (Routes A, B)
# ---------------------------------------------------------------

def f_hour_of_day(ev, mandate, ctx, bl, prior) -> float:
    """Feature 19. Raw temporal context."""
    return float(ev.timestamp.hour)


def f_outside_active_hours(ev, mandate, ctx, bl, prior) -> float:
    """Feature 20. Outside this principal's declared window.

    MEASURED: highest permutation importance of all features (0.22).
    """
    p = ctx.principals[ev.principal_id]
    h = ev.timestamp.hour
    return 0.0 if p.active_hour_start <= h <= p.active_hour_end else 1.0


def f_hour_deviation(ev, mandate, ctx, bl, prior) -> float:
    """Feature 21. Continuous version of 20, against OBSERVED history.

    Uses circular distance: 23:00 to 01:00 is 2 hours, not 22.
    """
    if bl.n_events == 0:
        return MISSING
    return circular_hour_distance(ev.timestamp.hour, bl.typical_hour)


def f_is_weekend(ev, mandate, ctx, bl, prior) -> float:
    """Feature 22. Weekday and weekend patterns differ."""
    return 1.0 if ev.timestamp.weekday() >= 5 else 0.0


def f_days_since_last_txn(ev, mandate, ctx, bl, prior) -> float:
    """Feature 23. Dormancy followed by sudden activity."""
    if bl.last_timestamp is None:
        return MISSING
    return (ev.timestamp - bl.last_timestamp).total_seconds() / 86400


# ---------------------------------------------------------------
# Group E — merchant relationship (Route A)
# PRINCIPAL-RELATIVE ONLY. See module docstring.
# ---------------------------------------------------------------

def f_merchant_new_to_principal(ev, mandate, ctx, bl, prior) -> float:
    """Feature 24. First time THIS principal has used THIS merchant.

    Asks "new to Priya", NOT "newly registered" -- the latter is a
    Route C signal and is excluded (Section 10.6.1).
    """
    if bl.n_events == 0:
        return MISSING
    return 0.0 if ev.merchant_id in bl.merchant_counts else 1.0


def f_txn_count_with_merchant(ev, mandate, ctx, bl, prior) -> float:
    """Feature 25. Familiarity depth with this specific merchant."""
    return float(bl.merchant_counts.get(ev.merchant_id, 0))


def f_category_new_to_principal(ev, mandate, ctx, bl, prior) -> float:
    """Feature 26. Category drift."""
    if bl.n_events == 0:
        return MISSING
    return 0.0 if ev.merchant_category in bl.category_counts else 1.0


def f_category_matches_agent_type(ev, mandate, ctx, bl, prior) -> float:
    """Feature 27. Purpose mismatch: is a grocery agent buying groceries?

    Independent of history, so it works from the very first transaction.
    """
    agent = ctx.agents[ev.agent_id]
    expected = AGENT_EXPECTED_CATEGORIES.get(agent.agent_type)
    if expected is None:
        return MISSING
    return 1.0 if ev.merchant_category in expected else 0.0


def f_merchant_concentration(ev, mandate, ctx, bl, prior) -> float:
    """Feature 28. Share of past spend at the top 3 merchants."""
    if bl.n_events == 0:
        return MISSING
    return bl.top3_share


# ---------------------------------------------------------------
# Group F — instruction source & maturity (Route A, cold start)
# ---------------------------------------------------------------

def f_is_external_content(ev, mandate, ctx, bl, prior) -> float:
    """Feature 29. Agent acted after processing untrusted content.

    NOT a label: ~10% of benign transactions use this too.
    """
    return 1.0 if ev.instruction_source == "external_content" else 0.0


def f_external_content_rate_deviation(ev, mandate, ctx, bl, prior) -> float:
    """Feature 30. Deviation matters more than the flag itself."""
    if bl.n_events < 5:
        return MISSING
    current = 1.0 if ev.instruction_source == "external_content" else 0.0
    return current - bl.ext_content_rate


def f_principal_history_days(ev, mandate, ctx, bl, prior) -> float:
    """Feature 31. Cold-start indicator.

    Lets the model learn to distrust its own baseline-derived features
    when history is thin (Section 10.7).
    """
    return bl.history_days


# ---------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------

_FUNCS = [
    f_mandate_age_days,
    f_days_since_confirmation,
    f_confirmation_count,
    f_merchant_in_mandate_scope,
    f_utilisation_rate,
    f_utilisation_velocity,
    f_amount_vs_reserve_ratio,
    f_days_to_expiry,
    f_txn_count_1h,
    f_txn_count_24h,
    f_txn_count_7d,
    f_velocity_ratio_1h,
    f_velocity_ratio_24h,
    f_interarrival_zscore,
    f_amount_sum_1h_ratio,
    f_amount_log_zscore,
    f_amount_percentile,
    f_amount_vs_median_ratio,
    f_hour_of_day,
    f_outside_active_hours,
    f_hour_deviation,
    f_is_weekend,
    f_days_since_last_txn,
    f_merchant_new_to_principal,
    f_txn_count_with_merchant,
    f_category_new_to_principal,
    f_category_matches_agent_type,
    f_merchant_concentration,
    f_is_external_content,
    f_external_content_rate_deviation,
    f_principal_history_days,
]

assert len(_FUNCS) == len(FEATURE_NAMES), \
    f"mismatch: {len(_FUNCS)} funcs, {len(FEATURE_NAMES)} names"


def extract(ev: EventRow, ctx: Context) -> dict:
    """All 31 features for one event, as a name -> value dict."""
    mandate = ctx.mandates[ev.mandate_id]
    prior = ctx.history_before(ev.principal_id, ev.timestamp)
    bl = ctx.baseline(ev.principal_id, ev.timestamp)
    return {
        name: fn(ev, mandate, ctx, bl, prior)
        for name, fn in zip(FEATURE_NAMES, _FUNCS)
    }


def to_vector(feats: dict) -> np.ndarray:
    """Dict -> array in FIXED order. Never rely on dict ordering."""
    return np.array([feats[n] for n in FEATURE_NAMES], dtype=float)


def extract_all(ctx: Context):
    """Extract for every event. Returns (event_ids, matrix)."""
    ids, rows = [], []
    for ev in ctx.events:
        ids.append(ev.event_id)
        rows.append(to_vector(extract(ev, ctx)))
    return ids, np.vstack(rows)


def main():
    t0 = time.perf_counter()
    ctx = Context.load()
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    ids, X = extract_all(ctx)
    t_extract = time.perf_counter() - t1

    print(f"load time    : {t_load:.1f}s")
    print(f"extract time : {t_extract:.1f}s for {len(ids):,} events")
    print(f"per event    : {t_extract / len(ids) * 1000:.2f} ms")
    print(f"matrix shape : {X.shape}   (want 31 columns)")

    print("\nper-feature summary (non-missing only):")
    print(f"{'feature':<34}{'miss%':>7}{'median':>10}{'p95':>10}{'max':>10}")
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        miss = np.isnan(col).mean() * 100
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            print(f"{name:<34}{miss:>6.1f}%{'--':>10}{'--':>10}{'--':>10}")
            continue
        print(f"{name:<34}{miss:>6.1f}%"
              f"{np.median(valid):>10.2f}"
              f"{np.percentile(valid, 95):>10.2f}"
              f"{valid.max():>10.2f}")

    n_inf = np.isinf(X).sum()
    print(f"\ninfinite values : {n_inf}  (want 0)")

    # Did the confirmation fix work? These must no longer be identical.
    a, b = X[:, 0], X[:, 1]
    ok = ~(np.isnan(a) | np.isnan(b))
    corr = np.corrcoef(a[ok], b[ok])[0, 1]
    print(f"corr(mandate_age, days_since_confirmation): {corr:.4f}")
    print(f"  (was 1.0000 before confirmations were simulated)")


if __name__ == "__main__":
    main()