"""
VIVEKA — feature extraction (Stage 1: Groups A and B).

Group A — mandate state    (features 1-6)   : Routes A, B
Group B — velocity         (features 7-13)  : Route B

Design: Section 10. Every feature has a stated purpose there.

THREE RULES ENFORCED HERE
  1. All history comes from context.history_before() -> leakage rule L5.
  2. Missing is NOT zero. A zero z-score means "exactly typical"; a
     missing z-score means "unknown". Conflating them tells the model
     an unknown is normal (Section 10.8).
  3. Feature order is fixed by FEATURE_NAMES. Column drift between
     train and predict is a silent, catastrophic failure.

NOT COMPUTED HERE, DELIBERATELY
  merchant_age_days, merchant_reputation_score, price_vs_category_median,
  merchant_volume_band. These fields exist in the data because a real
  system would have them, but they are Route C signals and Route C is
  the methodological control (Section 10.6.1). The extractor never
  reads them.

RATIO CLIPPING
  Several features are ratios against a baseline rate. When a principal
  has a very sparse history the denominator approaches zero and the
  ratio explodes -- interarrival_zscore reached 34,717 before clipping.
  Gradient boosting tolerates that silently; logistic regression does
  not, since one exploded row can dominate the fit. Clipping preserves
  "extremely high" while removing the numerical blow-up.
"""

import calendar
import math
import time
from datetime import timedelta

import numpy as np

from src.context import Context, EventRow, Baseline

MISSING = np.nan

# Clip bounds for ratio features (see RATIO CLIPPING above).
MAX_VELOCITY_RATIO = 100.0
MAX_ABS_ZSCORE = 20.0
MIN_GAP_SD_DAYS = 0.01

# Fixed order. Never reorder; append only. (Section 10.8)
FEATURE_NAMES = [
    # --- Group A: mandate state ---
    "mandate_age_days",
    "days_since_confirmation",
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
]


# ---------------------------------------------------------------
# Group A — mandate state (Routes A, B)
# ---------------------------------------------------------------

def f_mandate_age_days(ev, mandate, ctx, bl, prior) -> float:
    """Feature 1. Stale consent is higher risk: intent drifts from the
    context in which it was granted.

    NOTE: perfectly correlated with feature 2 in this dataset, because
    no step-up confirmations occur in the simulation. Kept and reported
    rather than silently dropped (Section 23).
    """
    return (ev.timestamp - mandate.created_at).total_seconds() / 86400


def f_days_since_confirmation(ev, mandate, ctx, bl, prior) -> float:
    """Feature 2. Time since the principal last actively affirmed the
    mandate. Poisoning takes time, so this rises independently of the
    poisoned behaviour (Section 14.3.2).

    In production this diverges from feature 1: every step-up approval
    refreshes last_confirmed_at (Section 12.7).
    """
    return (ev.timestamp - mandate.last_confirmed_at).total_seconds() / 86400


def f_utilisation_rate(ev, mandate, ctx, bl, prior) -> float:
    """Feature 3. Fraction of the monthly reserve already consumed,
    including this transaction."""
    if mandate.reserve_limit_paise <= 0:
        return MISSING
    consumed = ctx.consumed_before(mandate.mandate_id, ev.timestamp)
    return (consumed + ev.amount_paise) / mandate.reserve_limit_paise


def f_utilisation_velocity(ev, mandate, ctx, bl, prior) -> float:
    """Feature 4. THE KEY FEATURE (Section 10.4, Group A).

    Consumption rate relative to the month's elapsed pace.

    MEASURED: the median is ~0.43, not ~1.0 as Section 10 originally
    stated. Reserves carry 2-3x headroom by design (Section 9), so a
    full month of normal spending only reaches ~0.4 utilisation. What
    matters is not the absolute value but how far a transaction sits
    above the principal's own normal -- a drain pushes it several times
    higher while every individual transaction stays fully authorised.

    Floor on elapsed time prevents a division blow-up on the 1st of the
    month, where a single normal purchase would otherwise look infinite.
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
    """Feature 5. Share of total granted authority consumed in one act."""
    if mandate.reserve_limit_paise <= 0:
        return MISSING
    return ev.amount_paise / mandate.reserve_limit_paise


def f_days_to_expiry(ev, mandate, ctx, bl, prior) -> float:
    """Feature 6. End-of-mandate drain behaviour. Negative means the
    mandate has already expired (hard rule H2 territory)."""
    return (mandate.expires_at - ev.timestamp).total_seconds() / 86400


# ---------------------------------------------------------------
# Group B — velocity (Route B)
# ---------------------------------------------------------------

def _count_within(prior, ts, hours: float) -> int:
    """Prior events within a trailing window ending at ts."""
    cutoff = ts - timedelta(hours=hours)
    return sum(1 for e in prior if e.timestamp >= cutoff)


def f_txn_count_1h(ev, mandate, ctx, bl, prior) -> float:
    """Feature 7. Raw burst count. Absolute volume carries some signal
    independently of the principal's own rate."""
    return float(_count_within(prior, ev.timestamp, 1))


def f_txn_count_24h(ev, mandate, ctx, bl, prior) -> float:
    """Feature 8."""
    return float(_count_within(prior, ev.timestamp, 24))


def f_txn_count_7d(ev, mandate, ctx, bl, prior) -> float:
    """Feature 9."""
    return float(_count_within(prior, ev.timestamp, 24 * 7))


def f_velocity_ratio_1h(ev, mandate, ctx, bl, prior) -> float:
    """Feature 10. Burst RELATIVE to this principal (Section 10.2, F2).

    A global threshold like "5 per hour is suspicious" is meaningless:
    frequent for one person, routine for another.

    Clipped: a principal with a very low hourly_rate divides a small
    count by a tiny number and reached 678 before clipping.
    """
    if bl.n_events == 0 or bl.hourly_rate <= 0:
        return MISSING
    ratio = _count_within(prior, ev.timestamp, 1) / bl.hourly_rate
    return float(min(ratio, MAX_VELOCITY_RATIO))


def f_velocity_ratio_24h(ev, mandate, ctx, bl, prior) -> float:
    """Feature 11. Same, on a daily horizon. Clipped for the same reason."""
    if bl.n_events == 0 or bl.daily_rate <= 0:
        return MISSING
    ratio = _count_within(prior, ev.timestamp, 24) / bl.daily_rate
    return float(min(ratio, MAX_VELOCITY_RATIO))


def f_interarrival_zscore(ev, mandate, ctx, bl, prior) -> float:
    """Feature 12. Rhythm break. Negative means faster than usual.

    Needs at least 3 prior events for a meaningful standard deviation.
    The denominator is floored: a principal with near-simultaneous prior
    events has sd ~ 0, which produced z-scores over 30,000.
    """
    if bl.n_events < 3 or bl.last_timestamp is None:
        return MISSING
    sd = max(bl.sd_gap_days, MIN_GAP_SD_DAYS)
    gap = (ev.timestamp - bl.last_timestamp).total_seconds() / 86400
    z = (gap - bl.mean_gap_days) / sd
    return float(np.clip(z, -MAX_ABS_ZSCORE, MAX_ABS_ZSCORE))


def f_amount_sum_1h_ratio(ev, mandate, ctx, bl, prior) -> float:
    """Feature 13. Value concentration: spend in the trailing hour
    (including this transaction) against a normal week's spend."""
    if bl.n_events == 0 or bl.weekly_spend <= 0:
        return MISSING
    cutoff = ev.timestamp - timedelta(hours=1)
    recent = sum(e.amount_paise for e in prior if e.timestamp >= cutoff)
    return (recent + ev.amount_paise) / bl.weekly_spend


# ---------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------

_FUNCS = [
    f_mandate_age_days,
    f_days_since_confirmation,
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
]

assert len(_FUNCS) == len(FEATURE_NAMES), "feature list / function list mismatch"


def extract(ev: EventRow, ctx: Context) -> dict:
    """All Stage 1 features for one event, as a name -> value dict."""
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
    print(f"matrix shape : {X.shape}")

    print("\nper-feature summary (non-missing only):")
    print(f"{'feature':<26}{'miss%':>7}{'median':>12}{'p95':>12}{'max':>12}")
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        miss = np.isnan(col).mean() * 100
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            print(f"{name:<26}{miss:>6.1f}%{'--':>12}{'--':>12}{'--':>12}")
            continue
        print(f"{name:<26}{miss:>6.1f}%"
              f"{np.median(valid):>12.2f}"
              f"{np.percentile(valid, 95):>12.2f}"
              f"{valid.max():>12.2f}")

    n_inf = np.isinf(X).sum()
    print(f"\ninfinite values : {n_inf}  (want 0)")

    # --- correlation between features 1 and 2 (expected: 1.00) ---
    a, b = X[:, 0], X[:, 1]
    ok = ~(np.isnan(a) | np.isnan(b))
    corr = np.corrcoef(a[ok], b[ok])[0, 1]
    print(f"corr(f1, f2)    : {corr:.4f}  (expected 1.00, see Section 23)")


if __name__ == "__main__":
    main()