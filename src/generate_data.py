"""
VIVEKA — synthetic data generator.

Produces agent-mediated UPI transaction data with labelled attacks.
Design: Section 9. Leakage rules: Section 9.9.

Run:  python -m src.generate_data --seed 42
"""

import argparse
import csv
import json
import random
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.schema import (
    Principal, Agent, Mandate, Merchant, TransactionEvent, Label,
    SpendProfile, RhythmProfile, AgentType,
    MerchantCategory, VolumeBand, MandateStatus,
    Channel, InstructionSource, AttackRoute,
)

# --- Simulation constants (Section 9.7) ---
N_PRINCIPALS = 1200
N_MERCHANTS = 200
HISTORY_DAYS = 120
SIM_END = datetime(2026, 8, 1, 0, 0, 0)
SIM_START = SIM_END - timedelta(days=HISTORY_DAYS)
OUT_DIR = Path("data")

SPEND_PARAMS = {
    SpendProfile.LOW:    (40_000,  0.45),
    SpendProfile.MEDIUM: (120_000, 0.50),
    SpendProfile.HIGH:   (300_000, 0.55),
}

RHYTHM_DAYS = {
    RhythmProfile.WEEKLY:    2.5,
    RhythmProfile.BIWEEKLY:  5.0,
    RhythmProfile.IRREGULAR: 3.5,
}

AGENT_CATEGORIES = {
    AgentType.GROCERY:      [MerchantCategory.GROCERY, MerchantCategory.FOOD],
    AgentType.TRAVEL:       [MerchantCategory.TRAVEL],
    AgentType.SUBSCRIPTION: [MerchantCategory.OTHER, MerchantCategory.ELECTRONICS],
}

# Benign behaviour parameters (Section 9.5).
# These rates prevent leakage: if only attacks were out-of-scope /
# off-hours / external-content, those fields would be perfect label
# proxies (traps L1, L3).
P_OUT_OF_SCOPE   = 0.05
P_OFF_HOURS      = 0.06
P_AMOUNT_OUTLIER = 0.03
OUTLIER_MULT     = (3.0, 8.0)

BENIGN_INSTRUCTION_MIX = [
    (InstructionSource.SCHEDULED,        0.55),
    (InstructionSource.USER_DIRECT,      0.20),
    (InstructionSource.AGENT_AUTONOMOUS, 0.15),
    (InstructionSource.EXTERNAL_CONTENT, 0.10),
]

# Attacks must NOT map deterministically to one instruction_source, or
# that field becomes a label proxy (trap L3). Real injection arrives
# through more than one path.
A_INSTRUCTION_MIX = [
    (InstructionSource.EXTERNAL_CONTENT, 0.65),
    (InstructionSource.AGENT_AUTONOMOUS, 0.20),
    (InstructionSource.SCHEDULED,        0.15),
]

B_INSTRUCTION_MIX = [
    (InstructionSource.AGENT_AUTONOMOUS, 0.50),
    (InstructionSource.SCHEDULED,        0.35),
    (InstructionSource.USER_DIRECT,      0.15),
]

P_B_OFF_HOURS = 0.65   # not every drain happens at 3am

# --- Mandate lifecycle: makes H1/H2 reachable (Section 11.2.1) ---
P_MANDATE_EXPIRES = 0.04
P_MANDATE_REVOKED = 0.03

# --- Step-up confirmations (Section 12.7) ---
# In production every step-up approval refreshes last_confirmed_at.
# Without these events, days_since_confirmation is mathematically
# identical to mandate_age_days and contributes nothing to the model
# (permutation importance 0.0000 -- BUGLOG).
P_PRINCIPAL_CONFIRMS = 0.55        # share of principals who ever confirm
CONFIRMATIONS_PER_PRINCIPAL = (1, 4)

# --- Attack parameters (Section 9.6) ---
ATTACK_RATE  = 0.02
ROUTE_SPLIT  = {AttackRoute.A_INJECTION: 0.30,
                AttackRoute.B_DELEGATION_ABUSE: 0.50,
                AttackRoute.C_COUNTERFEIT_MERCHANT: 0.20}
B_BURST_SIZE = (3, 6)
P_STALE_VARIANT = 0.12   # share of Route B budget spent on stale credentials

# --- Splits (Section 9.10) ---
SPLIT_RATIOS = {"train": 0.60, "val": 0.20, "holdout": 0.20}


def make_merchants(rng):
    """Create the merchant pool.

    Provenance fields (registered_at, reputation_score, volume_band) exist
    because a real system would have them, but features.py never reads
    them -- that is the Route C control (Section 10.6.1).
    """
    merchants = []
    categories = list(MerchantCategory)
    for i in range(N_MERCHANTS):
        if rng.random() < 0.15:
            age_days = rng.randint(1, 60)
        else:
            age_days = rng.randint(200, 2000)
        merchants.append(Merchant(
            merchant_id=f"MER{i:04d}",
            category=rng.choice(categories),
            registered_at=SIM_END - timedelta(days=age_days),
            reputation_score=round(min(1.0, age_days / 1000 + rng.random() * 0.3), 3),
            volume_band=rng.choice(list(VolumeBand)),
        ))
    return merchants


def make_principals(rng):
    principals = []
    for i in range(N_PRINCIPALS):
        start_hour = rng.randint(6, 11)
        principals.append(Principal(
            principal_id=f"P{i:04d}",
            created_at=SIM_START - timedelta(days=rng.randint(30, 900)),
            spend_profile=rng.choice(list(SpendProfile)),
            rhythm_profile=rng.choice(list(RhythmProfile)),
            active_hour_start=start_hour,
            active_hour_end=min(23, start_hour + rng.randint(8, 13)),
        ))
    return principals


def assign_splits(principals, rng):
    """Split by PRINCIPAL, never by transaction (leakage trap L7).

    A random transaction-level split would put the same person in train
    and test, so the model would already know their baseline and recall
    would be fiction.
    """
    ids = [p.principal_id for p in principals]
    shuffled = ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    splits = {}
    for pid in shuffled[:n_train]:
        splits[pid] = "train"
    for pid in shuffled[n_train:n_train + n_val]:
        splits[pid] = "val"
    for pid in shuffled[n_train + n_val:]:
        splits[pid] = "holdout"
    return splits


def make_agents_and_mandates(principals, merchants, rng):
    """One agent and one mandate per principal.

    A small fraction of mandates expire or are revoked inside the window,
    so hard rules H1 (revoked) and H2 (expired) are reachable.

    revoked_at records WHEN revocation took effect. Status alone is not
    enough: without a timestamp, H1 would flag every transaction on that
    mandate including ones that were valid when they happened (BUGLOG).

    confirmation_times simulates step-up approvals. Each one refreshes
    the principal's active affirmation of the mandate, which is what
    makes days_since_confirmation differ from mandate_age_days.
    """
    agents, mandates = [], []
    mandate_end = {}

    for idx, p in enumerate(principals):
        agent_type = rng.choice(list(AgentType))
        allowed_cats = AGENT_CATEGORIES[agent_type]

        candidates = [m for m in merchants if m.category in allowed_cats]
        if len(candidates) < 4:
            candidates = merchants
        scope = rng.sample(candidates, k=rng.randint(2, 4))

        registered = SIM_START + timedelta(days=rng.randint(0, 20))
        created = registered + timedelta(hours=rng.randint(1, 48))

        # Reserve must cover a MONTH of normal spending, with headroom.
        # Size against the EXPECTED (mean) transaction, not the median:
        #   - a log-normal's mean is median * exp(sigma^2 / 2)
        #   - the 3% outlier injection multiplies amounts by 3-8x
        # Using the median under-provisions the reserve and makes benign
        # users breach hard rule H3, which would block real customers.
        median_spend, sigma = SPEND_PARAMS[p.spend_profile]
        mean_txn = median_spend * np.exp(sigma ** 2 / 2)
        mean_txn *= 1 + P_AMOUNT_OUTLIER * (np.mean(OUTLIER_MULT) - 1)
        mean_gap_days = RHYTHM_DAYS[p.rhythm_profile]
        expected_monthly_txns = 30.0 / mean_gap_days
        expected_monthly_spend = mean_txn * expected_monthly_txns
        reserve = int(expected_monthly_spend * rng.uniform(2.0, 3.2))

        status = MandateStatus.ACTIVE
        expires = created + timedelta(days=rng.randint(180, 540))
        revoked_at = None
        end = SIM_END
        roll = rng.random()
        if roll < P_MANDATE_EXPIRES:
            expires = created + timedelta(days=rng.randint(40, 100))
            end = min(end, expires)
        elif roll < P_MANDATE_EXPIRES + P_MANDATE_REVOKED:
            status = MandateStatus.REVOKED
            revoked_at = created + timedelta(days=rng.randint(40, 100))
            end = revoked_at

        # Step-up confirmations, spread across the mandate's active life.
        confirmations = []
        if rng.random() < P_PRINCIPAL_CONFIRMS:
            active_days = max((min(end, SIM_END) - created).days, 1)
            for _ in range(rng.randint(*CONFIRMATIONS_PER_PRINCIPAL)):
                offset = rng.uniform(1, active_days)
                confirmations.append(created + timedelta(days=offset))
            confirmations.sort()

        agents.append(Agent(
            agent_id=f"A{idx:04d}",
            principal_id=p.principal_id,
            agent_type=agent_type,
            registered_at=registered,
        ))

        mandates.append(Mandate(
            mandate_id=f"M{idx:04d}",
            principal_id=p.principal_id,
            agent_id=f"A{idx:04d}",
            merchant_scope=[m.merchant_id for m in scope],
            reserve_limit_paise=reserve,
            created_at=created,
            last_confirmed_at=created,
            expires_at=expires,
            status=status,
            revoked_at=revoked_at,
            confirmation_times=confirmations,
        ))
        mandate_end[f"M{idx:04d}"] = min(end, SIM_END)

    return agents, mandates, mandate_end


def _pick_weighted(options, rng):
    values = [o[0] for o in options]
    weights = [o[1] for o in options]
    return rng.choices(values, weights=weights, k=1)[0]


def _draw_amount_paise(median_paise, sigma, rng):
    """Log-normal draw. Spend is log-normal: most small, a few large."""
    value = np.random.lognormal(mean=np.log(median_paise), sigma=sigma)
    return max(1000, int(value))


def _make_event(ts, p, mandate, merchant, amount, source, rng):
    """Single construction point for events.

    Benign and attack rows are built by the SAME function so no
    structural artifact can distinguish them (leakage rule P4/L2).
    """
    return TransactionEvent(
        event_id=str(uuid.uuid4()),
        timestamp=ts,
        principal_id=p.principal_id,
        agent_id=mandate.agent_id,
        mandate_id=mandate.mandate_id,
        merchant_id=merchant.merchant_id,
        amount_paise=amount,
        merchant_category=merchant.category,
        item_count=rng.randint(1, 5),
        channel=Channel.AGENTIC,
        instruction_source=source,
    )


def generate_benign_transactions(principals, mandates, merchants, mandate_end, rng):
    """Normal agent activity, stopping when the mandate becomes inactive.

    Deliberate messiness (Section 9.5): ~3% amount outliers, ~5%
    out-of-scope merchants, ~6% off-hours. Without these, benign traffic
    is uniform, any detector scores near-perfect precision, and the
    evaluation means nothing.
    """
    merchant_by_id = {m.merchant_id: m for m in merchants}
    mandate_by_principal = {m.principal_id: m for m in mandates}
    all_merchant_ids = [m.merchant_id for m in merchants]

    events = []
    for p in principals:
        mandate = mandate_by_principal[p.principal_id]
        stop = mandate_end[mandate.mandate_id]
        median_paise, sigma = SPEND_PARAMS[p.spend_profile]
        mean_gap = RHYTHM_DAYS[p.rhythm_profile]

        scope = mandate.merchant_scope
        weights = sorted([rng.random() + 0.1 for _ in scope], reverse=True)
        scope_weighted = list(zip(scope, weights))

        current = mandate.created_at + timedelta(hours=rng.randint(2, 72))

        while current < stop:
            if rng.random() < P_OUT_OF_SCOPE:
                merchant_id = rng.choice(all_merchant_ids)
            else:
                merchant_id = _pick_weighted(scope_weighted, rng)
            merchant = merchant_by_id[merchant_id]

            if rng.random() < P_OFF_HOURS:
                hour = rng.randint(0, 23)
            else:
                hour = rng.randint(p.active_hour_start, p.active_hour_end)

            ts = current.replace(hour=hour, minute=rng.randint(0, 59),
                                 second=rng.randint(0, 59), microsecond=0)
            if ts >= stop:
                break

            amount = _draw_amount_paise(median_paise, sigma, rng)
            if rng.random() < P_AMOUNT_OUTLIER:
                amount = int(amount * rng.uniform(*OUTLIER_MULT))

            events.append(_make_event(
                ts, p, mandate, merchant, amount,
                _pick_weighted(BENIGN_INSTRUCTION_MIX, rng), rng))

            gap = rng.expovariate(1.0 / mean_gap)
            current = current + timedelta(days=min(gap, mean_gap * 5))

    events.sort(key=lambda e: e.timestamp)
    return events


def generate_attacks(principals, mandates, merchants, mandate_end,
                     benign, splits, rng):
    """Inject the three attack routes (Section 9.6).

    Route A - prompt injection hijack : features engineered (Section 10.5)
    Route B - delegation abuse        : features engineered
      B variant - stale credential    : caught by policy, not the model
    Route C - counterfeit merchant    : NO features; HELD-OUT ONLY.

    Route C is restricted to held-out principals so it cannot influence
    feature selection or threshold tuning. That is what makes it a real
    test of generalisation rather than a claimed one (Section 4.5).
    """
    merchant_by_id = {m.merchant_id: m for m in merchants}
    mandate_by_principal = {m.principal_id: m for m in mandates}
    principal_by_id = {p.principal_id: p for p in principals}

    benign_merchants = {e.merchant_id for e in benign}
    recent_merchants = [
        m for m in merchants
        if (SIM_END - m.registered_at).days < 90
        and m.merchant_id in benign_merchants
    ]
    holdout_principals = [p for p in principals
                          if splits[p.principal_id] == "holdout"]

    n_attacks = int(len(benign) * ATTACK_RATE / (1 - ATTACK_RATE))
    targets = {r: int(n_attacks * w) for r, w in ROUTE_SPLIT.items()}

    events, labels = [], []

    def window_for(mandate):
        lo = mandate.created_at + timedelta(days=7)
        hi = mandate_end[mandate.mandate_id]
        return (lo, hi) if hi > lo else None

    # ---------- Route A: prompt injection hijack ----------
    made = 0
    while made < targets[AttackRoute.A_INJECTION]:
        p = rng.choice(principals)
        mandate = mandate_by_principal[p.principal_id]
        win = window_for(mandate)
        if not win:
            continue
        lo, hi = win

        outside = [m for m in merchants
                   if m.merchant_id not in mandate.merchant_scope
                   and m.merchant_id in benign_merchants]
        if not outside:
            continue

        subtle = rng.random() < 0.5
        merchant = rng.choice(outside)
        median_paise, sigma = SPEND_PARAMS[p.spend_profile]
        amount = _draw_amount_paise(median_paise, sigma, rng)
        if not subtle:
            amount = int(amount * rng.uniform(2.0, 5.0))

        span = (hi - lo).total_seconds()
        ts = (lo + timedelta(seconds=rng.uniform(0, span))).replace(microsecond=0)

        ev = _make_event(ts, p, mandate, merchant, amount,
                         _pick_weighted(A_INSTRUCTION_MIX, rng), rng)
        events.append(ev)
        labels.append(Label(ev.event_id, True, AttackRoute.A_INJECTION,
                            "subtle" if subtle else "blatant"))
        made += 1

    # ---------- Route B: delegation abuse ----------
    # Deliberately COMPLIANT: merchant in scope, total under the reserve.
    # If it breached the mandate, hard rules would catch it and the ML
    # layer would have nothing to do (Section 9.6.2).
    b_burst_target = int(targets[AttackRoute.B_DELEGATION_ABUSE]
                         * (1 - P_STALE_VARIANT))
    made = 0
    while made < b_burst_target:
        p = rng.choice(principals)
        mandate = mandate_by_principal[p.principal_id]
        win = window_for(mandate)
        if not win:
            continue
        lo, hi = win

        burst = rng.randint(*B_BURST_SIZE)
        budget = int(mandate.reserve_limit_paise * rng.uniform(0.30, 0.55))
        per_txn = max(1000, budget // burst)

        span = (hi - lo).total_seconds()
        start = lo + timedelta(seconds=rng.uniform(0, max(span - 7200, 1)))

        if rng.random() < P_B_OFF_HOURS:
            b_hour = rng.choice([1, 2, 3, 4, 23])
        else:
            b_hour = rng.randint(p.active_hour_start, p.active_hour_end)
        start = start.replace(hour=b_hour,
                              minute=rng.randint(0, 59), microsecond=0)

        for _ in range(burst):
            merchant = merchant_by_id[rng.choice(mandate.merchant_scope)]
            amount = int(per_txn * rng.uniform(0.75, 1.25))
            ts = start + timedelta(minutes=rng.randint(0, 55))
            ev = _make_event(ts, p, mandate, merchant, amount,
                             _pick_weighted(B_INSTRUCTION_MIX, rng), rng)
            events.append(ev)
            labels.append(Label(ev.event_id, True,
                                AttackRoute.B_DELEGATION_ABUSE, "burst_drain"))
            made += 1

    # ---------- Route B variant: stale credential use ----------
    # Transactions AFTER the mandate was revoked or expired. Realistic
    # (an attacker reusing a dead credential) and it makes hard rules
    # H1 and H2 reachable. Caught by policy, not by the model -- that is
    # the correct outcome and it is what H1/H2 exist for.
    stale_target = max(int(targets[AttackRoute.B_DELEGATION_ABUSE]
                           * P_STALE_VARIANT), 10)
    made = 0
    guard = 0
    dead_mandates = [
        m for m in mandates
        if m.revoked_at is not None or m.expires_at < SIM_END
    ]
    while made < stale_target and dead_mandates and guard < 100_000:
        guard += 1
        mandate = rng.choice(dead_mandates)
        p = principal_by_id[mandate.principal_id]
        dead_from = mandate.revoked_at or mandate.expires_at
        if dead_from >= SIM_END:
            continue
        span = (SIM_END - dead_from).total_seconds()
        if span <= 60:
            continue
        ts = (dead_from + timedelta(seconds=rng.uniform(60, span))).replace(
            microsecond=0)
        merchant = merchant_by_id[rng.choice(mandate.merchant_scope)]
        median_paise, sigma = SPEND_PARAMS[p.spend_profile]
        amount = _draw_amount_paise(median_paise, sigma, rng)
        ev = _make_event(ts, p, mandate, merchant, amount,
                         _pick_weighted(B_INSTRUCTION_MIX, rng), rng)
        events.append(ev)
        labels.append(Label(ev.event_id, True,
                            AttackRoute.B_DELEGATION_ABUSE, "stale_credential"))
        made += 1

    # ---------- Route C: counterfeit merchant (HELD-OUT CONTROL) ----------
    made = 0
    guard = 0
    while (made < targets[AttackRoute.C_COUNTERFEIT_MERCHANT]
           and recent_merchants and holdout_principals and guard < 100_000):
        guard += 1
        p = rng.choice(holdout_principals)
        mandate = mandate_by_principal[p.principal_id]
        win = window_for(mandate)
        if not win:
            continue
        lo, hi = win

        merchant = rng.choice(recent_merchants)
        if merchant.merchant_id in mandate.merchant_scope:
            continue

        median_paise, sigma = SPEND_PARAMS[p.spend_profile]
        amount = int(_draw_amount_paise(median_paise, sigma, rng)
                     * rng.uniform(0.25, 0.5))

        span = (hi - lo).total_seconds()
        ts = (lo + timedelta(seconds=rng.uniform(0, span))).replace(microsecond=0)

        ev = _make_event(ts, p, mandate, merchant, amount,
                         _pick_weighted(BENIGN_INSTRUCTION_MIX, rng), rng)
        events.append(ev)
        labels.append(Label(ev.event_id, True,
                            AttackRoute.C_COUNTERFEIT_MERCHANT, "underpriced"))
        made += 1

    return events, labels


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_outputs(events, labels, principals, agents, mandates,
                  merchants, splits):
    """Write everything to data/. Labels go in a SEPARATE file (L10)."""
    OUT_DIR.mkdir(exist_ok=True)

    def rowify(obj):
        d = asdict(obj)
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
            elif isinstance(v, list):
                # datetimes inside a list need converting too
                d[k] = "|".join(
                    x.isoformat() if isinstance(x, datetime) else str(x)
                    for x in v)
            elif v is None:
                d[k] = ""
        return d

    ev_rows = [rowify(e) for e in events]
    _write_csv(OUT_DIR / "events.csv", ev_rows, list(ev_rows[0].keys()))

    lb_rows = [rowify(l) for l in labels]
    _write_csv(OUT_DIR / "labels.csv", lb_rows, list(lb_rows[0].keys()))

    for name, items in [("principals", principals), ("agents", agents),
                        ("mandates", mandates), ("merchants", merchants)]:
        rows = [rowify(i) for i in items]
        _write_csv(OUT_DIR / f"{name}.csv", rows, list(rows[0].keys()))

    with open(OUT_DIR / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    merchants = make_merchants(rng)
    principals = make_principals(rng)
    splits = assign_splits(principals, rng)
    agents, mandates, mandate_end = make_agents_and_mandates(
        principals, merchants, rng)
    benign = generate_benign_transactions(
        principals, mandates, merchants, mandate_end, rng)
    attacks, labels = generate_attacks(
        principals, mandates, merchants, mandate_end, benign, splits, rng)

    all_events = benign + attacks
    all_events.sort(key=lambda e: e.timestamp)

    benign_labels = [Label(e.event_id, False) for e in benign]
    all_labels = benign_labels + labels

    write_outputs(all_events, all_labels, principals, agents,
                  mandates, merchants, splits)

    # ---------------- diagnostics (Section 9.9) ----------------
    label_by_id = {l.event_id: l for l in all_labels}
    reserve_by_mandate = {m.mandate_id: m.reserve_limit_paise for m in mandates}

    counts = {}
    principals_seen = {}
    variants = {}
    for e in attacks:
        lab = label_by_id[e.event_id]
        r = lab.attack_route
        s = splits[e.principal_id]
        counts[(s, r)] = counts.get((s, r), 0) + 1
        principals_seen.setdefault((s, r), set()).add(e.principal_id)
        variants[lab.attack_variant] = variants.get(lab.attack_variant, 0) + 1

    benign_merchants = {e.merchant_id for e in benign}
    attack_merchants = {e.merchant_id for e in attacks}
    overlap = len(attack_merchants & benign_merchants) / len(attack_merchants)

    consumed = {}
    tripped_b = tripped_total = b_total = 0
    for e in all_events:
        key = (e.mandate_id, e.timestamp.year, e.timestamp.month)
        prior = consumed.get(key, 0)
        breach = prior + e.amount_paise > reserve_by_mandate[e.mandate_id]
        consumed[key] = prior + e.amount_paise
        lab = label_by_id[e.event_id]
        if breach:
            tripped_total += 1
        if (lab.attack_route == AttackRoute.B_DELEGATION_ABUSE
                and lab.attack_variant == "burst_drain"):
            b_total += 1
            if breach:
                tripped_b += 1

    split_sizes = {}
    for s in splits.values():
        split_sizes[s] = split_sizes.get(s, 0) + 1

    n_revoked = sum(1 for m in mandates if m.revoked_at is not None)
    n_expiring = sum(1 for m in mandates if m.expires_at < SIM_END)
    n_confirming = sum(1 for m in mandates if m.confirmation_times)
    n_confirmations = sum(len(m.confirmation_times) for m in mandates)

    print(f"seed              : {args.seed}")
    print(f"total events      : {len(all_events):,}")
    print(f"benign / attack   : {len(benign):,} / {len(attacks):,}")
    print(f"base rate         : {len(attacks) / len(all_events):.2%}")
    print(f"principals        : {split_sizes}")
    print("--- attacks by split (events / principals) ---")
    for s in ["train", "val", "holdout"]:
        parts = []
        for r in AttackRoute:
            n = counts.get((s, r), 0)
            k = len(principals_seen.get((s, r), []))
            parts.append(f"{r.value}={n}/{k}")
        print(f"  {s:8s} {'  '.join(parts)}")
    print("--- attack variants ---")
    for v, n in sorted(variants.items()):
        print(f"  {v:<18}: {n}")
    print("--- integrity checks ---")
    print(f"L1 attack merchants in benign : {overlap:.1%}  (want 100%)")
    print(f"L7 split is by principal      : yes")
    print(f"Route C outside holdout       : "
          f"{sum(counts.get((s, AttackRoute.C_COUNTERFEIT_MERCHANT), 0) for s in ['train','val'])}"
          f"  (want 0)")
    print(f"H3 fires on burst_drain       : {tripped_b}/{b_total} "
          f"= {tripped_b / max(b_total,1):.1%}  (want low)")
    print(f"H3 fires overall              : {tripped_total / len(all_events):.2%}")
    print(f"mandates revoked              : {n_revoked}  (H1 reachable)")
    print(f"mandates expiring in window   : {n_expiring}  (H2 reachable)")
    print(f"mandates with confirmations   : {n_confirming} "
          f"({n_confirmations} total confirmations)")
    print(f"files written to              : {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()