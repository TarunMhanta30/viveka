"""
VIVEKA — synthetic data generator.

Produces agent-mediated UPI transaction data with labelled attacks.
Design: Section 9. Leakage rules: Section 9.9.

Run:  python -m src.generate_data --seed 42
"""

import argparse
import random
import uuid
from datetime import datetime, timedelta

import numpy as np

from src.schema import (
    Principal, Agent, Mandate, Merchant, TransactionEvent,
    SpendProfile, RhythmProfile, AgentType,
    MerchantCategory, VolumeBand, MandateStatus,
    Channel, InstructionSource,
)

# --- Simulation constants (Section 9.7) ---
N_PRINCIPALS = 500
N_MERCHANTS = 200
HISTORY_DAYS = 120
SIM_END = datetime(2026, 8, 1, 0, 0, 0)
SIM_START = SIM_END - timedelta(days=HISTORY_DAYS)

# Spend profile -> (median txn in paise, spread)
SPEND_PARAMS = {
    SpendProfile.LOW:    (40_000,  0.45),
    SpendProfile.MEDIUM: (120_000, 0.50),
    SpendProfile.HIGH:   (300_000, 0.55),
}

# Rhythm -> mean days between transactions
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
# These rates exist to prevent leakage: if only attacks were
# out-of-scope / off-hours / external-content, those fields would be
# perfect label proxies (leakage traps L1, L3).
P_OUT_OF_SCOPE   = 0.05   # legitimate exploration of a new merchant
P_OFF_HOURS      = 0.06   # people do shop at odd times
P_AMOUNT_OUTLIER = 0.03   # a genuinely unusual large purchase
OUTLIER_MULT     = (3.0, 8.0)

BENIGN_INSTRUCTION_MIX = [
    (InstructionSource.SCHEDULED,        0.55),
    (InstructionSource.USER_DIRECT,      0.20),
    (InstructionSource.AGENT_AUTONOMOUS, 0.15),
    (InstructionSource.EXTERNAL_CONTENT, 0.10),
]


def make_merchants(rng: random.Random) -> list[Merchant]:
    """Create the merchant pool.

    Provenance fields (registered_at, reputation_score, volume_band) exist
    because a real system would have them, but features.py never reads
    them -- that is the Route C control (Section 10.6.1).
    """
    merchants = []
    categories = list(MerchantCategory)
    for i in range(N_MERCHANTS):
        # Most merchants are long-established; a minority are recent.
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


def make_principals(rng: random.Random) -> list[Principal]:
    """Create principals with varied spend and rhythm profiles."""
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


def make_agents_and_mandates(
    principals: list[Principal],
    merchants: list[Merchant],
    rng: random.Random,
) -> tuple[list[Agent], list[Mandate]]:
    """One agent and one mandate per principal."""
    agents, mandates = [], []

    for idx, p in enumerate(principals):
        agent_type = rng.choice(list(AgentType))
        allowed_cats = AGENT_CATEGORIES[agent_type]

        # Mandate scope: 2-4 merchants matching the agent's purpose.
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

        agents.append(Agent(
            agent_id=f"A{idx:04d}",
            principal_id=p.principal_id,
            agent_type=agent_type,
            registered_at=registered,
        ))

        # TODO (Pass 3): a fraction of mandates need short expiries and
        # REVOKED status, otherwise hard rules H1/H2 never fire and the
        # dataset only exercises H3/H4. See Section 11.2.1.
        mandates.append(Mandate(
            mandate_id=f"M{idx:04d}",
            principal_id=p.principal_id,
            agent_id=f"A{idx:04d}",
            merchant_scope=[m.merchant_id for m in scope],
            reserve_limit_paise=reserve,
            created_at=created,
            last_confirmed_at=created,
            expires_at=created + timedelta(days=rng.randint(180, 540)),
            status=MandateStatus.ACTIVE,
        ))

    return agents, mandates


def _pick_weighted(options: list[tuple], rng: random.Random):
    """Pick one item from [(value, weight), ...]."""
    values = [o[0] for o in options]
    weights = [o[1] for o in options]
    return rng.choices(values, weights=weights, k=1)[0]


def _draw_amount_paise(median_paise: int, sigma: float, rng: random.Random) -> int:
    """Log-normal draw. Spend is log-normal: most purchases small, a few large.

    np.random.lognormal(mean, sigma) has median exp(mean), so we pass
    log(median_paise) to centre the distribution on the profile's median.
    """
    value = np.random.lognormal(mean=np.log(median_paise), sigma=sigma)
    return max(1000, int(value))   # floor of Rs 10


def generate_benign_transactions(
    principals: list[Principal],
    mandates: list[Mandate],
    merchants: list[Merchant],
    rng: random.Random,
) -> list[TransactionEvent]:
    """Generate normal agent activity for every principal.

    Deliberate messiness (Section 9.5): ~3% amount outliers, ~5%
    out-of-scope merchants, ~6% off-hours. Without these, benign traffic
    is perfectly uniform, any detector scores near-perfect precision,
    and the evaluation means nothing.
    """
    merchant_by_id = {m.merchant_id: m for m in merchants}
    mandate_by_principal = {m.principal_id: m for m in mandates}
    all_merchant_ids = [m.merchant_id for m in merchants]

    events: list[TransactionEvent] = []

    for p in principals:
        mandate = mandate_by_principal[p.principal_id]
        median_paise, sigma = SPEND_PARAMS[p.spend_profile]
        mean_gap = RHYTHM_DAYS[p.rhythm_profile]

        # Habitual merchant weights: one dominates, as real habits do.
        scope = mandate.merchant_scope
        weights = sorted([rng.random() + 0.1 for _ in scope], reverse=True)
        scope_weighted = list(zip(scope, weights))

        current = mandate.created_at + timedelta(hours=rng.randint(2, 72))

        while current < SIM_END:
            # --- merchant ---
            if rng.random() < P_OUT_OF_SCOPE:
                merchant_id = rng.choice(all_merchant_ids)
            else:
                merchant_id = _pick_weighted(scope_weighted, rng)
            merchant = merchant_by_id[merchant_id]

            # --- hour ---
            if rng.random() < P_OFF_HOURS:
                hour = rng.randint(0, 23)
            else:
                hour = rng.randint(p.active_hour_start, p.active_hour_end)

            ts = current.replace(
                hour=hour,
                minute=rng.randint(0, 59),
                second=rng.randint(0, 59),
                microsecond=0,
            )
            if ts >= SIM_END:
                break

            # --- amount ---
            amount = _draw_amount_paise(median_paise, sigma, rng)
            if rng.random() < P_AMOUNT_OUTLIER:
                amount = int(amount * rng.uniform(*OUTLIER_MULT))

            events.append(TransactionEvent(
                event_id=str(uuid.uuid4()),
                timestamp=ts,
                principal_id=p.principal_id,
                agent_id=mandate.agent_id,
                mandate_id=mandate.mandate_id,
                merchant_id=merchant_id,
                amount_paise=amount,
                merchant_category=merchant.category,
                item_count=rng.randint(1, 5),
                channel=Channel.AGENTIC,
                instruction_source=_pick_weighted(BENIGN_INSTRUCTION_MIX, rng),
            ))

            # --- advance to next transaction ---
            gap_days = rng.expovariate(1.0 / mean_gap)
            gap_days = min(gap_days, mean_gap * 5)      # cap absurd gaps
            current = current + timedelta(days=gap_days)

    events.sort(key=lambda e: e.timestamp)
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    merchants = make_merchants(rng)
    principals = make_principals(rng)
    agents, mandates = make_agents_and_mandates(principals, merchants, rng)
    benign = generate_benign_transactions(principals, mandates, merchants, rng)

    # --- diagnostics: measure, do not assume (Section 9.9) ---
    scope_by_mandate = {m.mandate_id: set(m.merchant_scope) for m in mandates}
    reserve_by_mandate = {m.mandate_id: m.reserve_limit_paise for m in mandates}

    out_of_scope = sum(
        1 for e in benign if e.merchant_id not in scope_by_mandate[e.mandate_id]
    )
    ext_content = sum(
        1 for e in benign
        if e.instruction_source == InstructionSource.EXTERNAL_CONTENT
    )

    # Monthly spend per mandate vs its reserve limit.
    monthly: dict[tuple, int] = {}
    for e in benign:
        key = (e.mandate_id, e.timestamp.year, e.timestamp.month)
        monthly[key] = monthly.get(key, 0) + e.amount_paise
    utils = [monthly[k] / reserve_by_mandate[k[0]] for k in monthly]
    breaches = sum(1 for u in utils if u > 1.0)

    per_principal = len(benign) / len(principals)

    print(f"seed                 : {args.seed}")
    print(f"merchants            : {len(merchants)}")
    print(f"principals           : {len(principals)}")
    print(f"benign events        : {len(benign):,}")
    print(f"events per principal : {per_principal:.1f}")
    print(f"out-of-scope rate    : {out_of_scope / len(benign):.1%}")
    print(f"external-content rate: {ext_content / len(benign):.1%}")
    print(f"median utilisation   : {np.median(utils):.2f}")
    print(f"p95 utilisation      : {np.percentile(utils, 95):.2f}")
    print(f"benign H3 breaches   : {breaches / len(monthly):.1%} of mandate-months")


if __name__ == "__main__":
    main()