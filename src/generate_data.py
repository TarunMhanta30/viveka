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
    Principal, Agent, Mandate, Merchant,
    SpendProfile, RhythmProfile, AgentType,
    MerchantCategory, VolumeBand, MandateStatus,
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
        # Sizing it against a single transaction would make benign users
        # breach their own limit and fire hard rule H3 constantly.
        median_spend, _ = SPEND_PARAMS[p.spend_profile]
        mean_gap_days = RHYTHM_DAYS[p.rhythm_profile]
        expected_monthly_txns = 30.0 / mean_gap_days
        expected_monthly_spend = median_spend * expected_monthly_txns
        reserve = int(expected_monthly_spend * rng.uniform(1.4, 2.5))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    merchants = make_merchants(rng)
    principals = make_principals(rng)
    agents, mandates = make_agents_and_mandates(principals, merchants, rng)

    print(f"seed              : {args.seed}")
    print(f"merchants         : {len(merchants)}")
    print(f"principals        : {len(principals)}")
    print(f"agents            : {len(agents)}")
    print(f"mandates          : {len(mandates)}")
    print(f"sample scope size : {len(mandates[0].merchant_scope)}")
    print(f"sample reserve    : Rs {mandates[0].reserve_limit_paise / 100:,.2f}")


if __name__ == "__main__":
    main()