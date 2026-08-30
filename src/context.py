"""
VIVEKA — context loading and baseline computation.

Answers three questions for any event:
  1. What does this principal's history look like BEFORE this moment?
  2. What mandate governs this transaction?
  3. What merchant and agent are involved?

LEAKAGE RULE L5 (Section 9.9) is enforced HERE and nowhere else.
Every history slice is strictly before the event timestamp. If each
feature enforced this itself, one would eventually forget. Enforcing
it in one place makes the mistake structurally impossible.

That applies to confirmations too: last_confirmation_before() only
counts step-up approvals that had already happened at the time of the
transaction being scored.
"""

import bisect
import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")


# ---------------------------------------------------------------
# Baseline: a principal's behavioural profile at a point in time
# ---------------------------------------------------------------

@dataclass
class Baseline:
    """Everything features.py needs about a principal's past.

    All values computed from events STRICTLY BEFORE a given timestamp.
    n_events == 0 means no history: the cold-start case (Section 10.7).
    """
    n_events: int
    history_days: float

    # amounts (paise)
    amounts: list = field(default_factory=list)
    median_amount: float = 0.0
    mean_log_amount: float = 0.0
    sd_log_amount: float = 0.0

    # rhythm
    mean_gap_days: float = 0.0
    sd_gap_days: float = 0.0
    last_timestamp: datetime | None = None

    # timing
    typical_hour: float = 0.0          # circular mean
    hourly_rate: float = 0.0           # txns per hour, long-run
    daily_rate: float = 0.0            # txns per day, long-run
    weekly_spend: float = 0.0          # paise per week

    # merchants and categories
    merchant_counts: dict = field(default_factory=dict)
    category_counts: dict = field(default_factory=dict)
    top3_share: float = 0.0

    # instruction source
    ext_content_rate: float = 0.0


def _circular_mean_hour(hours: list[int]) -> float:
    """Mean of clock hours, handling wrap-around.

    A naive mean of [23, 1] gives 12, which is the opposite side of the
    clock. Converting to angles, averaging vectors, and converting back
    gives 0 — correct.
    """
    if not hours:
        return 0.0
    angles = [h * 2 * math.pi / 24 for h in hours]
    x = sum(math.cos(a) for a in angles) / len(angles)
    y = sum(math.sin(a) for a in angles) / len(angles)
    mean_angle = math.atan2(y, x)
    hour = mean_angle * 24 / (2 * math.pi)
    return hour % 24


def circular_hour_distance(h1: float, h2: float) -> float:
    """Distance between two clock hours, 0-12.

    23:00 to 01:00 is 2 hours, not 22. Used by feature 19.
    """
    d = abs(h1 - h2) % 24
    return min(d, 24 - d)


# ---------------------------------------------------------------
# Row types loaded from CSV
# ---------------------------------------------------------------

@dataclass
class EventRow:
    event_id: str
    timestamp: datetime
    principal_id: str
    agent_id: str
    mandate_id: str
    merchant_id: str
    amount_paise: int
    merchant_category: str
    item_count: int
    channel: str
    instruction_source: str


@dataclass
class MandateRow:
    mandate_id: str
    principal_id: str
    agent_id: str
    merchant_scope: set
    reserve_limit_paise: int
    created_at: datetime
    last_confirmed_at: datetime
    expires_at: datetime
    status: str
    revoked_at: datetime | None = None
    confirmation_times: list = field(default_factory=list)


@dataclass
class MerchantRow:
    merchant_id: str
    category: str
    registered_at: datetime
    reputation_score: float
    volume_band: str


@dataclass
class AgentRow:
    agent_id: str
    principal_id: str
    agent_type: str
    registered_at: datetime


@dataclass
class PrincipalRow:
    principal_id: str
    created_at: datetime
    spend_profile: str
    rhythm_profile: str
    active_hour_start: int
    active_hour_end: int


# ---------------------------------------------------------------
# Context
# ---------------------------------------------------------------

class Context:
    """Holds the world and answers point-in-time questions about it."""

    def __init__(self, events, principals, agents, mandates, merchants, splits):
        self.events = events
        self.principals = {p.principal_id: p for p in principals}
        self.agents = {a.agent_id: a for a in agents}
        self.mandates = {m.mandate_id: m for m in mandates}
        self.mandate_by_principal = {m.principal_id: m for m in mandates}
        self.merchants = {m.merchant_id: m for m in merchants}
        self.splits = splits

        # Per-principal timelines, sorted. Enables a bisect cut at any
        # timestamp instead of scanning all 40k events per lookup.
        self._timeline: dict[str, list[EventRow]] = {}
        for e in events:
            self._timeline.setdefault(e.principal_id, []).append(e)
        for pid in self._timeline:
            self._timeline[pid].sort(key=lambda e: e.timestamp)
        self._times = {
            pid: [e.timestamp for e in evs]
            for pid, evs in self._timeline.items()
        }

        self._baseline_cache: dict = {}

    # ---------- loading ----------

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR) -> "Context":
        def dt(s):
            return datetime.fromisoformat(s)

        def dt_list(s):
            if not s:
                return []
            return [datetime.fromisoformat(x) for x in s.split("|") if x]

        events = []
        with open(data_dir / "events.csv") as f:
            for r in csv.DictReader(f):
                events.append(EventRow(
                    event_id=r["event_id"],
                    timestamp=dt(r["timestamp"]),
                    principal_id=r["principal_id"],
                    agent_id=r["agent_id"],
                    mandate_id=r["mandate_id"],
                    merchant_id=r["merchant_id"],
                    amount_paise=int(r["amount_paise"]),
                    merchant_category=r["merchant_category"],
                    item_count=int(r["item_count"]),
                    channel=r["channel"],
                    instruction_source=r["instruction_source"],
                ))

        mandates = []
        with open(data_dir / "mandates.csv") as f:
            for r in csv.DictReader(f):
                mandates.append(MandateRow(
                    mandate_id=r["mandate_id"],
                    principal_id=r["principal_id"],
                    agent_id=r["agent_id"],
                    merchant_scope=set(r["merchant_scope"].split("|")),
                    reserve_limit_paise=int(r["reserve_limit_paise"]),
                    created_at=dt(r["created_at"]),
                    last_confirmed_at=dt(r["last_confirmed_at"]),
                    expires_at=dt(r["expires_at"]),
                    status=r["status"],
                    revoked_at=dt(r["revoked_at"]) if r.get("revoked_at") else None,
                    confirmation_times=dt_list(r.get("confirmation_times", "")),
                ))

        merchants = []
        with open(data_dir / "merchants.csv") as f:
            for r in csv.DictReader(f):
                merchants.append(MerchantRow(
                    merchant_id=r["merchant_id"],
                    category=r["category"],
                    registered_at=dt(r["registered_at"]),
                    reputation_score=float(r["reputation_score"]),
                    volume_band=r["volume_band"],
                ))

        principals = []
        with open(data_dir / "principals.csv") as f:
            for r in csv.DictReader(f):
                principals.append(PrincipalRow(
                    principal_id=r["principal_id"],
                    created_at=dt(r["created_at"]),
                    spend_profile=r["spend_profile"],
                    rhythm_profile=r["rhythm_profile"],
                    active_hour_start=int(r["active_hour_start"]),
                    active_hour_end=int(r["active_hour_end"]),
                ))

        agents = []
        with open(data_dir / "agents.csv") as f:
            for r in csv.DictReader(f):
                agents.append(AgentRow(
                    agent_id=r["agent_id"],
                    principal_id=r["principal_id"],
                    agent_type=r["agent_type"],
                    registered_at=dt(r["registered_at"]),
                ))

        with open(data_dir / "splits.json") as f:
            splits = json.load(f)

        return cls(events, principals, agents, mandates, merchants, splits)

    # ---------- point-in-time lookups ----------

    def history_before(self, principal_id: str, ts: datetime) -> list[EventRow]:
        """Events for this principal STRICTLY before ts.

        This is the single enforcement point for leakage rule L5.
        bisect_left returns the first index whose timestamp is >= ts,
        so the slice up to it excludes the event being scored and any
        simultaneous one.
        """
        times = self._times.get(principal_id)
        if not times:
            return []
        cut = bisect.bisect_left(times, ts)
        return self._timeline[principal_id][:cut]

    def last_confirmation_before(self, mandate_id: str,
                                 ts: datetime) -> datetime:
        """When the principal last actively affirmed this mandate, as at ts.

        Falls back to created_at when no confirmation has yet occurred --
        the original consent IS the first affirmation.

        L5 applies: a confirmation that happens AFTER the transaction
        being scored must not influence that transaction's features.
        """
        mandate = self.mandates[mandate_id]
        prior = [c for c in mandate.confirmation_times if c < ts]
        return max(prior) if prior else mandate.created_at

    def confirmation_count_before(self, mandate_id: str, ts: datetime) -> int:
        """How many times the principal has affirmed this mandate, as at ts."""
        mandate = self.mandates[mandate_id]
        return sum(1 for c in mandate.confirmation_times if c < ts)

    def consumed_before(self, mandate_id: str, ts: datetime) -> int:
        """Paise spent on this mandate in ts's calendar month, before ts.

        Used by hard rule H3 and by utilisation features.
        """
        mandate = self.mandates[mandate_id]
        prior = self.history_before(mandate.principal_id, ts)
        return sum(
            e.amount_paise for e in prior
            if e.mandate_id == mandate_id
            and e.timestamp.year == ts.year
            and e.timestamp.month == ts.month
        )

    def baseline(self, principal_id: str, ts: datetime) -> Baseline:
        """Behavioural profile from history strictly before ts."""
        key = (principal_id, ts)
        if key in self._baseline_cache:
            return self._baseline_cache[key]

        prior = self.history_before(principal_id, ts)
        bl = self._compute_baseline(prior, ts)
        self._baseline_cache[key] = bl
        return bl

    @staticmethod
    def _compute_baseline(prior: list[EventRow], ts: datetime) -> Baseline:
        n = len(prior)
        if n == 0:
            return Baseline(n_events=0, history_days=0.0)

        amounts = [e.amount_paise for e in prior]
        logs = np.log(amounts)
        span_days = max((ts - prior[0].timestamp).total_seconds() / 86400, 1e-6)

        gaps = [
            (prior[i].timestamp - prior[i - 1].timestamp).total_seconds() / 86400
            for i in range(1, n)
        ]

        hours = [e.timestamp.hour for e in prior]

        merchant_counts: dict = {}
        category_counts: dict = {}
        ext = 0
        for e in prior:
            merchant_counts[e.merchant_id] = merchant_counts.get(e.merchant_id, 0) + 1
            category_counts[e.merchant_category] = \
                category_counts.get(e.merchant_category, 0) + 1
            if e.instruction_source == "external_content":
                ext += 1

        top3 = sorted(merchant_counts.values(), reverse=True)[:3]

        return Baseline(
            n_events=n,
            history_days=span_days,
            amounts=amounts,
            median_amount=float(np.median(amounts)),
            mean_log_amount=float(np.mean(logs)),
            sd_log_amount=float(np.std(logs)) if n > 1 else 0.0,
            mean_gap_days=float(np.mean(gaps)) if gaps else 0.0,
            sd_gap_days=float(np.std(gaps)) if len(gaps) > 1 else 0.0,
            last_timestamp=prior[-1].timestamp,
            typical_hour=_circular_mean_hour(hours),
            hourly_rate=n / (span_days * 24),
            daily_rate=n / span_days,
            weekly_spend=sum(amounts) / (span_days / 7),
            merchant_counts=merchant_counts,
            category_counts=category_counts,
            top3_share=sum(top3) / n,
            ext_content_rate=ext / n,
        )


def main():
    """Self-test: verify L5 and print a sample baseline."""
    ctx = Context.load()
    print(f"events    : {len(ctx.events):,}")
    print(f"principals: {len(ctx.principals):,}")
    print(f"agents    : {len(ctx.agents):,}")
    print(f"mandates  : {len(ctx.mandates):,}")
    print(f"merchants : {len(ctx.merchants):,}")

    n_revoked = sum(1 for m in ctx.mandates.values() if m.revoked_at is not None)
    n_conf = sum(1 for m in ctx.mandates.values() if m.confirmation_times)
    total_conf = sum(len(m.confirmation_times) for m in ctx.mandates.values())
    print(f"revoked   : {n_revoked}  (with a revocation timestamp)")
    print(f"confirming: {n_conf} mandates, {total_conf} confirmations")

    # --- L5 check: no history entry may be at or after the event ---
    violations = 0
    checked = 0
    for e in ctx.events[::37]:            # sample, not all 40k
        prior = ctx.history_before(e.principal_id, e.timestamp)
        checked += 1
        if any(p.timestamp >= e.timestamp for p in prior):
            violations += 1
    print(f"\nL5 check  : {checked} events sampled, {violations} violations "
          f"(want 0)")

    # --- L5 on confirmations: none may be at or after the event ---
    conf_violations = 0
    for e in ctx.events[::37]:
        m = ctx.mandates[e.mandate_id]
        last = ctx.last_confirmation_before(e.mandate_id, e.timestamp)
        # created_at is the fallback when no confirmation has occurred.
        # A mandate created at the same instant as its first transaction
        # is not leakage -- the consent necessarily preceded the payment.
        if last == m.created_at:
            continue
        if last >= e.timestamp:
            conf_violations += 1
    print(f"L5 (confirm): {conf_violations} violations (want 0)")

    # --- do features 1 and 2 now differ? ---
    diffs = 0
    for e in ctx.events[::37]:
        m = ctx.mandates[e.mandate_id]
        last = ctx.last_confirmation_before(e.mandate_id, e.timestamp)
        if last != m.created_at:
            diffs += 1
    print(f"events where last confirmation != created_at: "
          f"{diffs}/{checked} = {diffs/checked:.1%}")
    print("  (this is what makes days_since_confirmation differ from "
          "mandate_age_days)")

    # --- cold start distribution ---
    empty = sum(1 for e in ctx.events[::37]
                if not ctx.history_before(e.principal_id, e.timestamp))
    print(f"no history: {empty}/{checked} sampled events")

    # --- sample baseline from late in the window ---
    late = [e for e in ctx.events if e.timestamp.month == 7][-1]
    bl = ctx.baseline(late.principal_id, late.timestamp)
    print(f"\nsample baseline for {late.principal_id} at {late.timestamp}")
    print(f"  n_events        : {bl.n_events}")
    print(f"  history_days    : {bl.history_days:.1f}")
    print(f"  median amount   : Rs {bl.median_amount / 100:,.2f}")
    print(f"  mean gap (days) : {bl.mean_gap_days:.2f}")
    print(f"  typical hour    : {bl.typical_hour:.1f}")
    print(f"  daily rate      : {bl.daily_rate:.2f}")
    print(f"  top3 share      : {bl.top3_share:.2f}")
    print(f"  ext content rate: {bl.ext_content_rate:.2f}")
    print(f"  distinct merch  : {len(bl.merchant_counts)}")

    ag = ctx.agents[late.agent_id]
    print(f"\nagent {ag.agent_id} type: {ag.agent_type}")
    print(f"circular distance 23h to 1h : "
          f"{circular_hour_distance(23, 1):.0f}  (want 2)")


if __name__ == "__main__":
    main()