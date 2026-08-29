"""
VIVEKA — audit trail (Section 13).

A LOG IS NOT AN AUDIT RECORD
  A log helps me debug. An audit record is EVIDENCE -- something a
  merchant, a regulator, or a dispute process relies on months later,
  when I am not there to explain it. That difference drives every
  choice in this file.

WHY IT IS A DELIVERABLE, NOT A BYPRODUCT
  ACP's Delegated Payment Spec assigns agentic chargeback liability to
  the merchant and their PSP, and industry participants working on UAP
  named the open problem directly: ensuring all parties have the
  information to review what happened when an agent goes wrong.
  This record is that information (Section 13.2).

FOUR REQUIREMENTS (Section 13.3)
  A1 Complete     -- everything needed to understand the decision
  A2 Replayable   -- feed it back, get the identical decision
  A3 Tamper-evident -- alteration is detectable
  A4 Minimal      -- nothing beyond what the purpose requires (DPDP)

WHY FEATURE VALUES ARE STORED, NOT RECOMPUTED
  The single most important decision here. Features are computed from
  history, and history GROWS. Recomputing next month uses a different
  history and produces different values -- so you would get a different
  decision and wrongly conclude the system is broken. The record must
  be a snapshot of what the system actually SAW, not a recipe for
  approximating it later (Section 13.4.1).

WHAT IS NOT RECORDED (Section 13.7)
  No card-shaped fields, no names or contact details, no demographics,
  no raw merchant content, no duplicated transaction history. Every
  field not stored is a field that cannot be stolen -- data
  minimisation is a breach-mitigation strategy, not only a compliance
  requirement.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

AUDIT_DIR = Path("data/audit")
GENESIS_HASH = "0" * 64

# DPDP Rule 6 sets a one-year log retention floor (Section 6.4.3).
RETENTION_DAYS = 365

# Stated purpose of processing, required under DPDP purpose limitation.
PROCESSING_PURPOSE = (
    "Integrity risk assessment of an agent-initiated payment, to detect "
    "divergence from the principal's authorised intent."
)


@dataclass
class AuditRecord:
    """Evidence for one decision. Field order matters for hashing."""
    audit_id: str
    event_id: str
    decided_at: str

    # pseudonymous references only -- no names, no contact details
    principal_id: str
    agent_id: str
    mandate_id: str
    merchant_id: str
    amount_paise: int

    # what the system SAW
    features: dict

    # layer 1
    rules_evaluated: list
    rules_fired: list
    rule_severity: str

    # layer 2
    model_version: str
    model_type: str
    ml_score: float | None
    score_suppressed: bool

    # decision
    thresholds: dict
    policy_version: str
    recommended_action: str
    top_factors: list
    reasons: list

    # LLM prose. NON-AUTHORITATIVE: generated after the decision and
    # never influencing it. If it ever phrases something misleadingly,
    # the authoritative record is top_factors and features.
    explanation_text: str | None
    explanation_authoritative: bool

    # DPDP
    processing_purpose: str
    retention_until: str

    # tamper evidence
    prev_hash: str
    record_hash: str = ""


def _canonical(record: dict) -> str:
    """Deterministic serialisation for hashing.

    sort_keys and a fixed separator mean the same content always
    produces the same string, on any machine, in any Python version.
    Without that, the hash chain would not survive a re-run.
    """
    payload = {k: v for k, v in record.items() if k != "record_hash"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str)


def compute_hash(record: dict) -> str:
    """SHA256 over the canonical form, including prev_hash.

    Because prev_hash is part of the hashed content, altering any past
    record breaks every hash after it. Detection becomes a linear walk.

    LIMITATION (Section 13.6): this makes tampering DETECTABLE, not
    IMPOSSIBLE. Someone with write access could rebuild the whole
    chain. Real immutability needs append-only storage.
    """
    return hashlib.sha256(_canonical(record).encode()).hexdigest()


def _clean(value):
    """NaN is not valid JSON. Store missing as null, never as 0.

    A zero z-score means 'exactly typical'. A missing z-score means
    'unknown'. Writing NaN as 0 would tell a future reader the system
    saw something it did not (Section 10.8).
    """
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


class AuditWriter:
    """Appends hash-chained records to a JSON Lines file."""

    def __init__(self, path: Path = AUDIT_DIR / "audit.jsonl",
                 policy_version: str = "1", model_version: str = "unknown",
                 model_type: str = "unknown"):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_version = policy_version
        self.model_version = model_version
        self.model_type = model_type
        self._prev_hash = self._last_hash()
        self._count = 0

    def _last_hash(self) -> str:
        """Resume the chain if the file already exists."""
        if not self.path.exists():
            return GENESIS_HASH
        last = None
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    last = line
        if last is None:
            return GENESIS_HASH
        return json.loads(last)["record_hash"]

    def write(self, event, decision, feature_values: dict,
              rules_evaluated: list, top_factors: list,
              explanation: str | None = None) -> AuditRecord:
        decided_at = datetime.now()
        rec = AuditRecord(
            audit_id=f"AUD-{event.event_id}",
            event_id=event.event_id,
            decided_at=decided_at.isoformat(),
            principal_id=event.principal_id,
            agent_id=event.agent_id,
            mandate_id=event.mandate_id,
            merchant_id=event.merchant_id,
            amount_paise=int(event.amount_paise),
            features={k: _clean(v) for k, v in feature_values.items()},
            rules_evaluated=list(rules_evaluated),
            rules_fired=list(decision.rules_fired),
            rule_severity=decision.rule_severity,
            model_version=self.model_version,
            model_type=self.model_type,
            ml_score=_clean(decision.ml_score),
            score_suppressed=bool(decision.score_suppressed),
            thresholds=dict(decision.thresholds),
            policy_version=self.policy_version,
            recommended_action=decision.recommended_action,
            top_factors=list(top_factors),
            reasons=list(decision.reasons),
            explanation_text=explanation,
            explanation_authoritative=False,
            processing_purpose=PROCESSING_PURPOSE,
            retention_until=(decided_at
                             + timedelta(days=RETENTION_DAYS)).isoformat(),
            prev_hash=self._prev_hash,
        )
        d = asdict(rec)
        d["record_hash"] = compute_hash(d)
        rec.record_hash = d["record_hash"]

        with open(self.path, "a") as f:
            f.write(json.dumps(d, default=str) + "\n")

        self._prev_hash = rec.record_hash
        self._count += 1
        return rec

    @property
    def count(self) -> int:
        return self._count


def verify_chain(path: Path = AUDIT_DIR / "audit.jsonl") -> dict:
    """Walk the chain and report the first break, if any.

    A test that never fails proves nothing, so tamper_test() below
    deliberately corrupts a record and confirms this catches it.
    """
    if not path.exists():
        return {"ok": False, "reason": "file not found", "checked": 0}

    prev = GENESIS_HASH
    n = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["prev_hash"] != prev:
                return {"ok": False, "reason": f"prev_hash mismatch at line {i}",
                        "checked": n}
            if compute_hash(rec) != rec["record_hash"]:
                return {"ok": False, "reason": f"content altered at line {i}",
                        "checked": n}
            prev = rec["record_hash"]
            n += 1
    return {"ok": True, "reason": "chain intact", "checked": n}


def replay(record: dict, thresholds_override: dict | None = None):
    """Re-derive the decision from a stored record (requirement A2).

    Uses ONLY what the record contains -- no history lookup, no
    recomputation. That is what makes it a true replay rather than a
    re-evaluation against today's data.
    """
    from src.fusion import Action, MIN_HISTORY_DAYS

    th = thresholds_override or record["thresholds"]
    severity = record["rule_severity"]
    score = record["ml_score"]

    if severity == "critical":
        rule_action = Action.BLOCK
    elif severity == "elevated":
        rule_action = Action.STEP_UP
    else:
        rule_action = Action.ALLOW

    if score is None:
        score_action = Action.ALLOW
    elif score >= th["t_block"]:
        score_action = Action.BLOCK
    elif score >= th["t_step"]:
        score_action = Action.STEP_UP
    else:
        score_action = Action.ALLOW

    order = {Action.ALLOW: 0, Action.STEP_UP: 1, Action.BLOCK: 2}
    return rule_action if order[rule_action] >= order[score_action] \
        else score_action


def main():
    """Self-test: write records, verify the chain, prove replay works,
    and prove tamper detection actually fires."""
    import pickle
    import shutil
    from src.context import Context
    from src.features import FEATURE_NAMES, extract
    from src.model import build_matrix
    from src.fusion import decide, load_thresholds, top_factors
    from src import rules as rules_mod

    # Fresh file each run so the chain check is meaningful.
    if AUDIT_DIR.exists():
        shutil.rmtree(AUDIT_DIR)

    ctx = Context.load()
    df = build_matrix(ctx)
    val = df[df.split == "val"].copy()

    with open("models/model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    model_type = bundle["chosen"]

    thresholds = load_thresholds()
    probs = model.predict_proba(val[FEATURE_NAMES].values)[:, 1]
    val_events = [e for e in ctx.events if ctx.splits[e.principal_id] == "val"]
    assert len(val_events) == len(val), "event/row misalignment"

    writer = AuditWriter(model_version=f"{model_type}-seed42",
                         model_type=model_type, policy_version="1")

    # Write a sample rather than all 7,826 -- enough to prove the
    # mechanism, small enough to inspect by hand.
    sample = 500
    rule_names = ["H1", "H2", "H3", "H4"]
    written = []

    for i in range(sample):
        ev = val_events[i]
        rr = rules_mod.evaluate(ev, ctx)
        feats = extract(ev, ctx)
        hist = feats["principal_history_days"]
        hist = 0.0 if (hist is None or np.isnan(hist)) else float(hist)
        d = decide(rr, float(probs[i]), hist, thresholds)
        rec = writer.write(ev, d, feats, rule_names, top_factors(feats))
        written.append((rec, d.recommended_action))

    size_kb = writer.path.stat().st_size / 1024
    print(f"records written : {writer.count:,}")
    print(f"file size       : {size_kb:,.0f} KB "
          f"({size_kb * 1024 / writer.count:,.0f} bytes/record)")

    # --- A3: chain intact ---
    res = verify_chain()
    print(f"\nchain verify    : {res['reason']}  "
          f"({res['checked']:,} records)")

    # --- A2: replay reproduces the decision ---
    mismatches = 0
    with open(writer.path) as f:
        for line, (_, action) in zip(f, written):
            if replay(json.loads(line)) != action:
                mismatches += 1
    print(f"replay mismatches: {mismatches}  (want 0)")

    # --- A3 proven: tamper detection must actually fire ---
    tampered = AUDIT_DIR / "tampered.jsonl"
    lines = writer.path.read_text().splitlines()
    rec = json.loads(lines[10])
    before = rec["recommended_action"]
    # Flip to a DIFFERENT action. An earlier version hardcoded "allow",
    # which was already the value on most records, so the tamper was a
    # no-op and the test passed without testing anything (BUGLOG).
    rec["recommended_action"] = "block" if before != "block" else "allow"
    assert rec["recommended_action"] != before, "tamper test did not tamper"
    lines[10] = json.dumps(rec, default=str)
    tampered.write_text("\n".join(lines) + "\n")
    bad = verify_chain(tampered)
    print(f"tamper test     : {'DETECTED' if not bad['ok'] else 'MISSED'} "
          f"-- {bad['reason']}")
    print(f"  (flipped record 10 from '{before}' to "
          f"'{rec['recommended_action']}')")

    # Second test: alter an amount rather than a decision.
    tampered2 = AUDIT_DIR / "tampered2.jsonl"
    lines2 = writer.path.read_text().splitlines()
    rec2 = json.loads(lines2[250])
    rec2["amount_paise"] = int(rec2["amount_paise"]) + 1
    lines2[250] = json.dumps(rec2, default=str)
    tampered2.write_text("\n".join(lines2) + "\n")
    bad2 = verify_chain(tampered2)
    print(f"tamper test 2   : {'DETECTED' if not bad2['ok'] else 'MISSED'} "
          f"-- {bad2['reason']}  (amount +1 paise)")

    # --- a real record, printed ---
    first_block = next((r for r, a in written if a == "block"), None)
    if first_block:
        print(f"\nsample BLOCK record:")
        print(f"  event      : {first_block.event_id[:8]}...")
        print(f"  action     : {first_block.recommended_action}")
        print(f"  severity   : {first_block.rule_severity}")
        print(f"  rules fired: {first_block.rules_fired}")
        print(f"  ml score   : {first_block.ml_score}")
        print(f"  top factors: {first_block.top_factors}")
        print(f"  reasons    : {first_block.reasons[:2]}")
        print(f"  retention  : {first_block.retention_until[:10]}")


if __name__ == "__main__":
    main()