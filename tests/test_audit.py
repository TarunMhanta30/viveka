"""
Audit trail unit tests (Section 18).

The audit record is EVIDENCE, not a log. It is what a merchant, a
regulator or a dispute process relies on months later, when nobody is
there to explain it.

Three properties must hold or the evidence claim is empty:
  A2 replayable    -- feed it back, get the identical decision
  A3 tamper-evident -- alteration is detectable
  A4 minimal       -- no card-shaped fields, no personal data

The tamper tests matter most. An earlier version of the tamper check
flipped a record's action to "allow" -- but 93% of records were already
"allow", so the test never actually tampered and passed for the wrong
reason (BUGLOG). A test that cannot fail proves nothing.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.audit import (AuditWriter, GENESIS_HASH, RETENTION_DAYS,
                       compute_hash, replay, verify_chain, _clean)
from src.context import EventRow
from src.fusion import Action, Decision
from src.rules import Severity


@pytest.fixture
def writer(tmp_path):
    """tmp_path gives each test a fresh directory, so tests never share
    a chain and cannot pass because of leftover state."""
    return AuditWriter(path=tmp_path / "audit.jsonl",
                       model_version="test-model-v1",
                       model_type="logistic_regression",
                       policy_version="1")


def _ev(i=0, amount=100_00):
    return EventRow(
        event_id=f"EV{i:04d}", timestamp=datetime(2026, 6, 10, 12),
        principal_id="P0001", agent_id="A0001", mandate_id="M0001",
        merchant_id="MER0001", amount_paise=amount,
        merchant_category="grocery", item_count=1,
        channel="agentic", instruction_source="scheduled")


def _decision(action=Action.ALLOW, score=0.12, severity="none",
              fired=None, suppressed=False):
    return Decision(
        recommended_action=action, ml_score=score,
        rule_severity=severity, rules_fired=fired or [],
        reasons=["test reason"],
        thresholds={"t_step": 0.10, "t_block": 0.75, "source": "derived"},
        score_suppressed=suppressed)


def _feats():
    return {"mandate_age_days": 52.4, "utilisation_rate": 0.31,
            "amount_log_zscore": float("nan"), "txn_count_1h": 0.0}


# ---------------------------------------------------------------
# A3 -- tamper evidence
# ---------------------------------------------------------------

def test_chain_intact_when_untouched(writer):
    for i in range(20):
        writer.write(_ev(i), _decision(), _feats(), ["H1", "H2", "H3"], [])
    res = verify_chain(writer.path)
    assert res["ok"] is True
    assert res["checked"] == 20


def test_first_record_links_to_genesis(writer):
    rec = writer.write(_ev(0), _decision(), _feats(), ["H1"], [])
    assert rec.prev_hash == GENESIS_HASH


def test_each_record_links_to_the_previous(writer):
    a = writer.write(_ev(0), _decision(), _feats(), ["H1"], [])
    b = writer.write(_ev(1), _decision(), _feats(), ["H1"], [])
    assert b.prev_hash == a.record_hash


def test_tamper_on_decision_is_detected(writer):
    """Flip a decision to its opposite. The value must genuinely change,
    or the test proves nothing."""
    for i in range(10):
        writer.write(_ev(i), _decision(), _feats(), ["H1"], [])

    lines = writer.path.read_text().splitlines()
    rec = json.loads(lines[5])
    before = rec["recommended_action"]
    rec["recommended_action"] = "block" if before != "block" else "allow"
    assert rec["recommended_action"] != before, "the test did not tamper"
    lines[5] = json.dumps(rec, default=str)

    tampered = writer.path.parent / "tampered.jsonl"
    tampered.write_text("\n".join(lines) + "\n")
    assert verify_chain(tampered)["ok"] is False


def test_tamper_on_one_paise_is_detected(writer):
    """The smallest possible change to a money field."""
    for i in range(10):
        writer.write(_ev(i), _decision(), _feats(), ["H1"], [])

    lines = writer.path.read_text().splitlines()
    rec = json.loads(lines[3])
    rec["amount_paise"] = int(rec["amount_paise"]) + 1
    lines[3] = json.dumps(rec, default=str)

    tampered = writer.path.parent / "tampered.jsonl"
    tampered.write_text("\n".join(lines) + "\n")
    res = verify_chain(tampered)
    assert res["ok"] is False
    assert "line 3" in res["reason"]


def test_deleting_a_record_is_detected(writer):
    """Removing a record breaks the prev_hash link of the next one."""
    for i in range(10):
        writer.write(_ev(i), _decision(), _feats(), ["H1"], [])

    lines = writer.path.read_text().splitlines()
    del lines[4]

    tampered = writer.path.parent / "deleted.jsonl"
    tampered.write_text("\n".join(lines) + "\n")
    assert verify_chain(tampered)["ok"] is False


def test_hash_is_deterministic():
    """Same content must hash identically on any machine and any run,
    or the chain cannot survive a re-verification."""
    rec = {"b": 2, "a": 1, "c": [3, 4], "record_hash": "ignored"}
    same_reordered = {"c": [3, 4], "a": 1, "b": 2, "record_hash": "other"}
    assert compute_hash(rec) == compute_hash(same_reordered)


# ---------------------------------------------------------------
# A2 -- replayability
# ---------------------------------------------------------------

def test_replay_reproduces_allow(writer):
    d = _decision(Action.ALLOW, score=0.05)
    writer.write(_ev(0), d, _feats(), ["H1"], [])
    rec = json.loads(writer.path.read_text().splitlines()[0])
    assert replay(rec) == Action.ALLOW


def test_replay_reproduces_step_up(writer):
    d = _decision(Action.STEP_UP, score=0.50)
    writer.write(_ev(0), d, _feats(), ["H1"], [])
    rec = json.loads(writer.path.read_text().splitlines()[0])
    assert replay(rec) == Action.STEP_UP


def test_replay_reproduces_block_from_rule(writer):
    """A critical rule blocks regardless of a low score. Replay must
    honour the rule floor, not just the threshold."""
    d = _decision(Action.BLOCK, score=0.01, severity="critical",
                  fired=["H2"])
    writer.write(_ev(0), d, _feats(), ["H1", "H2", "H3"], [])
    rec = json.loads(writer.path.read_text().splitlines()[0])
    assert replay(rec) == Action.BLOCK


def test_replay_handles_suppressed_score(writer):
    """Cold start: no score recorded. Replay must not crash on null."""
    d = _decision(Action.ALLOW, score=None, suppressed=True)
    writer.write(_ev(0), d, _feats(), ["H1"], [])
    rec = json.loads(writer.path.read_text().splitlines()[0])
    assert rec["ml_score"] is None
    assert replay(rec) == Action.ALLOW


def test_replay_uses_recorded_thresholds_not_current(writer):
    """The same score means different things under different thresholds.
    Without recording them, a past decision cannot be explained."""
    d = _decision(Action.STEP_UP, score=0.50)
    writer.write(_ev(0), d, _feats(), ["H1"], [])
    rec = json.loads(writer.path.read_text().splitlines()[0])
    assert rec["thresholds"]["t_step"] == 0.10
    # under a stricter policy the same score would have been allowed
    assert replay(rec, {"t_step": 0.9, "t_block": 0.95}) == Action.ALLOW


# ---------------------------------------------------------------
# A4 -- minimality and correctness of stored fields
# ---------------------------------------------------------------

def test_nan_is_stored_as_null_not_zero(writer):
    """A zero z-score means 'exactly typical'. Missing means 'unknown'.
    Writing NaN as 0 would tell a future reader the system saw something
    it did not."""
    writer.write(_ev(0), _decision(), _feats(), ["H1"], [])
    rec = json.loads(writer.path.read_text().splitlines()[0])
    assert rec["features"]["amount_log_zscore"] is None
    assert rec["features"]["txn_count_1h"] == 0.0


def test_clean_converts_nan_only():
    assert _clean(float("nan")) is None
    assert _clean(0.0) == 0.0
    assert _clean(np.float64(1.5)) == 1.5
    assert _clean(None) is None


def test_no_card_shaped_fields(writer):
    """PCI-DSS boundary (rule R1). No 13-19 digit numeric string may
    appear anywhere in the record, even synthetically."""
    import re
    writer.write(_ev(0), _decision(), _feats(), ["H1"], [])
    raw = writer.path.read_text()
    assert re.search(r"\b\d{13,19}\b", raw) is None


def test_no_personal_data_fields(writer):
    """A4 minimality: pseudonymous IDs only, no names or contact data."""
    writer.write(_ev(0), _decision(), _feats(), ["H1"], [])
    rec = json.loads(writer.path.read_text().splitlines()[0])
    for banned in ["name", "email", "phone", "address", "dob"]:
        assert banned not in rec


def test_retention_and_purpose_are_recorded(writer):
    """DPDP Rule 6: one-year retention floor, and purpose limitation
    requires a stated processing purpose on every record."""
    rec_obj = writer.write(_ev(0), _decision(), _feats(), ["H1"], [])
    decided = datetime.fromisoformat(rec_obj.decided_at)
    retain = datetime.fromisoformat(rec_obj.retention_until)
    assert (retain - decided).days == RETENTION_DAYS
    assert len(rec_obj.processing_purpose) > 20


def test_explanation_marked_non_authoritative(writer):
    """The LLM prose is generated after the decision and never affects
    it. If it ever phrases something misleadingly, the authoritative
    record is the structured factors."""
    rec = writer.write(_ev(0), _decision(), _feats(), ["H1"], [],
                       explanation="Looks fine to me")
    assert rec.explanation_authoritative is False


def test_model_and_policy_version_recorded(writer):
    """Without these, a decision cannot be reproduced later."""
    rec = writer.write(_ev(0), _decision(), _feats(), ["H1"], [])
    assert rec.model_version == "test-model-v1"
    assert rec.policy_version == "1"


def test_all_rules_evaluated_are_recorded(writer):
    """A reader must know which rules RAN, not only which fired --
    otherwise they cannot tell a passing rule from an absent one."""
    rec = writer.write(_ev(0), _decision(severity="critical",
                                         fired=["H2"]),
                       _feats(), ["H1", "H2", "H3"], [])
    assert rec.rules_evaluated == ["H1", "H2", "H3"]
    assert rec.rules_fired == ["H2"]