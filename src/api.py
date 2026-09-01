"""
VIVEKA — HTTP API (Section 8, revised).

Turns VIVEKA from a Python library into a SERVICE. A payment platform
does not want a library it must import; it wants an endpoint its
payment flow can call inline and get a decision from.

WHAT THE CALLER RECEIVES
  recommended_action, factor codes, an audit id. NOT the raw score,
  and NOT any threshold value -- including inside reason text.
  A score is a gradient; given gradients an attacker maps the decision
  boundary far more cheaply. RT4 measured this: withholding the score
  raises boundary discovery from 1 query to 8 (Section 14.3.4). An
  earlier version withheld the score field while printing it in prose
  right beside it, which returned the cost to 1 query (BUGLOG).

WHAT VIVEKA DOES NOT DO
  It does not block. The field is `recommended_action`. Under RBI's
  Authentication Directions 2025 the integrity of authentication is the
  ISSUER's responsibility. Claiming block authority would misstate
  where responsibility sits (rule R7).

FAIL-OPEN
  If scoring errors, the response is ALLOW with scored=false. In
  payments, a control that halts commerce when it breaks does more
  damage than the fraud it prevents (Section 8.8.2).

NOT PRODUCTION
  No authentication, no rate limiting, no TLS, no persistence beyond a
  local file. Section 24 lists every one of these as a production gap.
  Rate limiting matters most: RT4 showed the decision boundary is
  locatable in 8 unthrottled queries.
"""

import re
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import rules as rules_mod
from src.context import Context, EventRow
from src.features import FEATURE_NAMES, extract
from src.fusion import Action, decide, load_thresholds, top_factors
from src.pipeline import Viveka

# Loaded once at startup. Loading per request would add ~2s and
# make the latency figures meaningless.
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    ctx = Context.load()
    STATE["ctx"] = ctx
    STATE["viveka"] = Viveka(ctx, write_audit=True,
                             audit_path=Path("data/audit/api.jsonl"))
    STATE["loaded_in"] = time.perf_counter() - t0
    print(f"VIVEKA ready in {STATE['loaded_in']:.1f}s  "
          f"({len(ctx.events):,} events, "
          f"{len(ctx.principals):,} principals)")
    yield
    STATE.clear()


app = FastAPI(
    title="VIVEKA",
    description="Agent Integrity Layer for Delegated Payments",
    version="1.0.0",
    lifespan=lifespan,
)

# The demo page is opened from a file or a different port, so the
# browser blocks the call without this. Wide-open CORS is acceptable
# for a local demo and would NOT be in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------
# request / response
# ---------------------------------------------------------------

class ScoreRequest(BaseModel):
    """One agent-initiated payment to be scored.

    principal_id and mandate_id must exist -- VIVEKA needs the
    behavioural baseline and the consent artifact. A payment with no
    history is the cold-start case, handled in fusion.py.
    """
    principal_id: str = Field(..., examples=["P0001"])
    mandate_id: str = Field(..., examples=["M0001"])
    merchant_id: str = Field(..., examples=["MER0001"])
    amount_rupees: float = Field(..., gt=0, examples=[1250.00])
    timestamp: str | None = Field(
        None, description="ISO 8601. Defaults to the dataset's end date "
                          "so baselines are meaningful.")
    merchant_category: str = Field("grocery")
    item_count: int = Field(1, ge=1)
    instruction_source: str = Field(
        "scheduled",
        description="user_direct | scheduled | agent_autonomous | "
                    "external_content")


class ScoreResponse(BaseModel):
    event_id: str
    recommended_action: str
    factors: list[str]
    reasons: list[str]
    rules_fired: list[str]
    rule_severity: str
    scored: bool
    audit_id: str | None
    latency_ms: float
    # NOTE: ml_score is deliberately absent. See module docstring.


class ExplainResponse(ScoreResponse):
    """Same decision, plus the full feature vector.

    Separate endpoint because this is INTERNAL detail. An external
    caller gets ScoreResponse; an operator debugging a decision gets
    this. The score is still withheld -- the audit record has it.
    """
    features: dict
    thresholds: dict


# ---------------------------------------------------------------
# outbound redaction
# ---------------------------------------------------------------

def _redact(reason: str) -> str:
    """Remove numeric score and threshold values from outbound text.

    The score is withheld from the response field for a measured reason
    (RT4: withholding raises boundary discovery from 1 query to 8). A
    reason string reading "Risk score 0.76 at or above block threshold
    0.75" hands both values back in prose and defeats the mitigation.

    The audit record keeps the exact values. External callers do not.
    """
    if "Risk score" in reason or "threshold" in reason:
        cleaned = re.sub(r"[\d.]+", "", reason)
        return re.sub(r"\s{2,}", " ", cleaned).strip()
    return reason


# ---------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------

@app.get("/health")
def health():
    v = STATE.get("viveka")
    if v is None:
        raise HTTPException(503, "not ready")
    return {
        "status": "ready",
        "model": v.model_version,
        "n_features": len(FEATURE_NAMES),
        "n_rules": len(rules_mod.RULE_NAMES),
        "load_seconds": round(STATE["loaded_in"], 2),
        # Thresholds are NOT returned here either -- same reason as
        # the score. /explain exposes them for operators.
    }


def _build_event(req: ScoreRequest, ctx: Context) -> EventRow:
    if req.mandate_id not in ctx.mandates:
        raise HTTPException(404, f"unknown mandate {req.mandate_id}")
    if req.principal_id not in ctx.principals:
        raise HTTPException(404, f"unknown principal {req.principal_id}")

    mandate = ctx.mandates[req.mandate_id]
    if mandate.principal_id != req.principal_id:
        raise HTTPException(
            400, f"mandate {req.mandate_id} does not belong to "
                 f"{req.principal_id}")

    ts = (datetime.fromisoformat(req.timestamp) if req.timestamp
          else datetime(2026, 8, 1))

    return EventRow(
        event_id=f"API-{int(time.time() * 1000)}",
        timestamp=ts,
        principal_id=req.principal_id,
        agent_id=mandate.agent_id,
        mandate_id=req.mandate_id,
        merchant_id=req.merchant_id,
        amount_paise=int(round(req.amount_rupees * 100)),
        merchant_category=req.merchant_category,
        item_count=req.item_count,
        channel="agentic",
        instruction_source=req.instruction_source,
    )


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """Score one transaction. This is the endpoint a payment flow calls."""
    ctx, viveka = STATE["ctx"], STATE["viveka"]
    ev = _build_event(req, ctx)

    rr = rules_mod.evaluate(ev, ctx)
    r = viveka.process(ev)

    return ScoreResponse(
        event_id=r.event_id,
        recommended_action=r.recommended_action,
        factors=r.factor_codes,
        reasons=[_redact(x) for x in r.reasons],
        rules_fired=rr.fired,
        rule_severity=rr.severity.value,
        scored=r.scored,
        audit_id=r.audit_id,
        latency_ms=round(r.latency_ms, 2),
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ScoreRequest):
    """Same decision with the full feature vector, for an operator.

    Reasons are NOT redacted here and thresholds ARE returned: this is
    the internal view. The raw score is still absent -- the audit
    record is the only place that holds it.
    """
    ctx, viveka = STATE["ctx"], STATE["viveka"]
    ev = _build_event(req, ctx)

    feats = extract(ev, ctx)
    rr = rules_mod.evaluate(ev, ctx)
    r = viveka.process(ev)

    def clean(v):
        if v is None:
            return None
        if isinstance(v, (float, np.floating)):
            return None if np.isnan(v) else round(float(v), 4)
        return v

    return ExplainResponse(
        event_id=r.event_id,
        recommended_action=r.recommended_action,
        factors=r.factor_codes,
        reasons=r.reasons,
        rules_fired=rr.fired,
        rule_severity=rr.severity.value,
        scored=r.scored,
        audit_id=r.audit_id,
        latency_ms=round(r.latency_ms, 2),
        features={k: clean(v) for k, v in feats.items()},
        thresholds=viveka.thresholds,
    )


@app.get("/principals")
def principals(limit: int = 20):
    """List principals with a usable mandate, for the demo UI."""
    ctx = STATE["ctx"]
    out = []
    for pid, p in list(ctx.principals.items())[:limit]:
        m = ctx.mandate_by_principal.get(pid)
        if m is None:
            continue
        out.append({
            "principal_id": pid,
            "mandate_id": m.mandate_id,
            "agent_type": ctx.agents[m.agent_id].agent_type,
            "reserve_rupees": round(m.reserve_limit_paise / 100, 2),
            "merchant_scope": sorted(m.merchant_scope),
            "status": m.status,
            "spend_profile": p.spend_profile,
            "active_hours": [p.active_hour_start, p.active_hour_end],
        })
    return out


@app.get("/merchants")
def merchants(limit: int = 40):
    ctx = STATE["ctx"]
    return [{"merchant_id": m.merchant_id, "category": m.category}
            for m in list(ctx.merchants.values())[:limit]]


@app.get("/audit/{audit_id}")
def audit(audit_id: str):
    """Retrieve a decision record and verify the chain around it.

    This is the endpoint that makes the audit trail real rather than a
    claim: a dispute process can fetch the exact record.
    """
    import json
    from src.audit import verify_chain

    path = Path("data/audit/api.jsonl")
    if not path.exists():
        raise HTTPException(404, "no audit records yet")

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["audit_id"] == audit_id:
                return {"record": rec, "chain": verify_chain(path)}
    raise HTTPException(404, f"no record {audit_id}")


@app.post("/simulate-outage")
def simulate_outage(enable: bool = True):
    """Break the scorer on purpose, to demonstrate fail-open.

    A DEMO-ONLY endpoint. It would never exist in production, and its
    presence here is deliberate: showing the system failing safely is
    more informative than showing it working.
    """
    viveka = STATE["viveka"]

    if enable:
        if "real_model" not in STATE:
            STATE["real_model"] = viveka.model

        class BrokenModel:
            def predict_proba(self, X):
                raise RuntimeError("simulated scorer outage")

        viveka.model = BrokenModel()
        return {"outage": True,
                "message": "Scorer disabled. Transactions will fail open "
                           "-- allowed and flagged as unscored. Hard rules "
                           "still apply."}

    if "real_model" in STATE:
        viveka.model = STATE.pop("real_model")
    return {"outage": False, "message": "Scorer restored."}


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()