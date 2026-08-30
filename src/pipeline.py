"""
VIVEKA — the pipeline (Section 8).

Wires all eight components into one call. This is the file that makes
the architecture real rather than a diagram, and it is what the demo
runs.

    1 ingress          validate the event
    2 context loader   baseline, mandate, merchant, agent
    3 features         event + context -> 29 numbers
    4 hard rules       deterministic policy gates
    5 ML scorer        calibrated probability (grey zone only)
    6 decision gate    score + rules -> recommended action
    7 explanation      optional LLM prose, never affects the score
    8 audit writer     replayable, hash-chained record

CRITICAL RULES ENCODED HERE
  - A CRITICAL rule violation skips the ML layer entirely. There is
    nothing for a model to judge once policy is definitively broken,
    and running it anyway would invite the model to soften a fact.
  - Fail-open (Section 8.8.2). If scoring errors, the transaction is
    ALLOWED and flagged as unscored. In payments a control that halts
    commerce when it breaks does more damage than the fraud it stops.
  - VIVEKA emits `recommended_action`. It never blocks. Under RBI's
    Authentication Directions, authentication integrity is the
    ISSUER's responsibility (rule R7).
"""

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import rules as rules_mod
from src.rules import RULE_NAMES
from src.audit import AuditWriter
from src.context import Context
from src.features import FEATURE_NAMES, extract, to_vector
from src.fusion import Action, decide, load_thresholds, top_factors
from src.rules import Severity

MODEL_PATH = Path("models/model.pkl")


@dataclass
class PipelineResult:
    """What a caller receives. Note the field name."""
    event_id: str
    recommended_action: str      # NOT "action" -- VIVEKA advises
    factor_codes: list
    reasons: list
    audit_id: str | None
    latency_ms: float
    scored: bool                 # False if fail-open or skipped
    ml_score: float | None       # internal; NOT returned externally


class Viveka:
    """The assembled system."""

    def __init__(self, ctx: Context, model_path: Path = MODEL_PATH,
                 write_audit: bool = True, audit_path: Path | None = None):
        self.ctx = ctx
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.model_type = bundle["chosen"]
        self.model_version = f"{bundle['chosen']}-seed{bundle['seed']}"

        if list(bundle["features"]) != list(FEATURE_NAMES):
            raise RuntimeError(
                "Feature order changed since training. Column drift between "
                "train and predict is a silent, catastrophic failure "
                "(Section 10.8). Retrain before using this model.")

        self.thresholds = load_thresholds()
        self.write_audit = write_audit
        self.audit = AuditWriter(
            path=audit_path or Path("data/audit/pipeline.jsonl"),
            model_version=self.model_version,
            model_type=self.model_type,
            policy_version="1",
        ) if write_audit else None

        self.n_failed_open = 0

    # ---------- step 5, isolated so it can fail safely ----------

    def _score(self, feats: dict) -> float | None:
        vec = to_vector(feats).reshape(1, -1)
        return float(self.model.predict_proba(vec)[0, 1])

    # ---------- the whole path ----------

    def process(self, ev) -> PipelineResult:
        t0 = time.perf_counter()

        # 1 ingress
        if ev.mandate_id not in self.ctx.mandates:
            raise ValueError(f"unknown mandate {ev.mandate_id}")

        # 2 + 3 context and features
        feats = extract(ev, self.ctx)
        hist = feats["principal_history_days"]
        hist = 0.0 if (hist is None or np.isnan(hist)) else float(hist)

        # 4 hard rules
        rr = rules_mod.evaluate(ev, self.ctx)

        # 5 ML scorer -- skipped on a CRITICAL violation, and fail-open
        scored = True
        score = None
        if rr.severity != Severity.CRITICAL:
            try:
                score = self._score(feats)
            except Exception as exc:               # noqa: BLE001
                scored = False
                self.n_failed_open += 1
                rr.reasons.append(f"Scoring unavailable ({type(exc).__name__}); "
                                  f"failing open")
        else:
            scored = False   # deliberately not scored, not a failure

        # 6 decision gate
        d = decide(rr, score, hist, self.thresholds)

        # 7 explanation -- non-blocking, deliberately not implemented yet.
        # When added it receives FACTOR CODES ONLY, never transaction
        # data: no IDs, no amounts, no merchant names. That keeps payment
        # data inside India and out of a third party (Section 15.5).
        explanation = None

        factors = top_factors(feats)

        # 8 audit
        audit_id = None
        if self.audit is not None:
            rec = self.audit.write(ev, d, feats, RULE_NAMES, factors,
                                   explanation)
            audit_id = rec.audit_id

        return PipelineResult(
            event_id=ev.event_id,
            recommended_action=d.recommended_action,
            factor_codes=factors,
            reasons=d.reasons,
            audit_id=audit_id,
            latency_ms=(time.perf_counter() - t0) * 1000,
            scored=scored and score is not None,
            ml_score=d.ml_score,
        )

    def to_external(self, result: PipelineResult) -> dict:
        """What an external caller sees.

        The raw score is DELIBERATELY absent. A score is a gradient;
        given gradients an attacker can map the decision boundary far
        more cheaply (Section 14.3.4). Internal audit keeps the score.
        """
        return {
            "event_id": result.event_id,
            "recommended_action": result.recommended_action,
            "factors": result.factor_codes,
            "audit_id": result.audit_id,
        }


def main():
    """Self-test: run the holdout split end to end, and prove fail-open."""
    import shutil

    audit_path = Path("data/audit/pipeline.jsonl")
    if audit_path.exists():
        audit_path.unlink()

    ctx = Context.load()
    viveka = Viveka(ctx)

    print(f"model      : {viveka.model_version}")
    print(f"thresholds : t1={viveka.thresholds['t_step']:.2f} "
          f"t2={viveka.thresholds['t_block']:.2f} "
          f"[{viveka.thresholds['source']}]")

    events = [e for e in ctx.events
              if ctx.splits[e.principal_id] == "holdout"]
    print(f"\nprocessing {len(events):,} holdout events end to end...")

    t0 = time.perf_counter()
    results = [viveka.process(e) for e in events]
    total_s = time.perf_counter() - t0

    lat = np.array([r.latency_ms for r in results])
    counts = {}
    for r in results:
        counts[r.recommended_action] = counts.get(r.recommended_action, 0) + 1

    print(f"\nlatency  mean {lat.mean():6.2f} ms   "
          f"p50 {np.percentile(lat, 50):6.2f}   "
          f"p95 {np.percentile(lat, 95):6.2f}   "
          f"p99 {np.percentile(lat, 99):6.2f}")
    print("  (modelled, not load-tested -- Section 8.8.1)")
    print(f"total    {total_s:.1f}s for {len(events):,} events")

    print("\naction distribution:")
    for a in [Action.ALLOW, Action.STEP_UP, Action.BLOCK]:
        n = counts.get(a, 0)
        print(f"  {a:<9}: {n:6,}  ({n / len(results):.2%})")

    n_unscored = sum(1 for r in results if not r.scored)
    print(f"\nunscored (critical rule or fail-open): {n_unscored:,}")
    print(f"scoring failures (fail-open fired)   : {viveka.n_failed_open:,}")

    # --- prove fail-open actually works ---
    class BrokenModel:
        def predict_proba(self, X):
            raise RuntimeError("simulated model outage")

    broken = Viveka(ctx, write_audit=False)
    broken.model = BrokenModel()
    sample = [e for e in events if rules_mod.evaluate(e, ctx).severity
              != Severity.CRITICAL][:200]
    out = [broken.process(e) for e in sample]
    allowed = sum(1 for r in out if r.recommended_action == Action.ALLOW)
    print(f"\nfail-open test: model raised on {len(sample)} events")
    print(f"  allowed      : {allowed}/{len(sample)}  "
          f"(payments must not stop when the scorer dies)")
    print(f"  failures seen: {broken.n_failed_open}")

    # --- external response shape ---
    ex = next(r for r in results if r.recommended_action != Action.ALLOW)
    print(f"\nexternal response (note: no raw score):")
    for k, v in viveka.to_external(ex).items():
        print(f"  {k:<20}: {v}")

    # --- end-to-end trace, Section 8.4 ---
    print(f"\nSection 8.4 trace for {ex.event_id[:8]}...:")
    for i, reason in enumerate(ex.reasons, 1):
        print(f"  {i}. {reason}")


if __name__ == "__main__":
    main()