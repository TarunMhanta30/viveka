# VIVEKA

**An integrity layer for AI agent-initiated payments on UPI.**

Razorpay AI Buildathon 2026 — Track 02, AI Risk Manager

| | |
|---|---|
| **Live API** | https://viveka.onrender.com/health |
| **Demo console** | https://fluffy-youtiao-f1b79e.netlify.app/ |
| **Video** | https://youtu.be/QlimdxAMOmY |
| **Bug log** | [BUGLOG.md](BUGLOG.md) — 31 documented failures |

> The API sleeps on the free tier. The first request may take up to a minute
> to wake it. Open the health endpoint first, then the demo console.

---

## What this is

**VIVEKA is a scoring service, not a website.**

It is an HTTP API that a payment flow calls inline. It receives one
agent-initiated transaction and returns a decision in about six milliseconds.
That service is the deliverable.

The demo console exists so the service can be seen working. It is roughly ten
percent of the project. It runs no logic of its own — every number on it comes
from the API. Turn the API off and the console shows empty states, because
nothing on it is hardcoded.

The other ninety percent is the engine: data generation, feature extraction,
policy rules, model selection, cost modelling, decision fusion, audit trail,
evaluation, ablation, red teaming, and 54 unit tests. None of that has a visual
form. It is in `src/` and `eval/`, and its findings are in this document and in
the bug log.

---

## The problem

In February 2026, Razorpay and NPCI launched agentic UPI payments. A user
grants an AI agent a spending mandate once. After that, the agent transacts
without a PIN — groceries, cabs, subscriptions.

UPI Reserve Pay answers one question well: **may this agent spend?** It checks
that the mandate is active, the amount is within the reserve, and the merchant
is on the approved list.

Nothing answers the second question: **is it spending on what the user actually
intended?**

Consider a concrete case. Someone hides an instruction inside a product page
an agent reads. The agent buys something the user never asked for. The mandate
is valid. The amount is under the limit. The merchant is approved. Every
authorisation check passes, and the money is gone.

This is not a hypothetical category. It maps directly to **T6, Intent Breaking
and Goal Manipulation**, in the OWASP Top 10 for Agentic Applications published
in December 2025.

### Why this lands on Razorpay specifically

Under the Agentic Commerce Protocol's Delegated Payment Spec, chargeback
liability for agentic transactions sits with the **merchant and the PSP**.
Razorpay is the PSP. A control that reduces intent-divergent transactions
reduces liability that Razorpay currently carries.

### Where VIVEKA sits

Three rungs exist in agent payment trust:

| Rung | Question | Who answers it |
|---|---|---|
| Recognition | Is this agent genuine? | NPCI's Unified Agent Protocol, Visa Trusted Agent Protocol |
| Authorisation | May it spend this? | UPI Reserve Pay, Mastercard Agentic Tokens |
| **Integrity** | **Is it acting on the user's intent?** | **This is thin. VIVEKA is here.** |

VIVEKA does not replace authorisation. It is a second opinion underneath it.

---

## Architecture

Eight stages. The order is the design.

```
  event
    │
    ▼
 ┌─────────────────┐
 │ 01  Ingress     │  validates against a fixed schema
 └────────┬────────┘  money is integer paise, never float rupees
          ▼
 ┌─────────────────┐
 │ 02  Context     │  loads THIS principal's history, strictly before
 └────────┬────────┘  the event timestamp — single leakage barrier
          ▼
 ┌─────────────────┐
 │ 03  Features    │  31 values across 6 groups
 └────────┬────────┘  missing is NaN, never zero
          ▼
 ┌─────────────────┐
 │ 04  Policy      │  H1 revoked · H2 expired · H3 over reserve
 │     rules       │  facts, not probabilities — no model overrules them
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ 05  Model       │  calibrated probability, grey zone only
 └────────┬────────┘  skipped when history < 7 days
          ▼
 ┌─────────────────┐
 │ 06  Decision    │  rules can ESCALATE, never de-escalate
 │     gate        │  thresholds derived from a rupee cost model
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ 07  Explanation │  factor codes ranked by IQR-normalised deviation
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ 08  Audit       │  hash-chained record, replayable
 └────────┬────────┘
          ▼
   recommended_action + factors + audit_id      ~6 ms
```

### Why the order matters

**Context before features.** Every history slice is cut strictly before the
event timestamp, in one place — `Context.history_before()`. If each of the 31
features enforced this itself, one would eventually forget. Enforcing it once
makes the mistake structurally impossible.

**Rules before the model.** A revoked mandate is a fact. Passing it to a model
invites the model to overrule policy, which is indefensible to a merchant and
to a regulator. The rules run first and the model never sees their verdict.

**Rules escalate but never soften.** A critical rule can turn an allow into a
block. A low model score can never turn a rule violation into an allow.

**No language model in the decision path.** The output is a calibrated
probability from logistic regression, not generated text. Remove every LLM from
this system and every decision is identical.

**Fail-open.** If scoring errors, the response is `allow` with `scored: false`.
A control that halts commerce when it breaks does more damage than the fraud it
prevents. Hard rules still apply during an outage.

**It recommends, it does not block.** The field is `recommended_action`. Under
RBI's Authentication Directions 2025, the integrity of authentication is the
**issuer's** responsibility. Claiming block authority would misstate where
responsibility sits.

---

## The 31 features

| Group | Count | What it captures | Targets |
|---|---|---|---|
| A — mandate state | 8 | age, confirmations, scope, utilisation, expiry | Routes A, B |
| B — velocity | 7 | burst counts and rates relative to this user | Route B |
| C — amount | 3 | deviation from this user's own spend distribution | Routes A, B |
| D — temporal | 5 | hour, active-hours breach, dormancy, rhythm | Routes A, B |
| E — merchant relationship | 5 | familiarity with THIS merchant, for THIS user | Route A |
| F — instruction and maturity | 3 | instruction source, cold-start indicator | Route A |

**Everything is principal-relative.** `merchant_new_to_principal` asks "new to
this user", never "newly registered". That distinction is the Route C control,
explained below.

**Deliberately excluded:** merchant age, reputation score, price versus category
median, volume band — all Route C signals. Also no demographics, no location, no
device. The model cannot discriminate on attributes it never receives.

**Top five by permutation importance:** transactions in the last hour,
activity outside the user's normal window, length of history, mandate age, and
whether the merchant is in scope.

---

## Data

No real agentic fraud dataset exists for this problem. The data is synthetic
and generated from a fixed seed, so anyone can reproduce it exactly.

```
39,342 events · 1,200 principals · 200 merchants · 120 days
base rate 2.00%
train 720 principals · validation 240 · holdout 240
```

**Split by principal, never by transaction.** A random transaction split would
put the same person in train and test, so the model would already know their
baseline and recall would be fiction.

### Three attack routes

| Route | Attack | Features built for it? | Where it appears |
|---|---|---|---|
| A | Prompt injection hijack | Yes | all splits |
| B | Delegation abuse — slow drain inside the limit | Yes | all splits |
| **C** | **Counterfeit merchant, underpriced** | **No** | **holdout only** |

**Route C is the methodological control.** It has no engineered features and
never appears in training. Whatever recall it achieves is transfer, not
learning. Placing it in the holdout only means it cannot influence feature
selection or threshold tuning.

### Deliberate messiness

Benign traffic includes 5% out-of-scope merchants, 6% off-hours activity, and
3% amount outliers. Without this, benign behaviour is uniform, any detector
scores near-perfect precision, and the evaluation means nothing.

### Ten leakage rules, enforced and checked

The generator prints integrity diagnostics on every run: attack merchants must
also appear in benign traffic, Route C must be zero outside holdout, splits
must be by principal, and no feature may read a value from at or after the
event being scored. All pass.

---

## Results

The holdout set was scored **once**. Model, thresholds, feature reference, and
benign medians were all frozen before it ran.

### Held out — 240 unseen principals

| Metric | Value | |
|---|---|---|
| Recall | **76.5%** | 88 of 115 attacks caught |
| Precision | 28.9% | of which 163 of 217 are step-ups, not declines |
| Benign blocked | **0.70%** | 54 of 7,730 legitimate transactions |
| Benign stepped up | 2.11% | asked to confirm, not declined |
| PR-AUC | **0.611** | validation was 0.699 |
| ECE | 0.0022 | calibration error |

**On the PR-AUC gap.** Validation 0.699, holdout 0.611 — a real drop of 0.088.
The holdout has 115 attacks, which is a small sample, and some drop is expected.
It is reported rather than smoothed over.

### Recall by attack type — full path versus model alone

| Route | Full decision path | Model score alone |
|---|---|---|
| A — injection | 80.9% | 85.1% |
| B — delegation abuse | 73.5% | 69.1% |
| C — counterfeit merchant | 52.9% | **52.9%** |

**Route C at 52.9% is the generalisation result.** No feature targets it, it
never appeared in training, and the model still catches over half of it by
transfer from the shared out-of-scope property.

What it misses is **underpricing**. Route C's signature is an amount around
1.9 standard deviations *below* the user's normal, and no feature in this model
treats a low amount as suspicious. That gap is by design and it is reported.

### Cost

Thresholds are derived from a rupee cost model, not hand-picked. Four
parameters, three of them assumed:

| Parameter | Value | Source |
|---|---|---|
| Average fraud value | ₹4,266 | **measured** from the dataset |
| Dispute handling | ₹300 | assumed |
| Abandonment rate | 0.15 | assumed |
| Churn penalty | ₹2,000 | assumed |

Sensitivity analysis found that **only abandonment rate moves the answer**.
Doubling it shifts the step-up threshold from 0.10 to 0.40. The churn penalty
has zero effect, because no benign transaction scores above the block
threshold. Of three assumed parameters, one is decision-relevant — and it is
one Razorpay can measure directly.

| Configuration | Cost on holdout |
|---|---|
| Model only | ₹3,95,404 |
| Model and rules | ₹8,42,914 |
| Rules only, no ML | ₹11,73,631 |

**Against a rules-only baseline, the ML layer saves 39%.** That is the answer
to "why not just write rules".

### A finding that argued against my own architecture

The ablation showed the **rule layer more than doubles measured cost**. Rules
alone catch only 3.7% of attacks. On cost, the model alone is cheaper than the
model plus rules.

I kept the rules anyway, and I moved one of them.

**H4** — merchant outside the approved scope — was never a fact of the same
kind as the others. Authority exists; only intent is unclear. It was
miscategorised from the start, and the ablation made the cost of that visible:
it fired on 5.85% of all traffic, mostly legitimate customers trying a new shop.
It now lives in the model as `merchant_in_mandate_scope`, where it is weighed
against 30 other signals. Cost fell 13% and benign step-ups halved from 5.60%
to 2.30%. It is now the fifth most important feature.

**H1, H2 and H3 stayed.** They enforce consent that was withdrawn, expired, or
exceeded. A PSP cannot let a model approve a payment on withdrawn consent at any
price. **The cost model prices money. It does not price permission.**

---

## Red team

Four adversarial attacks against the frozen model, mapped to the OWASP Top 10
for Agentic Applications, December 2025.

| OWASP | Threat | Status | How |
|---|---|---|---|
| T1 | Memory and context poisoning | Partial | Median baselines resist gradual pull; no drift-rate limiting |
| T2 | Tool misuse | Out of scope | VIVEKA scores payments, not tool calls |
| T3 | Privilege compromise | **Covered** | Route B, mandate-state features |
| T4 | Resource overload | Partial | Velocity features detect bursts; VIVEKA itself is not rate limited |
| T5 | Cascading hallucination | Out of scope | No LLM in the scoring path |
| T6 | **Intent breaking and goal manipulation** | **Covered** | The core thesis |
| T7 | Misaligned and deceptive behaviour | Partial | Caught where it changes spend pattern |
| T8 | Repudiation and untraceability | **Covered** | Hash-chained replayable audit |
| T9 | Identity spoofing | Out of scope | Assumed handled upstream by UAP |
| T10 | Supply chain | Out of scope | Seven pinned dependencies |

Out-of-scope entries are boundary decisions with stated reasons, not gaps.

### Findings

**RT1 — adaptive evader.** An attacker who knows the thresholds and scales the
attack down. Detection: 70% at full strength, 53% at half, 37% at 30%.
Degradation is gradual, not a cliff. Halving the theft still gets you caught
half the time — evasion costs the attacker most of what they came for.

**RT2 — memory poisoning.** Inflate the victim's baseline first, then strike.
70% falls to 51.7% at 400% inflation.

**RT3 — mimicry. The worst case.** Study the victim, then spend only at their
familiar merchants at their normal hours. 70% falls to 50%. Every contextual
signal is fakeable. What survives is velocity and reserve utilisation — the
things an attacker cannot fake while still taking the money.

**RT4 — threshold probing.** The decision boundary was located in **eight
queries**. Withholding the raw score raises the cost from one query to eight,
not to infinity. **Rate limiting is the correct mitigation and it is not
implemented.**

**Nothing was remediated.** Fixing a weakness immediately after measuring it
invalidates the measurement.

---

## Why the API never returns a risk score

`/score` returns `recommended_action`, factor codes, and an audit id. It does
not return the probability.

A score is a gradient. Given gradients, an attacker maps the decision boundary
far more cheaply. RT4 measured exactly this.

An early version withheld the score field and then printed
`"Risk score 0.76 at or above block threshold 0.75"` in the reason text beside
it — returning the cost to one query. Outbound reason text is now stripped of
numeric values. The audit record keeps the exact figures.

A mitigation applied to one field and not to the text beside it is not a
mitigation.

---

## Audit trail

Every decision writes a record. Records are hash-chained: each contains the
hash of the one before it.

- **Tamper-evident.** Altering an amount by one paise breaks the chain and is
  detected. So is deleting a record entirely.
- **Replayable.** Feed a record back and the identical decision comes out.
  Thresholds are stored *in* the record, because the same score means different
  things under different policies.
- **Minimal.** Pseudonymous IDs only. No names, no contact data. A test asserts
  no card-shaped number appears anywhere.
- **DPDP-aligned.** Retention window and stated processing purpose on every
  record.
- **Missing is null, never zero.** A zero z-score means "exactly typical". Null
  means "unknown". Conflating them would tell a future reader the system saw
  something it did not.

---

## API

Base URL: `https://viveka.onrender.com`

```
GET  /health                    service status, model version, feature count
GET  /principals?limit=25       users with mandates, for the demo
GET  /merchants?limit=40        merchant list
POST /score                     the production endpoint
POST /explain                   same decision plus the full feature vector
GET  /audit/{audit_id}          retrieve a record and verify the chain
POST /simulate-outage           demo only — break the scorer deliberately
GET  /results/summary           headline metrics from the evaluation
GET  /results/holdout           full held-out evaluation
GET  /results/ablation          layer and feature ablation
GET  /results/redteam           four attacks, OWASP mapping
```

### Try it

```bash
curl -s https://viveka.onrender.com/health

curl -s -X POST https://viveka.onrender.com/score \
  -H "Content-Type: application/json" \
  -d '{"principal_id":"P0001","mandate_id":"M0001",
       "merchant_id":"MER0052","amount_rupees":1200}'
```

Returns:

```json
{
  "recommended_action": "allow",
  "factors": ["merchant_in_mandate_scope", "txn_count_1h", ...],
  "reasons": ["..."],
  "rules_fired": [],
  "rule_severity": "none",
  "scored": true,
  "audit_id": "AUD-API-...",
  "latency_ms": 5.87
}
```

### How Razorpay would integrate it

One POST call, inline in the payment flow, before the transaction is
authorised. The response is a signal, not a verdict — the issuer decides. It
adds roughly six milliseconds. If VIVEKA is unreachable, the caller treats it as
`allow` and continues, matching the service's own fail-open behaviour.

---

## Running it locally

```bash
git clone https://github.com/TarunMhanta30/viveka.git
cd viveka
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.api          # serves on http://127.0.0.1:8000
```

Models, thresholds, and evaluation results are committed, so the service starts
in about 18 seconds without recomputing anything.

To reproduce the full pipeline from scratch:

```bash
python -m src.generate_data --seed 42
python -m src.model --seed 42
python -m eval.build_reference
python -m eval.cost_model
python -m eval.evaluate          # spends the holdout — once
python -m eval.ablation
python -m eval.redteam
pytest tests/ -v                 # 54 tests
```

---

## Repository

```
src/
  schema.py          entities; money as integer paise
  generate_data.py   synthetic data, 10 leakage checks
  context.py         point-in-time history — the leakage barrier
  features.py        31 features in 6 groups
  rules.py           H1, H2, H3 — time-aware
  model.py           LR vs GB, decision rule written before training
  fusion.py          rule floor + score, fail-open
  audit.py           hash chaining, replay, tamper detection
  pipeline.py        end to end
  api.py             the HTTP service
eval/
  cost_model.py      thresholds from rupee cost
  build_reference.py benign medians for factor attribution
  evaluate.py        held-out evaluation
  ablation.py        layer and feature ablation
  redteam.py         four adversarial attacks, OWASP mapping
tests/               54 unit tests
BUGLOG.md            31 documented failures
demo/index.html      the console
```

---

## Testing

54 tests, all passing, in 0.3 seconds.

They test **behaviour that would fail silently**, not that functions return
numbers. Examples:

- The circular hour mean handles midnight wrap. A test caught it returning
  **24.0** — not a valid hour — for users whose average transaction time lands
  at midnight. That fed `hour_deviation` and was invisible in summary statistics.
- `history_before()` excludes events at the exact same timestamp, not just
  earlier ones.
- H1 does **not** fire before revocation took effect. This is a regression test:
  an earlier version flagged 682 legitimate transactions.
- Tampering with one paise is detected. So is deleting a record.
- A source-level test asserts `features.py` never reads merchant provenance —
  if it did, the Route C control would be invalid.

---

## What broke

**[BUGLOG.md](BUGLOG.md) — 31 entries, dated, with causes.**

Six are cases where something **passed for the wrong reason**:

- The **tamper test passed without tampering.** It set an action to `"allow"` on
  a record that was already `"allow"`. 93% of records are allow, so the tamper
  was a no-op and the hash matched because nothing changed. I nearly shipped a
  tamper-evidence claim backed by a test that could not fail.
- **Route C at 100% was not generalisation.** I claimed it, then found the hard
  rule was catching every one of them. Isolating the model dropped it to 38.4%,
  and after the H4 change, 52.9%. A per-route recall table that mixes rule and
  model decisions cannot tell you what the model did.
- **The cost model optimised a path the system does not use.** It swept
  thresholds on the ML score alone while the deployed path applies a rule floor.
  Fixing it raised reported cost from ₹121k to ₹510k. The old number was
  comfortable and wrong.
- **`instruction_source` was a 3.5× label proxy.** Benign rows used all the
  values, as the leakage rule required — but attacks used them *exclusively*. I
  had implemented one half of a two-sided rule.
- **H1 fired on 682 legitimate transactions.** The schema had a status field
  with no revocation timestamp, so "revoked" was true for all time.
- **The API leaked the score in prose** while carefully omitting it from the
  response field.

Three entries are cases where a measurement **contradicted a design decision I
had already defended**. In two, the design changed. In one it did not, because
cost is not the only thing a payment control optimises for.

---

## Honest limitations

1. **The data is synthetic.** No real agentic fraud dataset exists for this
   problem yet. Every number here describes behaviour on generated data.
2. **Three of four cost parameters are assumed.** Only one is
   decision-relevant, and Razorpay can measure it directly.
3. **Route A is likely optimistic.** The injected merchant is drawn uniformly
   from out-of-scope merchants, so it usually mismatches the agent's category
   too. Real injection would redirect to a more plausible merchant.
4. **Factor attribution is a heuristic, not attribution.** Six binary features
   have a benign IQR of zero, so any firing binary feature ranks near the top.
   Real per-decision attribution needs SHAP.
5. **Nine features have negative permutation importance.** Logistic regression
   is fitting slight noise on them — a known cost of choosing the interpretable
   model.
6. **Rate limiting is not implemented.** RT4 located the boundary in eight
   queries. This is the most important production gap.
7. **9.7% cold-start coverage gap.** Users with under 7 days of history get
   rules only; the model score is suppressed.
8. **Input spoofing sits above the trust boundary.** If the payload itself is
   forged upstream, VIVEKA scores the forgery.
9. **No production security.** No auth, no TLS beyond the platform's, local
   file storage, wide-open CORS for the demo.
10. **`days_since_confirmation` contributes nothing** even after the fix that
    made it mathematically distinct from mandate age. Reported as a fix that
    did not matter.

---

## Where AI was used, and where it was not

**Used:** design discussion, code review, and catching errors in reasoning —
including several of the bugs above. Claude Design built the demo console's
visual layer against an API contract I specified.

**Not used:** the model makes no LLM calls. The decision path is a calibrated
probability, and an LLM explanation layer was scoped in Section 4 and
deliberately never built, because removing it would leave every decision
identical. Right tool, right place — and a clear statement of where the tool
was not the right one.

Every design decision, every threshold, every architectural trade-off, and every
line of the engine is mine, and I can defend each one.

---

## Author

**Tarun Ganesh Mhanta**
MCA, JAIN (Deemed-to-be University), 2028
tarunmhanta30@gmail.com

Built in 11 days for the Razorpay AI Buildathon 2026, Track 02.

---

*Defence only. Synthetic data. No production deployment. The threat model
describes attacks in order to detect them, not to reproduce them.*
