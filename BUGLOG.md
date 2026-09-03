# Bug log — VIVEKA

Every failure, in the order it happened, with what caused it and what it
taught. Written as it occurred, not reconstructed afterwards.

Twenty-six entries across eleven days. Most were found by printing a
measurement rather than by something crashing — that is the pattern worth
noticing.

| Category | Count |
|---|---|
| Design errors caught before they reached results | 7 |
| Silent bugs found by measurement | 9 |
| Wrong claims I corrected about my own work | 5 |
| Tests and tooling that were themselves broken | 3 |
| Limitations reported rather than fixed | 2 |

---

## Environment

### 1 — Imports failed inside an active virtualenv
**27 Aug**

**Symptom.** `ModuleNotFoundError: No module named 'numpy'` with `(.venv)`
active and `pip install` having reported success.

**First hypothesis, wrong.** That a bare `pip` was resolving to system
Python. `which pip` disproved it — the venv's own pip was being used.

**Real cause.** WSL is Ubuntu 26.04, whose default Python is 3.14. numpy
2.1.3 publishes no wheel for 3.14, so pip fell back to a Meson source
build, which failed on a missing Python development dependency. The
failure scrolled past and I read only the last line.

**Fix.** Installed Python 3.12 via the deadsnakes PPA, deleted the venv and
rebuilt it. Installation became a wheel download instead of a compile.

**Lesson.** Check wheel availability for your interpreter before pinning
versions. The newest Python is not the safe choice for the scientific
stack — it leads the ecosystem rather than following it.

---

## Data generation

### 2 — Reserve limit sized against the wrong quantity
**27 Aug**

**Symptom.** Caught by inspection before any transaction was written.

**Cause.** The monthly reserve was set to `median_transaction × 8–20`. But
a weekly-rhythm principal makes roughly twelve transactions a month, so
normal spending would exceed their own limit and fire hard rule H3 on
benign traffic — destroying the false-positive baseline before the model
ever ran.

**Fix.** Size the reserve against expected *monthly* spend:
`median × 30 / mean_gap_days × 1.4–2.5`.

**Lesson.** A limit must be sized against the rate of consumption, not the
size of one event.

### 3 — Active-hours range could produce hour 24
**27 Aug**

**Cause.** `active_hour_end = start_hour + randint(8, 13)` with `start_hour`
up to 11 gives a maximum of 24. Hours run 0–23. This would have broken
timestamp construction in the next pass.

**Fix.** `min(23, ...)`.

**Lesson.** A derived time field needs its range checked at both ends, not
only the lower one.

### 4 — Benign traffic breached its own reserve in 14.5% of mandate-months
**27 Aug**

**Symptom.** The generator's diagnostic printed 14.5% of benign
mandate-months exceeding the reserve — meaning H3 would recommend blocking
legitimate customers at that rate.

**Cause.** Entry 2's fix used the *median* transaction. Real mean monthly
spend is about 1.28× higher: a log-normal's mean exceeds its median by
`exp(σ²/2)`, and the 3% amount-outlier injection multiplies those amounts
by 3–8×.

**Fix.** Size against expected (mean) spend including the outlier
contribution, and raise headroom from 1.4–2.5× to 2.0–3.2×. Breach rate
fell to 1.6%.

**Lesson.** Size a limit against the expected value of a distribution, not
a typical value. This was caught only because the generator prints measured
diagnostics rather than trusting its own parameters.

### 5 — Route B concentrated in too few principals to measure
**27 Aug**

**Symptom.** Route B — the most important attack class — appeared in only
16 principals out of 500, because each injection creates a burst of 6–10
events from one person.

**Why it mattered.** The split is by principal, so the held-out set would
carry roughly three Route B principals. Recall measured on three people is
not a measurement.

**Fix.** Reduced burst size and raised Route B's share of the attack budget,
roughly doubling distinct principals carrying it.

**Lesson.** With a principal-level split, statistical power comes from
distinct principals per class per split, not from total event count. Event
counts hid this entirely.

### 6 — Dataset scaled from 500 to 1,200 principals
**27 Aug**

Section 9.7 specified 500 principals and ~35,000 events. After entry 5's
fix, Route B still reached only seven held-out principals. Raised to 1,200,
which gave 21. Burst size reduced further to spread the same event budget
across more people.

**Recorded as a deviation from the design document**, made before any model
was trained and therefore before any result could have influenced it.

### 7 — Side effect: H3 fire rate on Route B rose
**27 Aug**

The burst-size reduction divided the same drain budget across fewer
transactions, so each was larger and more likely to cross the reserve. H3
firing on Route B rose from 0.7% to 5.5%.

**Accepted.** 94.5% of Route B remains invisible to policy, which is the
property the ML layer depends on. Recorded rather than silently tuned away.

### 8 — `instruction_source` was a partial label proxy
**27 Aug**

**Symptom.** A crosstab showed `agent_autonomous` in 53% of attacks versus
15% of benign — a 3.5× lift. Same problem with hour: attack mean 8.6,
benign 13.7.

**Cause.** Route A always used `external_content`; Route B always used
`agent_autonomous` and always drew off-hours. Leakage rule L3 required
benign rows to use these values too, and they did — but I missed the mirror
requirement that attacks must not use them *exclusively*.

**Fix.** Attacks now draw from route-specific weighted mixes, and Route B
uses off-hours only 65% of the time. Lift fell to ~2×.

**Lesson.** Leakage prevention runs both ways. It is not enough for the
benign class to look varied; the attack class must too.

---

## Features

### 9 — Three numerical blow-ups found by printing distributions
**28 Aug**

1. `interarrival_zscore` reached **34,717**. A principal with
   near-simultaneous prior events has a gap standard deviation near zero,
   so the division exploded. Fixed with a denominator floor and a clip
   to ±20.
2. `velocity_ratio_1h` reached **678** for the same reason on hourly rate.
   Clipped to 100.
3. `utilisation_velocity` median is **0.43**, not the ~1.0 claimed in
   Section 10.4. Reserves carry 2–3× headroom by design, so a full month
   of normal spending reaches only ~0.4 utilisation. The feature is sound;
   my stated interpretation of it was wrong, and Section 10 was corrected.

**Lesson.** Gradient boosting would have tolerated these silently. Logistic
regression would not — one exploded row can dominate the fit. Distribution
checks catch what model accuracy hides.

### 10 — The velocity clip compresses real signal, not just outliers
**28 Aug**

Route B's median `velocity_ratio_1h` sits *at* the clip ceiling of 100. The
clip is therefore trimming genuine signal, not only numerical blow-ups.

**Accepted and recorded** so the clip is a stated design choice rather than
a hidden one. The feature still separates cleanly — benign median is 0.00 —
and the alternative, unbounded ratios reaching 678, would destabilise the
model.

### 11 — Factor attribution ranked by magnitude, not by unusualness
**30 Aug**

**Symptom.** A step-up decision driven by an out-of-scope merchant reported
its top factors as `days_to_expiry`, `hour_of_day`, `mandate_age_days` —
none of which had anything to do with the decision.

**Cause.** `top_factors()` sorted by raw absolute value. `days_to_expiry`
(~307) and `mandate_age_days` (~52) topped every record simply for being
large numbers, while `merchant_new_to_principal` at 1.0 never appeared.
Requirement R9 was visibly unmet in every audit record and in the demo.

**Fix.** Rank by deviation from the benign median, scaled by that feature's
own interquartile range, computed on benign training rows only.

**Lesson.** Attribution needs a reference point. "Large" is not "unusual".

### 12 — The attribution fix is still a heuristic, not attribution
**30 Aug**

Six features are binary with a benign IQR of zero, so the spread floor makes
any firing binary feature rank near the top regardless of contribution —
`is_external_content` appeared on a decision it did not drive.

**Labelled as a known limitation rather than presented as attribution.**
Real per-decision attribution requires SHAP on the gradient boosting model.

**Lesson.** Replacing an obviously wrong ranking with a slightly less wrong
one is progress. Calling it attribution would be a false claim.

### 13 — Circular mean returned hour 24
**31 Aug, found by a unit test**

**Symptom.** `_circular_mean_hour([23, 1])` returned **24.0**. Hours are
0–23.

**Cause.** For a principal whose transactions average to midnight, `atan2`
returns a tiny negative angle; `% 24` turns it into 23.999999, which rounds
to 24.0. That value feeds `hour_deviation` through circular distance, so
the deviation for such a principal was wrong by 23 hours — and completely
invisible in summary statistics.

**Fix.** Snap values at or above 23.9999 back to 0.

**Lesson.** The bug was in the one function I had already written a warning
comment about. Knowing a function is tricky is not the same as testing it.

### 14 — The circular mean fix post-dates the held-out evaluation
**31 Aug**

The fix affects only principals whose mean transaction hour falls within
0.0001 of midnight — a handful of rows in 39,342.

**Not re-run.** Spending the held-out set a second time to correct a
rounding artifact would be a worse methodological error than the artifact.
Recorded rather than hidden.

---

## Rules

### 15 — H1 fired on 682 legitimate transactions; H2 never fired at all
**29 Aug**

**Symptom.** The rule fire-rate table showed H1 triggering on 1.71% of all
traffic with **zero** of those being attacks, and H2 at **zero fires**. Two
of four hard rules were broken or dead.

**Cause.** The mandate schema carried a `status` field with no revocation
*timestamp*, so H1 treated "revoked" as true for all time — flagging
transactions that occurred weeks before revocation, when consent was
entirely valid. Separately, transaction generation stopped at the mandate
end date, so nothing ever occurred after expiry and H2 was unreachable.

**Fix.** Added `revoked_at` to the schema, made H1 time-aware, and added a
"stale credential" attack variant — transactions after revocation or expiry
— which is realistic and makes both rules reachable.

**Verified after fix.** H1 false positives 682 → **0**. H2 fires 0 → 21.
Benign critical rate 2.2% → 0.5%.

**Lesson.** A status field without a timestamp is not enough to reason about
time. Rule fire-rate tables catch this; model accuracy never would.

A regression test now exists for this specific bug
(`test_h1_does_not_fire_before_revocation`).

---

## Evaluation and cost

### 16 — Route C being holdout-only inflates that split's base rate
**29 Aug**

Route C is confined to the held-out set by design. That raises the held-out
base rate to ~3.4% against validation's 1.3%, and PR-AUC rises mechanically
with base rate — so the two are not directly comparable.

**Handled** by reporting held-out metrics twice, with and without Route C.

**Lesson.** An experimental control placed in one split changes that split's
class balance. The control is still correct; the comparison needs care.

### 17 — Cost model optimised a decision path the system does not use
**30 Aug**

**Symptom.** The cost model reported 0% of benign transactions blocked. The
deployed path blocked 0.7%.

**Cause.** `cost_model.py` swept thresholds using the ML score alone, while
`fusion.py` applies the rule floor on top. H3 fires on benign reserve
breaches regardless of score.

**Fix.** Sweep candidate thresholds through the full `fusion.decide()` path.

**Consequences, all of them uncomfortable and all reported:**

1. Total cost rose from ₹121k to ₹510k. The first figure was fiction.
2. The step-up threshold moved from 0.10 to 0.45.
3. Saving over hand-picked thresholds collapsed from 10.4% to 1.6–2.5%.
   The cost surface is flat; threshold tuning is not where the value is.
4. A new baseline — rules only, no ML — cost ₹801k against ₹510k with the
   model. **That 36% is where the value actually is,** and it is the answer
   to "why not just write rules".

**Lesson.** Optimising a component in isolation produced a number that
looked good and was wrong. Optimising the whole path produced a
worse-looking number that is true, and revealed where the gain really came
from.

### 18 — Only one of three assumed cost parameters affects the answer
**30 Aug**

Sensitivity analysis showed churn penalty has **zero** effect on the derived
thresholds — ×2 and ×0.5 give identical results — because no benign
transaction scores above the block threshold, so the churn term is never
invoked.

Abandonment rate is the only assumed parameter that moves the answer.

**Useful outcome:** of three assumptions, exactly one is decision-relevant,
and it is one Razorpay can measure directly.

### 19 — Cold start affects 9.7% of traffic
**30 Aug**

Not a bug, but a coverage gap found during the same investigation: 9.7% of
validation events have under seven days of history, so the ML score is
suppressed and only hard rules apply.

### 20 — Route C at 100% recall was not evidence of generalisation
**31 Aug — a claim I had been making that was wrong**

**What I said.** Route C reaching 100% recall showed the model generalising
to an attack class it had never seen.

**What was actually happening.** Route C always uses an out-of-scope
merchant, so hard rule H4 fired on every one and flagged it. The same was
true of Route A at 100%. The rules caught them, not the model.

**Correction.** Isolating model-only recall changed the reading of every
route:

| | Full path | Model only |
|---|---|---|
| Route A | 100% | 50.8% |
| Route B | 75.6% | 73.2% |
| Route C | 100% | **38.4%** |

Route C generalisation is 38.4%, not 100%. Also revealed: model-only benign
false positive rate is 0.7% against 5.6% step-up on the full path — most
false positives came from H4 firing on legitimate new-merchant purchases,
not from the model.

**Lesson.** Aggregate metrics on a layered system hide which layer is
working. A per-route table that mixes rule and model decisions cannot tell
you what the model did.

### 21 — The ablation argued against my own architecture
**31 Aug**

**Finding.** On the held-out set, model-only costs ₹395,404. Model plus
rules costs ₹842,914. The rule layer more than doubles cost and catches
only 3.7% of attacks on its own.

**Decision: keep the rules anyway.** H1, H2 and H3 enforce consent that was
withdrawn, expired or exceeded. A payment service provider cannot delegate
that to a model at any price. The cost model prices money; it does not
price permission.

**But H4 was different.** H4 — merchant outside the approved list — was
never a policy fact. Authority exists; only intent is unclear. It was
miscategorised from the start, and the ablation made the cost of that
visible. Moved into the model as a feature, where it is weighed against 30
other signals instead of forcing a step-up on 5.85% of traffic.

**Result of that change:** total cost ₹526,932 → ₹458,765. Benign step-up
rate 5.6% → 2.3%. The feature is now the fifth most important in the model.

**Lesson.** An ablation that validates everything you built is an ablation
that was not testing anything.

### 22 — Two features contribute nothing, and one still doesn't after being fixed
**31 Aug**

`days_since_confirmation` had zero permutation importance because it was
mathematically identical to `mandate_age_days` — no step-up confirmations
occurred in the simulation, so the timestamp never advanced.

**Fix.** Added simulated confirmation events. Correlation fell from 1.0000
to 0.66, so the two features are now genuinely different.

**The fix did not matter.** Permutation importance remains zero. Logistic
regression already extracts that signal from mandate age.

**Reported as a fix that worked mechanically and failed to pay off.**

Separately: nine features show *negative* permutation importance —
shuffling them slightly improves the model, meaning it is fitting noise on
them. A known cost of shipping the more interpretable model over the
marginally more accurate one.

---

## Tests and tooling

### 23 — The tamper-detection test could not fail
**30 Aug**

**Symptom.** The audit chain tamper test reported "MISSED".

**Cause.** The chain was fine. The **test** was broken. It set
`recommended_action` to `"allow"` on record 10 — and since 93% of records
are already `"allow"`, the tamper was a no-op. The hash matched because
nothing had changed.

**Fix.** Flip to a value guaranteed to differ, assert the change actually
occurred, and add a second test that alters an amount by one paise. Both now
detect.

**Lesson.** A test that cannot fail proves nothing. I came close to shipping
a tamper-evidence claim backed by a test that never tampered.

### 24 — The API leaked the score it was designed to withhold
**1 Sep**

**Symptom.** The `/score` response carefully omitted `ml_score`, then
printed `"Risk score 0.76 at or above block threshold 0.75"` in the reason
text directly beside it.

**Why it mattered.** Red-team test RT4 measured that withholding the score
raises decision-boundary discovery from one query to eight. Printing it in
prose returns the cost to one query, defeating the mitigation entirely.

**Also found.** `rules_fired` and `rule_severity` were hardcoded empty in
`/score` — a revoked mandate would have shown no fired rules.

**Fix.** Redact numeric values from outbound reason strings; populate the
rule fields properly. The audit record keeps exact values; external callers
do not.

**Lesson.** A mitigation applied to one field and not to the text beside it
is not a mitigation.

### 25 — The leakage check itself was wrong
**1 Sep**

**Symptom.** The confirmation-timestamp leakage check reported one violation
out of 1,064 sampled events.

**Cause.** When no confirmation has occurred, the lookup falls back to the
mandate's creation time. For one mandate, creation and first transaction
share a timestamp — and the check flagged that as leakage. Consent existing
at the moment of the first payment is not future information; it is the
precondition for it.

**Fix.** Exclude the created-at fallback from the violation check. Result
is now zero.

**Lesson.** The data was correct. The check was wrong. Verify the verifier.

### 26 — Cosmetic: sklearn OptimizeWarning during calibration
**28 Aug**

`OptimizeWarning: Unknown solver options: iprint` appears five times during
logistic regression calibration. A scipy/sklearn version interaction with no
effect on results. **Left visible rather than suppressed** — hiding a warning
you have not diagnosed is how real problems get missed.

---

## Limitations reported rather than fixed

### A — Route A is easier than reality
**28 Aug**

The injected merchant is drawn uniformly from all out-of-scope merchants, so
it usually mismatches the agent's category as well. Real prompt injection
would more often redirect to a *plausible* merchant in the same category,
making `category_matches_agent_type` far less informative.

**Route A recall here is therefore optimistic.** Not corrected, because
regenerating would have invalidated downstream verification with eight days
remaining. Stated rather than hidden.

### B — Red team findings, none remediated
**1 Sep**

Four adversarial attacks against the frozen model, mapped to the OWASP Top
10 for Agentic Applications 2026:

| Test | Result |
|---|---|
| RT1 adaptive evader | 70% → 53% when the attacker halves the attack; below 50% only at 40% magnitude |
| RT2 memory poisoning | 70% → 51.7% at 400% baseline inflation |
| RT3 mimicry | 70% → 50%. The worst case |
| RT4 threshold probing | Boundary located in **8 queries** |

RT1's gradual decline is the useful result: evasion costs the attacker most
of their take. RT3 is the honest weak spot — contextual features are all
fakeable; velocity and utilisation are not, and they carry the residual.
RT4's mitigation is rate limiting, which is **not implemented**.

**Nothing was remediated.** Fixing a weakness immediately after measuring it
invalidates the measurement.

---

## What the pattern says

Nine of these were found by printing a measurement rather than by an error.
Five were corrections to claims I had already made about my own work. Three
were bugs in the tests and checks themselves.

The generator, the feature extractor, the rule layer and the evaluation all
print diagnostics on every run. That is why the bugs above were found on the
day they were introduced rather than on the day before submission.
