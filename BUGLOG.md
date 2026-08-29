## 27 Aug 2026
- Env: imports failed with ModuleNotFoundError despite an active venv.
  First hypothesis — bare `pip` resolving to system Python — was wrong;
  `which pip` confirmed the venv's own pip.
  Real cause: WSL is Ubuntu 26.04 (resolute), whose default Python is
  3.14. numpy 2.1.3 has no prebuilt wheel for 3.14, so pip fell back to
  a Meson source build, which failed on a missing Python dev dependency.
  Fix: installed python3.12 via the deadsnakes PPA, deleted and rebuilt
  the venv on 3.12. Install became a wheel download instead of a compile.
  Lesson: check wheel availability for your interpreter before pinning
  versions. The newest Python is not the safe choice for the scientific
  stack — it leads the ecosystem, not follows it.

  - Design bug in generator, caught before writing transactions.
  Reserve limit was sized as median_transaction × 8-20, but a weekly-rhythm
  principal makes ~12 transactions/month, so normal spending would exceed
  their own reserve and fire hard rule H3 on benign traffic, destroying the
  false-positive baseline.
  Fix: size reserve against expected MONTHLY spend (median × 30/mean_gap_days)
  × 1.4-2.5, giving normal utilisation ~0.4-0.7 with headroom.
  Lesson: a limit must be sized against the rate of consumption, not the size
  of one event.

- Off-by-one in principal active hours. active_hour_end computed as
  start_hour + randint(8,13) with start_hour up to 11, so the maximum was 24 —
  not a valid hour. Would have broken timestamp generation in Pass 2.
  Fix: clamp with min(23, ...).
  Lesson: any derived time field needs its range checked at both ends, not
  just the lower one.

  - Benign traffic breached its own reserve limit in 14.5% of mandate-months,
  which would fire hard rule H3 (Critical -> block) on legitimate users and
  destroy the false-positive baseline.
  Cause: reserve sized against the MEDIAN transaction amount. Actual mean
  monthly spend is ~1.28x higher, because a log-normal's mean exceeds its
  median by exp(sigma^2/2), and the 3% amount-outlier injection multiplies
  those amounts by 3-8x.
  Fix: size the reserve against expected (mean) spend including the outlier
  contribution, and raise headroom from 1.4-2.5x to 2.0-3.2x.
  Lesson: size a limit against the expected value of a distribution, not a
  typical value. Also: this was only caught because the generator prints
  measured diagnostics rather than assuming the parameters are right.


- Route B (delegation abuse) concentrated in only 16 principals out of 500,
  because each injection creates a burst of 6-10 events from one principal.
  Section 9.10 splits by principal, so the held-out set would carry only
  ~3 Route B principals — making recall on my most important attack class
  statistically meaningless.
  Fix: reduced burst size to 4-7 and raised Route B's share of the attack
  budget from 40% to 50%, roughly doubling the number of distinct
  principals carrying it.
  Lesson: with a principal-level split, what matters is not the number of
  attack EVENTS but the number of distinct PRINCIPALS carrying each attack
  class. Event counts alone hid this.

  - Section 9.7 specified 500 principals / ~35k events. Raised to 1200
  principals after measuring that Route B reached only 7 held-out
  principals — too few for a meaningful recall estimate on the most
  important attack class. Burst size reduced 4-7 to 3-6 to spread the
  same event budget across more principals.
  Lesson: with a principal-level split, statistical power is set by
  distinct principals per class per split, not by total event count.

  - Side effect of the burst-size reduction: H3 now fires on 5.5% of Route B
  events, up from 0.7%. Smaller bursts divide the same drain budget across
  fewer transactions, so each is larger and more likely to cross the reserve.
  Accepted: 94.5% of Route B remains invisible to hard rules, which is the
  property the ML layer depends on. Recorded rather than silently tuned away.

  - Data inspection revealed instruction_source was a partial label proxy:
  agent_autonomous appeared in 53% of attacks vs 15% of benign, because
  Route A always used external_content and Route B always agent_autonomous.
  Same issue with hour (attack mean 8.6 vs benign 13.7) since Route B always
  drew from [1,2,3,4,23].
  Section 9's rule L3 required benign rows to use these values too, which
  they did — but I missed the mirror requirement that attacks must not use
  them exclusively.
  Fix: attacks now draw instruction_source from route-specific weighted
  mixes, and Route B uses off-hours only 65% of the time.
  Lesson: leakage prevention runs both ways. It is not enough for the benign
  class to look varied; the attack class must too.


  - Three numerical bugs found by printing per-feature distributions rather
  than assuming the features were sane:
  1. interarrival_zscore reached 34,717 — a principal with near-simultaneous
     prior events has sd_gap ~ 0, so the division exploded. Fixed with a
     denominator floor and a clip to +/-20.
  2. velocity_ratio_1h reached 678 for the same reason on hourly_rate.
     Clipped to 100.
  3. utilisation_velocity median is 0.43, not the ~1.0 claimed in Section
     10.4. Cause: reserves carry 2-3x headroom by design (Section 9 fix), so
     full-month utilisation is ~0.4. Section 10 wording corrected; the
     feature is sound but the stated interpretation was wrong.
  Lesson: gradient boosting would have tolerated the extreme values silently.
  Logistic regression would not — one exploded row can dominate the fit.
  Distribution checks catch what model accuracy hides.

  - Route B median for velocity_ratio_1h sits at the clip ceiling of 100.
  The clip is therefore compressing genuine signal, not only trimming
  numerical blow-ups. Accepted: the feature still separates cleanly
  (benign median 0.00), and the alternative -- unbounded ratios reaching
  678 -- would destabilise logistic regression. Recorded so the clip is a
  stated design choice rather than a hidden one.

  - Route A realism limitation: the injected merchant is drawn uniformly from
  all out-of-scope merchants, so it usually mismatches the agent's category
  as well. Real injection would more often redirect to a plausible merchant
  in the same category, making category_matches_agent_type less informative.
  Route A recall here is therefore likely optimistic. Not corrected, because
  regenerating would invalidate downstream verification with 8 days left.
  Stated rather than hidden.

  - H1 (mandate revoked) fired on 682 legitimate transactions -- 1.71% of all
  traffic wrongly recommended for block. Cause: the Mandate schema carried a
  status field with no revocation timestamp, so the rule treated "revoked"
  as true for all time, including transactions weeks before revocation.
  Separately, H2 (expired) never fired at all, because transaction generation
  stopped at the mandate end date, so nothing ever occurred after expiry.
  Two of four hard rules were therefore broken or dead.
  Fix: added revoked_at to the schema, made H1 time-aware, and added a
  "stale credential" attack variant -- transactions after revocation or
  expiry -- which is realistic and makes both rules reachable.
  Lesson: a status field without a timestamp is not enough to reason about
  time. Rule fire-rate tables catch this; model accuracy never would.

  - Fix verified: H1 false positives went 682 -> 0 after adding revoked_at and
  making the rule time-aware. H2 went from dead (0 fires) to 21. Benign
  critical rate dropped 2.2% -> 0.5%.
  Note on the route table: Route A and C show ~100% elevated because both
  always use out-of-scope merchants, so H4 fires. That is rules doing rules'
  work. The model's real task on Route A is separating it from the 4.9% of
  BENIGN traffic that is also out-of-scope -- a harder problem than the
  headline number suggests.

  - Route C being holdout-only raises the holdout base rate to ~3.7% versus
  1.29% in validation. PR-AUC increases mechanically with base rate, so
  holdout and validation PR-AUC are not directly comparable. evaluate.py
  must report holdout both with and without Route C.
  Lesson: an experimental control placed in one split changes that split's
  class balance. The control is still correct; the comparison needs care.

- Cosmetic: sklearn emits OptimizeWarning "Unknown solver options: iprint"
  during LR calibration. A scipy/sklearn version interaction, no effect on
  results. Left visible rather than suppressed.

  - Sensitivity finding: churn penalty has ZERO effect on the derived
  thresholds (x2 and x0.5 give identical results), because no benign
  transaction scores above t2=0.75, so the churn term is never invoked.
  Abandonment rate is the only assumed parameter that moves the answer --
  doubling it shifts t1 from 0.10 to 0.40.
  Useful outcome: of three assumed parameters, only one is decision-relevant,
  and it is one Razorpay can measure directly.

- Methodological gap: cost_model.py sweeps thresholds using the ML score
  alone, but fusion.py applies the rule floor on top. Result: the cost model
  reported 0% benign blocked while the deployed path blocks 0.7% (H3 firing
  on benign reserve breaches). The thresholds were therefore optimised
  against a decision path the system does not use.
  Also: 9.7% of validation events have <7 days history, so the ML score is
  suppressed and only rules apply -- a real coverage gap, not a bug.
  Lesson: optimise against the decision the system actually makes, not
  against one component of it.

  - Fixed the cost model to sweep thresholds through the full fusion decision
  path rather than the ML score alone. Consequences:
  1. Total cost rose 121k -> 510k. The old figure was fiction; it omitted
     the rule floor that the deployed system applies.
  2. t1 moved 0.10 -> 0.45. With rules already catching most of Route A,
     aggressive score-based step-up adds friction without adding catches.
  3. Saving vs hand-picked thresholds collapsed from 10.4% to 1.6-2.5%.
     The cost surface is flat; threshold tuning is not where the value is.
  4. New baseline "rules only (no ML)" costs 801k vs 510k with ML -- a
     36.3% saving. THAT is where the value is, and it is the answer to
     "why not just write rules".
  Lesson: optimising a component in isolation produced a number that looked
  good and was wrong. Optimising the whole path produced a worse-looking
  number that is true, and revealed where the real gain comes from.

  - Tamper test reported "MISSED" -- but the chain was fine; the TEST was
  broken. It set recommended_action to "allow" on record 10, and since 93%
  of records are already "allow", the tamper was a no-op. The hash matched
  because nothing had changed.
  Fix: flip to a value guaranteed to differ, assert the change actually
  happened, and add a second test that alters an amount by one paise.
  Lesson: a test that cannot fail proves nothing. I nearly shipped a
  tamper-evidence claim backed by a test that never tampered.

  - top_factors() reported meaningless attribution. It ranked features by raw
  absolute value, so days_to_expiry (~307) and mandate_age_days (~52) topped
  every record simply for being large numbers, while the feature that
  actually drove the decision -- merchant_new_to_principal at 1.0 -- never
  appeared. Requirement R9 was visibly unmet, in every audit record and in
  the demo output.
  Fix: rank by deviation from the benign median, scaled by that feature's
  own interquartile range, computed on benign TRAIN rows only.
  Lesson: attribution needs a reference point. "Large" is not "unusual", and
  I would not have caught this without printing a real decision's factors.

  - The top_factors fix improved attribution but is still a heuristic, not
  true attribution. Six features are binary with benign IQR = 0, so the
  1e-6 spread floor makes any firing binary feature rank near the top
  regardless of contribution -- is_external_content appeared on a decision
  it did not drive.
  Labelled as a known limitation rather than presented as attribution.
  Real per-decision attribution needs SHAP on the gradient boosting model.
  Lesson: fixing an obviously-wrong ranking with a slightly-less-wrong one
  is progress, but calling it attribution would be a false claim.