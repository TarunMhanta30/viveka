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