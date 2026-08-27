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