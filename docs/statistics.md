# Statistics

Shared by all arms. The search runs over up to 1.8 × 10⁹ sources, so every number we report has to survive
the multiple-testing arithmetic and an instrument that produces most of the outliers.

## 1. False alarms are a budget

With N sources and a per-source false-alarm probability p, the expected number of false alarms is Np. At
N = 1.8 × 10⁹, a Gaussian 6σ tail (p ≈ 10⁻⁹) already gives ~2. Real tails are heavier, so analytic p-values
are wrong by orders of magnitude out there. Every threshold is therefore set from an **empirical** null and
stated as an expected false-alarm count over the searched set, with the vetting budget (~10³) as the
ceiling.

## 2. Empirical nulls

Nulls are built from the data so that they keep its systematics:

- **Mirror null.** Instrumental scatter is roughly symmetric; occultation is one-sided. The distribution of
  brightening statistics, computed in the same strata (magnitude, colour, ecliptic latitude, crowding, number
  of transits), estimates the dimming statistic's null. Brightening has its own astrophysics (flares,
  microlensing), which makes this null conservative in some strata. Measure that, don't assume it.
- **Strata must include the number of epochs.** Statistics that take an extreme over a light curve
  (minimum flux, largest gap, longest run) have nulls that widen with the number of points, roughly as
  √(2 ln n) for near-Gaussian noise. Gaia's epoch count follows the scanning law, mostly ecliptic latitude,
  so leaving it out of the strata produces false structure on the sky. Stratify on it, or convert to a
  per-point tail probability and apply the Šidák form 1 − (1 − p₁)ⁿ before calibrating.
- **Stratified permutation.** Re-assign epochs between sources in the same stratum that share scan
  history, which keeps the cadence and calibration systematics but breaks any star-specific signal.
- **Invariance nulls.** Gaia's redundancy gives tests whose null is exact: agreement across the 9 CCDs of a
  transit, across the two fields of view, between G and BP/RP. An artefact usually fails at least one.

**How far out a null reaches.** An empirical null of n draws resolves p down to 1/(n + 1), and the
add-one form (1 + #exceedances)/(1 + n) is valid for exchangeable draws. The mirror null has one draw per
source, so a stratum holding 10⁶ sources resolves p ≈ 10⁻⁶ directly. For N = 10⁹ and a budget of 10³ false
alarms the threshold is p ≈ 10⁻⁶, which sits at that resolution. Strata should therefore be coarse enough to
hold ≳ 10⁶ sources wherever the threshold has to be set.

Beyond the last few exceedances, p comes from a generalised Pareto fit to the null's top 1% (the
Pickands–Balkema–de Haan theorem makes this nearly assumption-free). Shape uncertainty compounds with
distance, so the default fit uses the 90th bootstrap percentile of the shape and never a short tail.
On test nulls of 10⁵ draws this costs a factor of a few to ~10² in p six decades out, and is too small by
more than 3× in ≲ 10% of nulls. Those far-tail p-values decide order within the shortlist, never whether
the threshold is met. Code: `cosmos_search.stats.nulls`.

## 3. Multiple testing

Within a source, combine statistics with a method robust to correlation (Cauchy combination, or minimum p
with a Šidák correction). Across sources, control the false discovery rate (Benjamini–Hochberg, or Storey
q-values when nearly all sources are null). Periodogram and dip searches carry a per-light-curve
look-elsewhere effect; calibrate it with injections and nulls, not analytically. Code:
`cosmos_search.stats.multiple`. The Cauchy combination stays valid under the strong dependence between
detectors run on the same light curve, which Fisher's method does not.

## 4. Injection-recovery

Inject into real light curves and real astrometric residuals so the systematics are real. Families should
span more than the signals we expect, because they measure how agnostic each detector is:

| Family | Examples |
|---|---|
| Astrophysical | Long-period eclipses, dust dips (reddened), transits, flares, spots, secular trends |
| Occulter geometry | Grey dips of varied depth, duration and recurrence; dips that are not periodic and not random |
| Non-astrophysical structure | Step changes, regularly spaced dips at intervals no orbit produces, number-theoretic spacings, chromatic oddities (BP dims while RP brightens) |
| Astrometric | Constant thrust, impulsive velocity changes, radiation-pressure accelerations correlated with flux |

The light-curve families are implemented in `cosmos_search/inject/lightcurve.py`, shared by every arm.
Occultations are parametrised physically (covering fraction × (1 − e^−τ(λ)), τ ∝ λ^−β), so grey, reddened
and blue dips come from one model. Hosts are drawn from the real sample with matched magnitude, colour and
sky distributions. Report
completeness as maps over the family's parameters, with Wilson intervals (`cosmos_search.inject.recovery`). Split injections into a visible **tuning** set and a
**sealed** set made by the truth repo; final numbers come only from the sealed set (blinding.md).

## 5. Ranking as a decision

A vetting slot is worth spending on candidate i in proportion to

  value_i ≈ P(real and unexplained | data)_i × P(cheaply verifiable)_i

where verifiability covers brightness (G ≲ 13 for amateur photometry), current activity, sky visibility and
TESS/ZTF coverage. The terms start from flat priors and are refitted as vetting outcomes and injection
recovery accumulate (with monotone constraints so a stronger anomaly never ranks lower, all else equal).

**Random audit.** Spend ~5–10% of vetting slots on candidates drawn at random from below the threshold,
with known inclusion probabilities. Weighting their outcomes by those probabilities gives an unbiased
estimate of what the ranking misses, and that estimate is the only check on its built-in assumptions.
Inclusion probabilities lean towards higher scores but have a floor, so every source can be drawn
(`cosmos_search.stats.audit`, Horvitz–Thompson with its standard error).

## 6. Decoys and controls for vetting

The main failure mode for LLM vetting is confabulation: a fluent, physical-sounding story for anything. Full
design in [vetting.md](vetting.md); the statistical points:

- **Roles.** Known-answer controls (KIC 8462852 and other studied dippers, ordinary variables, known artefacts)
  measure the false-"unexplained" rate per class. Injected decoys come in three scored roles: *incoherent*
  (no single physical screen does this; a confident natural story is a confabulation), *anomaly probes*
  (coherent, with no natural counterpart; they measure whether a real anomaly would survive), and *rare natural*
  (they measure over-calling "unexplained").
- **Stateless vetters change the mixing rule.** A Stage A agent sees one packet and no history, so a hidden
  in-stream fraction does not change its behaviour; per-packet indistinguishability does. Most calibration runs
  as a separate batch (~400 objects in V3), with ~5% drift checks in production. An audit classifier on packet
  features must not beat chance at separating planted from real packets.
- **Held-out prediction.** Part of each packet is withheld and the expert must predict it from its explanation.
  Predictive skill measures confabulation on real candidates, where no decoy label exists.
- **Reported figure.** The expected number of false shortlist entries, from the controls' confusion matrix under
  label shift, with a predictive interval. With ~10³ objects and per-object error rates of a few per cent this is
  tens of entries, so precision has to come from stacking filters with independent errors, measured jointly.

Vetters state probabilities; calibration is scored with log and Brier scores and reliability curves, and
recalibrated with a monotone map fitted on controls. A vetter configuration is accepted per class by Bayesian
sequential stopping on its false-pass rate.

## 7. Recalibration and freezing

Detectors, nulls and rankings change as evidence comes in (plan.md, principle 2). The constraint is order:
anything evaluated on the sealed injection set, and anything run on DR4 day one, uses a frozen, tagged
version written down before the data or the truth is seen. Changes after that are new versions with their
own evaluation.
