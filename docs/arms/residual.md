# Residual search: the simplest model that fits, then what is left

Code: `cosmos_search/residual/` (`model`, `cal`), on top of the EB filter (`cosmos_search/ebfilter/`,
docs/arms/eb-filter.md). Owner: "Simplest-model residual search" thread.

## Why

Up to the D2 freeze the chain asked, star by star, *is this an eclipsing binary?* and removed the star when the
answer was yes, while each new kind of anomaly added its own flag to the shortlist rule (v3b's recurrence flag, v4's
single-dip flag). Two problems follow from that framing.

- **A cut gives no partial credit.** The EB filter explains 17% of dips injected into real binaries (half B), because
  the dip that put the star in the queue was likeliest to be deepest when it landed on an eclipse, and the depth
  check (≤ 1.5 × the window's deepest other dip + 0.05 mag) passed the sum. The star then leaves the queue with its
  anomaly.
- **Flags are tested on raw dips.** The recurrence flag counts every dip event, so an eclipsing binary's own eclipses
  either look aperiodic (aliases near 1 d: 4 of 71 testable Gaia EBs flagged) or swamp an aperiodic train on the
  same star (the train's events become a minority among aligned eclipses).

From one step back, a binary is not a nuisance to remove but the best-instrumented host there is. Once its period
and eclipse profile are known, every epoch has a predicted brightness, so a dip the eclipses do not predict stands out
more cleanly than on a star whose own variability is unknown. The question for a technosignature search is not
*which known class is this star?* but *what does its light curve do that no simple model predicts?*

## The method

Per star, on every survey's epochs (Gaia G, ZTF g/r/i, TESS; BP/RP follow G's transits):

1. **Baseline.** The simplest model from a fixed class: constant, or a strict-period eclipse. The eclipse is the EB
   filter's compression test (Δ ≥ 7 bits over random dip placement, so P(false eclipse) ≤ 2⁻⁷ per star), with its
   purity (≥ 80% of a window's labelled epochs are dips) and orbit-duration checks; the target need not be covered.
   No other model is in the class yet: a single event and an aperiodic train are what the search is for, a trend is
   an anomaly family, and stochastic variability sets each band's robust σ and the mirror test.
2. **Refine the period on the depths.** The label search's grid leaves up to an eighth of a window of phase drift over
   the span. Within two grid steps either side, the frequency with the smoothest folded profile (least squared
   depth difference between phase-adjacent in-window epochs of a band, in σ) is taken. Both P and 2P are tried: at
   half the period a primary and a secondary of similar width fold onto one window, which labels cannot tell apart
   and depths can. The one leaving fewer residual dips is kept.
3. **Predict each epoch near an eclipse** (each window widened by one window width either side, because the eclipse's
   ingress can lie outside a coarse window). The prediction is the deeper of the epoch's two phase neighbours from
   *other cycles* (> 1 d away, so a dip never explains itself), the nearest at or before its phase and the nearest at
   or after, each within ⅛ of the window width. On a monotonic stretch of the profile the true depth lies between
   them. From the epoch's own band: tolerance TOL. From other bands: the star's own depth ratio between this band and
   the others (median over at least two other-cycle epochs of this band where the others predict ≥ 5σ), tolerance
   TOL; without a measured ratio, 2 × TOL, because eclipse depths differ between bands by colour and by blending
   (ZTF's PSF takes in neighbours Gaia resolves). No neighbour at that phase: the window's deepest other-cycle epoch
   (an eclipse is nowhere deeper than its deepest point), 2 × TOL.
4. **Residual dips.** A dip epoch (`ebfilter.search.labels`: ≥ 4σ low and below the band's second-largest
   brightening) outside every eclipse, or an epoch near an eclipse deeper than its prediction × (1 + tolerance) by
   ≥ 4 combined σ. Residual light curves drop the explained epochs and keep only the excess depth of residual ones.
5. **Verdict.**
   - `demote`: the EB filter explains the selected dip (as PR #45) *and* the selected dip is not a residual *and*
     no residual event (with ≥ MIN_EVENT_EPOCHS dip epochs) is left anywhere else. The star moves to the EB control
     pool. Demotion is a subset of what PR #45 explains, so anomaly loss cannot rise.
   - `eb_residual`: the baseline is an eclipse and something is left (the selected dip or another event). An
     eclipsing binary with an unexplained dip.
   - The recurrence test (`vetting.recurrence`, unchanged) also runs on the residual epochs: `residual_aperiodic`.

What this does not do yet: single events on residuals (the single-event test is in PR #46, owned by the single-dip
thread; its input would be the residual light curves), and 2–3 residual events, where neither recurrence (needs ≥ 4)
nor the single-event test (needs exactly 1) applies.

## Pre-registered (2026-10-21, before half A was run)

Data: the EB filter's joint Gaia DR3 + ZTF DR22 set (7,565 stars; `/mnt/project-files/ebfilter/`), halves by a hash
of source_id, joint mode only. Injections are drawn fresh (urandom seed, recorded) for each run, with the EB filter's
machinery (`ebfilter.cal`): every injection redrawn until its deepest G epoch would put the star in the queue.

Per half: every DPAC EB (`real_eb`) and non-EB host (`host`) with ≥ 100 ZTF epochs that would be queued, 300
synthetic EBs (`eb_synth`), 60 injections of each of the nine target families and of strict-period trains into
non-EB hosts (`anomaly`), and 120 each of grey dips and Poisson, prime and geometric trains into real DPAC EBs
(`eb_host`). Fixed, not tuned: RES_SIGMA = 4 (the dip threshold), CROSS_FACTOR = 2, neighbour width ⅛ window,
margin 1 window, ratio from ≥ 2 epochs at ≥ 5σ, Δ ≥ 7 bits and the filter's other constants.

**Choice on half A.** Over TOL ∈ {0.05, 0.1, 0.2, 0.4} and MIN_EVENT_EPOCHS ∈ {1, 2}: the setting with the smallest
eb_host loss (eb_host injections demoted) among those that demote at least 0.75 × as many real DPAC EBs as the EB
filter explains on the same stars; ties to more real-EB demotion (`residual.cal choose`). If none qualifies, ranking
v1 uses the filter as in PR #45 and the residual search is revised.

**Held-out test on half B** (fresh urandom seed drawn after this section is committed), at the chosen setting, with
95% Wilson intervals, paired against the EB filter on the same injections:

- D1, demotion for ranking v1: adopt if (a) eb_host loss ≤ half the filter's, (b) real-EB demotion ≥ 0.7 × the
  filter's, and (c) target-family loss in non-EB hosts ≤ the filter's (true by construction; reported). Otherwise
  ranking v1 demotes the filter's `explained` stars, as specified in PR #45.
- D2, `eb_residual` as a ranking flag (counted with fov, grey, tension, bright): adopt if it marks ≤ 15% of real
  DPAC EBs and ≥ 50% of eb_host injections. Otherwise it is reported in the packet and not used for priority.
- D3, recurrence on residuals in place of raw epochs for v3b's `aperiodic` input (rule v3c = S ≥ 0.35 or
  `residual_aperiodic`): adopt if (a) on trains in EB hosts, residual flags more than raw (paired gains > losses);
  (b) on trains in non-EB hosts, residual recall ≥ raw recall − 2 percentage points; (c) on testable real DPAC EBs,
  residual flags ≤ raw flags. The agents' part of the rule is unchanged, so this is a pipeline-level test; the
  flag's agents-plus-flag recall was measured on batch 4.

Also reported: synthetic-EB demotion, real queued non-EB hosts demoted and flagged, strict-period trains demoted, and
the real stars (no injection) with `eb_residual`, as exploratory leads.

## Ranking v1 (for the D2 freeze)

`python -m cosmos_search.dayone.rank requeue` (after `dayone.external fetch` on a prefetch list of the top tier-1–2
rows, then `python -m cosmos_search.residual` on it): demoted stars (D1's rule, or the filter's `explained` if D1
fails) move to tier 3 as known-answer controls, the audit draw is made over every star below the vetting cut
*including the demoted ones*, with known inclusion probabilities, so DR4 measures what demotion loses.
