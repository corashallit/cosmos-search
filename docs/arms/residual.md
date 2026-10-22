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
- D3, recurrence on residuals in place of raw epochs as the `aperiodic` input (rephrased after half A, see below:
  the input to ranking v1's priority flag, not a shortlist bypass): adopt if (a) on trains in EB hosts, residual flags more than raw (paired gains > losses);
  (b) on trains in non-EB hosts, residual recall ≥ raw recall − 2 percentage points; (c) on testable real DPAC EBs,
  residual flags ≤ raw flags. The agents' part of the rule is unchanged, so this is a pipeline-level test; the
  flag's agents-plus-flag recall was measured on batch 4.

Also reported: synthetic-EB demotion, real queued non-EB hosts demoted and flagged, strict-period trains demoted, and
the real stars (no injection) with `eb_residual`, as exploratory leads.

### Half A (calibration), and changes recorded before half B

Run with seed 12169237894723442052 (4,813 jobs; 175 queued real DPAC EBs, 46 queued non-EB hosts, 254 synthetic
EBs, 360 target-family and 49 strict-period injections, 177 eb_host injections that could be drawn).

| TOL | MIN_EVENT_EPOCHS | real EBs demoted | eb_host loss | qualifies (≥ 0.75 × 0.674) |
|---|---|---|---|---|
| 0.05 | 1 | 0.234 | 0.023 | no |
| 0.05 | 2 | 0.571 | 0.056 | **yes, chosen** |
| 0.1 | 1 / 2 | 0.240 / 0.571 | 0.034 / 0.068 | no / yes |
| 0.2 | 1 / 2 | 0.240 / 0.577 | 0.034 / 0.068 | no / yes |
| 0.4 | 1 / 2 | 0.251 / 0.589 | 0.034 / 0.073 | no / yes |

The EB filter alone explains 67.4% of the real EBs and 13.0% of the eb_host injections on these stars. **Chosen: TOL =
0.05, MIN_EVENT_EPOCHS = 2.** The tolerance barely matters; what matters is whether a one-epoch residual event counts.
With MIN_EVENT_EPOCHS = 1, 68% of real EBs keep some residual event, almost all of them single epochs (ZTF one-night
outliers and ingress mispredictions).

*Two changes after half A, recorded before half B was drawn:*

1. **Recurrence on residuals uses the same event definition.** Under an eclipse baseline, residual events with fewer
   than MIN_EVENT_EPOCHS in-dip epochs are dropped before the recurrence test. On half A without this, the residual
   flag fired on 18 of 175 real EBs against 14 on raw epochs: the lone epochs an eclipse leaves behind look
   aperiodic once the eclipses no longer outnumber them. Constant baselines are untouched (residual = raw).
2. **D3 is rephrased because of the red-team review's R1** (`/mnt/project-files/review/`): on real DR3 stars with ZTF,
   v3b's flag fires on 64% of testable non-EB variables (79% of RS CVn, 67% of SOLAR_LIKE) against 9% of EBs, and v3b
   lets it shortlist on its own. Ranking v1 therefore uses the flag as a priority flag, not a bypass (the shortlist
   rule stays frozen until the review's fixes land). D3 now decides which epochs feed that ranking flag, raw or
   residual, with the same criteria (a)–(c). Half B adds a population `real_class`: every real star of the half the
   raw test can test (the review's list), so both versions' flag rates are reported by DPAC class.

## Results on held-out half B

Seed 17678711268131745800 (urandom, drawn after commit 11ecace), 5,212 jobs: 187 queued real DPAC EBs, 54 queued
non-EB hosts, 260 synthetic EBs, 353 target-family and 50 strict-period injections, 194 eb_host injections, and 323
real stars the raw recurrence test can test, by DPAC class. TOL 0.05, MIN_EVENT_EPOCHS 2. Paired against the EB filter
(PR #45) on the same stars and injections; 95% Wilson intervals. Outputs: `/mnt/project-files/residual/res_B.parquet`
(and `res_A.parquet`, `res_A2.parquet` for half A).

| population | EB filter explains | residual search demotes | `eb_residual` |
|---|---|---|---|
| dips and trains injected into real EBs (eb_host) | 32/194 (16.5%) | **10/194 (5.2%; 2.8–9.2%)** | 64/194 (33%) |
| queued real DPAC EBs | 119/187 (63.6%) | **95/187 (50.8%; 43.7–57.9%)** | 28/187 (15.0%) |
| synthetic EBs | 144/260 (55.4%) | 113/260 (43.5%) | 41/260 (15.8%) |
| target-family injections, non-EB hosts | 1/353 (0.3%) | **0/353 (0–1.1%)** | 5/353 (1.4%) |
| strict-period trains | 25/50 | 22/50 | 4/50 |
| queued non-EB hosts | 10/54 | 6/54 | 5/54 |

By family, eb_host demoted: grey dips 5/60, Poisson trains 1/41, prime trains 3/50, geometric trains 1/43.

- **D1, adopted.** eb_host loss 5.2% ≤ half of the filter's 16.5%; real-EB demotion 0.80 × the filter's (≥ 0.7); no
  target-family loss. Ranking v1 demotes with the residual rule (`requeue --rule residual`, the default).
- **D2, not adopted.** `eb_residual` marks 15.0% of real EBs (at the limit) and only 33% of eb_host injections
  (criterion ≥ 50%): two thirds of the binaries with an injected dip have no eclipse baseline at all (the injected
  dips spoil the period's purity), so they stay in the queue without the flag. In a DR4 queue, where real EBs
  outnumber EBs with extra dips, the flag would mostly promote binaries with ZTF systematics. It is reported, not
  used for priority.
- **D3, adopted.** On trains injected into EBs the residual flag gains 7 and loses 2 against raw epochs; on trains in
  non-EB hosts it flags 36.2% against 36.8% (−0.7 points, within 2); on testable real EBs 11 against 13. Ranking
  v1's `aperiodic` priority flag reads the residual epochs.

**The flag on active stars (review R1, R16), real stars of both halves, testable ones:**

| DPAC class | testable | raw flag | residual flag |
|---|---|---|---|
| RS, SOLAR_LIKE, S | 111 | 82 (74%; 65–81%) | 74 (67%) |
| other non-EB variables (DSCT, LPV, CEP, …) | 33 | 23 (70%) | 18 (55%) |
| unclassified | 38 | 14 (37%; 23–53%) | 14 (37%) |
| AGN | 11 | 5 (45%) | 5 (45%) |
| ECL | 361 | 37 (10%) | 29 (8%) |
| injected trains, non-EB hosts (half B) | 152 | 56 (37%) | 55 (36%) |

The residual search does not touch this: active stars have a constant baseline, so their residuals are their raw
epochs. **Dip timing alone does not separate activity from occulters**: the flag fires more often on spotted stars
than on the injected trains it was built for. A prototype rotation baseline (per-season sinusoid and harmonic at the
Lomb–Scargle period, `/home/user/rs/rot/proto.py`, half A) did not help either: it found "rotation" in 60–70% of
injected trains and left SOLAR_LIKE flags unchanged (16 of 24). So under rule v3b, which lets the flag shortlist on
its own, every testable active star in the queue has about a two-in-three chance of reaching the shortlist whatever
the vetter says. Tier 2 lowers how many get vetted; it does not change that rate. The discriminant has to come
from something other than timing (dip colour, flares, the shape of the out-of-dip variability, X-rays), which is
the light-curve activity score the review asks for (R17), in a follow-up.

## Ranking v1 (for the D2 freeze)

`python -m cosmos_search.dayone.rank requeue` (after `dayone.external fetch` on a prefetch list of the top tier-1–2
rows, then `python -m cosmos_search.residual` on it): demoted stars (D1's rule, or the filter's `explained` if D1
fails) move to tier 3 as known-answer controls, the audit draw is made over every star below the vetting cut
*including the demoted ones*, with known inclusion probabilities, so DR4 measures what demotion loses.

Active stars (review R1): RS, SOLAR_LIKE, S and AGN are tier 2 in v1 (v0 left the first three in tier 1), the
`aperiodic` flag is one priority flag among fov, grey, tension and bright, and the queue carries `--n-active-controls`
(default 8) active-class stars as controls labelled `rotation_or_spots` (AGN: `extragalactic`), so the false-flag rate
is measured on the class that will crowd the shortlist. On DR4 day one DPAC classes may lag the photometry; an
activity model of our own (rotation period from ZTF/TESS, flares, eROSITA, `activityindex_espcs`) is the next step.
