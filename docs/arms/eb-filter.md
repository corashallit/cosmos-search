# Eclipsing-binary filter for the DR4 dimmers queue

Code: `cosmos_search/ebfilter/` (`search`, `decide`, `cal`), synthetic binaries in
`cosmos_search/inject/classes/eclipsing_binary.py`. Owner: "Eclipsing-binary filter for DR4" thread.

## Why our own filter

The DR3 rehearsal (docs/dayone.md) found two problems with leaning on DPAC for eclipsing binaries:

- 80% of tier 1 (1,747 of 2,190 G3 selections) had no DPAC classifier result, while VSX lists 9 of the 30
  ranked stars as eclipsing types. DR4's classifications may lag its photometry, and they label the star, not
  the dip.
- The recurrence flag (`vetting.recurrence`, rule v3b) flagged 4 of 71 testable Gaia EBs as aperiodic, at
  periods near 1 d where ZTF's nightly sampling aliases.

What the queue needs is a dip-level answer from the photometry itself: *is the dip that put this star in the
queue one eclipse of a strict-period binary?* An EB that stays costs one expert slot (budget ~10³); an anomaly
that is removed is lost. The filter is built to make the second error rare and to say how rare.

## The test, from first principles

**What distinguishes an eclipse from other dips is information, not shape.** An eclipse recurs at one period, so
once the period and phase are known, *which* epochs are in dip is predictable. An occulter on no single orbit
(dust, debris, a swarm, something engineered) does not have that property. So the test asks whether one period
compresses the dip pattern (minimum description length):

1. Label each epoch in each band (Gaia G, ZTF g/r/i, TESS; BP/RP share G's transits and are skipped) as dip
   (≥ 4 robust σ low and deeper than the band's second-largest brightening, a mirror test against heavy-tailed
   noise), flat (< 2 σ low) or neither (ingress, shallow eclipse in a noisy band: left out of both codes).
2. **H0 code:** the number of dip epochs and runs, then which of the C(N−K+1, E) C(K−1, E−1) arrangements of
   E runs among the N labelled epochs. No use of time.
3. **EB code:** a period from a fixed log-spaced grid (adjacent periods drift by a quarter-window over the span),
   a window width from {0.04, 0.1, 0.25, 0.6, 1.5, 4} d, a window start on a grid of a quarter of the window,
   optionally a second window (the secondary eclipse, any phase), one bit per band for whether it sees the
   eclipse. Epochs of participating bands inside a window are predicted to be in dip, so only the flats there
   are coded (as exceptions); everything else is coded as in H0.
4. **Statistic:** the saving Δ = L_H0 − L_EB in bits. The EB code pays for every parameter from a grid fixed
   before the data, so it is a valid prefix code, and the no-hypercompression inequality gives
   **P(Δ ≥ k | dips placed at random among this star's epochs) ≤ 2^−k.**

That bound is computed on the star's own epochs. Aliases (ZTF's 1-day, Gaia's scanning law, yearly gaps) and
the millions of trial periods are inside it: a period that lines up nightly dips buys nothing unless it also
keeps the flat epochs out of its window. Searching imperfectly only lowers Δ, so the bound stays valid. It needs
no simulation per star, which matters at 10⁴ candidates.

What it means in practice: with ZTF's ~1,000 epochs an EB needs about 6 eclipse events to compress; four can line
up by chance among the trial periods. With DR3 Gaia alone (~40 epochs) almost nothing compresses: the code
for one period and phase costs ~30 bits, and 40 epochs can't pay for it unless ~10 of them are in eclipse.
DR4 roughly doubles Gaia's epochs.

**Decision** (`ebfilter.decide`). A star is explained, i.e. moved out of the vetting queue into the EB control
pool, only if one of the best periods (aliases included) meets all three:

- Δ ≥ DELTA_BITS (the threshold below);
- the selected dip (the deepest Gaia G epoch) is in one of that solution's windows, and the window holds dips
  from other cycles at least 1/1.5 as deep (+0.05 mag; ranking v0's DPAC depth check). An EB with an extra dip at
  another phase, or a dip deeper than its eclipses, keeps its place in the queue;
- the eclipse fits an orbit: the dip epochs inside the window span at least D_lo, and two stars with
  R1 + R2 ≤ 2R* and M ≥ M* (R*, M* from G, BP−RP and parallax, `vetting.physics`) cannot eclipse longer than
  D_max = (P/π) asin(min(1, 2R*/a)) √((1+e)/(1−e)), e ≤ 0.9, at P or 2P. A strict period with longer dips is
  flagged `periodic_unphysical` and stays. Without a parallax with SNR ≥ 5 the check is skipped.

What the filter does not do: dips with one event (no period can be determined; the single-dip thread's test
covers them), contact binaries with no dip epochs under the mirror test (G3 does not select them either), and
depth or shape modelling (aperiodic occulters can have constant depths too).

## Calibration data

7,565 stars with both Gaia DR3 epoch photometry (from the kore cache, `(flags & 1) == 0`, G only) and ZTF DR22
light curves (LSDB HATS copy: IRSA's light-curve service timed out on the day), in eight ZTF HATS partitions at
b = −19° to +16°. 1,352 are DPAC eclipsing binaries. The joint table is on the share at
`/mnt/project-files/ebfilter/` (README there). Stars are split into halves A and B by a hash of source_id.

## Pre-registered (2026-10-21, before half A was scored)

`python -m cosmos_search.ebfilter.cal run --half A|B` builds, per half (`ebfilter/cal.py` docstring):

- 1,000 synthetic physical EBs (P log-uniform 0.3–300 d, q ~ U(0.1, 1), e up to 0.6 beyond 3 d, third light
  in half of them, primary = the host's own R*, M*, Teff) in non-EB hosts with ≥ 100 ZTF epochs;
- 150 injections of each anomaly family (grey, dust and blue dips, chromatic flip, Poisson, prime and geometric
  dip trains, secular fade, step: the nine *target* families) and of strict-period dip trains (reported
  separately: an occulter on one strict orbit is what an EB is, unless the duration check catches it);
- 75 each of grey dips and Poisson, prime and geometric trains injected into real DPAC EBs (`eb_host`);
- every DPAC EB and every non-EB host as observed that would be queued.

Every injection is redrawn until the star would be queued (its deepest G epoch ≥ 5σ low, lower than its
largest brightening is high, and moved ≥ 5σ by the injection), so all rates are conditional on Gaia having seen
the dip, as for real candidates. Each population is run on Gaia alone, on G + ZTF (joint) and, for EBs, on ZTF
alone.

**Threshold rule.** DELTA_BITS is the smallest integer k ≥ 7 for which, on half A in the joint mode, the pooled
loss over the nine target families is ≤ 1% and no single target family loses more than 3%. Nothing else is tuned
on A.

**Reported on half B** (with 95% Wilson intervals): recall on synthetic EBs (Gaia, ZTF, joint; overall and by
period); anomaly loss per target family and pooled (Gaia, joint); `eb_host` loss; strict-period train removal;
real DPAC EB recall on ZTF alone (no leak: DPAC's labels come from the Gaia epochs) and on joint data
(agreement, not independent); the share of queued non-DPAC-EB stars the filter explains.

*Changes before half A was scored,* found by the unit tests (synthetic cases, not calibration data), and the
half-A run restarted: (1) only periods within 10 bits of the best (odds ≥ 2^−10) may explain the target, because
a 1-in-several alias 170 bits worse than the true period could cover an extra dip by chance; (2) the duration
bound uses the whole run of dip epochs touching a window (no flat between, gaps < min(1.5 d, P/2)), because a
narrow window inside a 4-day dip otherwise claims a short eclipse.

**Decision.** Use the filter in ranking for the D2 freeze if, on half B, the pooled target-family loss has a
95% upper bound ≤ 3% and joint synthetic-EB recall is ≥ 0.5. Otherwise revise and test on fresh injections.

## Results

(After the runs.)
