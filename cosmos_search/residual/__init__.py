"""Explain each star with the simplest model that fits, then search what is left over.

Until now the dimmers chain asked, star by star, "is this an eclipsing binary?" and removed the star when the answer
was yes (ranking v0's DPAC check, the EB filter), while each new kind of anomaly got its own flag stacked onto the
shortlist rule (v3b's recurrence flag, v4's single-dip flag). Both habits answer the wrong question for a
technosignature search. What matters is not which known class a star belongs to but what its light curve does that
no simple model predicts. An eclipsing binary is the best-instrumented host there is: once its period and eclipse
profile are known, every epoch has a predicted brightness, so a dip that the eclipses do not account for stands out
more cleanly than on a star whose own variability is unknown. Rejecting the binary throws that away.

So, per star (`model.run`):

1. **Baseline.** The simplest model from a fixed class that explains the dips: constant (no dips), or a strict-period
   eclipse (the EB filter's compression test, `ebfilter.search`, with its decision layer's purity and duration
   checks). The eclipse is accepted only when it saves at least DELTA_BITS over random dip placement, the bound
   that makes P(false eclipse) <= 2^-7 per star.
2. **Flux-level prediction.** Under an eclipse baseline every epoch in an eclipse window gets the deepest depth the
   eclipse can have there: the depth at the same eclipse phase in other cycles (same band first, other bands with
   twice the tolerance, the window's deepest other-cycle epoch if no epoch shares the phase), times (1 + TOL).
3. **Residuals.** An epoch is a residual dip when it is a dip (`ebfilter.search.labels`) and either lies outside every
   window or is deeper than its prediction by RES_SIGMA combined sigma. Explained in-window epochs are dropped from
   the residual light curves; residual dips keep only the excess depth.
4. **What is left** is scored by the tests that already exist, now run on the residual light curves: the number of
   residual events, whether the selected dip is one of them, and the recurrence test (`vetting.recurrence`), so an
   aperiodic train on an eclipsing binary is no longer hidden by the binary's eclipses and an EB's own eclipses no
   longer count as aperiodic events.

A star whose baseline explains everything, the selected dip included, is a known-answer control; every other star
stays in the queue, ranked by what is left. docs/arms/residual.md has the derivation, the pre-registration and the
held-out results.
"""
