# Non-Keplerian astrometry

Narrow secondary arm. Question: does any star move in a way that gravity from unseen companions cannot
explain?

## Premise

An isolated system bound by gravity produces sky motion that is a sum of a 5-parameter single-star term
and Keplerian orbits (plus tiny relativistic and light-travel corrections). Anything else needs another
cause: a resolved or unresolved blend, a variable component shifting the photocentre, an instrument effect,
astrometric microlensing, or a force that is not gravity (thrust, radiation pressure on a sail, mass loss).
The search is model comparison across a nested family, with every natural non-Keplerian effect modelled
explicitly so that what remains is genuinely unexplained.

## Data

DR4 epoch astrometry: 1-D along-scan positions per CCD (129.6 × 10⁹ rows, 62 TB), with uncertainties and
scan angles. The June 2026 pre-release gives 12 sources (single stars of various magnitudes and parallaxes,
QSOs, and three non-single stars with published Keplerian orbits) and the "Gaia source update" tool that
fits the default single-star model. The `exogaia` package fits pre-release epoch astrometry too.

## Models, nested

1. Single star (5 parameters).
2. + constant acceleration (7) and jerk (9), as Gaia's own acceleration solutions do.
3. + one Keplerian, + two Keplerians.
4. Nuisance terms added to every model:
   - **Variability-induced motion**: the photocentre of an unresolved pair moves along the pair's axis
     when one component varies. With simultaneous epoch photometry, VIM predicts residual ∝ ΔF projected on
     the scan direction. Regress residuals on flux.
   - **Scan-angle-dependent blending**: residual as a low-order Fourier series in scan angle.
   - Perspective acceleration for nearby high-radial-velocity stars.

## Statistics

For each source, compare the best model in the family by evidence or an information criterion, then test
the best model's residuals for whiteness (runs, autocorrelation against time and scan angle). A candidate
has (a) a significant preference for structure beyond the best Keplerian model and (b) residuals that
survive the nuisance terms. Calibrate on the pre-release single stars and on residual streams with
injected non-gravitational accelerations (milestone A3), and apply the same multiple-testing treatment as
the dimmer arm.

## Log

Record changes of approach here, with the evidence that prompted them.
