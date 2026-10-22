# Non-Keplerian astrometry

Narrow secondary arm. Code: `cosmos_search/astrometry/`. Owner: "Astrometric non-Keplerian filter" thread.

Question: does any source move on the sky in a way that no gravitational or ordinary optical explanation fits, and
does any orbit or acceleration need a mass that nothing natural supplies?

## First principles: what can move, and what it costs

**A star's mass cannot be pushed measurably.** Gaia DR4 measures an acceleration to about 0.2 σ_tr mas/yr² (σ_tr
the per-transit along-scan error, ~90 transits over 5.5 yr on GOST cadence). A 5σ detection on a G = 12 star at
10 pc is 0.04 mas/yr² = 6 × 10⁻⁸ m/s². A photon rocket of power L gives thrust L/c, so moving one solar mass that
hard takes L = M a c ≈ 10⁵ L☉ of collimated light; at 100 pc or G = 17 it is 10⁶ L☉ or more
(`physics.luminosity_for_acceleration`). A Shkadov thruster on a Sun-like star manages ~10⁻¹² m/s². Brown dwarfs
gain a factor 20. So a non-Keplerian **mass** motion of a stellar source is excluded by five orders of magnitude,
whatever builds it.

**The light centroid is cheap to move.** What Gaia measures is the photocentre of the light in its window. That moves
without any force when light is added, removed or redistributed: an unresolved neighbour pulling the centroid as
the scan angle turns, one component of an unresolved pair varying (variability-induced motion, VIM), a dark lens
passing in front (astrometric microlensing), convection cells on a giant, an occulter large enough to unbalance a
resolved stellar disk, or the instrument. So the arm asks two questions:

1. **Light-centroid motion no natural model explains.** Every natural centroid mover is modelled explicitly, each
   model must also be physically possible, and only then is the residual examined.
2. **Gravity that needs too much mass.** Kepler's third law gives the least total mass any gravitational
   explanation needs. An orbit or acceleration that needs more than the luminous star (and adds no light) has a dark
   companion; one that needs more than ~100 M☉ is beyond every known stellar remnant. Both are individually
   valuable and cheap to verify with radial velocities.

A third use falls out of the same fits: **where a dip happens.** For the dimmer arms, the centroid shift during a
dip says whether the dimming source is the catalogued star or something beside it (below).

## Data

DR4 epoch astrometry: per FoV transit, nine AF CCD along-scan (AL) centroids with errors, scan angle, AL parallax
factor, flags. The June 2026 pre-release gives 12 sources (single stars, QSOs, HD 114762, Gaia-4, Gaia BH3) in the
final format, with ESA's `gaia-supdate` fitter. The AL model for a single star is linear (`epoch`, `models`):

    w = Δα* sin ψ + Δδ cos ψ + ϖ f_AL + μα* t sin ψ + μδ t cos ψ,   t from J2017.5

`epoch.per_transit` combines the CCDs of a transit (40 s, so no source motion within it) and keeps the
intra-transit χ², which measures whether the per-CCD errors are right for that source independently of any motion
model.

**Checked on the real pre-release** (`tests/test_astrometry.py`):

| check | result |
|---|---|
| Our AL parallax factor (Earth-Moon barycentre × 1.01, `geometry`) vs DR4's | max difference 2.5 × 10⁻³ |
| GOST's predicted scan angle vs DR4's `scan_pos_angle` (matched transits, 12 sources) | 0.002–0.003° |
| 5p parallaxes | HD 183633 1.001, LSPM J1324+6124 10.105, … (the published values) |
| BH3 orbit from scratch | P = 4188 d (published 4253 ± 98), e = 0.73, a₀ = 27.1 mas, f_A = 33 M☉ (M_BH³/M² ≈ 31) |
| HD 114762, Gaia-4 | P = 83.8 d, 579 d (published 83.9, 571) |
| Intra-transit χ²/dof | 0.95–1.10 for G ≥ 13; 1.2–3.1 for G < 12 (bright stars' per-CCD errors are underestimated) |
| Single-star residuals | Gaussian: kurtosis 3.1, none beyond 4σ in 631 |
| Full cascade on all 12 | singles and QSOs → 5p, the three binaries → K1, no non-Keplerian flag, FoV pairs consistent |

Per-transit noise for simulation, fitted to these 12 after the intra-transit inflation:
σ_tr² = 0.037² + 0.056² 10^{0.4(G−15)} + 0.012² 10^{0.8(G−15)} mas (0.040 at G 12, 0.051 at 14, 0.58 at 19,
1.3 at 20). G 15–18 is interpolated: no pre-release source sits there.

## The cascade (`detect.cascade`)

Per source, on per-transit abscissae, with the per-transit G flux (DR4 epoch photometry):

1. **Natural family**, each nested on the 5-parameter star, VIM (offset ∝ 1/F, two columns) in every model:
   5p; 7p and 9p (acceleration, jerk); K1, a photocentre orbit searched on the bare star and on top of an
   acceleration (circular periodogram 0.5 d – 40 yr, eccentricity × periastron grid, Nelder-Mead); K2, a second
   orbit on top of K1; **lens**, astrometric microlensing by a dark lens, chosen jointly with the magnification it
   must also produce in the photometry; **nb**, a point neighbour (separation, angle, flux ratio, relative motion,
   LSF width) whose pull on the centroid is switched by the window as the scan angle turns. Two nuisance blocks are
   tried on top of the winner, alone and together: odd scan-angle harmonics (3ψ, 5ψ), and VIM whose lever arm
   drifts linearly in time (a neighbour moving across the target's LSF changes both the flux it adds and the
   direction it pulls).
2. **Physical plausibility.** An orbit is removed from the natural family if, even with 100 M☉ in total, its
   periastron would be inside the luminous star (eccentric fits that park a periastron spike on one or two
   outlying transits), or if its mass function exceeds 100 M☉. A lens is removed if the photometry rejects its
   magnification (Δχ² > 25). The best remaining model by BIC is the natural explanation.
3. **Noise.** If the fitted VIM is large, flux errors themselves move the centroid; that variance is added and the
   cascade runs again. Single transits more than 5σ from the best natural model (at most 2% of transits) are
   dropped and the cascade runs again: a jump or any sustained signal is never one transit.
4. **Statistics.**
   - `dbic_cp`: a change point (position jump or proper-motion kick) on top of, or instead of, the best natural
     model. Positive when the data prefer the change point.
   - `dbic_implausible`: how much better a physically impossible orbit or lens fits than every possible natural
     model. Positive when the data demand what gravity or lensing cannot provide.
   - `gof_z`: Wilson–Hilferty z of the best natural model's χ².
   - `fov_pair_z`: do the residuals of transit pairs 106.5 min apart (preceding then following field of view) agree?
     Both fields see the same sky through different optics and calibrations; an astrophysical signal agrees, a
     per-FoV calibration jump does not. This is the arm's instrument veto (plan principle 4).
   - Also `ruwe` (what the catalogue would show), `k1_fA`, `k1_verdict`, `acc_min_mass`.

## Simulation (`simulate`)

Cadence and geometry from GOST predictions for 1,000 real stars spread over G 10–20.5 (thinned to 85%, the DR4
used/predicted ratio on the pre-release), each simulated source borrowing one star's position, G and parallax.
Noise from σ_tr(G), with a per-source error in the assumed σ (log-normal, 8%) and 0.3% of transits as 6σ
outliers.

Natural classes: single (45%), binary with a luminous companion (28%: Raghavan log-normal periods, q flat, light
ratio q⁴), dark companion (3%: WD/NS/BH), hierarchical triple (6%), blend (8%: neighbour at 30–1500 mas, flux ratio
0.01–1, half moving relative to the target at ~5 mas/yr; centroid and flux from a single-LSF fit to the two-Gaussian
image, hard 1-arcsec window, so a neighbour close enough to pull the centroid also adds flux),
VIM (4%: unresolved pair at 5–60 mas, periodic or red-noise variability), giant (5%: convective photocentre jitter,
6% of the angular radius, correlation time 0.1–1 yr), microlensing (1%, with its magnification). The detector's
neighbour model differs from the simulator's on purpose (small-pull approximation, logistic window edge, free
amplitude, LSF width fitted) so that calibration does not reward a model that matches the simulator exactly. Blends
also set the per-transit multi-source flag when the neighbour is more than 120 mas along scan and brighter than
b = 0.06.

Non-gravitational families, amplitude in units of σ_tr: **step** (centroid jump), **kick** (proper-motion change),
**thrust** (acceleration switched on for 20–60% of the span), **square** (periodic square-wave displacement, a
shape no orbit makes). Instrument family: **fov_jump** (a jump in one field of view only).

## Decision rule (`rule.py`, astrometry-v1)

Labels, in order of precedence:

- **instrument**: `fov_pair_z > FOV_Z`. The two fields of view disagree, so the signal is not on the sky.
- **neighbour**: more than `FRAC_MULTI` of the transits carry Gaia's multi-source flag. The centroid belongs to a
  known blend; no non-Keplerian claim is made for it. This label is a coverage cost: in DR3 (G < 16, ϖ/σ > 10, a
  random 1% of the sky) the share of sources with `ipd_frac_multi_peak` > 10% is 0.5% at RUWE < 1, 2% at
  RUWE 1–1.5, and 19–43% at RUWE 1.5–8, so roughly a third of the high-RUWE prefilter population goes to this
  label. Those sources need a resolved-neighbour model built from Gaia's own detections of the neighbour, a
  separate piece of work.
- **unexplained**: `S = max(dbic_cp, dbic_implausible) > S_MIN` and `S > REL × max(χ²_5p − (n − 5), 1)`. The data
  prefer a change point or a physically impossible model over every possible natural one, by a margin that is also
  a non-negligible share of the source's whole signal.
- **natural**: everything else.

### Pre-registered threshold procedure (written before calibration run 5 was read)

`FRAC_MULTI = 0.10`, from the DR3 numbers above (single-like stars exceed it at 0.5–2%). The other three are set
mechanically on run 5 (2,400 naturals, seed 51; 1,600 non-gravitational injections, seed 52; 400 FoV jumps,
seed 53; amplitudes log-uniform 1–50 σ_tr):

1. `FOV_Z`: the smallest of {4, 5, 6, 8, 10} that at most 0.5% of naturals and non-gravitational injections exceed.
2. `REL`: the largest of {0, 0.01, 0.02, 0.05, 0.1} that lowers recall of non-gravitational injections with
   amplitude ≥ 20 σ_tr by at most 2 percentage points against `REL = 0`.
3. `S_MIN`: the smallest of {20, 25, 30, 40, 50, 70, 100, 150, 200} at which at most 0.5% of naturals (all classes
   pooled, labels applied in the order above) are labelled unexplained.

### Blind test

A sealed batch (`scripts/astrometry_blind.py`): 800 naturals in the calibration mix, 240 non-gravitational
injections and 60 FoV jumps, amplitudes log-uniform 3–50 σ_tr, shuffled under anonymous ids, seed from
`os.urandom`, truth sealed with age to Cora's key and the reveal key. Flags are committed and pushed before the
reveal step will decrypt. Pass criteria, fixed with the thresholds:

- **P1** natural false-alarm rate: Clopper–Pearson 95% upper bound ≤ 1.5%.
- **P2** recall of non-gravitational injections with amplitude ≥ 20 σ_tr: at least run 5's recall minus 10 points.
- **P3** instrument: ≥ 80% of FoV jumps with amplitude ≥ 10 σ_tr labelled instrument, and ≤ 10% of all FoV jumps
  labelled unexplained.

The blind batch is drawn from the same simulator as the calibration, so it tests that the frozen thresholds and the
pipeline generalise to fresh draws and that nothing leaks; it cannot test whether the simulator's natural mix
resembles the sky. That test is DR4 itself (see "DR4 day one").

## Calibration

CALIBRATION_RESULTS

## DR3 rehearsal: the Kepler bound on Gaia's own solutions (`kepler_bound`)

DR3 has no epoch astrometry, but it has 169,227 astrometric orbits (`nss_two_body_orbit`) and 338,215 acceleration
solutions (`nss_acceleration_astro`). `scripts/astrometry_dr3_nss.py` fetches them with the gaia_source and FLAME
columns; data in `/mnt/project-files/astrometry/dr3/`.

For orbits: a₀ from the Thiele–Innes constants and f_A = (a₀/ϖ)³/P². For accelerations: the least mass that gives
the acceleration without visible curvature, M_min = (g/ϖ)³ (2T)⁴ / (2π)⁶, and for 9-parameter solutions the
circular-orbit mass g⁷/(4π² j⁴).

| | f_A or M_min > m₁ ("dark") | > 100 M☉ ("beyond") |
|---|---|---|
| orbits | 52 (36 without flags) | 23 (1 without flags) |
| accelerations | 44 (5 without flags) | 1 (flagged: goodness of fit 8.2) |

What fills the tail:

- **Gaia's scanning law.** 13 of the dark/beyond orbits have periods within 2% of 63.12/k days (k = 1–10), the
  precession period of the spin axis and its harmonics. 28 are `OrbitalAlternative` solutions with P < 100 d
  (17 of the 19 orbits above 100 M☉ with ϖ/σ > 5).
- **Low parallax** (14): f_A goes as ϖ⁻³.
- **Bad fits** (goodness of fit > 5): 8 orbits, 40 accelerations.

At the top of the unflagged list sit the known black holes: **Gaia BH1** (f_A = 11.1 M☉; it is flagged as a
half-year alias, P = 185.6 d vs 182.6, which is why aliases are flags and not vetoes: at DR3's frequency resolution
the two cannot be told apart) and **Gaia BH2** (6.7 M☉). The filter reproduces what took the field a year of
follow-up in 2022–23, from the catalogue alone and with no tuning on them.

**Leads to test on DR4 day one** (one DataLink pull each; DR4's 66 months cover both):

- **Gaia DR3 4041697094653956736** (G = 11.9, F star, R ≈ 2.2 R☉, ϖ = 1.81 mas, bulge direction l = 356.6°,
  b = −3.5°). Acceleration9 with g = 43 mas/yr² and jerk 67 mas/yr³: M_min = 230 M☉, circular-orbit mass ≈ 60 M☉ at
  P ≈ 4 yr. Goodness of fit 8.2 and a crowded field make a spurious solution likeliest. No paper mentions it (ADS
  full text, 2026-10-22). DR4 will show either an orbit with f_A of tens of M☉ or nothing.
- **Gaia DR3 4467000291193143808** (G = 15.5): Orbital, P = 1648 d (longer than DR3's 1038-d span), e = 0.88,
  f_A = 112 M☉. Discussed and set aside in El-Badry et al. 2023 (the BH1 paper). DR4 covers the full period.

## Dip localisation for the dimmer arms (`dips.localise`)

Two unresolved components, the varying one at offset ρ from the steady light: the centroid moves along ρ linearly in
1/F, with coefficient c, and |c| = ρ × (flux share of the steady part) ≤ ρ. So a fitted c is a lower bound on how
far the dimming source sits from the rest of the light. For one star dimmed by something in front of it, the
centroid can move by at most about the star's angular radius θ*.

Gaia measures along scan only, and a dip seen in one visit has one scan angle, so the bound is usually 1-D (along
that scan). On 3,000 simulated dips on GOST cadence:

| median 95% bound along scan, on-target dips | depth 3–10% | 10–20% | 20–50% |
|---|---|---|---|
| G 10–13 | 1.5 mas | 0.52 | 0.22 |
| G 13–15 | 2.0 | 0.64 | 0.25 |
| G 15–17 | 4.0 | 1.5 | 0.51 |
| G 17–19 | 6.0 | 3.2 | 1.4 |

On-target dips are called "neighbour" in 1 of 758. Dips on a neighbour are recognised at ≥ 72% for separations
30–100 mas at G < 15 and ≥ 92% beyond 100 mas, falling to 3–18% at G > 19. A deep dip on a bright star is placed on
the target to within ~0.25 mas, which excludes essentially every chance-aligned background star; a significant
offset smaller than θ* would mean an occulter large enough to unbalance a resolved disk, measurable only for nearby
giants (θ* of 0.1–10 mas).

## DR4 day one

In dependency order, from the release (2026-12-02):

1. **Kepler bound on the DR4 NSS tables** (`kepler_bound.orbits`, `kepler_bound.accelerations`): catalogue-level,
   minutes. Classes `dark` (needs a companion heavier than the luminous star and invisible) and `beyond` (needs more
   than 100 M☉, or periastron inside the star). Flags, not vetoes, for precession and half-year aliases, short
   `OrbitalAlternative` periods, low parallax, poor fit and crowding. Every unflagged `beyond` source and the top of
   `dark` get their epoch astrometry pulled through DataLink and the full cascade: the catalogue solution is one
   model; the cascade asks whether any possible model fits.
2. **The two DR3 leads**, first in the queue: Gaia DR3 4041697094653956736 (Acceleration9 solution, bulge field
   l = 356.6°, b = −3.5°, G = 11.9, minimum companion mass 230 M☉ with no visible curvature, goodness of fit 8.2, no
   literature) and 4467000291193143808 (f_A = 112 M☉ at P = 1,648 d, longer than DR3's span; in the El-Badry et al.
   2023 BH1 sample). DR4's 66-month span and per-transit data settle both: a real orbit closes, a crowding artefact
   shows up in the FoV-pair test or the multi-source flag.
3. **Epoch-level search.** Downloads, not CPU, set the scale: the full cascade costs ~3 s per source per core and
   the linear first stage (`detect.quick`: change point on the 9-parameter star, VIM and scan-angle terms) 7 ms.
   Prefilter on the catalogue's RUWE, which the injections show responds to the signals this arm can claim: RUWE >
   1.4 keeps 80% of 5–10 σ_tr steps and all square waves above 5 σ_tr; RUWE > 2 keeps only 17% of 5–10 σ_tr steps.
   In DR3 the G < 16, ϖ/σ > 10 sample has ~3.3 M sources at RUWE > 1.5 (55.6 M in all; a 1.1% random sample, 2026-10-22). Order: pull epoch data for
   RUWE > 1.4, run `quick` on all of it, then the cascade in descending `quick_S`; the decision rule above labels
   the output, and the top of `unexplained` goes to vetting within the expert budget.
4. **Neighbour stream**: sources labelled `neighbour` with large S get a resolved-neighbour model seeded from Gaia's
   own detection of the neighbour (position, G, proper motion). Not built yet.
5. **Dip localisation for the dimmer arms** (`dips.localise`): every dimmer candidate with epoch astrometry gets an
   on-target / neighbour / disk-scale verdict and a 95% bound on the light centre's offset during the dips.

## Log

- 2026-10-22: arm built from scratch. Reframed from "non-Keplerian motion" to light-centroid motion plus Kepler-bound
  tension after the energy bound above. First calibration (2,000 naturals) showed blends crossing the window edge
  and variable unresolved pairs as the dominant false alarms (43% and 66% with Δ BIC > 10 for a change point), and
  that eccentric orbits and microlensing absorb jumps; added the neighbour model, always-on VIM, flux-error
  propagation, the lens magnification check, periastron and mass-cap plausibility, 5σ transit clipping and the FoV
  pair test.
