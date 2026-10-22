# Population test for one-visit dips (pre-registered plan)

Status: **DR3 rehearsal run (results in §12).** The plan was frozen before any event was extracted from the real light curves. The commit that adds this
file is the pre-registration. Changes after it are listed in §10 with their reason and date, and the DR3 run
reports results under the rules as they stood here.

Why this exists: the single-dip step back (docs/dayone.md, "Step back") found that about three quarters of the
dips Gaia catches fall on one visit, and DR4 keeps the same visit pattern. One star at a time, a dip seen on one
visit cannot be told from an eclipse. Their *rate* can still be tested, star population against star
population. This file is the plan for that test on DR3, as a rehearsal for DR4 (2026-12-02).

## 1. The question, and what it can and cannot answer

**H0.** Which stars show a single one-visit dip is explained by star-level properties (type, brightness,
variability, crowding, cadence and sensitivity) together with eclipses by mundane companions and time-coherent
instrument events. Given those properties, the star's position in space carries no further information.

**H1.** Some other process puts one-off occultations on stars at a rate that depends on where they are: a
region of the Galaxy (on the sky and in distance) with more one-visit dips than its stars' properties predict.

Five points about the framing, from one or two steps back:

1. **Replication confirms stationarity, not strangeness.** Every steady astrophysical cause replicates: YSO
   dippers in a star-forming region, a crowded field, a class of variable star. The split-half test (§5) rules
   out flukes and short-lived artefacts. Only the attribution step (§6) separates "unexplained" from "known
   population". Its expected outcome on DR3 is a set of known regions, which double as positive controls.
2. **DR3 is a biased sample.** DR3 epoch photometry covers only the 11.75 M sources in `vari_summary`, chosen
   for variability, and sometimes chosen *because of* a dip. DR3 rates are not physical rates. The DR3 run
   rehearses the machinery and measures sensitivity to injected excesses. DR4 publishes epoch photometry for
   all sources, so the same test there has a well-defined denominator.
3. **The absolute eclipse rate is uncertain by about a factor of two** (binary fraction, period and mass-ratio
   distributions, and DR3's enrichment in binaries). So the test never compares an absolute count with an
   eclipse prediction. The forward-modelled eclipse rate enters as a covariate with a fitted coefficient (§3),
   and the search looks for structure in position that the covariates do not explain.
4. **No covariate may be computed from the dip.** DPAC's variability class is derived from the light curve,
   dips included: a population of odd dips would be classified as ECL or YSO, and conditioning on class would
   absorb exactly the excess we are looking for. Class, period searches and any EB classifier that reads the
   light curve are used only for attribution (§6), never in the null model. Covariates come from mean
   photometry, astrometry, crowding diagnostics and robust light-curve scale, which one dip barely moves.
5. **Stellar type cannot be tested the same way.** Mundane variability depends on type (YSOs, giants, active
   M dwarfs), and nothing models it per type from first principles. A per-type excess over the eclipse
   prediction is therefore reported descriptively (§7), not as a test.

## 2. Events

Applied identically to real light curves, injected light curves and the mirror (flux flipped about the median).

- **Input.** Gaia G epochs after Stage 0 cleaning (`vetting.stage0.clean_long`, with blend fractions), without
  DPAC variability rejections (`flags & 1`; a secondary run keeps them). Stars need ≥ 15 good G epochs.
- **Scale.** r = flux / median, σ = 1.4826 MAD(r), z = (1 − r) / σ (positive = fainter).
- **Threshold.** K = max(4, the second-largest brightening in z units). The second-largest, so a single bright
  outlier does not blind the star (as in `vetting.single_event`).
- **Visits.** G epochs split into visits at gaps > 1 d. A visit is usually one transit pair (two fields of view,
  106.5 min apart), sometimes two or three pairs 6 h apart.
- **Events.** In-dip epochs (z > K) in the same visit form one event. In-dip epochs in different visits join
  only if no flat epoch (z < 2) lies between them.
- **The class: single one-visit dip (SOV).** The star has exactly one event in its whole light curve; all its
  in-dip epochs lie in one visit; there are ≥ 2 of them (an artefact must hit both fields of view or two
  scans); and there is a flat epoch in an earlier visit and in a later one (bracketed).
- **Monitors, same rules otherwise.** SOV-1 (exactly one in-dip epoch: artefact-dominated, used to build
  the time masks and as a systematics monitor) and the mirror SOV (one-visit brightenings: flares, solar-system
  objects, scan-angle blending).

## 3. Null model

For each star s and half h (§5), the expected number of SOV events λ_{s,h} is built from two forward models
on the star's own epochs, σ and K. Both use the same event rule as §2 and put an event in half h only if its
visit belongs to h.

- **Generic exposure E_gen.** A reference one-off occulter: one event per 1000 d, start time uniform over the
  span, duration log-uniform 0.05 to 30 d, depth log-uniform 0.5% to 50%, trapezoid with random ingress. E_gen is
  the number of SOV events this process would produce on the star. It absorbs cadence, noise, the threshold and
  the scanning law (number and spacing of visits) in one number.
- **Eclipse exposure E_EB.** Companions drawn from `vetting.single_event`'s mundane population (stellar binaries
  with the Raghavan period distribution and dark bodies, eccentricity to 0.7, any impact parameter), host radius
  and mass from `vetting.physics.stellar_params`, a broad prior when the parallax is poor. E_EB is the expected
  number of SOV events from eclipses: an eclipse caught on one visit with no detectable repeat anywhere else.
- **Covariates x_s** (binned, one-hot, main effects plus a colour × absolute-magnitude grid): G (8 bins), BP−RP
  (8), M_G (6 + unknown), log(σ / median error) (6), log(E_EB / E_gen) (6), `ipd_frac_multi_peak` (3), RUWE
  (3), BP/RP blend fraction (3), number of visits (5), and |ecliptic latitude| (3). The last one tests for
  solar-system objects and the scanning law. It is kept only if a likelihood-ratio test on half A gives
  p < 0.01.
- **Model.** log λ_{s,h} = log E_gen,{s,h} + β·x_s, a Poisson GLM with a ridge penalty of 1, fitted per half on
  cells aggregated over identical covariate vectors.
- **Time masks (instrument events).** Per half, SOV, SOV-1 and mirror events are counted in 1 h time bins
  (whole sky) and in 1 h × HEALPix nside-2 bins. A bin is masked when its count exceeds the exposure-weighted
  expectation at binomial p < 10⁻³ / (number of bins). Events in masked bins are dropped and the exposure there
  is removed. Every day belongs to one half, so each half's mask uses only its own events.
- **Excluded a priori.** The LMC (within 8° of 80.9°, −69.8°) and the SMC (within 4° of 13.2°, −72.8°):
  different stellar populations at one distance, where position means something different. They are run as a
  secondary analysis.

## 4. Search (half A only)

- **Windows.** Sky discs centred on HEALPix nside-16 pixel centres (3,072), radii 1°, 2°, 4° and 8°. Each disc
  is crossed with a distance window: one, two or three contiguous shells with edges 0, 250, 500, 1000, 2000,
  4000, 8000 and ∞ pc (Bailer-Jones `r_med_photogeo`, so geometric+photometric), or all distances, including
  stars without a distance. About 2.3 × 10⁵ windows, built from nside-64 pixels × shells.
- **Statistic.** For each window with O > E, the Poisson log-likelihood ratio O ln(O/E) − (O − E). The search
  statistic is the maximum over windows.
- **Look-elsewhere.** 199 parametric null realisations: events redrawn star by star from Poisson(λ_s), maximum
  recomputed, no refit (checked on 10 realisations with refit). The family-wise p of a window is the fraction of
  realisations whose maximum reaches its LLR. As a robustness check on the top window, permute SOV labels within
  full covariate cells.
- **Candidates.** Greedily, the highest-LLR window with family-wise p < 0.05, then the highest among windows
  sharing no star with earlier picks, up to five. Each candidate's p comes from the same max-null, which is
  conservative for the second and later picks.
- **Freeze.** Candidate windows (centre, radius, shells), their O_A, E_A and p, and the code commit go into
  `population/candidates_A.json`. The file is committed, and its SHA-256 recorded, before half B is opened.

## 5. The split, and replication in half B

- **Split rule.** Every visit is assigned to a half by the day number of its first epoch, d =
  floor(BTJD of the first epoch). Half A if the low bit of splitmix64(d XOR 0x706f70756c617465) is 0, else half
  B. The salt is public: the split is not a secret, and half B stays blind because its events are sealed.
  - Day-level assignment puts every star seen on a given day into the same half. Instrument trouble on that day
    then lands in one half only.
  - A random day hash cannot resonate with the scanning law's 63-day precession, as alternating time blocks
    could.
  - Both halves see the same mix of instrument states and scan geometry, which contiguous halves would not.
- **Event halves.** A SOV event belongs to the half of its visit. Events are found on the full light curve
  (§2), so each physical event is in exactly one half. A long dip that touches visits in both halves is not a
  SOV event, so no single dip can count in both halves.
- **Sealing.** The event extractor writes half-A events in plaintext and half-B events (and the half-B mirror and
  SOV-1 sets) only as age ciphertext to `keys/cora.age.pub` and `keys/reveal.age.pub`. Exposures and covariates
  are not outcomes and stay in plaintext.
- **Replication.** Half-B events are decrypted only after `candidates_A.json` is committed. The half-B GLM is
  fitted on all half-B stars with the same covariates. For each frozen candidate k, it gives O_B,k and E_B,k
  and a one-sided Poisson p_k = P(N ≥ O_B,k | E_B,k). **Replicated** means p_k < 0.01, which is 0.05 family-wise
  over five. We also report the excess ratio (O − E) / E in both halves with 90% intervals; a replicated window
  whose half-B ratio is below a quarter of half A's is noted as weak.

## 6. Attribution (replicated windows only, before anything is called unexplained)

In order:

1. **Star-forming regions and young associations** within the window: Taurus, Ophiuchus, Lupus, Chamaeleon,
   Corona Australis, Orion, Perseus, Serpens, Sco-Cen, Cepheus Flare, NGC 2264/Mon R2, Cygnus OB, Carina. This
   is a list with centres, radii and distances from the literature, in `population/regions.py`.
2. **Clusters** (Hunt & Reffert 2023 members, Harris globulars).
3. **The class mix** of the window's SOV hosts against the sample (DPAC `best_class_name`).
4. **Crowding and blending:** the window's `ipd_frac_multi_peak`, RUWE and blend fraction against the sample,
   and whether the mirror class has a coincident excess (blending and scan-angle effects go both ways).
5. **Time structure:** whether the window's events cluster in time (a local instrument event the global masks
   missed).

6. **Be stars, OB associations, extinction:** the window's share of DPAC `BE|GCAS|SDOR|WR`, the OB associations in
   `population/regions.py`, and the members' median `ag_gspphot` against the sample at the same distance.

Whatever survives is "not attributed", never "unexplained": it means not on our lists, and only Stage B's literature
check can say more. Its members go to per-star vetting (Stage A/B with concurrent ZTF and TESS), and the window
becomes a follow-up target list.

## 7. Descriptive outputs (not tests)

- Observed / E_EB per colour × absolute-magnitude cell, in each half, with the halves' agreement.
- The fitted β, and the share of SOV events the eclipse covariate accounts for.
- Time-mask statistics: bins masked, events removed, their sky pattern.
- The mirror class run through the same search.

## 8. Calibration with injected excess populations

- **Universes.** Twelve copies of half A, each the real light curves plus injected one-off dips in 0, 1 or 2
  sealed excess regions. The injection is applied to the compact G light curves before event finding, so the
  event rule, masks and GLM all see the injected dips.
- **Regions.** Centre uniform on the sky outside the excluded zones, radius log-uniform 1–6°, and a random
  distance window (or all distances). In-region stars get a one-off occulter process: duration log-uniform
  0.1–10 d, depth log-uniform 1–30%, grey, with photon-scaled noise. The rate is set so the expected number of
  extra half-A SOV events, from E_gen, is log-uniform 5–300.
- **Sealing.** The plan is drawn from `os.urandom`, written only as age ciphertext to both keys, and decrypted
  in memory to apply. Candidates for all universes are committed, with SHA-256, before the plan is revealed.
- **Scoring.** A region counts as recovered if a candidate window has family-wise p < 0.05 and contains ≥ 50% of
  that region's injected SOV events. We report recovery against injected excess and radius, positional
  accuracy, and the number of candidates that do not match any injection (real-data windows, also present in a
  universe with no injection).
- **Known limit.** Injection happens after Stage 0 (the compact table is already cleaned), so Stage 0's effect
  on grey dips is not part of the measured recall. Its recall on grey dips was measured separately
  (docs/vetting.md §12).

## 9. DR4

Same code and the same rules as §2–§8, with the changes below. They were chosen on DR3 after both DR3 halves
were open (§13), so they are pre-registered for DR4 only; nothing in DR3 tests them.

- **Population.** DR4 epoch photometry covers all sources, so rates become physical and the null covariates
  include non-variable stars.
- **Split.** The split uses DR4's days, by the same rule and salt.
- **Null terms, fixed.** The null is `glm.DR4`: the §3 covariates, |ecliptic latitude| always (no LRT: with
  ~10⁹ stars every term would pass), and three more that never read a light curve:
  - `dens`: log₁₀ of all-Gaia sources per deg² in the star's nside-64 pixel (11 bins, edges 3.75 to 6.0). It
    stands for crowding: contaminated windows, background errors and on-board losses.
  - `vt`: tangential velocity 4.74 μ d (km/s, `r_med_photogeo`; edges 20, 40, 80, 150, 250, 400, plus unknown).
    It stands for population (thin disc, thick disc, halo), and so for age, activity and binary fraction.
  - `gal`: Galactocentric radius × height cells (R edges 3, 6, 10 kpc; |z| edges 0.3, 1, 3 kpc; R₀ 8.2 kpc;
    plus unknown). It stands for the bulge, bar, inner disc and outer disc. Its cells are kiloparsecs across, so
    at 2–8 kpc each spans 10–30° of sky, larger than the largest window. A broad excess over the whole bulge is
    therefore absorbed and is not a candidate; that is deliberate. A bulge-wide rate difference is a population
    effect before it is anything else. §13 measures what the terms absorb.
- **Overdispersion as a function of E, with the SOV-1 monitor (review R21).** On DR3, φ rises with a window's
  expected count within a scale, so a single φ per scale is too small for dense-sky windows and too large for
  sparse ones. Each scale's windows (E ≥ 5) are split into four quantile bins of E; each bin gets the robust φ of
  §10 item 6, made non-decreasing in E. The same curve is fitted for SOV-1 under the same null terms. A window's φ
  is the larger of the two curves, both interpolated in log E at the window's own E. The comparison happens at
  matched counts because SOV-1 has about five times SOV's count per window, and φ grows with the count. Taking
  SOV-1's per-scale φ at its own counts would have charged SOV for SOV-1's larger windows. Replication uses half B's
  own curves in the same way.
- **Window 1 of DR3 (l 348°, b −12°)** sits on SOV-1's strongest hotspot and was explained post hoc. It is
  not cited as a result anywhere. If DR4 finds a window there, it is reported with the SOV-1 count beside it.
- **Cadence before release.** GOST (`gaia/gost.py`) predicts each star's DR4 transits. On 4,950 DR3 stars spread
  over the sky (§13), GOST thinned to the measured retention reproduces the summed E_gen of the real DR3 epochs
  to 7%. On DR4's span it gives 82 epochs and 36 visits per star at the median (DR3: 44 and 19), and 2.07 times
  DR3's E_gen per star, evenly across ecliptic latitude and G. GOST stands in for DR4's epochs only before
  release: for DR4-cadence calibration universes and the power forecast. The DR4 analysis uses the real epochs.
- **Scan coincidence** (attribution, `analysis.scan_coincidence`). For each replicated window, count events whose
  deepest epochs fall within 0.1 d and 2° of another event in the window. Gaia crosses a 2° patch in seconds, so
  such a group is one scan, and independent dips almost never share one. The window is reported with its
  declustered count (each group counted once). It is attributed to the scanning when the declustered count is
  not significant.
- **Per-CCD G** (DR4) adds a time-mask and attribution check.
- **Sensitivity.** A window needs LLR/φ above the max-null's 95th percentile (11.9 on DR3), so an excess of x
  events over E is found when x ≳ 4.9 √(φ(E) E). On DR3-like stars alone, DR4's doubled exposure lowers the
  detectable fractional excess by up to √2, less where φ grows with E. All-source counts will set the real
  figure. DR4 half A will set the non-variable stars' rate, which DR3 cannot.

## 10. Changes after pre-registration

All made on 2026-10-21, before the half-A scan was run on the full data. Only pipeline mechanics and the count of
masked events had been seen at that point; nothing about where on the sky or at what distance events fall.

1. **Time-mask expectation conditioned on position (§3).** The first full mask used each cell's share of all epochs
   as its expectation. It masked 35% of half A's events, mostly in the hours that crossed event-rich regions.
   A steady spatial excess, which is the signal, would have been masked the same way. The expectation is now the
   cell's epochs times its own nside-2 pixel's events per epoch over the half; an hour's expectation is that,
   summed over pixels. The mask now removes 26% of half A's events, 11% of SOV events. Most of it is
   single-epoch brightenings (95,285 of 132,392 masked events), clustered on 123 days.
2. **Masks act on the deepest epoch's hour.** In the forward models, epochs in a masked cell are neither in dip nor
   flat. The half-B mask is computed from half-B events in memory and sealed. Half-B exposures are plaintext and
   therefore show which half-B time cells were masked. That reveals instrument events, not where the excess lies.
3. **E_gen floor.** E_gen below 10⁻⁴ is a Monte Carlo zero (64 draws). It is raised to 10⁻⁴, so an observed event
   never has rate zero.
4. **Eclipse-share bins.** Many stars have E_EB = 0 (no draw survives), so the sextile edges of log(E_EB / E_gen)
   tie and the lowest bins merge.
5. **Checks run on the 40-file dry run.** On 10 null realisations, the maximum LLR with a GLM refit and without one
   agreed (medians 7.0 and 7.0), so the null runs without a refit. The permutation check on the top window draws
   each covariate cell's count in the window from a hypergeometric, which is the same as permuting labels within
   cells.

6. **Overdispersion in the null (red-team review R12, approved by Cora).** The Poisson null is too narrow wherever
   structure the covariates miss (extinction, distance errors, unlisted populations) makes counts vary more than
   Poisson, and that inflates every window's significance in both halves. Each window scale (radius × distance
   window) now gets a dispersion φ = (1.4826 MAD of (O − E)/√E)², floored at 1, taken over all windows of that
   scale with E ≥ 5. The MAD keeps the few windows a real excess would fill from inflating φ. The search statistic
   is the quasi-Poisson LLR / φ against the Poisson max-null. Replication uses a negative binomial with variance
   φ_B·E, where φ_B is half B's own dispersion at the candidate's scale. Poisson-only candidates and p values are
   still reported, for comparison. The same φ is computed for the mirror and SOV-1 classes and reported as a check;
   the primary φ is SOV's own. Attribution (§6) gains Be stars, OB associations and extinction, and survivors are
   called "not attributed".

## 11. Data and code

- Compact light curves: `python scripts/population_stream_compact.py` (3,386 files, 9.3 GB, Stage 0
  applied with blend fractions). G flux and relative error, DPAC flags, FoV, BP/RP fractional change at the same
  transit. Regenerated from the bulk CDN files in about 70 min on 4 cores.
- Astrometry (`scripts/population_fetch_astrometry.py`): one TAP query per 1/24 of the sky, joining `vari_summary`, `gaia_source`, `external.gaiaedr3_distance`
  and `vari_classifier_result`. 11,754,237 rows, on the share at
  `data/gaia-dr3-epoch/aux/aux_astrometry.parquet`.
- Code: `cosmos_search/population/` (`run.py` drives the steps, `calibrate.py` the universes);
  `scripts/population_attribution.py` for §12's attribution refits. §13: `scripts/population_dr4_null.py`
  (LRTs, φ, frozen windows, rescan), `scripts/population_absorption.py` (count injections) and
  `scripts/population_gost_exposure.py` (GOST validation and forecast). The density map
  (`density_hpx6.parquet`, all-Gaia sources per nside-64 pixel), the GOST sample, its transits and the per-star
  exposures are on the share under `population/`.

## 12. DR3 results (2026-10-21)

**Summary.** The machinery works end to end on all of DR3, and its null model is honest only with the
overdispersion term. Half A gives five windows at family-wise p < 0.05. Four replicate in half B, which is what
steady populations do. DPAC class mix plus all-Gaia source density accounts for four of them. The fifth sits toward
the southern bulge and coincides with the strongest hotspot of the single-epoch artefact monitor. Under §6 it is
"not attributed", and the bulge population and DR3's known |b| 10–20° south processing anomaly are the leads.
Injected excess regions are recovered once they add about 12 single one-visit dips to half A, and not at 8 or fewer.

**Sample and events.** 11,327,348 stars with ≥ 15 good G epochs, outside the Magellanic zones. Half A: 26,531 SOV
events before the time masks and 23,131 after (half B: 25,727 after its own mask). The half-A mask removed 26% of all
events and 8.6% of epochs, over 421 whole-sky hours and 560 hour × pixel cells on 123 days. Most of what it removed
is single-epoch brightenings.

**Null model (half A).** The ecliptic-latitude term passed its test (LRT 564 on 2 df), so it stays: rates at
|β| > 15° are 0.6× those near the ecliptic, which points to solar-system objects or the scanning law. The forward
eclipse model carries real information: stars in its top sextile of eclipse share have 12× the SOV rate. Unknown
BP/RP blend fraction gives 3.5×. Overdispersion is large: φ = 2.2–3.5 for all-distance windows (up to 6.1 at some
scales), so a Poisson-only scan would overstate significance. Its maximum LLR is 150, against 43 for the quasi-Poisson
statistic. Refitting the GLM in null realisations leaves the null maximum unchanged (median 9.4 against 9.8).

**Candidates (frozen at 34a5fc2) and replication.**

| window (RA, Dec, radius, distance) | l, b | A: O / E | B: O / E (φ_B) | B p (NB) | replicated | E with class + density, A / B |
|---|---|---|---|---|---|---|
| 273.0, −45.0, 8°, all | 348°, −12° | 517 / 216 | 490 / 238 (4.2) | < 10⁻⁴ | yes | 333 / 385 |
| 286.9, +7.2, 8°, all | 41°, 0° | 559 / 316 | 606 / 341 (4.2) | < 10⁻⁴ | yes | 456 / 513 |
| 208.9, −48.1, 8°, all | 314°, +13° | 256 / 127 | 223 / 115 (4.2) | < 10⁻⁴ | yes | 175 / 164 |
| 284.1, −9.6, 8°, 500–4000 pc | 25°, −5° | 289 / 160 | 155 / 117 (3.2) | 0.033 | no | 226 / 159 |
| 256.5, −60.4, 8°, all | 330°, −12° | 514 / 328 | 583 / 407 (4.2) | 10⁻⁴ | yes | 407 / 517 |

**Attribution.**

- No window overlaps a listed star-forming region at its distance, apart from Serpens/Aquila (window 2) and UCL/LCC
  (window 3), whose hosts are not YSOs.
- The SOV hosts are mostly DPAC ECL, LPV and RS.
- Refitting with DPAC class, then with all-Gaia source density per nside-64 pixel (attribution only, §1 point 4),
  closes most of each excess. Re-scanning half A under class + density leaves one window at p < 0.05:
  window 1, with 517 observed against 333 expected (B: 490 against 385).
- Window 1 in more detail:
  - Under the class + density model, every bulge-direction window at |b| ≈ 12° and 2–8 kpc shows O/E of 1.2–1.4.
    Window 1's position is the strongest of them, at 1.7 in A and 1.4 in B.
  - The single-epoch dip monitor (SOV-1) has its top window there too (272.8°, −41.8°, 4–8 kpc; 2,038 against
    1,153).
  - The leads are the bulge population, which the colour–magnitude covariates do not encode, and the DR3
    rejected-point excess at |b| 10–20° south (docs/infra/gaia-epoch.md).
- The mirror class shows no excess there.

**Calibration (12 sealed universes, 15 regions; candidates committed before the reveal).**

- Recovered: 4 of 4 regions that added ≥ 12 half-A SOV events (12, 15, 32 and 41 events, ranks 1–3), and 0 of 11
  that added ≤ 8.
- The plan's target counts were rarely reached. A region's stars got many dips each, and a star with more than one
  event is no longer a single one-visit dip, so small regions saturate. A second calibration should spread
  strengths across regions of 10³–10⁴ stars.
- The real windows appear as candidates in every universe, as they should.

**Dispersion checks.** Under the same model, the mirror class has φ up to 4.4 and SOV-1 up to 10.9 (8° windows,
4–8 kpc and beyond). With SOV-1's φ the bulge windows would not pass. That is a reason to treat window 1 as an
instrument-or-population lead, not a candidate.

**What this means for DR4.** The test runs on all of DR4 with the same code. On DR3's 11 M stars, on 4 cores, the event
pass took about 5 min and the exposure pass about 70 min. Before DR4:

- Add source density to the null's covariates. It is not read from the dips, and §1 point 4 allows it.
- Add a bulge or Galactic-population term, for example a Besançon-model mix per window or kinematics from proper
  motions, so the null can represent populations.
- DR4's quiet stars change everything measured here. The covariate effects and φ must be re-estimated on DR4's
  half A.
- Use GOST's predicted transits (the coverage thread's `gaia/gost.py`) for exposure and scan-angle attribution.

## 13. DR3 check of the DR4 null (exploratory, 2026-10-22)

Both DR3 halves were open when this ran, so nothing here is a test. It shows what §9's changes do to the DR3
result and how much they cost in sensitivity. Outputs: `population/dr4_null_check.json`,
`population/dr4_absorption.json`, `population/gost_exposure.json`.

**New terms.** Added one at a time to the frozen null on half A, all three are kept by a wide margin: density
LRT 2,470 (10 df), v_tan 212 (7 df), Galactocentric cells 1,205 (16 df). Coefficients are natural-log rate
ratios in the full DR4 null. Pixels above 10⁴·⁵ sources per deg² run 0.3–0.5 above the sparsest, with a dip of
−0.8 in the 10⁴–10⁴·²⁵ bin. The rate rises with height at every radius, by 0.7–0.9 from |z| < 0.3 kpc to
|z| > 3 kpc. The 3–6 kpc ring runs 0.1–0.55 above the solar cylinder at the same height. Within R < 3 kpc the
rate matches the solar cylinder except in the plane, where it is 0.2 lower. Tangential velocity matters only at
the extremes: −0.46 above 400 km/s, and +0.59 for stars with no distance or proper motion.

**Overdispersion.** The new terms barely change it. At the 8°, all-distance scale, φ in half A is 6.1 under the
frozen null and 5.7 under the DR4 null (median over scales 2.2 and 2.2). The excess variance is not crowding or
Galactic structure as binned here. φ rises with a window's E: 2.7 at E ≈ 10 and 5.8 at E ≈ 275 (half A, 8°,
all distances). SOV-1 at the same scale runs 6.0 → 28 over E 10–1,300, and has about five times SOV's count per
window. The per-scale rule of §10 item 6 therefore understates φ for dense windows and overstates it for sparse
ones, and SOV-1's per-scale φ is not comparable to SOV's. §9's curves fix both.

**The five frozen windows under the DR4 null** (O/E, φ at the window's E, one-sided negative-binomial p; single
windows, no look-elsewhere):

| Window (ra, dec, radius, distance) | Half A, frozen null | Half A, DR4 null | Half B, DR4 null |
|---|---|---|---|
| 1. 273.0, −45.0, 8°, all | 517/216, p 7 × 10⁻⁶ | 517/339, φ 13, p 0.010 | 490/394, φ 20, p 0.14 |
| 2. 286.9, 7.2, 8°, all | 559/316, p 9 × 10⁻⁴ | 559/401, φ 15, p 0.031 | 606/445, φ 21, p 0.06 |
| 3. 208.9, −48.1, 8°, all | 256/127, p 8 × 10⁻⁴ | 256/161, φ 8, p 0.011 | 223/151, φ 14, p 0.07 |
| 4. 284.1, −9.6, 8°, 0.5–4 kpc | 289/160, p 0.001 | 289/181, φ 12, p 0.019 | 155/124, φ 11, p 0.19 |
| 5. 256.5, −60.4, 8°, all | 514/328, p 0.007 | 514/476, φ 17, p 0.32 | 583/606, φ 25, p 0.55 |

The density and Galactic terms raise E by 15–45%. The SOV-1 curve raises φ by 1.5–2×. None of the five would
replicate at p < 0.01 under the DR4 rules. Window 1, the one §12 left "not attributed", is about 1.5× the
DR4 null in both halves. That is in line with the other four and with a bulge-direction population that the
kiloparsec cells only partly capture.

**Half-A rescan under the DR4 rules.** The maximum LLR/φ over all windows is 12.6, against a null 95th
percentile of 11.9. One window passes (family-wise p 0.025): 146.3°, −27.3°, 4°, 250–500 pc, 12 events on
E 1.5. It has 1 event on E 1.3 in half B, so it does not replicate. All 12 hosts are DPAC `SOLAR_LIKE`, the dips
are shallow (1–6%, two in-dip epochs), and three share one scan (§9's scan coincidence: 10 declustered). It is a
low-count fluctuation among spotted stars, the class the single-dip arm already found to be its main false
positive.

**Absorption** (37 universes; 40 were planned, and a container restart stopped the run). Each universe adds
Poisson(f λ) events to the stars of one random region (radius 4°, 8° or 15°, a random shell window, away from
the real windows), with 7–600 added events. With and without the new terms, about 1–3% of an injected excess
moves into the refitted null: the kiloparsec cells fill with stars from far outside any one region. Recovery is
12/37 for the frozen null with per-scale φ and 11/37 for the DR4 null with the SOV-1 curves. By injected count
over √E: 1/13 and 1/13 below 3 (chance overlaps with nearby fluctuations); 0/9 and 0/9 from 3 to 12; 5/9 and 4/9 from 12 to 30; 6/6 and 6/6 above 30. The
DR4 rules cost little sensitivity. Everything in DR3 below about 12 √E is lost under either null, which matches
the §9 estimate x ≳ 4.9 √(φ E) for φ of 5–10.

**GOST.** 4,950 DR3 stars from 100 random compact files (HEALPix-8 ranges), queried in 1,000-target jobs. Over
DR3's span, GOST predicts every transit; Gaia kept a median 89% of them as good G epochs (91% at G < 17, 77% at
G 19–21). Thinned at random to 89%, GOST's summed E_gen is 0.93 of the real epochs' (unthinned 1.13). The
per-star scatter (16–84%: 0.52–1.60) is mostly Monte Carlo noise. Real losses cluster: stars that lost many
transits are overestimated by random thinning (1.48 at retention < 70%) and full-retention stars underestimated
(0.83). Cells sum many stars, so the 7% aggregate bias is what calibration on GOST cadence inherits. For
per-star use, thin by each star's measured retention. On DR4's span, the median star has 82 epochs and 36
visits (DR3 44 and 19). E_gen per star is 2.07× DR3's (median 2.06, 16–84% 1.17–3.74), within 2.0–2.25 in every
G and ecliptic-latitude bin. Half A holds 49.4% of it.

