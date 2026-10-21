# Population test for one-visit dips (pre-registered plan)

Status: **plan, frozen before any event was extracted from the real light curves.** The commit that adds this
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

Same code. Changes for DR4:

- DR4 epoch photometry covers all sources, so rates become physical and the H0 covariates can include
  non-variable stars.
- The split uses DR4's days, by the same rule and salt.
- Per-CCD G (DR4) adds a time-mask and attribution check.
- DR3 universes give the expected sensitivity, scaled by DR4's star counts and its doubled baseline.

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

- Compact light curves: `python /home/user/pop/stream_compact.py` (this run's container; 3,386 files, 9.3 GB, Stage 0
  applied with blend fractions). G flux and relative error, DPAC flags, FoV, BP/RP fractional change at the same
  transit. Regenerated from the bulk CDN files in about 70 min on 4 cores.
- Astrometry: one TAP query per 1/24 of the sky, joining `vari_summary`, `gaia_source`, `external.gaiaedr3_distance`
  and `vari_classifier_result`. 11,754,237 rows, on the share at
  `data/gaia-dr3-epoch/aux/aux_astrometry.parquet`.
- Code: `cosmos_search/population/` (`run.py` drives the steps, `calibrate.py` the universes).
