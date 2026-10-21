# Statistics

The census only works if "explained" has an operational definition that a classical pipeline can test and
an expert can be audited against. Everything below follows from that.

## 1. What "explained" means

Each object gets a set of candidate generative models M_k, each predicting every data modality we hold for
it: SED, CMD position, astrometry, variability, spectra, X-ray and radio levels. An object is **explained**
when at least one physically plausible model passes a posterior predictive check in *every* modality at a
pre-set level. It is **unexplained** otherwise, and goes to the expert queue.

Classical model families for tier 1, roughly in order of how much of the sample they should absorb:

| Family | Predicts | Main ingredients |
|---|---|---|
| Single main-sequence / evolved star | SED, CMD, activity level | MIST or PARSEC isochrones, BT-Settl / PHOENIX atmospheres |
| Unresolved binary (MS+MS, MS+WD, WD+BD) | SED excess, CMD offset, RUWE, NSS | pairs drawn from the single-star grid; Gaia NSS solutions |
| White dwarf (DA, DB, DC, DQ, DZ; magnetic flagged) | SED, CMD, XP features | Bergeron / Koester cooling tracks and atmospheres |
| Ultracool / brown dwarf | SED, CatWISE motion, IR colours | Sonora Bobcat / Elf Owl, BT-Settl CIFIST |
| Young star with disk or debris | IR excess at fixed T_dust, variability | modified blackbody excess on the stellar model |
| Variable classes | light-curve statistics | rotation, eclipsing, pulsator, flare models |
| Active star | X-ray, radio | L_X/L_bol saturation near 10⁻³, Güdel–Benz L_X/L_ν,R ≈ 10^15.5 Hz |

Extinction is near zero inside the Local Bubble, but not exactly zero toward the 100 pc edge; carry
E(B−V) from a 3D map (Edenhofer et al. 2024 or Bayestar19) as a nuisance parameter rather than fixing it.

The catch-all is important. Model comparison among only "known" models will always pick a winner, so each
object also needs an absolute goodness-of-fit, and the population itself provides an "ordinary" density
(empirical Bayes: the GCNS CMD and colour–colour density) against which low-density objects stand out.

## 2. Sample definition and the 100 pc boundary

- Parallax zero-point: apply Lindegren et al. (2021) per source (the `gaiadr3_zeropoint` package). At
  10 mas the correction (~−0.02 mas) is negligible for membership but matters for consistency with DR4.
- Parallax uncertainties are underestimated, by up to ~30% for bright sources (El-Badry et al. 2021, from
  wide binaries). Inflate σ_ϖ with their f(G) before computing membership.
- Spurious solutions: at ϖ > 10 mas, GCNS keeps 3.3 × 10⁵ of the 5.7 × 10⁵ DR3 rows, so up to ~40% are
  artefacts (crowded fields, close pairs). Use the Rybizki et al. (2022) astrometric fidelity and
  reproduce the GCNS classifier as a cross-check.
- Membership: P(r < 100 pc | ϖ, σ_ϖ) with a uniform space-density prior (nothing fancier is needed at
  ϖ/σ > 5). Carry the probability, do not hard-cut. Report expected counts as Σ P and treat objects with
  0.05 < P < 0.95 as boundary objects. Lutz–Kelker-type bias matters only for those.
- Pull a ϖ > 8 mas buffer (1.2 × 10⁶ rows) so boundary objects and injections near the edge have context.
- Completeness: Gaia's selection function (`gaiaunlimited`, Cantat-Gaudin et al. 2023) for the in-Gaia
  part. Known holes: brightest stars (G ≲ 3, use Hipparcos), T and Y dwarfs (too red and faint), white
  dwarfs fainter than G ≈ 20.7 (cool WDs beyond ~60–80 pc), and close pairs below Gaia's ~0.4–2 arcsec
  pair resolution. The non-Gaia population comes from CatWISE2020 motions, the UltracoolSheet and the
  20 pc census (Kirkpatrick et al. 2024).

## 3. Cross-matching

Nearby stars move fast. Barnard's star moves 10.4 arcsec/yr, and 2MASS (≈1998–2001) is ~17 years from the
Gaia epoch (J2016.0), so naive positional matching fails for exactly the objects we care about.

- Propagate every Gaia position to each survey's per-source epoch (not the survey's mean epoch) with the
  full 5-parameter covariance, then match within an error ellipse.
- Use Bayesian match probabilities with a source-density prior (Budavári & Szalay 2008). For X-ray
  (eROSITA, ~5 arcsec errors) and radio (RACS ~2–3 arcsec, VLASS ~0.5–1 arcsec) use likelihood-ratio
  counterpart assignment with priors on which stars emit (NWAY-style, Salvato et al. 2018).
- Measure the chance-coincidence rate empirically: re-run each match with the input positions shifted by
  1–2 arcmin in a few directions. That shifted-sky rate is the false-match floor for every multiwavelength
  "anomaly" downstream.
- Gaia's precomputed best-neighbour tables (2MASS, AllWISE, PS1) are a useful check, not a replacement;
  they use the mean epoch and will miss some fast movers.
- Solar-system objects contaminate single-epoch WISE and ZTF detections. Model them (Gaia DR3 SSO table,
  MPC ephemerides via SkyBoT) rather than leaving them to show up as "transients".

## 4. Multiple testing

With ~4 × 10⁵ objects, ~6 modalities and many features per modality, an analytic 3σ threshold produces
thousands of false anomalies. Plan:

1. Per object and modality, compute a posterior predictive p-value against the best model.
2. Calibrate those p-values **empirically**, not analytically. Gaia and survey errors are heavy-tailed and
   correlated, so the analytic null is wrong. The calibration set is the sample itself with known-ordinary
   objects (e.g. well-measured single main-sequence stars in clean regions) and null rehearsals with no
   injections.
3. Combine modalities per object (minimum p with a Šidák correction for the number of modalities, or a
   Cauchy combination, which is robust to correlation between modalities).
4. Control the false discovery rate across objects with Benjamini–Hochberg (or Storey's q-values if the
   null fraction is well below 1).
5. For periodograms and transit searches the look-elsewhere effect is per light curve: use bootstrapped or
   injection-calibrated false-alarm probabilities, not analytic ones.

The expert queue has a fixed capacity, so in practice the q-value threshold will be set by budget. That is
fine as long as we report the recall at that cutoff for each injected anomaly class (§5).

## 5. Injection-recovery

Inject into **real** data products, so real systematics stay in, at the level the pipeline reads them:
light curves for time-domain signals, photometry tables for SED signals, astrometric solutions for wobbles.

| Class | Injected into | Parameter grid |
|---|---|---|
| Transits / eclipses | TESS, ZTF, ASAS-SN light curves | depth, period, duration |
| Flares, dips, long-term fading | same | amplitude, rate, timescale |
| IR excess (disk, debris, "waste heat") | 2MASS/WISE/unWISE photometry | T_dust 50–1000 K, L_IR/L_* 10⁻⁵–10⁻¹ |
| Astrometric companion | Gaia RUWE / NSS; DR4 epoch astrometry | mass ratio, period |
| X-ray / radio emitter | eROSITA / VLASS / RACS catalogue level | flux relative to activity relations |
| Spectral oddities | XP, RVS, LAMOST/SDSS/DESI spectra | line strength, unexpected species |
| Kinematic outliers | proper motion / RV | velocity offset |

Report completeness C(class, parameters) as maps, not single numbers. Split injections into a **tuning
set** (visible, used to set thresholds) and a **sealed set** (encrypted truth, see blinding.md), and report
only sealed-set numbers as final.

## 6. Decoys and controls for expert subagents

The main failure mode for LLM experts is confabulation: producing a fluent, physically plausible story for
an object that has none. Decoys exist to measure that. Mix into the expert queue, at a fixed hidden
fraction (~10%):

- **Incoherent decoys**: injected signals with no consistent physical explanation (e.g. an IR excess
  inconsistent with any dust temperature and the measured luminosity). Correct answer: "unexplained" or
  "data problem". Rate of invented explanations = confabulation rate.
- **Coherent rare decoys**: physically consistent injections of rare scenarios (a WD+BD unresolved pair,
  an edge-on disk). Correct answer: that scenario.
- **Known-answer positives**: real, well-studied odd objects with identifiers stripped.
- **Negative controls**: ordinary objects that the classical tier passed. Correct answer: "ordinary".

Score accuracy per decoy type, and require experts to state a probability; check calibration with Brier
scores and reliability curves. An expert verdict is only usable if its calibration on decoys is acceptable.

**Budget.** At ~4 × 10⁵ objects, a classical tier that explains 99% leaves ~4,000 for experts; 99.9% leaves
~400. The DR3 rehearsal should tell us which regime we are in before we commit expert time to DR4.

## 7. DR3 rehearsal → DR4

- Validate the DR3 pipeline against existing censuses: GCNS (100 pc), CNS5 (25 pc, 5,931 objects),
  the 40 pc white dwarf sample (O'Brien et al. 2024), the 20 pc ultracool census (Kirkpatrick et al. 2024).
  Metrics: recovery of known classes, and the rate at which well-understood objects are flagged anomalous.
- Freeze code, model grids and thresholds before 2 December (`dr3-frozen` tag) and write the DR4 analysis
  plan down in the repo first. DR4 should be a re-run, not a fresh round of tuning.
- DR4 mean products (astrometry, XP, RVS, NSS, variability) slot straight into the frozen pipeline. Epoch
  astrometry and epoch XP/RVS spectra are new modalities; they get their own injection classes after the
  first DR4 pass.
