# Cross-match (`cosmos_search.xmatch`)

Milestone I5. One call takes a table of Gaia sources to calibrated match probabilities in another survey,
with the false-match rate measured on the real sky alongside:

```python
from cosmos_search.xmatch import crossmatch
res = crossmatch(gaia_table, "allwise")    # HATS or TAP chosen per survey
res.best      # one row per target (and survey group): best candidate, p_best, p_cal, p_none, p_chance_0.9
res.matches   # every (target, candidate) pair with P >= 0.01
res.report    # fitted error model, prior by G, false-match accounting, chi^2 check
```

```
python -m cosmos_search.xmatch targets.parquet --surveys allwise 2mass --out xm/
python -m cosmos_search.xmatch targets.parquet --context --out xm/    # one wide table, all default surveys
```

The DR4 dimmer arm uses it for the energy-budget test (G5: is the WISE flux this star's?) and for
light-curve cross-checks (ZTF, ASAS-SN, NEOWISE epochs, TESS via the TIC); the TESS arm uses it for TIC ids
and blending. Live validation: `scripts/xmatch_validate.py`.

## Why not a nearest-neighbour cone

The downstream question is "is this survey measurement this star's light, and how sure are we?". An
anomaly search is unusually sensitive to the wrong answer: a faint Gaia star handed its bright neighbour's
WISE detection has an IR excess, and an IR excess around an unexplained dimmer is exactly what we would
rank highest. So every design choice below is aimed at a probability that means what it says, plus an
independent empirical check of it.

**Positions at the survey's epoch.** Gaia's five-parameter solution is propagated to each survey row's own
epoch (linear motion, exact for zero radial velocity) with the full 5×5 covariance carried into the
tangent plane. Parallax is applied with ERFA's Earth position when the row's epoch is known to the day
(2MASS, NEOWISE), or added as variance ϖ²/2 when only a mean epoch is known (AllWISE, CatWISE, unWISE,
eROSITA, ZTF). Two-parameter sources get
a proper-motion prior from the robust spread of solved sources nearby, so a 2-parameter star is not
treated as fixed. A 1 arcsec/yr star moves 26 arcsec between 2MASS and Gaia; cone matching loses it.

**Bayesian posterior per target.** For target i and candidates j within the search radius,

```
P_ij = π LR_ij / ((1 − π) + π Σ_k LR_ik),    LR_ij = f(d_ij | C_ij) / ρ_i
```

f is the offset density under the error model, C_ij = C_gaia + s² C_survey + e² I, with a Gaussian core plus
a wide tail component (fraction ε, width τ) for the things a Gaussian misses: blends that pull a centroid,
orbital motion, bad epochs. ρ_i is the local density of survey sources, counted in an annulus around the
target and shrunk toward the regional density (Gamma–Poisson), so it is not fooled by small-number counts.
π is the counterpart fraction per Gaia G bin. There is no photometric prior: a colour prior would push
IR-excess objects toward "no match", which is backwards for this project.

**Calibrated from the data.** s, e, ε, τ and π per G bin maximise the marginal likelihood of all targets in
the batch under the same mixture the posterior uses (`calibrate.fit`; staged so small batches fit only π,
and the error model needs n_eff ≥ 50, the tail ≥ 200). If the model holds, the posterior is calibrated by
construction. Two checks say whether it holds: χ² of matches weighted by P and core membership should be
χ²₂ (mean 2, 1% above 9.21), and the shifted-sky false matches below should agree with the model's own.

**Competition.** The per-target posterior lets every Gaia star claim the same survey source. Every Gaia
source near a survey source, targets and non-target neighbours read from Gaia itself, now competes for it,
and each survey source is the counterpart of at most one of them (`compete.py`). The marginals of that
matching are computed by belief propagation (exact on trees; mean-field was ~1.7× overconfident on
synthetic fields). Two claimants share a survey source instead of competing only when its position cannot
tell them apart: their offsets differ by less than 2 combined positional sigmas (and less than half the
FWHM). This matters more than it sounds. A 2MASS source 0.07 arcsec from a bright star and 0.8 arcsec
from a faint one is the bright star's, even though 2MASS (FWHM 2.5 arcsec) cannot resolve the pair; its
position, good to ~0.1 arcsec, says whose light dominates it. Sharing by FWHM alone made 13 of the
Kepler-field 2MASS false matches; sharing by positional sigma made 2.5. Because claimed survey sources
are no longer available by chance, the chance-alignment density is reduced to the unclaimed part,
estimated per HEALPix order-6 region from the rows the pseudo-targets find.

## Measuring false matches on the real sky

Every target gets `n_shifts` pseudo-targets displaced in random directions by 45–120 arcsec (300–900 for eROSITA),
carrying the target's own astrometry, errors and prior. They are matched by exactly the same code, against
the same rows, competing with the same real claimants.

- **Plain pseudo-targets** have no counterpart by construction. The fraction accepted at P ≥ T is
  `fmr_shift`: P(false accept | no counterpart), measured on the real sky with all its clustering,
  artefacts and crowding.
- **Planted pseudo-targets** get a synthetic counterpart drawn from the fitted error model next to the real
  neighbours at the shifted position. They measure recall and `confusion`: P(a chance neighbour beats a
  true counterpart).
- `false_shift = Σ_targets (1 − π) fmr_shift + π confusion` is the empirical expected number of false
  accepts; `false_model = Σ_accepted (1 − P)` is the model's own. Agreement is the calibration check.

Per source, `p_chance_0.9` is the probability that, if this star had no counterpart, a chance source would
still be accepted at P ≥ 0.9: 1 − exp(−ρ_free A), with A the area where a chance source would pass,
averaged over the positional errors of background rows (chance sources are mostly faint, with larger
errors than a real counterpart). Its mean agrees with `fmr_shift` on synthetic fields.

**`p_cal`.** Where the two estimates disagree (crowded fields in wide-beam surveys, below), the model's
chance background is too simple. `p_cal = P / (P + κ(1 − P))` divides the posterior odds by the single
factor κ ≥ 1 that makes the model's expected false count over accepted matches equal the shifted sky's.
κ reads as "chance sources near these targets are κ times commoner than the model assumed", which is the
assumption the shifted sky tests. It is applied only when the shifted sky's false accepts hit at least 5
distinct survey rows, and never raises P. **Use `p_cal` for decisions and `p_match`/`p_best` for ranking.**
`report["shift_calibration"]` gives κ, `false_cal`, `false_model` and `false_shift`.

Two free-form maps were tried first: isotonic regression, then a logistic map in log-odds, both fitted on
the shifted-sky mixture (plain pseudo-targets weighted 1 − π, planted ones π). Both over-corrected 2–6× in
fields where the model held: accepted matches sit at 1 − P ≈ 10⁻⁴ and below, where a few thousand
pseudo-targets cannot resolve the shape of a map, and the planted half of the mixture does not reproduce
the real distribution of P closely enough. One parameter fitted to the one quantity the shifted sky
measures well is the version that survived.

## Surveys

| name | source | epoch | notes |
|---|---|---|---|
| `2mass` | HATS `two_mass` | per row (JD) | parallax exact; FWHM 2.5″ |
| `allwise` | HATS `allwise` | 2010.55 ± 0.6 fixed | **HATS copy column-shifted from `use_src` on**, see data-sources.md; only columns before it are read |
| `catwise2020` | IRSA TAP `catwise_2020` | `ra_pm/dec_pm` at 2015.4 | WISE beam 6.1″; positions from CatWISE's own motion fit |
| `unwise` | IRSA TAP | 2010–2017 coadds | crowded-field forced photometry; `fracflux` is unWISE's own blending measure |
| `neowise` | HATS | per detection (MJD) | one match per frame (`FRAME_NUM`, `SCAN_ID` groups): an IR light curve per star |
| `tic` | HATS (MAST S3) | 2000 | TIC carries Gaia DR2 ids; `ids.py` checks them through `dr2_neighbourhood`. FWHM = TESS pixel 21″ for blending |
| `erosita_dr2` | HATS | 2020.5 ± 0.75 | X-ray positions ~1–5″; prior 0.01; western Galactic hemisphere only |
| `ztf_dr22` | HATS `ztf_lc` | 2020.5 ± 2.5 | one object per field/CCD/quadrant/filter; grouped by `objectid // 10⁷` |
| `vsx` | HATS (2026-06) | 2000 | known variables; prior 0.05 |
| `asassn_vars` | VizieR II/366 | 2001.5 ± 4 | mixed epochs; carries Gaia DR3 and TIC ids of its own |
| `simbad` | SIMBAD TAP `basic` | 2000 | literature identity |

`n_local` is the number of survey rows found around a target; 0 means outside the survey footprint
(eROSITA-DE's half sky, ZTF's δ > −31°), which is different from "no counterpart".

## Validation (2026-10-21)

Three fields, Gaia DR3 from HATS, every survey calibrated on the field itself
(`scripts/xmatch_validate.py --cache DIR`):

| field | centre (l, b) | radius | Gaia sources |
|---|---|---|---|
| highlat | (196°, +78°) | 0.30° | 820 |
| kepler | (80°, +6.6°), around KIC 8462852 | 0.12° | 3,622 |
| plane | (240°, −2°) | 0.05° | 773 |

Expected false matches among targets accepted at P ≥ 0.9. "Shifted sky" is the empirical estimate,
"model" is Σ(1 − P); κ is the odds correction behind `p_cal` (so Σ(1 − p_cal) equals the shifted-sky
column wherever κ > 1). `fmr_shift` is the rate at which a star with no counterpart is matched anyway;
mean `p_chance` is the per-source prediction of the same thing. Error model: survey-error scale, floor,
tail fraction @ tail width.

| field | survey | accepted | false (shifted sky) | false (model) | κ | fmr_shift | mean p_chance | error model |
|---|---|---|---|---|---|---|---|---|
| highlat | 2mass | 376 | 0.3 | 0.2 | 1.00 | 0.0006 | 0.0004 | 1.14, 95 mas, 0.03 @ 848 mas |
| highlat | allwise | 617 | 2.9 | 2.2 | 1.30 | 0.0152 | 0.0156 | 1.09, 74 mas, 0.20 @ 699 mas |
| highlat | catwise2020 | 704 | 4.9 | 2.2 | 2.32 | 0.0524 | 0.0511 | 1.68, 184 mas, 0.11 @ 870 mas |
| highlat | unwise | 626 | 4.0 | 2.4 | 1.74 | 0.0470 | 0.0579 | 1.49, 166 mas, 0.15 @ 687 mas |
| highlat | neowise (per frame) | 13,053 | 173 | 99 | 1.80 | 0.0140 | 0.0166 | 1.29, 175 mas, 0.08 @ 1208 mas |
| highlat | ztf_dr22 | 4,322 | 4.3 | 2.1 | 2.13 | 0.0088 | 0.0070 | 1.00, 115 mas, 0.05 @ 416 mas |
| highlat | tic | 740 | 0.0 | 0.3 | 1.00 | 0.0000 | 0.0003 | 0.88, 1 mas, 0.06 @ 219 mas |
| kepler | 2mass | 1,152 | 2.2 | 0.6 | 3.97 | 0.0008 | 0.0003 | 0.97, 1 mas, 0.08 @ 393 mas |
| kepler | allwise | 768 | 5.2 | 3.0 | 1.75 | 0.0018 | 0.0010 | 2.02, 1 mas, 0.36 @ 657 mas |
| kepler | catwise2020 | 2,173 | 75 | 32 | 2.56 | 0.0740 | 0.0578 | 3.55, 44 mas, 0.44 @ 963 mas |
| kepler | unwise | 2,221 | 90 | 32 | 3.08 | 0.0979 | 0.0940 | 5.97, 79 mas, 0.37 @ 1003 mas |
| kepler | neowise (per frame) | 32,005 | 466 | 227 | 2.16 | 0.0027 | 0.0017 | 1.54, 86 mas, 0.24 @ 981 mas |
| kepler | ztf_dr22 | 11,974 | 11.1 | 12.8 | 1.00 | 0.0047 | 0.0047 | 1.00, 62 mas, 0.20 @ 235 mas |
| kepler | tic | 3,479 | 0.1 | 0.1 | 1.00 | 0.0006 | 0.0008 | 0.93, 1 mas, 0.05 @ 200 mas |
| plane | 2mass | 234 | 0.0 | 0.3 | 1.00 | 0.0000 | 0.0016 | 0.94, 1 mas, 0.11 @ 1043 mas |
| plane | allwise | 137 | 1.2 | 0.4 | 1.00 | 0.0019 | 0.0004 | 3.63, 1 mas, no tail |
| plane | catwise2020 | 374 | 13.1 | 5.3 | 2.65 | 0.0414 | 0.0305 | 3.21, 2 mas, 0.51 @ 998 mas |
| plane | unwise | 397 | 15.6 | 5.5 | 3.16 | 0.0608 | 0.0422 | 3.09, 76 mas, 0.53 @ 1071 mas |
| plane | neowise (per frame) | 3,621 | 93 | 33 | 3.11 | 0.0039 | 0.0026 | 1.42, 90 mas, 0.36 @ 1055 mas |
| plane | ztf_dr22 | 72 | 0.0 | 0.3 | 1.00 | 0.0000 | 0.0002 | 1.00, 67 mas, no tail |
| plane | tic | 749 | 0.0 | 0.1 | 1.00 | 0.0000 | 0.0008 | 0.91, 1 mas, 0.04 @ 200 mas |

eROSITA DR2 (6 matches, highlat only; kepler and plane fields are outside eROSITA-DE) and VSX (3, 27, 8)
are too sparse for these numbers to mean much; their priors are fixed (0.01, 0.05).

What the table says:

- **2MASS, TIC, ZTF and AllWISE outside crowded fields are well modelled**: χ² checks within 0.1 of 2
  (TIC 2.3 in Kepler, from TIC's own DR2-to-2000 propagation), false-match rates of 0.1% or lower.
- **The WISE family in crowded fields is where position-only matching is hard.** CatWISE and unWISE
  need survey errors scaled 3–6× and a tail holding 40–50% of matches at ~1 arcsec: at b = 2–7° the 6.1″
  beam blends several Gaia stars, so a "counterpart" position is a blend centroid. 4–10% of stars with no
  WISE counterpart of their own are still matched to something at P ≥ 0.9. The model underestimates that
  by 2.3–3.2×; κ corrects it. For G5 this means a WISE flux at low Galactic latitude needs `p_cal`,
  `p_chance_0.9` and beam contamination together, and a candidate with any of them marginal is a blend
  until shown otherwise.
- **NEOWISE single-epoch matches** are 2–3× more often false than the model thinks, probably from
  single-frame artefacts and moving objects near bright stars (not yet checked). Per-epoch IR light curves
  should use `p_cal` and expect a few percent of contaminating epochs.

**Against DPAC's best-neighbour tables** (our best match with P ≥ 0.5 vs `gaiadr3.tmass_psc_xsc_best_neighbour`
and `gaiadr3.allwise_best_neighbour`), there are no disagreements in any field: wherever both name a
counterpart it is the same one (2MASS 377/377, 1,153/1,153, 240/240; AllWISE 570/570, 560/560, 104/104).

- AllWISE matches DPAC does not have: 58, 247 and 38. 216 of the 247 in the Kepler field have `p_cal` ≥ 0.9,
  multiplicity ≈ 1 and a median separation of 0.75″. They look real but have not been checked
  one by one.
- Matches DPAC has and we do not: 2MASS 6, 31, 10; AllWISE 4, 26, 7. Most (68 of 84) get P < 0.3 here
  because a resolved Gaia neighbour is the better positional match. DPAC lets several Gaia sources share
  one 2MASS source; we give it to the star its position points at, which is the safer choice for IR
  photometry.

The full report (per G bin priors, per-threshold accounting) is `xmatch-validation-2026-10-21.json` next to
this file.

## Output columns

`best` (one row per target, per group for grouped surveys): `p_best`, `p_cal`, `p_none`,
`n_candidates`, `n_local`, `<survey>_id`, `sep_arcsec`, `survey_multiplicity` (Σ P over targets claiming
the same survey row; > 1 means a blend of our targets), `p_chance_0.9`. `matches` adds `chi2`, `log_lr`,
`rank`, the row's epoch and the survey's `keep` columns (photometry, flags).

`context()` / `--context` joins every survey into one wide table with `<survey>_p`, `<survey>_p_cal`,
`<survey>_p_chance`, `<survey>_multiplicity` and coverage, plus beam contamination from Gaia neighbours
(`blend.py`) for the W1/W2 (6.1″), W4 (12″) and TESS (21″) beams: number of Gaia sources in the beam and
the overlap-weighted RP flux fraction they contribute. For G5, an IR excess needs a confident match
(`p_cal`), low `p_chance_0.9`, multiplicity ≈ 1 and small beam contamination; each failure mode has its own
column so the vetting step can say which one fired.

## Access

HATS reads go through lsdb (fsspec/aiohttp), so they bypass `cosmos_search.net`. They still refuse to run
under `COSMOS_BLIND`, send the project User-Agent, retry transient errors with the shared client's
full-jitter backoff, and keep to 4 partition reads at a time. Cones are grouped by HEALPix pixel (0.5° by
default) so a target list costs one fetch per pixel. TAP surveys use the shared `net.Tap`.

## Limits and next steps

- Fields of 0.05–0.3° calibrate in seconds from cached cones; an all-sky run would calibrate per HEALPix
  region (the code takes any target list; region-wise batching is not written yet).
- κ is one number per run. If a batch mixes crowded and sparse sky, κ is an average; batch by region.
- Kepler-field 2MASS κ = 4.0 rests on 6 false accepts (2.2 ± 0.9 expected false); small-number κ values
  are noisy, and only apply above 5 distinct rows.
- The planted half of the shifted sky follows the fitted error model, so `confusion` and `recall` are only
  as good as that model. The plain half is model-free.
- ZTF groups use `objectid // 10⁷` (field, readout channel, filter), from the documented id layout; not
  yet checked against the HATS columns.
- AllWISE through HATS gives positions and profile-fit magnitudes only (the copy is column-shifted from
  `use_src` on, data-sources.md). Fluxes and other columns need IRSA TAP.
- ASAS-SN variables mix epochs (mean epoch 2001.5 ± 4 assumed); its own Gaia DR3 ids are a cross-check.
- Not done: Pan-STARRS/SkyMapper (for the optical SED), DECaLS, Spitzer, GALEX. The survey spec is a
  dataclass in `surveys.py`; adding one is a few lines plus a validation field.
- Upstream: the AllWISE HATS column shift is reported as [astronomy-commons/lsdb#1660](https://github.com/astronomy-commons/lsdb/issues/1660). When it is fixed, recheck and drop the column restriction in `surveys.py`.
