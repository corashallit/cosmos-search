# Slow changes

Main search since 2026-10-23. Code: `cosmos_search/slow/`. Owner: "Strategy review before DR4" thread (until a
dedicated thread takes it).

Question: does any star fade or brighten steadily over Gaia's 5.5 years, or dim for weeks to years, in a way that
neither the instrument nor known astrophysics explains?

## Why this is the main search

Gaia visits a typical star ~90 times over DR4's span (2014-07-25 to 2020-01-20), in clumps weeks apart. For a
1-day dip that sampling is a poor detector: it catches one on ~2% of stars, mostly on a single visit, where an
eclipse explains it as well as anything else (docs/arms/gaia-dimmers.md, docs/dayone.md). For changes longer than
~100 days it catches 90-100%, on several visits, and over the whole span it measures a star's trend to a few
hundredths of a percent per year (below). No other data set has years of millimagnitude photometry on one
calibration for over a billion stars, and DR4 is the first release with light curves for quiet stars: Boyajian's
star, for example, was not flagged variable in DR3, which therefore published no light curve for it.

Two outcomes are worth having. A find is cheap to verify, because one new brightness measurement shows whether
the fade continued. With no find, the search sets a limit on how common slowly dimming stars are; on Sun-like stars
at G < 15 that reaches ~0.1% per year, the rate a swarm of collectors built over ~10³ years would produce.

## Signals

| Signal | What it looks like in Gaia | Natural explanations to beat |
|---|---|---|
| Steady fade or brightening | Trend over 66 months after the shared calibration drift is removed; the same in G, BP and RP, with a colour that tells grey blocking from dust from a temperature change | Magnetic activity cycles (Sun ~0.1%, active stars several %), giants and long-period variables, young stars, high-proper-motion stars drifting past a neighbour |
| Long dimming episode (weeks to years) | Several consecutive visits below the star's level, smooth in time, in both fields of view and all bands | Disc eclipses in long-period binaries (ε Aur type), R CrB stars, UX Ori and dipper young stars, Be stars |
| Infrared response in the same years | NEOWISE W1/W2 single exposures (2014-2024) rise with a dust fade; a large cold opaque body or a change in the star itself gives none | — |

Instrumental explanations to beat: calibration drift shared by stars of similar magnitude and colour (`drift`),
steps where a fading star crosses a readout-mode boundary (G ≈ 11 and 13), crowding that changes with scan angle
(DR4's `phot_g_harmonic_amplitude` measures it), and DPAC-rejected transits.

## DR4 data products that matter here

From the draft data model of 2026-06-26 (gaia-dr4-prerelease-draft-data-model_2026-06-26.zip on the ESA ftp):

- `all_source_photometry` [TAP, nearly every source]: per-band Abbe values (`phot_g_abbe_value` in flux space,
  `phot_bp_abbe_value`, `phot_rp_abbe_value`), constancy probabilities (`phot_*_const_prob`), Stetson indices, flux
  percentiles, and `phot_g_harmonic_amplitude` / `_phase` (photometric residual against scan angle, large for
  doubles). The Abbe value is near 1 for noise and near 0 for a smooth change, so a whole-sky selection of slow
  changes, consistent across bands, runs on TAP on release day.
- `vari_summary` [TAP, ~4 × 10⁸ rows]: `abbe_mag_g_fov` with left- and right-tailed p-values.
- `epoch_photometry` [DataLink only]: light curves; `all_source_flags.has_epoch_photometry` says which sources have them.
- `xp_epoch_spectrum` [DataLink only, ~1.2 × 10¹¹ rows]: per-visit BP/RP spectra, for the wavelength dependence of
  a change on top candidates.

On DR3, `vari_summary.abbe_mag_g_fov < 0.2` keeps 78% of stars with a strong trend (|slope| > 10σ, 1,518 sampled)
and 4% of random variables (2,981 sampled). In DR4 most stars are quiet, so far fewer than 4% should pass.

## Milestones

| # | Milestone | Depends on | Status |
|---|---|---|---|
| S1 | Known answers matched to Gaia, DR4 columns checked | — | Done 2026-10-23 (below) |
| S2 | Drift correction from neighbours (`drift`), tested on stars it was not fitted on | — | Magnitude/colour drift done 2026-10-23; sky term measured 2026-10-23 (below): small, kept as a check; shared-date test in slow-v3 |
| S3 | Trend, curvature and long-episode detector; fake slow fades and long dips for recall; brightenings as the false-alarm estimate | S2 | First version done 2026-10-23 (below) |
| S4 | Whole-sky selection on TAP (Abbe values; fallback DR4 minus DR3 mean against neighbours), recall measured on DR3 | S1, S3 | Done 2026-10-23 (below); download routes in docs/infra/dr4-downloads.md |
| S5 | Checks per candidate: NEOWISE same years, colour of the change, position (`astrometry.dips`), ZTF and ASAS-SN after 2020, known-variable catalogues | S3 | First version done 2026-10-23 (below); position needs DR4 epoch astrometry |
| S6 | Pre-registered blind test of the detector: new fake shapes, fresh stars; known variable classes measured openly | S4, S5 | Run 2026-10-23 (below): fails on repeated slow cycles; Cora chose a slow period search, whose own blind test is written down (below) |
| S4b | Score every star from the bulk files; trends against neighbours; own bad-epoch test; reading order within kind of star; DataLink fallback list | S3, S7's trial | Done 2026-11-10 ("Every star scored" below) |
| S7 | Freeze before 2026-12-02, run on DR4 (short-dip and astrometry searches run as built alongside) | S6, DR4 | slow-v3 (fades read in time order, below); release-day thresholds from DR4's own stars ready (below); whole run tried on all of DR3 2026-10-30, then the band, eclipsing-binary and NEOWISE checks settled (below) |
| S8 | Verify top candidates: one new brightness measurement each; per-visit spectra for the best | S7 | |

## Known answers (S1)

`python -m cosmos_search.slow known --out known.parquet` (`slow/known.py`). On 2026-10-23: 985 stars, 967 matched
to a Gaia DR3 source within 2″ (median separation 0.07″), 573 of them with DR3 light curves. With `--mirror aip`
the positions are matched on the AIP copy of `gaiadr3.gaia_source`: on 2026-11-10 ESA's archive refused
synchronous queries (HTTP 503) for hours, and the mirror gave the same counts (967 matched, 573 with DR3 light
curves) in 19 minutes.

| Kind | Stars | Matched | DR3 light curve |
|---|---|---|---|
| Boyajian's star (Gaia DR3 2081900940499099136) | 1 | 1 | 0 |
| Gaia Science Alerts that faded, raised within DR4's span | 376 | 362 | 218 |
| VSX R CrB | 238 | 235 | 184 |
| VSX UX Ori | 331 | 330 | 135 |
| VSX DY Per | 39 | 39 | 36 |

Alerts need a change of about a magnitude, so these sit at the large-amplitude end; changes of 0.1-10% have no
ready-made list.

## Drift correction (S2)

`slow/drift.py`, checked with `scripts/slow_drift_check.py` on 2026-10-23. Input: a random 10% of the DR3 bulk files
(`dr3_pass --shard 0/10`, every light curve kept), 1,004,731 sources. Reference stars (scatter < 1.5× errors, no
epoch beyond 8 robust σ, ≥ 20 G epochs): 14,390, split by source_id hash into half A (7,136, drift fitted) and half
B (7,254, judged). Cells: 15-day time bins × G bins × three colour bins, ≥ 30 stars, with a pooled-colour fallback;
3,487 cells measured.

Half B, G band, trend in % per year (median over stars = the shared drift; spread = 1.4826 × MAD between stars):

| G | Stars | Median before | Median after | Spread after | Median error | Spread / error |
|---|---|---|---|---|---|---|
| 14-15 | 150 | -0.003 | +0.007 | 0.058 | 0.038 | 1.58 |
| 15-16 | 789 | -0.007 | +0.005 | 0.069 | 0.049 | 1.38 |
| 16-17 | 1,941 | -0.009 | -0.005 | 0.091 | 0.074 | 1.23 |
| 17-18 | 1,963 | +0.020 | +0.010 | 0.166 | 0.119 | 1.40 |
| 18-19 | 1,160 | +0.112 | -0.005 | 0.397 | 0.265 | 1.50 |
| 19-20 | 562 | +0.213 | +0.065 | 1.009 | 0.677 | 1.53 |

BP and RP behave the same way, with spread / error 1.0-1.2. Full table: `half_b_trends.csv` from the script.

What this shows:

- **The magnitude/colour drift comes out.** Where it is large (G > 18) the median trend falls from +0.11 to
  -0.005% per year (G 18-19) and from +0.21 to +0.07 (G 19-20), using a correction fitted on other stars.
- **Fades and brightenings stay symmetric after correction**: beyond 0.3% per year at > 5σ, 26 fades and 22
  brightenings in half B. Brightenings can serve as the false-alarm estimate for fades.
- **Spread between stars in G is 1.2-1.6× the errors**, against ~1.0-1.2 in BP and RP. G is the most precise band,
  so low-level variability of these DR3 variables shows there first; DR4's quiet stars will tell how much is the
  instrument.
- **A shared sky term remains.** Medians of trend significance in level-5 HEALPix pixels (~3.4 deg²) vary 1.87× as
  much as shuffling stars gives (p = 0.005, 180 pixels), and the pixel medians of half A and half B agree
  (r = 0.56, 147 pixels). Subtracting half A's pixel medians from half B only brings the ratio to 1.79, because
  ~25 reference stars per pixel give noisy medians. DR3's sample is variables only and its selection itself varies
  on the sky (docs/population.md §14), so this cannot yet separate instrument from population. Open: fit the sky
  term on the full DR3 stream (10× the reference stars), and on DR4's quiet stars.
- **A linear trend is the wrong statistic for episodic faders.** Of the 22 known answers in this sample, few show a
  significant DR3 trend: R CrB and UX Ori fades come in episodes, and at least 320 of the 376 fading alerts in
  DR4's span were raised after DR3's data end (2017-05-28), 229 of them in 2019. The detector (S3) needs a long-episode statistic beside the trend.

## Detector (S3)

`slow/detect.py`, checked with `scripts/slow_detect_check.py` on 2026-10-23 on the same 10% of DR3 (1,000,069
stars with at least 8 visits), after the drift correction fitted on half A.

Epochs closer than 2 days are averaged into one visit (the two fields of view and consecutive scans). Each star's
noise is the scatter of one visit measured from differences between consecutive visits, so it includes any
variability faster than the weeks between visits and every statistic below asks whether the slow part stands out
above the star's own fast scatter. Per star, in G:

- `slope_z`: the linear trend over the visits, in units of its error, taken from the larger of the visit noise
  and the scatter about the line. (With the visit noise alone, a median, outbursting and large-amplitude pulsating
  stars passed as steep fades: a few big jumps barely move a median. Fixed 2026-10-23 after the check sheet below
  showed them.)
- `s_dim`: the strongest dimming episode, over every run of at least 3 consecutive visits spanning at least 20
  days and at most 60% of the visits: the run's mean deficit below the star's median, times the square root of
  the number of visits, over the noise. The run's deepest visit is left out, so one deep visit (a short dip)
  never makes an episode. `s_bright` is the same for brightenings and is the false-alarm estimate.
- `abbe`: about 1 for noise and near 0 for a smooth change; DR4 publishes the same statistic for nearly every star.
- `depth_bp`, `depth_rp`: the BP and RP deficits over the G episode's dates, for the colour of the change.

A star passes if `s_dim` or `-slope_z` exceeds a threshold set so that 0.1% of quiet stars pass by brightening.

**False alarms.** In quiet stars of half B (never used for the drift) dimming and brightening are equally common
(beyond 8: 22 dimming episodes against 24 brightening; beyond 5, 17 fading trends against 13). Faint stars need
their own thresholds: at G ≥ 19 the quiet stars' tails reach more than twice as far in both directions, and the
stars there are mostly blue and slowly wandering, as quasars are. Thresholds at 0.1%:

| G | Quiet stars | `s_dim` | `-slope_z` |
|---|---|---|---|
| < 19 | 6,023 | 7.4 | 3.9 |
| ≥ 19 | 1,228 | 17.1 | 7.9 |

DR3 only has light curves for stars it classed as variable, so even these quiet stars are not typical; DR4's
thresholds have to come from DR4's own brightenings, at the far smaller false-alarm rates a billion stars need.
Among all DR3 light curves at G < 19, 33,833 pass against 30,895 brightening: a 10% excess of dimming among
DR3 variables, which this sample cannot attribute.

**Recall.** One fake of each family per quiet half-B star (21,762 fakes), injected into the raw light curve
before normalising and drift correction. Signal-to-noise below is the deepest G deficit times the square root of
the visits below half of it, over the host's noise. Share found at G < 19:

| Signal-to-noise | Steady fade | Fade that starts partway | Long dip (20-1,500 days) |
|---|---|---|---|
| < 5 | 16% | 1% | 0.2% |
| 5-10 | 92% | 21% | 2% |
| 10-20 | 99% | 61% | 27% |
| 20-40 | 100% | 81% | 73% |
| > 40 | 100% | 86% | 92% |

The steady fade is found close to the ideal (`slope_z` does nearly all of it). Long dips need about twice the
signal-to-noise: the search over every possible run costs some, leaving out the deepest visit costs some more,
and the measure above uses peak depth while rounded and tailed shapes spend most of their time shallower. On
DR3's 2.8 years a G 15-16 quiet star (noise per visit ~0.2%) has a 1% dip lasting a year at signal-to-noise ~15;
DR4 doubles the span and the visits. At G ≥ 19 almost nothing is found at the 0.1% thresholds.

**Known answers.** DR3 ends on 2017-05-28, before most fading alerts, so the 22 known answers with DR3 light
curves mostly show nothing (5 pass). The alerts' own Gaia light curves, one point per transit over DR4's span
(`python -m cosmos_search.slow alert-lcs`, 358 with enough visits), are a better check: 216 pass. Of the 241
whose light curves are smooth (Abbe < 0.5) 194 pass (80%); of the 117 that change faster than Gaia's visits
(one-day dips, erratic young stars) 22 pass, which is the intended behaviour, since the short-dip search covers
those. R CrB stars whose median state is the faint one show their recoveries as brightenings.

Open: the sky term from S2; thresholds per colour; families held out of all calibration for the blind test (S6).

## Whole-sky selection (S4)

**Superseded on 2026-11-10 for any run with bulk files** ("Every star scored" below): measured on injected steady fades,
this cut picks none up to 3% a year and 3% of the brightest at 5% a year. The table below counted only fakes the detector found among stars passing a
quiet-star cut that 14.5% of all stars pass. Kept as written, for the record and for `slow_search.py select`.

DR4's light curves come through DataLink only, so the detector runs on stars picked first from the per-star
columns of `all_source_photometry` (`slow/select.py`). For noise of variance σ² plus a change slower than the gap
between measurements, of variance s², consecutive differences hold only the noise, so the Abbé value is about
σ² / (σ² + s²) and the unit-weight variance about (σ² + s²) / err². Their combination

    slow_z = √n × uwv × (1 − Abbé)

is the variance of the slow part over the squared errors, scaled by its noise: near 0 for constant stars whatever
the error scale, large for slow changes, small for stars that vary faster than Gaia's visits (large uwv, Abbé
near 1). Cuts are set per G bin so a fixed share of stars passes.

On release day: draw a random sample with `select.sample_adql` (`random_index_as`), set the cuts per G bin from
it with `select.thresholds`, then run `select.adql(cuts)` over the sky in source_id slices. The Gaia archive's
ADQL has no CASE expression (checked on DR3, 2026-10-23), so each G bin is its own OR term; the query shape runs on
the DR3 archive.

Checked with `scripts/slow_select_check.py` on the same DR3 run: slow_z computed from each star's epochs (G,
uncorrected fluxes, as DR4 computes it), cuts set on quiet half-B stars, and the question asked of every fake the
detector finds at its 0.1% thresholds: did the selection keep it? At G < 19:

| Share of quiet stars selected | Long dips kept | Late-starting fades kept | Steady fades kept |
|---|---|---|---|
| 0.1% | 95% | 77% | 69% |
| 1% | 98% | 88% | 82% |
| 10% | 99% | 98% | 95% |

So a selection of 0.1-1% of stars keeps most of what the detector can find (the weakest steady fades, which the
trend statistic reaches and the variance-based score does not, are what it loses), and the light curves to fetch fall
from the whole catalogue to its 0.1-1%. The rate at which DataLink serves DR4 light curves sets which share
is affordable. DR3's DataLink served 1.5 light curves per second at best and 0.27 on a busy afternoon, with no
gain from parallel calls (docs/infra/gaia-epoch.md): 0.2-1.3 × 10⁵ a day. At that rate 0.1% of DR4's ~2 × 10⁹
stars takes two weeks to three months, and 1% ten times longer. Unless DR4 also offers bulk files, plan for a
share near 0.1%, highest scores first. DR3's quiet stars are not white-noise stars (median uwv 1.9-4.9 and
Abbé 0.88-0.94 by G bin: underestimated errors and correlated calibration), which is why the cuts come from the
stars themselves. At G ≥ 19 too few fakes are found to say anything. Not yet used: the BP and RP Abbé values
(a dust or grey change shows in all three bands), and `phot_g_harmonic_amplitude` to drop doubles.

## Checks per candidate (S5)

`slow/checks.py`, run on DR3 with `scripts/slow_checks.py`. Each check asks a question that the explanations
answer differently:

| Check | Grey occulter | Dust in front | Dust shell re-emitting | Change in the star |
|---|---|---|---|---|
| BP depth / RP depth (`colour`) | 1 | ~1.7 | ~1.7 | > 1 for cooling |
| NEOWISE W1/W2 change over the episode, or W1/W2 slope against Gaia's (`wise_change`, `wise_slope`) | same as G | ~0.05-0.07 of G | brighter (+5% in W2 per 1% of G at 300 K; far more if hotter) | well below G (Rayleigh-Jeans side) |
| ZTF after 2020-01-20 (`after`) | — | — | — | — |
| Other surveys over Gaia's dates and since, TESS variability ("Other surveys" below) | same change | same change, redder | same change, redder | same change; spotted stars vary within a month |

ZTF says whether a change went on, stopped or recovered after DR4's span. NEOWISE comes from IRSA's
`neowiser_p1bs_psd` single exposures (clean frames, one component, no contamination flags), averaged per ~6-month
visit. Position during the episode (`astrometry.dips`) needs DR4's epoch astrometry and is not in yet. Two more
checks came later: whether G's change agrees with BP's and RP's (`band_check`, "G against BP and RP" below) and
whether an eclipsing binary explains the dips (`eclipse`, "Eclipsing binaries among the candidates" below).

**NEOWISE errors (2026-10-30).** Both NEOWISE errors come from the scatter between visits, and DR3's 2.8 years give
3-6 visits outside a change: too few to measure a scatter. On the 280 calibration stars (vetting.md §14), with no
change injected, windows of 1-3 visits sat beyond 2 errors 9-12% of the time and beyond 3 errors 2-3% (a Gaussian
gives 4.6% and 0.3%); W1 trends over 4-8 visits 14-22% and 6-11%. Now the sample scatter is widened by Student's t
for the visits it rests on, held at or above 1.36 times the scatter within visits (the median ratio over those
stars, `WISE_FLOOR`), and the change's error counts the baseline's own error. The same windows then sit beyond 2
errors 2.8% (quiet stars) and 5.2% (variable stars) of the time, and beyond 3 errors 0.5% and 1.1%; trends 1.3% and
6.1%, 0.03% and 2.5%. Variable stars keep a heavier tail from their own slow variability.

Run on 2026-10-23 on the 40 strongest DR3 candidates at G < 17 (30 dimming episodes with smooth light curves,
10 fading trends). Every one carries a DPAC variability class: eclipsing binaries, RS CVn and other spotted stars,
quasars, Be stars, long-period variables. DR3's light curves are of stars already classed as variable, so this
was expected; the point was to see the checks work:

- The first version of the sheet showed outbursting and large-amplitude pulsating stars as the steepest "fades",
  which led to the trend-error fix above (S3).
- The steady fades that remain are smooth (Abbé 0.02-0.04). For two of the three spotted stars with NEOWISE
  coverage W1 fades at 0.4 and 0.6 of Gaia's rate (the third is too noisy to say): more than dust in front allows (0.07), less than grey blocking (1). That is what a
  change in the star's own temperature or spot coverage gives.
- NEOWISE covered 14 of the 30 episodes; the rest were shorter than its 6-month spacing or fell between visits.
  8 of the 40 stars had no clean NEOWISE exposures (crowded fields). ZTF had enough epochs either side of 2020 for 17.
- 16 of the 30 episodes are grey within 2σ and 5 dust-like. The grey ones are mostly eclipses of long-period
  binaries, which is why the class and the periodicity search (`ebfilter`) come first in the order of checks.

Output: `checks.csv` (one row per star) and the NEOWISE and ZTF epochs.

Calibrated on 2026-10-30 together with the blind review (docs/vetting.md §14, 432 DR3 objects): the colour and
NEOWISE errors hold on fakes (96% and 92% within 2σ), but at DR3's depths the checks barely separate natural stars
(13 of 253 have a colour neither grey nor dust-like). The review shortlists 44/50 anomalous fakes and 5.0% of
flagged natural stars at DR3's mix (2.8-7.5%), so DR4's review comes after automated cuts. It also reads half of
the G-only fakes as unexplained; a check of G's change against BP's and RP's is missing.

## Blind test (S6)

Written down and pushed before the batch is drawn. Nothing below changes after the draw.

**What is tested.** The detector of S3 with its settings frozen as `slow-v1` (`cosmos_search/slow/rule.py`): a
star is flagged when its dimming-episode score or its fading trend passes the threshold for its magnitude range
(G < 19: s_dim > 7.41 or slope_z < −3.94; G ≥ 19: s_dim > 17.1 or slope_z < −7.91), after correcting its epochs
with the drift table. Thresholds and drift table come from shard 0/10 of DR3 and are frozen in
`prereg/rules/slow-v1/` (sha256 of each file goes into the batch plan).

**Stars.** Shard 1/10 of DR3's epoch photometry (1,042,022 sources, none shared with shard 0). Quiet stars by the
same definition as in S2 (`drift.reference_mask`): 16,711, of which 14,060 at G < 19.

**Batch** (`scripts/slow_blind.py`, seed from the operating system, answers sealed with age to Cora's key and
the reveal key):

| Group | Stars | What it measures |
|---|---|---|
| Quiet stars as they are, G < 19 | 5,000 | false alarms |
| Quiet stars as they are, G ≥ 19 | 2,000 | false alarms |
| New shapes, 600 each, G < 19 | 2,400 | recall on shapes the detector was not designed or calibrated on |
| Calibrated shapes (long dip, late-starting fade, steady fade), 300 each, G < 19 | 900 | whether S3's recall holds on fresh stars |
| Late-starting brightenings, G < 19 | 300 | reported only: how often a brightening is flagged as a fade |

Each fake goes into its own quiet star, which never also appears as a natural star. Ids are anonymous and the
order is shuffled.

The four new shapes (`inject.lightcurve.HELD_OUT_SLOW`) were defined for this test and have not been run through
the detector. `sample()` refuses them unless a confirmatory batch asks, and a test checks that no calibration
code lists them:

- **Staircase fade:** 2-5 abrupt drops of 0.2-3% each that stay down.
- **Slow cycle:** repeated fades and full recoveries, period 150-1,000 days, depth 0.2-10%.
- **Accelerating fade:** a fade that starts slowly and speeds up (depth × x^p, p = 2-4), 0.2-10%.
- **Fade with dips:** a gradual fade of 0.2-5% with a few to ~20 short dips (1-20 days, 0.5-10%) after it starts.

Half of each are grey, half reddened (as for the calibrated shapes).

**Signal-to-noise** of a fake, as in S3: its deepest G deficit at the visits × √(visits below half that depth) ÷
the host star's own visit-to-visit noise.

**Pass criteria** (all three must hold):

| | Criterion | Expected from shard 0 |
|---|---|---|
| P1 | Natural stars flagged: at most 0.5% in each magnitude range | 0.23% (G < 19), 0.33% (G ≥ 19) |
| P2 | New shapes at G < 19 and signal-to-noise ≥ 20: pooled recall at least 70% with 95% lower bound at least 60%, and each shape at least 50% | no expectation (unseen shapes); the calibrated shapes give 83-100% |
| P3 | Calibrated shapes at signal-to-noise ≥ 20: each within 10 points of S3 | long dip 83%, late-starting fade 84%, steady fade 100% |

**What each outcome leads to.**

- All pass: the detector is frozen for DR4 as it stands (S7), and the recall quoted for DR4 is this batch's.
- P1 fails: the thresholds do not carry over between sets of stars; set them from both shards before DR4 (on DR4
  they come from DR4's own stars in any case) and say so.
- P2 fails for a shape: the detector misses that kind of change. Add a score aimed at it if one is simple; the
  shape is then spent, and its recall after the change is stated as not blind-tested.
- P3 fails: S3's recall figures were optimistic; find out why before freezing.

**Result** (batch `slow-blind-1`, flags archived in `prereg/verdicts/slow-blind-1/` and pushed before the
reveal; full numbers in the batch's `score.json`). The test fails as a whole, on P2:

| | Result | Pass? |
|---|---|---|
| P1 | 8 of 5,000 natural stars flagged at G < 19 (0.16%, 95% upper bound 0.29%); 1 of 2,000 at G ≥ 19 (0.05%) | yes |
| P2 | New shapes pooled: 558 of 838 (67%, lower bound 64%). Staircase fade 96% (324/339), accelerating fade 92% (116/126), fade with dips 55% (80/145), slow cycle 17% (38/228) | **no** (pooled under 70%; slow cycle under 50%) |
| P3 | Long dip 85% (47/55), late-starting fade 86% (73/85), steady fade 100% (50/50) | yes |

Reported only: 49 of 300 late-starting brightenings (16%) were flagged as fades, all by the episode score. The
stretch before a brightening looks like a dimming that never recovers inside the span. The checks per candidate
(ZTF after 2020, and whether an episode touches the start of the light curve) have to sort these out.

Why the misses (looked at after the reveal):

- **Slow cycle.** Recall falls with the number of cycles inside the span: 57% (30/53) for up to 1.5 cycles, 20%
  for 1.5-2, 4% for 2-3 and 1% above 3. The detector looks for one episode against the star's median. With
  several cycles the median sits partway down, and each trough holds only a few of DR3's ~24 visits. The
  signal-to-noise measure adds up every trough, which suits a search that stacks cycles and overstates what a
  one-episode search can see. These light curves are smooth (median Abbé 0.40 against 0.95 for quiet stars), so
  the selection score of S4 keeps them; the detector is what drops them.
- **Fade with dips.** When the fade is deeper than the deepest dip, 11 of 12 are found. When a short dip is the
  deepest point, 52%: those dips mostly fall in a single visit, which the slow search leaves out on purpose (the
  short-dip search covers them), so here too the signal-to-noise measure overstates.

What follows, as written above: the detector is not frozen as it stands, and the slow-cycle shape is now spent
(any recall quoted for it after a change is not blind-tested). A first idea, a score for smooth light curves that
lean to the faint side, does not work: this shape is a sinusoid, which has no lean, and in relative flux it looks
the same as a pulsating or spotted star. Scoring smoothness alone (Abbé below its 0.1% value for quiet stars,
0.36) was tried afterwards on the calibration stars (shard 0, 1,852 fakes): recall at signal-to-noise ≥ 20 goes
from 18% to 54%. It still misses cycles shorter than a third of the span (7% for more than 3 cycles), which need a
period search, and it flags 5.5% of DR3's G < 19 light curves, so on DR4 it would pass most of the whole-sky
selection. Cora chose to build a slow period search (2026-10-23, next section).

**Known variable classes** (active and spotted stars, giants and long-period variables, young stars, eclipsing
binaries) are measured openly on shard 1, after the reveal, from DPAC's classes. They need no blinding: the
question is how many such stars pass, which sets the workload for the checks per candidate (S5).

**Procedure** (as for the astrometry blind test): `make` draws the batch and seals the answers; `run` writes
`flags.csv`; `commit` archives the flags in `prereg/verdicts/slow-blind-1/` and refuses a dirty tree; the archive
commit is pushed; `reveal` refuses unless that commit is on origin and the hashes match.

## Slow period search

Cora's choice after the blind test above (2026-10-23): look for dimming that repeats. `cosmos_search/slow/period.py`,
calibrated with `scripts/slow_period_check.py` on shard 0/10.

**Method.** The detector's episode score, computed on the light curve folded at trial periods from 20 days to
two-thirds of the span (so at least 1.5 cycles are seen): the best window of visits that sit together in phase,
3 visits up to 60% of them, its deepest visit left out, against the star's own visit-to-visit scatter, after
removing a straight-line trend. Its mirror, repeated brightening, is the false-alarm estimate. A window counts as
repeating only if (a) its visits come from at least two cycles, (b) they form at least two groups in time, (c) the
visits between those groups are back at normal brightness (not more than 3σ faint) and (d) the window still shows
a 3σ deficit with its strongest group removed. Conditions (c) and (d) were added after a single long dip was found
passing as "repeating" on a test light curve: folded at a period shorter than the dip, or longer with one normal
visit sharing its phase, one dip looked like two. The eclipsing-binary filter's period test was not reused: it is
built for eclipses of hours to days.

**Speed and an outlier fix (2026-10-30, PR #78).** The window search and the repeat conditions are compiled with
numba, with numpy's summation order copied so results are bit-identical to the plain version: 255 → 23 ms per star
at DR4's sampling, 18 → 5 ms on DR3. The full DR3 trial (below) then showed stars with one extreme bright visit (up
to 10 times the star's flux) getting repeat scores in the hundreds: that visit tilted the straight-line trend
removed before folding, and the tilt read as a repeating dip. Visits beyond 20σ are now left out of that line's fit
(`period.DETREND_CLIP`; they stay in the search). This removed 94-100% of such flags; the quiet stars' threshold did
not move, and recall of the repeating long dip went from 57.0% to 55.9%.

**New shapes set aside before any tuning** (pushed in 5495052, before the search existed;
`inject.lightcurve.HELD_OUT_SLOW`): sawtooth dips (a fast drop and slow recovery each cycle), a drifting cycle
(period lengthening or shortening by 15-40% over the span), uneven dips (each cycle 20-100% as deep as the deepest)
and paired dips (two dips per period, the second at phase 0.2-0.8 and 20-100% as deep). Calibration used only an
equal, symmetric repeating long dip (`repeat_long_dip`) and the spent slow cycle.

**Noise yardstick.** Three were compared on a quarter of shard 0 (1,416 quiet stars, 3,834 other stars):

| Yardstick | Repeating long dips found (SNR ≥ 20) | Slow cycles found | Other stars flagged (dimming / brightening) |
|---|---|---|---|
| Visit-to-visit scatter (as the detector) | 69% | 25% | 3.0% / 2.0% |
| Scatter inside each visit | 79% | 55% | 22% / 22% |
| Scatter about the fitted two-level model | 31% | 17% | 1.7% / 1.6% |

Scatter inside a visit misses the variability between visits that most variable stars have, so it would pass a
fifth of DR3's light curves. The visit-to-visit scatter was kept. Its weak spot is the cycle of a few months:
with Gaia's visits one to two months apart, such a cycle shows up as visit-to-visit scatter and raises the
yardstick with it.

**Calibration** (shard 0/10, all 323 files): thresholds at 0.1% of quiet half-B stars' repeated brightenings,
G < 19: s_rep > 7.55; G ≥ 19: s_rep > 9.71. Quiet half-A stars: 5 repeated dimmings against 7 repeated
brightenings of 7,132. A 5% sample of all stars (49,230): 935 repeated dimmings against 881 brightenings. At G < 19
the search adds 486 stars (1.25%) to the 1,663 (4.3%) that the detector flags.

Found at G < 19, detector or period search together (detector alone in brackets), by expected signal-to-noise:

| | 10-20 | 20-40 | > 40 |
|---|---|---|---|
| Repeating long dip | 13% (2%) | 57% (14%) | 78% (54%) |
| Slow cycle | 4% (3%) | 17% (11%) | 31% (22%) |

Slow cycles with more than three cycles in the span are almost never found (1%), for the reason above. A single
long dip is called repeating in 6% of cases. DR4 has about twice the visits and span, which helps both.

**Blind test of the period search** (written down and pushed before the draw; same procedure as above,
`python scripts/slow_blind.py make --rule slow-v2 --batch slow-blind-2`):

- **Rule `slow-v2`** (`cosmos_search/slow/rule_v2.py`, files in `prereg/rules/slow-v2/`): slow-v1 or s_rep above
  its threshold.
- **Stars:** shard 2/10 of DR3, none shared with shards 0 or 1.
- **Batch:** 5,000 quiet stars as they are at G < 19 and 2,000 at G ≥ 19; 600 of each new shape and 300 each of
  the repeating long dip and the slow cycle, at G < 19; 300 repeated brightenings, reported only.
- **P1:** natural stars flagged at most 0.6% in each magnitude range (expected about 0.2%: 0.17% on quiet half A).
- **P2:** each new shape at signal-to-noise ≥ 20 found at least as often as the calibrated shape it most resembles
  in the same batch, minus 15 points: sawtooth, uneven and paired dips against the repeating long dip; the
  drifting cycle against the slow cycle.
- **P3:** the calibrated shapes at signal-to-noise ≥ 20 within 10 points of shard 0: repeating long dip 65%,
  slow cycle 25%.
- **What follows.** All pass: slow-v2 is frozen for DR4 with this batch's recall. P1 fails: thresholds from more
  quiet stars, as for slow-v1. P2 fails for a shape: its recall is stated as a weakness, and whether a fix is
  worth a third blind test is Cora's call. P3 fails: find out why before freezing.

**Result** (batch `slow-blind-2`, 10,300 light curves; flags archived in `prereg/verdicts/slow-blind-2/` and
pushed before the reveal). The test fails on P2:

| | Result | Pass? |
|---|---|---|
| P1 | 5 of 5,000 natural stars flagged at G < 19 (0.10%); 1 of 2,000 at G ≥ 19 (0.05%) | yes |
| P2 | Paired dips 49% (46/93), sawtooth dips 40% (33/83), uneven dips 40% (19/47), each against the repeating long dip's 65% minus 15 points; drifting cycle 11% (19/180) against the slow cycle's 19% | **no** (the three dip shapes) |
| P3 | Repeating long dip 65% (28/43), slow cycle 19% (18/93) | yes |

The detector alone finds, at signal-to-noise ≥ 20: 35% of repeating long dips, 24% of paired dips, 16% of
sawtooth dips, 15% of uneven dips, 11% of slow cycles and 4% of drifting cycles. With the period search added,
the same shapes are found 65%, 49%, 40%, 40%, 19% and 11% of the time. 6% of repeated brightenings were flagged
(reported only).

Why the misses (after the reveal): every miss at signal-to-noise ≥ 20 has a fold score below the threshold. None
was a strong fold turned away by the repeat conditions. Part of the gap is the signal-to-noise measure, which
credits signal that one window in phase cannot gather (the second dip of a pair, the shallow cycles of uneven
dips, the long tail of a sawtooth). Measured instead by the ideal (matched-filter) signal-to-noise, √Σ(deficit /
σ)² over the visits, the recalls at ≥ 20 are 53%, 43% and 46% against 67% for the repeating long dip: the gap
narrows a little but stays. Sawtooth dips with more than five cycles in the span are found 22% of the time.

What follows, as written above: the three dip shapes' recall is stated as a weakness, and whether a fix is worth
a third blind test is Cora's call (asked 2026-10-23).

**Attempts to improve it after the blind test** (2026-10-23, Cora: worth improving a little). Tuned only on
shard 0 and on shapes already spent (the four of slow-blind-1 and slow-blind-2). Every variant is compared at the
same false-alarm rate: its threshold is the 0.1% point of repeated brightenings among all 12,040 quiet stars at
G < 19 (both halves; thresholds from half B alone, about 6,000 stars, rest on 6 stars and move by ±1 between
runs, enough to reorder variants). Recall is at signal-to-noise ≥ 20, G < 19, fakes in a quarter of shard 0's
files (142-529 per shape).

| Variant | Quiet stars flagged | All stars flagged | Repeating long dip | Paired | Sawtooth | Uneven | Slow cycle |
|---|---|---|---|---|---|---|---|
| Current (slow-v2) | 0.12% | 2.6% | 66% | 58% | 56% | 50% | 18% |
| Plus weaker folds whose BP and RP visits are ≥ 5σ faint too | 0.12% (0.13% brightening) | 3.9% | 70% | 59% | 59% | 54% | 22% |
| Same, BP and RP ≥ 3σ | 0.13% (0.21% brightening) | 4.6% | 73% | 64% | 66% | 61% | 28% |
| Noise from the lower quartile of visit differences | 0.11% | 1.2% | 49% | 28% | 30% | 25% | 3% |
| Each window scored with the noise of the visits outside it ¹ | 0.12% | 1.2% | 47% | 32% | 37% | 27% | 1% |
| Plus two separate dimming episodes, no period ¹ | 0.22% (0.15% brightening) | 3.0% | 66% | 56% | 54% | 51% | 23% |

¹ Thresholds from half B only (6,039 stars); same fakes otherwise.

Earlier the same day six other changes to the fold score were tried (noise from the scatter about the fitted
dip, BP and RP averaged into G, a per-scan sky offset from neighbouring stars, the repeat conditions applied to
brightenings too, the baseline from visits outside the window, the whole window without leaving out its deepest
visit). None beat the current score at the same false-alarm rate.

What was learned:

- **Why strong repeating dips are missed.** A dip that repeats every few visits puts a large step into a third or
  more of the differences between consecutive visits, and the visit noise (from those differences) rises with
  it: shard-0 fakes with 1-9% dips went from 0.1-0.7% noise to 0.4-7%, and some were missed at a signal-to-noise
  of 80. The two fixes in the table that estimate the noise differently both lose more than they gain. Quiet
  stars' brightest windows profit from the same freedom, so the threshold rises 2-3 times. With DR3's 15-25
  visits per star there is no noise estimate that ignores a repeating dip without also ignoring chance runs.
  DR4's roughly doubled number of visits narrows this.
- **BP and RP agreement** is the one change that raises recall on every shape at the same quiet-star rate: 3-4
  points at ≥ 5σ, 7-10 points at ≥ 3σ. Its cost is the share of all stars flagged, 2.6% → 3.9-4.6%: real
  variable stars, which dim in all three bands at once. That is 50-80% more stars for the checks (S5) and for
  people to look at, for a few points of recall. Not adopted (Cora, 2026-10-23). `period.star_scores` reports the BP and RP
  agreement of the chosen window (`rep_band_z`) for use when checking a candidate.
- **The gap between strict and irregular repeats** stays at 8-16 points in every variant. The three shapes that
  failed slow-blind-2 sit 8-16 points below the repeating long dip on shard 0 too, against a margin of 15, so the
  blind test was close to a coin toss for uneven dips. A fold gathers repeats that keep their period, depth and
  shape, and the irregular shapes give some of that up by construction.

No variant is worth a third blind test, and Cora chose to keep slow-v2 as frozen (2026-10-23), with the recall of slow-blind-2 stated as its
weakness on irregular repeats.

## Fades in time order (slow-v3)

Changes after slow-v2, made on 2026-10-23 to fix a misreading the blind tests reported, not to chase recall on a
shape; the shapes searched for are the same. `cosmos_search/slow/rule_v3.py`, files in `prereg/rules/slow-v3/`.
Checked with `scripts/slow_detect_check.py` and `scripts/slow_fade_compare.py` on shard 0/10 (where the thresholds
are set) and on shard 3/10, whose stars no part of the slow-change work had used, with shard 0's thresholds.

**What was wrong.** The episode score measures runs of visits against the star's median, so it takes whatever
level the star spends most of its visits at as its normal one. A star that brightens for good partway through and
spends more than half its visits bright shows its earlier visits as a dimming episode. At signal-to-noise ≥ 20,
53% of late-starting brightenings were flagged as fades on shard 0 (16% in slow-blind-1, counted over all
signal-to-noise). Read the other way round, a star that fades early and spends most visits faint shows its early
visits as a brightening, and only the trend score could find it.

**The change** (`detect.ordered_episodes`). Runs are read in time order. A faint run that opens the light curve,
or follows visits that are themselves faint by more than half the run's depth, means the star got brighter: it
scores as a rise (`s_rise`). A bright run that opens the light curve means the star got fainter afterwards: it
scores as a fade (`s_fade`, kind "drop"). Runs with normal visits before them score as before ("dip", or "fade"
when they reach the last visit). The two scores are exact mirror images (flipping a light curve swaps them; a test
checks it), so rises stay the false-alarm estimate for fades.

**What it costs.** A dimming already under way at the first visit and over inside the span looks exactly like a
brightening; Gaia alone cannot tell which level is normal. On shard 0 these were 21% of the fake long dips (their
start is drawn up to half a duration before the first visit, and durations reach 1,500 days against DR3's ~1,000)
and every one of the long dips the change lost. They now go to a second list (`early`: faint at the first visit,
normal later, change date not shared), which brightening stars also land in. Surveys from before 2014
(Pan-STARRS, Catalina, ASAS-SN's first years) tell the two apart for any star on it. DR4's span is twice DR3's,
so fewer dips straddle its start.

**An instrument effect it uncovered.** With brightenings no longer absorbing them, 50 quiet shard-0 stars at
G < 19 showed a drop in the 1% tail, against 24 jumps. Most are faint (G 18-19) red stars in the Galactic bulge
and plane, and they drop on the same visit as hundreds of other stars in the same level-6 pixel (~0.84 deg²): on
BTJD 269.2 (2015-09-07), 383 of 6,103 stars in one bulge pixel drop by more than s = 4, and other pixels show the
same at their own dates. A step shared by a crowd on one scan is instrumental (crowding, which depends on scan
direction, is the likely cause); a real occulter changes one star. `sky.shared_changes` counts, for each star,
the other stars in its pixel whose change (in either direction, s > 4) falls within 1.5 days of its own, and sets
the change aside when that count is beyond chance (Poisson probability below 10⁻³ given the pixel's strong
changes spread over its distinct dates). It applies to both directions, so the mirror stays a mirror. On DR4 the
test runs over the selected stars of each pixel; crowd artefacts pass the selection together, so they are there to
be counted. The old median score had hidden the effect by counting these stars as brightenings, which raised its
threshold: its episode score flagged 4-5 quiet stars against 12-18 mirror brightenings, so it ran below the
intended 0.1% and lost recall for it.

**Thresholds** (0.1% of quiet stars of both halves of shard 0, mirror with shared dates set aside): G < 19,
s_fade > 6.04 (was s_dim > 7.41 in slow-v1, set on half B only); G ≥ 19, 17.09. The trend thresholds (3.94, 7.84)
and the period search (s_rep > 7.55, 9.71) are unchanged.

**Result.** Found at signal-to-noise ≥ 20, G < 19 (episode and trend scores; the period search is unchanged):

| | Shard 0, median score (slow-v1/v2) | Shard 0, slow-v3 | Shard 3 (fresh), median score | Shard 3 (fresh), slow-v3 |
|---|---|---|---|---|
| Late-starting brightening flagged as a fade | 53% | 2.5% | 57% | 2.0% |
| Late-starting fade | 84% | 90% | 84% | 92% |
| Steady fade | 100% | 100% | 100% | 100% |
| Long dip, fading list | 85% | 68% | 83% | 68% |
| Long dip, fading or `early` list | | 86% | | 83% |

At signal-to-noise 10-20 slow-v3 finds 67% of late-starting fades (shard 3) against 62%, and 29% of long dips
against 27%.

Quiet stars at G < 19 flagged as fading by the episode score: shard 0, 11 of 11,994 (mirror 13); shard 3, 18 of
14,139 (mirror 11), so the episode score now runs at the intended 0.1%. With the trend score added (whose own
fades outnumber its brightenings, 19 against 12 and 26 against 15, as among DR3 variables generally), 0.23% and
0.31% of quiet stars are flagged, against 0.19% and 0.22% with the median score and against the blind tests'
bound of 0.5-0.6%. Among all DR3 light curves at G < 19, the fading list holds 4.8% of stars (4.6% on shard 3)
and the `early` list 1.2% (1.1%); these stars are DR3's variables, so neither share carries over to DR4's
mostly quiet stars.

The `early` list at the same threshold flags 1-4 quiet stars per shard (mirror: 5).

## Sky term (S2)

`slow/sky.py`. The magnitude and colour drift leaves a sky pattern in quiet stars' trends (S2). Gaia observes every
star in a small patch in the same scans, so the pattern was modelled as an offset per patch and visit, measured
from the patch's other stars, as differential photometry uses comparison stars: a precision-weighted mean of the
stars' visit deviations (each star weighted by its own scatter, stars more than 4σ off left out), shrunk towards
zero by τ² / (τ² + error²), with τ the scatter of true offsets found by maximum likelihood. Each star is corrected
from the stars of the other half, so it never corrects itself.

On shard 0, at level 6 (~0.84 deg², median 65 stars per pixel; DR3 has too few quiet stars per pixel, 2-5, so every
star with a light curve served, weighted by its scatter): τ = 0.13% in G, 0.20% in BP and 0.09% in RP per visit
(median 30 comparison stars per pixel and visit), real but small. The correction lowers the sky pattern in quiet stars' trends from 1.44 to 1.21 times what
shuffling stars between pixels gives (p 0.01). At 0.1% false alarms it moves recall by under one point and the
quiet stars flagged by one or two. It is not in slow-v3. It costs neighbours' light curves for every star, and on
DR4 those come with the sample only when the sample is whole sky patches. It stays available as a check for
candidates (`sky.offsets` on a candidate and its fetched neighbours), next to the shared-date test above, which is
the part of the sky term that changes results.

## Release-day thresholds

`slow/release.py`, `scripts/slow_release_thresholds.py`. Every threshold above comes from DR3's quiet stars, which
are DR3 variables that happen to be quiet. On release day the same 0.1% false-alarm level is set again from DR4's
own stars:

1. Sample: every star in a few hundred random small sky patches (`release.patch_ranges`, HEALPix level 9,
   ~0.013 deg² each). Each star has the same chance of being drawn whatever the sky density, and its neighbours
   come with it for the shared-date test. Fetched by source_id range, which is how the AIP mirror is fast
   (`epoch.mirror_epoch_photometry`) and how the bulk files are split (docs/infra/dr4-downloads.md).
2. Quiet stars from the light curves themselves (`release.quiet_ids`, the S2 definition).
3. Drift tables fitted on each half of the quiet stars, each correcting the other half's stars, so no star is
   corrected with a table it helped fit and both halves count towards the thresholds (half B alone left ~6 stars
   beyond the 0.1% point on DR3, and thresholds moved by ±1 between runs). The search itself uses the table
   fitted on all quiet stars.
4. Scores for every star (shared-date test), period search for quiet stars, then the 0.1% points of the mirrors
   per magnitude range with the count beyond them and a bootstrap 16-84% interval (`release.thresholds`).
5. Rule files in slow-v3's format plus `release.json` (`release.write_rule`).

Since 2026-11-10 ("Every star scored" below) the script needs no catalogue (each star's G and colour come from its
light curve; `--aux` is optional), scores every star with the compiled summary, measures each trend against the
median of nearby stars of similar brightness before taking the trend's 0.1% point, and runs the period search only
on the quiet stars whose bound on its score is among the largest 8%, checking that every star left out is bounded
below the threshold found (it widens the share until that holds). The sample can be a share of the bulk files:
`slow_search.py pass --files-share 0.002 --keep-lightcurves all`. `--drift-share` fits the drift table on a
random share of the quiet stars when the sample is larger than memory.

How many: the false-alarm rate at a threshold set from N quiet stars is known to about 1/√(0.001 N), so ±20%
needs 25,000 quiet stars per magnitude range (`release.stars_needed`). At G < 19 that is about 30,000 stars with
light curves; at ~1,500 light curves per second from the AIP mirror, under a minute; at ESA DataLink's 0.3-1.5 per
second, 6-28 hours.

**Checked on DR3** (shard 0, run end to end by the script, 2026-10-23): 1,004,731 stars, 11,657 quiet at G < 19
and 2,212 at G ≥ 19 by the light-curve definition (the frozen thresholds used 11,994 and 2,389 by the DR3 features'
definition). Thresholds with their bootstrap 16-84% intervals, against the frozen slow-v3 values:

| | Release-day code | Interval | Frozen |
|---|---|---|---|
| G < 19, episode (s_fade) | 5.87 | 5.65-6.41 | 6.04 |
| G < 19, trend (slope_z) | 3.82 | 3.78-3.95 | 3.94 |
| G < 19, repeat (s_rep) | 7.55 | 6.97-7.69 | 7.55 |
| G ≥ 19, episode | 15.33 | 14.82-16.73 | 17.09 |
| G ≥ 19, trend | 7.64 | 7.38-7.73 | 7.84 |
| G ≥ 19, repeat | 10.05 | 9.00-11.52 | 9.71 |

The code reproduces the frozen values within their sampling noise, apart from the faint episode threshold, which
sits on 3 stars beyond the 0.1% point. That is the case for ~25,000 quiet stars per magnitude range on release day.

## Every star scored (2026-11-10)

`slow/summary.py`, `slow/search.py` (`score_all`), `slow/kinds.py`, `slow/fallback.py`, `scripts/slow_search.py`
(`pass`, `flag`, `cuts`), `scripts/slow_inject_test.py`, `scripts/slow_fallback.py`. The whole-project review of
2026-11-10 measured that the archive-side selection above (S4) picks none of the steady fades up to 3% a year
injected into quiet FGK dwarfs: its 0.1% cut on `slow_z` sits among stars with very large scatter. The bulk
light-curve files are read in full anyway, so the run now scores every star in them and chooses nothing before a
light curve is read. Everything below is measured on DR3's Andromeda field, the one part of DR3 with a light curve
for every star: 1,080,003 stars, 1,003,245 with enough visits for scores, 122 million epochs.

**Summary of every star.** `summary.star_summary` gives, in one compiled loop over stars, the trend and its
error, the time-ordered episode scores with their kinds and windows, the visit scatter and Abbe value, each
band's trend on the transits all three bands have, and the quiet-star test. On 3,803 real stars the trend and
episode numbers equal the detector's (`detect.star_features`): scores, windows and kinds bit for bit, trends to
2 × 10⁻¹². So the trend and episode parts of the rule are tested on every star at every magnitude. G and BP-RP
come from the light curve's median fluxes (against the catalogue: median difference 0.0006 mag, scatter 0.022
mag in G), so no catalogue query comes before the lists.

**Period search only where it can pass.** A window of k visits scores the sum of k − 1 of the star's detrended
visit deviations over √(k − 1) σ, and no k − 1 visits sum lower than the k − 1 lowest. That bound (`rep_bound`)
was at or above the period search's score for all 1,423 stars tested, with a median ratio of 1.35. The period
search runs only on stars brighter than G 16 whose bound exceeds the repeat threshold: 6,103 of the field's stars.
The detector itself runs only on stars within 0.8 of a threshold, for the per-band depths. On 20,000 stars the
new pass and the detector plus period search on every star flag the same stars: fading 79 and 79, early 10 and
10, rising 69 and 69, repeated brightening 50 and 50, no star differing.

**Trends against neighbours.** The drift correction is fitted on quiet stars, which are 1% of stars brighter than
G 14. After it the median trend of the field's stars is +0.94 errors at G 12-13 (+0.07% a year), +0.24 at G 13-14
and under 0.15 fainter. DR4's trend errors are about 2.7 times smaller, so the same offset would be about 2.5
errors. `summary.reference_table` takes the median trend of all stars in each sky pixel and 0.5 mag range of G, at
the finest HEALPix level (6 down to 0) where the cell holds 50 stars, and `apply_reference` measures each star
against it with the median's error added to its own. 98% of stars take their reference at level 6. The median
is then under 0.01 errors in every magnitude range (by construction; `flag` writes both medians to
`summary.json` as the release-day check), and at G 12-13 the stars beyond 3.8 errors go from 51 brightening and 13
fading to 20 and 20 (measured with DR3's marks).

**Epochs without rejection marks.** DR3 marks the epochs its variability processing rejected (10% of G epochs in
one file of this field) and `drift.normalise` drops them. DR4's bulk epoch photometry has no such column (found by
the release-day practice run). `drift.own_good` now drops epochs whose error is over twice the star's median error
in that band, unless more than 20% of the star's epochs would go. Each setting was run end to end with its own
thresholds from the field's quiet stars; G < 19:

| Epochs dropped | Quiet stars | Thresholds (episode, trend, repeat) | Fading / rising lists | Above strength 5 | 0.5% a year fade listed | 1% a year |
|---|---|---|---|---|---|---|
| DR3's marks | 265,172 | 5.43, 3.65, 6.93 | 3,008 / 2,883 | 28 / 29 | 63% | 83% |
| None | 165,381 | 5.54, 4.71, 7.33 | 2,751 / 4,145 | 53 / 82 | 43% | 71% |
| Our own test, marks ignored | 278,452 | 5.42, 3.68, 6.77 | 3,960 / 3,707 | 54 / 106 | 64% | 82% |

With no test, bright outliers widen the brightening mirror, the trend threshold rises by a third and small fades
are lost. With our own test the thresholds and the recall return to what the marks give. The lists are still 30%
longer and the far tail holds twice as many fades and nearly four times as many brightenings as with the marks:
the artefacts left are mostly bright ones, such as close pairs that are resolved in some scan directions. Tried
and dropped: a clip of isolated spikes against their neighbours in time (lists 5% longer, recall unchanged). A
clip about the star's median is never used: it removes the deep episodes of quiet stars. DR4's per-transit flags
(a neighbour in the window, attitude, satellite events; `gaia/dr4.py` reads them into `flags`) are the next thing
to test against this in the first hours.

**Injection test.** `scripts/slow_inject_test.py` puts fakes into real light curves and scores them as the run
does: the summary, the reference built from the real stars and looked up with the host's magnitude, the
shared-date test against the real stars of the host's sky pixel, then the rule. 4,800 hosts brighter than G 19
(600 from each magnitude bin) and 1,200 fainter, 16 kinds of fake each, marks ignored, thresholds from the field.
Share listed as fading, beside the share the earlier archive-side cut would have picked at all:

| Steady fade, % a year | 0.3 | 0.5 | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|---|---|
| New run, G < 13 | 49% | 77% | 89% | 94% | 95% | 97% | 99% |
| New run, G 13-16 | 59% | 80% | 90% | 94% | 96% | 97% | 98% |
| New run, G 16-19 | 15% | 38% | 69% | 90% | 94% | 96% | 97% |
| New run, G ≥ 19 | 1% | 2% | 6% | 26% | 46% | 68% | 88% |
| New run, quiet hosts on no list, G < 19 | 37% | 61% | 86% | 99% | 100% | 100% | 100% |
| Archive-side cut, G < 13 | 0% | 0% | 0% | 0% | 0% | 3% | 60% |
| Archive-side cut, G 13-16 | 0% | 0% | 0% | 0% | 0% | 0% | 30% |
| Archive-side cut, G 16-19 | 0% | 0% | 0% | 0% | 0% | 0% | 2% |

By kind of star at 1% a year: FGK dwarfs 92% (2,456 hosts), giants and subgiants 91%, M dwarfs 52%, stars with
uncertain parallaxes 61%. Untouched hosts are on the fading list 1.1% of the time and the same fakes turned into
brightenings land on the rising list at the fades' rates (83% at 1% a year) and on the fading list 1.4% of the
time. Drawn families (`inject.lightcurve`), all hosts with G < 19: steady fades of any size and colour 63%, ramps
38%, long dips 16% (55% of those deeper than 10%), repeated long dips 14%, slow cycles 10%; the archive-side cut
picks 1% of the long dips and none of the rest. The long-dip and cycle numbers are the detector's own limits on
DR3's 19 visits a star, which this work does not change.

**Reading order within kind of star.** By raw strength the lists start with known variables. `cuts` now gives
each listed star its kind (`kinds.star_kind`: parallax, absolute magnitude, colour, no extinction correction) and
each fade its standing against the brightenings of the same kind, `q_kind` (`kinds.rank`, a knockoff-style
estimate of the share of chance and ordinary variability among the kind's fades at least as strong). The fading
list is written in reading order: FGK dwarfs, M dwarfs, hot dwarfs, giants, then the rest; within a kind, stars
without a variability class first, then by `q_kind`, then brightest first. In the field (fading / brightening
above each strength):

| Kind | > 1 | > 2 | > 3 | > 5 |
|---|---|---|---|---|
| FGK dwarf | 1,713 / 1,571 | 129 / 199 | 34 / 83 | 11 / 25 |
| M dwarf | 394 / 629 | 30 / 68 | 7 / 28 | 1 / 8 |
| Giant or subgiant | 253 / 210 | 42 / 35 | 18 / 13 | 7 / 2 |
| No measurable motion (quasars, M31's stars) | 2,801 / 2,620 | 753 / 720 | 233 / 229 | 34 / 26 |

Dwarfs brighten at least as often as they fade at every strength, so no dwarf has `q_kind` below 0.5 and their
order falls back on brightness; only giants (7 stars) and hot dwarfs do. Far out the brightenings outnumber the
fades (bright artefacts, above), so `q_kind` overstates chance there. What the order buys is position: a fake FGK
dwarf fading 3% a year has a median strength of 7.8 and two real FGK dwarfs of the field's 1,713 listed ahead of
it; at 1% a year, strength 3.3 and 22 ahead.

**Cost.** The pass took 0.9 min for the field on 4 cores: 0.2 ms per star per core at DR3's sampling, against
14.7 ms for the detector and period search on every star. For DR4's ~2 × 10⁹ stars at about twice the epochs
that is 200 to 300 core-hours, less than the download. Stored: about 70 bytes a star (every star brighter than G
19, a tenth of the fainter ones, any star near a threshold), so about 60 GB for DR4, plus light curves for the
0.7% of stars that are brighter than G 19 and near a threshold on the fading side, or in the random 0.1%. The thresholds step took 3.5 min for the
field.

**DataLink fallback list.** If the bulk files are late or absent, light curves come one star at a time at 0.3 to
1.5 a second, and the order of fetching decides what can be found. `slow_fallback.py list` writes one order:
the 2,849 stars DR3 showed still fading at its end, each with a prediction written before DR4 (the DR3 trend
continued to 2020-01-20, the change expected since DR3's last visit, and the significance expected over DR4's
span; `fallback.outcome` then calls each continued, slowed, stopped or reversed), then the known answers (942
stars once the 25 that are both an alert and a VSX star are counted once; none is on the watch-list), then
the stars `slow_fallback.py abbe` selects on release day: brighter than G 16 with a G Abbe value in the lowest 1%
of their magnitude bin in `all_source_photometry`. In the injection test the Abbe value alone at a 1% share picks
59% of 1% a year fades and 98% of 3% a year fades on quiet hosts, where `slow_z` at 0.1% picks none.
The first two lists together are 3,791 stars: one to four hours of DataLink at 1 to 0.3 light curves a second.

## Release day, end to end

`scripts/slow_search.py` runs the search step by step; `slow/search.py` holds the scoring and flagging. On DR4:

1. Routes: `python -m cosmos_search.gaia.dr4_routes --out routes.json` (docs/infra/dr4-downloads.md).
2. Thresholds from DR4's own stars: `slow_search.py pass --out S --dr4-bulk --files-share 0.002 --keep-lightcurves
   all` reads a random share of the bulk files (whole patches of sky) and keeps their light curves, then
   `slow_release_thresholds.py --lightcurves S/lightcurves --out RULE --check-against prereg/rules/slow-v3`
   (section above). Without bulk files: `slow_search.py sample --out S --patches 300 --level 9` from a mirror.
3. Every star scored: `slow_search.py pass --out R --rule RULE --dr4-bulk`, split over machines with `--shard i/n`
   and resumed from done-markers ("Every star scored" below). Each bulk file is downloaded, read and deleted. It
   writes a summary row for every star brighter than G 19 and a tenth of the fainter ones (plus any star near a
   threshold), the full scores of the stars that need them, and the light curves of stars near a threshold on the
   fading side. No star is chosen before its light curve is read, and no catalogue query is needed: each star's G
   and colour come from its light curve.
4. Lists: `slow_search.py flag --run R --rule RULE --out L`: each star's trend against the median of nearby stars
   of similar brightness, the shared-date test, then the fading list, the `early` list and the rising list, one
   sky region at a time. `L/summary.json` holds the median trend by magnitude before and after the reference
   (the first-hour check that trends sit at zero) and the counts above each strength.
5. Cuts: `slow_search.py cuts --run L` looks up parallax, proper motion, colour, BP/RP blend counts and the
   variability class of every star above strength 1 either way (the run's one catalogue query, through
   `targets.releases`, so DR4's table names are in one place). It marks stars with no measurable motion as
   `still` (quasars, galaxies, stars in other galaxies), gives each star its kind (`kinds.star_kind`) and each
   fade its standing against the brightenings of the same kind (`q_kind`), adds whether each fade's G change
   agrees with BP and RP (`band_verdict`, "G against BP and RP" below), and writes the fading list in reading
   order: ordinary dwarfs with good parallaxes first, stars without a variability class before those with one.
   `L/cuts_summary.json` has the fading and brightening counts above each strength by kind of star.
6. If the bulk files are late or absent: `slow_fallback.py` ("DataLink fallback list" below) gives the order in
   which to fetch light curves one star at a time, and `slow_search.py pass --lightcurves DIR` scores whatever
   has arrived. `slow_search.py select`, the earlier archive-side choice of stars, is kept only for comparison:
   it picks none of the steady fades it was meant to find ("Every star scored").
7. First-hour checks on the lists: the median trend by magnitude (`L/summary.json`, `trend_offset_by_G`), the
   fading against the brightening counts by kind (`L/cuts_summary.json`), and the watch-list's predictions
   (`slow_fallback.py check`).
8. Reading: the fading list in its reading order, kind by kind, each down to where its fades stop standing
   clear of the same kind's brightenings. The checks:
   `slow_checks.py --fading L/fading.parquet --lightcurves R/lightcurves --out L/checks --skip-still --skip-band
   --skip-class ECL --n 1000`. DR4's eclipsing-binary class is left out first; `--lightcurves` adds the
   eclipsing-binary filter on G and ZTF (`eb_explained`, its period and duration), kept beside each star's
   review as information (its test failed for repeating long dips, "Eclipsing binaries among the candidates"
   below).
   Stars marked `g_only` or `bp_rp_only` are not checked but stay on the fading list with their verdict and are
   read after the rest, as the still stars are.
9. Figures for the stars worth showing: `viz_candidates.py figures --run ... --ids ...` (docs/viz.md).
10. Blind review of the stars the checks covered (docs/vetting.md §14-§16), `scripts/slow_review.py`:
    `files --checks L/checks --fading L/fading.parquet --lightcurves R/lightcurves --rule RULE --hosts
    S/lightcurves --out D` writes one review file per star, built as the calibration's files were
    (`slow.vet.review_file`), and among them planted fades of known answer: 30, or 5% of the stars where that is
    more, put into quiet stars of the thresholds' sample, kept when the run would have flagged them, and built
    into files by the same function (docs/vetting.md §16). Which files are fakes is written only encrypted.
    `launch --dir D` prints the arguments for `scripts/stage_a_reviews.js` (the model and effort of
    `prereg/reviewers/slow-cal-1.json`; run with the Workflow tool, never the Agent tool, from a thread Cora has
    not typed in); `stage_a_audit.py --reviewer prereg/reviewers/slow-cal-1.json` counts only reviews served by
    that reviewer and handed nothing but their task; `vetting commit --audit --archive --plan D/plan.json.age`
    records the verdicts; and `table --dir D --checks L/checks` writes `D/reviewed.csv`, the list to read, with
    the eclipsing-binary check beside each verdict, and `D/fakes.json`: how many planted fades the reviewer
    shortlisted and whether that is consistent with the calibration (at least 20 of 30). If it is not, no star
    is set aside on the batch's verdicts. The files hold the calibration's four tests and nothing from the other
    surveys of "Other surveys" below, where a fake's host would be seen unchanged. `eb_explained` goes into the
    files only with `--with-eclipse` ("Eclipsing binaries among the candidates", decision of 2026-11-11). A star
    whose NEOWISE or ZTF fetch failed in the checks is left out and listed, so that its checks are run again
    first.
11. The limit on how common steady fades are among FGK dwarfs, find or no find (docs/arms/occurrence.md, "On
    DR4"): on day one `slow_occurrence.py` on the light curves of step 2, with the pass and flag steps run on
    that directory; then on every star, from `R/summary`, `L/reference.parquet`, the light curves the pass keeps
    of its random share of stars, and the dwarfs' parallaxes fetched in sky slices (`slow_occurrence.py aux
    --sky 4`).

### Trial on all of DR3 (2026-10-30)

Every step that does not need DR4, on every DR3 star with a light curve: 11.6 M stars from ESA's bulk files (3,386
files, 74 GB, streamed from the CDN), thresholds from 1,000 random level-5 sky patches fetched from AIP (1.07 M
stars), and the selection query as DR4 will run it, on DR3's `gaia_source` with a stand-in score.

**Time and failures.**

| Step | DR3 trial | For DR4 |
|---|---|---|
| Selection (ESA TAP, 768 slices, 5 jobs) | 14.6 min, no failed slice | 4-5 s a slice on 1.8 × 10⁹ rows |
| Sample for thresholds (AIP) | ~500 light curves/s with 8 workers | AIP refuses one query over ~10⁴ stars; dense patches are now split in quarters |
| Thresholds | 17 min for 1.07 M stars | |
| Scores | 11.8 h on 4 cores (272 stars/s, 14.7 ms per star per core), no failed file | ~25-35 ms per star per core at DR4's sampling: ~20 core-hours for a 0.1% selection (2.5 × 10⁶ stars), ten times that for 1% |
| Lists | 100 s, 4.0 GB peak (10.8 GB in one pass) | memory follows the largest sky region (1/48 of the sky) |
| Motion | 6 min for 1.47 M stars (30 uploaded queries) | |
| Checks | 27 s a star with 6 at once, no failed fetch | IRSA allows 4 calls at once (NEOWISE ~2 min, ZTF ~1 min each): 10³ candidates in ~7.5 h |

Fixed on the way: the period search's outlier bug (above), the shared-date test comparing every star with every
patch (33 min → 46 s on 4.7 M stars, same results), joins against whole-sky tables inside per-file and per-region
loops (seconds each time), and the checks running one star after another.

**Thresholds** (quiet stars of the 1,000 patches; frozen slow-v3 values for comparison):

| | Quiet stars | Episode (s_fade) | Trend (slope_z) | Repeat (s_rep) |
|---|---|---|---|---|
| G < 19, all patches | 65,867 | 5.63 | 3.80 | 7.00 |
| G < 19, M31 field | 53,569 | 5.50 | 3.86 | 6.94 |
| G < 19, elsewhere | 12,298 | 5.92 | 3.41 | 7.17 |
| G < 19, frozen | 11,994 | 6.04 | 3.94 | 7.55 |
| G ≥ 19, all patches | 70,815 | 6.37 | 3.90 | 7.66 |
| G ≥ 19, M31 field | 68,962 | 5.59 | 3.64 | 7.41 |
| G ≥ 19, elsewhere | 1,853 | 15.27 | 7.41 | 9.26 |
| G ≥ 19, frozen | 2,389 | 17.09 | 7.84 | 9.71 |

The M31 field (5.5° around M31, Gaia's GAPS release) is the only part of DR3 with light curves for every star, so
it is the closest thing to DR4's population. Elsewhere DR3 published light curves only for stars already found
variable, and its faint "quiet" stars are mostly quasars. So DR4's faint thresholds will likely sit near 5.6, not
17; the release-day script sets them from DR4's own stars, so nothing frozen needs to change. The list step used the
all-patches values.

**Lists.** G < 19: 9.01 M stars scored, 565,241 fading (6.3%), 45,697 `early`, 365,905 rising. G ≥ 19: 2.60 M
scored, 324,915 fading (12.5%), 59,787 `early`, 317,769 rising. These are DR3's variable stars, so the share is far
above the 0.1% the thresholds allow for quiet stars. In the M31 field, which is mostly ordinary stars, 0.49% (G < 19)
and 0.37% (G ≥ 19) are on the fading list, and the rising list is as long or longer at every strength:

| Strength above | 1 | 1.5 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|---|
| M31 field G < 19, fading / rising | 2,818 / 4,010 | 602 / 1,032 | 256 / 495 | 88 / 160 | 16 / 40 | 3 / 6 |
| Elsewhere G < 19, fading / rising | 611,688 / 443,339 | 227,913 / 166,270 | 115,000 / 82,366 | 45,098 / 29,244 | 12,098 / 5,923 | 1,344 / 453 |

In a population of ordinary stars the fading count does not stand clear of the rising count at any strength. So
DR4's lists will hold far more stars than anyone can read, nearly all of them chance passes and ordinary
variables; they are for reading from the top, and the rising count beside them says how far down reading still
pays.

**Shared-date test.** It set aside 72,210 fades and 49,359 brightenings at G < 19 that had passed the episode
threshold (14,834 and 9,746 at G ≥ 19). Almost all are in dense fields: 1-2% of strong changes in level-6 pixels
with under 400 scored stars, 33% at 900-2,100 and 58% above. These are not borderline cases: a median 128 other
stars in the pixel changed within 1.5 days, 6 times what chance gives. 76% of the fades set aside fall in five
10-day windows (mid-June to early July 2015 in the Galactic plane, late August to early October 2015 mostly towards
the bulge), each across 180-650 pixels at once, so they come from a few scans, not single stars. In the M31 field
the test sets aside about 20% of strong fades and 17% of strong brightenings, most of them in the second half of
June 2015, in both directions. It works as intended and treats both directions alike. Its cost is any real
change that happens on one of those few dates in a dense field.

**Star types at the top.** By DPAC class, the 1,000 strongest fades at G < 19 are 82% eclipsing binaries (mostly
flagged by episode and repeat score together). With those left out, half the next 1,000 are quasars ("AGN"), then
spotted stars, long-period variables and Be stars. At G ≥ 19, 93% of the fading list has no measurable parallax
or proper motion, and 99% of those are DPAC quasars. On the 10,000 strongest fades in each magnitude range, that
test picked 9,665 of 9,688 DPAC quasars at G ≥ 19 and 2,367 of 2,376 at G < 19, and only 20 and 15 other stars
(mostly long-period variables and eclipsing binaries). With those stars left out, fading stands clear of rising at G < 19 at every strength (9,075 against
3,011 above 5); at G ≥ 19 rising outnumbers fading above strength 3, from outbursting cataclysmic variables (254
rising against 126 fading).

**Checks** on the 60 strongest fades at G < 17 that DPAC does not call eclipsing binaries: all are known variables
(18 spotted stars, 10 Be stars, 7 long-period variables, 4 R CrB stars, 4 δ Scuti, 2 quasars, 1 microlensing
event) or sit in DR3's short-timescale (11) and microlensing (3) tables. Nine of the 11 short-timescale stars repeat
every 20-42 days, and all 11 dip by 16-34% at the repeat: most likely eclipsing binaries DPAC's classifier did not
label, which the eclipsing-binary filter is built for. NEOWISE measured a change for 24 of the 44 episodes but a
slope for only 2 of the 14 steady fades: 9 of those 14 lie within 5° of the Galactic plane, where few NEOWISE
exposures pass the cleaning, and DR3's 2.8 years give NEOWISE 5-6 visits at best. ZTF covered 22 of the 27 stars
north of -30°.

**Known answers.** Of 77 known fading alerts with DR3 light curves, 24 are on the fading list and 3 on `early`; 70
of the 77 alerts were raised in 2018-2020, after DR3's light curves end (May 2017), so this measures little. DY Per
stars 6 of 8, R CrB stars 17 of 41 (18 more on the rising list, recovering from a decline), UX Ori stars 12 of 67.

**Still to do before DR4** (as of the trial). A check of G's change against BP's and RP's (docs/vetting.md §14),
and the eclipsing-binary filter on the candidates. Both done the same day (below), with NEOWISE's errors (S5).

## G against BP and RP (written 2026-10-30, before any fake is drawn)

**Why.** The calibration run (docs/vetting.md §14) found the blind review reads a change seen in G alone as
unexplained: 13 of 25 G-only fakes were shortlisted, 12 as unexplained. Nothing told it whether G's change agrees
with BP's and RP's. An exploratory cut found there (G's depth outside the BP and RP depths by more than 3 combined
errors) marked 14 of those 25 and none of 85 other fakes, and needs fresh fakes before use.

**The check** (`checks.band_agreement`, `checks.band_check`). G's passband covers BP's and RP's, so any change in
the source's own light changes G by d_G = w d_BP + (1 − w) d_RP, with w, BP's share of the G signal, between 0.1
and 0.9 from M dwarfs to A stars (as Stage 0, docs/vetting.md §12). That holds for grey, dusty and chromatic
changes alike. The check measures how far d_G lies outside that range, in combined errors (z), using the score
that gave the star its strength: the episode's depths in G, BP and RP over the same dates; the G, BP and RP trends
(each band's own straight line); or, for a repeated dip, the BP and RP visits at the times of G's window. The
detector and period search now record what that needs (`dim_depth_err`, `slope_bp`, `slope_rp`, `rep_depth_bp`,
`rep_depth_rp` and their errors). Verdicts, at |z| > 3:

- `g_only`: G changed more than any mix of BP and RP allows, on a star whose BP and RP transits are unblended
  (under 5% blended): an instrument fault in G.
- `bp_rp_only`: G changed less than both: the change is mostly in BP and RP's 3.5 × 2.1 arcsec window and outside
  G's 0.7 × 2.1, so a neighbour.
- `untested`: a band missing, or G deeper on a blended star (a constant neighbour in BP and RP's window dilutes
  their changes, which reads the same way), or no blend count.
- `agree` otherwise.

On release day `slow_search.py cuts` (formerly `motion`) adds `band_verdict` to the fading list, and
`slow_checks.py --skip-band` leaves out `g_only` and `bp_rp_only` stars. The rising list is not cut, so its count
stays slightly high, which errs towards reading less of the fading list.

**Test on fresh fakes.** Hosts are quiet stars at G < 19 from the 1,000 random sky patches of the DR3 trial (never
used as fakes before; stars in `rehearsal-inputs/exclude/all_used_ids.txt` left out), half from the M31 field and
half from elsewhere, each host used once. Shapes drawn uniformly from the six of the calibration run (long dip,
late-starting fade, steady fade, staircase fade, accelerating fade, repeating long dip), injected into the raw G, BP
and RP epochs, then the trial's drift table, detector and period search. A fake is kept only if the trial
thresholds flag it (with its host's real neighbours in the shared-date test) at the slot's signal-to-noise (S3's
definition, in G).

| Group | Fakes | Hosts | S/N | Should be |
|---|---|---|---|---|
| Grey, dust, re-emitting dust, a cooling star | 75 each | any | ≥ 20 | not cut |
| Grey | 100 | any | 6-20 | not cut (reported) |
| G alone (BP and RP unchanged) | 100 | unblended | ≥ 20 | `g_only` |
| G alone | 50 | blended | ≥ 20 | reported (`untested` by design) |
| A neighbour: BP and RP change, G sees a fraction 0.1-0.6 of it | 50 | any | ≥ 20 in G | reported (`bp_rp_only`) |

**Pass marks** (one-sided 95% Clopper-Pearson):

- **B1** (keeps real changes): of the 300 real-light fakes at S/N ≥ 20, those cut (`g_only` or `bp_rp_only`) have
  an upper bound ≤ 2% (at most 1 of 300).
- **B2** (catches G-only faults): of the 100 G-only fakes on unblended hosts, those marked `g_only` have a lower
  bound ≥ 30% (at least 38 of 100).

Reported whatever the outcome: each group by flavour, shape, score that set the strength, magnitude and region;
`untested` shares; and on DR3's natural stars, the verdicts of 3,000 random fades from the trial's fading list
and of the 2,000 strongest (no still stars), by DPAC class.

**What each outcome leads to.** B1 and B2 pass: `--skip-band` is part of the release-day run before the review.
B1 fails: the check is shown to reviewers as a number, never used as a cut, and the misses are read. B2 fails with
B1 passing: the cut is still used (it costs nothing real), with its low recall stated.

**Result (2026-10-30).** Both pass, twice: before and after a fix that DR3's own stars called for.

| Group | Fakes | First run | After the fix |
|---|---|---|---|
| Grey, dust, re-emitting dust, a cooling star (S/N ≥ 20) | 300 | 0 cut | 0 cut |
| Grey, S/N 6-20 | 100 | 1 `g_only` | 0 cut |
| G alone, unblended host | 100 | 54 `g_only` | 51 `g_only` |
| G alone, blended host | 50 | 25 `untested`, 25 `agree` | 22 `untested`, 28 `agree` |
| A neighbour | 50 | 23 `bp_rp_only` | 20 `bp_rp_only` |

B1: 0 of 300 cut, upper bound 1.0% (≤ 2%: pass). B2: 54, then 51, of 100 marked, lower bounds 45% and 42% (≥ 30%:
pass). G-only fakes are caught 38 of 54 times at G < 17 and 13 of 46 at G ≥ 17, where BP's and RP's own noise
hides a missing change, and more often for episodes (31 of 53) than trends (17 of 41).

*What DR3's stars showed.* In the first run the check marked 362 of the 2,000 strongest natural fades (18%) and 35
of 3,000 random ones as G-only, 309 of them DPAC eclipsing binaries. Their raw transits show G, BP and RP falling
together. Two faults in our measurement made the bands differ. DPAC rejects 2-5% of epochs in one band and not
the others, no more often in a dip than outside one (measured on 600 of these stars); a dip seen in one or two
transits that lost its BP or RP epoch then reads as G alone. Fakes cannot show this, since they are injected into
the epochs that survived. And BP's and RP's visits at an episode's edges fell outside a window set by G's visit
times, which differ from theirs by minutes, while the repeat score detrended BP and RP without the clipping G
gets, so deep dips tilted their lines. Now every band comparison (this check and the colour of the change) uses
only transits that all three bands have (`detect.same_transits`), the window is widened by half the visit gap, and
BP and RP are detrended as G is. The test then ran again from the same seed (the kept fakes differ a little: each
slot keeps whichever fake finishes first). Natural fades after the fix: 47 of the strongest 2,000 (2.4%: 32
eclipsing binaries, 7 long-period variables, 4 cataclysmic variables) and 16 of 3,000 random (0.5%) marked G-only;
1 and 6 BP/RP-only.

So, as written above, `--skip-band` is part of the release-day run. The stars it skips are at these rates real
variables as well as instrument faults, so they stay on the fading list with their verdict and are read after the
rest. Data: `/mnt/project-files/slow-changes/band-test/`.

## Eclipsing binaries among the candidates (written 2026-10-30, before any fake is drawn)

**Why.** On the DR3 trial 82% of the 1,000 strongest fades at G < 19 are DPAC eclipsing binaries, and 11 of the 60
strongest that DPAC does not call one repeat every 20-42 days with dips of 16-34%, most likely unlabelled
eclipsing binaries. The project's own filter (`ebfilter`, docs/arms/eb-filter.md) asks whether one strict period
explains which epochs are in dip, and whether the dips are short enough for two stars at that period. With Gaia's
~40 DR3 epochs alone it explains almost nothing (≤ 2%, an information limit): it needs ZTF (dec > −30°). So it runs
inside the candidate checks, which fetch ZTF anyway: `slow_checks.py --lightcurves R/lightcurves` reads each
candidate's G epochs from the run's kept light curves and adds `eb_explained` (and the period, bits saved and
whether the period is unphysical for an eclipse) to checks.csv. The selected dip is the deepest G epoch, as in
the filter's own calibration, and the threshold stays at its frozen 7 bits. Explained stars leave the review for
the eclipsing-binary list.

**Test.** Fakes in quiet hosts at G < 19 from the DR3 trial's sky patches, north of −30° with at least 100 ZTF
epochs, each host used once and none used by the G-against-BP-and-RP test: 50 repeating long dips and 50 drawn
uniformly from the other five shapes (long dip, late-starting fade, steady fade, staircase fade, accelerating
fade), grey, injected into Gaia's raw epochs and the host's ZTF epochs (repeating dips keep repeating through ZTF's
years), kept if the trial thresholds flag them at S/N ≥ 20.

- **E1** (keeps real slow changes): fakes explained as eclipsing binaries, upper bound (one-sided 95%) ≤ 5%, i.e.
  at most 1 of 100.

Reported: explained share of DPAC eclipsing binaries among the strongest fades north of −30° (G < 17, 60 stars,
with ZTF), and of the 60 strongest non-eclipsing fades already checked in the trial.

**What each outcome leads to.** E1 passes: explained stars leave the review. E1 fails: the verdict is shown to the
reviewer only, and the fakes it explained are read.

**Result (2026-10-30): E1 fails.** 98 of the 100 slots filled in 30 rounds (two repeating long dips never reached
S/N ≥ 20 on a free host). 9 of 98 fakes explained, upper bound 15.5% (> 5%). All 9 are repeating long dips (9 of
48); none of the 50 other shapes. The filter found each one's injected period (within 0.3%, one at 1.14×), and the
duration check let them through: the injected dips last 4-35 days (5-23% of periods of 50-305 days), but the
filter's lower bound on duration, built for dips of hours, breaks a run of dip epochs at any gap over 1.5 days
and so measured 0.7-4 days on ZTF's cadence. Two more were flagged `periodic_unphysical` (dips too long for any
stellar pair), which is the right reading.

Real stars, reported: 44 of the 52 DPAC eclipsing binaries at G < 17 with ZTF epochs are explained (8 of 60 lie
south of −30°); the other 8 have a strict period, but the deepest Gaia epoch falls outside its eclipses. Of the 60
checked non-eclipsing fades, 23 have ZTF epochs and 3 are explained, all three without a DPAC class (among the 11
short-timescale stars that repeat every 20-42 days).

So `eb_explained` goes into `checks.csv` and the review packet as information, and no star leaves the review on
it. A strictly periodic long dip and a long-period eclipsing binary differ only in how long the dips last, which is
the kind of object this search is for. Measuring duration from the folded light curve, rather than from unbroken
runs of epochs, would tell them apart; it would need its own test. Data: `/mnt/project-files/slow-changes/eb-test/`.

**Decision (2026-11-11): the check stays out of the review files and stands beside each verdict.** The consequence
written above for a failed E1 has two parts. The first, that no star leaves the review on `eb_explained`, holds.
The second, that the verdict is shown to the reviewer, was never built into a calibration: the 432 files of the
reviewer's calibration (docs/vetting.md §14, run the same day) carry four tests and no eclipsing-binary check,
and no batch since has carried it. When release day's review files were written (`scripts/slow_review.py`, PR
#88) the check was left out unless `--with-eclipse` is given. That default is kept, and this is a departure from
the text above, for these reasons:

- A reviewer is measured on one kind of file (docs/vetting.md §7). Recall of 44 of 50 and a false shortlist of
  5.0% at the flagged mix belong to files without the check. Adding a test that says "explained: 1" makes a
  reviewer whose rates nobody has measured, on the day the rates are quoted.
- The check is wrong in the direction that matters. It explains 9 of 48 repeating long dips, the objects this
  search is for. How far the reviewer follows it is unknown, so the loss of recall on that shape could be
  anything from nothing to a fifth.
- The gain is unmeasured too. DPAC's eclipsing binaries are left out before the checks (`--skip-class ECL`), and
  of the 23 other fades with ZTF epochs the check explained 3.
- Nothing is lost by waiting. `eb_explained`, its period and whether the period is unphysical are in
  `checks.csv`, in the batch's key, and beside every verdict in `D/reviewed.csv`, where the person reading the
  list sees both. From 2026-11-11 each batch also reports how many of its planted fades the check would have
  explained (`D/fakes.json`, docs/vetting.md §16).

Showing it to the reviewer needs a calibration batch whose files carry it, scored as §14 was: about 250 reviews
(the 50 fakes of R1, the repeating long dips among them, and 200 natural stars), about 2.7 million output tokens
on the adopted reviewer. It has not been run. `--with-eclipse` remains for that batch; `files.json` then records
that the files are not of the calibrated kind, and `table` says that §14's rates do not describe their reviews.

## Other surveys (2026-11-10)

A fade in Gaia can be Gaia's own doing (an instrument or calibration effect, or a neighbour that only Gaia's
window separates), and DR4 stops on 2020-01-20. Other telescopes answer three questions that Gaia cannot answer
about itself. `cosmos_search/surveys/` reads them, `slow/independent.py` compares, and `scripts/slow_checks.py`
adds the result to every candidate's row.

| Question | Column | Reading |
|---|---|---|
| Did another telescope see the same change over Gaia's own dates? | `same_k` ± `same_k_err`, `same_years` | k is the survey's change as a multiple of what Gaia's predicts for its band: 1 the same change, 0 none. `seen` (k above 0 by 3 errors), `not_seen` (even half of Gaia's change ruled out), `opposite`, `undecided`, `no_data` |
| What happened after Gaia's last visit? | `after`, `after_ratio` or `after_recovered` | A trend: its rate since, as a multiple of Gaia's (`went_on`, `stopped`, `reversed`). An episode still under way at Gaia's last visit: the share of its depth recovered since (`fainter`, `still_faint`, `recovered`) |
| Is the star active enough for spots to do it? | `tess_amp`, `tess_quiet`, `activity_ratio` | Gaia's change as a multiple of the star's variability within a month in TESS |

Each also comes per survey (`skymapper_k`, `ztf_after`, ...), with `<survey>_dilution`: the light of Gaia
neighbours inside that survey's beam over the star's own, by which the expected change is reduced.

**How the comparison is made.** Epochs are averaged into visits, then into 180-day bins inside each group that
has its own zero point (a filter and image type, a camera, a ZTF field and CCD quadrant). The fit gives every
group its own level and one k for the survey, so nothing depends on zero points between groups, and a camera
change cannot read as a step. Gaia's change is scaled to each band from its BP and RP values (a change twice as
deep in BP as in RP is expected deeper in u and shallower in z). A bin's error is its formal error plus a floor
measured on constant stars, in quadrature; if the bins scatter about the fit by more than that, the error of k
grows by the same factor, and it is widened by Student's t for the few bins it rests on.

**Measured on constant stars.** 640 Gaia DR3 stars that DPAC does not call variable, with RUWE < 1.4 and a G
scatter below the median of their magnitude, |b| > 10°, in four G ranges from 11 to 18.5
(`configs/surveys/constants-*.csv`; `scripts/survey_calibrate.py`). The floor is the scatter of 180-day bin means
beyond their quoted errors. The test compares each constant star with a made-up Gaia trend of −2% per year over
DR4's dates, and again with that trend injected into the survey's epochs.

| Survey | Years | Sky | Stars it can test | Floor | Trend error over DR4's dates (median, % per year) | Injected trend seen | Constant star read as seen / opposite |
|---|---|---|---|---|---|---|---|
| SkyMapper DR4 | 2014-03 to 2021-09 | δ < +16° | 279 of 316 | 0.44% | 0.22 (G 11-13), 0.21 (13-15), 0.33 (15-17), 0.97 (17-18.5) | 86%, 93%, 79%, 33% | 0 / 1 of 279 |
| NOIRLab NSC DR2 | 2012-09 to 2019-11 | mostly δ < +30°, G > 14.5 | 62 of 167 | 0.57% | 0.35 (G 15-17), 1.9 (17-18.5) | 64%, 42% | 0 / 0 of 62 |
| ZTF (IRSA) | 2018-03 on | δ > −30°, G > 13 | 224 of 232 | 0.30% | 0.33 (G 13-15), 0.29 (15-17), 0.63 (17-18.5) | 82%, 80%, 59% | 2 / 3 of 224 |
| ATLAS | 2015-07 on | all (south of −50° from 2022) | not measured | provisional 1.5% | | | |
| ASAS-SN | 2012 on | all, G 10.5-16.5 | not measured | provisional 1.5% | | | |

The ratio of k to its error has a scatter of 0.9 on SkyMapper's constants, so its errors are about right.
NOIRLab's are too large by about half (scatter 0.5): with a median of five bins per star the widening for few
bins dominates, which errs on the safe side. So SkyMapper confirms a trend of 0.7% per year at G < 15 and 1% per
year at G 15-17 over DR4's dates, for nearly every star south of +16°. NOIRLab has enough exposures for one faint
star in three. Of 341 constant stars tested in the two surveys, none read as `seen` and one as `opposite`.

ZTF overlaps only DR4's last 1.9 years, so over Gaia's dates it asks whether a trend was still running then, to
1% per year at G 13-17. Its errors are right (scatter 1.0), yet five constant stars of G 15-17 read as `seen` or
`opposite`. In each the change is 1-2% over the two years and shows in two or three filters and in overlapping
fields, so the stars themselves changed: "constant" here means quiet in Gaia DR3, which ends ten months before
ZTF starts. About two quiet stars in a hundred drift by that much, and a `seen` for a fade of 1% per year carries
that false-match rate. Brighter than G 13 ZTF saturates (floor 0.45%, and 4 of 58 constant stars read
`opposite`), so it is not read for those stars.

Every trend error includes 0.1% per year for the survey's own drift (`TREND_FLOOR`). ZTF showed why: with six
years of data after 2020 its bin errors predict a trend error of 0.06% per year, while the trends of 230 constant
stars scatter by 0.12% per year around a median fade of 0.03% per year. Without the floor 11 of the 230 read as
`reversed`; with it 2 do, and 1 reads as `went_on` (it faded by 0.6% per year for six years). The verdicts also
ask first what a survey rules out: a change in Gaia's direction that is measured well but is under half of
Gaia's is `not_seen`, however far it stands above zero.

After DR4's last visit SkyMapper has at most 1.6 years, so its trend error there is 1.2 to 7% per year and it can
only say `went_on` for fast fades. ZTF carries the years since for the northern sky: its trend error after 2020
is 0.11 to 0.13% per year at every G from 13 to 18.5 (the drift floor dominates), so it tells whether a fade of
0.4% per year or more went on. With the made-up trend continued into ZTF's epochs, 97% of 230 stars read
`went_on`; left constant, 91% read `stopped`. South of −30° the years since rest on ATLAS, and for stars brighter
than G 13 on ASAS-SN.

SkyMapper needed one change from the first version: with one group per filter, constant stars of G 17-18.5 read
as fading (median k / error +0.5, 4 of 62 `seen`), because the short early exposures catch a faint star only when
noise lifts it. With one group per filter and image type the bias is gone (+0.1, 0 of 61).

**A dimming still under way at Gaia's last visit** was tested the same way (`survey_calibrate.py null --kind
episode`): a made-up dimming of 5% over the two years before the last visit, and afterwards the star stays as
faint, recovers, or dims by as much again.

| Survey, Gaia's last visit | Over Gaia's dates: depth error (median), made-up dimming seen, constant stars `seen` / `opposite` | Afterwards: error, read `recovered` when it recovers, `fainter` when it doubles, constant stars `fainter` / `recovered` |
|---|---|---|
| SkyMapper, DR4 | 0.9% of the flux, 70%, 2 / 2 of 243 | 1.0%, 40%, 64%, 0 / 0 of 249 |
| SkyMapper, DR3 (2017-05) | 1.8%, 53%, 1 / 0 of 151 | 1.0%, 38%, 69%, 1 / 0 of 229 |
| ZTF, DR4 (dimming from 2019-01) | 0.47%, 96%, 0 / 0 of 222 | 0.33%, 90%, 99%, 2 / 0 of 227 |
| NOIRLab, DR4 | 1.1%, 65%, 0 / 0 of 23 | no epochs after 2019-11 |

This test found two faults, both fixed. Survey bins from the 30 days after Gaia's last visit were counted as
"back to normal" over Gaia's dates and as "still faint" afterwards, which pulled ZTF's depth down to 0.9 of the
true one and a full recovery down to 0.9; those bins now count on neither side. And ZTF's level after 2020
against its level in 2019 scatters by 0.34% on constant stars where its bin errors predict 0.27%, so 7 of 227
constant stars read `fainter`; every depth and recovery now carries 0.2% of the flux for that (`LEVEL_FLOOR`,
the counterpart of the trend floor), which leaves 2.

**On stars with a known Gaia change.** The DR3 watch-list (`configs/surveys/known-dr3.csv`: 2,849 stars with a
fade in DR3, four in five of them DPAC-classed spotted stars) over DR3's dates, 2014-07 to 2017-05:
SkyMapper has epochs for 1,746 of the 1,769 south of +16°, but its first years are thin, and only 471 have enough
bins on both sides of Gaia's change to be compared. Of those 115 read `seen`, 355 `undecided`, none `not_seen` and
one `opposite` (a star with no DPAC class and a Gaia fade of 1.4%, where SkyMapper shows a brightening of 12 ± 3%;
not followed up). For the 113 whose SkyMapper error is under a third of Gaia's change, 82 are `seen` and 31
`undecided`, with a median k of 0.89 (10th to 90th percentile 0.5 to 1.4). A k just under 1 is expected, since a
list picked for large Gaia changes overstates them a little. Where SkyMapper can judge, Gaia's fades are real and
of the size Gaia gave.

After DR3's last visit, to 2021-09: of 195 stars whose dimming was still under way and which SkyMapper could
judge, 14 had recovered, 104 were about as faint and 77 fainter still, by a median of the depth again. The
comparison is with the mean level inside Gaia's faint window, so a fade still deepening at Gaia's last visit reads
`fainter` even if it stopped there. Of 49 stars with a trend, 9 went on at more than half Gaia's rate, 22 did not
and 18 reversed. For a spotted star, then, staying faint for years after Gaia is the usual case: `still_faint` or
`fainter` does not set a candidate apart, and `recovered` is the rarer outcome (7%). ZTF cannot be tested on
these stars. It starts ten months after DR3 ends, so it has no level from inside Gaia's window to compare with
(a sample of 193 gave `no_data` for every episode); for DR4 it covers the last 1.9 years.

**TESS variability as a measure of activity.** `surveys/tess_activity.py` measures the 5th to 95th percentile
range of each sector's light curve in 3-hour bins, noise removed, from TGLC's undetrended fluxes (sectors 1-55,
Tmag < 16), after dropping the first day of each orbit and dividing each orbit by a fitted line. Without the line
constant stars show a median of 1% at G 11-13 and 3% at G 14.5-16, nearly all of it drift over an orbit. With it:

| G | Constant stars with a sector | Median | 90% below (`quiet_floor`) |
|---|---|---|---|
| 11-13 | 96 | 0.34% | 1.6% |
| 13-14.5 | 58 | 0.54% | 2.9% |
| 14.5-16.5 | 107 | 1.5% | 9.5% |

63% of the constant stars asked for had a TGLC light curve. The line costs any rotation slower than an orbit
(about 13 days).

This check turned out weaker than hoped. Of 975 DPAC-classed RS CVn stars on the watch-list with a TGLC sector,
81% show no more variability within a month than constant stars do (68% at G 13-14.5), although Gaia saw them
change by a median of 7%. Their spots change the mean level over years far more than they modulate it over days,
or they rotate slower than an orbit. So a quiet TESS light curve does not rule activity out. What the check can
say: an amplitude above the floor is direct evidence of spots (1 RS CVn star in 5 shows it, 1 in 3 at G 13-14.5,
against 1 constant star in 10 by construction), and Gaia's change over the TESS amplitude (`activity_ratio`, with the floor standing in for
a quiet star) has a known range for spotted stars: median 0.9, 90% below 2.9, 99% below 7.0. A candidate far
above that range is unlike nearly all known spotted stars; with floors this high that takes a change of more than
about 15% at G < 13 and 30% at G 13-14.5. Gaia's own rotation-modulation and Ca II activity tables are likely a
better measure of activity and are not read yet.

**Not yet tested against the live service.**

- ATLAS needs an account's token (`ATLAS_TOKEN`). The reader runs a job on reduced images (difference images
  step when ATLAS replaced its reference images in 2018-10 and 2020-02, inside Gaia's last years), and its
  protocol is tested against a local server. The first session started after the token was added runs
  `survey_calibrate.py fetch --surveys atlas`, `floors` and `null` on the constant stars and replaces the
  provisional floor.
- ASAS-SN answers only on port 9006, which cloud containers cannot open. On 2026-11-10 the first try from kore
  ended within two minutes with the whole hawaii.edu network dropping kore's address, most likely because the
  reachability check opened two empty connections that looked like a port scan. The check is now one ordinary
  request, retries are limited to two, and ASAS-SN is read only when named (`--surveys ...,asassn`). Its floor is
  provisional and its reader has not seen live data.

**In a blind test.** Fakes are injected into Gaia's epochs only, so every fake would read `not_seen` here and
give itself away. Until the fakes are also injected into these epochs (`dayone.external.apply` takes this table:
same columns and band names), the survey columns must stay out of what a reviewer of fakes sees.

**Running it.** `scripts/slow_checks.py` reads ZTF, ATLAS, SkyMapper and NOIRLab by default and names once any
survey it cannot reach; `--surveys none` turns the comparison off and `--no-tess` the TESS step. SkyMapper takes
about 3 s a star, NOIRLab 15-25 s, TESS a few seconds. Gaia neighbours for the dilution come from ESA's archive,
or from the AIP mirror when ESA rejects the query. Epochs go to `surveys.parquet` and the TESS rows per sector to
`tess.parquet`.

## Log

- **2026-11-11.** Blind review on release day (Release day, end to end, step 10; docs/vetting.md §16). Every
  review batch now holds planted fades of known answer, built into review files by the function that builds a
  star's and written with sealed answers; the batch's result says whether its reviewer still finds what the
  calibration's reviewer found. The eclipsing-binary check stays out of the review files and beside each
  verdict, a recorded departure from E1's written consequence (Eclipsing binaries among the candidates), because
  no calibration has measured the reviewer on files that carry it. Review files hold nothing from the other
  surveys.
- **2026-11-11.** Other surveys as checks on a fade (Other surveys, above; PR #87). Readers for SkyMapper,
  NOIRLab, ATLAS, ASAS-SN and TESS, and a comparison in every candidate's row: whether another telescope saw
  Gaia's change over Gaia's own dates, what happened after Gaia's last visit, and how variable TESS shows the star
  to be within a month. Measured on 640 constant stars. SkyMapper confirms a trend of 0.7% per year at G < 15 for
  nearly every star south of +16°. ZTF (north of −30°, G > 13) tells whether a fade of 0.4% per year went on
  after 2020 and sees 96% of 5% dimmings in DR4's last year. One or two constant stars in 100 read `seen` or
  `opposite`; in ZTF these are stars that did change by 1-2%. On DR3's known faders SkyMapper confirmed 82 of the
  113 it could judge and contradicted none, and of those still faint at DR3's last visit 7% had recovered by
  2021. TESS is a weak test of activity: four in five known spotted faders look as quiet as constant stars.
  ATLAS and ASAS-SN are built and tested offline only; ATLAS waits for a session that has the token, ASAS-SN
  for the hawaii.edu block of 2026-11-10 to lift.

- **2026-11-11.** The known answers are in the DataLink fallback list (3,791 stars with the watch-list). ESA's
  archive refused the position queries, so `slow known` gained `--mirror aip`.

- **2026-11-11.** The occurrence limit reads the pass's per-star table and the flag step's reference, and the
  DR3 field was run again that way (docs/arms/occurrence.md): fewer than 1 in 101 FGK dwarfs fade at 0.2% a
  year, 1 in 7,879 at 1%, 1 in 58,623 at 5%. Found on the way: the trend left after the drift correction depends
  on colour. Stars redder than BP−RP 2 brighten by 0.06 to 0.22% a year and the bluest by about 0.1%, at every
  magnitude, while stars between 0.5 and 1.8 sit within 0.04% of zero (table in docs/arms/occurrence.md, "The
  leftover trend depends on colour"). The drift table has one colour range above BP−RP 1.3 and the neighbour
  reference none, so neither removes it. At DR3's errors that is +0.3 to +0.5 errors for the reddest stars;
  at DR4's it would be about +1. The occurrence limit takes the dwarfs' own median out. The lists do not yet.
- **2026-11-10.** Upper limit on how common steady fades are among FGK dwarfs (docs/arms/occurrence.md), built
  and run on the 195,572 FGK dwarfs of DR3's Andromeda field: fewer than 1 in 110 fade at 0.2% a year, 1 in
  7,545 at 1%, 1 in 51,304 at 5% (95%). Recall comes from steady fades injected into every one of those stars.
  Every fading star past the cut counts as a possible real fade; subtracting chance fades estimated from
  brightening stars was dropped as the limit, because on DR3 the two are not equally common far out (30
  brightening against 10 fading beyond 8 errors) and a subtracted limit then falls below the truth. Settings for
  DR4 frozen in `prereg/occurrence/steady-fades-v1.json`.
- **2026-11-10.** Every star scored ("Every star scored" above). The archive-side selection picked none of the
  injected steady fades up to 3% a year; the run now summarises every star of the bulk files in one compiled pass
  (0.2 ms per star per core) and flags the same stars as the detector on every star. Trends are measured against
  neighbours of similar brightness (the +0.94 error offset at G 12-13 is gone). DR4's bulk files carry no rejected-
  epoch marks, so the scoring has its own test, which restores the thresholds and recall the marks gave; lists are
  30% longer. Injected fades of 1% a year: 82% listed over all hosts brighter than G 19, 86% on quiet ones. The
  cuts step ranks within kind of star, and a DataLink fallback list with a prediction per watch-list star exists.

- **2026-10-30.** Last checks before DR4. G against BP and RP: passes its pre-written test twice (0 of 300 real
  changes cut; 54, then 51, of 100 G-only fakes caught) and is a cut on release day. DR3's own fades showed that
  DPAC drops 2-5% of epochs in one band only, which made one-transit eclipses look like G-only changes; every band
  comparison now uses transits all three bands have (18% → 2.4% of the strongest natural fades marked). The
  eclipsing-binary filter runs inside the checks on G and ZTF (44 of 52 DPAC eclipsing binaries explained) but
  fails its test on repeating long dips (9 of 98 fakes explained), so it informs the review and cuts nothing.
  NEOWISE's change and trend errors were too small with DR3's few visits; widened (S5).

- **2026-10-30.** Whole search tried on all 11.6 M DR3 stars with light curves (Release day, end to end; PRs #78,
  #79): every step ran without a failed file, slice or fetch; scoring 14.7 ms per star per core. Fixed a period
  search bug (one extreme bright visit faked repeat scores in the hundreds), made the shared-date test and list step
  fast and the list step's memory bounded, and ran the checks several stars at a time. New step: stars without
  measurable parallax or proper motion (`still`, nearly all quasars) set aside, 93% of the faint fading list. DR4's
  faint thresholds will likely sit near 5.6 (M31 field) instead of DR3's 17. Among ordinary stars the rising list
  is as long as the fading list at every strength, so DR4's lists are read from the top.
- **2026-10-30.** Checks and blind review calibrated on 432 DR3 objects (S5, docs/vetting.md §14): recall of
  anomalous fakes 44/50 (pass), false shortlist 26/322 pooled (upper bound 11.0%, fail; 5.0% at the flagged mix),
  colour and NEOWISE errors hold. Three cataclysmic variables carry the repeat-flag artefact fixed in PR #78.
- **2026-10-23.** Fades in time order (slow-v3, above): late-starting brightenings no longer pass as fades (53% →
  2-2.5% at signal-to-noise ≥ 20), late-starting fades found 90-92% instead of 84%, on shard 0 and on fresh shard 3.
  Dips already under way at the first visit move to a second list. A crowd of stars stepping on the same scan in
  crowded fields, hidden before, is now set aside by a shared-date test. A sky offset per patch and visit is real
  (0.13% in G) but changes results by under a point. Release-day threshold code ready and checked on DR3 (above).
- **2026-10-23.** Ten ways to improve the slow period search (above): none worth a third blind test. BP and RP
  agreement adds 3-10 points of recall but 50-80% more flagged stars; dips that repeat every few visits inflate
  the noise estimate, and no estimate tried avoids that without raising the threshold as much. slow-v2 stays
  frozen for DR4 (Cora's choice).
- **2026-10-23.** Slow period search blind test (above): false alarms 0.10% (pass); calibrated shapes 65% and
  19% (pass); new repeating dip shapes 40-49% against a bar of 50% (fail). The period search still doubles or
  triples what the detector finds on every repeating shape.
- **2026-10-23.** Slow period search (above): repeating long dips at signal-to-noise 20-40 found 57% of
  the time instead of 14%; slow cycles stay weak (17%). Its blind test written down before the draw.
- **2026-10-23.** Blind test (S6, above). False alarms 0.16% (pass), calibrated shapes 85-100% (pass), new shapes
  67% pooled (fail): staircase and accelerating fades over 90%, repeated slow cycles 17%. How to cover
  repeated cycles is open (above).
- **2026-10-23.** Blind test written down (S6, above) before drawing the batch: four new fake shapes, fresh
  stars from shard 1/10, frozen thresholds and drift table, pass criteria P1-P3.
- **2026-10-23.** Checks per candidate (S5, above): colour, NEOWISE change or slope against what grey, dust and
  re-emission predict, ZTF after 2020. On DR3's 40 strongest candidates all are known variable classes; spotted
  stars' fades are neither grey nor dusty in W1, as a change in the star predicts.
- **2026-10-23.** Whole-sky selection (S4, above): selecting 0.1% of stars by √n × uwv × (1 − Abbé) per G bin
  keeps 69-95% of what the detector finds at G < 19; 1% keeps 82-98%. DR3's DataLink rate points to ~0.1%.
- **2026-10-23.** Detector first version (S3, above): fades found near the ideal, long dips need about twice the
  signal-to-noise; faint stars (G ≥ 19, mostly quasars) need their own thresholds; 80% of smooth fading alerts
  pass at DR4's sampling. Trend error made robust to rare large jumps the same day (S5 check sheet).
- **2026-10-23.** Made the main search after a step-back review (Cora's decision, 2026-10-23). DR3 numbers behind
  it: trend error 0.03% per year at G 13-14 and 0.05 at G 15-16 for the quietest DR3 stars over 2.6 years, with
  star-to-star spread only 1.2-1.3× the error; median trend of quiet stars +0.02 to +0.1% per year at G < 17
  (shared calibration drift); fades and brightenings beyond 0.5% per year at > 10σ equally common at every G.
