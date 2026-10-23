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
| S2 | Drift correction from neighbours (`drift`), tested on stars it was not fitted on | — | Magnitude/colour drift done 2026-10-23; sky term open (below) |
| S3 | Trend, curvature and long-episode detector; fake slow fades and long dips for recall; brightenings as the false-alarm estimate | S2 | First version done 2026-10-23 (below) |
| S4 | Whole-sky selection on TAP (Abbe values; fallback DR4 minus DR3 mean against neighbours), recall measured on DR3 | S1, S3 | Done 2026-10-23 (below); DataLink download rate open |
| S5 | Checks per candidate: NEOWISE same years, colour of the change, position (`astrometry.dips`), ZTF and ASAS-SN after 2020, known-variable catalogues | S3 | First version done 2026-10-23 (below); position needs DR4 epoch astrometry |
| S6 | Pre-registered blind test of the detector: new fake shapes, fresh stars; known variable classes measured openly | S4, S5 | Written down 2026-10-23 (below); batch not yet drawn |
| S7 | Freeze before 2026-12-02, run on DR4 (short-dip and astrometry searches run as built alongside) | S6, DR4 | |
| S8 | Verify top candidates: one new brightness measurement each; per-visit spectra for the best | S7 | |

## Known answers (S1)

`python -m cosmos_search.slow known --out known.parquet` (`slow/known.py`). On 2026-10-23: 985 stars, 967 matched
to a Gaia DR3 source within 2″ (median separation 0.07″), 573 of them with DR3 light curves.

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

ZTF says whether a change went on, stopped or recovered after DR4's span. NEOWISE comes from IRSA's
`neowiser_p1bs_psd` single exposures (clean frames, one component, no contamination flags), averaged per ~6-month
visit. Position during the episode (`astrometry.dips`) needs DR4's epoch astrometry and is not in yet.

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

**Known variable classes** (active and spotted stars, giants and long-period variables, young stars, eclipsing
binaries) are measured openly on shard 1, after the reveal, from DPAC's classes. They need no blinding: the
question is how many such stars pass, which sets the workload for the checks per candidate (S5).

**Procedure** (as for the astrometry blind test): `make` draws the batch and seals the answers; `run` writes
`flags.csv`; `commit` archives the flags in `prereg/verdicts/slow-blind-1/` and refuses a dirty tree; the archive
commit is pushed; `reveal` refuses unless that commit is on origin and the hashes match.

## Log

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
