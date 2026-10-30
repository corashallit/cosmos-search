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
| S7 | Freeze before 2026-12-02, run on DR4 (short-dip and astrometry searches run as built alongside) | S6, DR4 | slow-v3 (fades read in time order, below); release-day thresholds from DR4's own stars ready (below) |
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

## Log

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
