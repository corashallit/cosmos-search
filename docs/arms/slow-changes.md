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
| S3 | Trend, curvature and long-episode detector; fake slow fades and long dips for recall; brightenings as the false-alarm estimate | S2 | |
| S4 | Whole-sky selection on TAP (Abbe values; fallback DR4 minus DR3 mean against neighbours), recall measured on DR3 | S1, S3 | |
| S5 | Checks per candidate: NEOWISE same years, colour of the change, position (`astrometry.dips`), ZTF and ASAS-SN after 2020, known-variable catalogues | S3 | |
| S6 | Pre-registered blind test: new fakes; controls from active stars, giants, young stars, known eclipsing binaries | S4, S5 | |
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

## Log

- **2026-10-23.** Made the main search after a step-back review (Cora's decision, 2026-10-23). DR3 numbers behind
  it: trend error 0.03% per year at G 13-14 and 0.05 at G 15-16 for the quietest DR3 stars over 2.6 years, with
  star-to-star spread only 1.2-1.3× the error; median trend of quiet stars +0.02 to +0.1% per year at G < 17
  (shared calibration drift); fades and brightenings beyond 0.5% per year at > 10σ equally common at every G.
