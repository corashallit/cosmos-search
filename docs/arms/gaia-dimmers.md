# Aperiodic dimmers in Gaia epoch photometry

Main arm. Goal for DR4 day one: a ranked, vetted list of stars whose brightness drops in ways that neither
the instrument nor known astrophysics explains, ordered by how cheaply each can be checked.

## What the data can see

Gaia measures each source in **field-of-view transits**: typically 50–150 over DR4's 66 months, at times set
by the scanning law (pairs 106.5 min apart from the two fields of view, then groups weeks apart). Within one
transit:

- G is measured on 9 CCDs in about 40 s (DR4 releases per-CCD G as `epoch_photometry_ccd`);
- BP and RP are measured within about a minute, by a separate instrument;
- the scan angle, field of view and position in the focal plane are all known.

That sampling cannot resolve the shape of a dip lasting hours to days, but it does constrain three things
well: **whether dimming episodes exist, their colour, and slow trends over 5.5 years**. Dip shape and timing
belong to the TESS arm.

## Signals, from first principles

Rather than searching for "Boyajian-like" light curves, start from what an occulter or an engineered change
to a star's output must do, and from what Gaia can test.

| Signal | Why it matters | Gaia test |
|---|---|---|
| **Dimming episodes** | Something passes in front of, or modulates, the star | Low flux in ≥ 1 transit, reproduced on all 9 CCDs, in both fields of view where both are sampled, and in BP and RP |
| **Dip colour** | Small dust grains redden (extinction rising toward the blue); bodies much larger than the wavelength are grey; some physics makes dips bluer | R = (ΔF/F)_RP / (ΔF/F)_BP. R ≈ 1 is grey. Plot against depth; natural dust dippers sit at R < 1 |
| **Secular fading or brightening** | Boyajian's star faded ~0.34 %/yr over Kepler's first ~3 years, then ~2.5 % in 200 days (Montet & Simon 2016); quiet main-sequence stars change far less | Trend over 66 months relative to stars with the same magnitude, colour and sky region (which removes calibration drift); extend with DR3, Hipparcos/Tycho for bright stars |
| **Energy budget** | Absorbed starlight must come out somewhere. Covering fraction f ≈ depth × duty cycle implies an infrared excess L_IR ≈ f L_* at T ≈ 280 K × (L_*/L_☉)^¼ (r/1 au)^-½ | Predict the W3/W4 excess from the measured dimming and compare with AllWISE/unWISE. A dimmer with *no* excess where one is predicted is more interesting than one with it: the material is cold, far out, reflective or not absorbing |
| **Asymmetry** | Instrumental noise is roughly symmetric about the median, real occultation is one-sided | Compare the faint tail with the bright tail for the same star and for its strata (statistics.md §2) |

The usual "no infrared excess" cut removes young-star dippers, but used as a binary filter it throws away
information. The energy-budget comparison is quantitative instead, and it covers both cases: Project
Hephaistos looks for infrared excess *without* dimming, while we look for dimming without the excess it
predicts.

## Natural explanations to beat

Eclipsing binaries with long periods (including disk-occulted systems in the ε Aur family, which sparse
sampling over 5.5 years is well suited to finding), young-star dippers and UX Ori stars (with IR excess),
R CrB stars (deep, reddened, carbon-rich), Be stars, rotational modulation from large spots, planetary
transits that land in a transit by chance, and occultations by solar-system bodies (single-transit dips;
check against Gaia's SSO ephemerides).

Instrumental explanations, which will dominate: crowding and blends whose flux depends on scan angle,
saturation and gating for G ≲ 13, cosmic rays and CCD defects (caught by the per-CCD test), field-of-view
calibration offsets (caught by the FoV test), and photometric outliers DPAC already flags (respect
`rejected_by_photometry` / `rejected_by_variability`, but test them, don't trust them).

## Day-one strategy for DR4

DR4 epoch photometry is 121.8 × 10⁹ rows (7.2 TB). Two routes reach it, at very different rates (measured on
DR3 on 2026-10-21; docs/infra/targets.md):

- **DataLink** caps at 50 source IDs per call and costs ~11 s per call plus ~1.1 s per source. Calls in
  parallel gave no speed-up, so it delivers ~1.5 sources/s: 10⁴ sources in ~2 h, 10⁵ in ~18 h.
- **Bulk CDN files** stream and parse at ~29 MB/s on one 4-core container, CPU-bound. If DR4's epoch
  photometry is on the CDN at release, as DR3's was, a full pass takes ~3 days on one container and
  proportionally less when the file list is split across several sessions.

The bulk stream changes the design. It reaches every source, including the transits DPAC rejected, which the
`vari_summary` prefilter cannot see. On DR3 so far, 7.9% of sources have a rejected point ≥ 3σ faint in an
otherwise quiet light curve, and the prefilter recalls 0.3% of them. With the full stream, the prefilter is
no longer a selection: it only decides what we look at first. So:

1. **TAP-side prefilter** on DR4 `vari_summary` (~4 × 10⁸ rows), using statistics that respond to a
   one-sided faint tail: skewness, the minimum relative to the median in units of the per-source noise, and
   the number of low points. It is prepared on DR3 and adapted to the draft data model (G7).
2. **DataLink on day one** for the top ~10⁴ by prefilter score, with per-CCD G. This gives the first ranked
   candidates within hours of release.
3. **Full bulk stream from day one**, with detectors run on each file as it lands. Only compact outputs are
   kept (storage.md):
   - a feature row for every source above a loose cut, plus full light curves for a random sample drawn with
     known inclusion probabilities (`stats.audit`), which give injection hosts and recall measurements;
   - per-stratum **tail sketches** for the null (`stats.TailSketch`). Each stratum keeps its largest
     dimming and brightening statistics exactly (up to a fixed count, by default 10⁵) and a fine histogram
     of the rest. The mirror null then stays exact where thresholds are set (statistics.md §2), sketches
     from different files and sessions merge exactly, and nothing near 1.8 × 10⁹ rows is kept.
   - The stream must resume after a container is reclaimed, so it records which files it has processed.
4. **Fallback if DR4 has no bulk files at release.** The random full-light-curve channel then goes through
   DataLink at ~10⁵ sources per ~18 h, which is still large enough to measure the rejected-transit channel
   and the prefilter's recall. DataLink limits for DR4 may differ from DR3's, so re-measure them on the day.
5. Rank across all channels, calibrate against nulls, and send the top ~10³ to vetting.

The DR3 rehearsal (G1–G6) runs the same chain on DR3's 11.75 M variable sources. DR3 has only per-transit G,
so the per-CCD test is DR4-only.

## Ranking

Score = calibrated significance after invariance tests × physics flags (grey dips, energy-budget tension,
secular trend) × verifiability (G ≲ 13 for amateur photometry, recent activity, TESS coverage, sky
position) × not already explained (SIMBAD/VSX types, literature). The weights are not fixed by hand. They
start flat, then are fitted to injection recovery and vetting outcomes, with a random-audit stream below the
threshold so we can estimate what the ranking misses (statistics.md §5).

## Log

Record changes of approach here, with the evidence that prompted them.

- **2026-10-21.** The random-controls pull became a search channel too. The archive thread found that the
  `vari_summary` prefilter is blind to dips in DPAC-rejected transits (7.9% of DR3 sources processed so far
  have a quiet light curve with a ≥ 3σ faint rejected point; the prefilter recalls 0.3% of them). Sampling
  is with known inclusion probabilities so the channel's yield extrapolates to the whole sky.
- **2026-10-21.** DataLink caps at 50 IDs per call (~1.5 sources/s), while bulk CDN streaming runs at ~29 MB/s.
  Day one becomes a DataLink pull of the prefilter's top ~10⁴ plus a full bulk stream, so the prefilter only
  sets the order of work. The random DataLink channel stays as the fallback if DR4 has no bulk files.
