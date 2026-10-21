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
| **Dimming episodes** | Something passes in front of, or modulates, the star | Low flux in ≥ 1 transit, reproduced on all 9 CCDs, in both fields of view where both are sampled, and in BP and RP. G spans both passbands, so a change in the source's light puts the G change between the BP and RP changes; an epoch outside that interval is an instrument fault and is dropped before any statistic, on the bright side too (`vetting.stage0`, docs/vetting.md §12) |
| **Dip colour** | Small dust grains redden (extinction rising toward the blue); bodies much larger than the wavelength are grey; some physics makes dips bluer | R = (ΔF/F)_RP / (ΔF/F)_BP. R ≈ 1 is grey. Plot against depth; natural dust dippers sit at R < 1 |
| **Secular fading or brightening** | Boyajian's star faded ~0.34 %/yr over Kepler's first ~3 years, then ~2.5 % in 200 days (Montet & Simon 2016); quiet main-sequence stars change far less | Trend over 66 months relative to stars with the same magnitude, colour and sky region (which removes calibration drift); extend with DR3, Hipparcos/Tycho for bright stars |
| **Energy budget** | Absorbed starlight must come out somewhere. Covering fraction f ≈ depth × duty cycle implies an infrared excess L_IR ≈ f L_* at T ≈ 280 K × (L_*/L_☉)^¼ (r/1 au)^-½ | Predict the W3/W4 excess from the measured dimming and compare with AllWISE/unWISE. A dimmer with *no* excess where one is predicted is more interesting than one with it: the material is cold, far out, reflective or not absorbing |
| **Per-event physics** | Sparse epochs rarely reveal timing structure: on 12 DR3 hosts, 1 in 4,800 aperiodic dip trains was recognisable as aperiodic (docs/vetting.md §10). What one event can carry is a contradiction with every orbit around this star | A dip seen in both fields of view lasts ≥ 106.5 min, and one seen on consecutive scans lasts ≥ ~6 h. For an occulter on a bound orbit, a duration D across a star of radius R* and mass M* needs speed ≲ 2R*/D (for occulters smaller than the star; an extended cloud loosens this), hence a ≳ G M* D² / (4 R*²) and P ≳ 2π a / v. That sets a minimum recurrence, to compare with the dips the epochs show. Depth gives the occulter's projected size, ≥ √depth × R*, against the companion that size implies and its light. Colour (grey vs reddened) separates bodies from dust. R* and M* come from parallax, G and BP−RP |
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
   - Statistics are computed after the stage-0 band-consistency mask (`vetting.stage0.clean_long`). It drops
     G epochs whose change BP and RP cannot account for, on both sides, so the mirror stays symmetric.
     Injections pass through the same mask, so recovery is measured after cleaning. Dropped epochs are counted
     per source and per verdict as features, because their rate is an instrument diagnostic in its own right.
   - The stream must resume after a container is reclaimed, so it records which files it has processed.
4. **Fallback if DR4 has no bulk files at release.** The random full-light-curve channel then goes through
   DataLink at ~10⁵ sources per ~18 h, which is still large enough to measure the rejected-transit channel
   and the prefilter's recall. DataLink limits for DR4 may differ from DR3's, so re-measure them on the day.
5. Rank across all channels, calibrate against nulls, and send the top ~10³ to vetting.

The DR3 rehearsal (G1–G6) runs the same chain on DR3's 11.75 M variable sources. DR3 has only per-transit G,
so the per-CCD test is DR4-only.

## Ranking

Score = calibrated significance after invariance tests × physics flags (grey dips, energy-budget tension,
secular trend, per-event orbit tension) × verifiability (G ≲ 13 for amateur photometry, recent activity, TESS coverage, sky
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
- **2026-10-21.** DR3 pass complete (11,754,237 sources). The `vari_summary` prefilter, which selects 1.1% of
  sources, recalls 37% of deep dips confirmed in both fields of view (70% at G < 13), 21% of dips with no
  partner transit, and 0.3% of dips only in DPAC-rejected transits. Controls pass at 1.2%. So even on the
  dips it was built for, a prefilter-only search would miss most of them; the full stream is the search and
  the prefilter only orders it. Rejected-faint-point rates rise with the number of transits (4.0% at
  n_g < 20 to 8.5% at 80+) and double at G < 13, so a fallback random pull is stratified by G × n_g. The
  rate is also high at Galactic |b| 10–20° (8.9%, against 4.8–6.1% elsewhere), within every n_g bin and
  more so in the south. Until a sky map explains it, treat rejected-dip candidates from that band with
  suspicion.
- **2026-10-21.** Stage 0 band consistency joins the detector chain (vetting PR #25). In pilot 3, about half the
  detector's non-EB triggers were G-only dips with BP and RP flat. DPAC had not flagged those epochs. The mask
  catches 71–77% of single-transit G-only dips at ≥ 10σ and passes grey, dust and flip changes (≤ 0.3% flagged).
  The DR4 stream features (schema `dr4-features-v1`, archive thread) apply it before the robust statistics.
  They also keep the raw extremes and the per-source fault counts, so the effect of cleaning can be audited.
  G3 on DR3 runs without the mask. Its `fov` statistic needs the partner transit in the other field of view,
  so a fault in a single transit cannot pass it on its own.
- **2026-10-21.** Stage 0 removes real eclipses in crowded fields. In a DR3 LMC chunk, two of the dips it
  removed at ≥ 8σ belong to DPAC eclipsing binaries (P = 14.3 d and 60.8 d), and both have BP and RP blended
  in every transit. Blending dilutes the dip in BP and RP, so a real G dip falls outside the mixing interval and
  reads as a G fault. Error inflation at the deepest point removed the other one. The injection test missed
  this because it injects undiluted changes. Fixed in vetting PR #30 (docs/vetting.md §12). The allowed G
  change now accounts for a measured per-source dilution. An inflated-error epoch is dropped only when neither
  BP, RP nor the partner transit corroborates it. Diluted injections are part of the test, and none of 6,800 is
  vetoed. The three LMC sources keep their deep points. The cost is G-only fault recall: 41–45% at ≥ 10σ
  (59–66% on unblended stars), and blended stars are effectively untested. We accept that, because a mask that
  removes real dips does more harm than faults that reach vetting. G3 still reports both the raw and the
  cleaned extremes (`single_raw` vs `single`).
- **2026-10-21.** First G3 run on all 11.75 M cleaned DR3 sources (archive thread) selected nothing: 0 at q ≤ 0.1
  for `single`, `fov` and `single_raw`. Symmetry by stratum explains it. At 30 robust σ, G 8–12 has 20–43 times
  more dim than bright exceedances (G = 10, n ≥ 20: 151 vs 3). G 19–20 has only bright ones (G = 20, n ≥ 20:
  0 vs 689), because a faint star's dip is capped by its S/N while spikes are not. Pooled over all sources, at
  |W| ≥ 200, there were 369 dim winners and 875 bright, down from 1,386 bright before Stage 0. Competition now
  runs per G bin (statistics.md §2), and the pooled q is kept for comparison.
- **2026-10-21.** G3 with per-G-bin competition on all 11.75 M cleaned DR3 sources. Selections at q ≤ 0.1:
  16,288 for `single` (100 at q ≤ 0.01), 6,624 for `fov`, and 14,696 for `single_raw`, so cleaning adds 11%.
  Pooled competition gives 0 for all three. The `fov` selections lie entirely at G 8–15. The deepest are
  eclipsing-binary-like: depths 25–65%, R 0.78–1.05, 1–5 episodes, RUWE ~1. That is expected, because DR3 epoch
  photometry is DPAC's variables. Known-class removal and the per-event physics term (G2, G6) come next.
  Two consequences for DR4 follow:
  - **No power at G ≥ 16.** Those bins have 220–300 k dim winners each but select nothing in `fov`, because bright
    spikes keep pace with dips at every threshold. Most DR4 sources are this faint. Per-CCD G in DR4 is the
    fix to plan for: a cosmic ray or a neighbour's crossing hits one or two of the 9 CCDs of a transit, while
    an occultation lasting more than ~45 s dims all 9. Requiring all-CCD consistency on both sides of the
    mirror should remove most of the bright tail at faint G and restore symmetry where the search needs it.
    It needs testing on the DR4 data model (G7) and on injections before it is relied on. The June draft data
    model puts per-CCD G (`epoch_photometry_ccd`) in DataLink only, so on day one the test runs on a pull of the
    top ~10⁴ by |W| in each G ≥ 16 bin, ~9 h at DR3 rates. The pull is chosen by |W| regardless of sign, so
    each source's dim/bright coin stays fair and knockoff+ on the CCD-cleaned W of the pulled set stays valid.
    Choosing by the dim score alone would bias it. The test is `gaia.dr4.ccd_at_extremes` (PR #36): `frac_ccd_*`
    and a trimmed coherent shift `z_trim_*`, which drops the 2 most extreme CCDs on that side, at each source's
    faintest and brightest transits.
  - **The floor binds.** At q ≤ 0.01 each G bin needs 100 discoveries, so `fov` reports none. Lists for
    vetting use q ≤ 0.1.
