# Plan

## Guiding principles

These apply to every arm and every thread.

1. **Reason from first principles, not from astronomy habit.** Standard astronomical practice encodes
   decades of human assumptions about what is worth looking for. Use it as evidence, not as a template.
   Ask what physics, information theory, statistics and engineering say an anomaly would look like,
   and what the data can and cannot distinguish. Cross-pollinate: astroengineering (energy budgets,
   occulter geometry, propulsion), statistics (calibrated tails, permutation nulls, multiple testing),
   data science (representation learning, active learning), and maths.
2. **Recalibrate as results come in.** Every detector, threshold and ranking is provisional. Injection
   recovery, random audits and vetting outcomes feed back into what we look for. Write down what changed
   and why in the arm doc.
3. **Hypothesis-driven and hypothesis-free, side by side.** Physically motivated detectors (grey dips,
   energy-budget tension, non-gravitational motion) run alongside detectors that only ask "how surprising
   is this, given everything like it?". Injections span families neither kind was designed for, so we
   measure how agnostic each really is.
4. **The instrument is the main source of anomalies.** In a survey of 1.8 × 10⁹ sources, the weirdest
   object is almost always a calibration, crowding or scanning artefact. Gaia's redundancy (9 CCDs per
   transit, two fields of view 106.5 min apart, simultaneous G/BP/RP, a known scanning law) is the
   strongest tool we have against that, and it beats any prior about what aliens build.
5. **Rank by value, not by weirdness.** A candidate is worth a vetting slot in proportion to
   P(real and unexplained) × how cheaply it can be verified (brightness, observability, recurrence).
   The vetting budget is ~10³ candidates, with ~10⁴ as the outer limit.
6. **Measure ourselves.** Injection-recovery and blinded decoys run throughout (blinding.md). A shortlist
   is only credible with its measured recall and false-positive rate attached.
7. **No invented schedules.** Milestones are ordered by dependency. The only dates are external.

## External dates

| Date | Event |
|---|---|
| June 2026 (done) | Gaia DR4 pre-release: epoch astrometry for 12 sources, draft DR4 data model, "Gaia source update" tool |
| **2 December 2026** | **Gaia DR4** (66 months, ~400 TB). Remaining DPAC tools released the same day |
| early 2027 | DESI DR2 |

## Milestones

| # | Milestone | Depends on | Owner |
|---|---|---|---|
| **Infrastructure** | | | |
| I1 | Shared HTTP/TAP client with per-host policies and cache ✓ | — | Archive thread |
| I2 | Gaia DR3 epoch photometry streamed to per-source features (bulk CDN files and DataLink) ✓ | I1 | Archive thread |
| I3 | Versioned target queries with a freeze manifest | I1 | Archive thread |
| I4 | Durable store | Cora | Cora |
| I5 | Proper-motion-aware cross-match ✓ (`xmatch/`; docs/infra/xmatch.md): 2MASS, AllWISE, CatWISE2020, unWISE, NEOWISE epochs, TIC, eROSITA DR2, ZTF DR22, VSX, ASAS-SN, SIMBAD. Calibrated per batch, Gaia neighbours compete for survey sources, false-match rate measured on the shifted sky per run, `p_cal` and `p_chance_0.9` per source | I1 | Cross-match thread |
| **Gaia dimmers (main)** | | | |
| G1 | DR3 rehearsal sample: full DR3 epoch photometry (11.75 M) plus random controls ✓ (features for every source; docs/infra/gaia-epoch.md) | I2 | Archive thread |
| G2 | Detectors v0: invariance tests (FoV pairs, bands: stage-0 mask ✓ `vetting/stage0.py`), dip colour, secular trend, surprisal score. The per-candidate path (DataLink → `stage0.clean_long` → detector) is not written yet | G1 | TESS thread's survey-agnostic detector in `dimming/`, then Gaia-specific tests |
| G3 | Empirical nulls (brightening mirror, stratified permutation) and calibrated false-alarm rates. Mirror competition per G bin ✓ on all of DR3 (`scripts/dr3_g3.py`; arms/gaia-dimmers.md log). No power yet at G ≥ 16; stratified permutation not started | G2 | DR4 plan thread (library ✓ `stats/`); detector owner (strata and runs) |
| G4 | Injection families (library ✓ `inject/lightcurve.py`) and recovery maps on DR3 (binning ✓ `inject/recovery.py`) | G2 | DR4 plan thread (library); detector owner (maps) |
| G5 | Energy-budget test against 2MASS/WISE ✓ `energy/` (docs/arms/energy-budget.md): excess and shell/thin deficit per star, calibrated on halves A/B/C, packet context and rank columns | I5, G2 | Waste-heat thread |
| G6 | Ranking with value term, plus a random-audit stream below threshold (audit sampler ✓ `stats/audit.py`) | G3, G4, G5 | unassigned |
| G7 | DR4 adapters from the draft data model (per-CCD G, new vari tables); I3 and G7 in review (docs/infra/targets.md) | I3 | Archive thread |
| **TESS dimmers (side)** | | | |
| T1–T4 | FFI access, detector in `dimming/`, injections, 10⁴-target comparison run | I1 | TESS thread |
| T5 | TESS cross-check of Gaia candidates | T1, G6 | TESS thread |
| **Astrometry (secondary)** | | | |
| A1 | Reproduce the 12 pre-release sources with the source-update tool | I1 | unassigned |
| A2 | Nested model fits (single, acceleration, 1–2 Keplerians) plus VIM and scan-angle terms | A1 | unassigned |
| A3 | Injections of non-gravitational motion into real residuals | A2 | unassigned |
| **Vetting** | | | |
| V1 | Packet builder, anonymiser and leak scan; FP-class catalogue; Stage 0 cross-candidate coincidence, band consistency and SSO check ✓ | G6 for real input | Vetting thread |
| V2 | Stage A / Stage B agents, decoy generator, commit-then-reveal, scoring, held-out targets ✓. Known-answer control lists (Gaia: vetting; TESS: `tess/controls.csv`) | V1, truth repo ✓ | Vetting thread |
| V3 | Statistics ✓ (label-shift shortlist FP, recalibration, allocation, sequential acceptance, simulation). Calibration run of ~400 on DR3 controls and decoys; then rehearsal on DR3 candidates | V2, G6, I4 | Vetting thread |
| **Decision and freeze** | | | |
| D1 | Compare arms on equal footing: recovery on injections and vetting cost per surviving candidate; reallocate effort | G4, T4, A3, V3 | DR4 plan thread → Cora |
| D2 | Freeze `dr4-day-one`: queries, thresholds, code, pre-registered analysis | D1, G7; before DR4 | DR4 plan thread |
| D3 | DR4 day one: TAP prefilter → DataLink pull of the top ~10⁴; full bulk stream of epoch photometry with detectors per file (falls back to a random DataLink channel if there are no bulk files); for G ≥ 16, a per-CCD DataLink pull of the top ~10⁴ by \|W\| per G bin (sign-blind, so competition stays valid; `gaia.dr4.ccd_at_extremes`, docs/infra/targets.md) → ranking → vetting | D2, DR4 release | all |
| D4 | Shortlist with measured recall and false-positive rate; decide on outreach to technosignature groups | D3 | Cora |
| D0 | Day-one dress rehearsal on DR3 ✓ (docs/dayone.md, `dayone/`): whole chain with 6,000 sealed injections. Fixed the DataLink flag parser and the `fov` cut asymmetry; recall is set by sampling and a ~100σ G3 threshold; pre-registered shortlist empty, 0/5 injections called unexplained. Changes for D2 listed there | G3, I5, V2 | DR4 day-one rehearsal thread |

## Open decisions

| Item | Status |
|---|---|
| Day-one stream capacity (arms/gaia-dimmers.md) | Is DR4 epoch photometry on the CDN at release (DR3's was)? If yes, how many sessions split the bulk stream. If no, the size of the random DataLink channel, set at D1 from DR3 yields |
| Durable store | Deferred; ~50–100 GB is enough for this track (storage.md) |
| Vetting budget | ~10³, ~10⁴ outer limit (decided) |
| Collaborators for follow-up | After D4. Candidates: Wright group (Penn State), Project Hephaistos (Uppsala), Boyajian (LSU) |
| AAVSO account for ground follow-up | When we have a bright candidate |
| WISE moving-object search, POSS-I plates | Parked |
