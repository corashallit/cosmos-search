# Aperiodic dimmers in TESS full-frame images

Side arm and rehearsal for the Gaia arm. Goal: a ranked list of stars whose light drops in ways neither the
instrument nor known astrophysics explains, with a measured recall and false-alarm rate, plus the numbers
decision D1 needs (recovery on injections, vetting cost per surviving candidate). The detector lives in
`cosmos_search/dimming/` and takes any light curve (times, relative flux, errors, flags, band), so the Gaia
arm runs the same code.

## What the data can see

QLP light curves (Huang et al. 2020; MAST HLSP, public S3 bucket) cover stars to Tmag ≈ 13.5 in early
sectors and fainter since, one ~27 d sector at a time, 200 s cadence from sector 56 on, with a ~1 d gap
mid-sector and scattered-light stretches near each orbit's perigee. Single-band, so no dip colour. TGLC (Han &
Brandt 2023) covers fainter stars with Gaia-informed PSF photometry and is keyed by Gaia DR3 source_id, which
makes it the bridge for T5 (TESS cross-check of Gaia candidates).

Light curves are streamed from S3 and discarded; only features and events are kept (~10 MB per 10⁴ stars).

What one sector can and cannot test:

| Signal | One sector | Notes |
|---|---|---|
| Dips of 1 h to ~4 d | Yes | The main channel. Longer dips are partly absorbed by any baseline |
| Shape (ingress/egress, flat bottoms, asymmetry) | Yes, for bright stars | Gaia cannot resolve shape |
| Recurrence and timing structure | Within 27 d; across sectors for the CVZ | Prime/geometric spacings are injected |
| Secular fades | No | ≤ 0.4% over a sector for the injected priors; this is Gaia's job (5.5 yr) |
| Colour | No | Gaia arm |

## Method, and why

Starting points are first principles rather than the Kepler/TESS planet-search habit of detrending and
searching for periodic boxes.

1. **One-sided baseline.** Occultation only removes light. The trend is fitted with asymmetric clipping
   (points well below are excluded and bridged), so a day-long dip is not absorbed into the baseline the way
   a symmetric spline (QLP's own detrending, 0.3–2.5 d breaks) absorbs it. We read QLP's raw SAP flux for
   this reason.
2. **Shape-agnostic multi-scale scan.** A moving mean of the residual over windows of 1 h to 3.6 d, divided
   by its own measured scatter at that scale. Dividing by the measured scatter of the moving mean (a robust
   "β factor") makes the statistic answer "how unusual is this for this star at this scale", so red noise
   and stellar variability raise the bar automatically.
3. **Heteroscedastic noise.** A per-point noise level (moving RMS of point-to-point differences) instead of
   one global sigma, because noise is not stationary within a sector.
4. **Mirror null.** The same statistic on the light curve reflected about 1 measures how often noise,
   systematics and variability produce a "dip" (statistics.md §2). p-values come from
   `cosmos_search.stats` in Tmag strata.
5. **Physics vetoes rather than significance cuts.**
   - *Background coupling.* Fit flux against background outside the dip, predict the in-dip change from
     the background, and ask what fraction of the dip it explains. Only counted when the background *rose*
     unusually during the dip, because over-subtracting a rising background is how background makes a star
     look faint; a background that falls with the flux is the star's own light in the annulus, which is
     evidence for an on-target dip. Centroids are reported to the vetters but not used to veto: when a
     blended star dims, its light-weighted centroid moves towards its neighbours, so the centroid "explains"
     real dips.
   - *Aperture ratio.* A real on-target dip has the same fractional depth in every aperture. Only for dips
     under 0.1 d, because QLP's alternate apertures are spline-detrended.
   - *One-sided events.* A dip that starts or ends on a data gap has no baseline on one side and cannot be
     told from a ramp, the commonest TESS systematic.
   - *Occultation kinematics.* An occulter of radius r crossing at speed v changes flux over 2r/v and removes
     at most (r/R*)²; so the deepest change possible within one cadence Δt is (v_max Δt / 2R*)², with
     v_max = 2 v_esc(star). About 3% for a Sun-like star at 200 s. Faster steps cannot be occultations
     (flares in the mirror, instrumental jumps, or the star changing its own output); they go to a separate
     "sudden change" channel instead of polluting the occultation channel.
   - *Common mode.* Events shared by improbably many stars on the same camera/CCD in the same time bin
     (binomial test) are instrumental. An event is vetoed when at least half its time is in such bins.
6. **Hypothesis-free score alongside.** An isolation forest plus kNN distance on noise-normalised features
   (extreme statistic and red-noise level at every scale, tail ratios, quantile skew, Cody M and Q,
   bimodality, von Neumann ratio), standardised within Tmag strata, asks "how unusual is this light curve
   among 10⁴ others" without a template.
7. **Injections** (shared `inject/lightcurve.py` plus `inject/classes/dimming.py`): grey/dust/blue dips, dip
   trains with Poisson, periodic, prime and geometric spacing, fades, steps, flares, chromatic flips, and
   Boyajian-like complexes, shallow swarms, strictly periodic dips of random depth, and telegraph (discrete
   level) switching. Each star is processed three ways: real, mirrored, and with one injected signal, so
   recovery is measured on real noise and real systematics.

## Running it

    python -m cosmos_search.tess.bakeoff sample  --out data/bakeoff/v0 --sectors 60 66 72 78 84 90 --n 10000
    python -m cosmos_search.tess.bakeoff process --out data/bakeoff/v0 --workers 12
    python -m cosmos_search.tess.analyse --out data/bakeoff/v0
    python -m cosmos_search.tess.controls --run data/bakeoff/v0
    python -m cosmos_search.tess.vet build --out data/bakeoff/v0 --controls cosmos_search/tess/controls.csv

`analyse` writes `summary.json`, `recovery.csv` (one row per injection) and `candidates.parquet`; `controls`
scores the known-answer stars against the same nulls; `vet` writes a Stage A batch through the shared
vetting package.

## Known-answer controls

`cosmos_search/tess/controls.csv` lists 196 stars with literature evidence (bibcodes in the file, method in
`controls_notes.md`): YSO dippers (39), aperiodic deep dimmers including exocomet hosts and ASAS-SN dippers
(24), R CrB stars (40), UX Ori / Herbig Ae/Be (48), long or disk eclipses (8), detached EBs as a reference
(30), and documented TESS false alarms (7). Each row gives the QLP sectors in which the dimming is
documented, so each star is tested in its event sectors and in up to two quiet sectors. The quiet sectors
of the same stars are the best negatives available: same star, same pipeline, different state.

## Results, v0 (2026-10-21)

9,973 QLP light curves, ~1,660 each from sectors 60, 66, 72, 78, 84 and 90, each searched three ways (real,
mirrored, one injection). Outputs (summary, per-injection recovery, candidates, controls, Stage A batch) are on
the project share under `tess-dimmers/v0/`.

**No population excess.** 103 real light curves pass the dip channel at FAR 10⁻², against ~100 expected from
the mirror null, and none survive Benjamini–Hochberg at q = 0.1. Real and mirrored data have surviving events
on 1,655 and 1,566 stars. Mirror-null thresholds at FAR 10⁻²: SNR 29.5 for events up to 3 h (flares in the
mirror set this) and 12.7 for longer ones.

**Recovery at FAR 10⁻²**, by peak depth over the star's per-cadence noise:

| Family | 1–3 | 3–10 | ≥ 10 | All |
|---|---|---|---|---|
| Boyajian-like complex | 0.10 | 0.39 | 0.77 | 0.45 |
| Grey dip | 0.08 | 0.27 | 0.58 | 0.27 |
| Dip train, periodic | 0.07 | 0.30 | 0.80 | 0.36 |
| Dip train, primes | 0.01 | 0.08 | 0.70 | 0.29 |
| Dust dip | 0.01 | 0.20 | 0.54 | 0.17 |
| Shallow swarm | 0.00 | 0.04 | 0.12 | 0.02 |
| Telegraph | 0.03 | 0.17 | 0.18 | 0.06 |

The population anomaly score adds most for prime-spaced trains (0.29 → 0.40 with either channel). Secular
fades and steps are invisible in one sector by construction. The largest single loss is common mode: 21%
of injected detections fall in epochs where many stars on the CCD dip together (mostly sector starts), which
is the case for subtracting the CCD's common mode rather than vetoing it.

**Known-answer controls** against the same null (fraction of sectors flagged by either channel, event
sectors vs quiet sectors of the same stars): aperiodic deep dimmers 0.29 vs 0.08; long or disk eclipses 0.43
vs 0.00; YSO dippers 0.57 vs 0.46; UX Ori 0.54 vs 0.46; R CrB 0.21 vs 0.18; detached EBs 0.51 (event only).
Stars that dip all the time are caught by the anomaly score (YSO dippers 0.55) and rarely by the dip channel
(0.02), because the per-star red-noise factor makes a dip ordinary for a star that always dips. That is the
intended split: the dip channel looks for rare events on otherwise quiet stars. R CrB declines last weeks
and are mostly invisible within one sector; that is the Gaia arm's job.

**Stage A pilot** (13 packets: the 8 highest-value candidates, 3 controls, 2 decoys; blind subagents, one
read of their own packet each, audited from transcripts; key sealed with a session-only age identity until
the verdicts were committed). All 8 real candidates were judged ordinary: 7 eclipsing binaries (four with
period and secondary worked out from the packet) and one blended EB; the highest P(unexplained) was 0.20.
Controls: 3/3 in the correct group. Decoys: the prime-spaced grey train got P(unexplained) 0.55 and the
periodic grey decoy was called an EB, both correct. Cost per packet: ~55k context tokens (Claude Code
subagent overhead included; ~10k through the Messages API) and ~2.6k output tokens, 22–82 s.

**Vetting cost per surviving candidate.** Zero of 8 top candidates survived, so the survival rate is below
~0.37 (95%), and the dominant contaminant is ordinary EBs. Stage A on the whole 193-candidate queue would
cost ~10⁷ tokens per 10⁴ stars, and with a control false-unexplained rate of a few per cent would put ~4
false entries on the shortlist. A deterministic EB filter (two or more equal-depth dips at a common period,
or a secondary at phase 0.5) ahead of Stage A is the cheap fix; most of the queue is EBs.

**Decoys leak.** Two experts spotted the injection: one saw the dip missing from QLP's alternate apertures,
another saw the flux error scale with the flux inside the dip. Decoys must be applied to every flux channel
and leave the errors alone, or Stage A learns to recognise them.

## Milestones

| # | Milestone | Depends on | Status |
|---|---|---|---|
| T1 | QLP/TGLC access, streamed through `cosmos_search.net` | I1 | Done |
| T2 | Detector in `dimming/`, survey-agnostic | — | v0 done |
| T3 | Injection families and recovery maps | inject core | v0 done (visible tuning set) |
| T4 | 10⁴-target comparison run: recovery, false alarms, vetting cost per survivor | T1–T3, V2 | v0 done (below) |
| T5 | TESS cross-check of Gaia candidates via TGLC (Gaia source_id keyed) | T1, G6 | Not started |

## Log

Changes of approach, with the evidence that prompted them.

- **Scattered light removed, not modelled** (500-star pilot, sector 70/72). QLP SAP flux swings by tens of
  percent where the background rises by 10³ near perigee; one Tmag 14.6 star gave 20 "dips" at ≥ 5σ, all
  in flagged cadences (bits 2048 and 2³⁰). Those cadences are now dropped; injections falling there go
  unrecovered, so the cost is inside the recovery numbers.
- **Nuisance vetoes recalibrated against injections.** A white-noise z-score of background and centroid
  shifts vetoed 26% (background) and 17–50% (centroid) of *injected* dips, which by construction have no
  coupling to either. Replaced by the coupling regression above: injected loss 4.7%, while it still removes
  50% of real and 40% of mirrored events. The aperture-ratio veto removes 1.3% of injected and 29% of real
  events once the injections are also applied to the alternate apertures (they had not been at first,
  which made every injected dip look off-target).
- **Mirrored flares dominate the short-event null.** In the pilot the mirror produced more surviving events
  than the real data, nearly all at the 1 h scale. Two responses: the kinematic split above, and separate
  nulls for short (≤ 3 h) and long events, so flares do not set the bar for day-long dips. At FAR 10⁻² the
  SNR threshold is ~24 for short events and ~8.5 for long ones.
- **QLP file versions.** Sectors 74–79 are delivered as `v02`; the first sample silently skipped sector 78.
- **Shared library fix.** Dip-train priors assumed Gaia's multi-year span and failed on 27 d; fixed upstream
  (PR #4), and the spacing now scales with the span.
- **Controls overturned three vetoes** (known dimmers in their documented event sectors, 2026-10-21).
  - Centroid coupling removed a third of real dimmers' dips. When a blended star dims, the centroid moves
    towards the neighbours, so centroid change is a consequence of the dip rather than its cause. Centroids
    are now evidence for the vetters, not a veto.
  - The background veto now needs the background to have *risen* (placement z > 2). Of background-"explained"
    dips, 86% in quiet sectors had a rising background against 48% in event sectors; the direction rule
    vetoes 3% of event-sector dips and 12% of quiet-sector dips (was 9% and 16%).
  - QLP's alternate apertures are detrended with 0.3-2.5 d spline breaks, which ate 0.1-0.3 d dips of
    known YSO dippers and failed them on aperture ratio. Injections, multiplied in after that detrending,
    could not show this. The ratio is now used below 0.1 d only.
- **Kinematic edge test on median-3 flux.** Single-cadence outliers (cosmic rays, hot pixels) registered as
  two impossible steps and sent real dips to the "sudden change" channel.
- **Edges: one-sided events only.** The 0.1 d segment-edge veto removed 15% of event-sector dips. Measured
  against the mirror, the excess is entirely in events that touch the edge (mirror/real 2.7 at < 0.01 d,
  0.4 at 0.01-0.1 d), so the veto is now "starts or ends on a gap".
- **Common mode by coverage.** "Any overlap with a flagged 0.25 d bin" vetoed long dips that brushed one
  bin. Now at least half the event's bins must be flagged. Sector 78 still loses ~17% of injection windows
  to its first days, when cameras 3-4 share a large ramp. Subtracting the CCD's common mode (estimated from
  its ~100 searched stars) would recover that time; it needs a two-pass pipeline and is not done yet.

