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
   - *Coupling regression.* Fit flux against background and centroids outside the dip, predict the in-dip
     change from those channels, and ask what fraction of the dip they explain. An occultation has nothing
     to do with the background; a scattered-light dip is mostly background.
   - *Aperture ratio.* A real on-target dip has the same fractional depth in every aperture.
   - *Occultation kinematics.* An occulter of radius r crossing at speed v changes flux over 2r/v and removes
     at most (r/R*)²; so the deepest change possible within one cadence Δt is (v_max Δt / 2R*)², with
     v_max = 2 v_esc(star). About 3% for a Sun-like star at 200 s. Faster steps cannot be occultations
     (flares in the mirror, instrumental jumps, or the star changing its own output); they go to a separate
     "sudden change" channel instead of polluting the occultation channel.
   - *Common mode.* Events shared by improbably many stars on the same camera/CCD in the same time bin
     (binomial test) are instrumental.
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
    python -m cosmos_search.tess.vet build --out data/bakeoff/v0 --controls cosmos_search/tess/controls.csv

`analyse` writes `summary.json`, `recovery.csv` (one row per injection) and `candidates.parquet`; `vet`
writes a Stage A batch through the shared vetting package.

## Milestones

| # | Milestone | Depends on | Status |
|---|---|---|---|
| T1 | QLP/TGLC access, streamed through `cosmos_search.net` | I1 | Done |
| T2 | Detector in `dimming/`, survey-agnostic | — | v0 done |
| T3 | Injection families and recovery maps | inject core | v0 done (visible tuning set) |
| T4 | 10⁴-target comparison run: recovery, false alarms, vetting cost per survivor | T1–T3, V2 | Running |
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
