# Concurrent photometry for Gaia DR4 one-look dips

Thread "Concurrent ZTF and TESS coverage", 2026-10-21. Code: `cosmos_search/dayone/concurrent.py`,
`cosmos_search/gaia/gost.py`, `cosmos_search/dayone/external.py`. Data: `/mnt/project-files/concurrent/`.

**Question.** Gaia sees a star in short visits weeks apart, so about three quarters of the dips it catches land
on one transit pair, and the single-event test cannot argue those (docs/dayone.md, single dips, step back). DR4
(2014-07-25 to 2020-01-20) overlaps ZTF (from 2018-03-17, 34% of the span) and TESS sectors 1–20. How many
one-look dips does photometry taken at the same time make testable?

**Answer.** Few. On DR4's real cadence, with real ZTF and TESS light curves:

| caught 1–10 d dips, testable share | depth 1% | 3% | 10% | 30% |
|---|---|---|---|---|
| Gaia alone | 22% | 23% | 22% | 22% |
| + ZTF + TESS, single-event test as it is | 23% | 23% | 23% | 23% |
| + ZTF + TESS, lone/join rules relaxed (below) | 23% | 24% | 25% | 25% |
| + ZTF, relaxed, stars with ZTF data, G 13–17 | 23% | 24% | 30% | 31% |

Concurrent archival photometry raises the testable share by one point as the test stands, and by 3–9 points
with the rule changes below, only for dips of ≥ 10% on northern stars of G 13–17. It cannot remove the one-look
problem. Duty cycles set the ceiling. A TESS sector is 27 d of a 2,005 d span, so about 1% of one-look dips on a
star with TESS have a concurrent TESS epoch. ZTF covers a third of the span, sees a star for part of each year,
revisits a band every ~3 nights, and needs dips of several per cent to see them.

## Method

- **Sample.** 1,000 random `gaiadr3.gaia_source` stars, 200 in each G bin (10–13, 13–15, 15–17, 17–19,
  19–20.5), in `random_index` order.
- **Gaia DR4 cadence.** The published commanded scanning law stops at DR3's end (2017-05-28). ESA's Gaia
  Observation Forecast Tool (GOST) predicts every field-of-view transit up to 2025-01-15, so `gaia.gost`
  queries it (≤ 10,000 targets per call; 1,000 took 133 s). Median 89 predicted DR4 transits per star; DR4/DR3
  predicted ratio 1.96. DR3's `matched_transits` over GOST's DR3-span prediction is 0.98, 0.97, 0.96, 0.94 and
  0.82 per G bin, and predicted transits are thinned at random by that share. Gaia epochs get Gaussian noise
  at DR3's median per-transit G error for the star's magnitude (`gaia_err.csv`, floor 1 mmag).
- **External data as they are.** ZTF from IRSA's light-curve service (catflags 0, every data release) for the
  610 stars north of −31°: 69–91% of them have ZTF, with a median of 86–477 epochs per G bin inside DR4's span.
  TGLC for TESS sectors 1–20: 63%, 61% and 43% of stars at G 10–13, 13–15 and 15–17, none fainter.
- **Injections.** Grey trapezoid dips (ingress a fifth of the duration), durations 0.1–30 d, depths 1–30%.
  Dip centres are drawn uniformly over the times when a G epoch moves by ≥ 5σ (the decoys' rule), 30 per star
  and cell, and multiplied into the real ZTF and TESS fluxes. No dilution in TESS, which favours TESS.
- **Testable** is the single-event test's own gate (`vetting.single_event` `_band_epochs` and `_event`): one event
  across surveys, ≥ 2 in-dip epochs including G, beyond one transit pair, bracketed by normal epochs.

Gaia alone reproduces the step back: 75% of caught 1–10 d dips are one-look on DR4 cadence (77% at 1 d, 69% at
30 d, 99% at 0.1 d), and 21–24% are testable.

## What else this found

1. **The TESS data that DR4 overlaps were being thrown away.** `external._bin` binned TGLC to 30 min and dropped
   bins with fewer than two points. Sectors 1–26 are 30-min FFIs, so every epoch from them was dropped (one star:
   937 clean cadences in sector 4, 0 kept). This explains the "about 1 TESS epoch per star in the window" of the
   single-dip calibration. Fixed: single-point bins are kept. Their error is the robust point-to-point
   scatter / sqrt(n), inflated by a red-noise factor (`external.red_noise_beta`, review finding R18): the median
   over 0.25, 0.5, 1 and 2 d of the robust scatter of bin means over what white noise predicts, at least 1. It is
   1.0–1.6 on the sectors checked, and its 95th percentile on pure white noise is 1.21. The estimator is robust,
   so a short excursion (a 13% systematic dip on 2.6% of one sector's points) stays an event and does not inflate
   every error. The gates and tests above use the robust scatter of the fluxes, not these errors, so the
   coverage numbers do not change. The errors reach the Stage A packets and Stage 0's cross-band check.
2. **The fetch now ranks TESS sectors by closeness to the Gaia dip** (`t_deepest`; without it, sectors inside
   DR4's span first) before `max_sectors` cuts the list. Sector times are in `tess/sectors.csv`, measured from one
   light curve per sector (S1–S98).
3. **The single-event test sets concurrent ZTF epochs aside.** Its lone-epoch rule keeps an external in-dip
   epoch only if another in-dip epoch of the same band lies within a day. ZTF revisits a band every ~3 nights,
   so a 30-d dip that ZTF sees on every visit is discarded epoch by epoch, including a ZTF epoch 1.5 min from
   the Gaia epoch. The cross-survey rule then splits Gaia from ZTF epochs more than a day apart. The relaxed
   variant (`concurrent.relax_lone`, `concurrent.event_joined`) keeps an external epoch whose neighbour in the
   same band is also in dip, or that has an in-dip epoch of another band within a day, and joins surveys unless
   an informative normal epoch lies between. For ZTF on one-look dips of ≥ 10%, this raises the testable share
   about threefold (for 10 d dips at G 13–15, from 1.9% to 10.1% of all one-look dips).
4. **Real TESS light curves often break the test.** "Exactly one event across surveys" fails when the star's
   TESS light curve has its own dip-like systematics. Adding TGLC turns 30% of Gaia-testable dips and 34% of
   one-look dips into "more than one event" (ZTF: 9% and 8%). On day one, adding TESS to a packet as it is costs
   more single-event tests than it enables.
5. **Faint stars gain nothing.** At G 19–20.5, ZTF's per-epoch noise is comparable to the dips Gaia catches,
   and TESS has no data.

## Evidence when testable

When concurrent ZTF does make a one-look dip testable (relaxed rules), the evidence is about as strong as for a
dip Gaia alone saw on several visits. On 150 injected dips of each kind (depth ≥ 3%), the single-event Monte Carlo
gives, for dips of ≥ 1 d, a median log10(one-off / eclipse) of 3.1 with ZTF and 3.4 with Gaia alone; 63% and 74%
reach the flag level (≥ 2.5). This is recall on injected dips only. The relaxed rules also let real ZTF outliers
count, so their false-positive rate on eclipsing binaries and active stars needs the single-dip calibration
before any rule uses them.

## What would raise the ceiling

- **ATLAS** (forced photometry; one Hawaii unit from 2015, a second from 2017; about every second night,
  four exposures within an hour, to ~19 mag) overlaps about three quarters of DR4's span, and its same-night quads corroborate
  themselves. As a rough scaling from ZTF's duty cycle, not a measurement, it should give about 1.5–2× ZTF's
  concurrent coverage north of −45°. It needs a free account at fallingstar-data.com; the server is reachable
  from the container.
- **ASAS-SN** (V 2014–2018, g 2017 onward, near-daily, whole sky, ~8" pixels, a few per cent precision at
  G ≲ 16) covers all of DR4's span. Sky Patrol v2's server (port 9006) is unreachable from the container and
  from kore's VM; it would need a session with open network.
- Neither changes the arithmetic much for shallow dips: only TESS and Gaia reach 1% at G > 13, and TESS
  overlaps DR4 for one sector per star.

## Reproduce

    python -m cosmos_search.dayone.concurrent sample --per-bin 200 --out cz/sample.parquet
    python -m cosmos_search.dayone.concurrent fetch --sample cz/sample.parquet --out cz/ --surveys gost,tess
    python -m cosmos_search.dayone.concurrent fetch --sample cz/sample_north.parquet --out cz/ --surveys ztf
    python -m cosmos_search.dayone.concurrent run --dir cz/ --gaia-err cz/gaia_err.csv --draws 30 --out cz/grid.parquet

`sample_north.parquet` is the sample with dec > −31°. IRSA answers in ~30 s per star with four at a time, so the
ZTF fetch for 610 stars takes the longest.
