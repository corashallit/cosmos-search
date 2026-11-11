# Plan

The target: on Gaia DR4's release (2026-12-02), search its epoch photometry for stars that fade or brighten over
years and for dimming episodes of weeks to years that neither the instrument nor known astrophysics explains,
and hand over a short list with a measured recall and false-positive rate. Two smaller searches (short dips,
non-Keplerian astrometry) run on DR4 as built. The order of work on the day is in
[release-day.md](release-day.md).

## Guiding principles

1. **Reason from first principles.** Standard astronomical practice encodes decades of assumptions about what is
   worth looking for. Use it as evidence. Ask what physics, statistics and engineering say an anomaly would look
   like, and what the data can and cannot distinguish.
2. **Recalibrate as results come in.** Every detector, threshold and ordering is provisional. When a result
   comes in, question the framing one or two steps above the fix. Write down what changed and why in the search's
   page.
3. **The instrument is the main source of anomalies.** In a survey of 2 × 10⁹ sources the strangest object is
   almost always a calibration, crowding or scanning artefact. Gaia's redundancy (9 CCDs per transit, two fields
   of view 106.5 min apart, simultaneous G, BP and RP, a known scanning law) is the strongest tool against that.
4. **Order by value.** A candidate is worth a review in proportion to the chance it is real and unexplained times
   how cheaply it can be verified (brightness, observability, recurrence). Ordinary stars come before known
   variable classes. The review budget is about 10³ candidates, 10⁴ at the outside.
5. **Measure ourselves.** Injected signals and blinded decoys run throughout ([blinding.md](blinding.md)). A
   list is only credible with its measured recall and false-positive rate attached, and fading stars are always
   read against brightening stars of the same kind.
6. **No invented schedules.** Work is ordered by dependency. The only dates are external.

## External dates

| Date | Event |
|---|---|
| June 2026 (done) | DR4 pre-release: epoch astrometry for 12 sources, the draft data model |
| 2026-11-25 12:00 CET | ESA's Gaia archive closes until the release |
| **2026-12-02, about 12:00 CET** | **Gaia DR4** (66 months of data; epoch photometry for every source, 1.2 × 10¹¹ transits, 7.2 TB) |

## Where things stand

| Search | State | Page |
|---|---|---|
| **Slow changes** (main since 2026-10-23) | Detector for trends and long episodes, slow period search, fades in time order, thresholds set from the release's own stars, automated cuts, checks per candidate against NEOWISE, colour and ZTF, figures. Tried end to end on all 11.6 M DR3 stars with light curves (2026-10-30). Blind review calibrated on 432 DR3 objects: recall of anomalous fakes 44/50, false shortlist 5.0% at the flagged mix | [arms/slow-changes.md](arms/slow-changes.md) |
| Short dips | Built and frozen (shortlist rule v5); runs on DR4 as is | [arms/gaia-dimmers.md](arms/gaia-dimmers.md), [dayone.md](dayone.md) |
| Non-Keplerian astrometry | Rule astrometry-v1 adopted after a sealed blind test; runs as built | [arms/astrometry.md](arms/astrometry.md) |
| Dips in TESS full-frame images | Built; a cross-check for Gaia candidates | [arms/tess-dimmers.md](arms/tess-dimmers.md) |

Shared pieces, all built: archive client with per-host limits and a cache ([infra/net.md](infra/net.md)); Gaia
epoch photometry readers for DR3 and DR4, bulk files and DataLink ([infra/gaia-epoch.md](infra/gaia-epoch.md),
[infra/dr4-downloads.md](infra/dr4-downloads.md)); table and column names per release
([infra/targets.md](infra/targets.md)); cross-match to eleven catalogues with a measured false-match rate
([infra/xmatch.md](infra/xmatch.md)); injection and blinding ([blinding.md](blinding.md)); statistics
([statistics.md](statistics.md)); review of candidates ([vetting.md](vetting.md)); figures ([viz.md](viz.md)).

Milestone labels used in older pages (I1 to I6 infrastructure, S1 to S8 slow changes, G1 to G7 short dips, T1 to T5
TESS, A1 to A3 astrometry, V1 to V3 review, D0 the DR3 practice of the short-dip search) are all done except the
ones listed below; their history is in each page's log and in git.

## Before the release

From the whole-project review of 2026-11-10, in the order they depend on each other:

| # | Work | Depends on | State |
|---|---|---|---|
| R1 | Score every star from the bulk files: a summary for each star, the full detector to about G 16, trends measured against neighbours of similar brightness. The archive-side selection kept 0% of injected fades up to 3% per year and 47% at 10% per year | DR4 file reader | In progress |
| R2 | Order candidates within kind of star, ordinary dwarfs first. Ordered by strength alone, the list starts with known variables | R1 | In progress |
| R3 | A DataLink list by value, for a release without bulk files (ESA's DR4 page names DataLink only) | R1 | In progress |
| R4 | An upper limit on how common steady fades are among ordinary stars | R1 | Built and run on DR3 ([arms/occurrence.md](arms/occurrence.md)): of 188,485 FGK dwarfs in the Andromeda field, fewer than 1 in 7,800 fades at 1% a year. The 3.6% of dwarfs whose light is mixed with a neighbour's are left out, because Gaia's G photometry of them drifts (settings v2, 2026-11-11). Settings frozen in `prereg/occurrence/`. On DR4 it runs on the threshold sample at once and on every star from R1's per-star table, with the dwarfs' parallaxes fetched in sky slices. The whole-sky route was tried on the DR3 field only |
| R5 | Readers for ASAS-SN, ATLAS, SkyMapper, NOIRLab and TESS, so a fade can be checked in an independent survey (ZTF is the only one today) | none | In progress |
| R6 | Name the model in the reviewing agents' files and repeat the 432-object calibration on it | none | In progress |
| R7 | Release-day order of work: DR4 file and DataLink readers tried end to end on fake DR4 files built from DR3, first-hour checks, a scheduled start | none | Written and practised ([release-day.md](release-day.md)); the practice run is repeated when R1 lands, and the scheduled start waits for Cora's word |
| R8 | Licence, citation file, a Zenodo copy and an announcement plan | Cora | Licence and citation file in place; the deposit and any announcement are Cora's to make |

A practice run at the size of all of DR3 on kore needs Cora to start it there.

## Release day and after

| # | Step | Depends on |
|---|---|---|
| D2 | Freeze before release: code on `main`, thresholds procedure, review instructions and the written predictions under `prereg/` | R1 to R7 |
| D3 | Release day: first-hour checks, thresholds from DR4's own stars, every star scored, lists, automated cuts ([release-day.md](release-day.md)) | D2, the release |
| D4 | Checks per candidate against independent surveys, figures, blind review of the top of the lists in the order of R2 | D3, R5, R6 |
| D5 | Short list with its measured recall and false-positive rate and the occurrence limit; Cora decides on outreach and publication | D4, R4 |

## Open decisions

| Item | State |
|---|---|
| Routes to DR4's light curves | Decided on the day by `gaia.dr4_routes`: bulk files, a mirror's table, or DataLink only ([infra/dr4-downloads.md](infra/dr4-downloads.md)) |
| Epochs DPAC would have rejected | The draft DR4 table does not mark them; the scoring applies its own rule ([release-day.md](release-day.md)) |
| One outside astronomer to read the method before release | Cora |
| Zenodo deposit, announcement | Cora, after D5 or earlier if she chooses |
| Follow-up partners | After D5. Candidates: Wright group (Penn State), Project Hephaistos (Uppsala), Boyajian (LSU) |
| AAVSO account for ground follow-up | When there is a bright candidate |
