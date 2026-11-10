# Release day: order of work

Gaia DR4 becomes public on **2026-12-02 at about 12:00 CET** (11:00 UTC, 06:00 in Ontario). ESA's archive is
closed from 2026-11-25 12:00 CET until then. This page is the order of work for that day: what must be in place
before, the checks of the first hour, the run, and what to do when DR4 differs from what the code expects. Steps
are in dependency order; the only dates are ESA's.

The search itself is described in [arms/slow-changes.md](arms/slow-changes.md); the routes to the light curves in
[infra/dr4-downloads.md](infra/dr4-downloads.md).

## What is known and what is not

| | Known | Not known until release |
|---|---|---|
| Tables and columns | Draft data model of 2026-06-26 (`configs/dr4/draft-data-model-2026-06-26.json`); every release-day query is held to it by `tests/test_releases.py` | Schema name (`gaiadr4` assumed), last-minute renames |
| Epoch photometry layout | One row per transit, float32 fluxes, flag words (`gaia/dr4.py`) | Whether bulk files exist, their partitioning, file sizes, row order |
| Routes | DataLink is confirmed by ESA (DR4 content page, 2026-09-14) | Bulk files on the CDN (approved plan, not announced); a mirror's table |
| Rejected epochs | The draft `epoch_photometry` has no column for epochs rejected by variability processing; `vari_epoch_flags` (DataLink) covers only sources that went through it | Whether DPAC adds one |

## Before the archive closes (2026-11-25)

| Item | State |
|---|---|
| DR3 light curves of the 2,849 watch-list stars (fading at DR3's end, G < 17), to compare with DR4's continuation | Saved 2026-11-10: `/mnt/project-files/release-day/dr3-reference/watchlist/` |
| DR3 light curves of 2,360 ordinary stars over G 6.7 to 20.9 from 60 sky regions, for the DR4-against-DR3 check | Saved 2026-11-10: `/mnt/project-files/release-day/dr3-reference/random/` |
| Known answers matched to Gaia (Boyajian's star, fading alerts, R CrB / UX Ori / DY Per stars) | `/mnt/project-files/release-day/known.parquet` |
| DR3 per-star table (G, BP-RP, classes) for the 11.6 M DR3 stars with light curves | `/mnt/project-files/data/gaia-dr3-epoch/aux/` |
| All DR3 epoch photometry files | On kore (`~/cosmos-data`, 72.8 GB) |

## The day before

1. `main` is green and `python scripts/dr4_practice.py` passes in a fresh container (below).
2. `GAIA_USER` and `GAIA_PASSWORD` are set in the environment; the first archive call on release day logs in.
3. kore is on, awake and on mains power, with `~/cosmos-data/cosmos-search` at `main`.

## First hour

Each step gates the ones below it. A step that reports `problems` stops the run until fixed; `notes` are read and
the run goes on.

| # | Check | Command | If it fails |
|---|---|---|---|
| 1 | Is DR4 out, and by which routes | `python -m cosmos_search.gaia.dr4_routes --out routes.json` (exit code 3: no route answers yet) | Check again each hour |
| 2 | Tables and columns the queries name | `columns_missing` in `routes.json`, or `python -m cosmos_search.targets check --release dr4` | Edit `RELEASES["dr4"]` in `targets/releases.py` (one map for every query), re-run |
| 3 | One bulk file, smallest then largest | `python -m cosmos_search.gaia.dr4_bulk list`, then `python -m cosmos_search.gaia.dr4_check file KEY` | Column names: `ALIASES` / `COLUMNS` in `gaia/dr4.py`. Times outside DR4's span: `gaia_time_to_btjd`. Stars not contiguous: the reader sorts each file in memory, so lower `--workers` |
| 4 | First light curves through DataLink: Boyajian's star, the known answers, the DR3 reference stars | `python -m cosmos_search.gaia.dr4_datalink --ids LIST --out DL` | The response layout is new: `read_response` in `gaia/dr4_datalink.py` names the columns it found |
| 5 | DR4 against DR3 for the same stars | `python -m cosmos_search.gaia.dr4_check compare --dr4 DL/lightcurves --dr3 /mnt/project-files/release-day/dr3-reference/random/lightcurves.parquet` | No star in both: map ids through `dr3_neighbourhood` (`--id-map`). Flux scale or time offset: units or zero points differ from DR3's; fix the reader before anything is scored |
| 6 | Rejected epochs | Does `routes.json` list a `vari_epoch_flags` folder among the bulk files? Does step 3 report a rejection column? | Neither is expected. The scoring uses its own rule for bad epochs (below) |
| 7 | Class names | `SELECT phot_variable_best_class, COUNT(*) FROM gaiadr4.all_source_flags GROUP BY 1` | Update the eclipsing-binary class given to `slow_checks.py --skip-class` |
| 8 | Trend zero point | After the threshold sample is scored: median trend in units of its error, by G magnitude and by sky region | An offset at G < 14 was +0.85 errors in DR3 and would be about +2.3 at DR4's precision; it has to be removed before thresholds are set |

Steps 1 to 3 need no science decision and take what the archive's load allows. Step 4 is the first data anyone
on the project sees: Boyajian's star has no light curve in DR3 and gets one in DR4.

## The run

By route, fastest first (`routes.json` says which exist):

1. **Bulk files.** Thresholds from a random share of the files, then every star scored as the files pass, split
   over machines with `--shard i/n`; each raw file is deleted once read. Commands in
   [arms/slow-changes.md](arms/slow-changes.md), "Release day, end to end".
2. **A mirror's table, no bulk files.** Whole-sky selection on ESA's `all_source_photometry`, light curves of the
   selected stars from the mirror by source_id range.
3. **DataLink only.** The watch-list, the known answers and the top of the selection, at 0.1 to 2.0 light curves
   a second (2,849 DR3 watch-list stars took 24 min on 2026-11-10, no failed call); check the other routes daily.

Then lists, automated cuts, the checks per candidate and figures, as in "Release day, end to end".

Machines:

| Machine | What it has | Use |
|---|---|---|
| Cloud container (one per thread) | 4 cores, 15 GB memory, about 30 GB disk, 120 MB/s from ESA's CDN | The first pass, several containers each with its own `--shard i/n` |
| kore | 10 cores, 16 GB memory, 840 GB free disk | One or more shards with at most 6 workers; where the kept light curves and summaries are stored. Its sessions only act on messages Cora writes there |
| `/mnt/project-files/` | Shared by every thread, durable | Rules, thresholds, per-star summaries, lists, `routes.json`, first-hour reports |

Sizing from the practice run (below): reading and converting takes 1.9 core-seconds per million transits, so
DR4's 1.2 × 10¹¹ transits cost about 64 core-hours before any scoring; the download is 7.2 TB, 17 h for one
container's network. A worker needs about 1.2 GB of memory per 1.4 million transits it holds (30,000 stars at
DR3's sampling, about 15,000 at DR4's).

## Practice run

`gaia/dr4_fake.py` builds files in DR4's expected layout from DR3 bulk files and serves them, a file listing and a
DataLink endpoint on localhost; `scripts/dr4_practice.py` points the release-day code at that and runs it
unchanged.

```
python -m cosmos_search.gaia.dr4_fake build --out FAKE --m31 --random 60 --keep-dr3     # or --dr3 DIR on kore
python scripts/dr4_practice.py --root FAKE --out OUT --flaky 0.02
```

Result of 2026-11-10 (one cloud container, 4 cores), on the M31 field and 60 random DR3 files:

| | |
|---|---|
| Fake DR4 files | 233 files, 4.18 GB, 42,981,250 transits of 1,084,551 stars; largest file 258 MB |
| Listing | every file found with its checksum; the per-CCD folder and catalogue metadata left out |
| Download and read | 27 s for everything, no failed file with 2% of requests answered 503 (7 refused, all retried); 108 ranged requests for the large files |
| Against DR3 read directly | 119,179,938 light-curve points of 3,232,030 star-and-band pairs: same epochs, times, CCD rows, fields of view and photometry-rejection flags; fluxes equal to float32 rounding (6 × 10⁻⁸) |
| DataLink reader | 200 stars through the stand-in endpoint, equal to the files; the same reader against ESA's real DataLink for DR3: 2,849 stars, no failed call |
| First-hour file check | no problems; one note (no rejected epochs marked) |
| Memory | a worker's peak over the run 1.5 GB; on the largest file alone 1.2 GB (1.7 GB before the fix below) |

What the practice run and the checks leading to it found:

| Finding | What was done |
|---|---|
| Several release-day queries named columns DR4's `gaia_source` and `vari_classifier_result` will not have (`astrometric_params_solved`, blend counts, `phot_bp_rp_excess_factor`, `has_epoch_photometry`, `best_class_name`) | One map per release, `targets/releases.py`, used by the cuts, the checks and the figures; held to the draft data model by a test (PR #84) |
| No reader for DR4's DataLink layout, the one route ESA has confirmed | `gaia/dr4_datalink.py`, any packaging the service may choose (PR #86) |
| The file reader held a whole file in memory when every star is kept | Reads whole stars a piece at a time; a file whose stars are scattered is detected and sorted (PR #86) |
| Converting 30,000 stars at once needed 1.7 GB | Converted a slice at a time: 1.2 GB, in 40% less time (PR #86) |
| The draft table does not mark the epochs DR3's variability processing rejected (5.2% of points in the M31 field; 42% of them have an error over twice the star's median, 38% are bright outliers) | Measured on 49,902 stars of one sky pixel with DR3's thresholds: without the marks the fading list grows from 177 to 256 and 9 of its top 20 are stars absent from the list made with the marks. The scoring gets its own rule for bad epochs (`slow/drift.py`, with R1 of [plan.md](plan.md)); the reader keeps every epoch |
| ESA's TAP answered 503 to small queries while four DataLink calls were running | Run the per-star lookups and DataLink pulls one after the other on release day, or from different containers |

Not covered by a practice run: DR4's real file sizes and row order, the archive under release-day load, and a
run at the size of all of DR3 on kore.

## Scheduled start

Proposed to Cora on 2026-11-10 and created only once she agrees. Two scheduled tasks, both delivered to the
thread "Plan release-day order of work":

| When (Ontario) | What runs |
|---|---|
| 2026-12-01 09:00, once | The day-before list above: tests on `main`, the practice run on a small fake set, archive login, kore reachable. Reports anything not ready |
| 2026-12-02 from 05:00, each hour | First-hour step 1. While DR4 is not out, nothing else happens. Once a route answers: steps 2 to 8, a message to Cora with the first light curve of Boyajian's star, then the run. The hourly task is removed at that point; if DR4 is not out by 2026-12-03 06:00 it drops to twice a day |

kore takes part once Cora writes the start message in a thread on kore; the cloud containers do not wait for it.
