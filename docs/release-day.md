# Release day: order of work

Gaia DR4 becomes public on **2026-12-02 at about 12:00 CET** (11:00 UTC, 06:00 in Ontario). ESA's archive is
closed from 2026-11-25 12:00 CET until then. This page is the order of work for that day: what must be in place
before, the checks of the first hour, the run, and what to do when DR4 differs from what the code expects. Steps
are in dependency order; the only dates are ESA's. Nothing starts by itself: the day's work begins when Cora
writes in a thread (last section).

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
| Known answers matched to Gaia (Boyajian's star, fading alerts, R CrB / UX Ori / DY Per stars) | Built 2026-11-11 on the AIP copy of the DR3 source table (`python -m cosmos_search.slow known --out known.parquet --mirror aip`; ESA's synchronous queries were refused for an hour): 985 stars, 967 matched. `/mnt/project-files/release-day/known.parquet`; the same table is in the fallback fetch list `/mnt/project-files/score-every-star/fallback/fallback_dr3.parquet` |
| DR3 per-star table (G, BP-RP, classes) for the 11.6 M DR3 stars with light curves | `/mnt/project-files/data/gaia-dr3-epoch/aux/` |
| All DR3 epoch photometry files | On kore (`~/cosmos-data`, 72.8 GB) |

## The day before

1. `main` is green and `python scripts/dr4_practice.py` passes in a fresh container (below).
2. `GAIA_USER` and `GAIA_PASSWORD` are set in the environment; the first archive call on release day logs in.
3. kore is on, awake and on mains power, with `~/cosmos-data/cosmos-search` at `main` and its environment
   reinstalled from it (`uv pip install -e '.[dev,plots]'`; on 2026-11-11 it lacked `healpy`, which only the
   population code imports).

## First hour

Each step gates the ones below it. A step that reports `problems` stops the run until fixed; `notes` are read and
the run goes on.

| # | Check | Command | If it fails |
|---|---|---|---|
| 1 | Is DR4 out, and by which routes | `python -m cosmos_search.gaia.dr4_routes --out routes.json` (exit code 3: no route answers yet) | Run it again later; nothing is scheduled |
| 2 | Tables and columns the queries name | `columns_missing` in `routes.json`, or `python -m cosmos_search.targets check --release dr4` | Edit `RELEASES["dr4"]` in `targets/releases.py` (one map for every query), re-run |
| 3 | One bulk file, smallest then largest | `python -m cosmos_search.gaia.dr4_bulk list`, then `python -m cosmos_search.gaia.dr4_check file KEY` | Column names: `ALIASES` / `COLUMNS` in `gaia/dr4.py`. Times outside DR4's span: `gaia_time_to_btjd`. Stars not contiguous: the reader sorts each file in memory, so lower `--workers` |
| 4 | First light curves through DataLink: Boyajian's star, the known answers, the DR3 reference stars | `python -m cosmos_search.gaia.dr4_datalink --ids LIST --out DL` | The response layout is new: `read_response` in `gaia/dr4_datalink.py` names the columns it found |
| 5 | DR4 against DR3 for the same stars | `python -m cosmos_search.gaia.dr4_check compare --dr4 DL/lightcurves --dr3 /mnt/project-files/release-day/dr3-reference/random/lightcurves.parquet` | No star in both: map ids through `dr3_neighbourhood` (`--id-map`). Flux scale or time offset: units or zero points differ from DR3's; fix the reader before anything is scored |
| 6 | Rejected epochs | Does `routes.json` list a `vari_epoch_flags` folder among the bulk files? Does step 3 report a rejection column? | Neither is expected. The scoring uses its own rule for bad epochs (below) |
| 7 | Class names | `SELECT phot_variable_best_class, COUNT(*) FROM gaiadr4.all_source_flags GROUP BY 1` | Update the eclipsing-binary class given to `slow_checks.py --skip-class` |
| 8 | Trend zero point | After the threshold sample is scored: `trend_offset_by_G` in `L/summary.json` (median trend in units of its error, by G magnitude), and the same by sky region | An offset at G < 14 was +0.85 errors in DR3 and would be about +2.3 at DR4's precision; it has to be removed before thresholds are set |

Steps 1 to 3 need no science decision and take what the archive's load allows. Step 4 is the first data anyone
on the project sees: Boyajian's star has no light curve in DR3 and gets one in DR4.

## The run

By route, fastest first (`routes.json` says which exist):

1. **Bulk files.** Thresholds from a random share of the files, then every star scored as the files pass, split
   over machines with `--shard i/n`; each raw file is deleted once read:

   ```
   python scripts/slow_search.py pass --out S --dr4-bulk --files-share 0.002 --keep-lightcurves all
   python scripts/slow_release_thresholds.py --lightcurves S/lightcurves --out RULE --check-against prereg/rules/slow-v3
   python scripts/slow_search.py pass --out R --rule RULE --dr4-bulk [--shard i/n] [--durable DIR]
   python scripts/slow_search.py flag --run R --rule RULE --out L
   python scripts/slow_search.py cuts --run L
   ```

   Before the scoring pass, read `RULE/release.json`: the sample has to hold at least `stars_needed_for_20pct`
   quiet stars (25,000). With fewer, the thresholds' intervals are wide; raise `--files-share` and repeat the
   first two commands.

2. **A mirror's table, no bulk files**, or **DataLink only**: `scripts/slow_fallback.py` gives the order in which
   to pull light curves (the watch-list, the known answers, then the archive's own statistics), at 0.1 to 2.0
   light curves a second through DataLink (2,849 DR3 watch-list stars took 24 min on 2026-11-10, no failed call);
   check the other routes daily.

Then the reading of the lists, the checks per candidate, figures and the blind review:

```
python scripts/slow_checks.py --fading L/fading.parquet --lightcurves R/lightcurves --out L/checks --skip-still --skip-band --skip-class ECL --n 1000
python scripts/slow_review.py files --checks L/checks --fading L/fading.parquet --lightcurves R/lightcurves --rule RULE --hosts S/lightcurves --out D
python scripts/slow_review.py launch --dir D          # arguments for scripts/stage_a_reviews.js (Workflow tool)
python scripts/stage_a_audit.py --transcripts <session>/subagents --prompts D/prompts --replies D/replies --out D/audit.json --reviewer prereg/reviewers/slow-cal-1.json
python -m cosmos_search.vetting commit --packets D/products.json --replies D/replies --vetter slow-cal-1 --batch D/batch --archive prereg/verdicts/<name> --audit D/audit.json --plan D/plan.json.age
git push
python scripts/slow_review.py table --dir D --checks L/checks        # D/reviewed.csv, the list to read; D/fakes.json
```

The limit on how common steady fades are among FGK dwarfs, which is reported with or without a find
([arms/occurrence.md](arms/occurrence.md), "On DR4": first on the threshold sample, then on every star):

```
python scripts/slow_occurrence.py aux --sky 4 --schema gaiadr4 --out A        # the dwarfs' parallaxes, in sky slices
python scripts/slow_occurrence.py inject --lightcurves R/lightcurves --random-share 0.001 --aux A --rule RULE --out O
python scripts/slow_occurrence.py limit --stars R/summary --reference L/reference.parquet --aux A --injections O/injections --out O/limit
```

Each step is described in [arms/slow-changes.md](arms/slow-changes.md), "Release day, end to end".

**Planted fades.** `files` puts fakes of known answer among the stars' review files: 30, or 5% of the stars
where that is more, in quiet stars of the thresholds' sample (`--hosts S/lightcurves`, so keep S until the
review files are built). Which files are fakes is written only encrypted, and `table` opens it once the verdicts
are pushed. `D/fakes.json` then says how many the reviewer shortlisted. At least 20 of 30 (34 of 50) is
consistent with the calibration. With fewer, no star is set aside on that batch's verdicts: the list is read in
the checks' order with the verdict as a column ([vetting.md](vetting.md) §16). The review files hold nothing
from the other surveys the checks read, where a fake's host would be seen unchanged, and the eclipsing-binary
check stays beside each verdict in the table.

**Launching the reviews.** The launcher hands every reviewer, before its task, the last message Cora typed in
the thread that launches them, when that thread's context holds one (found in the practice of 2026-11-10,
below; [vetting.md](vetting.md) §15, second amendment). A reviewer that has read a message about the release is
not the reviewer the calibration measured, and its review does not count. A message that reached the thread
through the coordinator is not handed on. So on release day:

1. Launch the reviews from a thread Cora has not typed in. The thread doing the day's work, which starts when
   Cora writes in it, asks the coordinator for a separate thread for the reviews. That thread needs D's
   `prompts/` and an empty `replies/` where it can reach them (D built under `/mnt/project-files/`, or those two
   copied there and the replies copied back). It also runs the audit, because the reviewers' transcripts are in
   the session that launched them, and hands back `audit.json` with the replies. It never needs the key.
2. `scripts/stage_a_reviews.js` checks first: one small subagent reports whether anything precedes its task, and
   if something does the script stops with no review launched. Then launch from another thread, or from this
   one after its context has been summarized (in the practice, a launch after that was handed the task only).
3. The audit reads every transcript whatever the check said, and a review that was handed more than its task
   does not count. If the audit and the check disagree, launch one review, audit it, and launch the rest only
   when it counts. A wrong launch costs reviews, never a wrong verdict in the batch.

Machines:

| Machine | What it has | Use |
|---|---|---|
| Cloud container (one per thread) | 4 cores, 15 GB memory, about 30 GB disk, 120 MB/s from ESA's CDN | The first pass, several containers each with its own `--shard i/n` |
| kore | 10 cores, 16 GB memory, 171 GiB free disk on 2026-11-11 (263 GiB once the practice files under `~/cosmos-data/practice-dr4/` are deleted) | One or more shards with 6 workers; where the kept light curves and summaries are stored. A session there starts when Cora says so in the thread that needs it |
| `/mnt/project-files/` | Shared by every thread, durable | Rules, thresholds, per-star summaries, lists, `routes.json`, first-hour reports |

Sizing from the practice runs (below): reading and converting takes 1.9 core-seconds per million transits in a
cloud container and 1.2 on kore, so DR4's 1.2 × 10¹¹ transits cost about 64 core-hours in the cloud, or 40 on
kore, before any scoring; the download is 7.2 TB, 17 h for one container's network. A worker needs about 1.2 GB
of memory per 1.4 million transits it holds (30,000 stars at DR3's sampling, about 15,000 at DR4's); six workers
on kore peaked at 1.33 GB each over all of DR3. Download, reading and scoring together cost 10 core-seconds per million
transits on kore (all of DR3 in 15.5 min with 6 workers), which is about 330 core-hours for DR4. That is an upper
figure: DR3's stars were selected as variable, and 14% of them got the full scoring against 4% of
the Andromeda field's ordinary stars.

## Practice run

`gaia/dr4_fake.py` builds files in DR4's expected layout from DR3 bulk files and serves them, a file listing and a
DataLink endpoint on localhost; `scripts/dr4_practice.py` points the release-day code at that and runs it
unchanged.

```
python -m cosmos_search.gaia.dr4_fake build --out FAKE --m31 --random 60 --keep-dr3     # or --dr3 DIR on kore
python scripts/dr4_practice.py --root FAKE --out OUT --flaky 0.02
```

On a machine that holds DR3's files (kore), all of DR3:

```
python -m cosmos_search.gaia.dr4_fake build --out FAKE --dr3 DIR --workers 6 [--resume]
python scripts/dr4_practice.py --root FAKE --out OUT --dr3 DIR --flaky 0.02 --workers 6
```

**All of DR3 on kore (2026-11-10).** Every DR3 light curve, through the release-day reader, compared with DR3
read directly:

| | |
|---|---|
| Fake DR4 files | 12,288 files (every order-5 pixel), 54.3 GB, 555,868,797 transits of 11,754,237 stars, from DR3's 3,386 files (72.8 GB); largest file 322 MB. Built in 19.7 min with 6 workers |
| Download and read | 189 s for everything with 6 workers, no failed file with 2% of requests answered 503 (254 refused, all retried); 837,000 transits per core-second |
| Against DR3 read directly | 1,554,036,289 light-curve points of 35,222,072 star-and-band pairs: all equal (times identical, fluxes to float32 rounding, 6 × 10⁻⁸). Reading DR3 itself took 704 s |
| DataLink reader | 200 stars through the stand-in endpoint, equal to the files |
| First-hour file check | no problems on the smallest and the largest file; one note (no rejected epochs marked) |
| Memory | a worker's peak 1.33 GB; the build's last step 2.7 GB in one process; no swapping |
| Disk | 52 GB of fake files and 34 GB of output under `~/cosmos-data/practice-dr4/`, with the scoring practice's 6 GB; all of it is deleted once both practices are recorded (Cora's choice, 2026-11-11), and the fake files can be built again in 20 min |

Nothing had to be changed for macOS. kore's checkout was on the branch `kore-mirror-verify-props`, not `main`;
the practice ran from a separate working copy.

**M31 field and 60 random DR3 files (2026-11-10, one cloud container, 4 cores):**

| | |
|---|---|
| Fake DR4 files | 233 files, 4.18 GB, 42,981,250 transits of 1,084,551 stars; largest file 258 MB |
| Listing | every file found with its checksum; the per-CCD folder and catalogue metadata left out |
| Download and read | 27 s for everything, no failed file with 2% of requests answered 503 (7 refused, all retried); 108 ranged requests for the large files |
| Against DR3 read directly | 119,179,938 light-curve points of 3,232,030 star-and-band pairs: same epochs, times, CCD rows, fields of view and photometry-rejection flags; fluxes equal to float32 rounding (6 × 10⁻⁸) |
| DataLink reader | 200 stars through the stand-in endpoint, equal to the files; the same reader against ESA's real DataLink for DR3: 2,849 stars, no failed call |
| First-hour file check | no problems; one note (no rejected epochs marked) |
| Memory | a worker's peak over the run 1.5 GB; on the largest file alone 1.2 GB (1.7 GB before the fix below) |

**Scoring, lists, checks and review files (2026-11-10, same container, same 233 fake files through the stand-in
server).** The commands of "The run" above, with DR3's catalogue for the lookups:

| Step | Result |
|---|---|
| Sample pass, every light curve kept | 1,084,551 stars in 0.8 to 1.0 min on 4 cores, no failed file (after the fix below) |
| Thresholds | G < 19: 5.41 (episode), 3.72 (trend), 6.68 (repeat), against 5.42, 3.68, 6.77 from the scoring work's own run on the M31 field |
| Scoring pass with the thresholds | 1.4 min; 1,017,063 stars with scores |
| Lists | 16,941 stars on the fading list |
| Cuts | catalogue rows of 30,680 of 30,682 stars in 19 s; 9,957 without measurable motion |
| Checks, 12 strongest | 312 s with 6 at once, no failed fetch |
| Review files | 12 files in 2.8 min (most of it the AllWISE and 2MASS cross-match), none refused by the leak scan. Two stars fainter than G 19 had no kept light curve and were pulled through DataLink first |

**Scoring at the size of all DR3 on kore (2026-11-11).** The 12,288 fake files of the practice above through the
stand-in server (2% of requests answered 503), 6 workers, the commands of "The run" as far as the lists:

| Step | Result |
|---|---|
| Sample pass, 5% of the files, every light curve kept | 615 files, 552,224 stars in 39 s, no failed file |
| Thresholds | 42 s, 1.6 GB of memory. Stopped on the first try with a fault that only macOS shows (below); passed once fixed |
| Scoring pass | 12,288 files, no failed file (247 requests refused, all retried). 11,754,237 stars read and 11,606,315 scored in 15.5 min: 12,700 stars a second. 1,593,355 stars got the full scoring and 414,761 a period search. A worker used 1.0 to 1.4 GB |
| Lists | 110 s, 2.8 GB of memory in one process. Fading: 407,925 stars brighter than G 19 (4.5%) and 92,166 fainter |
| Trend zero point (first-hour check 8) | Median trend in units of its error, by G magnitude: between -0.18 and +0.24 before the comparison with neighbouring stars (largest at G 13 to 14), 0.003 to 0.007 after it in every magnitude bin |
| Disk | sample 1.6 GB, thresholds 0.15 GB, scoring output 4.1 GB (624,396 kept light curves 2.1 GB, statistics 1.0 GB, per-star summaries 0.9 GB), lists 0.3 GB |

The thresholds of this run say nothing about DR4's. Outside the Andromeda field DR3 has light curves only for
stars DPAC selected as variable: 3.6% of the sample were quiet stars (19,702, under the 25,000 the rule asks
for), and the thresholds at G < 19 came out at 5.55 (episode), 6.76 (trend) and 6.45 (repeat), the trend's
interval 5.66 to 7.37. The same split shows on the 233-file set: its Andromeda-field stars alone (903,292 stars,
60% quiet) give 5.41, 3.72 and 6.69, and its other 201 files alone (181,259 stars, 1.2% quiet) give 4.88, 4.47
and 7.02. DR4 has a light curve for every star, so its sample is of the first kind. The selection also makes
this run's lists long.

**Review step on real candidates (2026-11-10).** From the checks of the DR3 trial's 60 strongest fades that DPAC
does not call eclipsing binaries: 60 review files in 10.6 min (cross-match of 60 stars scattered over the sky),
none refused, none left out. Six of them were reviewed through `scripts/stage_a_reviews.js` by the adopted
reviewer: six replies in 7 min, all in the reply format, about 180,000 tokens a review. **None counts**: the audit
found the last message Cora wrote in the launching thread in every reviewer's context (below). Committed as an
uncounted practice batch (not archived), the table step listed two of the six on the shortlist and read the two
R CrB stars and the two repeating dimmers as their known kinds. On 2026-11-11, after this thread's context had been
summarized, one of the six was reviewed again the same way: the reviewer was handed its task only, the audit
counted it, `vetting commit --audit` committed that one review, and the table step listed the other five as not
reviewed.

What the practice runs and the checks leading to them found:

| Finding | What was done |
|---|---|
| Several release-day queries named columns DR4's `gaia_source` and `vari_classifier_result` will not have (`astrometric_params_solved`, blend counts, `phot_bp_rp_excess_factor`, `has_epoch_photometry`, `best_class_name`) | One map per release, `targets/releases.py`, used by the cuts, the checks and the figures; held to the draft data model by a test (PR #84) |
| No reader for DR4's DataLink layout, the one route ESA has confirmed | `gaia/dr4_datalink.py`, any packaging the service may choose (PR #86) |
| The file reader held a whole file in memory when every star is kept | Reads whole stars a piece at a time; a file whose stars are scattered is detected and sorted (PR #86) |
| Converting 30,000 stars at once needed 1.7 GB | Converted a slice at a time: 1.2 GB, in 40% less time (PR #86) |
| The draft table does not mark the epochs DR3's variability processing rejected (5.2% of points in the M31 field; 42% of them have an error over twice the star's median, 38% are bright outliers) | Measured on 49,902 stars of one sky pixel with DR3's thresholds: without the marks the fading list grows from 177 to 256 and 9 of its top 20 are stars absent from the list made with the marks. The scoring gets its own rule for bad epochs (`slow/drift.py`, with R1 of [plan.md](plan.md)); the reader keeps every epoch |
| ESA's TAP answered 503 to runs of small queries for about an hour on 2026-11-10, with and without DataLink calls running, then answered normally (30,680 rows in 19 s) | Answers are cached, so a run that stops on a 503 goes on where it stopped when started again. The refusals came back on 2026-11-11 (logged-in queries too), and the known-answers table was matched on the AIP copy of the DR3 source table instead (`--mirror aip`, 985 cone queries in 19 min) |
| Worker processes started after the first archive call shared its open connection; two of them sending byte-range requests down one socket read each other's answers, so the first large file of every scoring pass failed its checksum (and would have been read wrongly from an archive that publishes no checksums) | A worker now opens connections of its own (`net/client.py`, with a test that hangs without the fix); three passes over the 233 files, no failed file (PR #88) |
| The launcher of the blind reviews now hands every reviewer the last message Cora wrote in the launching thread, before its task. The audit looked for the task in the first message only and reported every review as missing | The audit finds such reviews and refuses them as not blind (`scripts/stage_a_audit.py`, PR #88). "Launching the reviews" above says how to launch reviews that count |
| The scoring pass keeps light curves only for stars brighter than G 19, but the fading list also holds fainter stars; the checks' eclipsing-binary test and the review files need the light curve | `slow_review.py files` lists such stars in `left_out.csv` and takes a second directory of light curves pulled through DataLink; or check only stars brighter than G 19 (`slow_checks.py --max-g 19`) |
| On macOS a worker process starts as a new interpreter and does not see what the parent put in the script's module state; the thresholds step stopped on kore with `KeyError: 'drift_share'` (scoring practice, 2026-11-11) | The thresholds, injection-test, blind-test and period-check scripts hand the state to their workers; a test runs the thresholds step with macOS's start method (PR #92) |
| The comparison with DR3 held every star's sums in one process, too much for all of DR3 on a 16 GB machine; memory figures were 1000 times too high on macOS | Compared a 48th of the sky at a time; the build can resume (PR #88) |

Not covered by a practice run: DR4's real file sizes and row order, the archive under release-day load, the
cuts, checks and review files at the size of all of DR3 (they ran on the 233-file set), and a whole batch of
reviews that the audit counts (one review did).

## Start

No scheduled task starts anything (Cora's choice, 2026-11-10). The day's work begins when Cora writes in a
thread on or after 2026-12-02; DR4 is due at about 06:00 in Ontario.

| When | What is run, by hand |
|---|---|
| The day before (2026-12-01) | The day-before list above: tests on `main`, the practice run on a small fake set, archive login, kore on and at `main` |
| Release day | First-hour step 1. While no route answers, nothing else; run it again later. Once a route answers: steps 2 to 8, a message to Cora with the first light curve of Boyajian's star, then the run |

kore takes part once Cora says so in the thread that runs the day; the cloud containers do not wait for it.
