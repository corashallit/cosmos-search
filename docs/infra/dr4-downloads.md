# Getting DR4's light curves

How the slow-change search (and the other Gaia searches) will get DR4 epoch photometry on and after 2026-12-02,
measured on DR3 on 2026-10-23. Code: `gaia/dr4_routes.py` (release-day check), `gaia/dr4_bulk.py` (bulk files),
`gaia/epoch.py` (DataLink; the AIP mirror reader `mirror_epoch_photometry`).

## Routes, fastest first

| Route | Measured on DR3 | DR4 status | Suits |
|---|---|---|---|
| ESA bulk files on the CDN (`dr4_bulk pull`) | 32 MB/s per stream, 122 MB/s with 8 streams from one cloud container; kore pulled DR3's 73 GB at 45 MB/s (2026-10-21) | Planned: ESA's approved note puts transit-level epoch photometry on the CDN as Parquet (below); nothing under `Gaia/gdr4/` yet | Any selection size: all 7.2 TB stream past in ~17 h per container at 122 MB/s, and only kept stars are written |
| AIP's database table (`epoch.mirror_epoch_photometry`) | ~1,500 light curves/s, ~3,000/s with 4 queries (by source_id range) | AIP served DR2's and DR3's light curves as a table; no DR4 announcement | A selection of millions: 2.5 × 10⁷ in 2-5 h |
| ESA DataLink (`epoch.datalink_epoch_photometry`) | 0.1-1.5 light curves/s whatever the request shape (below) | DataLink only for per-CCD G | ≤ ~10⁴ stars; per-CCD G for the top candidates |

## What ESA has said

- DR4 content page (last updated 2026-09-14): `epoch_photometry` 121,845,163,512 transit rows, 7.2 TB, served by
  DataLink; `epoch_photometry_ccd` 23 TB, DataLink. "The data (or a subset thereof) will also be served from our
  partner data centres (CDS, ASDC, ARI, AIP and Flatiron)", without saying which tables.
- Technical note GAIA-CZ-TN-ESA-JDB-128 "A proposal for the bulk-download repository for Gaia DR4" (issue 1.1,
  2024-10-21, status Approved; public copy on the IVOA ParquetInAstronomy wiki): the CDN serves all TAP and DataLink
  products except six large epoch products (epoch astrometry 72 TB, epoch images 166 TB, `EpochPhotometryCcd` 32 TB,
  RVS epoch spectra 61 TB, XP epoch spectra and crowding 169 TB). Transit-level `EpochPhotometry` (9 TB in the
  note's estimate) is in. Format Parquet (VOParquet, per the IVOA wiki, 2025), partitioned in an evolution of HATS,
  one Apache Iceberg table per product; CDN for at least the first 6 months. The excluded products can be had in bulk
  "on request" by email to the Gaia Helpdesk (sftp, as for affiliated data centres). DataLink requests for more than
  one product will return a list of URLs to fetch (curl, wget or Metalink), "easier (albeit not faster)".
- Archive down from 2026-11-25 12:00 CET until the release, 2026-12-02 ~12:00 CET.

Nothing newer confirms or cancels the bulk plan; the CDN's top level (`gaia.eu-1.cdn77-storage.com/?prefix=Gaia/`)
holds gdr1, gdr2, gdr3, gedr3 and gfpr only.

## Partner centres (checked 2026-10-23)

| Centre | DR3 light curves | DR4 plans found |
|---|---|---|
| AIP (gaia.aip.de, Daiquiri/PostgreSQL TAP) | `gaiadr3.epoch_photometry` and `gaiadr2.epoch_photometry` tables; async queues 30 s, 5 min, 2 h, 24 h; result download as CSV, VOTable, FITS, Parquet | none |
| ARI Heidelberg (TAP) | no epoch photometry table | none |
| CDS VizieR | no table under I/355 | none |
| Flatiron (FlatHUB) | gaia_source only | none |
| ASDC | portal only | none |

So AIP is the only partner with a record of serving light curves as a table. Whether, and when, it loads DR4's
1.2 × 10¹¹ rows is not public.

## ESA DataLink, measured 2026-10-23

Logged in, DR3, csv unless stated. The 50-id cap still holds (51 ids: HTTP 500).

| Request | Time | Notes |
|---|---|---|
| 1 id | 2.2 s | |
| 10 ids | 23 s | first byte at 2.7 s, then the body streams star by star |
| 50 ids | 122 s | ~2.4 s per star |
| 10 ids, ecsv / fits / votable | 52 / 46 / 74 s | csv is fastest |
| 10 ids, `DATAMODEL_STANDARD` | 140 s | zip of per-source files |
| 10 ids, anonymous | 118 s | |
| 4 × 10 ids in parallel | 181 s wall (0.22/s) | |
| 8 × 5 ids in parallel | 27 s wall (1.5/s) | |
| 4 × 5 ids one after another | 214 s (0.09/s) | ~21:20 UTC, busy archive |
| 4 × 5 ids in parallel, straight after | 66 s (0.30/s) | 3× the serial rate at the same moment |
| repeat: serial, then parallel | 184 s, then 87 s | 2× again |

The cost is per star on the server and swings by 10× with the archive's load. Format aside, the one shape that
helps is several small calls at once, which `dayone.datalink` now does by default (4 calls of 5 ids; `--workers`,
`--chunk`). Overall 0.1-1.5 light curves/s, 10⁴-10⁵ a day. The DR4 selection of 0.1-1% of ~2.5 × 10⁹ stars (2.5 × 10⁶ to 2.5 × 10⁷
light curves) would take months to years this way.

## How many light curves, and where they live

The slow-change selection keeps 0.1-1% of stars (`slow/select.py`; recall in docs/arms/slow-changes.md): 2.5 × 10⁶
to 2.5 × 10⁷ light curves. In the shared schema (`epoch.SCHEMA`, zstd) DR3 light curves take ~3.9 kB a star, so
10-100 GB, plus ~10 GB for a 0.1% random reference sample. That fits on kore, even with the HATS mirror there.
The whole release does not fit anywhere we have, and nothing stages it: `dr4_bulk pull` downloads one raw file per
worker into WORK/tmp, keeps the chosen stars' transits, writes their light curves and deletes the raw file (also when
reading fails). Per-star summary statistics for every star come from ESA's `all_source_photometry` on TAP (755 GB as
a table), so none are written here either.

Peak disk per machine is about `--workers` × the largest raw file, plus the kept light curves of its shard. DR4 file
sizes are not known yet; DR3's epoch files reached 2.2 GB and ESA's note aims for even sizes, so 8 workers need
≲ 20 GB of scratch. Before each download the streamer checks for the file's size plus 2 GB free and otherwise fails
that file, which the next run retries. On kore, set `--out` inside ~/cosmos-data so scratch lands on the big disk.

Whether every star should be screened as the files pass (the detector run per file, with only its flags and the
selected light curves kept) is a choice for the slow-change search; the API's `keep_all` exists for that hook and has
no command-line flag, so a pull cannot write the whole release by accident.

Processing a DR4-shaped Parquet file (built from a DR3 bulk file: 286,289 transits, 21.8 MB): converting every
transit to `epoch.SCHEMA` runs at ~15 MB/s per core; keeping a 1% share runs at ~70 MB/s per core (mostly reading).
So a 4-core container keeping a selection is limited by its network (~120 MB/s), and one converting everything by CPU
(~60 MB/s) plus whatever the detector costs.

## Release-day steps

1. `python -m cosmos_search.gaia.dr4_routes --out routes.json` once the archive reopens: lists DR4 Parquet under
   `Gaia/gdr4/` on the CDN, `epoch_photometry` tables at ESA, AIP and ARI, and times one DataLink call; prints which
   route to use.
2. Bulk files up: `python -m cosmos_search.gaia.dr4_bulk list`, read one file and check its columns against
   `gaia.dr4.COLUMNS`, then `dr4_bulk pull --out WORK --durable DEST --select selection.parquet --random-share 0.001
   --shard i/n` on kore and on several cloud containers. Resumes from done-markers; raw files are deleted as read.
3. No bulk files, AIP table up: whole-sky selection on ESA TAP (`slow/select.py`), light curves from AIP by
   source_id range.
4. Neither: DataLink for the top of the selection, and re-check the CDN and AIP daily.
5. Per-CCD G (DataLink only, not on the CDN): top ~10⁴ candidates through DataLink. For many more, the Gaia
   Helpdesk can grant bulk access on request.

The dry run on DR3 (`--release "Gaia DR3" --schema gaiadr3 --cdn-prefix Gaia/gdr3/Photometry/`) finds AIP's table,
no Parquet on the CDN (DR3's files are csv.gz) and DataLink at 0.66 light curves/s.
