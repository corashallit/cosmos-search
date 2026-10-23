# Storage

## What we have

| Store | Size | Lifetime | Use |
|---|---|---|---|
| Container local disk | ~30 GB free | Gone when the session container is reclaimed | Scratch and cache only |
| GitHub repo | keep < 1 GB, files < 100 MB | Durable | Code, docs, configs, small summary tables |
| Project file share (`/mnt/project-files/`) | Reports 1 PB, network mount | Durable across threads | Candidate durable store; slow for random access, fine for tile-sized Parquet |
| kore (Cora's Mac), `~/cosmos-data` | 926 GB disk | Durable | Local copies of the external data the checks read (below), the Gaia DR3 epoch files, and DR4 working space |

## Estimates for this track

Per-source sizes are DR4 totals divided by 1.8 × 10⁹ sources.

| Product | Sources | Size |
|---|---|---|
| DR3 epoch photometry, all variables (bulk CDN files) | 1.175 × 10⁷ | 74 GB transfer; stream to features (~1–2 GB kept) |
| DR3 `vari_summary` subset | ~10⁷ | ~2 GB |
| DR4 epoch photometry, full bulk stream | 1.8 × 10⁹ | 7.2 TB transfer, not kept. Kept: feature rows above a loose cut (~10⁷ × ~100 B ≈ 1 GB), random full light curves (~10⁶ × ~4 kB ≈ 4 GB), per-stratum null tail sketches (< 1 GB) |
| DR4 per-CCD G via DataLink (~13 kB/source) | ~10⁴ | ~0.1 GB |
| DR4 epoch astrometry via DataLink (~34 kB/source; 62 TB in total, too large to stream) | ~10⁴ | ~0.3 GB |
| WISE / 2MASS / SIMBAD cross-match rows | ~10⁶ | ~1 GB |
| TESS FFI cutouts | streamed per target | features only, < 1 GB |
| Vetting packets, injections, run manifests | — | < 5 GB |
| **Durable total** | | **~20–30 GB** |

The DR3 rehearsal and the DR4 bulk stream both pass through local disk one file at a time without a durable store; only their outputs need one. The durable store
matters once injected products from the truth repo need to reach the analysis side (milestone V3) and for
keeping DR4 pulls across sessions.

## Ask

**~50–100 GB of durable storage** before DR4, which leaves room for two pipeline versions and a larger DR4
pull if the prefilter needs widening. Options, in order of preference: the project file share; a cloud
bucket (the container has `gsutil`/`gcloud`); Zenodo for frozen releases.

## Local data store (kore)

Archives go down (SkyBoT and Miriade were both returning 502 on 2026-10-23; IRSA's ZTF service takes ~20 s a
call; data.lsdb.io serves ~2 MB/s per connection), so on kore the external data the cross-match and vetting read
live on local disk and the code reads them first. `cosmos_search.net.local.data_dir()` is the root:
`COSMOS_DATA_DIR`, else `~/cosmos-data` when it exists. Containers have neither and behave as before.

| What | Where under `~/cosmos-data` | Size on disk | Read by | Built by |
|---|---|---|---|---|
| Gaia DR3 `gaia_source`, the 34 columns the matcher and `dayone.rank.context` read (positions, astrometry and its correlations, G/BP/RP, RUWE, IPD, blending counts) | `mirror/hats/gaia_dr3` | ~190 GB | `xmatch` neighbours and blending (`HatsSource`) | `scripts/mirror_hats.py` from MAST's S3 copy |
| TIC v8.2, matcher columns | `mirror/hats/tic` | ~60 GB | `xmatch` | same, MAST S3 |
| 2MASS PSC, AllWISE (columns before `use_src` only, see data-sources.md), eROSITA-DE DR2 (all columns), VSX 2026-06 (all columns), ZTF DR22 object index (positions, ids, filter, field, CCD, epochs; no light curves) | `mirror/hats/<survey>` | ~10, ~25, 1.3, 0.5, ~45 GB | `xmatch`; VSX also `slow.known` | same, data.lsdb.io |
| SIMBAD `basic`, ASAS-SN variables (VizieR II/366), matcher columns | `mirror/hats/simbad`, `mirror/hats/asassn_vars` | ~2 GB | `xmatch` (otherwise TAP-only) | `scripts/mirror_tap.py` |
| Asteroid ephemeris: every MPCORB orbit (1.58 M) integrated with ASSIST (DE440 + 16 asteroids), states every 20 d 2014-06 to 2027-01, plus hourly Sun, Earth, Gaia and TESS positions | `mirror/sso/ephemeris` | ~17 GB | `vetting.sso` (first; SkyBoT/Miriade as fallback) | `scripts/build_sso.py` |
| Response cache: TAP results, ZTF light curves (IRSA service), TESScut sector lists, TGLC/QLP light curves, DataLink pulls made with `cache=True` | `cache/http` | grows to 100 GB (LRU) | everything through `cosmos_search.net` | filled as runs go |
| Gaia DR3 epoch photometry bulk files | `gaia-dr3-epoch-raw` | 73 GB | DR3 rehearsal stream | `~/cosmos-data/cache-tools/fetch_gaia_epoch_raw.py` |

How the code chooses: `xmatch.mirror.local(name, columns)` returns a mirror when its `mirror.json` exists and it
holds every column asked for; otherwise the read goes remote exactly as before. Mirrors are byte-for-byte the
same values (columns rewritten with zstd and byte-stream-split floats, which is lossless): cones read from them
were compared with the remote catalogues row for row. `COSMOS_MIRROR=off` turns the mirrors off;
`COSMOS_SSO=remote` sends the asteroid check to SkyBoT. Under `COSMOS_BLIND` every mirror and cache read is
refused, as remote reads are.

Not cached, and why:

- **NEOWISE single exposures** (3.7 TB HATS; the matcher's 17 columns would still be ~500 GB) and **ZTF light
  curves** (7.9 TB): per-star reads instead, kept in the response cache (`dayone.external.ztf`).
- **CatWISE2020 and unWISE** (~2 × 10⁹ rows each, IRSA TAP only, no bulk Parquet): cone queries, kept in the
  response cache.
- **Gaia DR3 `l`, `b`, `ecl_lat`** (46 GB): computed from ra/dec when needed. The rest of `gaia_source`
  (~570 GB) is not used.
- **DR4 tables**: not released until 2026-12-02. DR4 epoch photometry (7.2 TB) streams through, as planned.

Refresh: rerun the builder; each skips what is already complete. MPCORB ages (new discoveries, improved orbits),
so rebuild the asteroid ephemeris before a DR4 vetting run (`scripts/build_sso.py` elements, integrate, observers; ~2 h).
Budget: the mirrors and ephemeris take ~350 GB of kore's 926 GB, leaving ~400 GB for DR4 working data.
