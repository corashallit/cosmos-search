# Storage

## What we have

| Store | Size | Lifetime | Use |
|---|---|---|---|
| Container local disk | ~30 GB free | Gone when the session container is reclaimed | Scratch and cache only |
| GitHub repo | keep < 1 GB, files < 100 MB | Durable | Code, docs, configs, small summary tables, encrypted truth |
| Project file share (`/mnt/project-files/`) | Reports 1 PB, network mount | Durable across threads | Candidate for durable Parquet products. Slow for random access; fine for tile-sized files. Whether it is intended for ~100 GB of data is Cora's call. |

## Estimates

Per-object sizes are from DR3 schema widths; counts are from live queries (README).

### DR3 rehearsal

| Product | Objects | Size if kept |
|---|---|---|
| Gaia DR3 source + AP rows, ϖ > 8 mas | 1.2 × 10⁶ | ~2 GB |
| XP continuous coefficients + correlations | 2.5 × 10⁵ | ~3 GB |
| RVS spectra, epoch photometry | 2–3 × 10⁴ | < 1 GB |
| Cross-match tables (2MASS, WISE ×3, eROSITA, VLASS, RACS, PS1, GALEX) | 10⁶ | ~1–2 GB |
| NEOWISE single exposures | 10⁶ × ~250 | ~25 GB raw → features ~0.2 GB |
| ZTF light curves | ~4 × 10⁵ | ~30 GB raw → features ~0.2 GB |
| TESS light curves (all sectors, FFI + 2-min) | ~3 × 10⁵ | ~1–2 TB raw → features ~1 GB |
| ASAS-SN light curves | ~10⁵ | ~5 GB raw → features < 0.1 GB |
| Spectra (LAMOST, SDSS, DESI) | ~10⁵ | ~10 GB raw → keep ~2 GB |
| Model grids (synthetic photometry) | — | ~5 GB |
| **Durable total if light curves are streamed** | | **~15–25 GB** |

TESS is the outlier: ~1–2 TB of transfer even though we keep almost none of it. Reading from MAST's public
S3 bucket is the efficient route. At a sustained 50 MB/s that is 6–12 hours of wall time, which is fine,
but it is the one stage where a cache cannot fit locally, so it must run fetch → features → discard per tile.

### DR4

Gaia's DR4 product list gives totals for 1.8 × 10⁹ sources; the 100 pc buffer is ~1.2 × 10⁶ of those
(~7 × 10⁻⁴). Scaling the per-source products:

| DR4 product | Full release | 100 pc buffer share |
|---|---|---|
| `epoch_astrometry` | 62 TB | ~40 GB |
| `epoch_photometry` | 7.2 TB | ~5 GB |
| `xp_epoch_spectrum` | 119 TB | ~80 GB |
| `rvs_epoch_spectrum` | 49 TB | ~10–40 GB (nearby stars are over-represented among RVS targets) |
| `xp_continuous_mean_spectrum` | 22 TB | ~15 GB |
| `epoch_image` | 113 TB | skip |
| TAP summary tables (NSS, variability, AP) | ~0.5 TB | ~1 GB |

That is ~150–180 GB if pulled whole, or ~100 GB without epoch XP spectra (which can be streamed to features
like TESS). Each DataLink call takes ≤ 5,000 sources, so ~250 calls per product; partner data centres (CDS, ARI,
AIP, Flatiron) will mirror DR4, which spreads the load.

## Ask

Provision **~250 GB of durable storage** before 2 December. That covers the DR3 products (~25 GB) and DR4
without epoch XP spectra (~100 GB), with room for two pipeline versions side by side. Options, in
order of preference:

1. The project file share, if Cora is comfortable putting ~250 GB there.
2. A cloud bucket (the container already has `gsutil`/`gcloud`), with credentials as an environment secret.
3. Zenodo for frozen releases (50 GB per record), alongside one of the above for working data.

Nothing larger is needed unless we decide to keep raw TESS or DR4 epoch XP spectra.
