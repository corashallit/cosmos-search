# Storage

## What we have

| Store | Size | Lifetime | Use |
|---|---|---|---|
| Container local disk | ~30 GB free | Gone when the session container is reclaimed | Scratch and cache only |
| GitHub repo | keep < 1 GB, files < 100 MB | Durable | Code, docs, configs, small summary tables |
| Project file share (`/mnt/project-files/`) | Reports 1 PB, network mount | Durable across threads | Candidate durable store; slow for random access, fine for tile-sized Parquet |

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
