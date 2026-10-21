# Gaia epoch photometry (`cosmos_search.gaia`)

Milestone I2: Gaia DR3 epoch photometry streamed to per-source features, plus the reader that DR4 will reuse.

## Access routes

| Route | Use | Notes |
|---|---|---|
| Bulk CDN (`gaia.dr3_pass`) | All 11,754,237 DR3 sources with epoch photometry | 3,386 `csv.gz` files, 74 GB, named by HEALPix-8 range, MD5-checked. Median file 16 MB, largest 2.2 GB (Galactic bulge), so files are parsed in 2,000-source chunks |
| DataLink (`epoch.datalink_epoch_photometry`) | Targeted pulls, ≤ 50 sources per call | `DATA_STRUCTURE=DATAMODEL_GAIA` (the old `INDIVIDUAL`/`RAW` now return 500). The archive rejects more than 50 IDs per call since the 2026-10-14 release. Measured 2026-10-21: ~11 s per call plus ~1.1 s per source, and 4 parallel calls were no faster than one (1.5 sources/s overall). That is ~2 h for 10⁴ sources and ~8 days for 10⁶, so DataLink is for short lists |

Both go through `cosmos_search.net` and produce the same table.

## Light-curve table

One row per observation, sorted by (source_id, band, t). Shared with the TESS arm, whose detector reads the
first six columns.

| Column | Type | Meaning |
|---|---|---|
| `source_id` | int64 | Gaia DR3 source_id |
| `band` | dictionary string | `G`, `BP`, `RP` |
| `t` | float64 | BTJD = BJD_TDB − 2457000. Gaia publishes BJD_TCB − 2455197.5; the TCB→TDB rate term (≈ 18.6 s in 2015) is applied |
| `flux`, `err` | float64 | e⁻ s⁻¹ as published |
| `flags` | int32 | bit 1: DPAC variability reject (or negative flux); bit 2: BP/RP photometry reject; bit 4: G noisy across CCDs |
| `transit_id` | int64 | G, BP and RP of one transit share it |
| `fov` | int8 | 1 preceding, 2 following field of view, decoded from `transit_id` |
| `ccd_row` | int8 | 1–7, decoded from `transit_id` |
| `n_obs` | int8 | CCDs in the transit-averaged G (0 for BP/RP) |

Observations with no published flux are dropped rather than flagged.

## Features (`features.source_features`, version `dr3-features-v0`)

Ingredients for the detectors in docs/arms/gaia-dimmers.md, one row per source. Design choices, and why:

- **Robust scale.** Deviations are in units of 1.4826 × MAD of the good G fluxes (`z_rob`). A standard
  deviation would be inflated by the very dip we want, which is how a deep, rare dip hides itself. The
  published errors are kept too (`z_err`, `err_med_g`), and their ratio `excess_g` separates stars that
  vary from stars that are measured badly.
- **Every faint-side statistic has a bright-side mirror** (`min_*`/`max_*`, `n_low*`/`n_high*`,
  `partner_agree_low`/`_high`). Instrument noise is close to symmetric, an occulter is not, so the bright
  side gives an empirical null per stratum (magnitude, colour, sky region) without a noise model.
- **Gaia's redundancy.** For the deepest (and brightest) point: the z of its partner transit in the other
  field of view, ~106.5 min later (`min_partner_z`), and the same-transit BP and RP fractional deviations
  (`min_frac_bp`, `min_frac_rp`) for dip colour. `partner_agree_low` is the fraction of the source's faint
  3σ points whose partner is also below −2σ. A real dip lasting hours should appear in both fields of view;
  a cosmic ray, a CCD defect or a FoV-dependent calibration error should not.
- **DPAC rejections are tested, not trusted.** Rejected points are excluded from the scale but counted when
  they are faint outliers (`n_rej_low3`). A rejection rule tuned on ordinary variables can discard a real
  deep dip.
- **Episodes, not points.** `n_low_episodes` groups faint 3σ points separated by less than a day, so a
  pair of FoV transits counts once.
- **Trend.** OLS slope of G (fraction per year) on points within 5σ, so dips don't drive it. Its formal
  error is too small for most DR3 variables (correlated variability), so it is a ranking ingredient, not a
  selection cut.
- `control` marks a deterministic 1% random subset (splitmix64 of source_id with a public salt). These are
  controls, not injections, so the salt being public does not affect blinding.

## What the DR3 pass keeps

Features for every source. Full light curves only where `keep_light_curve` is true: `min_z_rob ≤ −5`,
`n_low3 ≥ 3` or `n_rej_low3 ≥ 1`, the mirrored bright-side criteria, or `control`. That is ~15–25% of
sources, because the DR3 sample is already DPAC's variables. Fractional-depth cuts are not used: at G ≈ 20,
5% scatter is normal, and they kept ~80% of sources in a trial file.

Per file, the pass writes features, kept light curves and (optionally) a JSON sketch. With `--durable DIR`
they are copied to `DIR/{features,lightcurves,sketches}/` and a done-marker is written to `DIR/done/`, after
which local copies are deleted; files with a done-marker are skipped, so a reclaimed container or another
session resumes where the last stopped. `--shard i/n` takes the files whose CRC-32 name hash is i mod n,
so n sessions split the list without coordinating. `--feature-cut` keeps only feature rows matching a
pandas query (default: all). Light-curve Parquet metadata records the control fraction and salt, so the
random controls' inclusion probability (0.01) travels with the data.

```
python -m cosmos_search.gaia.dr3_pass --out /tmp/work --durable /mnt/project-files/data/gaia-dr3-epoch --shard 0/2 --workers 5
python -m cosmos_search.gaia.dr3_pass --durable /mnt/project-files/data/gaia-dr3-epoch --out x --consolidate /mnt/project-files/data/gaia-dr3-epoch/merged
```

`cosmos_search.gaia.strata` tabulates the rate of a feature-defined class (default: quiet except for a
DPAC-rejected faint point) by magnitude, colour and crowding, for sizing a stratified random pull.

The pass is one process per file and mostly CPU-bound: on this 4-core container, 5 workers ran at ~70% CPU
and ~29 MB/s of compressed input. More cores, or more sessions via `--shard`, is where speed comes from.
