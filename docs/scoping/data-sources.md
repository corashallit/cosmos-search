# Data sources

The current target list. Checked 2026-10-21. Rate limits are what each service documents; where nothing is documented, we follow
the polite-client defaults at the bottom. Items marked *(verify)* came from secondary or older pages.

## Contact email

Most of these services do not ask bots to register; the convention is a descriptive User-Agent with a
contact. Every request we make carries:

```
User-Agent: cosmos-search/<version> (100 pc census; mailto:cora.shallit@gmail.com)
```

Services that need an account tied to an email are listed under **Auth**. Account creation involves
accepting terms and setting a password, so Cora needs to do it; credentials go into environment secrets,
never into the repo.

## Core

| Source | Release to use | Role | Coverage | Access | Auth | Limits / policy |
|---|---|---|---|---|---|---|
| **Gaia** | DR3 now; DR4 on 2026-12-02 | Astrometry, G/BP/RP, XP and RVS spectra, NSS, variability, AP | All sky, G ≲ 20.7 | ESA TAP+ (`gea.esac.esa.int/tap-server/tap`), DataLink; HATS copy at `data.lsdb.io/hats/gaia_dr3` | Account in env vars `GAIA_USER` / `GAIA_PASSWORD` (cookie login via `Gaia.login`) | Anonymous: async 3 × 10⁶ rows, 90 min; registered: unlimited rows, 120 min, 20 GB job space; DataLink ≤ 5,000 sources per call *(verify; from the older FAQ)*. Since the 2026-10-14 archive release: **≤ 5 jobs executing in parallel per user, queue of 50, excess rejected**; DataLink structures renamed RAW → `DATAMODEL_GAIA`, INDIVIDUAL → `DATAMODEL_STANDARD`; DataLink limits changed (see the Archive user guide). Bulk catalogue download explicitly supported. |
| **Hipparcos** (van Leeuwen 2007) | I/311 | Astrometry for G ≲ 3 stars Gaia handles poorly | All sky, bright | VizieR TAP | none | CDS limits (below) |
| **2MASS** PSC | final | JHK_s, epoch ~1998–2001 | All sky | IRSA TAP `fp_psc`; Gaia `tmass_psc_xsc_best_neighbour`; HATS `two_mass` | none | IRSA: none published; TAP async, avoid >2 GB per sync query |
| **AllWISE** | final | W1–W4 | All sky | IRSA TAP `allwise_p3as_psd` | none | as IRSA |
| **CatWISE2020** | final | W1/W2 + proper motions; finds brown dwarfs Gaia misses | All sky | IRSA `catwise_2020`, VizieR II/365, Data Lab `catwise2020.main` | none | as host |
| **unWISE** | 2019 catalogue (neo7 coadds) | Deeper W1/W2, crowded fields | All sky | IRSA `unwise_2019`, Data Lab `unwise_dr1` | none | as host |
| **NEOWISE-R** single exposure | final (mission ended 2024) | W1/W2 time series 2014–2024 | All sky, ~2 visits/yr | IRSA TAP `neowiser_p1bs_psd`; HEALPix Parquet on S3 `nasa-irsa-wise/wise/neowiser/catalogs/p1bs_psd/healpix_k5/` | none | Catalog engine truncates at 2 GB; use S3 Parquet for bulk |
| **unTimely** (Meisner et al. 2023) | v1 | Deep W1/W2 time series, ~16 epochs, 2010–2020 | All sky | IRSA (DOI 10.26131/IRSA580), ~4 TB total | none | bulk files; we read by sky tile |
| **TESS** | Sectors to date | Light curves: SPOC 2-min/20-s, QLP, TGLC, TESS-SPOC FFI | ~All sky, bright to T ≈ 16 | MAST (`astroquery.mast`), TESScut; public S3 `stpubdata/tess/public` incl. HATS TIC | none for public data | TESScut ≤ 10 simultaneous calls per user (503 above). TICA cutouts discontinued Aug 2025. |
| **ZTF** | DR24 (2026-01-22) | g/r/i light curves | δ > −31°; saturates G ≈ 12.5–13 | IRSA light-curve API, TAP `ztf_objects_dr24`, bulk Parquet; HATS up to `ztf_dr22` | none (forced-photometry service needs email registration) | IRSA asks for bulk downloads instead of many API calls |
| **ASAS-SN** Sky Patrol v2 | live | V/g light curves for bright stars (fills ZTF saturation gap) | All sky, ~8 < g < 17 | `skypatrol` (`pyasassn.client`) | none | Batch queries of up to 10⁶ light curves; bandwidth-limited. Throttling not documented. |
| **VLASS** | Epoch 1 & 2 QL component catalogues (CIRADA) | 3 GHz detections / limits | δ > −40° | CIRADA downloads; Epoch 1 also VizieR J/ApJS/255/30 | none | static files |
| **RACS** | low DR1, mid, high, low2/low3 | 0.9–1.7 GHz | δ ≲ +30° to +49° by band | CASDA TAP (`AS110.racs_*`), VizieR J/other/PASA/38.58, 41.3, 42.38, 43.89 | Catalogue TAP anonymous; images and cutouts need an OPAL account | CASDA async TAP |
| **eROSITA-DE** | **DR2** (eRASS:3, 2026-07-31) | 0.2–2.3 keV detections, upper limits | Western Galactic hemisphere only | `erosita.mpe.mpg.de/dr2`; HATS `erosita_dr2_erass3`; upper-limit server | none | static files + upper-limit service |
| **ROSAT** 2RXS | final | Shallower X-ray for the eastern half eROSITA-DE lacks | All sky | VizieR / HEASARC | none | as host |
| **LAMOST** | DR11 v2.0 (intl.), DR12 v1.1 | Low/medium-res optical spectra | δ ≳ −10° | TAP `www.lamost.org/<dr>/voservice/tap` (LRS `public.catalogue`, 12,605,485 rows in DR12 v1.1), SSA, direct FITS `…/<dr>/spectrum/fits/<obsid>` | None for TAP, SSA or single-spectrum FITS (tested anonymously 2026-10-21 on both releases). Account via China-VO OAuth only for FTP / bulk packages and the `pylamost` open API, which takes a token as a query parameter: env var `LAMOST_TOKEN` | not documented; keep to 2 concurrent |
| **SDSS** | **DR20** (2026-07-30) | BOSS/APOGEE spectra incl. SDSS-V Milky Way Mapper | Mostly north + MWM fields | SkyServer / CAS, CasJobs for large queries, SAS files | CasJobs account (email) for big jobs | CAS sync limits; use CasJobs |
| **DESI** | DR1 (DR2 expected early 2027) | Spectra of MWS targets | DESI footprint | NOIRLab Data Lab `desi_dr1.*`, SPARCL for spectra | Anonymous TAP; account for MyDB | Data Lab limits not documented |
| **SIMBAD** | live | Identifiers, types, bibliography | — | TAP (`simbad.cds.unistra.fr/simbad/sim-tap`) | none | **> 6 queries/s from one IP gets a temporary blacklist.** Upload tables to TAP instead of looping. |
| **VizieR** | live | Published catalogues | — | TAPVizieR, X-Match | none | same CDS policy |
| **ADS** | live | Literature record per object | — | ADS API | Token is a proxy-injected bearer secret for `api.adsabs.harvard.edu`; send no Authorization header ourselves | 5,000 search queries/day per token, resets 00:00 UTC; limits in response headers |

## Supporting sources

| Source | Role | Access |
|---|---|---|
| GALEX GR6/7 | UV: separates white dwarfs and WD+M pairs, measures chromospheric activity | MAST TAP / `astroquery.mast` |
| Pan-STARRS1 DR2, SkyMapper DR4 | Optical SED points between Gaia and 2MASS; some close pairs Gaia merges | MAST (PS1), HATS `skymapper_dr4` |
| 4XMM-DR14, Chandra CSC 2.1 | Deeper pointed X-ray where it exists | HEASARC TAP |
| NASA Exoplanet Archive | Known planets explain TESS / RV / astrometric signals | Exoplanet Archive TAP |
| UltracoolSheet, Montreal White Dwarf Database | Curated brown dwarf and white dwarf lists | static downloads |
| AAVSO VSX | Known variable classifications as priors | HATS `vsx` |

## Rate-limit summary and client defaults

Documented hard limits: SIMBAD/CDS (≤ 6 queries/s per IP), TESScut (≤ 10 concurrent), ADS (5,000/day),
Gaia row and time quotas, IRSA catalog engine (2 GB per query). Nothing else publishes numbers.

Client defaults for every service, enforced in one shared HTTP layer:
- at most 2 concurrent connections per host (TESScut: 8), at most 2 requests/s to CDS;
- prefer one TAP upload join of 10⁴ rows over 10⁴ cone searches;
- exponential backoff with jitter on 429/503/connection refused, and stop for an hour on repeated blacklist
  signals rather than retrying;
- content-addressed cache of every response (key = service + query hash), so reruns cost nothing;
- prefer static bulk copies (S3 Parquet, HATS, VizieR downloads) over live query services for anything
  sky-wide.

## Footprint gaps to keep in mind

X-ray is half-sky at eROSITA depth; VLASS and RACS overlap in −40° < δ < +30° and each covers one pole;
ZTF and LAMOST are northern; TESS is nearly all-sky but shallow. A missing detection outside a footprint is
not a non-detection, so every modality carries a coverage flag and an upper limit where one exists.
