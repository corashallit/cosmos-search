# Software

## Pipeline shape

```
 select ─► crossmatch ─► fetch time series / spectra ─► features ─► classical models ─► anomaly scores
   │            │                  │ (streamed, cached)       │              │                  │
 Gaia +      PM-aware,         per object, features       Parquet,      tier 1 fits,      FDR-ranked
 Hipparcos + chance-match      kept, raw data dropped     HEALPix-      posterior         expert queue
 CatWISE     rate measured                                partitioned   predictive p      ─► experts (blinded.md)
                                       ▲
                       injections enter here, from sealed truth (blinding.md)
```

Every stage reads and writes Parquet partitioned by HEALPix order 5 (12,288 tiles; ~30 objects per tile
at 100 pc). Each stage is a pure function of its inputs plus a config hash, so any tile can be recomputed
and two runs can be diffed.

## Libraries

| Need | Choice | Notes |
|---|---|---|
| Archive access | `pyvo`, `astroquery`, `requests` | One shared session with the User-Agent, throttling and cache from data-sources.md |
| Sky-wide joins without TAP | `lsdb` + HATS catalogues (`data.lsdb.io/hats`: Gaia DR3, 2MASS, WISE, ZTF, eROSITA DR2, TESS TIC, SkyMapper, VSX) | Streams only the tiles we touch; no rate limits |
| Tables | `pyarrow`, `polars`, `duckdb` | DuckDB queries Parquet in place, which keeps memory low on a 15 GB machine |
| Coordinates / PM propagation | `astropy` (`SkyCoord.apply_space_motion`) | Propagate with covariance to each survey epoch |
| Sky partitioning | `astropy-healpix` or `healpy`, `mocpy` for footprints | MOCs give coverage flags for every survey |
| Gaia specifics | `gaiaxpy` (XP sampling, synthetic photometry), `gaiadr3_zeropoint`, `gaiaunlimited` (selection function) | DR4 tools will need checking when released |
| Dust | `dustmaps` (Edenhofer 2024, Bayestar19) | Small but non-zero toward 100 pc |
| Stellar models | MIST / PARSEC isochrones, BT-Settl, PHOENIX, Sonora Bobcat/Elf Owl, Koester / Bergeron WD grids, `synphot`/`speclite` filters | Precompute synthetic photometry grids once |
| Inference | `numpy`/`scipy`, `jax` + `numpyro` for batched fits, `dynesty` for hard cases | Batched JAX fits keep 4 × 10⁵ SED fits to minutes |
| Light curves | `lightkurve`, `astropy.timeseries` (Lomb–Scargle, BLS), `transitleastsquares`, `wotan`, `celerite2` | Feature extraction runs on the fly per object |
| Spectra | `specutils`, SPARCL client (DESI), SDSS/LAMOST readers | |
| Cross-match statistics | own Bayesian matcher; `nway` for X-ray/radio | |
| Multiple testing | `statsmodels` (BH, Storey) | |
| Anomaly ranking cross-check | `scikit-learn` isolation forest, normalising flows (`zuko`) | Only as a second opinion; explicit-model residuals stay primary |
| Blinding | `age` / `pyrage` | See blinding.md |
| Workflow | `snakemake` or plain Python + Make | Tile-level parallelism over 4 cores |
| Tests / CI | `pytest`, GitHub Actions | Unit tests and a small synthetic tile only; CI never touches real data |

## Repo layout (proposed)

```
cosmos_search/
  net/          shared HTTP client: User-Agent, throttle, backoff, cache
  sample/       membership probabilities, fidelity, selection function
  xmatch/       PM propagation, Bayesian matching, shifted-sky rates
  fetch/        per-survey adapters (TESS, ZTF, ASAS-SN, NEOWISE, spectra)
  features/     light-curve, SED and spectral features
  models/       classical model tier
  score/        p-value calibration, FDR, expert queue
  inject/       injection generators (distributions public, draws sealed)
  experts/      packet builder, anonymiser, agent definitions, scoring
configs/        one YAML per run; hashed into every output
docs/scoping/   these documents
scripts/        probes and one-off utilities
tests/
keys/           age public key only
```
