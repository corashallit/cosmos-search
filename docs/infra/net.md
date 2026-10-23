# Shared archive client (`cosmos_search.net`)

Every outbound request goes through `get_client()`, a process-wide `requests.Session` subclass. Being one
object per process is what makes the limits hold when many threads share it; pyvo and astroquery take it
as `session=`.

```python
from cosmos_search.net import get_client, s3_list, Tap

c = get_client()
c.get_bytes(url, missing_ok=True)          # bytes, or None on 404
c.get(url, params=..., cache=True)         # opt-in response cache
c.download(url, dest, md5=...)             # streamed to disk, checksum verified, retried
with c.streamed(url) as path: ...          # temp file, deleted afterwards
for e in s3_list("https://stpubdata.s3.amazonaws.com", "mast/hlsp/qlp/", delimiter="/"): ...
Tap(url, job_slots=2).query(adql, uploads={"ids": table})   # async job, deleted after fetch, cached
```

Gaia: `cosmos_search.gaia.archive.tap()` logs in with `GAIA_USER` / `GAIA_PASSWORD` (cookie) and returns a
`Tap` with 4 job slots.

## Behaviour

| Concern | What the client does |
|---|---|
| Identification | `User-Agent: cosmos-search/<version> (Gaia DR4 anomaly search; mailto:cora.shallit@gmail.com)` |
| Concurrency | Per-host semaphore (table below). `download()` holds the slot for the whole transfer; a plain `stream=True` request releases it when headers arrive |
| Rate | Minimum interval between request starts, across threads |
| Transient failures | 429, 500, 502, 503, 504, connection errors, timeouts, truncated bodies and md5 mismatches are retried up to 6 times with full-jitter exponential backoff (base × 2ⁿ, capped), or after `Retry-After` when the server sends it (≤ 1 h) |
| Blacklist signals | For CDS, 403/429 count as strikes; 2 in a row pause every CDS host for 1 h and raise `HostBlocked` instead of retrying. Elsewhere, 3 calls in a row that exhaust their retries pause the host for 10 min |
| Daily budgets | ADS: when `X-RateLimit-Remaining` ≤ 50, further calls raise `BudgetExhausted` until `X-RateLimit-Reset` |
| Cache | Off by default; `cache=True` per call. Content-addressed (SHA-256 of method, URL with sorted query, body), atomic writes, LRU pruning at `COSMOS_CACHE_MAX_GB` (default 5; 100 on a machine with a local data store) under `COSMOS_CACHE_DIR` (default `~/.cache/cosmos-search/http`, or `<data dir>/cache/http` on a machine with a local data store, see [storage.md](../storage.md#local-data-store-kore)). Only 200 responses are stored. TAP results are cached as raw VOTable keyed by service, whitespace-normalised ADQL, `maxrec` and upload contents |
| Blind runs | `COSMOS_BLIND=1` makes every request **and every cache read** raise `BlindModeError`, because the cache holds un-injected data (blinding.md §2) |
| TAP columns | Lower-cased on parse (Gaia async jobs return `SOURCE_ID`) |

Limits are per process. Running several processes against one host multiplies them; use threads inside
one process instead.

## Per-host policy

The only place these numbers are kept; the code is `cosmos_search/net/policy.py`. The most specific
matching rule wins (host suffix, then path prefix).

| Rule | Concurrent | Min interval | Notes and source |
|---|---|---|---|
| `cds.unistra.fr`, `u-strasbg.fr` (shared) | 2 | 0.5 s | CDS blacklists > 6 queries/s per IP; 403/429 are blacklist signals, 1 h pause |
| `gea.esac.esa.int` | 4 | 0.1 s | Gaia archive: ≤ 5 executing jobs per user since 2026-10-14; `Tap` keeps 4 job slots |
| `cdn.gea.esac.esa.int`, `cdn77-storage.com` | 8 | — | Gaia bulk files and their S3-style listing |
| `amazonaws.com` | 32 | — | Public S3 (stpubdata, nasa-irsa-wise); backoff base 0.5 s |
| `mast.stsci.edu/tesscut` | 8 | — | TESScut returns 503 above 10 simultaneous calls per user |
| `mast.stsci.edu` | 4 | — | Nothing documented |
| `api.adsabs.harvard.edu` | 2 | 0.2 s | 5,000 queries/day per token; stop at 50 remaining. The proxy injects the token; send no Authorization header |
| `data.lsdb.io` | 8 | — | Static HATS Parquet |
| anything else | 2 | — | Polite default |

## Gaia bulk files

DR3 epoch photometry for all 11,754,237 sources in `vari_summary` is 74 GB in 3,386 `csv.gz` files under
`cdn.gea.esac.esa.int/Gaia/gdr3/Photometry/epoch_photometry/`, named by HEALPix level-8 index range and
listed with MD5 sums in `_MD5SUM.txt`. The CDN's directory pages are JavaScript; list the bucket through
`s3_list("https://gaia.eu-1.cdn77-storage.com", "Gaia/gdr3/...")` instead.
