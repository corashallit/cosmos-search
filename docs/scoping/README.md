# 100 pc census: scoping

Status: first scoping pass, 2026-10-21. Gaia DR4 is scheduled for **2 December 2026** (ESA), six weeks out.
Everything here is a proposal; the plan was deliberately loose, and several parts of it are restructured below.

| Doc | What it covers |
|---|---|
| [statistics.md](statistics.md) | What "explained" means, sample definition, cross-matching, multiple testing, injection-recovery, decoys, DR3 → DR4 validation |
| [data-sources.md](data-sources.md) | Every archive: release to use, coverage, access route, auth, rate policy, where the contact email goes |
| [software.md](software.md) | Libraries, pipeline shape, repo layout |
| [storage.md](storage.md) | Volume estimates for DR3 and DR4, and what we need to provision |
| [blinding.md](blinding.md) | Whether blinding works in a single repo, and the design that makes it work well enough |

## Summary

**Sample.** Gaia DR3 has 574,531 sources with ϖ > 10 mas, but a large fraction are spurious astrometric
solutions in crowded fields. The Gaia Catalogue of Nearby Stars (GCNS, Smart et al. 2021), built on the same
EDR3 astrometry that DR3 reuses, keeps 331,312 after a classifier. Expect a working sample of
**~3.3–4 × 10⁵ objects**, plus a few thousand that Gaia cannot see (T/Y dwarfs, the coolest white dwarfs,
the brightest naked-eye stars). Counts are from live archive queries; see `scripts/probe_gaia_counts.py`.

| Gaia DR3 selection | Rows |
|---|---|
| ϖ > 8 mas (buffer for boundary objects) | 1,211,740 |
| ϖ > 10 mas | 574,531 |
| ϖ > 10 mas, ϖ/σ > 5 | 541,958 |
| ϖ > 10 mas, ϖ/σ > 10 | 372,226 |
| ϖ > 10 mas with XP continuous spectra | 247,841 |
| ϖ > 10 mas with RVS spectra | 20,100 |
| ϖ > 10 mas with epoch photometry | 26,593 |
| ϖ > 10 mas with a non-single-star solution | 10,236 |
| GCNS main table | 331,312 |

**Data sources.** The current target list, with releases, coverage, access and rate policies, is in
[data-sources.md](data-sources.md). It is the single source of truth for what the pipeline ingests.

**Storage.** The DR3 rehearsal fits in the ~30 GB of local disk if we stream light curves and keep only
features (≈15–25 GB durable). DR4 for 100 pc is ~150–180 GB if pulled whole (~100 GB without epoch XP
spectra). Provisioning is deferred until closer to December; until then, nothing durable is written beyond
the repo. See [storage.md](storage.md).

**Blinding.** Truth and its randomness live in a second private repo that no Claude session can
reach, because the Claude GitHub App is not installed on it. A GitHub Actions job there generates
injections and decoys and publishes only the injected products plus truth encrypted to Cora's key. Inside
this repo, blind runs read a frozen snapshot with the network off, and expert subagents work in two stages:
a blind stage on anonymised packets with no tools, then an unblinded literature stage that never sees
decoys. See [blinding.md](blinding.md).

## Open items

| Item | Status |
|---|---|
| Gaia Archive, ADS, SciServer credentials | Done, verified |
| Durable storage, ~250 GB (options in storage.md) | Deferred; needed before DR4 |
| Truth repo with the Claude GitHub App *not* installed (blinding.md §1) | Set up, smoke run passed; Cora removing Claude's access |
| `age` key pairs (Cora's, reveal); public keys committed to `keys/` | Done |
| Expert budget (statistics.md §6) | Decided: plan for a queue of ~10³ objects; ~10⁴ is plausible but near the limit. The classical tier must explain ≳ 99.7% of the sample |
| Scope of "everything": stars, brown dwarfs, compact objects and their companions/disks. Solar-system bodies are modelled as contaminants; the ISM and the Local Bubble are context | Proposed |

## Milestones

Ordered by dependency, with no durations. The only fixed dates are external: Gaia DR4 on 2 December 2026,
DESI DR2 expected early 2027.

| # | Milestone | Depends on | Done |
|---|---|---|---|
| 1 | Shared archive client: User-Agent, throttling, backoff, response cache | — | |
| 2 | Sample selection v0 on DR3: membership probabilities, fidelity cut, ϖ > 8 mas buffer | 1 | |
| 3 | Proper-motion-aware cross-match to 2MASS / WISE / CatWISE / eROSITA / VLASS / RACS, with shifted-sky false-match rates | 2 | |
| 4 | Time-series and spectra fetchers (TESS, ZTF, ASAS-SN, NEOWISE, LAMOST/SDSS/DESI) streaming to features | 1, 2 | |
| 5 | Classical model tier v0: SED + CMD + astrometric + variability residuals | 3, 4 | |
| 6 | Calibrated p-values, FDR ranking, expert queue | 5 | |
| 7 | Injection classes in `cosmos_search/inject/` feeding the truth-repo workflow; first completeness maps | 4, 5; truth repo ✓ | |
| 8 | Durable store, so injected products reach the analysis side | Cora | |
| 9 | Expert stages A and B with decoys and known-answer controls, rehearsed on DR3 | 6, 7, 8 | |
| 10 | Validation against GCNS, CNS5, 40 pc WDs, 20 pc ultracools | 5, 6 | |
| 11 | Freeze code and thresholds (`dr3-frozen` tag), pre-register the DR4 analysis | 9, 10; before DR4 | |
| 12 | DR4: rerun mean-product tiers blind; epoch products follow | 11; DR4 release | |
