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

**Corrections to the initial source list.**
- eROSITA-DE **DR2** (eRASS:3, released 31 July 2026) supersedes DR1. It still covers only the western
  Galactic hemisphere (180° < l < 360°), so X-ray coverage is half-sky; add ROSAT 2RXS for the other half.
- SDSS is at **DR20** (30 July 2026), which carries SDSS-V Milky Way Mapper spectra of many nearby M dwarfs
  and white dwarfs. LAMOST is at DR11 v2 (international) / DR12 v1.1. DESI DR1 is public; DR2 is expected
  early 2027, after DR4.
- ZTF is at DR24 (2026-01-22). It saturates around G ≈ 12.5–13, so ASAS-SN and TESS carry the bright end.
- Worth adding: Hipparcos (stars too bright for good Gaia astrometry), GALEX (UV, essential for white dwarfs
  and chromospheric activity), Pan-STARRS1 / SkyMapper (optical SED fill-in), CatWISE2020 motions for
  objects Gaia misses, NASA Exoplanet Archive, UltracoolSheet, and the Montreal White Dwarf Database.

**Storage.** The DR3 rehearsal fits in the ~30 GB of local disk if we stream light curves and keep only
features (≈15–25 GB durable). DR4 epoch data for 100 pc is ~150–180 GB if pulled whole (~100 GB without epoch XP spectra). The local disk is
ephemeral and the repo cannot hold data, so we need a durable store before 2 December. Details and the ask
are in [storage.md](storage.md).

**Blinding.** A single repo plus a shared container cannot blind a determined reader, but it can be made
blind by construction against the realistic leaks: truth tables are encrypted to a key only Cora holds,
injections are drawn from unlogged randomness, and expert subagents work in two stages, a blind stage on
anonymised packets with no tools and no network, then an unblinded literature stage that never sees decoys.
The one leak this does not close is an agent comparing injected data against the live archive, which is why
blind runs read from a frozen snapshot. If Cora wants stronger guarantees, move injection to a GitHub
Actions job in a second private repo. See [blinding.md](blinding.md).

## Decisions needed from Cora

1. **Durable storage**: approve ~250 GB of durable storage (options in storage.md).
2. **Accounts**: register a Gaia Archive account and an ADS API token with cora.shallit@gmail.com, and store
   the credentials as environment secrets (never in the repo). Optional: LAMOST, SDSS CasJobs.
3. **Blinding key**: generate an `age` key pair locally and commit only the public key.
4. **Expert budget**: the size of the expert queue is the main cost driver (statistics.md §6). A queue of
   ~10³ objects is plausible; ~10⁴ is expensive.
5. **Scope of "everything"**: proposal is compact objects, stars, brown dwarfs and their companions/disks.
   Solar-system bodies are technically within 100 pc but are treated as contaminants to model, not targets;
   the ISM and the Local Bubble are context, not census entries.

## Six-week plan to DR4

| Week (Mon) | Milestone |
|---|---|
| 26 Oct | Polite archive client, sample selection v0 on DR3, durable store in place |
| 2 Nov | Proper-motion-aware cross-match to 2MASS / WISE / CatWISE / eROSITA / VLASS / RACS |
| 9 Nov | Classical model tier v0: SED + CMD + astrometric + variability residuals |
| 16 Nov | Injection harness and sealed-truth tooling; first completeness maps |
| 23 Nov | Expert-stage rehearsal with decoys and known-answer controls on DR3 |
| 30 Nov | Freeze code and thresholds (`dr3-frozen` tag), pre-register the DR4 analysis |
| 2 Dec | DR4: rerun mean-product tiers blind; epoch products follow |
