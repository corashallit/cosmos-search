# cosmos-search

Be among the first to search Gaia DR4 (released 2 December 2026) for anomalies with an artificial or
otherwise unexplained origin, and hand over a short, vetted, easy-to-verify list with a measured
false-positive rate.

The bet: one person with pre-validated queries and an agent vetting loop can move faster on a narrow question
than a survey team, in exactly the kind of search where volunteers historically found the famous objects
(Boyajian's star from Planet Hunters; the brown dwarf "The Accident" from Backyard Worlds).

## Arms

| Arm | Data | Status | Doc |
|---|---|---|---|
| **Aperiodic dimmers** (main) | Gaia DR4 epoch photometry; DR3 as rehearsal | Designing | [docs/arms/gaia-dimmers.md](docs/arms/gaia-dimmers.md) |
| **Aperiodic dimmers in TESS FFIs** (side, cross-check) | TESS full-frame images | Thread working | [docs/arms/tess-dimmers.md](docs/arms/tess-dimmers.md) |
| **Non-Keplerian astrometry** (narrow, secondary) | Gaia DR4 epoch astrometry and NSS; DR3 NSS and DR4 pre-release as rehearsal | Rule astrometry-v1 adopted (blind test passed) | [docs/arms/astrometry.md](docs/arms/astrometry.md) |

Shared infrastructure: archive client and Gaia access ([docs/infra/](docs/infra/)), injection and blinding
([docs/blinding.md](docs/blinding.md)), statistics ([docs/statistics.md](docs/statistics.md)), vetting ([docs/vetting.md](docs/vetting.md)).

## Start here

- [docs/plan.md](docs/plan.md): guiding principles, milestones in dependency order, open decisions.
- [CONTRIBUTING.md](CONTRIBUTING.md): branches, ownership, conventions for the threads working in parallel.
- [docs/data-sources.md](docs/data-sources.md), [docs/storage.md](docs/storage.md).

Earlier scoping for a full 100 pc census is in git history (before commit `7dc9a9a`).
