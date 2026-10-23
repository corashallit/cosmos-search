# cosmos-search

Be among the first to search Gaia DR4 (released 2 December 2026) for anomalies with an artificial or
otherwise unexplained origin, and hand over a short, vetted, easy-to-verify list with a measured
false-positive rate.

The bet: one person with pre-validated queries and an agent vetting loop can move faster on a narrow question
than a survey team, in exactly the kind of search where volunteers historically found the famous objects
(Boyajian's star from Planet Hunters; the brown dwarf "The Accident" from Backyard Worlds).

## Searches

| Search | Data | Status | Doc |
|---|---|---|---|
| **Slow changes** (main since 2026-10-23): fades and brightenings over years, dimming episodes of weeks to years | Gaia DR4 epoch photometry and `all_source_photometry`; DR3 as trial run | Known answers and drift correction built | [docs/arms/slow-changes.md](docs/arms/slow-changes.md) |
| **Short dips** | Gaia DR4 epoch photometry | Built and frozen: runs on DR4 as is, no further tuning | [docs/arms/gaia-dimmers.md](docs/arms/gaia-dimmers.md), [docs/dayone.md](docs/dayone.md) |
| **Non-Keplerian astrometry** | Gaia DR4 epoch astrometry and NSS | Rule astrometry-v1 adopted (blind test passed); runs as built | [docs/arms/astrometry.md](docs/arms/astrometry.md) |
| **Dips in TESS FFIs** (cross-check) | TESS full-frame images | Built | [docs/arms/tess-dimmers.md](docs/arms/tess-dimmers.md) |

Why slow changes lead: Gaia's ~90 visits over 5.5 years catch a 1-day dip on ~2% of stars, usually once, but catch
90-100% of changes longer than ~100 days and measure a star's trend to a few hundredths of a percent per year.

Shared infrastructure: archive client and Gaia access ([docs/infra/](docs/infra/)), injection and blinding
([docs/blinding.md](docs/blinding.md)), statistics ([docs/statistics.md](docs/statistics.md)), vetting ([docs/vetting.md](docs/vetting.md)).

## Start here

- [docs/plan.md](docs/plan.md): guiding principles, milestones in dependency order, open decisions.
- [CONTRIBUTING.md](CONTRIBUTING.md): branches, ownership, conventions for the threads working in parallel.
- [docs/data-sources.md](docs/data-sources.md), [docs/storage.md](docs/storage.md).

Earlier scoping for a full 100 pc census is in git history (before commit `7dc9a9a`).
