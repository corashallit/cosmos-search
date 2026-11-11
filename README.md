# cosmos-search

A search of Gaia DR4 (released 2 December 2026) for stars that change in ways neither the instrument nor known
astrophysics explains, aiming to be among the first to look and to hand over a short, checked, easy-to-verify
list with a measured recall and false-positive rate.

The bet: one person with code written and practised before the release, and a review of candidates whose error
rates are measured with injected signals, can move faster on a narrow question than a survey team, in the kind
of search where volunteers found the famous objects (Boyajian's star from Planet Hunters; the brown dwarf "The
Accident" from Backyard Worlds).

## Searches

| Search | Data | State | Page |
|---|---|---|---|
| **Slow changes** (the main search): fades and brightenings over years, dimming episodes of weeks to years | Gaia DR4 epoch photometry for every star, read from ESA's bulk files or DataLink | Built and tried end to end on all 11.6 M DR3 stars with light curves; scoring of every DR4 star from the bulk files is being finished | [docs/arms/slow-changes.md](docs/arms/slow-changes.md) |
| Short dips | Gaia DR4 epoch photometry | Built and frozen: runs on DR4 as is | [docs/arms/gaia-dimmers.md](docs/arms/gaia-dimmers.md), [docs/dayone.md](docs/dayone.md) |
| Non-Keplerian astrometry | Gaia DR4 epoch astrometry and non-single-star tables | Rule adopted after a sealed blind test; runs as built | [docs/arms/astrometry.md](docs/arms/astrometry.md) |
| Dips in TESS full-frame images | TESS | Built; a cross-check for Gaia candidates | [docs/arms/tess-dimmers.md](docs/arms/tess-dimmers.md) |

Why slow changes lead: Gaia's ~90 visits over 5.5 years catch a 1-day dip on ~2% of stars, usually once, but catch
90-100% of changes longer than ~100 days and measure a star's trend to a few hundredths of a percent per year.
With or without a find, the slow-change search gives an upper limit on the share of Sun-like dwarfs fading
steadily at each rate ([docs/arms/occurrence.md](docs/arms/occurrence.md)).

## Start here

- [docs/plan.md](docs/plan.md): principles, where things stand, what is left before the release, in dependency order.
- [docs/release-day.md](docs/release-day.md): the order of work on 2 December, the first-hour checks, and the
  practice run on fake DR4 files built from DR3.
- [docs/infra/](docs/infra/): archive client, Gaia readers, routes to DR4's light curves, cross-match.
  [docs/blinding.md](docs/blinding.md), [docs/statistics.md](docs/statistics.md), [docs/vetting.md](docs/vetting.md)
  and [docs/viz.md](docs/viz.md): injected signals and blinding, statistics, the review of candidates, figures.
- [CONTRIBUTING.md](CONTRIBUTING.md): branches, ownership and conventions for the threads working in parallel.
- [docs/data-sources.md](docs/data-sources.md), [docs/storage.md](docs/storage.md).

```
pip install -e '.[dev,plots]'
pytest -q -m "not network"
```

## Licence and citation

Code: BSD 3-Clause ([LICENSE](LICENSE)). Documentation, written results, candidate lists and figures: Creative
Commons Attribution 4.0 International ([LICENSE-CC-BY-4.0](LICENSE-CC-BY-4.0)). To cite, see
[CITATION.cff](CITATION.cff).

This work has made use of data from the European Space Agency (ESA) mission Gaia
(https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis Consortium (DPAC,
https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC has been provided by national
institutions, in particular the institutions participating in the Gaia Multilateral Agreement.

Earlier scoping for a full 100 pc census is in git history (before commit `7dc9a9a`).
