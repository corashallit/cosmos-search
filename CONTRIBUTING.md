# Contributing

Several Claude threads work in this repo at once. These rules keep them out of each other's way.

## Branches

- `main` is the integration branch. Never push work-in-progress to it.
- Each thread works on its own branch from `main` and opens a pull request into `main`.
- Rebase on `main` before merging; merge only when CI is green. Squash-merge is fine.
- Keep PRs to one package or concern where possible.

## Ownership

| Path | Owner thread | Contents |
|---|---|---|
| `cosmos_search/net/` | Archive client and Gaia epoch photometry | HTTP client, TAP layer, cache |
| `cosmos_search/gaia/` | Archive client and Gaia epoch photometry | DataLink, epoch photometry, DR4 adapters |
| `cosmos_search/targets/` | Archive client and Gaia epoch photometry | Versioned ADQL target queries, freeze manifest |
| `cosmos_search/tess/` | TESS aperiodic dimmers | FFI access, light curves |
| `cosmos_search/dimming/` | TESS aperiodic dimmers (first), then shared | Dip detection and features used by both arms |
| `cosmos_search/inject/` | DR4 plan and repo layout | Injector core and sealing; arms add classes in `inject/classes/` |
| `cosmos_search/stats/` | DR4 plan and repo layout | Empirical nulls and tail p-values, FDR, p-value combination, audit sampling |
| `cosmos_search/vetting/`, `.claude/agents/` | Candidate vetting | Packets, anonymiser, Stage 0 tests, expert agents, commit/reveal, calibration (V1–V3) |
| `cosmos_search/xmatch/` | Multi-survey cross-match | Proper-motion-aware matching (I5) |
| `cosmos_search/dayone/` | DR4 day-one dress rehearsal | Glue for the day-one chain: DataLink pull order, ranking v0, packet building and scoring (docs/dayone.md) |
| `cosmos_search/ebfilter/`, `inject/classes/eclipsing_binary.py` | Eclipsing-binary filter for DR4 | Dip-level EB test (compression over periods), its decision layer and calibration (docs/arms/eb-filter.md) |
| `cosmos_search/residual/` | Simplest-model residual search | Eclipse baseline, flux-level residuals, ranking v1's demotion rule (docs/arms/residual.md) |
| `cosmos_search/astrometry/` | Astrometric non-Keplerian filter | Epoch loader, nested fits, cascade, simulator, decision rule, Kepler bound, dip localisation (A1–A3) |
| `docs/` | DR4 plan and repo layout | Arms own their page under `docs/arms/` and `docs/infra/` |

To change a package you don't own, message its owner thread first.

## Code

- Python ≥ 3.11. Dependencies go in `pyproject.toml`; resolve conflicts there by rebasing, not by deleting
  other threads' entries.
- Every outbound request goes through `cosmos_search.net` once it exists (User-Agent with the contact email,
  throttling, cache). Respect the limits in `docs/data-sources.md`.
- Tests live in `tests/` and run offline. Mark anything that needs a live archive with
  `@pytest.mark.network`; CI skips those.
- Never commit data, credentials, or plaintext injection truth (see `docs/blinding.md`).
