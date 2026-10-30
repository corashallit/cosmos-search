# Release-day figures

Code: `cosmos_search/viz/`, driven by `scripts/viz_candidates.py`. Gallery of DR3 examples (private to the
project): https://claude.ai/artifact/SANH9rfcwL9Gi2XKopRUkQ

One set of figures per candidate: light curves, strength against the full distribution, DPAC classes,
colour-magnitude diagram, BP/RP depth ratio and NEOWISE W1/W2 change. Style from cora.earth (colours, mono labels in
highlight boxes, square corners), each figure in a light and a dark version; the page follows the reader's
light/dark setting and has a toggle.

## On release day

Input: a slow-change run directory (docs/arms/slow-changes.md, "Release day, end to end") with `lists/`, `checks/`
and `rule-*/` (release.json, drift_table.parquet, stats.parquet).

1. Comparison sample, once per run (ESA TAP, a few minutes):

       python scripts/viz_candidates.py reference --run RUN --out REF --release dr4

   300,000 random stars of the threshold sample with G, BP − RP and DPAC class, plus 400,000 stars with
   parallax / error > 10 for the colour-magnitude diagram.
2. Figures and a page for selected stars:

       python scripts/viz_candidates.py figures --run RUN --ref REF --release dr4 --ids 123 456 --out OUT \
           --lightcurves DIR --title "Gaia DR4 candidates" --artifact

   Up to six figures per star (below) as `OUT/<id>-<figure>-<light|dark>.png` and `.svg`; `OUT/index.html` shows
   them with captions and PNG links, and `--artifact` also writes `OUT/artifact.html` for claude.ai. Light curves
   come from `--lightcurves` (the files the scoring step kept) or, for DR3, the AIP mirror.
3. Other figures: the plot functions take plain DataFrames, and `style.render(fn, stem, ...)` draws any of them in
   both themes. `sheet.py` shows the calls for the standard set.

The page inlines every figure as SVG in both themes, about 0.2 MB per figure pair and 0.7 MB for the DPAC-class
figure. Keep a page under about 15 stars (the Artifact limit is 16 MB) or split it.

## Figures

| Figure | Function | Content |
|---|---|---|
| Light curves | `lightcurve.slow_change` | Gaia visit means in G, then BP and RP, then ZTF and NEOWISE where available, as ΔF/F about the median (Δmag beyond 40%). Orange: detected episode (band) or trend (line), size in the label. Grey band: ±2σ visit-to-visit scatter of non-variable stars at the same G. |
| Strength distribution | `population.strength_hist` | Histogram of strength (score / threshold) for every star scored in the same magnitude range, fading above the axis and brightening below (log counts). Brightening is the sign-reversed statistic, so it estimates the false-alarm count at each strength. The candidate is marked with its rank among all fades. |
| DPAC classes | `population.known_classes` | Abbe value against episode depth for the comparison sample, one DPAC class per panel. |
| Colour-magnitude diagram | `population.hr_diagram` | M_G against BP − RP; with parallax under 3σ, a line at the candidate's BP − RP only. |
| BP/RP depth ratio | `colour.change_colour` | BP depth / RP depth against G depth, with reference lines for a grey occulter, dust (A ∝ λ^−1.5) and a temperature drop. |
| NEOWISE W1/W2 | `colour.infrared` | W1 and W2 change over the episode dates, measured and predicted per model (from `slow_checks.py`). |
| Short dip | `lightcurve.short_dip`, `population.dip_depth` | All G epochs with the dip marked, G/BP/RP within ±30 h of the minimum, and single-epoch depth against G for the comparison sample. |
| Astrometry | `astrometry.along_scan`, `astrometry.kepler_bound` | Along-scan residuals against the 5-parameter model, before and after an alternative model; f_A against period for all DR3 orbital solutions, Gaia BH1 and BH2 marked. |

Captions describe the figure and do not state a conclusion about the source.

## Colours

From the cora.earth style sheet (2026-10-30), in `style.TOKENS`:

| Role | Light | Dark |
|---|---|---|
| figure background (`box`) | #f5f5f5 | #1b1b1b |
| comparison sample: density from the page colour through `blue` | #4b64a3 | #93a8dc |
| candidate (`orange`) | #e56900 | #fd923e |
| Gaia G / BP / RP | ink / blue / red #b2443f | ink / blue / red #e48a83 |
| ZTF g / r / i | blue / red / amber #9a6814 | blue / red / amber #e0b462 |
| NEOWISE W1 / W2 | purple #7d52b0 / amber | purple #bb9de6 / amber |
| DPAC classes | ink dots, one class per panel | same |

Colour pairs that share a panel were checked with the dataviz skill's palette validator: blue/red, purple/amber
and blue/orange pass every check in light mode; in dark mode they pass the colour-blind and contrast checks but sit
above the validator's lightness band, as the site's dark palette does. Green fails against amber and orange and is
not used.

## DR3 caveat

DR3 epoch photometry covers only the 11.6 million stars DPAC flagged as variable, so the DR3 comparison sample is
variable stars, 77% of them with a DPAC class. At G = 15 the classed stars have about three times the
visit-to-visit scatter of the unclassed ones, so the grey band uses unclassed stars below threshold. In DR4 every
star has epoch photometry and the comparison sample is all stars.

## Log

- 2026-10-30: built; gallery of six DR3 stars of known type (RS CVn, eclipsing binary, quasar, Be, long-period
  variable, R CrB), two injected grey signals (−1% yr⁻¹ trend; 2% dip over 300 d), a short dip on an eclipsing
  binary and Gaia BH3 from the DR4 pre-release. Reference data: /mnt/project-files/viz/ref-dr3/. Rebuild with
  `viz_candidates.py gallery --run /mnt/project-files/slow-changes/trial-dr3 --ref /mnt/project-files/viz/ref-dr3
  --out OUT --features ... --aux ... --orbits /mnt/project-files/astrometry/dr3/orbits_kb.parquet --artifact`.
- 2026-10-30: wording changed to standard technical terms throughout (Cora); the cumulative "how rare" curves
  replaced by a mirrored strength histogram with the candidate's rank.
