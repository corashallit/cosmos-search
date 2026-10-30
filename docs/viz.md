# Release-day figures

Code: `cosmos_search/viz/`, driven by `scripts/viz_candidates.py`. Gallery of DR3 examples (private to the
project): https://claude.ai/artifact/SANH9rfcwL9Gi2XKopRUkQ

One set of figures per candidate: light curves, strength against the full distribution, DPAC classes of the most
similar strong fades, colour-magnitude diagram, BP/RP depth ratio and NEOWISE W1/W2 change. Pages carry a list of
terms (`sheet.TERMS`): Gaia G, BP, RP and BP − RP, ZTF and NEOWISE bands, ΔF/F, visit, strength, DPAC class, Abbe
value, M_G, AL and f_A. Style from cora.earth (colours, mono labels in
highlight boxes, square corners), each figure in a light and a dark version; the page follows the reader's
light/dark setting and has a toggle.

## On release day

Input: a slow-change run directory (docs/arms/slow-changes.md, "Release day, end to end") with `lists/`, `checks/`
and `rule-*/` (release.json, drift_table.parquet, stats.parquet).

1. Reference data, once per run (ESA TAP, a few minutes):

       python scripts/viz_candidates.py reference --run RUN --out REF --release dr4

   `pop.parquet`: 300,000 random stars of the threshold sample with G, BP − RP and DPAC class. `hr.parquet`: 400,000
   stars with parallax / error > 10 for the colour-magnitude diagram. `top.parquet`: the 20,000 strongest fades
   (`--top`) with parallax, proper motion, their errors and DPAC class. `--parts top` redoes one part only.
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
| DPAC classes of similar stars | `population.similar_classes` | Share of each DPAC class among the 100 strongest fades nearest the candidate, against its share among all of them (log scale, with both shares as a table). Nearest in episode depth (mag) and duration, Abbe value, BP − RP, G, parallax / error and proper motion / error (`data.similarity_features`), each scaled by 1.4826 MAD. The footnote gives how isolated the candidate is: its median distance to the 100 nearest against the same for every comparison star. |
| Colour-magnitude diagram | `population.hr_diagram` | M_G against BP − RP; with parallax under 3σ, a line at the candidate's BP − RP only. |
| BP/RP depth ratio | `colour.change_colour` | BP depth / RP depth against G depth, with reference lines for a grey occulter, dust (A ∝ λ^−1.5) and a temperature drop. |
| NEOWISE W1/W2 | `colour.infrared` | W1 and W2 change over the episode dates, measured and predicted per model (from `slow_checks.py`). |
| Short dip | `lightcurve.short_dip`, `population.dip_depth` | All G epochs with the dip marked, G/BP/RP within ±30 h of the minimum, and single-epoch depth against G for the comparison sample. |
| Astrometry | `astrometry.along_scan`, `astrometry.kepler_bound` | Along-scan residuals against the 5-parameter model, before and after an alternative model; f_A against period for all DR3 orbital solutions, Gaia BH1 and BH2 marked. |

Captions describe the figure and do not state a conclusion about the source.

### DPAC classes: how well the figure matches DPAC

The first version plotted Abbe value against episode depth with one DPAC class per panel. Among strong fades nearly
every class is smooth (median Abbe value 0.04-0.13 for all but eclipsing binaries and cataclysmic variables), so
five of the six gallery stars landed in the quasar cloud, and the gallery quasar itself sat off the top because a
drop from six times the median read as 530% deep. Depth is now in magnitudes, and classes are compared by nearest
neighbours over seven numbers.

On the 20,000 strongest DR3 fades (19,501 with one of the eight named classes), leaving each star out of its own
neighbours:

| Class | Stars | Largest share among the 100 nearest is its class | Own class among the 100 nearest (median) |
|---|---|---|---|
| eclipsing binaries | 9,314 | 100% | 98 |
| quasars | 8,415 | 100% | 100 |
| spotted stars | 728 | 94% | 94 |
| long-period variables | 599 | 91% | 94 |
| Be stars | 235 | 91% | 98 |
| cataclysmic variables | 124 | 44% | 24 |
| young stars | 65 | 6% | 16 |
| R CrB stars | 21 | 0% | 7 |

Overall 98%. Rare classes seldom lead, but other stars rarely have 10 or more of them among their nearest (under
0.3% for young stars, cataclysmic variables and R CrB stars). Taking instead the class with the largest ratio of the
two shares gives 80% (82% averaged over classes; eclipsing binaries 62%), so the figure shows both shares and the
caption says how to read them. Gallery stars: the quasar, eclipsing binary, spotted star and long-period variable
lead with their own class (92-98 of 100); the Be star has 34 Be stars and 40 spotted stars among its nearest (1.2%
and 3.7% overall); the R CrB star has none, because most of its DR3 visits fall in declines and the detected episode
is only 0.17 mag deep.

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
- 2026-10-30: DPAC class figure replaced by classes of the 100 most similar strong fades (Cora found the Abbe/depth
  panels put most gallery stars in the quasar panel); band ranges in captions and labels, and a terms list on each
  page. Reference data gains `top.parquet`.
