# Release-day figures

Code: `cosmos_search/viz/`, driven by `scripts/viz_candidates.py`. Gallery of DR3 examples: see the log below.

The figures answer, for one star at a time, what Gaia saw, how unusual that is among ordinary stars, which known
kind of variable it resembles, and what the change looks like across wavelengths. They are drawn in the style of
cora.earth (its colours, mono labels in tight highlight boxes, square corners) in a light and a dark version, and
the page that holds them follows the reader's light/dark setting and has the site's toggle.

## On release day

The slow-change search's outputs are the input (docs/arms/slow-changes.md, "Release day, end to end"): a run
directory with `lists/`, `checks/` and `rule-*/` (release.json, drift_table.parquet, stats.parquet).

1. Comparison sample, once per run (ESA TAP, a few minutes):

       python scripts/viz_candidates.py reference --run RUN --out REF --release dr4

   300,000 random stars of the threshold sample with their G, BP-RP and DPAC class, plus 400,000 stars with
   parallaxes better than 10% for the HR diagram.
2. Figures and a page for the stars worth showing:

       python scripts/viz_candidates.py figures --run RUN --ref REF --release dr4 --ids 123 456 --out OUT \
           --lightcurves DIR --title "Gaia DR4: first candidates"

   Each star gets up to six figures (below) as `OUT/<id>-<figure>-<light|dark>.png` and `.svg`, and `OUT/index.html`
   shows them all with captions and links to the PNGs. Light curves come from `--lightcurves` (the files the
   scoring step kept) or, for DR3, the AIP mirror.
3. Anything else (a figure for a talk, a different comparison): the plot functions take plain DataFrames, and
   `style.render(fn, stem, ...)` draws any of them in both themes. `sheet.py` shows how the standard set is called.

The page inlines every figure as SVG in both themes, about 0.2 MB per figure pair and 0.7 MB for the known-classes
figure. Keep a page under about 15 stars (the Artifact limit is 16 MB) or split it.

## The figures

| Figure | Function | What it shows |
|---|---|---|
| Light curves | `lightcurve.slow_change` | Every Gaia visit in G, then BP and RP, then ZTF and NEOWISE where they have data, as the change from the star's normal level. The detected change is the orange band (an episode) or line (a trend) with its size in the label; a grey band shows an ordinary star's visit-to-visit scatter at that brightness. Changes beyond 40% switch the axes to magnitudes. |
| How rare | `population.how_rare` | How many of the run's stars fade at least as strongly, and how many brighten as strongly. Brightening is the mirror of fading, so its curve is what chance and ordinary variability give; a fade is worth reading where the fading curve stands clear of it. |
| Known classes | `population.known_classes` | Smoothness (Abbe value) against depth for all compared stars, one DPAC class per panel. |
| HR diagram | `population.hr_diagram` | Where the star sits by colour and absolute magnitude; with a parallax under 3σ, only its colour. |
| Colour of the change | `colour.change_colour` | BP depth ÷ RP depth against the G depth, with the ratios an opaque body, interstellar-type dust and a cooling star predict. |
| Infrared | `colour.infrared` | NEOWISE's change over the episode's dates against each explanation's prediction (from `slow_checks.py`). |
| Short dip | `lightcurve.short_dip`, `population.dip_depth` | The dip among all the star's epochs, zoomed to hours in three bands, and its depth against ordinary stars of the same brightness. |
| Astrometry | `astrometry.along_scan`, `astrometry.kepler_bound` | Positions along each scan after the single-star model, before and after an alternative model; the orbit's mass function against every catalogue orbit, with Gaia BH1 and BH2 marked. |

Captions say how to read each figure and never state a conclusion about the star.

## Colours

From cora.earth's style sheet (2026-10-30), in `style.TOKENS`:

| Role | Light | Dark |
|---|---|---|
| figure background (`box`) | #f5f5f5 | #1b1b1b |
| ordinary stars: density from the page colour through `blue` | #4b64a3 | #93a8dc |
| the star a figure is about (`orange`) | #e56900 | #fd923e |
| Gaia G / BP / RP | ink / blue / red #b2443f | ink / blue / red #e48a83 |
| ZTF g / r / i | blue / red / amber #9a6814 | blue / red / amber #e0b462 |
| NEOWISE W1 / W2 | purple #7d52b0 / amber | purple #bb9de6 / amber |
| known classes | ink dots, one class per panel | same |

Pairs that share a panel were checked with the dataviz skill's palette validator: blue/red, purple/amber and
blue/orange pass every check in light mode; in dark mode they pass the colour-blind and contrast checks but are
lighter than its suggested band, which is how the site's dark theme is meant to look. Green with amber or orange
fails, so green is not used.

## DR3 caveat

DR3 published light curves only for the 11.6 million stars DPAC flagged as variable, so in the DR3 gallery
"ordinary stars" are those, and three in four of them carry a DPAC class. At G = 15 the classed ones scatter about
three times more from visit to visit than the rest, so the grey noise band uses stars below threshold with no
class. In DR4 every star has a light curve and the comparison sample is the whole sky.

## Log

- 2026-10-30: built; gallery of six real DR3 stars (spotted, eclipsing binary, quasar, Be, long-period, R CrB),
  two injected fakes (1% a year grey fade, 2% grey dimming episode over 300 days), a short dip on an eclipsing
  binary and Gaia BH3 from the DR4 pre-release. Reference data: /mnt/project-files/viz/ref-dr3/.
