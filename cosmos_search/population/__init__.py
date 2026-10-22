"""Population test for single one-visit dips (docs/population.md, the pre-registered plan).

- `split`: visits and the day-hash split into halves A and B.
- `events`: the event rule, vectorised over every star in a light-curve table.
- `exposure`: forward models of the expected number of events per star and half (generic occulter, eclipses).
- `glm`: the Poisson null model on aggregated covariate cells.
- `scan`: sky x distance windows, the likelihood-ratio scan and its parametric null.
- `inject`: sealed excess populations for calibration.
"""
