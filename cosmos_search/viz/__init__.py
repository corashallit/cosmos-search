"""Release-day figures for candidates, in cora.earth's style (light and dark).

    style        colours, fonts, figure helpers, render() to PNG + SVG in both themes
    data         loading and normalising what the figures draw (epochs, other surveys, list strengths)
    lightcurve   light curves of one star: slow changes (fades, brightenings, dimming episodes) and short dips
    population   candidate vs comparison sample: strength histogram, DPAC classes of similar stars, colour-magnitude
                 diagram, dip depth
    colour       BP/RP depth ratio, NEOWISE W1/W2 change against model predictions
    astrometry   along-scan residuals against an alternative model, and the Kepler bound on orbits
    sheet        the standard set of figures for one candidate, with titles and plain-language captions
    html         a page of figures with the site's theme toggle and hover text

scripts/viz_candidates.py drives these on release day; docs/viz.md says how.
"""
