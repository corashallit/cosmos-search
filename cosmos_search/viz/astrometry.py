"""Astrometry figures: along-scan residuals against the single-star model, and orbital solutions against the
Kepler bound.

    along_scan(src, alt)            # residuals per transit: single-star model against the alternative model
    kepler_bound(orbits, cand)      # period against the mass function of every catalogue orbit

Gaia measures each position along its scan direction only, so residuals are along-scan offsets in mas, one per
field-of-view transit (astrometry.epoch.per_transit).
"""

import numpy as np

from cosmos_search.astrometry import epoch as aepoch
from cosmos_search.viz import style
from cosmos_search.viz.style import c, px

KNOWN = {  # Gaia DR3 source ids of known dark companions, for labels on the Kepler-bound plane
    4373465352415301632: "Gaia BH1",
    5870569352746779008: "Gaia BH2",
}


def _year(t_jyear):
    return aepoch.DR4_REF_JYEAR + np.asarray(t_jyear, float)


def along_scan(src, alt, alt_label="orbit", name=None, width=style.WIDTH, heading=None):
    """Two rows sharing time: the along-scan offsets left after removing the star's own motion as `alt` fits it
    (parallax and proper motion), with `alt`'s extra signal at each transit in orange; then what is left after the
    whole of `alt`. src: astrometry.fit.Source; alt: a Fit whose first five columns are the single-star model
    (Source.kepler(), Source.changepoint(), ...)."""
    five = src.five()
    a, x = alt.columns, alt.x
    star = a[:, :5] @ x[:5]
    extra = a[:, 5:] @ x[5:]
    resid_star = src.y - star
    resid_all = src.y - a @ x
    chi_five = five.chi2 / max(src.n - five.k, 1)
    chi_alt = alt.chi2 / max(src.n - alt.k, 1)
    yr = _year(src.t)
    order = np.argsort(yr)
    height = 520
    fig = style.figure(width, height)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], left=78 / width, right=1 - 20 / width, top=1 - 96 / height,
                          bottom=50 / height, hspace=0.42)
    a0 = fig.add_subplot(gs[0, 0])
    a1 = fig.add_subplot(gs[1, 0], sharex=a0)
    style.title(fig, heading or (f"{name}: along-scan residuals" if name else "Along-scan residuals"),
                sub="per-transit AL position minus the 5-parameter part of the fit")
    a0.errorbar(yr, resid_star, yerr=src.s, fmt="none", ecolor=style.mix("fg", "box", 0.35), elinewidth=px(1),
                zorder=3)
    p = a0.scatter(yr, resid_star, s=px(4.5) ** 2 * 1.6, c=c("fg"), edgecolors=c("box"), linewidths=px(0.75),
                   zorder=5, label="measured")
    style.hover(p, [f"{y:.3f}: {r:+.3f} mas ± {s:.3f}" for y, r, s in zip(yr, resid_star, src.s)])
    a0.scatter(yr[order], extra[order], s=px(3) ** 2 * 1.6, c=c("orange"), edgecolors="none", zorder=6,
               label=f"{alt_label} model")
    a0.axhline(0, color=c("rule"), lw=px(1), zorder=1)
    a0.set_ylabel("AL residual (mas)")
    a0.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.1, handletextpad=0.2)
    style.panel_label(a0, f"5-parameter model: χ²/dof = {style.number(chi_five)}")
    a0.tick_params(labelbottom=False)
    a1.errorbar(yr, resid_all, yerr=src.s, fmt="none", ecolor=style.mix("fg", "box", 0.35), elinewidth=px(1),
                zorder=3)
    a1.scatter(yr, resid_all, s=px(4) ** 2 * 1.6, c=c("fg"), edgecolors=c("box"), linewidths=px(0.75), zorder=5)
    a1.axhline(0, color=c("rule"), lw=px(1), zorder=1)
    lim = max(np.percentile(np.abs(resid_all), 99) * 1.3, np.median(src.s) * 3)
    a1.set_ylim(-lim, lim)
    a1.set_ylabel("AL residual (mas)")
    style.year_axis(a1)
    style.panel_label(a1, f"5-parameter + {alt_label}: χ²/dof = {style.number(chi_alt)}")
    return fig


def kepler_bound(orbits, cand=None, known=KNOWN, width=style.WIDTH, heading=None, label="candidate"):
    """Every catalogue orbit by period (days) and mass function f_A (solar masses), both log: f_A is the least
    companion mass the photocentre's orbit needs, so points above the star's own mass need a companion heavier than
    the star that gives no light, and points above 100 solar masses need more than any known dark object. orbits
    needs period, f_A, m1 and source_id (astrometry.kepler_bound.orbits); cand a dict with period and f_A."""
    o = orbits[np.isfinite(orbits["f_A"]) & (orbits["f_A"] > 0) & np.isfinite(orbits["period"])]
    height = 470
    fig = style.figure(width, height)
    ax = fig.add_axes([80 / width, 56 / height, 1 - 110 / width, 1 - 150 / height])
    style.title(fig, heading or "Astrometric mass function vs period",
                sub=f"{len(o):,} Gaia DR3 orbital solutions; f_A = minimum mass of a dark companion")
    style.density(ax, o["period"], o["f_A"], (np.linspace(-0.3, 4.0, 110), np.linspace(-9, 4, 110)),
                  log_x=True, log_y=True)
    dark = o["f_A"] > o["m1"]
    ax.scatter(o.loc[dark, "period"], o.loc[dark, "f_A"], s=px(3) ** 2 * 1.6, c=c("fg"), edgecolors="none",
               zorder=5, rasterized=True, label="f_A > M₁")
    ax.axhline(100, color=c("fg"), lw=px(1), ls=(0, (4, 3)), zorder=4)
    style.note(ax, 0.01, 100, "100 M☉", transform=ax.get_yaxis_transform(), ha="left", va="bottom")
    for sid, text in (known or {}).items():
        k = o[o["source_id"] == sid]
        if len(k):
            px_, py_ = float(k["period"].iloc[0]), float(k["f_A"].iloc[0])
            ax.scatter([px_], [py_], s=px(6) ** 2 * 1.6, facecolors="none", edgecolors=c("fg"), linewidths=px(1.5),
                       zorder=7)
            style.tag(ax, px_, py_, text, ha="left", dx=14)
    if cand is not None:
        cp, cf = float(cand["period"]), float(cand["f_A"])
        style.candidate_marker(ax, cp, cf)
        right = np.log10(cp) > 3.2   # near the right edge: label on the left
        style.chip(ax, cp, cf, cand.get("label", label), ha="right" if right else "left", dx=-14 if right else 14)
    ax.set_xlim(10 ** -0.3, 10 ** 4.0)
    ax.set_ylim(1e-8, 1e4)
    ax.set_xlabel("P (d)")
    ax.set_ylabel("f_A (M☉)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.1, handletextpad=0.2)
    return fig
