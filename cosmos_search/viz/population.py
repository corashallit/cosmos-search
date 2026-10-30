"""Where a candidate sits among ordinary stars and DPAC's known classes.

    how_rare(pop, strength, group="G<19")                 # tail of fading strengths, with the brightening mirror
    known_classes(pop, cand, x="abbe", y="dim_depth")     # small multiples: one known class per panel
    hr_diagram(ref, cand)                                 # colour against absolute brightness
    dip_depth(pop, cand)                                  # deepest single epoch against brightness (short dips)

`pop` is the comparison sample (see data module); `cand` a dict or Series with the same columns for the candidate.
Every panel draws the population in the site's blue ramp and the candidate in orange.
"""

import numpy as np
import pandas as pd

from cosmos_search.viz import data, style
from cosmos_search.viz.style import c, px

DEFAULT_CLASSES = ("ECL", "AGN", "LPV", "RS", "BE|GCAS|SDOR|WR", "YSO")


def _get(cand, key, default=np.nan):
    try:
        v = cand[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if v is None else v


def how_rare(fade, rise, strength, n_stars=None, label="this star", width=style.WIDTH, heading=None, where=""):
    """How many stars fade at least as strongly as the candidate, and how many brighten as strongly.

    fade, rise: fading and brightening strengths of every star scored in the candidate's magnitude range (or of
    the lists, which hold every star past strength 1 in either direction); strength: the candidate's. Strength is
    each score over its threshold (1 = at the threshold), as the lists order stars. Getting brighter is the mirror
    of getting fainter, so its curve is what chance and ordinary variability give at each strength: a fade is worth
    reading where the fading curve stands clear of it. n_stars: how many stars were scored, for the subtitle."""
    f = np.asarray(fade, float)
    r = np.asarray(rise, float)
    f, r = f[np.isfinite(f)], r[np.isfinite(r)]
    lo = max(min(np.percentile(f, 1), np.percentile(r, 1), 1.0) * 0.85, 0.3) if len(f) and len(r) else 0.85
    hi = max(f.max() if len(f) else 1, r.max() if len(r) else 1, strength) * 1.6
    at = np.logspace(np.log10(lo), np.log10(hi), 200)
    nf, nr = data.survival(f, at), data.survival(r, at)
    height = 400
    fig = style.figure(width, height)
    ax = fig.add_axes([80 / width, 76 / height, 1 - 104 / width, 1 - 174 / height])
    n = n_stars or max(len(f), len(r))
    style.title(fig, heading or "How rare is a change this strong?",
                sub=f"{n:,} stars{' ' + where if where else ''} scored; 1 on the bottom axis is the threshold")
    for counts, col, ls in ((nr, c("muted"), (0, (4, 3))), (nf, c("blue"), "-")):
        m = counts > 0
        ax.step(at[m], counts[m], where="post", color=col, lw=px(2), ls=ls, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style.plain_log(ax.xaxis)
    style.plain_log(ax.yaxis)
    for share in (0.2, 0.42, 0.08):   # where the curves are labelled: clear of the candidate's label
        xl = at[int(len(at) * share)]
        if abs(np.log10(xl / strength)) >= 0.3:
            break
    kf, kr = int(data.survival(f, [xl])[0]), int(data.survival(r, [xl])[0])
    ax.text(xl, max(kf, 1) * 1.5, "getting fainter", color=c("fg"), ha="left", va="bottom", fontsize=px(style.FONT_PX))
    ax.text(xl, max(kr, 1) / 1.8, "getting brighter", color=c("muted"), ha="right", va="top",
            fontsize=px(style.FONT_PX))
    fig.text(80 / width, 10 / height, "getting brighter mirrors getting fainter: it shows what chance and ordinary "
             "variability give", color=c("muted"), fontsize=px(style.FONT_PX), ha="left", va="bottom")
    ax.axvline(1, color=c("rule"), lw=px(1), zorder=0)
    k, kb = int(data.survival(f, [strength])[0]), int(data.survival(r, [strength])[0])
    y = max(k, 1)
    ax.plot([strength, strength], [0.6, y], color=c("orange"), lw=px(2), zorder=6, solid_capstyle="butt")
    style.candidate_marker(ax, strength, y)
    top = max(nf.max(), nr.max(), 2) * 3
    text = f"{label}: {strength:.3g}× threshold"
    if np.log(strength / lo) / np.log(hi / lo) < 0.5:   # room on the right, where the curves fall away
        style.chip(ax, strength * 1.12, min(y * 1.8, top / 4), text, ha="left", va="bottom")
    else:
        style.chip(ax, strength / 1.12, min(y * 4, top / 4), text, ha="right", va="bottom")
    style.note(ax, 0.99, 0.97, f"as strong or stronger: {k:,} fading, {kb:,} brightening", ha="right", va="top",
               color=c("fg"))
    ax.set_ylim(0.6, top)
    ax.set_xlim(at[0], at[-1])
    ax.set_xlabel("strength (score ÷ threshold)")
    ax.set_ylabel("stars at least this strong")
    return fig


def known_classes(pop, cand, x="abbe", y="dim_depth", classes=DEFAULT_CLASSES, class_col="best_class_name",
                  x_label="smoothness: Abbe value (near 0: one smooth change; 1: jumps about like noise)",
                  y_label="depth of the change (%)", log_x=False, log_y=True, y_scale=100.0, ncols=3,
                  max_dots=600, width=style.WIDTH, heading=None, seed=0):
    """Small multiples: every comparison star in the ramp, one known class per panel as ink dots, the candidate in
    orange in every panel. Shows which known kind of variable, if any, behaves like the candidate on two numbers.
    y_scale multiplies the y column (100 for fractions shown in percent)."""
    rng = np.random.default_rng(seed)
    xs = pop[x].to_numpy(float)
    ys = pop[y].to_numpy(float) * y_scale
    cx, cy = float(_get(cand, x)), float(_get(cand, y)) * y_scale
    ok = np.isfinite(xs) & np.isfinite(ys)
    if log_x:
        ok &= xs > 0
    if log_y:
        ok &= ys > 0

    def lims(v, log, extra):
        lo, hi = np.percentile(v, (0.5, 99.5))
        if np.isfinite(extra) and (extra > 0 or not log):
            lo, hi = min(lo, extra), max(hi, extra)
        if log:
            return 10 ** (np.log10(lo) - 0.08), 10 ** (np.log10(hi) + 0.08)
        pad = 0.04 * (hi - lo)
        return lo - pad, hi + pad

    xl, yl = lims(xs[ok], log_x, cx), lims(ys[ok], log_y, cy)
    bx = np.linspace(*(np.log10(xl) if log_x else xl), 60)
    by = np.linspace(*(np.log10(yl) if log_y else yl), 46)
    classes = [k for k in classes if (pop[class_col] == k).any()] if class_col in pop else []
    nrows = int(np.ceil(len(classes) / ncols)) or 1
    left, right, gap = 70, 16, 14
    pw = (width - left - right - gap * (ncols - 1)) / ncols
    ph = pw * 0.75
    top, rowgap = 76, 36
    height = int(top + nrows * (ph + rowgap) + 50)
    fig = style.figure(width, height)
    style.title(fig, heading or "Does it behave like a known kind of variable?",
                sub="blue: all stars compared · dots: one DPAC class per panel · orange: the candidate")
    for i, k in enumerate(classes):
        r, col = divmod(i, ncols)
        x0 = (left + col * (pw + gap)) / width
        y0 = 1 - (top + rowgap + r * (ph + rowgap) + ph) / height
        ax = fig.add_axes([x0, y0, pw / width, ph / height])
        style.density(ax, xs[ok], ys[ok], (bx, by), log_x=log_x, log_y=log_y)
        m = ok & (pop[class_col] == k).to_numpy()
        idx = np.flatnonzero(m)
        if len(idx) > max_dots:
            idx = rng.choice(idx, max_dots, replace=False)
        ax.scatter(xs[idx], ys[idx], s=px(2.6) ** 2 * 1.6, c=c("fg"), edgecolors="none", alpha=0.7, zorder=5,
                   rasterized=True)
        if np.isfinite(cx) and np.isfinite(cy):
            style.candidate_marker(ax, cx, cy, size=8)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        style.panel_label(ax, data.CLASS_SHORT.get(k, data.CLASS_NAMES.get(k, k)))
        style.note(ax, 0.98, 0.96, f"{int(m.sum()):,}", ha="right", va="top")
        if col:
            ax.tick_params(labelleft=False)
        if i + ncols < len(classes):
            ax.tick_params(labelbottom=False)
    fig.text(left / width, 8 / height, x_label, color=c("muted"), fontsize=px(style.FONT_PX), ha="left",
             va="bottom")
    fig.text(12 / width, 1 - (top + rowgap + (nrows * (ph + rowgap) - rowgap) / 2) / height, y_label,
             color=c("muted"), fontsize=px(style.FONT_PX), rotation=90, ha="left", va="center")
    return fig


REGIONS = (  # where the main features of a colour-magnitude diagram sit, for plain labels (BP-RP, M_G, align)
    (3.25, 10.2, "main sequence", "left"),
    (2.65, -1.2, "giants", "left"),
    (0.6, 14.4, "white dwarfs", "left"),
)


def hr_diagram(ref, cand, width=style.WIDTH, heading=None, regions=REGIONS, label="candidate"):
    """Colour against absolute G magnitude (the Hertzsprung-Russell diagram as Gaia sees it, no extinction
    correction) for a reference sample with parallaxes better than 10%, and the candidate with its error. ref and
    cand need phot_g_mean_mag, bp_rp, parallax (mas); cand also parallax_error."""
    r = ref[(ref["parallax"] > 0) & ref["bp_rp"].notna()]
    mg = r["phot_g_mean_mag"] + 5 * np.log10(r["parallax"] / 100.0)
    height = 560
    fig = style.figure(width, height)
    ax = fig.add_axes([74 / width, 56 / height, 1 - 100 / width, 1 - 150 / height])
    style.title(fig, heading or "What kind of star is it?",
                sub=f"{len(r):,} Gaia stars with distances better than 10%, by colour and true brightness")
    style.density(ax, r["bp_rp"], mg, (np.linspace(-0.6, 4.2, 150), np.linspace(-5, 17, 160)))
    for bx, my, text, ha in regions:
        style.note(ax, bx, my, text, transform=ax.transData, ha=ha, va="center")
    g, col = float(_get(cand, "phot_g_mean_mag")), float(_get(cand, "bp_rp"))
    plx, plx_e = float(_get(cand, "parallax")), float(_get(cand, "parallax_error"))
    ax.set_xlim(-0.6, 4.2)
    ax.set_ylim(17, -5)
    if np.isfinite(plx) and np.isfinite(plx_e) and plx > 3 * plx_e:
        m = g + 5 * np.log10(plx / 100.0)
        lo = g + 5 * np.log10((plx - plx_e) / 100.0)
        hi = g + 5 * np.log10((plx + plx_e) / 100.0)
        ax.plot([col, col], [lo, hi], color=c("orange"), lw=px(2), zorder=17, solid_capstyle="butt")
        style.candidate_marker(ax, col, m)
        style.chip(ax, col, m, label, ha="left", dx=14)
    else:
        why = "no parallax" if not np.isfinite(plx) else f"parallax {plx:.2f} ± {plx_e:.2f} mas"
        fig.text(14 / width, 1 - 58 / height, f"{label}: {why}, too uncertain to place it on the diagram; the line "
                 "marks its colour", ha="left", va="top", fontsize=px(style.FONT_PX), color=c("fg"))
        ax.axvline(col, color=c("orange"), lw=px(2), zorder=17)
        style.chip(ax, col, 0.97, f"{label}'s colour", transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.set_xlabel("colour, BP − RP (bluer ← → redder)")
    ax.set_ylabel("absolute G magnitude (brighter up)")
    return fig


def dip_depth(pop, cand, depth="min_frac", g_col="phot_g_mean_mag", classes=("ECL",), class_col="best_class_name",
              width=style.WIDTH, heading=None, label="candidate"):
    """Short dips: the deepest single epoch of every comparison star (fractional drop, percent, log) against its
    brightness, eclipsing binaries as ink dots, and the candidate. Shows how deep a one-epoch dip ordinary stars
    of the same brightness reach."""
    g = pop[g_col].to_numpy(float)
    dd = -100.0 * pop[depth].to_numpy(float) if np.nanmedian(pop[depth]) < 0 else 100.0 * pop[depth].to_numpy(float)
    ok = np.isfinite(g) & np.isfinite(dd) & (dd > 0)
    height = 440
    fig = style.figure(width, height)
    ax = fig.add_axes([74 / width, 56 / height, 1 - 100 / width, 1 - 150 / height])
    style.title(fig, heading or "How deep do ordinary stars dip?",
                sub=f"the deepest single epoch of {ok.sum():,} stars, against their brightness")
    style.density(ax, g[ok], dd[ok], (np.linspace(np.nanpercentile(g[ok], 0.2), np.nanpercentile(g[ok], 99.8), 90),
                                      np.linspace(-1.5, 2.0, 80)), log_y=True)
    if class_col in pop:
        for k in classes:
            m = ok & (pop[class_col] == k).to_numpy()
            idx = np.flatnonzero(m)[:1500]
            ax.scatter(g[idx], dd[idx], s=px(2.4) ** 2 * 1.6, c=c("fg"), edgecolors="none", alpha=0.6, zorder=5,
                       rasterized=True, label=data.CLASS_NAMES.get(k, k))
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.1, handletextpad=0.2)
    cg = float(_get(cand, g_col))
    cd = float(_get(cand, depth))
    cd = -100 * cd if cd < 0 else 100 * cd
    style.candidate_marker(ax, cg, cd)
    style.chip(ax, cg, cd, label, ha="left", dx=14)
    ax.set_xlabel("G magnitude (fainter →)")
    ax.set_ylabel("deepest single epoch (% fainter)")
    return fig
