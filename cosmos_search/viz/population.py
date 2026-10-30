"""A candidate against the comparison sample and the DPAC variability classes.

    strength_hist(fade, rise, strength, n, bulk)         # mirrored histogram: fading up, brightening down
    similar_classes(top, cand)                            # DPAC classes of the most similar strong fades
    hr_diagram(ref, cand)                                 # colour against absolute brightness
    dip_depth(pop, cand)                                  # deepest single epoch against brightness (short dips)

`pop` is the comparison sample (see data module); `cand` a dict or Series with the same columns for the candidate.
Every panel draws the population in the site's blue ramp and the candidate in orange.
"""

import numpy as np
import pandas as pd

from cosmos_search.viz import data, style
from cosmos_search.viz.style import c, px

SIMILAR_CLASSES = ("ECL", "AGN", "LPV", "RS", "BE|GCAS|SDOR|WR", "YSO", "CV", "RCB")


def _get(cand, key, default=np.nan):
    try:
        v = cand[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if v is None else v


def strength_hist(fade, rise, strength, n_stars=None, bulk=None, label="candidate", width=style.WIDTH,
                  heading=None, where="", lo=0.03, per_decade=10):
    """Fading and brightening strength of every star scored, as one mirrored histogram: fading above the axis,
    brightening (the same statistic with the sign reversed, so a false-alarm estimate) below it, counts on log
    scales, the candidate as an orange line with its rank among the fades.

    fade, rise: strengths of every star at or above the threshold (1) in the candidate's magnitude range, from the
    lists. bulk: (fade, rise) strengths of a random sample of the same stars, scaled to n_stars for the bins below
    the threshold, which the lists do not hold. Strength is score / threshold; stars below `lo` are not shown."""
    f = np.asarray(fade, float)
    r = np.asarray(rise, float)
    f, r = f[np.isfinite(f) & (f >= 1)], r[np.isfinite(r) & (r >= 1)]
    n = n_stars or (len(f) + len(r))
    top_s = max(f.max() if len(f) else 1, r.max() if len(r) else 1, strength)
    k0, k1 = int(np.floor(np.log10(lo) * per_decade)), int(np.ceil(np.log10(top_s * 1.05) * per_decade)) + 1
    edges = 10 ** (np.arange(k0, k1 + 1) / per_decade)   # 1 is an edge, so no bin straddles the threshold
    above = edges[:-1] >= 1 - 1e-9
    counts = {}
    for key, tail, b in (("fade", f, 0), ("rise", r, 1)):
        c_ = np.histogram(tail, edges)[0].astype(float)
        if bulk is not None:
            smp = np.asarray(bulk[b], float)
            smp = smp[np.isfinite(smp)]
            if len(smp):
                c_ = np.where(above, c_, np.histogram(smp, edges)[0] * n / len(smp))
        counts[key] = c_

    height = 460
    fig = style.figure(width, height)
    left, right, top, bottom = 84, 24, 104, 64
    ph = (height - top - bottom) / 2
    a_up = fig.add_axes([left / width, (bottom + ph) / height, 1 - (left + right) / width, ph / height])
    a_dn = fig.add_axes([left / width, bottom / height, 1 - (left + right) / width, ph / height], sharex=a_up)
    rank = int(np.sum(f > strength)) + 1
    k, kb = int(np.sum(f >= strength)), int(np.sum(r >= strength))
    style.title(fig, heading or "Fade strength vs all stars scored",
                sub=f"{n:,} stars{', ' + where if where else ''} · fading above the axis, brightening below")
    ymax = max(counts["fade"].max(), counts["rise"].max(), 2) * 2.5
    for ax, key, col, text in ((a_up, "fade", c("blue"), "fading"), (a_dn, "rise", c("muted"), "brightening")):
        cnt = counts[key]
        for sel, shade in ((~above, 0.45), (above, 1.0)):   # below threshold lighter
            m = sel & (cnt > 0)
            bars = ax.bar(edges[:-1][m], cnt[m] - 0.5, width=np.diff(edges)[m], bottom=0.5, align="edge",
                          color=style.mix(col, "box", shade) if shade < 1 else col, edgecolor=c("box"),
                          linewidth=px(1), zorder=3)
            style.hover_bars(bars, [f"{text} {a:.3g}–{b:.3g}: {v:,.0f} stars" for a, b, v in
                                    zip(edges[:-1][m], edges[1:][m], cnt[m])])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.5, ymax)
        style.plain_log(ax.yaxis)
        ax.set_yticks([v for v in 10.0 ** np.arange(0, 10, 2) if v < ymax and (ax is a_up or v > 1)])
        style.note(ax, 0.01, 0.92 if ax is a_up else 0.08, text, color=c("fg"), ha="left",
                   va="top" if ax is a_up else "bottom", fontweight="bold")
        ax.axvline(1, color=c("fg"), lw=px(1), ls=(0, (4, 3)), zorder=4)
        ax.axvline(strength, color=c("orange"), lw=px(2), zorder=6)
    a_dn.invert_yaxis()
    a_up.tick_params(labelbottom=False, bottom=False)
    a_up.spines["bottom"].set_visible(False)
    a_dn.spines["top"].set_visible(True)
    a_dn.spines["top"].set_color(c("rule"))
    a_dn.set_xlim(lo, edges[-1])
    style.plain_log(a_dn.xaxis)
    a_dn.set_xticks([v for v in 10.0 ** np.arange(-3, 5) if lo <= v <= edges[-1]])
    a_dn.set_xticklabels([("1 = threshold" if v == 1 else style._plain_number(v)) for v in a_dn.get_xticks()])
    late = np.log(strength / lo) / np.log(edges[-1] / lo) > 0.6
    style.chip(a_up, strength, 1.0, f"{label}: {strength:.3g}, rank {rank:,} of {n:,}",
               transform=a_up.get_xaxis_transform(), ha="right" if late else "left", va="bottom",
               dx=-4 if late else 4, dy=4)
    style.note(a_dn, strength, 0.0, f"≥ {strength:.3g}: {k:,} fading, {kb:,} brightening",
               transform=a_dn.get_xaxis_transform(), ha="right" if late else "left", va="bottom", color=c("fg"),
               fontsize=px(style.FONT_PX))
    a_dn.set_xlabel("strength (score / threshold)")
    fig.text(14 / width, (bottom + ph) / height, "stars per bin", rotation=90, ha="left", va="center",
             color=c("muted"), fontsize=px(style.FONT_PX))
    if bulk is not None:
        fig.text(left / width, 8 / height, "lighter bars: below threshold, scaled from the comparison sample",
                 color=c("muted"), fontsize=px(style.FONT_PX), ha="left", va="bottom")
    return fig


def _pct(v):
    """A share in percent, to two significant figures below 10%."""
    if v <= 0:
        return "0"
    return f"{v:.0f}%" if v >= 9.5 else f"{v:.2g}%"


def similar_classes(top, cand, k=100, classes=SIMILAR_CLASSES, class_col="best_class_name", exclude=None,
                    label="candidate", where="", width=style.WIDTH, heading=None):
    """DPAC classes among the k comparison stars nearest the candidate, against each class's share among all
    comparison stars: one row per class, shares on a log scale, an open dot for all and an orange dot for the
    nearest, with both shares in a table at the right.

    top: the comparison stars (the strongest fades of the run, data module) with the class in class_col; cand: the
    candidate's data.SIMILAR_COLUMNS; exclude: the candidate's source id, if it is among the comparison stars.
    Distance is Euclidean over data.similarity_features, each scaled by its spread (1.4826 MAD) over the
    comparison stars; features the candidate lacks are left out. A class whose share among the nearest is far
    above its share among all is the known type the candidate is closest to. The footnote compares the median
    distance to the candidate's k nearest with the same for every comparison star."""
    from scipy.spatial import cKDTree

    ft = data.similarity_features(top)
    fc = data.similarity_features(pd.DataFrame([{k_: _get(cand, k_) for k_ in data.SIMILAR_COLUMNS}])).iloc[0]
    use = [f for f in ft.columns if np.isfinite(fc[f])]
    ok = np.isfinite(ft[use].to_numpy(float)).all(axis=1)
    ids = top["id"] if "id" in top else top.get("source_id")
    if exclude is not None and ids is not None:
        ok &= ids.to_numpy() != int(exclude)
    z = ft.loc[ok, use].to_numpy(float)
    mid = np.median(z, axis=0)
    spread = 1.4826 * np.median(np.abs(z - mid), axis=0)
    spread = np.where(spread > 0, spread, z.std(axis=0) + 1e-12)
    z = (z - mid) / spread
    zc = (fc[use].to_numpy(float) - mid) / spread
    k = min(k, len(z) - 1)
    tree = cKDTree(z)
    dc, ic = tree.query(zc, k)
    typical = np.median(tree.query(z, k + 1)[0][:, 1:], axis=1)
    farther = float(np.mean(typical < np.median(dc)))

    raw = top.loc[ok, class_col] if class_col in top else pd.Series(np.nan, index=top.index[ok])
    missing = (raw.isna() | raw.astype(object).isin(["nan", "None", ""])).to_numpy()
    raw = raw.astype(object).to_numpy()
    lab = np.where(missing, "none", np.where(np.isin(raw, classes), raw, "other"))
    rows = [k_ for k_ in (*classes, "other", "none") if (lab == k_).any()]
    names = {**{k_: data.CLASS_SHORT.get(k_, data.CLASS_NAMES.get(k_, k_)) for k_ in classes},
             "other": "other DPAC classes", "none": "no DPAC class"}
    n_all = {k_: int(np.sum(lab == k_)) for k_ in rows}
    n_near = {k_: int(np.sum(lab[ic] == k_)) for k_ in rows}
    s_all = {k_: 100 * n_all[k_] / len(lab) for k_ in rows}
    s_near = {k_: 100 * n_near[k_] / k for k_ in rows}

    row_h, top_px, bottom = 30, 104, 104
    left, table = 196, 176
    height = int(top_px + row_h * len(rows) + bottom)
    fig = style.figure(width, height)
    ax = fig.add_axes([left / width, bottom / height, 1 - (left + table) / width, row_h * len(rows) / height])
    style.title(fig, heading or "DPAC classes of the most similar stars",
                sub=f"{k} nearest of {len(lab):,} comparison stars: the strongest fades{', ' + where if where else ''}")
    y = np.arange(len(rows))[::-1].astype(float)
    lo = 10 ** np.floor(np.log10(min(v for v in s_all.values() if v > 0)) - 0.3)
    for yi, k_ in zip(y, rows):
        ax.axhline(yi, color=c("rule"), lw=px(1), zorder=1)
        a, b = s_all[k_], s_near[k_]
        if b > 0:
            ax.plot([a, b], [yi, yi], color=c("muted"), lw=px(2), zorder=2, solid_capstyle="butt")
        dot = ax.scatter([a], [yi], s=px(9) ** 2, facecolors=c("box"), edgecolors=c("muted"), linewidths=px(2),
                         zorder=4)
        style.hover(dot, [f"{names[k_]}: {n_all[k_]:,} of {len(lab):,} comparison stars ({_pct(a)})"])
        if b > 0:
            dot = style.candidate_marker(ax, b, yi, size=9)
            style.hover(dot, [f"{names[k_]}: {n_near[k_]} of the {k} nearest ({_pct(b)})"])
    ax.set_xscale("log")
    ax.set_xlim(lo, 130)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    style.plain_log(ax.xaxis)
    ax.set_xticks([v for v in 10.0 ** np.arange(-4, 3) if lo <= v <= 100])
    ax.set_xticklabels([f"{style._plain_number(v)}%" for v in ax.get_xticks()])
    ax.set_yticks(y)
    ax.set_yticklabels([names[k_] for k_ in rows], color=c("fg"))
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("share of stars", labelpad=2)
    # the shares as a table at the right
    xa, xb = 1 - (table - 70) / width, 1 - 14 / width
    for x_, head in ((xa, f"{k} nearest"), (xb, "all")):
        fig.text(x_, (bottom + row_h * len(rows) + 16) / height, head, ha="right", va="center",
                 fontsize=px(style.FONT_PX), color=c("muted"))
    for yi, k_ in zip(y, rows):
        yy = (bottom + row_h * (yi + 0.5)) / height
        fig.text(xa, yy, _pct(s_near[k_]), ha="right", va="center", fontsize=px(style.FONT_PX), color=c("fg"),
                 fontweight="bold" if s_near[k_] > 0 else "normal")
        fig.text(xb, yy, _pct(s_all[k_]), ha="right", va="center", fontsize=px(style.FONT_PX), color=c("muted"))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker="o", ms=px(9), mfc=c("orange"), mec=c("box"), mew=px(2),
                      label=f"{label}: {k} nearest"),
               Line2D([], [], ls="none", marker="o", ms=px(8), mfc=c("box"), mec=c("muted"), mew=px(2),
                      label="all comparison stars")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-left / (width - left - table), 1.0),
              ncol=2, borderaxespad=0.6, handletextpad=0.3, columnspacing=1.6)
    for i, line in enumerate((f"nearest in {data.SIMILAR_BY},", "each scaled by its spread over the comparison stars",
                              f"median distance to the {k} nearest: larger than for {farther:.0%} of comparison "
                              "stars")):
        fig.text(14 / width, (10 + 18 * (2 - i)) / height, line, color=c("muted"), fontsize=px(style.FONT_PX),
                 ha="left", va="bottom")
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
    style.title(fig, heading or "Colour–magnitude diagram",
                sub=f"{len(r):,} Gaia stars, parallax / error > 10, no extinction correction")
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
        fig.text(14 / width, 1 - 58 / height, f"{label}: {why} (under 3σ); line marks BP − RP only",
                 ha="left", va="top", fontsize=px(style.FONT_PX), color=c("fg"))
        ax.axvline(col, color=c("orange"), lw=px(2), zorder=17)
        style.chip(ax, col, 0.97, f"{label} BP − RP", transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.set_xlabel("BP − RP: blue minus red magnitude (larger is redder)")
    ax.set_ylabel("M_G, absolute G (mag)")
    return fig


def dip_depth(pop, cand, depth="min_frac", g_col="phot_g_mean_mag", classes=("ECL",), class_col="best_class_name",
              width=style.WIDTH, heading=None, label="candidate"):
    """Short dips: the deepest single epoch of every comparison star (fractional drop, percent, log) against its
    brightness, eclipsing binaries as ink dots, and the candidate. Shows how deep a one-epoch dip comparison stars
    of the same brightness reach."""
    g = pop[g_col].to_numpy(float)
    dd = -100.0 * pop[depth].to_numpy(float) if np.nanmedian(pop[depth]) < 0 else 100.0 * pop[depth].to_numpy(float)
    ok = np.isfinite(g) & np.isfinite(dd) & (dd > 0)
    height = 440
    fig = style.figure(width, height)
    ax = fig.add_axes([74 / width, 56 / height, 1 - 100 / width, 1 - 150 / height])
    style.title(fig, heading or "Deepest single-epoch dip vs G",
                sub=f"{ok.sum():,} comparison stars; dots: DPAC eclipsing binaries")
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
    ax.set_xlabel("G (mag)")
    ax.set_ylabel("max single-epoch depth (%)")
    return fig
