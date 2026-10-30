"""Light curves of one source, with the detected change and the visit-to-visit scatter of non-variable stars of the
same G marked.

    fig = slow_change(epochs, change, others=others, noise=0.002, name="Gaia DR3 1807455283392608768")
    fig = short_dip(epochs, t_dip=1234.5, name=...)

Flux is shown as ΔF/F about the median in percent, brighter up, against decimal year; beyond 40% the axes switch to
magnitudes (brighter up).
"""

import numpy as np
import pandas as pd

from cosmos_search.viz import data, style
from cosmos_search.viz.style import c, px

KIND_TEXT = {
    "dip": "dip: {days:.0f} d, depth {depth:.2g}%",
    "fade": "fade from {start:.1f}: depth {depth:.2g}%",
    "drop": "drop at {start:.1f}: depth {depth:.2g}%",
    "trend": "trend: −{slope:.2g}% yr⁻¹",
}
KIND_TEXT_MAG = {   # on magnitude axes the depth is read off the plotted visits, in magnitudes
    "dip": "dip: {days:.0f} d, ΔG = {depth:.1f} mag",
    "fade": "fade from {start:.1f}: ΔG = {depth:.1f} mag",
    "drop": "drop at {start:.1f}: ΔG = {depth:.1f} mag",
    "trend": "trend: −{slope:.2g}% yr⁻¹",
}


def _pct(x):
    return 100.0 * np.asarray(x, float)


MAG_ABOVE = 0.4   # beyond a 40% change either way (robust range of G visits), the axes switch to magnitudes


def choose_scale(frac):
    """'percent' for changes of tens of percent, 'mag' beyond (R CrB stars, Miras, outbursts)."""
    f = np.asarray(frac, float)
    f = f[np.isfinite(f)]
    if not len(f):
        return "percent"
    lo, hi = np.percentile(f, (2, 98))
    return "mag" if (hi > MAG_ABOVE or lo < -MAG_ABOVE) else "percent"


def to_scale(frac, err, scale):
    """Fractional change and its error in the axis unit: percent, or magnitudes (fainter positive; the axis is
    inverted so brighter stays up)."""
    f, e = np.asarray(frac, float), np.asarray(err, float)
    if scale == "mag":
        with np.errstate(invalid="ignore", divide="ignore"):
            return -2.5 * np.log10(np.clip(1 + f, 1e-4, None)), 1.0857 * e / np.clip(1 + f, 1e-4, None)
    return 100.0 * f, 100.0 * e


def _depth_in(depth, scale):
    """A fade's depth (fraction) as an axis value: negative percent, or positive magnitudes."""
    if scale == "mag":
        return float(-2.5 * np.log10(max(1 - abs(depth), 1e-4)))
    return -100.0 * abs(depth)


UNIT = {"percent": "ΔF/F (%)", "mag": "Δmag"}


def _limits(values, errs, extra=(), pad=0.12, q=(0.5, 99.5)):
    v = np.asarray(values, float)
    e = np.nan_to_num(np.asarray(errs, float))
    v = v[np.isfinite(v)]
    if not len(v):
        return -1, 1
    lo, hi = np.percentile(v, q)
    lo = min(lo - np.median(e), *extra) if extra else lo - np.median(e)
    hi = max(hi + np.median(e), *extra) if extra else hi + np.median(e)
    span = max(hi - lo, 1e-3)
    return lo - pad * span, hi + pad * span


def _series(ax, t, y, err, colour, label=None, size=5, hover_fmt=None, n=None, zorder=5):
    """Visit means with error bars, points beyond the axis drawn as arrows at its edge."""
    yr = style.year(t)
    lo, hi = sorted(ax.get_ylim())
    top_is_high = ax.get_ylim()[1] > ax.get_ylim()[0]
    inside = (y >= lo) & (y <= hi)
    ax.errorbar(yr[inside], y[inside], yerr=err[inside], fmt="none", ecolor=style.mix(colour, "box", 0.45),
                elinewidth=px(1), capsize=0, zorder=zorder - 1)
    pts = ax.scatter(yr[inside], y[inside], s=px(size) ** 2 * 1.6, c=colour, edgecolors=c("box"),
                     linewidths=px(1), zorder=zorder, label=label)
    if hover_fmt:
        nn = n[inside] if n is not None else np.ones(inside.sum(), int)
        style.hover(pts, [hover_fmt.format(yr=a, y=b, e=d, n=k) for a, b, d, k in
                          zip(yr[inside], y[inside], err[inside], nn)])
    for high, m, ylim in ((True, y > hi, hi), (False, y < lo, lo)):
        if m.any():
            ax.scatter(yr[m], np.full(m.sum(), ylim), marker="^" if high == top_is_high else "v", s=px(6) ** 2,
                       c=colour, edgecolors="none", zorder=zorder, clip_on=False)
    return pts


def _bin(t, f, e, days):
    """Means in bins of `days` (for dense surveys), with errors from the scatter or the propagated errors."""
    k = np.floor((t - t.min()) / days).astype(int)
    df = pd.DataFrame({"k": k, "t": t, "f": f, "e": e})
    g = df.groupby("k")
    out = g.agg(t=("t", "mean"), f=("f", "mean"), sd=("f", "std"), n=("f", "size"), e2=("e", lambda x: np.sqrt(np.sum(x ** 2))))
    err = np.where(out["n"] > 1, out["sd"] / np.sqrt(out["n"]), out["e2"])
    err = np.maximum(err, out["e2"] / out["n"])
    return out["t"].to_numpy(), out["f"].to_numpy(), np.nan_to_num(err), out["n"].to_numpy()


def _mag_depth(t, y, change):
    """How much fainter (mag) the visits in the change's window are than the rest, medians of the plotted values.
    On magnitude axes this replaces the search's depth, which is measured after clipping extreme visits."""
    t0, t1 = change.get("t0"), change.get("t1")
    if t0 is None:
        return np.nan
    inside = (t >= t0) & (t <= (t1 if t1 is not None and change.get("kind") == "dip" else np.inf))
    if not inside.any() or inside.all():
        return np.nan
    return float(np.median(y[inside]) - np.median(y[~inside]))


def _change_label(change, scale="percent", mag_depth=np.nan):
    kind = change.get("kind", "dip")
    t0, t1 = change.get("t0"), change.get("t1")
    depth = _pct(abs(change.get("depth", np.nan)))
    slope = _pct(abs(change.get("slope", np.nan)))
    fmt = KIND_TEXT.get(kind, "{kind}")
    if scale == "mag" and np.isfinite(mag_depth):
        fmt, depth = KIND_TEXT_MAG.get(kind, "{kind}"), mag_depth
    days = (t1 - t0) if t0 is not None and t1 is not None else np.nan
    return fmt.format(kind=kind, days=days, depth=float(depth), slope=float(slope),
                      start=float(style.year(t0)) if t0 is not None else np.nan)


def slow_change(epochs, change=None, others=None, noise=None, release=None, name=None, heading=None, facts=None,
                scale="auto", width=style.WIDTH):
    """Light curves of one candidate for the slow-change search.

    epochs: the star's Gaia epochs (data module); change: dict with kind ('dip', 'fade', 'drop', 'trend'), t0, t1
    (BTJD, the episode window), depth (fraction) and slope (fraction a year) as far as known; others: ZTF and
    NEOWISE (data.others); noise: visit-to-visit scatter of non-variable stars at this G (fraction, from
    data.typical_noise); release: (label, t_end) for the line where Gaia's data end; facts: a line about the star
    for under the title; scale: 'percent', 'mag' or 'auto' (magnitudes for changes beyond MAG_ABOVE)."""
    gv = data.visits(epochs, "G")
    if len(gv) < 3:
        return None
    bp, rp = data.visits(epochs, "BP"), data.visits(epochs, "RP")
    oth = others if others is not None and len(others) else pd.DataFrame(columns=["band", "t", "flux", "err"])
    has_ztf = oth["band"].astype(str).str.startswith("ztf").any()
    has_wise = oth["band"].astype(str).isin(["W1", "W2"]).any()
    rows = [("G", 3.0), ("colour", 2.0)] + ([("ztf", 1.7)] if has_ztf else []) + ([("wise", 1.7)] if has_wise else [])
    heights = [h for _, h in rows]
    height = 92 + 72 * sum(heights) + 26 * len(rows)
    fig = style.figure(width, height)
    top = 1 - 82 / height
    gs = fig.add_gridspec(len(rows), 1, height_ratios=heights, left=78 / width, right=1 - 20 / width,
                          top=top, bottom=46 / height, hspace=0.45)
    axes = {}
    for i, (key, _) in enumerate(rows):
        axes[key] = fig.add_subplot(gs[i, 0], sharex=axes.get("G"))
    if scale == "auto":
        scale = choose_scale(gv["y"])
    unit = "%" if scale == "percent" else " mag"
    style.title(fig, heading or (f"{name}: light curves" if name else "Light curves"),
                sub=facts or "visit means (epochs within 2 d); brighter up")

    t_all = [gv["t"].to_numpy()]
    if len(oth):
        t_all.append(oth["t"].to_numpy(float))
    t_lo, t_hi = min(map(np.min, t_all)), max(map(np.max, t_all))
    x0, x1 = style.year(t_lo) - 0.15, style.year(t_hi) + 0.15

    # G
    ax = axes["G"]
    y, e = to_scale(gv["y"], gv["err"], scale)
    extra = [0.0]
    if change and np.isfinite(change.get("depth", np.nan)) and scale == "percent":   # on magnitudes the visits set it
        extra.append(_depth_in(change["depth"], scale))
    band = None
    if noise and np.isfinite(noise):
        band = 2 * float(to_scale(noise, 0, scale)[0]) if scale == "percent" else 2 * 1.0857 * noise
        extra += [-band, band]
    lo, hi = _limits(y, e, extra)
    ax.set_ylim((hi, lo) if scale == "mag" else (lo, hi))
    ax.set_xlim(x0, x1)
    if band:
        ax.axhspan(-band, band, color=c("hl"), lw=0, zorder=0)
        if 2 * band / (hi - lo) * 72 * 3.0 > 10:   # only where the band is tall enough to read as one
            roomy_high = (hi - band) >= (-band - lo)  # write on the side of the band with more room
            if scale == "mag":                        # inverted axis: high values are at the bottom
                yb, va = (band, "top") if roomy_high else (-band, "bottom")
            else:
                yb, va = (band, "bottom") if roomy_high else (-band, "top")
            style.note(ax, 0.995, yb, "grey: ±2σ visit scatter, non-variable stars at this G",
                       transform=ax.get_yaxis_transform(), ha="right", va=va)
    ax.axhline(0, color=c("rule"), lw=px(1), zorder=1)
    if change:
        t0, t1 = change.get("t0"), change.get("t1")
        if change.get("kind") != "trend" and t0 is not None and t1 is not None:
            for a in axes.values():
                a.axvspan(style.year(t0), style.year(t1), color=style.mix("orange", "box", 0.16), lw=0, zorder=0)
            late = (style.year(t0) - x0) / (x1 - x0) > 0.6   # label from the episode's end when it is late
            label = _change_label(change, scale, _mag_depth(gv["t"].to_numpy(), y, change))
            style.chip(ax, style.year(t1 if late else t0), 1.0, label,
                       transform=ax.get_xaxis_transform(), ha="right" if late else "left", va="bottom")
        show = change.get("show_slope", change.get("kind") == "trend")
        if scale == "mag":   # a line straight in flux only reads as a trend where the change is modest
            show = change.get("kind") == "trend"
        if np.isfinite(change.get("slope", np.nan)) and show:
            tm = np.median(gv["t"])
            tt = np.linspace(gv["t"].min(), gv["t"].max(), 60)
            ff = np.median(gv["y"]) + change["slope"] * (tt - tm) / 365.25
            keep = 1 + ff > 0.05
            ax.plot(style.year(tt[keep]), to_scale(ff[keep], 0, scale)[0], color=c("orange"), lw=px(2), zorder=6,
                    solid_capstyle="butt")
            if change.get("kind") == "trend":
                style.chip(ax, style.year(tt[0]), 1.0, _change_label({**change, "kind": "trend"}),
                           transform=ax.get_xaxis_transform(), va="bottom")
    _series(ax, gv["t"].to_numpy(), y, e, c("fg"),
            hover_fmt="{yr:.2f}: {y:+.2f}" + unit + " ± {e:.2f}" + unit + " ({n} epochs)", n=gv["n"].to_numpy())
    ax.set_ylabel("G, " + UNIT[scale])

    # BP and RP
    ax = axes["colour"]
    sc = {b: to_scale(v["y"], v["err"], scale) for v, b in ((bp, "BP"), (rp, "RP")) if len(v)}
    vals = np.concatenate([v[0] for v in sc.values()]) if sc else np.zeros(1)
    errs = np.concatenate([v[1] for v in sc.values()]) if sc else np.zeros(1)
    lo, hi = _limits(vals, errs, [0.0] + extra[1:2])
    ax.set_ylim((hi, lo) if scale == "mag" else (lo, hi))
    ax.axhline(0, color=c("rule"), lw=px(1), zorder=1)
    for v, band in ((bp, "BP"), (rp, "RP")):
        if band in sc:
            _series(ax, v["t"].to_numpy(), sc[band][0], sc[band][1], c(style.BAND_TOKEN[band]),
                    label=style.BAND_LABEL[band], size=4.5,
                    hover_fmt=band + " {yr:.2f}: {y:+.2f}" + unit + " ± {e:.2f}" + unit, n=v["n"].to_numpy())
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.2,
              borderaxespad=0.1)
    ax.set_ylabel(UNIT[scale])
    style.panel_label(ax, "Gaia BP (330–680 nm), RP (630–1050 nm)")

    for key, bands, label, days in (("ztf", ("ztf_g", "ztf_r", "ztf_i"), "ZTF g, r, i (15 d means)", 15.0),
                                    ("wise", ("W1", "W2"), "NEOWISE W1, W2 (visit means)", None)):
        if key not in axes:
            continue
        ax = axes[key]
        series = []
        for b in bands:
            d = oth[oth["band"] == b]
            if not len(d):
                continue
            t, f, er = d["t"].to_numpy(float), d["flux"].to_numpy(float), d["err"].to_numpy(float)
            n = np.ones(len(t), int)
            if days:
                t, f, er, n = _bin(t, f, er, days)
            series.append((b, t, *to_scale(f - 1, er, scale), n))
        vals = np.concatenate([s[2] for s in series])
        errs = np.concatenate([s[3] for s in series])
        lo, hi = _limits(vals, errs, [0.0])
        ax.set_ylim((hi, lo) if scale == "mag" else (lo, hi))
        ax.axhline(0, color=c("rule"), lw=px(1), zorder=1)
        for b, t, f, er, n in series:
            _series(ax, t, f, er, c(style.BAND_TOKEN[b]), label=style.BAND_LABEL[b], size=4,
                    hover_fmt=style.BAND_LABEL[b] + " {yr:.2f}: {y:+.2f}" + unit + " ± {e:.2f}" + unit + " ({n} points)",
                    n=n)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(series), handletextpad=0.2,
                  columnspacing=1.2, borderaxespad=0.1)
        ax.set_ylabel(UNIT[scale])
        style.panel_label(ax, label)

    if release and len(oth) and oth["t"].max() > release[1]:
        for a in axes.values():
            a.axvline(style.year(release[1]), color=c("muted"), lw=px(1), ls=(0, (3, 3)), zorder=2)
        style.note(axes[rows[-1][0]], style.year(release[1]), 0.0, f" end of {release[0]}",
                   transform=axes[rows[-1][0]].get_xaxis_transform(), ha="left", va="bottom")
    for key, a in axes.items():
        if key != rows[-1][0]:
            a.tick_params(labelbottom=False)
    style.year_axis(axes[rows[-1][0]])
    return fig


def short_dip(epochs, t_dip, window_hours=30.0, noise=None, name=None, heading=None, width=style.WIDTH):
    """A short dip: the whole G light curve with the dip's epochs marked, and a zoom on the hours around it in
    every band. epochs as for slow_change, t_dip the deepest epoch's time (BTJD)."""
    ep = epochs.copy()
    ep["code"] = data.band_of(ep["band"])
    g = ep[ep["code"] == 0].sort_values("t")
    if len(g) < 3:
        return None
    height = 470
    fig = style.figure(width, height)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], left=78 / width, right=1 - 16 / width, top=1 - 92 / height,
                          bottom=52 / height, wspace=0.22)
    a0, a1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    style.title(fig, heading or (f"{name}: short dip" if name else "Short dip"),
                sub="single epochs; brighter up")
    near = np.abs(ep["t"] - t_dip) * 24 < window_hours
    y = _pct(g["frac"])
    a0.set_ylim(*_limits(y, _pct(g["err_frac"]), [0.0, float(_pct(ep.loc[near, "frac"]).min())], q=(0, 100)))
    if noise and np.isfinite(noise):
        a0.axhspan(-2 * _pct(noise), 2 * _pct(noise), color=c("hl"), lw=0, zorder=0)
    a0.axhline(0, color=c("rule"), lw=px(1), zorder=1)
    gm = near.loc[g.index].to_numpy()
    pts = a0.scatter(style.year(g["t"][~gm]), y[~gm], s=px(4) ** 2 * 1.6, c=c("fg"), edgecolors=c("box"),
                     linewidths=px(0.75), zorder=5)
    style.hover(pts, [f"{a:.3f}: {b:+.2f}%" for a, b in zip(style.year(g["t"][~gm]), y[~gm])])
    a0.scatter(style.year(g["t"][gm]), y[gm], s=px(6) ** 2 * 1.6, c=c("orange"), edgecolors=c("box"),
               linewidths=px(1), zorder=6)
    yd = float(y[gm].min()) if gm.any() else 0.0
    late = (style.year(t_dip) - a0.get_xlim()[0]) / np.ptp(a0.get_xlim()) > 0.5
    style.chip(a0, style.year(t_dip), yd, "dip", ha="right" if late else "left", va="center",
               dx=-14 if late else 14)
    a0.set_ylabel("G, ΔF/F (%)")
    style.year_axis(a0)
    style.panel_label(a0, "Gaia G, all epochs")

    z = ep[near].copy()
    z["h"] = (z["t"] - t_dip) * 24
    for code, band in ((1, "BP"), (2, "RP"), (0, "G")):
        d = z[z["code"] == code]
        if not len(d):
            continue
        col = c(style.BAND_TOKEN[band])
        a1.errorbar(d["h"], _pct(d["frac"]), yerr=_pct(d["err_frac"]), fmt="none",
                    ecolor=style.mix(col, "box", 0.45), elinewidth=px(1), zorder=3)
        p = a1.scatter(d["h"], _pct(d["frac"]), s=px(5.5) ** 2 * 1.6, c=col, edgecolors=c("box"),
                       linewidths=px(1), zorder=5, label=style.BAND_LABEL[band])
        style.hover(p, [f"{band} {a:+.1f} h: {b:+.2f}%" for a, b in zip(d["h"], _pct(d["frac"]))])
    a1.axhline(0, color=c("rule"), lw=px(1), zorder=1)
    a1.set_xlim(-window_hours, window_hours)
    a1.set_xlabel("t − t_dip (h)")
    a1.legend(loc="lower right", handletextpad=0.1, borderaxespad=0.3)
    style.panel_label(a1, f"G, BP, RP within ±{window_hours:.0f} h")
    return fig
