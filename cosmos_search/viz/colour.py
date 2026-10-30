"""What the change looks like across wavelengths: the colour of a fade in Gaia's bands, and the infrared response.

    change_colour(pop, cand)     # BP depth / RP depth against G depth, with grey, dust and cooling lines
    infrared(cand)               # NEOWISE W1/W2 change during the episode against what each explanation predicts

These are the questions the candidate checks ask (slow/checks.py): an opaque body blocks every band alike, dust in
front dims blue more than red and barely touches the infrared, a dust shell that absorbs the light glows in the
infrared, and a change in the star itself follows its temperature.
"""

import numpy as np

from cosmos_search.slow import vet
from cosmos_search.viz import style
from cosmos_search.viz.style import c, px


def _get(cand, key, default=np.nan):
    try:
        v = cand[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if v is None else v


def band_ratio(flavour, teff=5800.0):
    """BP depth / RP depth that one explanation predicts (slow.vet.ratios)."""
    r = vet.ratios(flavour, teff=teff)
    return r["BP"] / r["RP"]


def change_colour(pop, cand, min_z=5.0, width=style.WIDTH, heading=None, label="candidate"):
    """BP depth over RP depth (y) against the G depth (x, percent, log) for comparison stars whose change is
    measured at min_z sigma in both bands, with the ratios grey blocking, interstellar-type dust and a cooling
    star predict. pop and cand need dim_depth, depth_bp, depth_bp_err, depth_rp, depth_rp_err (fractions); cand
    also bp_rp for the cooling line."""
    d = pop
    bp, rp = d["depth_bp"].to_numpy(float), d["depth_rp"].to_numpy(float)
    ebp, erp = d["depth_bp_err"].to_numpy(float), d["depth_rp_err"].to_numpy(float)
    g = 100 * d["dim_depth"].to_numpy(float)
    ok = (bp > min_z * ebp) & (rp > min_z * erp) & (g > 0)
    ratio = np.where(ok, bp / np.where(rp > 0, rp, np.nan), np.nan)
    height = 460
    fig = style.figure(width, height)
    ax = fig.add_axes([74 / width, 74 / height, 1 - 300 / width, 1 - 168 / height])
    style.title(fig, heading or "Is the change grey, dusty, or the star itself?",
                sub=f"{ok.sum():,} comparison stars whose change is measured in both BP and RP")
    style.density(ax, g[ok], ratio[ok], (np.linspace(-1.0, 2.0, 80), np.linspace(-0.2, 3.2, 70)), log_x=True)
    teff = vet.teff(float(_get(cand, "bp_rp", 1.0)))
    lines = [(1.0, "grey: an opaque body"),
             (band_ratio("dust"), "dust like the interstellar kind"),
             (band_ratio("temperature", teff), f"the star cooling (from {teff:,.0f} K)")]
    for yv, text in lines:
        ax.axhline(yv, color=c("fg"), lw=px(1), ls=(0, (4, 3)), zorder=6)
        ax.text(1.02, yv, text, transform=ax.get_yaxis_transform(), color=c("muted"), va="center", ha="left",
                fontsize=px(style.FONT_PX))
    cb, cr = float(_get(cand, "depth_bp")), float(_get(cand, "depth_rp"))
    cbe, cre = float(_get(cand, "depth_bp_err")), float(_get(cand, "depth_rp_err"))
    cg = 100 * float(_get(cand, "dim_depth"))
    if np.isfinite(cb) and np.isfinite(cr) and cr > 0 and cg > 0:
        cy = cb / cr
        cye = abs(cy) * np.hypot(cbe / cb if cb else np.inf, cre / cr)
        ax.errorbar([cg], [cy], yerr=[min(cye, 5)], fmt="none", ecolor=c("orange"), elinewidth=px(2), zorder=17)
        style.candidate_marker(ax, cg, cy)
        style.chip(ax, cg, cy, label, ha="left", dx=14)
    ax.set_ylim(-0.2, 3.2)
    ax.set_xlabel("depth in G (%)")
    ax.set_ylabel("BP depth ÷ RP depth")
    fig.text(74 / width, 14 / height, "dust grains much larger than a micron block every band alike too",
             color=c("muted"), fontsize=px(style.FONT_PX), ha="left", va="bottom")
    return fig


ROWS = (("W1", "W1, 3.4 µm"), ("W2", "W2, 4.6 µm"))
PREDICTIONS = (("pred_grey", "grey blocking"), ("pred_dust_in_front", "dust in front"),
               ("pred_temperature", "the star cooling"), ("pred_dust_shell_300K", "dust shell glowing, 300 K"),
               ("pred_dust_shell_1000K", "dust shell glowing, 1000 K"))


def infrared(cand, width=style.WIDTH, heading=None, label="measured"):
    """NEOWISE change during the episode in W1 and W2 (percent of normal, brighter positive) against each
    explanation's prediction, one panel per band: the explanation whose dot falls inside the orange band fits.
    From a slow_checks row (W1_change, W1_err, W1_pred_*; same for W2); the cooling-star prediction comes from
    slow.vet.ratios when dim_depth and bp_rp are present. The axis is linear within ±1% and logarithmic beyond."""
    have = [b for b, _ in ROWS if np.isfinite(float(_get(cand, f"{b}_change")))]
    if not have:
        return None
    depth = float(_get(cand, "dim_depth"))
    teff = vet.teff(float(_get(cand, "bp_rp", 1.0)))
    temp = vet.ratios("temperature", teff=teff)
    rows = [text for _, text in PREDICTIONS]
    height = 150 + 30 * len(rows)
    fig = style.figure(width, height)
    style.title(fig, heading or "Does it glow in the infrared?",
                sub="NEOWISE's change over the episode's dates (orange band, ±1σ) against each explanation (dots)")
    left, gap = 190, 26
    pw = (width - left - 20 - gap * (len(have) - 1)) / len(have)
    for i, b in enumerate(have):
        ax = fig.add_axes([(left + i * (pw + gap)) / width, 58 / height, pw / width, 1 - 138 / height])
        m, e = 100 * float(_get(cand, f"{b}_change")), 100 * float(_get(cand, f"{b}_err"))
        ax.axvspan(m - e, m + e, color=style.mix("orange", "box", 0.35), lw=0, zorder=1)
        ax.axvline(m, color=c("orange"), lw=px(2), zorder=2)
        xs = []
        for j, (key, text) in enumerate(PREDICTIONS):
            v = float(_get(cand, f"{b}_{key}"))
            if key == "pred_temperature" and np.isfinite(depth):
                v = -abs(depth) * temp[b]
            yrow = len(rows) - 1 - j
            ax.axhline(yrow, color=c("rule"), lw=px(1), zorder=0)
            if np.isfinite(v):
                inside = abs(100 * v - m) <= e
                p = ax.scatter([100 * v], [yrow], s=px(7) ** 2 * 1.6, c=c("fg") if inside else c("box"),
                               edgecolors=c("fg"), linewidths=px(1.5), zorder=5)
                style.hover(p, [f"{text}: {100 * v:+.2f}%"])
                xs.append(100 * v)
        xs += [m - e, m + e]
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.6)
        lo, hi = min(xs), max(xs)
        ax.set_xlim(lo - 0.3 * abs(lo) - 1, hi + 0.3 * abs(hi) + 1)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(list(reversed(rows)) if i == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        style.plain_log(ax.xaxis)
        style.panel_label(ax, dict(ROWS)[b])
    fig.text(left / width, 12 / height, "change during the episode (%; fainter is negative) · filled dots fit",
             color=c("muted"), fontsize=px(style.FONT_PX), ha="left", va="bottom")
    return fig
