"""The standard set of figures for one candidate, with titles and captions for a page.

    ctx = Context(pop=pop, lists=lists, hr=hr, top=top, release=("Gaia DR4", 1869.0), label="DR4")
    figs = slow_candidate(ctx, row, epochs, "out/123", others=others, checks=checks_row, astro=astro_row)
    figs += short_dip_candidate(ctx, row, epochs, t_dip, "out/123")

Each entry is {"key", "title", "caption", "paths"} with paths as `style.render` returns them, ready for
`html.page`. Captions say how to read the figure in plain words; they never state a conclusion about the star.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.slow import vet
from cosmos_search.viz import colour, data, lightcurve, population, style


# Definitions for a page of these figures (html.terms), in the order a reader meets them.
TERMS = (
    ("G, BP, RP", "Gaia photometric bands: G 330–1050 nm, BP 330–680 nm (blue), RP 630–1050 nm (red)."),
    ("BP − RP", "Colour index: BP magnitude minus RP magnitude. Larger is redder."),
    ("ΔF/F, Δmag", "Flux change relative to the median flux; the same change in magnitudes (positive = fainter)."),
    ("visit", "Gaia epochs within 2 d of each other, averaged."),
    ("ZTF g, r, i", "Zwicky Transient Facility bands, about 480, 640 and 790 nm."),
    ("NEOWISE W1, W2", "Infrared bands at 3.4 and 4.6 µm (NEOWISE survey, 2013–2024)."),
    ("strength", "Detection score divided by the threshold for its magnitude range; 1 = threshold."),
    ("DPAC class", "Variability class from the Gaia Data Processing and Analysis Consortium "
                   "(vari_classifier_result)."),
    ("Abbe value", "Mean squared difference of consecutive visits over twice the variance: about 1 for "
                   "uncorrelated scatter, near 0 for a smooth change."),
    ("M_G", "Absolute G magnitude, G + 5 log₁₀(parallax / 100 mas)."),
    ("AL", "Along-scan: the direction in which Gaia measures positions."),
    ("f_A", "Astrometric mass function: the minimum mass of an unseen companion, from the orbit size and period."),
)


@dataclass
class Context:
    """What every candidate is compared against, loaded once per run.

    pop: comparison sample (detector columns, phot_g_mean_mag, bp_rp, data.strengths columns, best_class_name);
    lists: {group: (fading strengths, brightening strengths, stars scored)} over the whole run (the lists);
    hr: reference sample for the HR diagram (phot_g_mean_mag, bp_rp, parallax); top: the strongest fades of the run
    with astrometry and DPAC class (data module), for the classes of similar stars; release: (name, last BTJD);
    label: data release name for captions."""
    pop: pd.DataFrame
    lists: dict = field(default_factory=dict)
    hr: pd.DataFrame = None
    top: pd.DataFrame = None
    release: tuple = ("Gaia DR3", 902.0)
    label: str = "DR3"


def _get(row, key, default=np.nan):
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else v


def change_of(row):
    """The detected change of a list row as lightcurve.slow_change wants it."""
    t0, t1, kind = vet.change_window(row)
    by = str(_get(row, "by", ""))
    return {"kind": kind, "t0": t0, "t1": t1, "depth": float(_get(row, "dim_depth")),
            "slope": float(_get(row, "slope")), "show_slope": "trend" in by}


def facts(row, cls=None, astro=None):
    """One line of catalogue values for under the light-curve title."""
    parts = [f"G = {float(_get(row, 'phot_g_mean_mag')):.1f}", f"BP − RP = {float(_get(row, 'bp_rp')):.2f}"]
    if cls:
        parts.append("DPAC class: " + data.CLASS_NAMES.get(cls, cls))
    else:
        parts.append("DPAC class: none")
    if astro is not None and np.isfinite(float(_get(astro, "parallax"))):
        p, e = float(_get(astro, "parallax")), float(_get(astro, "parallax_error"))
        parts.append(f"parallax {p:.2f} ± {e:.2f} mas")
    return " · ".join(parts)


def slow_candidate(ctx, row, epochs, out_dir, others=None, checks=None, astro=None, cls=None, name=None,
                   label="candidate", sid=None, themes=style.THEMES):
    """Figures for one slow-change candidate: light curves, strength distribution, DPAC classes of similar stars,
    colour-magnitude diagram, BP/RP depth ratio and (with NEOWISE in `checks`) the W1/W2 change. row: the star's row from the lists
    (or detector output with the same columns); epochs: its Gaia epochs (data.epochs); sid: its source id when
    the row cannot hold it exactly (a row of floats rounds 19-digit ids)."""
    out_dir = Path(out_dir)
    sid = int(sid if sid is not None else _get(row, "id", _get(row, "source_id", 0)))
    name = name or f"Gaia {ctx.label} {sid}"
    g = float(_get(row, "phot_g_mean_mag"))
    grp = str(data.group([g])[0])
    noise = data.typical_noise(ctx.pop, g) if "sigma_visit" in ctx.pop else None
    entries = []

    def add(key, fn, *args, title, caption, **kw):
        paths = style.render(fn, out_dir / f"{sid}-{key}", *args, themes=themes, **kw)
        if paths:
            entries.append({"key": key, "title": title, "caption": caption, "paths": paths})

    add("lightcurve", lightcurve.slow_change, epochs, change_of(row), others=others, noise=noise,
        release=ctx.release, name=name, facts=facts(row, cls, astro),
        title="Light curves",
        caption="Gaia visit means (epochs within 2 d) as ΔF/F about the median; magnitudes when the range exceeds "
                "40%. Orange: detected episode or trend. Grey band: ±2σ visit-to-visit scatter of non-variable "
                "stars at the same G. Lower panels where available: Gaia BP (330–680 nm) and RP (630–1050 nm); ZTF "
                "g, r, i (about 480, 640, 790 nm; 15 d means); NEOWISE W1 (3.4 µm) and W2 (4.6 µm).")
    strength = float(_get(row, "strength"))
    if grp in ctx.lists and np.isfinite(strength):
        f, r, n = ctx.lists[grp]
        pg = ctx.pop[ctx.pop["group"] == grp] if "group" in ctx.pop else None
        bulk = (pg["fade_strength"], pg["rise_strength"]) if pg is not None and len(pg) else None
        add("strength", population.strength_hist, f, r, strength, n_stars=n, bulk=bulk, where=f"{grp.replace('<', ' < ').replace('>=', ' ≥ ')}, {ctx.label}",
            label=label, title="Fade strength distribution",
            caption="Histogram of strength (score / threshold) for every star scored in the same magnitude range: "
                    "fading above the axis, brightening below (log counts). Brightening is the same statistic with "
                    "the sign reversed, so at each strength it estimates the false-alarm count among the fades. "
                    "Bars at strength ≥ 1 count every star in the run; lighter bars below 1 are scaled from the "
                    "comparison sample. Rank: position of the candidate among all fades.")
    if ctx.top is not None and len(ctx.top):
        cand = {k: _get(row, k) for k in data.SIMILAR_COLUMNS}
        cand.update({k: _get(astro, k) for k in data.SIMILAR_COLUMNS if k.startswith(("parallax", "pm"))})
        add("classes", population.similar_classes, ctx.top, cand, exclude=sid, label=label, where=ctx.label,
            title="DPAC classes of similar stars",
            caption="DPAC variability classes (vari_classifier_result) of the 100 comparison stars nearest the "
                    f"candidate in {data.SIMILAR_BY}, each scaled by its spread. Comparison stars: the "
                    "strongest fades of the run. Open dots: share of each class among all "
                    "comparison stars; orange: among the 100 nearest. On the 20,000 strongest DR3 fades, the "
                    "largest class among the 100 nearest matches DPAC's class for 98% of classed stars. Rare "
                    "classes seldom lead: young stars, cataclysmic variables and R CrB stars typically have 7 to "
                    "24 of their own class among the nearest, a count fewer than 0.3% of other stars reach.")
    if ctx.hr is not None and astro is not None:
        hr_row = {"phot_g_mean_mag": g, "bp_rp": _get(row, "bp_rp"), "parallax": _get(astro, "parallax"),
                  "parallax_error": _get(astro, "parallax_error")}
        add("hr", population.hr_diagram, ctx.hr, hr_row, label=label, title="Colour–magnitude diagram",
            caption="Absolute G magnitude M_G = G + 5 log₁₀(parallax / 100 mas), no extinction correction, "
                    "against the colour BP − RP (BP magnitude minus RP magnitude; larger is redder) for Gaia stars "
                    "with parallax / error > 10.")
    if np.isfinite(float(_get(row, "depth_bp"))) and np.isfinite(float(_get(row, "depth_rp"))):
        col_row = {k: _get(row, k) for k in ("dim_depth", "depth_bp", "depth_bp_err", "depth_rp", "depth_rp_err",
                                             "bp_rp")}
        add("colour", colour.change_colour, ctx.pop, col_row, label=label,
            title="BP/RP depth ratio",
            caption="Depth of the episode in Gaia BP (330–680 nm) over its depth in RP (630–1050 nm), against "
                    "its depth in G (330–1050 nm). Reference lines: grey occulter (ratio 1), dust with "
                    "A ∝ λ^−1.5, and a temperature drop from the T_eff implied by BP − RP.")
    if checks is not None:
        ir = dict(checks)
        ir.setdefault("dim_depth", _get(row, "dim_depth"))
        ir.setdefault("bp_rp", _get(row, "bp_rp"))
        add("infrared", colour.infrared, ir, title="NEOWISE W1/W2",
            caption="NEOWISE infrared W1 (3.4 µm) and W2 (4.6 µm) change over the episode dates, measured (orange, ±1σ) and "
                    "predicted per model (dots; filled = within 1σ). Dust that absorbs the light re-emits in W1/W2; "
                    "an opaque cold occulter and a temperature drop do not.")
    return entries


def short_dip_candidate(ctx, row, epochs, t_dip, out_dir, name=None, label="candidate", sid=None,
                        themes=style.THEMES):
    """Figures for one short-dip candidate: the dip in context and zoomed, and the dip depth against the comparison
    sample by G (needs min_frac in ctx.pop and in row)."""
    out_dir = Path(out_dir)
    sid = int(sid if sid is not None else _get(row, "id", _get(row, "source_id", 0)))
    name = name or f"Gaia {ctx.label} {sid}"
    g = float(_get(row, "phot_g_mean_mag"))
    noise = data.typical_noise(ctx.pop, g, quiet=False) if "sigma_visit" in ctx.pop else None
    entries = []
    paths = style.render(lightcurve.short_dip, out_dir / f"{sid}-dip", epochs, t_dip, noise=noise, name=name,
                         themes=themes)
    if paths:
        entries.append({"key": "dip", "title": "Short dip", "paths": paths,
                        "caption": "Left: all G epochs, dip epochs in orange. Right: G, BP and RP near the minimum. "
                                   "The two fields of view give transit pairs 106 min apart; the next pair follows "
                                   "hours to weeks later."})
    if "min_frac" in ctx.pop and np.isfinite(float(_get(row, "min_frac"))):
        paths = style.render(population.dip_depth, out_dir / f"{sid}-dipdepth", ctx.pop, row, label=label,
                             themes=themes)
        entries.append({"key": "dipdepth", "title": "Single-epoch depth vs G", "paths": paths,
                        "caption": "Largest single-epoch flux decrease of each comparison star against G. Dots: DPAC "
                                   "eclipsing binaries, which reach the largest single-epoch depths."})
    return entries
