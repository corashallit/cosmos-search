"""The standard set of figures for one candidate, with titles and captions for a page.

    ctx = Context(pop=pop, lists=lists, hr=hr, release=("Gaia DR4", 1869.0), label="DR4")
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


@dataclass
class Context:
    """What every candidate is compared against, loaded once per run.

    pop: comparison sample (detector columns, phot_g_mean_mag, bp_rp, data.strengths columns, best_class_name);
    lists: {group: (fading strengths, brightening strengths, stars scored)} over the whole run (the lists);
    hr: reference sample for the HR diagram (phot_g_mean_mag, bp_rp, parallax); release: (name, last BTJD);
    label: data release name for captions."""
    pop: pd.DataFrame
    lists: dict = field(default_factory=dict)
    hr: pd.DataFrame = None
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
    """One line about the star for under the light curve's title."""
    parts = [f"G = {float(_get(row, 'phot_g_mean_mag')):.1f}", f"BP−RP = {float(_get(row, 'bp_rp')):.2f}"]
    if cls:
        parts.append("DPAC class: " + data.CLASS_NAMES.get(cls, cls))
    else:
        parts.append("no DPAC class")
    if astro is not None and np.isfinite(float(_get(astro, "parallax"))):
        p, e = float(_get(astro, "parallax")), float(_get(astro, "parallax_error"))
        parts.append(f"parallax {p:.2f} ± {e:.2f} mas")
    return " · ".join(parts)


def slow_candidate(ctx, row, epochs, out_dir, others=None, checks=None, astro=None, cls=None, name=None,
                   label="this star", sid=None, themes=style.THEMES):
    """Figures for one slow-change candidate: light curves, how rare the change is, known classes, HR diagram,
    colour of the change and (with NEOWISE in `checks`) the infrared response. row: the star's row from the lists
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
        caption="Each dot is one Gaia visit, as the change from the star's normal level (the median); brighter is "
                "up. The orange band marks the change the search found. The grey band around zero is the "
                "visit-to-visit scatter of an ordinary star this bright. Lower panels: Gaia's blue and red "
                "bands over the same dates, then other telescopes where they have data.")
    strength = float(_get(row, "strength"))
    if grp in ctx.lists and np.isfinite(strength):
        f, r, n = ctx.lists[grp]
        add("rare", population.how_rare, f, r, strength, n_stars=n, where=f"at {grp.replace('<', ' < ').replace('>=', ' ≥ ')} in {ctx.label}",
            label=label, title="How rare is a change this strong?",
            caption="Number of stars whose fading (blue) or brightening (grey, dashed) is at least as strong as the "
                    "value on the bottom axis. The search's threshold is 1. Brightening mirrors fading, so its "
                    "curve shows what chance and ordinary variability produce; a fade stands out where the blue "
                    "curve runs above the grey one.")
    cand = {"abbe": _get(row, "abbe"), "dim_depth": _get(row, "dim_depth")}
    if "best_class_name" in ctx.pop:
        add("classes", population.known_classes, ctx.pop, cand, title="Known kinds of variable star",
            caption="Two numbers the search measures: how smooth the light curve is (bottom) and how deep the "
                    "change is (side). Blue shows every star compared; each panel's dots are the stars Gaia's "
                    "variability team (DPAC) put in one class. A candidate that sits among a class's dots behaves "
                    "like that class on these two numbers.")
    if ctx.hr is not None and astro is not None:
        hr_row = {"phot_g_mean_mag": g, "bp_rp": _get(row, "bp_rp"), "parallax": _get(astro, "parallax"),
                  "parallax_error": _get(astro, "parallax_error")}
        add("hr", population.hr_diagram, ctx.hr, hr_row, label=label, title="What kind of star is it?",
            caption="Colour against true brightness (from the parallax distance, with no correction for dust) for "
                    "nearby Gaia stars. Stars sit on the main sequence while they burn hydrogen, then swell into "
                    "giants; white dwarfs are the cooling cores left at the end.")
    if np.isfinite(float(_get(row, "depth_bp"))) and np.isfinite(float(_get(row, "depth_rp"))):
        col_row = {k: _get(row, k) for k in ("dim_depth", "depth_bp", "depth_bp_err", "depth_rp", "depth_rp_err",
                                             "bp_rp")}
        add("colour", colour.change_colour, ctx.pop, col_row, label=label,
            title="The colour of the change",
            caption="How much deeper the change is in Gaia's blue band than in its red band. An opaque body blocks "
                    "both alike (ratio 1); small dust grains dim blue light more; a star that cools dims most in "
                    "the blue too, by an amount set by its temperature.")
    if checks is not None:
        ir = dict(checks)
        ir.setdefault("dim_depth", _get(row, "dim_depth"))
        ir.setdefault("bp_rp", _get(row, "bp_rp"))
        add("infrared", colour.infrared, ir, title="The infrared over the same dates",
            caption="NEOWISE's change at 3.4 and 4.6 µm while Gaia saw the change, against what each explanation "
                    "predicts. Dust that absorbs starlight warms up and glows in the infrared; an opaque cold body "
                    "or a cooling star does not.")
    return entries


def short_dip_candidate(ctx, row, epochs, t_dip, out_dir, name=None, label="this star", sid=None,
                        themes=style.THEMES):
    """Figures for one short-dip candidate: the dip in context and zoomed, and its depth against ordinary stars of
    the same brightness (needs min_frac in ctx.pop and in row)."""
    out_dir = Path(out_dir)
    sid = int(sid if sid is not None else _get(row, "id", _get(row, "source_id", 0)))
    name = name or f"Gaia {ctx.label} {sid}"
    g = float(_get(row, "phot_g_mean_mag"))
    noise = data.typical_noise(ctx.pop, g, quiet=False) if "sigma_visit" in ctx.pop else None
    entries = []
    paths = style.render(lightcurve.short_dip, out_dir / f"{sid}-dip", epochs, t_dip, noise=noise, name=name,
                         themes=themes)
    if paths:
        entries.append({"key": "dip", "title": "The dip", "paths": paths,
                        "caption": "Left: every Gaia epoch of the star, the dip's epochs in orange. Right: the hours "
                                   "around the dip in all three bands. Gaia sees a star twice about 106 minutes "
                                   "apart (its two fields of view), then again some hours or weeks later."})
    if "min_frac" in ctx.pop and np.isfinite(float(_get(row, "min_frac"))):
        paths = style.render(population.dip_depth, out_dir / f"{sid}-dipdepth", ctx.pop, row, label=label,
                             themes=themes)
        entries.append({"key": "dipdepth", "title": "How deep do ordinary stars dip?", "paths": paths,
                        "caption": "The deepest single epoch of every star compared, against its brightness. "
                                   "Eclipsing binaries (dots) reach the deepest single-epoch dips."})
    return entries
