"""House style for release-day figures, taken from cora.earth (light and dark), and the helpers every plot shares.

Colours are the site's tokens (style.css on cora.earth, 2026-10-30): square corners, mono labels in tight boxes, one
sequential ramp from the page colour through `blue`, and `orange` for the thing a figure is about. Here that means:

| Role | Token | Used for |
|---|---|---|
| figure surface | `box` | every figure's background, like the site's figure panels |
| ordinary stars | ramp from `bg` through `blue` | densities of the comparison population |
| the candidate | `orange` (`orange_fg` on it) | the star a figure is about, its episode, its labels |
| Gaia G / BP / RP | `fg` / `blue` / `red` | light curves, by wavelength |
| ZTF g / r / i | `blue` / `red` / `amber` | light curves after Gaia |
| NEOWISE W1 / W2 | `purple` / `amber` | infrared light curves |
| known classes | `fg` dots, one class per panel | small multiples, so no class needs its own hue |

Colour pairs that share a panel were checked with the dataviz skill's validator (2026-10-30): blue/red, purple/amber
and blue/orange pass every all-pairs check in light mode on `box`; in dark mode they pass colour-blind and normal-vision
separation and contrast, but sit lighter than the skill's dark band (the site's dark inks are pastels by design). Never
put green and amber, or green and orange, in one panel: those pairs fail.

    from cosmos_search.viz import style
    paths = style.render(make_figure, "out/star-123-lightcurve", data)   # light and dark PNG + SVG

Inside a plot function, read colours with `style.c("orange")` (current theme) and build figures with
`style.figure(...)`; `render` calls the function once per theme.
"""

import contextlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb  # noqa: E402

TOKENS = {
    "light": {
        "bg": "#ffffff", "fg": "#161616", "muted": "#5e5e5e", "box": "#f5f5f5", "box_head": "#ebebeb",
        "hl": "#e2e3ea", "selected": "#bfc8e3", "rule": "#d4d4d4", "accent": "#4b64a3", "accent_fg": "#ffffff",
        "blue": "#4b64a3", "purple": "#7d52b0", "amber": "#9a6814", "orange": "#e56900", "orange_fg": "#161616",
        "green": "#2f7a4c", "red": "#b2443f", "grey": "#8c8c8c",
    },
    "dark": {
        "bg": "#121212", "fg": "#e4e4e4", "muted": "#9b9b9b", "box": "#1b1b1b", "box_head": "#262626",
        "hl": "#2d2f38", "selected": "#3f4866", "rule": "#383838", "accent": "#93a8dc", "accent_fg": "#121212",
        "blue": "#93a8dc", "purple": "#bb9de6", "amber": "#e0b462", "orange": "#fd923e", "orange_fg": "#121212",
        "green": "#7cc79a", "red": "#e48a83", "grey": "#7d7d7d",
    },
}
THEMES = tuple(TOKENS)

BAND_TOKEN = {"G": "fg", "BP": "blue", "RP": "red", "ztf_g": "blue", "ztf_r": "red", "ztf_i": "amber",
              "W1": "purple", "W2": "amber"}
BAND_LABEL = {"G": "G", "BP": "BP (blue)", "RP": "RP (red)", "ztf_g": "ZTF g", "ztf_r": "ZTF r", "ztf_i": "ZTF i",
              "W1": "W1 (3.4 µm)", "W2": "W2 (4.6 µm)"}

MONO = ["Berkeley Mono", "SF Mono", "Menlo", "Cascadia Mono", "DejaVu Sans Mono"]
SANS = ["Inter", "DejaVu Sans"]
DPI = 100            # figure sizes are given in pixels at this dpi, like the site's 44rem (704 px) text column
PNG_SCALE = 2        # PNGs are written at twice that, for sharp text on high-density screens
WIDTH = 720          # default figure width in pixels
FONT_PX = 11         # the site's figure labels: 11 px mono, muted
TITLE_PX = 13

_current = {"name": "light"}


def px(n):
    """Points for a size given in pixels at DPI (matplotlib sizes text and lines in points)."""
    return n * 72.0 / DPI


def c(name):
    """A token's hex colour in the current theme."""
    return TOKENS[_current["name"]][name]


def current():
    return _current["name"]


# --- colour mixing, as CSS color-mix(in oklab, a p%, b) ------------------------------------------------------------


def _lin(u):
    u = np.asarray(u, float)
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def _gam(u):
    u = np.clip(np.asarray(u, float), 0, 1)
    return np.where(u <= 0.0031308, 12.92 * u, 1.055 * u ** (1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def _oklab(hexcolour):
    lms = _M1 @ _lin(to_rgb(hexcolour))
    return _M2 @ np.cbrt(lms)


def _from_oklab(lab):
    lms = np.linalg.solve(_M2, lab) ** 3
    return to_hex(_gam(np.linalg.solve(_M1, lms)))


def mix(a, b, p):
    """CSS color-mix(in oklab, a p, b) for p in 0..1, with a and b token names or hex colours."""
    a, b = (c(x) if not str(x).startswith("#") else x for x in (a, b))
    return _from_oklab(p * _oklab(a) + (1 - p) * _oklab(b))


def ramp(n=8):
    """n steps of the site's sequential ramp: page colour + 12% blue, then blue (at 75%), then blue mixed with ink."""
    out = []
    for x in np.linspace(0, 1, n):
        if x <= 0.75:
            p = 0.12 + (1 - 0.12) * x / 0.75
            out.append(mix("blue", "bg", p))
        else:
            p = 1 - 0.3 * (x - 0.75) / 0.25
            out.append(mix("blue", "fg", p))
    return out


def cmap(n=64):
    """The sequential ramp as a matplotlib colormap (zero should be left unpainted, so the surface shows)."""
    return LinearSegmentedColormap.from_list("cora_ramp", ramp(n), N=256)


# --- figures -------------------------------------------------------------------------------------------------------


def rc():
    """matplotlib settings for the current theme."""
    return {
        "figure.dpi": DPI, "savefig.dpi": DPI, "figure.facecolor": c("box"), "savefig.facecolor": c("box"),
        "axes.facecolor": "none", "axes.edgecolor": c("rule"), "axes.linewidth": px(1),
        "axes.labelcolor": c("muted"), "axes.labelsize": px(FONT_PX), "axes.titlesize": px(FONT_PX),
        "axes.titlecolor": c("fg"), "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": px(8), "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "axes.axisbelow": True, "axes.prop_cycle": matplotlib.cycler(color=[c("fg")]),
        "xtick.color": c("rule"), "ytick.color": c("rule"), "xtick.labelcolor": c("muted"),
        "ytick.labelcolor": c("muted"), "xtick.labelsize": px(FONT_PX), "ytick.labelsize": px(FONT_PX),
        "xtick.major.size": px(4), "ytick.major.size": px(4), "xtick.minor.size": px(2), "ytick.minor.size": px(2),
        "xtick.major.width": px(1), "ytick.major.width": px(1),
        "font.family": "monospace", "font.monospace": MONO, "font.size": px(FONT_PX), "text.color": c("fg"),
        "legend.frameon": False, "legend.fontsize": px(FONT_PX), "legend.labelcolor": c("muted"),
        "lines.linewidth": px(2), "lines.markersize": px(6), "patch.linewidth": 0,
        "grid.color": c("rule"), "grid.linewidth": px(1), "hatch.color": c("rule"),
        "svg.fonttype": "none", "svg.hashsalt": "cosmos-search", "image.interpolation": "nearest",
    }


@contextlib.contextmanager
def theme(name):
    """Use one of THEMES for the figures made inside the block."""
    if name not in TOKENS:
        raise ValueError(f"unknown theme {name!r}; one of {THEMES}")
    prev = _current["name"]
    _current["name"] = name
    try:
        with matplotlib.rc_context(rc()):
            yield
    finally:
        _current["name"] = prev


def figure(width=WIDTH, height=420, **kw):
    """A figure of width x height pixels on the figure surface."""
    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI, **kw)
    fig._viz_hover = {}
    return fig


def title(fig, text, sub=None, y=None):
    """The figure's title in the site's heading style (bold mono in a tight highlight box), top left, and an
    optional plain line under it."""
    h = fig.get_figheight() * DPI
    y = 1 - 14 / h if y is None else y
    fig.text(14 / (fig.get_figwidth() * DPI), y, text, ha="left", va="top", fontsize=px(TITLE_PX),
             fontweight="bold", color=c("fg"), bbox=dict(boxstyle="square,pad=0.15", fc=c("hl"), ec="none"))
    if sub:
        fig.text(14 / (fig.get_figwidth() * DPI), y - 24 / h, sub, ha="left", va="top", fontsize=px(FONT_PX),
                 color=c("muted"))


def panel_label(ax, text, loc="left"):
    """A small bold label for one panel, at its top left (or right), in ink."""
    x, ha = (0.0, "left") if loc == "left" else (1.0, "right")
    ax.text(x, 1.02, text, transform=ax.transAxes, ha=ha, va="bottom", fontsize=px(FONT_PX), fontweight="bold",
            color=c("fg"))


def note(ax, x, y, text, transform=None, **kw):
    """Muted text, for explanations inside a plot."""
    kw.setdefault("fontsize", px(FONT_PX))
    kw.setdefault("color", c("muted"))
    return ax.text(x, y, text, transform=transform or ax.transAxes, **kw)


def _shifted(ax, transform, dx, dy):
    tr = transform or ax.transData
    if dx or dy:
        from matplotlib.transforms import offset_copy
        tr = offset_copy(tr, fig=ax.figure, x=px(dx), y=px(dy), units="points")
    return tr


def chip(ax, x, y, text, transform=None, ha="left", va="center", dx=0, dy=0, **kw):
    """The candidate's label: bold text in a filled orange box (the site's 'chip'), dx/dy pixels from (x, y)."""
    return ax.text(x, y, text, transform=_shifted(ax, transform, dx, dy), ha=ha, va=va, fontsize=px(FONT_PX),
                   fontweight="bold", color=c("orange_fg"), zorder=20,
                   bbox=dict(boxstyle="square,pad=0.25", fc=c("orange"), ec="none"), **kw)


def tag(ax, x, y, text, transform=None, ha="left", va="center", dx=0, dy=0, **kw):
    """A label for a reference mark (known object, model line): ink text in a tight highlight box."""
    return ax.text(x, y, text, transform=_shifted(ax, transform, dx, dy), ha=ha, va=va, fontsize=px(FONT_PX),
                   color=c("fg"), zorder=15, bbox=dict(boxstyle="square,pad=0.2", fc=c("hl"), ec="none"), **kw)


def dimension(ax, x0, x1, y, text, transform=None):
    """A dimension line across [x0, x1] with its length written in the middle, like the site's ├──┤ scale bars."""
    tr = transform or ax.transData
    ax.plot([x0, x1], [y, y], color=c("muted"), lw=px(1), transform=tr, zorder=12, solid_capstyle="butt")
    for x in (x0, x1):
        ax.plot([x, x], [y, y], marker="|", ms=px(10), mew=px(1), color=c("muted"), transform=tr, zorder=12)
    ax.text((x0 + x1) / 2, y, text, transform=tr, ha="center", va="center", fontsize=px(FONT_PX),
            fontweight="bold", color=c("fg"), zorder=13, bbox=dict(boxstyle="square,pad=0.2", fc=c("box"), ec="none"))


def candidate_marker(ax, x, y, size=9, **kw):
    """The candidate: an orange dot with a surface-coloured ring, so it stands clear of whatever is under it."""
    return ax.scatter([x], [y], s=(px(size)) ** 2 * 1.6, c=c("orange"), edgecolors=c("box"), linewidths=px(2),
                      zorder=18, **kw)


def density(ax, x, y, bins, log_x=False, log_y=False, weights=None, zorder=1):
    """2-D histogram of the comparison population in the sequential ramp, log-scaled counts, empty bins left
    unpainted so the surface shows through. Returns the mesh."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if log_x:
        ok &= x > 0
    if log_y:
        ok &= y > 0
    x, y = x[ok], y[ok]
    w = None if weights is None else np.asarray(weights, float)[ok]
    bx, by = bins
    h, ex, ey = np.histogram2d(np.log10(x) if log_x else x, np.log10(y) if log_y else y, bins=(bx, by), weights=w)
    h = np.ma.masked_less_equal(h, 0)
    ex = 10 ** ex if log_x else ex
    ey = 10 ** ey if log_y else ey
    from matplotlib.colors import LogNorm
    vmax = max(float(h.max()) if h.count() else 1.0, 1.0)
    mesh = ax.pcolormesh(ex, ey, h.T, cmap=cmap(), norm=LogNorm(vmin=1, vmax=vmax), shading="flat",
                         rasterized=True, zorder=zorder, linewidth=0)
    if log_x:
        ax.set_xscale("log")
        plain_log(ax.xaxis)
    if log_y:
        ax.set_yscale("log")
        plain_log(ax.yaxis)
    return mesh


def ramp_legend(fig, rect, low="fewer", high="more stars", ticks=None):
    """The site's ramp legend: 'fewer [ramp] more ...' in one muted line."""
    ax = fig.add_axes(rect)
    ax.imshow(np.linspace(0, 1, 128)[None, :], aspect="auto", cmap=cmap(), extent=(0, 1, 0, 1))
    ax.set_axis_off()
    ax.text(-0.04, 0.5, low, transform=ax.transAxes, ha="right", va="center", color=c("muted"), fontsize=px(FONT_PX))
    ax.text(1.04, 0.5, high, transform=ax.transAxes, ha="left", va="center", color=c("muted"), fontsize=px(FONT_PX))
    return ax


def _plain_number(v, _pos=None):
    if v == 0:
        return "0"
    a = abs(v)
    sign = "−" if v < 0 else ""
    for div, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if a >= div:
            return f"{sign}{a / div:g}{suffix}"
    if a < 1e-3:   # 10⁻⁶ rather than 1e-06
        e = int(np.floor(np.log10(a) + 1e-9))
        sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        m = a / 10 ** e
        return f"{sign}10{sup}" if abs(m - 1) < 1e-6 else f"{sign}{m:g}×10{sup}"
    return f"{sign}{a:g}"


def number(v):
    """A statistic for a label: 17,400 rather than 1.74e+04, three significant figures below 1000."""
    v = float(v)
    if not np.isfinite(v):
        return "–"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}"


def plain_log(axis):
    """Tick labels on a log axis as plain numbers (0.1, 1, 10, 1k, 1M) instead of powers of ten."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axis.set_major_formatter(FuncFormatter(_plain_number))
    axis.set_minor_formatter(NullFormatter())


def hover(artist, labels):
    """Hover text for each point of a scatter or marker line, written as SVG <title>s (browsers show them as
    tooltips). Points keep their order, so labels[i] goes with the i-th point."""
    fig = artist.figure
    gid = f"hover-{len(fig._viz_hover)}"
    artist.set_gid(gid)
    fig._viz_hover[gid] = [str(s) for s in labels]
    return artist


def year(btjd):
    """Decimal year for a BTJD (BJD - 2457000) time, for date axes (J2000.0 = JD 2451545.0)."""
    return 2000.0 + (np.asarray(btjd, float) + 2457000.0 - 2451545.0) / 365.25


def btjd(yr):
    return (np.asarray(yr, float) - 2000.0) * 365.25 + 2451545.0 - 2457000.0


def year_axis(ax, label="year"):
    """Whole years as labels on a decimal-year axis (half years too when it spans under three years), quarters as
    unlabelled ticks. Call after the x limits are set."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(MultipleLocator(1.0 if hi - lo >= 3 else 0.5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(axis="x", which="minor", length=px(3))
    if label:
        ax.set_xlabel(label)


# --- output --------------------------------------------------------------------------------------------------------

ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
_SVG = "{http://www.w3.org/2000/svg}"


def _svg_with_hover(raw, hover_map):
    """Add <title> children to the points of each hover group, and a class on the group for hover styles."""
    if not hover_map:
        return raw
    root = ET.fromstring(raw)
    for g in root.iter(_SVG + "g"):
        labels = hover_map.get(g.get("id"))
        if labels is None:
            continue
        g.set("class", "viz-hover")
        for e, text in zip(g.iter(_SVG + "use"), labels):
            t = ET.SubElement(e, _SVG + "title")
            t.text = text
    return ET.tostring(root, encoding="unicode")


def save(fig, stem, formats=("png", "svg")):
    """Write fig as stem.png (at PNG_SCALE x) and stem.svg (text kept as text, hover titles added). Returns paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    if "png" in formats:
        p = stem.with_suffix(".png")
        fig.savefig(p, dpi=DPI * PNG_SCALE, metadata={"Software": None})
        out["png"] = p
    if "svg" in formats:
        import io
        buf = io.StringIO()
        fig.savefig(buf, format="svg", dpi=DPI * PNG_SCALE, metadata={"Date": None, "Creator": None})
        raw = re.sub(r"<\?xml[^>]*\?>\s*", "", buf.getvalue())
        raw = re.sub(r"<!DOCTYPE[^>]*>\s*", "", raw)
        p = stem.with_suffix(".svg")
        p.write_text(_svg_with_hover(raw, getattr(fig, "_viz_hover", {})))
        out["svg"] = p
    return out


def render(fn, stem, *args, themes=THEMES, formats=("png", "svg"), **kwargs):
    """Call fn(*args, **kwargs) once per theme and save each figure as <stem>-<theme>.{png,svg}.
    Returns {theme: {format: path}}. fn returns a Figure, or None when it has nothing to draw."""
    out = {}
    for name in themes:
        with theme(name):
            fig = fn(*args, **kwargs)
            if fig is None:
                return {}
            out[name] = save(fig, f"{stem}-{name}", formats)
            plt.close(fig)
    return out
