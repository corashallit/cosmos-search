"""Versioned ADQL target queries for the Gaia dimmer prefilter (docs/infra/targets.md).

The prefilter runs where the data are (Gaia TAP), on vari_summary, so DR4 day one needs no bulk download.
It is two-sided by construction. For each source:

    d_faint  = (max_mag_g_fov − median_mag_g_fov) / mad_mag_g_fov      (deepest dimming, robust σ)
    d_bright = (median_mag_g_fov − min_mag_g_fov) / mad_mag_g_fov      (its mirror)

Thresholds are set per magnitude bin from the bright tail, so each bin passes the same fraction of its
mirror null, and the bright-side selections come along as a decoy set (target–decoy, as in proteomics).
A source_id-modulo control set gives prefilter-independent randoms for measuring recall.

The Gaia archive's ADQL has no CASE, LEAST or UNION, so bins come from FLOOR and per-bin thresholds from
an OR chain. Changing a template or its parameters is a new version: frozen manifests record the
rendered ADQL hash.
"""

from dataclasses import dataclass, field

from .releases import tables

MAG_BIN = 0.5  # magnitude bin width; bin index = FLOOR(median_mag_g_fov / MAG_BIN)
D_STEP = 0.25  # histogram resolution in robust sigma
CONTROL_MODULUS = 4093  # prime; MOD(source_id, 4093) = 0 keeps ~0.024% of sources

_D = {
    "faint": "(max_mag_g_fov - median_mag_g_fov) / mad_mag_g_fov",
    "bright": "(median_mag_g_fov - min_mag_g_fov) / mad_mag_g_fov",
}
_GOOD = "mad_mag_g_fov > 0 AND num_selected_g_fov >= {min_transits}"
_MAG_BIN = f"FLOOR(median_mag_g_fov / {MAG_BIN})"


@dataclass(frozen=True)
class Query:
    name: str
    version: int
    description: str
    template: str
    defaults: dict = field(default_factory=dict)

    def render(self, release, **params):
        p = {**self.defaults, **params}
        return " ".join(self.template.format(t=tables(release), **p).split())


HISTOGRAM = Query(
    name="dimmer_tail_histogram",
    version=1,
    description="Counts per (magnitude bin, d bin) for one tail: the empirical null (bright) or target (faint)",
    template=f"""
SELECT mag_bin, d_bin, COUNT(*) AS n FROM (
  SELECT {_MAG_BIN} AS mag_bin, FLOOR({{d}} / {D_STEP}) AS d_bin
  FROM {{t[vari_summary]}} WHERE {_GOOD} {{extra_where}}
) AS h GROUP BY mag_bin, d_bin
""",
    defaults={"min_transits": 10, "extra_where": ""},
)

PREFILTER = Query(
    name="dimmer_prefilter",
    version=1,
    description="Sources over their magnitude bin's threshold on either tail, plus modulo controls",
    template=f"""
SELECT source_id, num_selected_g_fov, median_mag_g_fov, mad_mag_g_fov, min_mag_g_fov, max_mag_g_fov,
       skewness_mag_g_fov, outlier_median_g_fov, {_MAG_BIN} AS mag_bin,
       {_D["faint"]} AS d_faint, {_D["bright"]} AS d_bright, MOD(source_id, {CONTROL_MODULUS}) AS control_mod
FROM {{t[vari_summary]}}
WHERE {_GOOD} {{extra_where}} AND (MOD(source_id, {CONTROL_MODULUS}) = 0 OR {{threshold_terms}})
""",
    defaults={"min_transits": 10, "extra_where": ""},
)

REGISTRY = {q.name: q for q in (HISTOGRAM, PREFILTER)}


def render_histogram(release, side, **params):
    return HISTOGRAM.render(release, d=_D[side], **params)


def threshold_terms(thresholds):
    """OR chain: a source passes if either tail is over its magnitude bin's threshold.

    thresholds maps mag_bin (int) -> d. Bins without a threshold pass nothing (apart from controls).
    """
    terms = []
    for b, d in sorted(thresholds.items()):
        lo, hi = b * MAG_BIN, (b + 1) * MAG_BIN
        rng = f"median_mag_g_fov >= {lo:g} AND median_mag_g_fov < {hi:g}"
        terms.append(f"({rng} AND ({_D['faint']} >= {d:.4f} OR {_D['bright']} >= {d:.4f}))")
    return "(" + " OR ".join(terms) + ")" if terms else "1 = 0"


def render_prefilter(release, thresholds, **params):
    return PREFILTER.render(release, threshold_terms=threshold_terms(thresholds), **params)
