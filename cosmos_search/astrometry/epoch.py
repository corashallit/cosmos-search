"""Gaia DR4 epoch astrometry: per-CCD along-scan (AL) abscissae, flattened and combined per field-of-view transit.

DR4 publishes one row per FoV transit with per-CCD arrays (SM + AF1..AF9, 10 entries). The AL observation model
(gaia-supdate, Lindegren et al. 2012) for a single star is linear:

    w = dra* sin(psi) + ddec cos(psi) + plx * f_AL + pmra* t sin(psi) + pmdec t cos(psi) [+ colour term]

with w = centroid_pos_al (mas, local-plane AL coordinate relative to the reference point ra0, dec0), psi =
scan_pos_angle, f_AL = parallax_factor_al and t in Julian years from J2017.5 (TCB, barycentric). Every model in
`models` adds columns to this design matrix.

    from cosmos_search.astrometry import epoch
    ccd = epoch.read_votable("GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml")   # one row per CCD
    tr = epoch.per_transit(ccd)                                                # one row per FoV transit
"""

import numpy as np
import pandas as pd

from cosmos_search.gaia.epoch import decode_transit_id

TCB_REF_JYEAR = 2010.0  # obs_time_tcb counts ns from 2010-01-01T00:00:00 TCB (gaia-supdate constants)
DR4_REF_JYEAR = 2017.5
NS_PER_JYEAR = 365.25 * 86400e9
JD_J2010 = 2455197.5  # J2010.0 = 2010-01-01T00:00 TCB
BTJD0 = 2457000.0

ARRAY_COLS = ["obs_time_tcb", "scan_pos_angle", "colour_factor_al", "centroid_pos_al", "centroid_pos_error_al",
              "used_by_agis_al", "ccd_proc_flags", "ipd_error_al", "gates"]
SCALAR_COLS = ["source_id", "transit_id", "ra0", "dec0", "agis_source_excess_noise", "obs_time_bary_corr",
               "parallax_factor_al", "nu_eff_used_in_astrometry", "multipeak", "blended", "g_mag", "g_class",
               "transit_proc_flags"]


def _jyear_from_tcb(ns, bary_ns):
    """Barycentric Julian years from J2017.5 for an obs_time_tcb (ns since 2010.0) plus its barycentric correction."""
    return TCB_REF_JYEAR + (np.asarray(ns, float) + np.asarray(bary_ns, float)) / NS_PER_JYEAR - DR4_REF_JYEAR


def flatten(table):
    """Archive-format epoch astrometry (astropy Table or DataFrame with per-CCD arrays) -> one row per CCD.

    Adds `ccd` (0 = SM, 1..9 = AF1..AF9), `t` (Julian years from J2017.5, barycentric) and `btjd` (days)."""
    df = table.to_pandas() if hasattr(table, "to_pandas") else table
    rows = []
    for rec in df.itertuples(index=False):
        r = rec._asdict()
        n = len(np.atleast_1d(r["obs_time_tcb"]))
        out = {c: np.repeat(r[c], n) for c in SCALAR_COLS if c in r}
        for c in ARRAY_COLS:
            if c in r:
                a = np.ma.asarray(r[c])
                v = np.ma.filled(a, False) if a.dtype == bool else np.ma.filled(a.astype(float), np.nan)
                out[c] = np.asarray(v)[:n] if len(np.atleast_1d(v)) >= n else np.full(n, np.nan)
        out["ccd"] = np.arange(n)
        rows.append(pd.DataFrame(out))
    ccd = pd.concat(rows, ignore_index=True)
    ccd["t"] = _jyear_from_tcb(ccd["obs_time_tcb"], ccd["obs_time_bary_corr"])
    ccd["btjd"] = JD_J2010 + (ccd["t"] + DR4_REF_JYEAR - TCB_REF_JYEAR) * 365.25 - BTJD0
    ccd["used_by_agis_al"] = ccd["used_by_agis_al"].astype(bool)
    return ccd


def read_votable(path):
    """DR4 (pre-release) epoch astrometry VOTable -> one row per CCD (see `flatten`)."""
    from astropy.io.votable import parse_single_table

    return flatten(parse_single_table(str(path)).to_table())


def select(ccd, used_only=True, drop_sm=True):
    """The CCD observations AGIS used (AF only by default): the input every fit here starts from."""
    m = np.isfinite(ccd["centroid_pos_al"]) & np.isfinite(ccd["centroid_pos_error_al"])
    if drop_sm:
        m &= ccd["ccd"] > 0
    if used_only:
        m &= ccd["used_by_agis_al"]
    return ccd[m].reset_index(drop=True)


def per_transit(ccd):
    """Combine CCDs of one FoV transit (~40 s, so no source motion inside) into a weighted-mean abscissa.

    Returns one row per transit with t, psi (rad), f_AL, w, sigma_w (formal, from the per-CCD errors), the
    intra-transit chi^2 and its degrees of freedom (n_ccd - 1). The intra-transit chi^2 measures whether the per-CCD
    errors are right for this source, independently of any model of its motion: it is the noise yardstick."""
    c = select(ccd)
    wt = 1.0 / c["centroid_pos_error_al"].to_numpy() ** 2
    c = c.assign(_wt=wt, _ww=wt * c["centroid_pos_al"].to_numpy(), _wpsi=wt * c["scan_pos_angle"].to_numpy(),
                 _wtt=wt * c["t"].to_numpy(), _wcf=wt * c["colour_factor_al"].fillna(0).to_numpy())
    g = c.groupby(["source_id", "transit_id"], sort=False)
    s = g[["_wt", "_ww", "_wpsi", "_wtt", "_wcf"]].sum()
    out = pd.DataFrame({"t": s["_wtt"] / s["_wt"], "psi": np.deg2rad(s["_wpsi"] / s["_wt"]),
                        "w": s["_ww"] / s["_wt"], "sigma_w": 1.0 / np.sqrt(s["_wt"]),
                        "colour_factor": s["_wcf"] / s["_wt"]})
    first = g[["parallax_factor_al", "g_mag", "ra0", "dec0", "agis_source_excess_noise", "btjd"]].first()
    out = out.join(first)
    out["n_ccd"] = g.size()
    mean = out["w"].reindex(pd.MultiIndex.from_frame(c[["source_id", "transit_id"]])).to_numpy()
    chi = (c["centroid_pos_al"].to_numpy() - mean) ** 2 * wt
    out["chi2_intra"] = pd.Series(chi).groupby([c["source_id"], c["transit_id"]], sort=False).sum().to_numpy()
    out["dof_intra"] = out["n_ccd"] - 1
    out = out.reset_index().rename(columns={"parallax_factor_al": "f_al"})
    out["fov"] = decode_transit_id(out["transit_id"].to_numpy())[1]
    if "transit_proc_flags" in c:
        tpf = c.groupby(["source_id", "transit_id"], sort=False)["transit_proc_flags"].first().to_numpy()
        out["multi_source"] = np.isin(tpf.astype(np.int64) & 0xF, [1, 2, 3])
    return out.sort_values(["source_id", "t"], ignore_index=True)
