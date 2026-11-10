"""Does another survey see the change Gaia saw, and did it go on after Gaia's last visit?

Gaia gives a candidate's change as a trend (fraction of the flux per year between its first and last visit) or as
an episode (a depth over a faint window). `cosmos_search.surveys` reads the same star from telescopes that share
nothing with Gaia's instrument or calibration. Three questions, each a number with an error:

- `same_years`: over Gaia's own dates, the survey's change as a multiple k of what Gaia's change predicts for
  that survey's band. k = 1 is the same change, k = 0 is no change. A Gaia fault gives 0; so does a change in a
  neighbour that only Gaia's narrow window separates.
- `after`: the trend after Gaia's last visit, as a fraction of Gaia's flux per year and as a multiple of Gaia's
  trend (1 went on at the same rate, 0 stopped, negative turned around); for an episode still under way at
  Gaia's last visit, how much of the depth has been recovered since.
- `activity`: the star's change over Gaia's dates as a multiple of its variability within a month in TESS
  (`surveys.tess_activity`). Starspots that rotate into view and a spot cycle are the same spots, so a star whose
  rotational modulation is 5% can fade 5% by activity alone, and one that is flat to 0.3% cannot.

What Gaia's change predicts for another band. The change's colour is known from BP and RP: taking the fractional
change as linear in 1/wavelength between BP (0.51 um) and RP (0.78 um) covers a grey screen (flat), interstellar-like
dust (extinction close to 1/wavelength in the optical) and a temperature change (Wien side) to first order.
`band_factor` is that line at the survey band's wavelength over its value at G's; it stays 1 unless BP and RP
differ by more than twice their combined error, and is kept within BAND_FACTOR_RANGE.

Blending. A survey with a wide point-spread function measures the star plus its neighbours, so the star's
fractional change is diluted by 1 + c, with c the neighbours' flux inside the beam over the star's (`dilution`,
from Gaia's own catalogue, as `xmatch.blend`). ASAS-SN's beam is 16 arcsec and ATLAS's 5; ZTF, SkyMapper and DECam
are near 1-3.

Errors. A survey's quoted errors leave out what limits a slow change: zero points that drift between seasons,
reference images and cameras. So epochs are first averaged into visits (within VISIT_GAP_DAYS), then into bins
of BIN_DAYS inside each group (the unit with its own zero point), and every bin's error gets the survey's
SEASON_FLOOR added in quadrature: the scatter of such bin means on constant stars beyond their quoted errors
(`scripts/survey_calibrate.py`). The fit gives every group its own level, so nothing rests on zero points
between cameras, fields or filters. If the bins scatter about the fit by more than their errors (the star's own
variability), the error of k is scaled up by that factor, and widened by Student's t for the few bins it rests on.

Verdicts leave a margin for the band factor and the dilution, neither of which is known to better than tens of
percent: "seen" needs k above 0 by K_SIGMA errors; "not_seen" needs the survey to rule out even half of Gaia's
change (k + K_SIGMA errors < HALF); "opposite" is a change of the other sign; the rest is "undecided".
"""

import numpy as np
import pandas as pd
from scipy import stats

from cosmos_search.inject.lightcurve import BAND_WAVELENGTH_UM
from cosmos_search.surveys import FWHM_ARCSEC

VISIT_GAP_DAYS = 0.5
BIN_DAYS = 180.0
PAD_DAYS = 30.0            # a survey bin this close to Gaia's first or last visit counts as inside Gaia's dates
MIN_BINS = 4
K_SIGMA = 3.0
HALF = 0.5
CLIP_SIGMA = 5.0
BAND_FACTOR_RANGE = (0.25, 4.0)
NEIGHBOUR_ARCSEC = 32.0    # twice the widest beam read here (ASAS-SN's 16 arcsec)
BP_UM, RP_UM, G_UM = BAND_WAVELENGTH_UM["BP"], BAND_WAVELENGTH_UM["RP"], BAND_WAVELENGTH_UM["G"]

# Scatter of 180-day bin means on constant stars beyond their quoted errors, as a fraction of the flux
# (scripts/survey_calibrate.py, docs/arms/slow-changes.md "Other surveys"). PROVISIONAL marks surveys not yet
# reachable for that measurement: their value is the largest measured one, until the script is run on them.
SEASON_FLOOR = {"ztf": 0.010, "skymapper": 0.015, "noirlab": 0.015, "asassn": 0.015, "atlas": 0.015}
PROVISIONAL = ("asassn", "atlas")


def band_factor(band, d_bp=np.nan, e_bp=np.nan, d_rp=np.nan, e_rp=np.nan):
    """The change expected in `band` over the change in G, from the BP and RP changes (module notes). 1 when BP
    and RP are missing or agree within twice their combined error, and for a band with no listed wavelength."""
    lam = BAND_WAVELENGTH_UM.get(band)
    if lam is None or not (np.isfinite(d_bp) and np.isfinite(d_rp) and np.isfinite(e_bp) and np.isfinite(e_rp)):
        return 1.0
    if abs(d_bp - d_rp) <= 2 * np.hypot(e_bp, e_rp):
        return 1.0
    beta = (d_bp - d_rp) / (1 / BP_UM - 1 / RP_UM)
    at = lambda um: d_rp + beta * (1 / um - 1 / RP_UM)
    if at(G_UM) == 0:
        return 1.0
    return float(np.clip(at(lam) / at(G_UM), *BAND_FACTOR_RANGE))


def dilution(sep_arcsec, dmag, fwhm_arcsec):
    """c: flux of neighbours inside a beam of the given FWHM over the star's own, for neighbours at sep_arcsec that
    are dmag fainter (negative: brighter) in G. Gaussian overlap exp(-r^2 / 4 sigma^2), as `xmatch.blend`."""
    sep, dmag = np.asarray(sep_arcsec, float), np.asarray(dmag, float)
    ok = np.isfinite(sep) & np.isfinite(dmag)
    sigma = fwhm_arcsec / 2.3548
    return float(np.sum(10 ** (-0.4 * dmag[ok]) * np.exp(-sep[ok] ** 2 / (4 * sigma ** 2))))


def neighbours_adql(ra, dec, schema="gaiadr3", radius_arcsec=NEIGHBOUR_ARCSEC):
    """Gaia sources within radius_arcsec of (ra, dec), the star itself among them."""
    return (f"SELECT source_id, ra, dec, phot_g_mean_mag FROM {schema}.gaia_source WHERE 1 = CONTAINS("
            f"POINT('ICRS', ra, dec), CIRCLE('ICRS', {float(ra):.7f}, {float(dec):.7f}, {radius_arcsec / 3600:.7f}))")


def dilutions(neighbours, source_id, ra, dec, g_mag, names):
    """{survey: c} for the named surveys from a `neighbours_adql` table (`dilution` with each survey's FWHM)."""
    nb = neighbours[neighbours["source_id"].to_numpy() != int(source_id)]
    nb = nb[np.isfinite(nb["phot_g_mean_mag"].to_numpy(float))]
    sep = 3600 * np.hypot((nb["ra"].to_numpy(float) - ra) * np.cos(np.radians(dec)), nb["dec"].to_numpy(float) - dec)
    dmag = nb["phot_g_mean_mag"].to_numpy(float) - float(g_mag)
    return {s: dilution(sep, dmag, FWHM_ARCSEC[s]) for s in names}


def visits(lc, gap=VISIT_GAP_DAYS):
    """Epochs averaged into visits per survey, band and group: inverse-variance mean, with the larger of the
    formal error and (for 3 or more epochs) the scatter of the epochs over sqrt(n)."""
    if not len(lc):
        return pd.DataFrame(columns=["survey", "band", "group", "t", "flux", "err", "n"])
    d = lc.sort_values(["survey", "band", "group", "t"], kind="stable")
    key = d[["survey", "band", "group"]].astype(str).agg("|".join, axis=1).to_numpy()
    t = d["t"].to_numpy(float)
    new = np.r_[True, (key[1:] != key[:-1]) | (np.diff(t) > gap)]
    vid = np.cumsum(new) - 1
    w = 1 / d["err"].to_numpy(float) ** 2
    f = d["flux"].to_numpy(float)
    sw = np.bincount(vid, w)
    n = np.bincount(vid)
    mean = np.bincount(vid, w * f) / sw
    var = np.bincount(vid, w * (f - mean[vid]) ** 2) / sw
    err = 1 / np.sqrt(sw)
    with np.errstate(invalid="ignore", divide="ignore"):
        scatter = np.sqrt(var * n / np.maximum(n - 1, 1) / n)
    err = np.where(n >= 3, np.maximum(err, scatter), err)
    first = np.flatnonzero(new)
    return pd.DataFrame({"survey": d["survey"].to_numpy()[first], "band": d["band"].to_numpy()[first],
                         "group": d["group"].to_numpy()[first], "t": np.bincount(vid, w * t) / sw, "flux": mean,
                         "err": err, "n": n})


def bins(v, edges=(), width=BIN_DAYS, floors=SEASON_FLOOR):
    """Visits averaged into bins of `width` days per survey, band and group, never across one of `edges` (BTJD:
    Gaia's first and last visit, an episode's start and end). A bin's error is the larger of its formal error and
    the scatter of its visits over sqrt(n), with the survey's season floor added in quadrature."""
    cols = ["survey", "band", "group", "t", "flux", "err", "n"]
    if not len(v):
        return pd.DataFrame(columns=cols)
    edges = np.sort(np.asarray([e for e in edges if np.isfinite(e)], float))
    t = v["t"].to_numpy(float)
    seg = np.searchsorted(edges, t, side="right")
    start = np.where(seg > 0, edges[np.maximum(seg - 1, 0)], np.floor(t.min() / width) * width) if len(edges) \
        else np.full(len(t), np.floor(t.min() / width) * width)
    # Before the first edge, count bins backwards from it so that the edge stays a bin boundary.
    if len(edges):
        b = np.where(seg > 0, np.floor((t - start) / width), -1 - np.floor((edges[0] - t) / width))
    else:
        b = np.floor((t - start) / width)
    d = v.assign(_seg=seg, _bin=b.astype(np.int64), _w=1 / v["err"].to_numpy(float) ** 2)
    d["_wf"], d["_wt"] = d["_w"] * d["flux"], d["_w"] * d["t"]
    g = d.groupby(["survey", "band", "group", "_seg", "_bin"], sort=True)
    out = g.agg(sw=("_w", "sum"), swf=("_wf", "sum"), swt=("_wt", "sum"), n=("flux", "size"),
                sd=("flux", "std")).reset_index()
    out["flux"], out["t"] = out["swf"] / out["sw"], out["swt"] / out["sw"]
    err = 1 / np.sqrt(out["sw"].to_numpy())
    sd = out["sd"].to_numpy(float) / np.sqrt(out["n"].to_numpy())
    err = np.where(out["n"].to_numpy() >= 3, np.maximum(err, np.nan_to_num(sd)), err)
    floor = out["survey"].map(lambda s: floors.get(s, max(floors.values()))).to_numpy(float)
    out["err"] = np.hypot(err, floor)
    return out[cols]


def _wls(a, y, err):
    """Weighted least squares: (coefficients, covariance, chi2)."""
    aw, yw = a / err[:, None], y / err
    beta, *_ = np.linalg.lstsq(aw, yw, rcond=None)
    cov = np.linalg.pinv(aw.T @ aw)
    return beta, cov, float(np.sum((yw - aw @ beta) ** 2))


def fit(b, x, min_bins=MIN_BINS):
    """Fit bin fluxes as (level of each group) + k x, with x the change Gaia predicts for each bin. Returns
    dict(k, k_err, n_bins, n_groups, chi2_dof), k NaN when the bins cannot tell (too few, or x the same in every
    bin of each group). One pass drops bins more than CLIP_SIGMA errors from the first fit. The error of k is
    scaled by sqrt(chi2/dof) when above 1 and widened by Student's t for the degrees of freedom."""
    none = {"k": np.nan, "k_err": np.nan, "n_bins": 0, "n_groups": 0, "chi2_dof": np.nan}
    if not len(b):
        return none
    key = b[["survey", "band", "group"]].astype(str).agg("|".join, axis=1).to_numpy()
    y, err, x = b["flux"].to_numpy(float), b["err"].to_numpy(float), np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(err) & (err > 0) & np.isfinite(x)
    for attempt in range(2):
        # A group informs k only if x differs between its bins.
        use = ok.copy()
        for g in np.unique(key[ok]):
            m = ok & (key == g)
            if m.sum() < 2 or np.ptp(x[m]) == 0:
                use &= ~m
        names = np.unique(key[use])
        dof = int(use.sum()) - len(names) - 1
        if use.sum() < min_bins or dof < 1:
            return none
        a = np.column_stack([(key[use][:, None] == names[None, :]).astype(float), x[use]])
        beta, cov, chi2 = _wls(a, y[use], err[use])
        resid = np.abs(y[use] - a @ beta) / err[use]
        scale = max(1.0, float(np.sqrt(chi2 / dof)))
        bad = resid > CLIP_SIGMA * scale
        if not bad.any() or attempt == 1:
            break
        ok[np.flatnonzero(use)[bad]] = False
    widen = float(stats.t.ppf(0.975, dof) / stats.norm.ppf(0.975))
    return {"k": float(beta[-1]), "k_err": float(np.sqrt(cov[-1, -1]) * scale * widen), "n_bins": int(use.sum()),
            "n_groups": len(names), "chi2_dof": chi2 / dof}


def verdict(k, k_err, k_sigma=K_SIGMA, half=HALF):
    """ "seen", "not_seen", "opposite" or "undecided" (module notes); "no_data" when k is missing."""
    if not (np.isfinite(k) and np.isfinite(k_err) and k_err > 0):
        return "no_data"
    if k - k_sigma * k_err > 0:
        return "seen"
    if k + k_sigma * k_err < 0:
        return "opposite"
    if k + k_sigma * k_err < half:
        return "not_seen"
    return "undecided"


def combine(parts):
    """Inverse-variance mean of independent (k, k_err) pairs; (nan, nan) when none is finite."""
    p = [(k, e) for k, e in parts if np.isfinite(k) and np.isfinite(e) and e > 0]
    if not p:
        return np.nan, np.nan
    w = np.array([1 / e ** 2 for _, e in p])
    return float(np.sum(w * np.array([k for k, _ in p])) / w.sum()), float(1 / np.sqrt(w.sum()))


def gaia_change(star):
    """What Gaia measured, from a row of the fading list: dict(kind "trend" or "episode", t_first, t_last, and for
    a trend slope (fraction per year) with the BP and RP slopes; for an episode depth, t0, t1 and the BP and RP
    depths). The kind follows `strength_by` (the score that set the star's strength), else `by`; a repeating
    dimming has no single window and returns kind "repeat"."""
    get = (lambda k: star.get(k, np.nan)) if hasattr(star, "get") else (lambda k: getattr(star, k, np.nan))
    num = lambda k: float(get(k)) if get(k) is not None and pd.notna(get(k)) else np.nan
    by = get("strength_by")
    if by is None or (isinstance(by, float) and np.isnan(by)) or by == "":
        text = str(get("by"))
        by = "episode" if "episode" in text else "trend" if "trend" in text else "repeat"
    out = {"kind": str(by), "t_first": num("t_first"), "t_last": num("t_last")}
    if by == "trend":
        out.update(slope=num("slope"), d_bp=num("slope_bp"), e_bp=num("slope_bp_err"), d_rp=num("slope_rp"),
                   e_rp=num("slope_rp_err"))
    elif by == "episode":
        out.update(depth=num("dim_depth"), t0=num("dim_t0"), t1=num("dim_t1"), d_bp=num("depth_bp"),
                   e_bp=num("depth_bp_err"), d_rp=num("depth_rp"), e_rp=num("depth_rp_err"),
                   to_end=str(get("fade_kind")) in ("fade", "drop"))
    return out


def predicted(b, change, c=None):
    """Gaia's change at each bin for k = 1: a fraction of the group's flux, scaled by the band factor and divided
    by 1 + c for the bin's survey (c: {survey: dilution}). Bins outside Gaia's dates get NaN."""
    t = b["t"].to_numpy(float)
    inside = (t >= change["t_first"] - PAD_DAYS) & (t <= change["t_last"] + PAD_DAYS)
    f = np.array([band_factor(bd, change.get("d_bp", np.nan), change.get("e_bp", np.nan),
                              change.get("d_rp", np.nan), change.get("e_rp", np.nan)) for bd in b["band"]])
    dil = 1 + np.array([(c or {}).get(s, 0.0) for s in b["survey"]], float)
    if change["kind"] == "trend":
        m = change["slope"] * (t - change["t_first"]) / 365.25
    elif change["kind"] == "episode":
        m = np.where((t >= change["t0"]) & (t <= change["t1"]), -change["depth"], 0.0)
    else:
        return np.full(len(t), np.nan)
    return np.where(inside, m * f / dil, np.nan)


def same_years(lc, change, c=None, floors=SEASON_FLOOR):
    """Per survey and for all together: k over Gaia's own dates (module notes).

    lc: a `surveys` table for one star. change: `gaia_change`. c: {survey: dilution}.
    Returns {survey: dict(k, k_err, n_bins, n_groups, chi2_dof, verdict)} plus the key "all" (surveys combined as
    independent measurements)."""
    if change["kind"] not in ("trend", "episode") or not len(lc):
        return {"all": {"k": np.nan, "k_err": np.nan, "verdict": "no_data"}}
    edges = [change["t_first"] - PAD_DAYS, change["t_last"] + PAD_DAYS]
    if change["kind"] == "episode":
        edges += [change["t0"], change["t1"]]
    b = bins(visits(lc), edges=edges, floors=floors)
    x = predicted(b, change, c)
    out = {}
    for s in pd.unique(b["survey"]):
        m = (b["survey"] == s).to_numpy()
        r = fit(b[m], x[m])
        r["verdict"] = verdict(r["k"], r["k_err"])
        out[str(s)] = r
    k, e = combine([(r["k"], r["k_err"]) for r in out.values()])
    out["all"] = {"k": k, "k_err": e, "verdict": verdict(k, e)}
    return out


def after(lc, change, c=None, floors=SEASON_FLOOR):
    """What each survey saw after Gaia's last visit (module notes).

    For a trend: `slope` after t_last in Gaia-equivalent fraction per year (the survey's slope divided by its
    band factor and multiplied by 1 + c) with `slope_err`, `ratio` = slope over Gaia's with `ratio_err`, and a
    verdict on the ratio read as a k: "seen" here means the fade went on, so the names are "went_on", "stopped"
    (rules out half of Gaia's rate), "reversed" and "undecided".
    For an episode still under way at Gaia's last visit: `recovered`, the level after t_last minus the level
    inside the faint window as a fraction of the depth (0 still as faint, 1 back to normal), with its error,
    from groups that have bins on both sides; verdict "still_faint" (recovered + K_SIGMA errors < HALF),
    "recovered" (recovered - K_SIGMA errors > HALF) or "undecided".
    Returns {survey: {...}} plus "all"."""
    names = {"seen": "went_on", "not_seen": "stopped", "opposite": "reversed"}
    if change["kind"] not in ("trend", "episode") or not len(lc):
        return {"all": {"verdict": "no_data"}}
    t_last = change["t_last"]
    if change["kind"] == "trend":
        b = bins(visits(lc), edges=[t_last + PAD_DAYS], floors=floors)
        b = b[b["t"] > t_last + PAD_DAYS]
        f = np.array([band_factor(bd, change["d_bp"], change["e_bp"], change["d_rp"], change["e_rp"])
                      for bd in b["band"]])
        dil = 1 + np.array([(c or {}).get(s, 0.0) for s in b["survey"]], float)
        x = change["slope"] * (b["t"].to_numpy(float) - t_last) / 365.25 * f / dil
        out = {}
        for s in pd.unique(b["survey"]):
            m = (b["survey"] == s).to_numpy()
            r = fit(b[m], x[m])
            out[str(s)] = {"ratio": r["k"], "ratio_err": r["k_err"], "slope": r["k"] * change["slope"],
                           "slope_err": r["k_err"] * abs(change["slope"]), "n_bins": r["n_bins"],
                           "t_end": float(b.loc[m, "t"].max()) if m.any() else np.nan,
                           "verdict": names.get(verdict(r["k"], r["k_err"]), verdict(r["k"], r["k_err"]))}
        k, e = combine([(r["ratio"], r["ratio_err"]) for r in out.values()])
        v = verdict(k, e)
        out["all"] = {"ratio": k, "ratio_err": e, "slope": k * change["slope"], "slope_err": e * abs(change["slope"]),
                      "verdict": names.get(v, v)}
        return out
    if not change.get("to_end"):
        return {"all": {"verdict": "no_data"}}
    b = bins(visits(lc), edges=[change["t0"], t_last + PAD_DAYS], floors=floors)
    t = b["t"].to_numpy(float)
    b = b[(t >= change["t0"])]
    t = b["t"].to_numpy(float)
    f = np.array([band_factor(bd, change["d_bp"], change["e_bp"], change["d_rp"], change["e_rp"]) for bd in b["band"]])
    dil = 1 + np.array([(c or {}).get(s, 0.0) for s in b["survey"]], float)
    x = np.where(t > t_last + PAD_DAYS, change["depth"] * f / dil, 0.0)   # k = 1: back up by the full depth
    out = {}
    for s in pd.unique(b["survey"]):
        m = (b["survey"] == s).to_numpy()
        r = fit(b[m], x[m], min_bins=3)
        out[str(s)] = {"recovered": r["k"], "recovered_err": r["k_err"], "n_bins": r["n_bins"],
                       "verdict": _recovery(r["k"], r["k_err"])}
    k, e = combine([(r["recovered"], r["recovered_err"]) for r in out.values()])
    out["all"] = {"recovered": k, "recovered_err": e, "verdict": _recovery(k, e)}
    return out


def _recovery(k, e, k_sigma=K_SIGMA, half=HALF):
    if not (np.isfinite(k) and np.isfinite(e) and e > 0):
        return "no_data"
    if k + k_sigma * e < half:
        return "still_faint"
    if k - k_sigma * e > half:
        return "recovered"
    return "undecided"


def total_change(change):
    """The size of Gaia's change as a fraction of the flux: the trend times its span, or the episode's depth."""
    if change["kind"] == "trend":
        return abs(change["slope"]) * (change["t_last"] - change["t_first"]) / 365.25
    if change["kind"] == "episode":
        return abs(change["depth"])
    return np.nan


def activity(change, tess_amp):
    """Gaia's total change over the star's variability within a month in TESS (`surveys.tess_activity`: the 5th to
    95th percentile range, noise removed). NaN when either is missing or the TESS range is not above its noise."""
    total = total_change(change)
    if not (np.isfinite(total) and tess_amp is not None and np.isfinite(tess_amp) and tess_amp > 0):
        return np.nan
    return float(total / tess_amp)


def columns(same, aft, prefix=""):
    """One flat row of the results for a checks table: per survey `<s>_k`, `<s>_k_err`, `<s>_same`; the combined
    `same_k`, `same_k_err`, `same_years`; and the after results likewise (`after_ratio`, `after_slope`, ... or
    `after_recovered`, ...), with `after` the combined verdict."""
    row = {}
    for s, r in same.items():
        if s == "all":
            row.update({f"{prefix}same_k": r["k"], f"{prefix}same_k_err": r["k_err"],
                        f"{prefix}same_years": r["verdict"]})
        else:
            row.update({f"{prefix}{s}_k": r["k"], f"{prefix}{s}_k_err": r["k_err"], f"{prefix}{s}_bins": r["n_bins"],
                        f"{prefix}{s}_same": r["verdict"]})
    for s, r in aft.items():
        name = f"{prefix}after" if s == "all" else f"{prefix}{s}_after"
        for key, val in r.items():
            if key == "verdict":
                row[name] = val
            elif key != "n_bins":
                row[f"{name}_{key}"] = val
    return row
