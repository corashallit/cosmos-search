"""Slow-change candidates as blind-review packets, and the colour laws the calibration fakes use.

The blind review (docs/vetting.md §4) was built for short dips. A slow-change candidate reaches it as an ordinary
`vetting.candidate` dict: drift-corrected G, BP and RP epochs, ZTF epochs, NEOWISE visit means as bands W1 and W2,
the detected change as the packet's one detected dip, and the per-candidate checks of `slow.checks` as tests whose
descriptions say what they compute and what each explanation predicts (never a calibration figure). The short-dip
pipeline tests (recurrence, single event) answer a different question and are left out of these packets.

Fakes for calibrating the review (docs/vetting.md §14) take a shape from `inject.lightcurve`, read its change in
G, and carry it into every other band as a fixed fraction (`inject.lightcurve.BandScaled`). `ratios` gives those
fractions for each flavour: grey blocking, dust in front, dust that absorbs and re-emits, a change in the star's
temperature, and an artefact in G alone.
"""

import io

import numpy as np
import pandas as pd

from cosmos_search.energy import budget
from cosmos_search.inject.lightcurve import BAND_WAVELENGTH_UM, REF_UM
from cosmos_search.slow import checks, detect

BANDS = ("G", "BP", "RP", "ztf_g", "ztf_r", "ztf_i", "W1", "W2")
FLAVOURS = ("grey", "dust", "reemit", "temperature", "g_only")
HC_OVER_K_UM = 14387.77     # hc/k in micrometre kelvin
SHELL_TEMPS_K = (300.0, 600.0, 1000.0)
ZTF_SPLIT_MJD = checks.DR4_END_MJD
NEOWISE_GAP_DAYS = 30.0
NEOWISE_MIN_PER_VISIT = 3


def _planck_dlnT(lam_um, teff):
    """d ln B_lambda / d ln T for a blackbody: x e^x / (e^x - 1), x = hc / (lambda k T)."""
    x = HC_OVER_K_UM / (lam_um * teff)
    return x * np.exp(x) / np.expm1(x)


def ratios(flavour, teff=5800.0, beta=1.5, t_shell=300.0):
    """Fraction of the G change that each band in BANDS shows, for one flavour (positive: the same sign as G).

    grey: all bands alike. dust: (lambda / lambda_G)^-beta, small changes. reemit: optical as dust, W1/W2 brighten
    by what a shell at t_shell re-emits when it absorbs the G deficit's share of the starlight (`checks.reemission`).
    temperature: the band ratios of a blackbody at teff that cools (d ln B / d ln T at each wavelength over G's).
    g_only: G changes, nothing else does."""
    lam = {b: BAND_WAVELENGTH_UM[b] for b in BANDS}
    if flavour == "grey":
        return {b: 1.0 for b in BANDS}
    if flavour in ("dust", "reemit"):
        out = {b: float((lam[b] / REF_UM) ** -beta) for b in BANDS}
        if flavour == "reemit":
            rem = checks.reemission(1.0, teff, temps=(t_shell,))
            out.update({b: -float(rem[b][0]) for b in ("W1", "W2")})
        return out
    if flavour == "temperature":
        g = _planck_dlnT(REF_UM, teff)
        return {b: float(_planck_dlnT(lam[b], teff) / g) for b in BANDS}
    if flavour == "g_only":
        return {b: (1.0 if b == "G" else 0.0) for b in BANDS}
    raise ValueError(f"unknown flavour {flavour!r}")


def snr(sig, tv, sigma_visit):
    """Signal-to-noise as in S3 (docs/arms/slow-changes.md): the deepest G deficit at the visit times x
    sqrt(visits beyond half of it) / the host's visit noise."""
    k = np.abs(1 - np.asarray(sig.factor(tv, "G"), float))
    d = float(k.max()) if len(k) else 0.0
    n = int(np.sum(k > d / 2)) if d > 0 else 0
    return d * np.sqrt(n) / sigma_visit if sigma_visit > 0 else np.nan


# --- other surveys -------------------------------------------------------------------------------------------


def ztf_raw(ra, dec):
    """ZTF epochs as `dayone.external.ztf` fetches them (same service and cache), keeping each ZTF object's id so
    that a signal can be injected before each object is normalised to its own median. Columns oid, band, t
    (BTJD), mag, magerr."""
    from cosmos_search.dayone import external
    from cosmos_search.net import get_client
    from cosmos_search.net.local import persistent

    url = external.ZTF_LC.format(ra=ra, dec=dec, r=external.ZTF_RADIUS_ARCSEC / 3600)
    body = get_client().get_bytes(url, cache=persistent())
    rows = pd.read_csv(io.BytesIO(body)) if body else pd.DataFrame()
    out = []
    for oid, r in rows.groupby("oid") if len(rows) else ():
        ok = (r["catflags"] == 0) & np.isfinite(r["mag"]) & np.isfinite(r["magerr"])
        band = external.ZTF_BANDS.get(str(r["filtercode"].iloc[0]))
        if ok.sum() < external.ZTF_MIN_EPOCHS or band is None:
            continue
        out.append(pd.DataFrame({"oid": int(oid), "band": band, "t": r.loc[ok, "hjd"].to_numpy(float) - 2457000.0,
                                 "mag": r.loc[ok, "mag"].to_numpy(float),
                                 "magerr": r.loc[ok, "magerr"].to_numpy(float)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["oid", "band", "t", "mag", "magerr"])


def ztf_normalised(raw, signal=None):
    """band, t, flux, err with each ZTF object at its own median, after multiplying in `signal` if given."""
    if not len(raw):
        return pd.DataFrame(columns=["band", "t", "flux", "err"])
    flux = 10 ** (-0.4 * (raw["mag"].to_numpy(float) - 20.0))
    err = flux * raw["magerr"].to_numpy(float) * np.log(10) / 2.5
    if signal is not None:
        k = np.ones(len(raw))
        for b in raw["band"].unique():
            m = (raw["band"] == b).to_numpy()
            k[m] = signal.factor(raw["t"].to_numpy(float)[m], b)
        flux, err = flux * k, err * k
    med = pd.Series(flux).groupby(raw["oid"].to_numpy()).transform("median").to_numpy()
    return pd.DataFrame({"band": raw["band"].to_numpy(), "t": raw["t"].to_numpy(float), "flux": flux / med,
                         "err": err / med})


def neowise_visits(clean, signal=None):
    """NEOWISE visit means from `checks.neowise_clean` output (mjd, band, flux, err), after multiplying in
    `signal` at each exposure if given: band, t (BTJD), flux (relative to the median visit), err (scatter within
    the visit / sqrt(n)), n."""
    out = []
    for b in ("W1", "W2"):
        d = clean[clean["band"] == b]
        if not len(d):
            continue
        mjd, f = d["mjd"].to_numpy(float), d["flux"].to_numpy(float)
        if signal is not None:
            f = f * signal.factor(mjd - checks.BTJD_T0_MJD, b)
        order = np.argsort(mjd)
        mjd, f = mjd[order], f[order]
        g = np.r_[0, np.cumsum(np.diff(mjd) > NEOWISE_GAP_DAYS)]
        v = pd.DataFrame({"g": g, "mjd": mjd, "f": f}).groupby("g").agg(
            mjd=("mjd", "mean"), flux=("f", "mean"), sd=("f", "std"), n=("f", "size"))
        v = v[v["n"] >= NEOWISE_MIN_PER_VISIT]
        if not len(v):
            continue
        med = float(np.median(v["flux"]))
        out.append(pd.DataFrame({"band": b, "t": v["mjd"].to_numpy() - checks.BTJD_T0_MJD,
                                 "flux": v["flux"].to_numpy() / med,
                                 "err": (v["sd"] / np.sqrt(v["n"])).to_numpy() / med, "n": v["n"].to_numpy()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["band", "t", "flux", "err", "n"])


# --- the checks on one candidate -----------------------------------------------------------------------------


def change_window(row):
    """(t0, t1, kind) of the detected change on the Gaia clock: the episode window when the star has one and was
    flagged by it (or only by the repeat search), else the whole span for a trend."""
    by = str(row.get("by", ""))
    has_episode = bool(row.get("fade_kind")) and np.isfinite(row.get("dim_t0", np.nan))
    if has_episode and ("episode" in by or "trend" not in by):
        return float(row["dim_t0"]), float(row["dim_t1"]), str(row["fade_kind"])
    return float(row["t_first"]), float(row["t_last"]), "trend"


def band_changes(epochs, row):
    """Fractional G, BP and RP deficit of the detected change and its error, per band: for an episode the mean
    deficit of the visits inside its window against the median of those outside (`detect._band_depth`); for a
    trend, minus the OLS slope of the visit means times the span, with the slope's error from the residuals."""
    t0, t1, kind = change_window(row)
    out = {}
    for code, name in ((0, "G"), (1, "BP"), (2, "RP")):
        e = epochs[epochs["band"] == code].sort_values("t")
        t, y = e["t"].to_numpy(float), e["frac_corr"].to_numpy(float)
        if kind != "trend":
            out[name] = detect._band_depth(t, y, t0, t1)
            continue
        if len(t) < detect.MIN_VISITS:
            out[name] = (np.nan, np.nan)
            continue
        tv, yv = detect.visits(t, y)
        if len(tv) < detect.MIN_VISITS:
            out[name] = (np.nan, np.nan)
            continue
        x = tv - tv.mean()
        slope, icpt = np.polyfit(x, yv, 1)
        res = yv - slope * x - icpt
        err = np.sqrt(max(np.sum(res ** 2) / (len(x) - 2), detect.visit_sigma(yv) ** 2) / np.sum(x * x))
        span = float(tv[-1] - tv[0])
        out[name] = (float(-slope * span), float(err * span))
    return out


def teff(bp_rp):
    return float(budget.teff_from_bp_rp(bp_rp if bp_rp is not None and np.isfinite(bp_rp) else 0.82))


def tests(row, changes, neowise_clean, ztf, bp_rp):
    """The packet's tests for one candidate: detector scores, colour of the change, NEOWISE and ZTF."""
    t0, t1, kind = change_window(row)
    by = str(row.get("by", "")).replace("+", ", ")
    out = {"slow_change": {
        "s_fade": row.get("s_fade"), "slope_pct_per_yr": 100 * row.get("slope", np.nan), "slope_z": row.get("slope_z"),
        "s_rep": row.get("s_rep"), "rep_period_d": row.get("rep_period"), "abbe": row.get("abbe"),
        "n_visits": row.get("n_visits"), "sigma_visit": row.get("sigma_visit"),
        "description": (
            "The slow-change search's scores on Gaia G visits (epochs less than 2 days apart averaged), each in "
            "units of the star's own visit-to-visit noise sigma_visit (from differences between consecutive "
            "visits), so variability faster than the weeks between visits does not count. Calibration drift shared "
            "by stars of similar magnitude and colour has been removed from G, BP and RP. s_fade: the strongest run "
            "of consecutive faint visits lasting at least 20 days, read in time order. slope_z: the linear trend "
            "over all visits in units of its error (slope_pct_per_yr: the trend). s_rep: the strongest dimming "
            "that repeats at one period with at least two separate events (rep_period_d). abbe: von Neumann ratio "
            "of the visit means, about 1 for noise and near 0 for a smooth change. The search flagged this star "
            f"by: {by or 'none'}. The detected change ({kind}) is the packet's detected dip.")}}

    (dbp, ebp), (drp, erp) = changes["BP"], changes["RP"]
    col = checks.colour(dbp, ebp, drp, erp)
    out["colour_of_change"] = {
        "depth_g": changes["G"][0], "depth_g_err": changes["G"][1], "depth_bp": dbp, "depth_bp_err": ebp,
        "depth_rp": drp, "depth_rp_err": erp, "ratio_bp_over_rp": col["ratio"], "ratio_err": col["ratio_err"],
        "z_grey": col["z_grey"], "z_dust": col["z_dust"],
        "description": (
            "Fractional depth of the detected change in G, BP and RP (mean deficit of the visits inside the "
            "change's dates against the median of those outside; for a trend, the fitted change over the span), "
            "and ratio = BP depth / RP depth. z_grey = (ratio - 1) / ratio_err, z_dust = (ratio - 1.7) / ratio_err. "
            "A body much larger than a micron, or an opaque swarm, dims both bands alike (ratio 1). Interstellar-like "
            "dust dims BP more (ratio about 1.7). A star that cools also dims BP more (about 1.4-1.5 for a Sun-like "
            "star), so a ratio above 1 needs the infrared to tell dust from temperature.")}

    lc = neowise_clean
    depth = changes["G"][0] if np.isfinite(changes["G"][0]) else row.get("dim_depth", np.nan)
    t_eff = teff(bp_rp)
    mjd0, mjd1 = t0 + checks.BTJD_T0_MJD, t1 + checks.BTJD_T0_MJD
    nw = {}
    star = ratios("temperature", t_eff)
    if kind == "trend":
        ws = checks.wise_slope(lc, mjd0, mjd1)
        g_slope = -depth / max(t1 - t0, 1.0) * 365.25 if np.isfinite(depth) else np.nan
        nw["g_slope_per_yr"] = g_slope
        for b in ("W1", "W2"):
            nw[f"{b}_slope_per_yr"], nw[f"{b}_slope_err"], nw[f"{b}_visits"] = ws[b]
            nw[f"{b}_pred_grey"] = g_slope
            nw[f"{b}_pred_dust_in_front"] = g_slope * checks.WISE_OVER_G_DUST[b]
            nw[f"{b}_pred_star_temperature"] = g_slope * star[b]
        what = ("W1/W2 trend of the visit means over the same years as Gaia's (fraction of the median per year), "
                "beside Gaia's G trend (g_slope_per_yr) and what each explanation predicts for it")
    else:
        wc = checks.wise_change(lc, mjd0, mjd1)
        pred = checks.wise_predictions(max(depth, 0.0) if np.isfinite(depth) else 0.0, t_eff)
        for b in ("W1", "W2"):
            nw[f"{b}_change"], nw[f"{b}_err"], nw[f"{b}_visits_in"] = wc[b]
            for k, v in pred[b].items():
                nw[f"{b}_pred_{k}"] = v
            nw[f"{b}_pred_star_temperature"] = -max(depth, 0.0) * star[b] if np.isfinite(depth) else np.nan
        what = ("fractional change of the mean W1/W2 visit flux inside the change's dates against the visits outside "
                "(negative: fainter), with its error from the scatter of the outside visits, beside what each "
                "explanation predicts for the G depth")
    nw["description"] = (
        "NEOWISE W1 (3.4 micron) and W2 (4.6 micron) clean single exposures, averaged per visit (NEOWISE sees a "
        "field for a few days every ~6 months from 2014 on; bands W1 and W2 in the light curves are these visit "
        f"means, each relative to the median visit). Shown: {what}. pred_grey: grey blocking dims W1/W2 as much as "
        "G. pred_dust_in_front: dust dims them by only a few per cent of G. pred_dust_shell_300K / _1000K: dust "
        "that absorbs that share of the starlight and re-emits it at that temperature brightens them. "
        "pred_star_temperature: the star itself cooling dims them less than G (Rayleigh-Jeans side).")
    out["neowise_change"] = nw

    zt = {}
    for b, v in checks.after(ztf).items() if len(ztf) else ():
        zt[f"{b}_change_after_2020"] = v["change"]
        zt[f"{b}_slope_after_2020_per_yr"] = v["slope_post_per_yr"]
        zt[f"{b}_n_before"], zt[f"{b}_n_after"] = v["n_pre"], v["n_post"]
    zt["description"] = (
        "ZTF from 2018 on, after Gaia's light curve ends: per band, the median level after 2020-01-20 against "
        "2018-2019 (change, as a fraction) and the slope after 2020-01-20 (fraction per year). Each ZTF field, CCD "
        "and filter is normalised to its own median, so a step between them is an offset, not a change on the "
        "star. Absent bands had too few epochs on one side.")
    out["ztf_after"] = zt
    return {k: {kk: _plain(vv) for kk, vv in v.items()} for k, v in out.items()}


def _plain(x):
    """Python scalars, so that the packet builder keeps every numeric field (it skips numpy integers)."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    return x


def candidate(candidate_id, row, epochs, source, context, neowise=None, ztf=None, tests_=None):
    """A `vetting.candidate` dict for one slow-change candidate.

    epochs: this star's drift-corrected epochs (band 0/1/2, t, frac_corr, err_frac, fov); neowise: visit means
    (`neowise_visits`); ztf: normalised epochs (`ztf_normalised`); tests_: output of `tests`."""
    lcs = []
    for code, name in ((0, "G"), (1, "BP"), (2, "RP")):
        e = epochs[epochs["band"] == code].sort_values("t")
        if len(e) < 10:
            continue
        lcs.append({"band": name, "time": e["t"].tolist(), "flux": (1 + e["frac_corr"]).tolist(),
                    "flux_err": e["err_frac"].tolist(), "meta": {"fov": e["fov"].astype(int).tolist()}})
    for name, tab in (("ztf", ztf), ("neowise", neowise)):
        if tab is None or not len(tab):
            continue
        for b, d in tab.groupby("band", sort=True):
            d = d.sort_values("t")
            if len(d) >= (10 if name == "ztf" else 3):
                lcs.append({"band": b, "time": d["t"].tolist(), "flux": d["flux"].tolist(),
                            "flux_err": d["err"].tolist(), "meta": {}})
    t0, t1, kind = change_window(row)
    g = epochs[epochs["band"] == 0]
    tv, yv = detect.visits(g.sort_values("t")["t"].to_numpy(float), g.sort_values("t")["frac_corr"].to_numpy(float))
    inside = (tv >= t0) & (tv <= t1)
    t_min = float(tv[inside][np.argmin(yv[inside])]) if inside.any() else float((t0 + t1) / 2)
    depth = (tests_ or {}).get("colour_of_change", {}).get("depth_g", row.get("dim_depth", np.nan))
    dip = {"kind": {"dip": "dimming episode, normal before and after", "fade": "fade, faint to the last visit",
                    "drop": "drop, bright at first and fainter afterwards", "trend": "steady trend over the span"}
           .get(kind, kind),
           "t_start": t0, "t_end": t1, "t_min": t_min, "duration": t1 - t0, "depth": depth,
           "visits_in": int(inside.sum())}
    return {"candidate_id": candidate_id, "arm": "gaia", "source": source, "lightcurves": lcs, "dips": [dip],
            "tests": tests_ or {}, "context": context}
