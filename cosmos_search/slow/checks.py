"""Checks per candidate for the slow-change search (S5): what the change looks like outside Gaia's G band.

Each check turns one question into numbers that a natural explanation predicts differently:

- `colour`: depth in BP over depth in RP during the G episode (from `detect`). A grey occulter (a body much larger
  than the wavelength, or an opaque swarm) dims both equally, ratio 1. Interstellar-like dust dims BP more,
  ratio ~1.7 (DUST_BP_RP). A cooling star also dims BP more, so a ratio above 1 needs the infrared to tell
  dust from temperature.
- `wise_change`: NEOWISE W1 and W2 inside the episode's dates against outside. Grey blocking dims W1/W2 as much
  as G; dust in front dims them by only a few percent of that (WISE_OVER_G_DUST); dust that absorbs starlight
  and re-emits it at a few hundred kelvin brightens them (`reemission`). A change in the star itself dims W1/W2
  far less than G (Rayleigh-Jeans side).
- `wise_slope`: for a steady fade, the W1/W2 trend over the same years, to compare with Gaia's slope.
- `after`: ZTF from 2018 on, inside and past DR4's end (2020-01-20): did the change go on, stop or recover?

Inputs are plain tables so each check can be fed from whichever archive answers on the day.
"""

import numpy as np
import pandas as pd

from cosmos_search.energy import budget

BTJD_T0_MJD = 56999.5          # epoch photometry times are BTJD = JD - 2457000 = MJD - 56999.5
DR4_END_MJD = 58868.0          # 2020-01-20
DUST_BP_RP = 1.7               # A_BP / A_RP for interstellar-like dust and a Sun-like star (Gaia EDR3 extinction law)
WISE_OVER_G_DUST = {"W1": 0.07, "W2": 0.05}   # A_W / A_G for interstellar-like dust, approximately
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP"
NEOWISE_RADIUS_ARCSEC = 3.0


def colour(depth_bp, err_bp, depth_rp, err_rp):
    """dict(ratio, ratio_err, z_grey, z_dust): BP over RP fractional depth and how far it sits from grey (1) and
    from interstellar-like dust (DUST_BP_RP), in units of its error (first-order propagation)."""
    if not (np.isfinite(depth_bp) and np.isfinite(depth_rp)) or depth_rp <= 0:
        return {"ratio": np.nan, "ratio_err": np.nan, "z_grey": np.nan, "z_dust": np.nan}
    r = depth_bp / depth_rp
    e = abs(r) * np.hypot(err_bp / max(abs(depth_bp), 1e-12), err_rp / depth_rp)
    return {"ratio": float(r), "ratio_err": float(e), "z_grey": float((r - 1) / e) if e > 0 else np.nan,
            "z_dust": float((r - DUST_BP_RP) / e) if e > 0 else np.nan}


def neowise_adql(ra, dec, radius_arcsec=NEOWISE_RADIUS_ARCSEC):
    """NEOWISE-R single exposures around (ra, dec) at IRSA. Positions should be propagated to ~2019 first."""
    return ("SELECT mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, qual_frame, cc_flags, nb, qi_fact, saa_sep, "
            "moon_masked FROM neowiser_p1bs_psd WHERE CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {radius_arcsec / 3600:.7f})) = 1")


def neowise_clean(rows):
    """Good single exposures (quality frame > 0, no contamination flags in W1/W2, one component, qi_fact = 1,
    away from the SAA and the Moon) as mjd, band, flux (relative to the median) and err."""
    r = pd.DataFrame(rows)
    if not len(r):
        return pd.DataFrame(columns=["mjd", "band", "flux", "err"])
    cc = r["cc_flags"].astype(str).str.ljust(2, "0")
    ok = ((r["qual_frame"] > 0) & (r["nb"] == 1) & (r["qi_fact"] >= 1) & (r["saa_sep"] > 5)
          & (r["moon_masked"].astype(str).str[:2] == "00"))
    out = []
    for i, b in enumerate(("W1", "W2")):
        m, e = f"w{i + 1}mpro", f"w{i + 1}sigmpro"
        good = ok & (cc.str[i] == "0") & r[m].notna() & r[e].notna()
        mag = r.loc[good, m].to_numpy(float)
        if not len(mag):
            continue
        flux = 10 ** (-0.4 * (mag - np.median(mag)))
        out.append(pd.DataFrame({"mjd": r.loc[good, "mjd"].to_numpy(float), "band": b, "flux": flux,
                                 "err": flux * r.loc[good, e].to_numpy(float) * np.log(10) / 2.5}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["mjd", "band", "flux", "err"])


def _visits(mjd, flux, gap=30.0):
    """NEOWISE visits a field for a few days every ~6 months: mean flux per visit and its error from the scatter."""
    order = np.argsort(mjd)
    mjd, flux = mjd[order], flux[order]
    g = np.r_[0, np.cumsum(np.diff(mjd) > gap)]
    df = pd.DataFrame({"g": g, "mjd": mjd, "flux": flux})
    s = df.groupby("g").agg(mjd=("mjd", "mean"), flux=("flux", "mean"), sd=("flux", "std"), n=("flux", "size"))
    s["err"] = s["sd"] / np.sqrt(s["n"])
    return s[s["n"] >= 3].reset_index(drop=True)


def wise_change(lc, t0_mjd, t1_mjd):
    """Per band: fractional change of the mean visit flux inside [t0, t1] against the visits outside, with its
    error (from the visit-to-visit scatter outside, so it includes the star's own slow variability)."""
    out = {}
    for b in ("W1", "W2"):
        d = lc[lc["band"] == b]
        if len(d) < 6:
            out[b] = (np.nan, np.nan, 0)
            continue
        v = _visits(d["mjd"].to_numpy(float), d["flux"].to_numpy(float))
        inside = (v["mjd"] >= t0_mjd) & (v["mjd"] <= t1_mjd)
        if inside.sum() < 1 or (~inside).sum() < 3:
            out[b] = (np.nan, np.nan, int(inside.sum()))
            continue
        base = float(np.median(v.loc[~inside, "flux"]))
        scatter = float(np.std(v.loc[~inside, "flux"], ddof=1))
        out[b] = (float(v.loc[inside, "flux"].mean() / base - 1), scatter / base / np.sqrt(inside.sum()),
                  int(inside.sum()))
    return out


def wise_slope(lc, t0_mjd, t1_mjd):
    """Per band: linear trend of the visit means inside [t0, t1], in fraction of the median per year, with its
    error from the residual scatter (None below 4 visits). For a steady fade, compare with Gaia's slope: grey
    blocking gives the same slope, dust in front WISE_OVER_G_DUST of it, a re-emitting shell the opposite sign."""
    out = {}
    for b in ("W1", "W2"):
        d = lc[(lc["band"] == b) & (lc["mjd"] >= t0_mjd) & (lc["mjd"] <= t1_mjd)]
        v = _visits(d["mjd"].to_numpy(float), d["flux"].to_numpy(float)) if len(d) else d
        if len(v) < 4:
            out[b] = (np.nan, np.nan, len(v))
            continue
        x = (v["mjd"].to_numpy() - v["mjd"].mean()) / 365.25
        y = v["flux"].to_numpy() / np.median(v["flux"])
        slope, icpt = np.polyfit(x, y, 1)
        res = y - (slope * x + icpt)
        err = np.sqrt(np.sum(res ** 2) / (len(x) - 2) / np.sum(x ** 2))
        out[b] = (float(slope), float(err), len(v))
    return out


def reemission(delta, teff, temps=(300.0, 600.0, 1000.0)):
    """Fractional W1/W2 brightening if a shell absorbs a fraction delta of the star's light and re-emits it at
    each temperature (energy.budget, blackbody grains). dict band -> array over temps."""
    e = budget.excess_per_f(teff, np.asarray(temps, float))[0]
    return {b: delta * e[:, budget.BANDS.index(b)] for b in ("W1", "W2")}


def wise_predictions(delta, teff):
    """Expected fractional W1/W2 change for a G deficit delta under each explanation."""
    rem = reemission(delta, teff)
    return {b: {"grey": -delta, "dust_in_front": -delta * WISE_OVER_G_DUST[b],
                "dust_shell_300K": float(rem[b][0]), "dust_shell_1000K": float(rem[b][2])}
            for b in ("W1", "W2")}


def after(ztf, split_mjd=DR4_END_MJD):
    """Per ZTF band: median flux before and after split_mjd (each light curve normalised to its own median, so
    the change is a fraction) and the slope after it, in fraction per year. ztf: band, t (BTJD), flux."""
    out = {}
    for b, d in ztf.groupby("band"):
        mjd = d["t"].to_numpy(float) + 2457000.0 - 2400000.5
        f = d["flux"].to_numpy(float)
        pre, post = f[mjd < split_mjd], f[mjd >= split_mjd]
        if len(pre) < 10 or len(post) < 10:
            continue
        x = (mjd[mjd >= split_mjd] - split_mjd) / 365.25
        slope = float(np.polyfit(x, post, 1)[0]) if np.ptp(x) > 0.5 else np.nan
        out[b] = {"pre": float(np.median(pre)), "post": float(np.median(post)),
                  "change": float(np.median(post) / np.median(pre) - 1), "slope_post_per_yr": slope,
                  "n_pre": len(pre), "n_post": len(post)}
    return out
