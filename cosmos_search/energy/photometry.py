"""From catalogue rows to what the energy-budget fit needs: per band a flux, its error, whether it is usable, whether
it counts only as an upper bound, and the photosphere template.

Canonical input columns (the Gaia-archive reference join in reference.py uses these names; `from_context` renames
the xmatch context table into them):

    ks_m, ks_msigcom, t_ph_qual (2MASS JHK quality letters), t_mates (other Gaia stars sharing this 2MASS source)
    w1mpro..w4mpro, w1mpro_error..w4mpro_error, w_ph_qual, w_cc_flags, w_ext_flag, w_mates
    optional: blend_w1 (Gaia neighbour flux in the 6" beam over the star's), blend_w4 (12" beam), sigma_var_mag

Fluxes are Vega-relative (10^(-0.4 m)); every quantity the fit uses is a ratio within one band, so zero points
never enter.

Rules (why in brackets):
- A 2MASS or WISE detection (quality A, B or C) is a measurement with its catalogue error.
- A detection below SNR_MEASURE counts only as an upper bound: a faint source is detected when its noise is up, so
  its flux is biased high (measured on held-out reference stars: W3/W4 detections of faint stars sat +0.7/+2.1
  sigma above their photospheres). Saturated bands are left out.
- WISE 'U' is a 95% upper limit, computed by AllWISE as measured flux + 2 sigma. It enters as 0 +- limit / 2, so a
  model at the limit sits 2 sigma high and one at 1.5x the limit 3 sigma.
- Anything that could add flux from elsewhere only bounds the star's flux from above: a WISE contamination flag
  (diffraction spike, persistence, halo, ghost) in that band, an extended source, another Gaia star sharing the
  WISE or 2MASS source, or Gaia neighbours contributing more than BLEND_MAX of the star's flux in the beam. [A
  blend can fake an excess but not hide one; the deficit test needs only upper bounds.]
- Gaia G (mean over 2014-2017) and 2MASS K (1998-2001) anchor the photosphere's scale. Neither was taken with WISE,
  so their errors get sigma_var_mag added in quadrature. W1-W4 are simultaneous in AllWISE.
"""

import numpy as np
import pandas as pd

from .budget import BANDS, teff_from_bp_rp

LN10_25 = np.log(10) / 2.5
BLEND_MAX = 0.05
SNR_MEASURE = 5.0
# Brighter than these the band is saturated and AllWISE's profile fit drifts (W1-W3 moved by 0.04 mag across W1 6-8
# in the reference); a saturated band is left out.
SATURATION = {"W1": 8.1, "W2": 6.7, "W3": 3.8, "W4": -0.4, "K": 4.5}
W_BANDS = ("W1", "W2", "W3", "W4")


def _str(v, n):
    return v if isinstance(v, str) and len(v) >= n else "X" * n


def _col(raw, name, default):
    """Numeric column as an array, or `default` everywhere when the table lacks it."""
    if name not in raw:
        return np.full(len(raw), default, float)
    return pd.to_numeric(raw[name], errors="coerce").fillna(default).to_numpy(float)


def _flux(m):
    return 10 ** (-0.4 * np.asarray(m, float))


def design(raw, locus):
    """raw: canonical photometry (DataFrame); locus: DataFrame from photosphere.Locus.predict with c_<colour>,
    s_<colour> (mag). Returns the fit table (budget.fit's columns) indexed like raw."""
    raw = raw.reset_index(drop=True)
    locus = locus.reset_index(drop=True)
    n = len(raw)
    out = pd.DataFrame(index=raw.index)
    wq = raw["w_ph_qual"].map(lambda v: _str(v, 4))
    cc = raw["w_cc_flags"].map(lambda v: _str(v, 4) if isinstance(v, str) else "0000")
    ext = _col(raw, "w_ext_flag", 0) > 0
    w_mates = _col(raw, "w_mates", 0) > 0
    blend_w1 = _col(raw, "blend_w1", 0) > BLEND_MAX
    blend_w4 = _col(raw, "blend_w4", 0) > BLEND_MAX
    for i, b in enumerate(W_BANDS, start=1):
        m = raw[f"w{i}mpro"].to_numpy(float)
        e = raw[f"w{i}mpro_error"].to_numpy(float)
        q = wq.str[i - 1].to_numpy()
        det = np.isin(q, list("ABC")) & np.isfinite(m) & np.isfinite(e) & (e > 0) & (m > SATURATION[b])
        ul = (q == "U") & np.isfinite(m)
        phi = _flux(m)
        out[f"d_{b}"] = np.where(det, phi, np.where(ul, 0.0, np.nan))
        out[f"s_{b}"] = np.where(det, phi * e * LN10_25, np.where(ul, phi / 2, np.nan))
        out[f"use_{b}"] = det | ul
        flagged = cc.str[i - 1].to_numpy() != "0"
        beam = blend_w4 if b == "W4" else blend_w1
        low_snr = e > 2.5 / np.log(10) / SNR_MEASURE
        out[f"bound_{b}"] = det & (flagged | ext | w_mates | beam | low_snr)
    km, ke = raw["ks_m"].to_numpy(float), raw["ks_msigcom"].to_numpy(float)
    kq = raw["t_ph_qual"].map(lambda v: _str(v, 3)).str[2].to_numpy()
    kdet = np.isin(kq, list("ABC")) & np.isfinite(km) & np.isfinite(ke) & (ke > 0) & (km > SATURATION["K"])
    phi_k = _flux(km)
    out["d_K"] = np.where(kdet, phi_k, np.nan)
    out["use_K"] = kdet
    gm = _col(raw, "phot_g_mean_mag", np.nan)
    foe = _col(raw, "phot_g_mean_flux_over_error", np.nan)
    gdet = np.isfinite(gm) & (foe > 0)
    phi_g = _flux(gm)
    out["d_G"] = np.where(gdet, phi_g, np.nan)
    out["use_G"] = gdet
    out["bound_G"] = np.zeros(n, bool)
    t_mates = _col(raw, "t_mates", 0) > 0
    out["bound_K"] = kdet & (t_mates | (ke > 2.5 / np.log(10) / SNR_MEASURE))

    # Photosphere template relative to W1, from the locus colours: K = W1 + c_K1, Wj = W1 - c_1j, W4 = W3 - c_34.
    c = {k: locus[f"c_{k}"].to_numpy(float) for k in ("G1", "K1", "12", "13", "34")}
    sm = {k: locus[f"s_{k}"].to_numpy(float) for k in ("G1", "K1", "12", "13", "34")}
    P = {"W1": np.ones(n), "G": 10 ** (-0.4 * c["G1"]), "K": 10 ** (-0.4 * c["K1"]), "W2": 10 ** (0.4 * c["12"]), "W3": 10 ** (0.4 * c["13"]),
         "W4": 10 ** (0.4 * (c["13"] + c["34"]))}
    s_mod = {"W1": np.zeros(n), "G": np.hypot(sm["G1"], _col(raw, "sigma_var_mag", 0.0)), "K": sm["K1"], "W2": sm["12"], "W3": sm["13"], "W4": np.hypot(sm["13"], sm["34"])}
    var = _col(raw, "sigma_var_mag", 0.0)
    s_mod["K"] = np.hypot(s_mod["K"], var)
    # Photosphere scale for the model errors: W1 if it is a clean measurement, else K.
    w1_clean = out["use_W1"].to_numpy() & ~out["bound_W1"].to_numpy() & (out["d_W1"].to_numpy() > 0)
    a0 = np.where(w1_clean, out["d_W1"].to_numpy(), np.where(kdet, phi_k / P["K"], np.nan))
    s_k_meas = np.where(kdet, phi_k * ke * LN10_25, np.nan)
    for b in BANDS:
        out[f"P_{b}"] = P[b]
        model = a0 * P[b] * s_mod[b] * LN10_25
        meas = {"K": s_k_meas, "G": phi_g / foe}.get(b, out[f"s_{b}"].to_numpy() if f"s_{b}" in out else None)
        # Upper limits already carry their own width; the template error matters only for detections.
        is_ul = (out[f"d_{b}"].to_numpy() == 0) if b not in ("K", "G") else np.zeros(n, bool)
        out[f"s_{b}"] = np.where(is_ul, meas, np.hypot(meas, np.nan_to_num(model)))
        out[f"use_{b}"] = out[f"use_{b}"].to_numpy() & np.isfinite(P[b]) & np.isfinite(out[f"s_{b}"].to_numpy())
    out["teff"] = teff_from_bp_rp(raw["bp_rp"].to_numpy(float))
    out["a0"] = a0
    return out


def sigma_var_from_gaia(raw):
    """Epoch-to-epoch scatter of G in magnitudes, from gaia_source alone: the mean's flux error times sqrt(n_obs).
    It includes photon noise, so it is an upper bound on the star's own variability; good enough for the K term."""
    foe = _col(raw, "phot_g_mean_flux_over_error", np.nan)
    nobs = _col(raw, "phot_g_n_obs", np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((foe > 0) & (nobs > 0), np.sqrt(nobs) / foe / LN10_25, np.nan)


CONTEXT_MAP = {
    "2mass_k_m": "ks_m", "2mass_k_msigcom": "ks_msigcom", "2mass_ph_qual": "t_ph_qual",
    "allwise_w1mpro": "w1mpro", "allwise_w1sigmpro": "w1mpro_error", "allwise_w2mpro": "w2mpro",
    "allwise_w2sigmpro": "w2mpro_error", "allwise_w3mpro": "w3mpro", "allwise_w3sigmpro": "w3mpro_error",
    "allwise_w4mpro": "w4mpro", "allwise_w4sigmpro": "w4mpro_error", "allwise_ph_qual": "w_ph_qual",
    "allwise_cc_flags": "w_cc_flags", "allwise_ext_flg": "w_ext_flag",
    "blend_wise_w1_contamination": "blend_w1", "blend_wise_w4_contamination": "blend_w4",
}


def from_context(wide, gaia=None):
    """Canonical photometry from an xmatch context table (`cosmos_search.xmatch.context`), joined to the Gaia
    columns (bp_rp, phot_g_mean_mag, parallax, ...) from `gaia` on source_id when given.

    The xmatch survey_multiplicity (sum of P over Gaia claimants of one survey row) stands in for DPAC's
    number_of_mates: above 1.2 the row is shared, so it bounds rather than measures. A survey match below
    P_MATCH_MIN is treated as no match (the band is missing, not an upper limit)."""
    df = wide.rename(columns=CONTEXT_MAP).copy()
    for s, col in (("allwise", "w_mates"), ("2mass", "t_mates")):
        df[col] = (_col(wide, f"{s}_multiplicity", 0.0) > 1.2).astype(int)
        if f"{s}_p_cal" in wide:
            weak = ~(_col(wide, f"{s}_p_cal", 0.0) >= P_MATCH_MIN)
            cols = ["ks_m", "ks_msigcom"] if s == "2mass" else [f"w{i}mpro{e}" for i in range(1, 5) for e in ("", "_error")]
            df.loc[weak, [c for c in cols if c in df]] = np.nan
    if gaia is not None:
        g = gaia.drop(columns=[c for c in gaia.columns if c in df.columns and c != "source_id"])
        df = df.merge(g, on="source_id", how="left")
    if "sigma_var_mag" not in df:
        df["sigma_var_mag"] = sigma_var_from_gaia(df)
    return df


P_MATCH_MIN = 0.9
