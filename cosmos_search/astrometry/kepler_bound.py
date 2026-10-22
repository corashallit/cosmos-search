"""Catalogue-level filter: orbits and accelerations whose gravitational explanation needs an extraordinary mass.

Runs on Gaia's own non-single-star (NSS) solutions (DR3 now, DR4 on day one), before any epoch data is pulled.

For each astrometric orbit: a0 from the Thiele-Innes constants, the astrometric mass function f_A = (a0/plx)^3/P^2
(physics.py: the total mass any gravitational explanation needs), and the luminous star's mass m1 (FLAME, else a
main-sequence guess from M_G). Classes:
    dark        f_A > m1: the companion outweighs the luminous star yet adds no light (WD/NS/BH, or a hidden
                luminous pair); Shahaf et al.'s AMRF class III
    beyond      f_A > M_CAP (100 Msun): heavier than any known stellar remnant
For each acceleration solution: M_min = (g/plx)^3 (2T)^4 / (2 pi)^6 (no curvature over the span T), and for
9-parameter solutions also the circular-orbit mass g^7 / (4 pi^2 j^4) with P = 2 pi g / j.

Vetoes (natural and instrumental explanations seen in the DR3 tail):
    prec_alias  period within 2% of 63.12/k d (k = 1..10), the precession period of Gaia's spin axis and its
                harmonics: the scanning law, not the star
    year_alias  period within 3% of 365.25/k d (k = 1..3): parallax leaking into the orbit
    low_plx     parallax_over_error < 10: f_A goes as plx^-3
    bad_fit     goodness_of_fit > 5: the published model does not describe the data
    alt_short   OrbitalAlternative solution with P < 100 d (17 of DR3's 19 f_A > 100 orbits)
    crowded     ipd_frac_multi_peak > 10 or ipd_frac_odd_win > 10 (resolved neighbour or odd windows)
"""

import numpy as np
import pandas as pd

from . import physics
from .fit import thiele_innes

PRECESSION_DAYS = 63.12
DR3_SPAN_YR = 2.84


def _near(period, base, ks, tol):
    k = np.round(base / period)
    return (k >= ks[0]) & (k <= ks[1]) & (np.abs(period * k / base - 1) < tol)


def luminous_mass(df):
    mg = df["phot_g_mean_mag"] + 5 * np.log10(np.clip(df["parallax"], 1e-3, None) / 100)
    guess = np.clip(10 ** (-0.1 * (mg - 4.7)), 0.1, 3.0)
    return np.where(df["mass_flame"].notna(), df["mass_flame"], guess) if "mass_flame" in df else guess


def orbits(df):
    out = df.copy()
    out["a0_mas"] = [thiele_innes(a, b, f, g)["a0_mas"] for a, b, f, g in
                     zip(df["a_thiele_innes"], df["b_thiele_innes"], df["f_thiele_innes"], df["g_thiele_innes"])]
    out["f_A"] = physics.mass_function(out["a0_mas"], out["parallax"], out["period"] / 365.25)
    out["m1"] = luminous_mass(df)
    p = out["period"]
    out["prec_alias"] = _near(p, PRECESSION_DAYS, (1, 10), 0.02)
    out["year_alias"] = _near(p, 365.25, (1, 3), 0.03)
    out["low_plx"] = out["parallax"] / out["parallax_error"] < 10
    out["bad_fit"] = out["goodness_of_fit"] > 5
    out["alt_short"] = out["nss_solution_type"].str.startswith("OrbitalAlternative") & (p < 100)
    out["crowded"] = (out["ipd_frac_multi_peak"] > 10) | (out["ipd_frac_odd_win"] > 10)
    veto = ["prec_alias", "year_alias", "low_plx", "bad_fit", "alt_short", "crowded"]
    out["vetoed"] = out[veto].any(axis=1)
    out["veto"] = out[veto].apply(lambda r: ",".join(c for c in veto if r[c]), axis=1)
    out["cls"] = np.where(out["f_A"] > physics.M_CAP, "beyond", np.where(out["f_A"] > out["m1"], "dark", ""))
    return out


def accelerations(df, span_yr=DR3_SPAN_YR):
    out = df.copy()
    g = np.hypot(df["accel_ra"], df["accel_dec"])
    out["g"] = g
    out["g_err"] = np.sqrt((df["accel_ra"] * df["accel_ra_error"]) ** 2 +
                           (df["accel_dec"] * df["accel_dec_error"]) ** 2) / g
    plx = np.clip(df["parallax"], 1e-3, None)
    out["m_min"] = physics.acceleration_min_mass(g, plx, span_yr)
    j = np.hypot(df["deriv_accel_ra"], df["deriv_accel_dec"])
    gp, jp = g / plx, j / plx
    out["m_circ"] = np.where(j > 0, gp ** 7 / (4 * np.pi ** 2 * jp ** 4), np.nan)
    out["p_circ_yr"] = np.where(j > 0, 2 * np.pi * g / j, np.nan)
    out["m1"] = luminous_mass(df)
    out["low_plx"] = df["parallax"] / df["parallax_error"] < 10
    out["bad_fit"] = df["goodness_of_fit"] > 5
    out["crowded"] = (df["ipd_frac_multi_peak"] > 10) | (df["ipd_frac_odd_win"] > 10)
    out["vetoed"] = out[["low_plx", "bad_fit", "crowded"]].any(axis=1)
    out["cls"] = np.where(out["m_min"] > physics.M_CAP, "beyond", np.where(out["m_min"] > out["m1"], "dark", ""))
    return out


def summary(orb, acc):
    rows = []
    for name, d in (("orbits", orb), ("accel", acc)):
        for cls in ("dark", "beyond"):
            m = d["cls"] == cls
            rows.append({"table": name, "class": cls, "n": int(m.sum()), "after_veto": int((m & ~d["vetoed"]).sum())})
    return pd.DataFrame(rows)
