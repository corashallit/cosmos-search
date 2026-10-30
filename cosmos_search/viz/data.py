"""Inputs for the figures: a star's epochs as visits, other surveys' epochs, and the comparison population.

Plot functions take plain DataFrames so the same code serves DR3 trial output and DR4:

- epochs: one row per Gaia epoch with `band` (0 G, 1 BP, 2 RP, or the names), `t` (BTJD), `frac` (fractional change
  from the star's normal level, drift-corrected where available) and `err_frac`; `drift.normalise` gives this.
- others: one row per epoch or visit of another survey with `band` (ztf_g, ztf_r, ztf_i, W1, W2), `t` (BTJD), `flux`
  (1 at the star's normal level) and `err`, as `slow.vet.ztf_normalised` and `slow.vet.neowise_visits` give them.
- population: one row per star of the comparison sample with the detector's columns (`slow.search.score`), plus
  `phot_g_mean_mag` and `bp_rp`; `strengths` adds the fading and brightening strengths the lists are ordered by.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.slow import detect, drift

BAND_CODE = {"G": 0, "BP": 1, "RP": 2}
BAND_NAME = {0: "G", 1: "BP", 2: "RP"}

# Plain names for DPAC's variability classes (vari_classifier_result.best_class_name), for panel titles.
CLASS_NAMES = {
    "ECL": "eclipsing binaries", "AGN": "quasars and other active galaxies", "LPV": "long-period variables",
    "RS": "spotted stars (RS CVn)", "BE|GCAS|SDOR|WR": "Be and other emission-line stars", "YSO": "young stars",
    "SOLAR_LIKE": "Sun-like active stars", "RR": "RR Lyrae stars", "CEP": "Cepheids",
    "DSCT|GDOR|SXPHE": "δ Scuti and γ Dor pulsators", "CV": "cataclysmic variables", "RCB": "R CrB stars",
    "MICROLENSING": "microlensing events", "ELL": "ellipsoidal binaries", "WD": "white dwarfs", "SN": "supernovae",
    "S": "short-timescale variables", "SYST": "symbiotic stars",
}


# Shorter names for small-multiple panel titles (at most ~24 characters).
CLASS_SHORT = {
    "ECL": "eclipsing binaries", "AGN": "quasars", "LPV": "long-period variables", "RS": "spotted stars",
    "BE|GCAS|SDOR|WR": "Be stars", "YSO": "young stars", "SOLAR_LIKE": "Sun-like active stars",
    "DSCT|GDOR|SXPHE": "δ Scuti, γ Dor", "RR": "RR Lyrae", "CEP": "Cepheids", "CV": "cataclysmic variables",
}


def band_of(x):
    """Band codes 0/1/2 from codes or names."""
    x = pd.Series(x)
    if x.dtype == object or str(x.dtype).startswith(("string", "category")):
        return x.astype(str).map(BAND_CODE).to_numpy()
    return x.to_numpy().astype(int)


def epochs(lc, cells=None, table=None):
    """Epochs (source_id, band, t, frac, err_frac, fov) from a light-curve table (gaia.epoch.SCHEMA), with `frac`
    drift-corrected when the drift table and the star's cells (drift.star_cells) are given."""
    ep = drift.normalise(lc)
    if cells is not None and table is not None and len(ep):
        ep = ep[ep["source_id"].isin(cells.index)].copy()
        ep["frac"] = ep["frac"].to_numpy() - drift.drift_at(ep, cells, table)
    return ep


def visits(ep, band):
    """Visit means of one band: t, y (fractional change), err, n. Epochs closer than detect.VISIT_GAP_DAYS are one
    visit; err is the larger of the propagated epoch errors and the scatter inside the visit over sqrt(n)."""
    e = ep[band_of(ep["band"]) == BAND_CODE.get(band, band)].sort_values("t")
    if not len(e):
        return pd.DataFrame(columns=["t", "y", "err", "n"])
    t, y = e["t"].to_numpy(float), e["frac"].to_numpy(float)
    err = e["err_frac"].to_numpy(float)
    new = np.r_[True, np.diff(t) > detect.VISIT_GAP_DAYS]
    idx = np.cumsum(new) - 1
    n = np.bincount(idx).astype(float)
    tv, yv = np.bincount(idx, t) / n, np.bincount(idx, y) / n
    prop = np.sqrt(np.bincount(idx, err ** 2)) / n
    dev2 = np.bincount(idx, (y - yv[idx]) ** 2)
    scat = np.sqrt(dev2 / np.maximum(n - 1, 1)) / np.sqrt(n)
    return pd.DataFrame({"t": tv, "y": yv, "err": np.maximum(prop, np.where(n > 1, scat, 0)), "n": n.astype(int)})


def others(*tables):
    """One `others` table (band, t, flux, err) from tables already in that form: `slow.vet.ztf_normalised` for ZTF,
    `slow.vet.neowise_visits` for NEOWISE. Empty or missing tables are skipped."""
    parts = [t[["band", "t", "flux", "err"]] for t in tables if t is not None and len(t)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["band", "t", "flux", "err"])


def thresholds(release_json):
    """{group: {"s": .., "z": .., "rep": ..}} from a rule directory's release.json (slow_release_thresholds.py)."""
    meta = json.loads(Path(release_json).read_text())
    return {g: {k: v["value"] for k, v in d.items()} for g, d in meta["thresholds"].items()}


def group(g_mag):
    """Magnitude range of the rule ('G<19' or 'G>=19') per star."""
    return np.where(np.asarray(g_mag, float) < 19, "G<19", "G>=19")


def strengths(pop, thr, g_col="phot_g_mean_mag"):
    """Fading and brightening strength per star, as the lists order them (slow.search.flag): the largest of each
    score over its threshold. Brightening uses the mirror scores (s_rise_ns or s_rise, +slope_z, s_rep_bright)."""
    grp = group(pop[g_col])
    th = pd.DataFrame(thr).T.reindex(grp)
    s_th, z_th = th["s"].to_numpy(), th["z"].to_numpy()
    r_th = th["rep"].to_numpy() if "rep" in th else np.full(len(pop), np.inf)

    def col(name, alt=None):
        if name in pop:
            return pop[name].to_numpy(float)
        if alt and alt in pop:
            return pop[alt].to_numpy(float)
        return np.full(len(pop), np.nan)

    def best(*parts):
        p = np.vstack(parts)
        return np.max(np.where(np.isfinite(p), p, -np.inf), axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        fade = best(col("s_fade") / s_th, -col("slope_z") / z_th, col("s_rep") / r_th)
        rise = best(col("s_rise_ns", "s_rise") / s_th, col("slope_z") / z_th, col("s_rep_bright") / r_th)
    return pd.DataFrame({"fade_strength": fade, "rise_strength": rise, "group": grp}, index=pop.index)


def typical_noise(pop, g_mag, width=0.5, g_col="phot_g_mean_mag", quiet=True, class_col="best_class_name"):
    """Median visit-to-visit scatter (fraction) of comparison stars within +-width mag of g_mag (or in its drift G bin
    when pop has no magnitudes); with quiet=True only stars below the thresholds in both directions and without a
    DPAC variability class count, so the band shows what an ordinary star does. (DR3 published light curves only
    for stars DPAC flagged as variable: there the classed ones scatter about 3x more than the rest at G = 15.)"""
    if g_col in pop:
        m = np.abs(pop[g_col].to_numpy(float) - g_mag) < width
    else:
        m = pop["gbin"].to_numpy() == np.digitize([g_mag], drift.G_EDGES[1:-1])[0]
    if quiet and "fade_strength" in pop:
        m &= ((pop["fade_strength"] < 1) & (pop["rise_strength"] < 1)).to_numpy()
    if quiet and class_col in pop and (m & pop[class_col].isna().to_numpy()).sum() >= 50:
        m &= pop[class_col].isna().to_numpy()
    s = pop.loc[m, "sigma_visit"].to_numpy(float)
    return float(np.nanmedian(s)) if len(s) else np.nan


def survival(x, at):
    """Number of values >= each of `at` (counts for a tail curve)."""
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    return len(x) - np.searchsorted(x, np.asarray(at, float), side="left")
