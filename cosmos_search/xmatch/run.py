"""One call from a Gaia target table to a calibrated match table with its false-match accounting."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .astrometry import Astrometry
from .calibrate import (DEFAULT_G_BINS, apply_odds, best_per_unit, chance_probability, fit, odds_correction,
                        strata, summary)
from .compete import competition
from .geometry import healpix_groups
from .match import ErrorModel, gather, posteriors
from .sources import AutoSource
from .surveys import GAIA_DR3, SURVEYS

MIN_CALIBRATION_TARGETS = 300
CHANCE_THRESHOLD = 0.9
COMPETE_REGION_ORDER = 6  # ~0.9 deg pixels over which the claimed background fraction is averaged
KEEP_P = 0.01  # pairs below this are dropped from the output table


@dataclass
class Result:
    survey: str
    matches: pd.DataFrame     # one row per (target, candidate) with P >= KEEP_P
    best: pd.DataFrame        # one row per target and survey group: best candidate or none
    model: ErrorModel
    prior: np.ndarray         # per stratum
    report: dict = field(default_factory=dict)


def crossmatch(gaia, survey, *, source=None, n_shifts=2, seed=0, calibrate=True, model=None, prior=None,
               g_bins=DEFAULT_G_BINS, group_side_deg=0.5, id_col="source_id", workers=4, compete=True,
               gaia_catalog=GAIA_DR3):
    """Match Gaia targets to one survey.

    gaia: table with gaia_source columns. survey: name in SURVEYS or a Survey.
    calibrate: fit the error scale, floor and per-magnitude prior from these targets (needs a few hundred;
    below MIN_CALIBRATION_TARGETS the given or default model and prior are used and the report says so).
    compete: let every Gaia source near a survey source compete for it (compete.py), with Gaia neighbours
    read from gaia_catalog through `source`. Unresolved pairs (closer than half the survey FWHM) share.
    """
    survey = SURVEYS[survey] if isinstance(survey, str) else survey
    source = source or AutoSource()
    gaia_df = gaia.to_pandas() if hasattr(gaia, "to_pandas") else pd.DataFrame(gaia)
    astro = Astrometry.from_table(gaia_df)
    ids = np.asarray(gaia_df[id_col]) if id_col in gaia_df else np.zeros(0)
    nb_source = getattr(source, "hats", source)
    cands = gather(gaia_df, survey, source, n_shifts=n_shifts, seed=seed, group_side_deg=group_side_deg,
                   astrometry=astro, workers=workers,
                   neighbours=(nb_source, gaia_catalog, ids) if compete else None)
    gmag = np.asarray(gaia_df.get("phot_g_mean_mag", pd.Series(np.nan, index=gaia_df.index)), float)
    stratum = strata(gmag, g_bins)
    n_strata = len(g_bins) - 1
    model = model or ErrorModel(floor_mas=survey.floor_mas)
    if prior is None:
        prior = np.full(n_strata, survey.prior)
    report = {"survey": survey.name, "n_targets": len(gaia_df), "fetch": cands.stats, "calibrated": False,
              "compete": bool(compete)}
    nbrs = cands.nbrs
    if compete:
        nb_stratum = strata(nbrs.gmag, g_bins)
        blend_r = survey.fwhm_arcsec * 500.0  # half the FWHM, in mas: the most that sharing a row can span
        region = healpix_groups(astro.ra, astro.dec, COMPETE_REGION_ORDER)
        report["neighbour_pairs"] = len(nbrs)
        cands.log_comp, cands.claimed = competition(cands, nbrs, model, prior[stratum], prior[nb_stratum],
                                                    blend_r, region=region)
    if calibrate and len(gaia_df) >= MIN_CALIBRATION_TARGETS:
        model, prior, info = fit(cands, stratum, n_strata=n_strata, model=model)
        if compete:
            # Competition depends on the model; one more round with the fitted one.
            cands.log_comp, cands.claimed = competition(cands, nbrs, model, prior[stratum], prior[nb_stratum],
                                                        blend_r, region=region)
            model, prior, info = fit(cands, stratum, n_strata=n_strata, model=model)
        report["calibrated"] = True
        report["fit"] = info
    report["prior_by_g"] = {_bin_label(g_bins, i): float(prior[i]) for i in range(n_strata)}
    pi_t = prior[stratum]
    report.update(summary(cands, model, pi_t, stratum, seed=seed))
    p_chance = chance_probability(cands, model, pi_t, CHANCE_THRESHOLD)
    report["p_chance_check"] = {
        "threshold": CHANCE_THRESHOLD, "mean_p_chance": float(np.mean(p_chance)),
        "fmr_shift": next(r["fmr_shift"] for r in report["thresholds"] if r["threshold"] == CHANCE_THRESHOLD)}
    t = next(r for r in report["thresholds"] if r["threshold"] == CHANCE_THRESHOLD)
    p_pair, p_none, _, _ = posteriors(cands, model, pi_t)
    _, best_p = best_per_unit(cands, p_pair, p_none)
    acc_p = best_p[(cands.unit_shift < 0) & (best_p >= CHANCE_THRESHOLD)]
    kappa = odds_correction(acc_p, t["false_shift"], t["false_rows_shift"])
    matches, best = tables(cands, gaia_df, model, pi_t, id_col=id_col, kappa=kappa)
    report["shift_calibration"] = {
        "kappa": kappa, "threshold": CHANCE_THRESHOLD, "accepted": int(len(acc_p)),
        "false_model": t["false_model"], "false_shift": t["false_shift"],
        "false_cal": float(np.sum(1 - apply_odds(acc_p, kappa))),
        "accepted_cal": int((best.p_cal >= CHANCE_THRESHOLD).sum())}
    pc = pd.Series(p_chance, index=np.asarray(gaia_df[id_col]) if id_col in gaia_df else np.arange(len(gaia_df)))
    best[f"p_chance_{CHANCE_THRESHOLD:g}"] = best[id_col].map(pc).to_numpy()
    matches[f"p_chance_{CHANCE_THRESHOLD:g}"] = matches[id_col].map(pc).to_numpy()
    return Result(survey.name, matches, best, model, prior, report)


def tables(cands, gaia_df, model, pi_t, *, id_col="source_id", kappa=1.0):
    survey = cands.survey
    p_pair, p_none, chi2, llr = posteriors(cands, model, pi_t)
    cal = lambda p: apply_odds(p, kappa)
    real = cands.shift < 0
    ids = np.asarray(gaia_df[id_col]) if id_col in gaia_df else np.arange(len(gaia_df))

    # Survey-side multiplicity: how many Gaia targets claim each survey row (sum of P). Values well above 1
    # mean one survey source is the counterpart of a blend of Gaia sources.
    mult = np.bincount(cands.row[real], weights=p_pair[real], minlength=len(cands.rows))
    n_cand = np.bincount(cands.unit[real], minlength=len(cands.unit_target))

    keep = real & (p_pair >= KEEP_P)
    rows = cands.rows.iloc[cands.row[keep]].reset_index(drop=True)
    sep = np.hypot(cands.d[keep, 0], cands.d[keep, 1]) / 1e3
    m = pd.DataFrame({
        id_col: ids[cands.target[keep]],
        f"{survey.name}_id": rows[survey.id].to_numpy(),
        "p_match": p_pair[keep],
        "p_cal": cal(p_pair[keep]),
        "p_none": p_none[cands.unit[keep]],
        "sep_arcsec": sep,
        "chi2": chi2[keep],
        "log_lr": llr[keep],
        "n_candidates": n_cand[cands.unit[keep]],
        "survey_multiplicity": mult[cands.row[keep]],
        "epoch": rows["_epoch"].to_numpy(),
    })
    for c in survey.keep:
        if c in rows and c not in m:
            m[c] = rows[c].to_numpy()
    m = m.sort_values([id_col, "p_match"], ascending=[True, False]).reset_index(drop=True)
    m["rank"] = m.groupby(id_col).cumcount() + 1

    best_i, best_p = best_per_unit(cands, p_pair, p_none)
    ru = cands.unit_shift < 0
    bi = best_i[ru]
    has = bi >= 0
    best = pd.DataFrame({
        id_col: ids[cands.unit_target[ru]],
        "group": cands.unit_group[ru],
        "p_best": best_p[ru],
        "p_cal": cal(best_p[ru]),
        "p_none": p_none[ru],
        "n_candidates": n_cand[ru],
        "n_local": cands.n_local[cands.unit_target[ru]] if cands.n_local is not None else -1,
    })
    best[f"{survey.name}_id"] = None
    best.loc[has, f"{survey.name}_id"] = cands.rows[survey.id].to_numpy()[cands.row[bi[has]]] if has.any() else []
    best["sep_arcsec"] = np.nan
    best.loc[has, "sep_arcsec"] = np.hypot(cands.d[bi[has], 0], cands.d[bi[has], 1]) / 1e3
    best["survey_multiplicity"] = np.nan
    best.loc[has, "survey_multiplicity"] = mult[cands.row[bi[has]]]
    return m, best


def _bin_label(bins, i):
    lo, hi = bins[i], bins[i + 1]
    if not np.isfinite(lo):
        return f"G<{hi:g}"
    if not np.isfinite(hi):
        return f"G>={lo:g}"
    return f"{lo:g}<=G<{hi:g}"
