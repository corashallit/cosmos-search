"""Synthetic skies with known truth, for testing the matcher the way we test detectors: by injection.

A field of Gaia sources with realistic-ish motions, a survey that detected a known subset of them at
another epoch with known errors, and an unrelated background at a known density. The matcher must recover
the error model, the counterpart fraction, calibrated posteriors and the false-match rate.
"""

import numpy as np
import pandas as pd

from .astrometry import parallax_factors
from .geometry import MAS_PER_ARCSEC, offset_position, random_annulus
from .surveys import Survey


def sim_survey(name="sim", *, epoch=2000.0, parallax_mode="none", per_row_epoch=False, groups=False,
               floor_mas=50.0, search_radius_mas=3000.0, shift_annulus_arcsec=(30.0, 60.0)):
    return Survey(
        name, ra="ra", dec="dec", id="sid", error=("axes", "sx", "sy", 1.0), epoch=epoch,
        epoch_col="jyear" if per_row_epoch else None, epoch_span=0.5 if per_row_epoch else 0.0,
        parallax_mode=parallax_mode, floor_mas=floor_mas, search_radius_mas=search_radius_mas,
        shift_annulus_arcsec=shift_annulus_arcsec, group_by=("grp",) if groups else (),
        fwhm_arcsec=0.1,  # the simulated survey resolves everything: no blends
    )


def sim_field(n=3000, *, seed=0, ra0=150.0, dec0=30.0, radius_deg=0.5, survey=None, p_counterpart=0.7,
              bg_per_arcmin2=0.5, sigma_cat_mas=(80.0, 300.0), true_scale=1.3, true_floor_mas=60.0,
              frac_high_pm=0.02, frac_unsolved=0.1, n_groups=1, tail_frac=0.0, tail_mas=0.0):
    """Return (gaia DataFrame, survey DataFrame, truth array: survey sid of each Gaia source's counterpart
    or -1 per group). truth has shape (n, n_groups)."""
    rng = np.random.default_rng(seed)
    survey = survey or sim_survey()
    # Gaia sources uniform in a cap around (ra0, dec0)
    r = radius_deg * 3.6e6 * np.sqrt(rng.uniform(0, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    ra, dec = offset_position(ra0, dec0, r * np.sin(phi), r * np.cos(phi))
    g = rng.uniform(12, 20.8, n)
    plx = np.abs(rng.lognormal(np.log(0.8), 0.8, n))
    pmra = rng.normal(-3, 6, n)
    pmdec = rng.normal(-4, 6, n)
    hi = rng.uniform(size=n) < frac_high_pm
    pmra[hi] = rng.normal(0, 400, hi.sum())
    pmdec[hi] = rng.normal(0, 400, hi.sum())
    plx[hi] += 20.0
    solved = rng.uniform(size=n) > frac_unsolved
    solved[hi] = True
    ref = np.full(n, 2016.0)
    # Gaia's measurement errors (small); reported = true + noise
    s_pos = 0.02 + 0.3 * 10 ** (0.4 * (g - 20))  # mas
    s_pm = 2 * s_pos
    gaia = pd.DataFrame({
        "source_id": np.arange(n) + 10**15, "ra": ra, "dec": dec, "ref_epoch": ref,
        "parallax": np.where(solved, plx + rng.normal(0, s_pos), np.nan),
        "pmra": np.where(solved, pmra + rng.normal(0, s_pm), np.nan),
        "pmdec": np.where(solved, pmdec + rng.normal(0, s_pm), np.nan),
        "ra_error": s_pos, "dec_error": s_pos, "parallax_error": np.where(solved, s_pos, np.nan),
        "pmra_error": np.where(solved, s_pm, np.nan), "pmdec_error": np.where(solved, s_pm, np.nan),
        "phot_g_mean_mag": g,
    })

    rows = []
    truth = np.full((n, n_groups), -1, dtype=np.int64)
    sid = 0
    for grp in range(n_groups):
        has = rng.uniform(size=n) < p_counterpart
        k = has.sum()
        if per_row := bool(survey.epoch_col):
            t = survey.epoch + rng.uniform(-survey.epoch_span, survey.epoch_span, k)
        else:
            t = np.full(k, survey.epoch)
        dt = t - 2016.0
        xi = pmra[has] * dt
        eta = pmdec[has] * dt
        if survey.parallax_mode == "exact":
            fx, fy = parallax_factors(ra[has], dec[has], t)
            xi, eta = xi + plx[has] * fx, eta + plx[has] * fy
        sx = rng.uniform(*sigma_cat_mas, k)
        sy = sx * rng.uniform(0.8, 1.2, k)
        tail = rng.uniform(size=k) < tail_frac
        ex = np.sqrt((true_scale * sx) ** 2 + true_floor_mas**2 + tail * tail_mas**2)
        ey = np.sqrt((true_scale * sy) ** 2 + true_floor_mas**2 + tail * tail_mas**2)
        sra, sdec = offset_position(ra[has], dec[has], xi + rng.normal(0, ex), eta + rng.normal(0, ey))
        ids = np.arange(sid, sid + k)
        sid += k
        truth[np.nonzero(has)[0], grp] = ids
        rows.append(pd.DataFrame({"sid": ids, "ra": sra, "dec": sdec, "sx": sx, "sy": sy, "grp": grp,
                                  "jyear": t if per_row else survey.epoch}))
        # Unrelated background over a slightly larger cap, so edge targets see the same density.
        area = np.pi * ((radius_deg + 0.1) * 60) ** 2
        nb = rng.poisson(bg_per_arcmin2 * area)
        rb = (radius_deg + 0.1) * 3.6e6 * np.sqrt(rng.uniform(0, 1, nb))
        pb = rng.uniform(0, 2 * np.pi, nb)
        bra, bdec = offset_position(ra0, dec0, rb * np.sin(pb), rb * np.cos(pb))
        bsx = rng.uniform(*sigma_cat_mas, nb)
        rows.append(pd.DataFrame({"sid": np.arange(sid, sid + nb), "ra": bra, "dec": bdec, "sx": bsx,
                                  "sy": bsx, "grp": grp,
                                  "jyear": survey.epoch + rng.uniform(-survey.epoch_span, survey.epoch_span, nb)}))
        sid += nb
    return gaia, pd.concat(rows, ignore_index=True), truth
