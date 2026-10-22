"""The cascade: best natural model by BIC, then what the residual still holds.

Natural family (each nested on the 5-parameter base): 5p, 7p, 9p, K1 (on 5p or 7p, whichever BIC prefers), K2 on K1,
lens. The winner then gets the nuisance blocks (VIM if flux is given, odd scan-angle harmonics for blends) if they
lower BIC. Only after the natural family has had every chance does the cascade try the non-gravitational change
points (step, kick) on top of it.

Statistics per source (all from the same fits):
    best            name of the best natural model
    dbic_cp         BIC(best natural) - BIC(best natural + change point), > 0 favours the change point
    gof_z           Wilson-Hilferty z of the best natural model's chi2 (residual excess, ~N(0,1) when the model and
                    the noise are right)
    ruwe            sqrt(chi2_5p / (n - 5)): what the DR4 catalogue's RUWE would show (prefilter input)
    leverage        share of K1's chi2 gain carried by two transits
    k1_fA, k1_verdict
                    astrometric mass function of the K1 orbit and whether it is physically possible (physics.py);
                    impossible orbits are removed from the natural family before the best model is chosen
    dbic_implausible
                    BIC(best possible natural model) - BIC(best impossible orbit): > 0 when the data demand an orbit
                    gravity cannot provide (needs > 100 Msun, or periastron inside the star)
    acc_min_mass    smallest companion mass that produces the 7p acceleration without visible curvature
    fov_pair_z      do the two fields of view agree on the residual? (large: instrument, not sky)
"""

import numpy as np

from . import models as M, physics
from .fit import Source, vim_index

LENS_PHOT_DCHI2 = 25.0
CLIP_Z = 5.0
MAX_CLIP_FRAC = 0.02


def covariance_diag(fit, src):
    return np.diag(M.covariance(fit.columns, src.s))


def fov_pair_z(t, fov, resid, sigma, max_gap_days=0.08):
    """Residual consistency of the two fields of view: transits 106.5 min apart (preceding then following FoV) see
    the same sky through different optics and calibrations, so their residuals should agree within noise. A per-FoV
    calibration jump does not. Wilson-Hilferty z of sum (r1 - r2)^2 / (s1^2 + s2^2) over such pairs."""
    if fov is None:
        return np.nan
    fov = np.asarray(fov)
    i = np.where((np.diff(t) * 365.25 < max_gap_days) & (fov[1:] != fov[:-1]))[0]
    if len(i) < 5:
        return np.nan
    chi2 = np.sum((resid[i] - resid[i + 1]) ** 2 / (sigma[i] ** 2 + sigma[i + 1] ** 2))
    return float(wilson_hilferty(chi2, len(i)))


def wilson_hilferty(chi2, dof):
    dof = max(dof, 1)
    return ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / np.sqrt(2 / (9 * dof))


def cascade(data, full=True):
    """Run the cascade; if the fitted variability-induced motion is large enough that the flux errors themselves move
    the photocentre by more than ~1/3 sigma, inflate sigma by that propagated error and run it again."""
    out = _cascade(data, full)
    if out.get("vim_noise_ratio", 0) > 0.33:
        sig = np.sqrt(np.asarray(data["sigma"]) ** 2 + out.pop("_vim_var"))
        data = {**data, "sigma": sig}
        out = _cascade(data, full)
        out["vim_inflated"] = True
    # single-transit outliers (> CLIP_Z sigma from the best natural model; AGIS would have down-weighted them): drop at
    # most MAX_CLIP_FRAC of the transits and run again. A jump or any sustained signal is never one transit.
    z = np.abs(out.pop("_resid_z"))
    bad = np.where(z > CLIP_Z)[0]
    if len(bad):
        bad = bad[np.argsort(-z[bad])][:max(1, int(MAX_CLIP_FRAC * len(z)))]
        keep = np.ones(len(z), bool)
        keep[bad] = False
        data = {k: (np.asarray(v)[keep] if isinstance(v, np.ndarray) and np.ndim(v) == 1 and len(v) == len(z) else v)
                for k, v in data.items()}
        out = _cascade(data, full)
        out.pop("_resid_z")
        out["n_clipped"] = int(len(bad))
    out.pop("_vim_var", None)
    return out


def _cascade(data, full=True):
    src = Source(data["t"], data["psi"], data["f_al"], data["w"], data["sigma"], data.get("flux"),
                 data.get("flux_err"))
    n = src.n
    fits = {}
    implausible = {}
    fits["5p"] = src.five()
    fits["7p"] = src.accel(2)
    fits["9p"] = src.accel(3)
    out = {"n": n, "chi2_5p": fits["5p"].chi2, "ruwe": float(np.sqrt(fits["5p"].chi2 / (n - 5)))}
    ms = data.get("multi_source")
    out["frac_multi"] = float(np.mean(ms)) if ms is not None and len(ms) else np.nan
    if full:
        # an orbit on the bare star, and one on top of an acceleration (a wide third body); acceleration and a long
        # period trade off, so both are searched and BIC decides
        k1 = src.kepler(base=src.base(1))
        if fits["7p"].bic(n) < fits["5p"].bic(n):
            k1a = src.kepler(base=src.base(2))
            k1 = k1a if k1a.bic(n) < k1.bic(n) else k1
        fits["K1"] = k1
        out.update(k1_period=k1.params["period_yr"], k1_ecc=k1.params["ecc"], k1_a0=k1.params["a0_mas"],
                   k1_leverage=k1.params["leverage"], k1_plx=float(k1.x[2]))
        if k1.bic(n) < min(fits["5p"].bic(n), fits["7p"].bic(n), fits["9p"].bic(n)):
            fits["K2"] = src.kepler2(k1)
        fits["lens"] = src.lens()
        out["lens_phot_dchi2"] = fits["lens"].params["phot_dchi2"]
        if out["lens_phot_dchi2"] > LENS_PHOT_DCHI2:
            implausible["lens"] = fits.pop("lens")
        fits["nb"] = src.neighbour()
        out.update(nb_rho=fits["nb"].params["rho"], nb_b=fits["nb"].params["b"])
    # physical plausibility: an orbit that needs > M_CAP or puts periastron inside the star is not a natural model
    r1 = data.get("r1", physics.main_sequence_radius(data.get("m1", 1.0)))
    for name in ("K1", "K2"):
        if name not in fits:
            continue
        fk = fits[name]
        orbits = [fk.params] if name == "K1" else [fk.params["inner"], fk.params["outer"]]
        plx_err = float(np.sqrt(covariance_diag(fk, src)[2]))
        verdicts = []
        for o in orbits:
            fa, v = physics.kepler_verdict(o["period_yr"], o["ecc"], o["a0_mas"], float(fk.x[2]), plx_err, r1)
            verdicts.append((fa, v))
        fas = [fa for fa, _ in verdicts if np.isfinite(fa)]
        out[f"{name.lower()}_fA"] = float(max(fas)) if fas else np.nan
        bad = [v for _, v in verdicts if v in ("rp_inside_star", "over_mass_cap")]
        out[f"{name.lower()}_verdict"] = bad[0] if bad else ("unknown" if any(v == "unknown" for _, v in verdicts)
                                                              else "ok")
        if bad:
            implausible[name] = fits.pop(name)
    g7 = fits["7p"].x[5:7]
    out["acc_g"] = float(np.hypot(*g7))
    out["acc_min_mass"] = float(physics.acceleration_min_mass(out["acc_g"], max(fits["7p"].x[2], 1e-3), src.span))
    bics = {k: v.bic(n) for k, v in fits.items()}
    best = min(bics, key=bics.get)
    bf = fits[best]
    out.update({f"bic_{k}": v for k, v in bics.items()})
    # how strongly the data prefer a physically impossible orbit over every possible natural model
    out["dbic_implausible"] = float(max([bf.bic(n) - f.bic(n) for f in implausible.values()] + [-np.inf]))
    # nuisance blocks on top of the winner (non-linear parameters held)
    nuis = src.nuisance()
    if nuis:
        cand = src.solve(np.concatenate([bf.columns] + nuis, axis=1), best + "+N", k_extra=bf.k - bf.columns.shape[1])
        if cand.bic(n) < bf.bic(n):
            bf = cand
    vi = vim_index(src, bf.columns)
    if vi and src.flux_err is not None:
        c = bf.x[vi]
        u_err = src.flux_err  # fractional flux error ~ fractional error of 1/F
        g = 1.0 / src.flux
        disp = (c[0] * np.sin(src.psi) + c[1] * np.cos(src.psi)) / g.mean() * g * u_err
        out["_vim_var"] = disp ** 2
        out["vim_noise_ratio"] = float(np.sqrt(np.mean(disp ** 2) / np.mean(src.s ** 2)))
        out["vim_amp"] = float(np.hypot(*c))
    out["_resid_z"] = (src.y - bf.columns @ bf.x) / src.s
    out["fov_pair_z"] = fov_pair_z(src.t, data.get("fov"), src.y - bf.columns @ bf.x, src.s)
    out["best"] = bf.name
    out["k_best"] = bf.k
    out["chi2_best"] = bf.chi2
    out["gof_z"] = float(wilson_hilferty(bf.chi2, n - bf.k))
    out["plx_best"] = float(bf.x[2])
    # non-gravitational change points on top of the best natural model
    # ... and instead of it: a change point on the bare 5-parameter star (the orbit may have absorbed the jump)
    for kind in ("step", "kick"):
        cp = src.changepoint(kind, base=bf.columns)
        cp.k = bf.k + 3
        out[f"dbic_{kind}"] = float(bf.bic(n) - cp.bic(n))
        out[f"t_{kind}"] = cp.params["t_k"]
        alt = src.changepoint(kind, base=src.base(1) if not nuis else np.concatenate([src.base(1)] + nuis, axis=1))
        alt.k = alt.columns.shape[1] + 1
        out[f"dbic_{kind}_alt"] = float(bf.bic(n) - alt.bic(n))
    out["dbic_cp"] = max(out["dbic_step"], out["dbic_kick"], out["dbic_step_alt"], out["dbic_kick_alt"])
    return out
