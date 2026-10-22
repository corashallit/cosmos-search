"""Where on the sky does a dip happen? Astrometric localisation of a dimming event, for the dimmer arms.

Two unresolved components, the varying one at offset rho from the steady one: the photocentre sits at rho F_var/F, so
as the total flux F changes the photocentre moves along rho, linearly in 1/F, with

    dx / d(1/F) * <1/F>  =  c,   |c| = rho x (flux share of the steady component).

So the fitted coefficient c (mas per unit fractional change of 1/F, the VIM block) is a lower bound on how far the
dimming source sits from the rest of the light: |rho| >= |c|. For one star dimmed by something in front of it, the
light centroid can move by at most about the star's angular radius theta_*, so

    |c| consistent with 0      the dip is on the target, to within the upper limit (a vetting fact for dimmers)
    |c| >> theta_*             the dip belongs to an unresolved neighbour (blend), separation >= |c|
    0 < |c| <~ theta_*         an occulter large enough to unbalance the stellar disk (only measurable for
                               nearby giants, where theta_* reaches 0.1-10 mas): the rare, interesting case

The dip epochs carry all the information: c is measured from the transits whose flux departs from the median.
"""

import numpy as np

from . import models as M
from .fit import Source


def localise(t, psi, f_al, w, sigma, flux, theta_star_mas=0.0, base_order=1):
    """Fit 5p (+acceleration) + VIM and return the dip offset c (mas, RA*/Dec), its covariance, and verdicts."""
    src = Source(t, psi, f_al, w, sigma, flux)
    cols = src.base(base_order)
    vi = [j for j in range(cols.shape[1]) if np.array_equal(cols[:, j], src.fixed[0][:, 0])
          or np.array_equal(cols[:, j], src.fixed[0][:, 1])]
    x, chi2 = M.wls(cols, src.y, src.s)
    cov = M.covariance(cols, src.s)[np.ix_(vi, vi)]
    c = x[vi]
    chi2_c = float(c @ np.linalg.solve(cov, c))  # 2-dof significance of any offset
    # Gaia measures along scan only, and a dip seen in one visit has one scan angle, so c is often constrained in one
    # direction only. Report the bound along the best-constrained axis (1-D 95%) and the 2-D one.
    evals, evecs = np.linalg.eigh(cov)
    smin, smax = float(np.sqrt(evals[0])), float(np.sqrt(evals[-1]))
    upper_axis = float(abs(c @ evecs[:, 0]) + 1.96 * smin)
    upper = float(np.hypot(*c) + 2.45 * smax)
    offset = chi2_c > 13.8  # 2-dof p < 1e-3
    lower = float(max(np.hypot(*c) - 2.45 * smax, abs(c @ evecs[:, 0]) - 1.96 * smin))
    if not offset:
        verdict = "on_target"
    elif lower > max(theta_star_mas, 0.0) * 1.5:
        verdict = "neighbour"
    else:
        verdict = "disk_scale"
    return {"c_ra": float(c[0]), "c_dec": float(c[1]), "c_abs": float(np.hypot(*c)), "sigma_min": smin,
            "sigma_max": smax, "upper95_axis": upper_axis, "upper95": upper, "lower95": lower, "chi2_offset": chi2_c,
            "verdict": verdict, "chi2": float(chi2)}
