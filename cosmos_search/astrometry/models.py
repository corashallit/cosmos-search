"""Nested models for AL abscissae: single star, acceleration, Keplerian orbits, and the natural non-Keplerian terms.

Every model is linear in most of its parameters, so a fit is a weighted least-squares solve over a design matrix
whose columns are built from (t, psi, f_AL) and, for the non-linear parameters (period, eccentricity, periastron
time, lens timing, change-point time), a grid or a bounded optimiser over the profiled chi^2.

Column families (all multiply sin(psi) for the RA* part and cos(psi) for the Dec part of an offset):
    base5    position, parallax, proper motion (Gaia's 5-parameter model)
    accel    1/2 g t^2 (mas/yr^2) and jerk 1/6 j t^3: Gaia's 7- and 9-parameter acceleration models
    kepler   Thiele-Innes (B, G, A, F) x elliptic rectangular coordinates (X, Y) of one photocentre orbit
    vim      variability-induced motion: offset proportional to 1/F(t) - <1/F>, any unresolved pair with one
             variable component (photocentre at rho * F2/(F1+F2); linear in 1/F whichever component varies)
    blend    odd scan-angle harmonics cos/sin(3 psi), (5 psi): a fixed unresolved neighbour pulls the AL centroid by
             an odd function of psi (the first harmonic is the position offset itself), so even harmonics are not
             a blend signature
    neighbour
             a point neighbour's pull on the AL centroid, switched by the window as the scan angle turns
             (the dominant natural non-Keplerian signal in simulation)
    lens     astrometric microlensing centroid shift theta_E (tau e_phi + u0 e_perp) / (tau^2 + u0^2 + 2)
    step     position jump at t_k (non-gravitational: no force moves a mass instantaneously)
    kick     proper-motion change at t_k (an impulse: energetically implausible for a star, see docs)
"""

import numpy as np

MAS_PER_RAD = 180 / np.pi * 3.6e6
DAYS_PER_YEAR = 365.25


def _pair(x, psi):
    """Columns for an offset whose RA* and Dec components are both x: (x sin psi, x cos psi)."""
    return np.stack([x * np.sin(psi), x * np.cos(psi)], axis=-1)


def base5(t, psi, f_al):
    s, c = np.sin(psi), np.cos(psi)
    return np.stack([s, c, f_al, t * s, t * c], axis=-1)


def accel(t, psi, order=2):
    cols = [_pair(0.5 * t ** 2, psi)]
    if order >= 3:
        cols.append(_pair(t ** 3 / 6.0, psi))
    return np.concatenate(cols, axis=-1)


def kepler_xy(t, period, ecc, tperi):
    """Elliptic rectangular coordinates X = cos E - e, Y = sqrt(1-e^2) sin E (t, period, tperi in years).

    Broadcasts: t (..., n) with period/ecc/tperi (..., 1)."""
    m = 2 * np.pi * (t - tperi) / period
    e = np.asarray(ecc, float)
    ea = m + e * np.sin(m)
    for _ in range(30):  # Newton on Kepler's equation; converges for e <= 0.95 in < 30 steps
        d = (ea - e * np.sin(ea) - m) / (1 - e * np.cos(ea))
        ea = ea - d
        if np.nanmax(np.abs(d)) < 1e-10:
            break
    return np.cos(ea) - e, np.sqrt(1 - e ** 2) * np.sin(ea)


def kepler(t, psi, period, ecc, tperi):
    """Thiele-Innes columns (B, G, A, F): dra* = B X + G Y, ddec = A X + F Y (Gaia NSS convention)."""
    x, y = kepler_xy(t, period, ecc, tperi)
    s, c = np.sin(psi), np.cos(psi)
    return np.stack([x * s, y * s, x * c, y * c], axis=-1)


def vim(psi, flux):
    g = 1.0 / np.asarray(flux, float)
    return _pair((g - g.mean()) / g.mean(), psi)


def blend(psi, harmonics=(3, 5)):
    return np.concatenate([np.stack([np.cos(k * psi), np.sin(k * psi)], axis=-1) for k in harmonics], axis=-1)


NB_LSF_MAS = 40.0
NB_WINDOW_MAS = 1100.0
NB_EDGE_MAS = 50.0


def neighbour(t, psi, rho, theta, b, dmu=(0.0, 0.0), lsf=NB_LSF_MAS):
    """AL pull of an unresolved point neighbour at (rho, theta) with flux ratio b <= 1, moving at dmu (mas/yr).

    Deliberately not the simulator's pull (simulate._blend_pull, exact two-Gaussian peak with a hard window): the
    detector uses the small-pull approximation b d g / (1 + b g), g = exp(-d^2 / 4 s^2), with the LSF width s a free
    parameter (the true image model is unknown) and a logistic window edge, and its amplitude is a free linear
    parameter, so a mismatch between the true and assumed image model is absorbed rather than rewarded."""
    ox = rho * np.sin(theta) + dmu[0] * t
    oy = rho * np.cos(theta) + dmu[1] * t
    d = ox * np.sin(psi) + oy * np.cos(psi)
    dac = ox * np.cos(psi) - oy * np.sin(psi)
    g = np.exp(-d ** 2 / (4 * lsf ** 2))
    win = 1 / (1 + np.exp((np.abs(dac) - NB_WINDOW_MAS) / NB_EDGE_MAS)) / (
        1 + np.exp((np.abs(d) - NB_WINDOW_MAS) / NB_EDGE_MAS))
    return (b * d * g / (1 + b * g) * win)[..., None]


def lens(t, psi, t0, te, u0):
    tau = (t - t0) / te
    d = tau ** 2 + u0 ** 2 + 2.0
    return np.concatenate([_pair(tau / d, psi), _pair(1.0 / d, psi)], axis=-1)


def step(t, psi, tk):
    return _pair((t >= tk).astype(float), psi)


def kick(t, psi, tk):
    return _pair(np.clip(t - tk, 0, None), psi)


# ---------------------------------------------------------------- solving


def wls(a, y, sigma):
    """Weighted least squares. a (..., n, p), y (n,), sigma (n,) -> (params (..., p), chi2 (...)).

    Batched over leading axes of `a` so that a whole grid of non-linear parameters is solved at once."""
    w = 1.0 / sigma
    aw = a * w[:, None]
    yw = y * w
    ata = np.einsum("...np,...nq->...pq", aw, aw)
    aty = np.einsum("...np,n->...p", aw, yw)
    p = aw.shape[-1]
    ata = ata + 1e-12 * np.trace(ata, axis1=-2, axis2=-1)[..., None, None] * np.eye(p)
    x = np.linalg.solve(ata, aty[..., None])[..., 0]
    r = yw - np.einsum("...np,...p->...n", aw, x)
    return x, np.einsum("...n,...n->...", r, r)


def covariance(a, sigma):
    aw = a / sigma[:, None]
    return np.linalg.pinv(aw.T @ aw)


def bic(chi2, k, n):
    return chi2 + k * np.log(n)
