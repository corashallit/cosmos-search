"""The energy budget: light an occulter absorbs must come out again, as heat, somewhere.

A star dimmed by a time-averaged fraction <delta> along our line of sight has absorbers on that line. How much of the
star's whole output they intercept depends on how they are spread around the star, which one sightline cannot see:

- shell: absorbers spread evenly over the sky as seen from the star (a swarm, an envelope, an R CrB dust shell).
  The fraction of all directions covered is then the fraction of time ours is, so f = <delta>.
- thin: everything on one orbit through our sightline, no wider than the star. A body of projected area A at
  distance r dims the star by A / (pi R*^2) for a fraction R* / (pi r) of its orbit and intercepts A / (4 pi r^2) of
  L*, so f = (pi / 4) <delta> R* / r. This is the least absorbing arrangement that still produces the dimming.

The absorbed power comes out at the absorber's temperature T, and T fixes r: a flat plate facing the star has
T = T* (R* / r)^(1/2), so R* / r = (T / T*)^2 and f_thin(T) = (pi / 4) <delta> (T / T*)^2. Small grains run hotter
than a blackbody at the same r; allowing a factor 2 in T halves this again, so the thin floor used here is
(pi / 8) <delta> (T / T*)^2. Both floors assume the absorber is dark (albedo 0); a reflective one absorbs (1 - albedo)
of what it blocks.

So the test is a function of T. For each temperature on a grid this module fits the star's K and WISE photometry
with a photosphere plus a blackbody (or a lambda^-1 modified blackbody) carrying a fraction f of L*, and reports the
largest f the photometry allows (f_ul) and how strongly it rejects the f each geometry needs. Upper limits (WISE
'U' quality) enter as a measurement of zero with sigma = limit / 2; contaminated or blended detections enter as
upper bounds only (one-sided), because a blend can add flux but cannot remove it. The excess is computed relative
to WISE W1, which AllWISE measured at the same epochs as W2-W4, so the star's own variability between 2MASS (1998-2001)
and WISE (2010) only enters through K, whose error carries it.

Band shapes are trapezoids that roughly follow the relative response curves; the star's spectrum is a blackbody at
T* from BP-RP. Both are crude, but the zero points cancel (an excess is a ratio in the same band), and the geometric
factor between the shell and thin cases is already orders of magnitude.
"""

import numpy as np

BANDS = ("K", "W1", "W2", "W3", "W4")
# Trapezoid (rise from a to b, flat to c, fall to d), microns: 2MASS Ks and WISE W1-W4, approximately.
BAND_SHAPES = {"K": (1.95, 2.00, 2.30, 2.36), "W1": (2.75, 3.00, 3.70, 3.90), "W2": (4.00, 4.20, 5.10, 5.35),
               "W3": (7.5, 9.0, 13.5, 16.5), "W4": (19.5, 21.0, 24.0, 27.5)}
HC_K_UM = 14387.77          # hc / k in micron kelvin
T_GRID = np.geomspace(50.0, 1800.0, 37)
T_SUB = 1500.0              # dust sublimates around 1500-2000 K; the summaries stop here
BETAS = (0.0, 1.0)          # emissivity index: 0 = blackbody (large grains, bodies), 1 = small grains beyond ~1 micron
F_GRID = np.concatenate([[0.0], np.geomspace(1e-7, 3.0, 161)])

# Dwarf BP-RP -> Teff (Pecaut & Mamajek 2013, the 2022 table's Gaia colours), coarse; reddening makes T* low, which
# makes the predicted excess smaller (conservative for a deficit).
_TEFF_TABLE = np.array([
    (-0.30, 20000), (-0.12, 12000), (-0.04, 10500), (0.00, 9700), (0.21, 8080), (0.37, 7220), (0.56, 6510),
    (0.72, 5920), (0.82, 5770), (0.85, 5660), (0.98, 5280), (1.06, 5040), (1.42, 4440), (1.67, 4050), (1.84, 3850),
    (2.03, 3660), (2.17, 3560), (2.43, 3430), (2.85, 3210), (3.24, 3060), (4.09, 2810), (4.50, 2680), (5.0, 2500)])


def teff_from_bp_rp(bp_rp):
    c = np.asarray(bp_rp, float)
    return np.interp(c, _TEFF_TABLE[:, 0], _TEFF_TABLE[:, 1])


def _lam_grid(band, n=200):
    a, b, c, d = BAND_SHAPES[band]
    lam = np.linspace(a, d, n)
    r = np.interp(lam, [a, b, c, d], [0.0, 1.0, 1.0, 0.0])
    return lam, r


def _planck_lam(lam_um, T):
    """B_lambda up to a constant: lam^-5 / (exp(hc / lam k T) - 1)."""
    x = HC_K_UM / (lam_um * T)
    with np.errstate(over="ignore"):
        return lam_um ** -5.0 / np.expm1(np.minimum(x, 700.0))


_LAM_ALL = np.geomspace(0.05, 5000.0, 4000)


def _norm(T, beta):
    """Integral of emissivity * B_lambda over all wavelengths (same constants as _planck_lam)."""
    T = np.atleast_1d(T)
    eps = np.minimum(1.0, (1.0 / _LAM_ALL)) ** beta if beta else 1.0
    y = eps * _planck_lam(_LAM_ALL[None, :], T[:, None])
    return np.trapezoid(y, _LAM_ALL, axis=1)


def band_fraction(band, T, beta=0.0):
    """Response-weighted (photon-counting) in-band flux of an emitter at T, per unit bolometric flux, up to a
    band constant that cancels in ratios."""
    lam, r = _lam_grid(band)
    T = np.atleast_1d(np.asarray(T, float))
    eps = np.minimum(1.0, 1.0 / lam) ** beta if beta else 1.0
    num = np.trapezoid(r * lam * eps * _planck_lam(lam[None, :], T[:, None]), lam, axis=1)
    return num / _norm(T, beta)


def excess_per_f(teff, T, beta=0.0):
    """E_j / f for each band in BANDS: in-band flux of an emitter carrying a fraction f of L*, over the star's own
    in-band flux, per unit f. Shape (len(teff), len(T), len(BANDS))."""
    teff = np.atleast_1d(np.asarray(teff, float))
    T = np.atleast_1d(np.asarray(T, float))
    out = np.empty((len(teff), len(T), len(BANDS)))
    for j, b in enumerate(BANDS):
        dust = band_fraction(b, T, beta)               # (nT,)
        star = band_fraction(b, teff, 0.0)             # (n,)
        out[:, :, j] = dust[None, :] / star[:, None]
    return out


# --- geometry ---------------------------------------------------------------


def f_shell(delta):
    return np.asarray(delta, float)


def f_thin(delta, T, teff):
    """Least absorbing arrangement (module docstring): (pi / 8) <delta> (T / T*)^2."""
    return np.pi / 8 * np.asarray(delta, float)[..., None] * (np.asarray(T, float)[None, :]
                                                              / np.asarray(teff, float)[..., None]) ** 2


def t_floor_kinematic(duration_max_days, depth, r_star, m_star, teff):
    """Coldest a bound occulter can be, given that the whole event lasted at most duration_max_days.

    The occulter covers depth * pi R*^2 at mid-event, so its extent is at least sqrt(depth) R* and it crosses at
    least that distance within the bracket: v >= sqrt(depth) R* / D. A bound orbit has v^2 <= 2 G M* / r, so
    r <= 2 G M* D^2 / (depth R*^2), and a fast-rotating blackbody there has T = T* (R* / 2r)^(1/2). Dust runs
    hotter, so this is a floor. Returns kelvin (nan without stellar parameters)."""
    from cosmos_search.vetting.physics import DAY, G_MSUN, R_SUN

    if not (r_star and m_star and depth and depth > 0 and duration_max_days and duration_max_days > 0):
        return float("nan")
    d = duration_max_days * DAY
    rs = r_star * R_SUN
    r_max = 2 * G_MSUN * m_star * d ** 2 / (depth * rs ** 2)
    return float(teff * np.sqrt(rs / (2 * r_max)))


# --- the fit ----------------------------------------------------------------


def _chi2_profile(d, s, w_meas, m, bound):
    """min over a >= 0 of chi^2 for data d, sigma s, model a * m; `bound` bands count only where a * m > d.

    Shapes (..., n_bands). Two passes of the active set suffice: bounds only ever pull a down."""
    w = np.where(w_meas & ~bound, 1.0 / s ** 2, 0.0)
    a = _a_hat(d, w, m)
    for _ in range(2):
        active = bound & (a[..., None] * m > d)
        w = np.where((w_meas & ~bound) | active, 1.0 / s ** 2, 0.0)
        a = _a_hat(d, w, m)
    chi2 = np.sum(w * (d - a[..., None] * m) ** 2, axis=-1)
    return chi2, a


def _a_hat(d, w, m):
    num = np.sum(w * d * m, axis=-1)
    den = np.sum(w * m * m, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(np.where(den > 0, num / den, np.nan), 0, None)


def fit(phot, delta=None, z_ul=2.0, betas=BETAS, t_grid=T_GRID, f_grid=F_GRID):
    """Energy-budget fit for each row of `phot` (see photometry.design for the columns: d_<band>, s_<band>,
    use_<band>, bound_<band>, P_<band>, teff).

    Returns a dict of arrays:
      chi2_0          photosphere only
      z_excess        sqrt(chi2_0 - min over T, beta, f >= 0 of chi2): how strongly an excess is preferred
      t_best, f_best  the temperature and fraction at that minimum
      f_ul            (n, nT): largest f with chi2 - chi2_min(T) <= z_ul^2, the least constraining beta
      and, with delta (time-averaged dimming, lower bound): z_shell, z_thin (n, nT), how strongly the photometry
      rejects the f each geometry needs at each T (0 where it does not), the smaller over beta.
    """
    d = np.stack([phot[f"d_{b}"] for b in BANDS], -1).astype(float)
    s = np.stack([phot[f"s_{b}"] for b in BANDS], -1).astype(float)
    use = np.stack([phot[f"use_{b}"] for b in BANDS], -1).astype(bool) & np.isfinite(d) & (s > 0)
    bound = np.stack([phot[f"bound_{b}"] for b in BANDS], -1).astype(bool) & use
    P = np.stack([phot[f"P_{b}"] for b in BANDS], -1).astype(float)
    use &= np.isfinite(P)
    d, s, P = np.where(use, d, 0.0), np.where(use, s, 1.0), np.where(use, P, 0.0)
    teff = np.asarray(phot["teff"], float)
    n, nT, nF = len(d), len(t_grid), len(f_grid)

    chi2_0, _ = _chi2_profile(d, s, use, P, bound)
    chi2_min_T = np.full((n, nT), np.inf)
    f_hat_T = np.zeros((n, nT))
    f_ul = np.zeros((n, nT))
    z_shell = np.full((n, nT), np.inf) if delta is not None else None
    z_thin = np.full((n, nT), np.inf) if delta is not None else None
    if delta is not None:
        delta = np.asarray(delta, float)
        need_shell = np.broadcast_to(f_shell(delta)[:, None], (n, nT))
        need_thin = f_thin(delta, t_grid, teff)
    for beta in betas:
        e = excess_per_f(teff, t_grid, beta)            # (n, nT, 5)
        # chi2 on the f grid: (n, nT, nF)
        m = P[:, None, None, :] * (1 + f_grid[None, None, :, None] * e[:, :, None, :])
        c2, _ = _chi2_profile(d[:, None, None, :], s[:, None, None, :], use[:, None, None, :], m,
                              bound[:, None, None, :])
        cmin = c2.min(axis=2)
        fh = f_grid[c2.argmin(axis=2)]
        # f_ul: largest grid f whose chi2 stays within z_ul^2 of this beta's minimum, above f_hat.
        ok = (c2 - cmin[..., None] <= z_ul ** 2) & (f_grid[None, None, :] >= fh[..., None])
        idx = np.where(ok.any(axis=2), nF - 1 - np.argmax(ok[..., ::-1], axis=2), 0)
        ful = f_grid[idx]
        better = cmin < chi2_min_T
        chi2_min_T = np.where(better, cmin, chi2_min_T)
        f_hat_T = np.where(better, fh, f_hat_T)
        f_ul = np.maximum(f_ul, ful)
        if delta is not None:
            for need, z in ((need_shell, z_shell), (need_thin, z_thin)):
                mm = P[:, None, :] * (1 + need[..., None] * e)
                cn, _ = _chi2_profile(d[:, None, :], s[:, None, :], use[:, None, :], mm, bound[:, None, :])
                zz = np.sqrt(np.clip(cn - cmin, 0, None)) * (need > fh)
                np.minimum(z, zz, out=z)
    k = np.argmin(chi2_min_T, axis=1)
    rows = np.arange(n)
    out = {"chi2_0": chi2_0, "z_excess": np.sqrt(np.clip(chi2_0 - chi2_min_T[rows, k], 0, None)),
           "t_best": t_grid[k], "f_best": f_hat_T[rows, k], "f_ul": f_ul, "chi2_min_T": chi2_min_T,
           "n_bands": use.sum(axis=1)}
    if delta is not None:
        out["z_shell"], out["z_thin"] = z_shell, z_thin
    return out


def excluded_above(z, t_grid=T_GRID, z_thr=3.0, t_max=T_SUB):
    """Lowest grid temperature T such that every grid temperature from T up to t_max is rejected at z >= z_thr
    (nan if t_max itself is not). "The absorbed light cannot be coming out hotter than this.\""""
    z = np.atleast_2d(z)
    sel = t_grid <= t_max
    zz = z[:, sel] >= z_thr
    tg = t_grid[sel]
    out = np.full(len(z), np.nan)
    for i, row in enumerate(zz):
        if not row[-1]:
            continue
        j = len(row) - 1
        while j > 0 and row[j - 1]:
            j -= 1
        out[i] = tg[j]
    return out


def f_ul_at(f_ul, T, t_grid=T_GRID):
    """f_ul interpolated (in log) at temperature T."""
    lg = np.log10(np.clip(f_ul, 1e-12, None))
    return 10 ** np.array([np.interp(np.log10(T), np.log10(t_grid), row) for row in np.atleast_2d(lg)])
