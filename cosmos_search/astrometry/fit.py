"""Fitting the nested family to one source's per-transit AL abscissae.

`Source` holds the data (t in years from J2017.5, psi in rad, f_AL, w and sigma in mas, optional flux) and fits:

    five        5-parameter single star
    accel7/9    + acceleration / + jerk
    kepler      + one photocentre orbit (circular frequency grid, then eccentric refinement of the best peaks)
    kepler2     + a second orbit on top of the first (hierarchical triple, or binary + wide companion)
    lens        + astrometric microlensing by an unseen lens
    vim, blend  nuisance blocks that can be added to any model
    step, kick  non-gravitational change points

Every non-linear search profiles the linear parameters out with the Frisch-Waugh projection: candidate columns
are projected orthogonally (in the weighted metric) to the columns already in the model, so each grid point costs
a q x q solve with q = 2..4.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from . import models as M

# Orbit search range. A period much longer than the span is an acceleration; below ~0.5 d the 106.5-min FoV pair
# spacing and the 6-h spin dominate the window and every peak aliases.
P_MIN_DAYS = 0.5
P_MAX_YEARS = 40.0
OVERSAMPLE = 6
ECC_GRID = (0.0, 0.3, 0.55, 0.75, 0.9)
N_TPERI = 12
N_PEAKS = 8
N_REFINE = 2
E_MAX = 0.97


@dataclass
class Fit:
    name: str
    k: int  # number of free parameters (linear + non-linear)
    chi2: float
    params: dict = field(default_factory=dict)
    columns: np.ndarray = None  # design matrix of the linear part
    x: np.ndarray = None  # linear solution

    def bic(self, n):
        return M.bic(self.chi2, self.k, n)


class Source:
    def __init__(self, t, psi, f_al, w, sigma, flux=None, flux_err=None):
        self.t = np.asarray(t, float)
        self.psi = np.asarray(psi, float)
        self.f = np.asarray(f_al, float)
        self.y = np.asarray(w, float)
        self.s = np.asarray(sigma, float)
        self.flux = None if flux is None else np.asarray(flux, float)
        # VIM rides in every model when per-transit flux is known (DR4 always has it): 2 columns, and without it a
        # variable unresolved pair looks like anything from an orbit to a jump
        self.fixed = [] if self.flux is None or np.ptp(self.flux) == 0 else [M.vim(self.psi, self.flux)]
        self.n = len(self.y)
        self.flux_err = None if flux_err is None else np.asarray(flux_err, float) * np.ones(self.n)
        self.span = self.t.max() - self.t.min()
        self.sw = 1.0 / self.s

    # ---------------------------------------------------------------- linear core

    def solve(self, a, name, k_extra=0, params=None):
        x, chi2 = M.wls(a, self.y, self.s)
        return Fit(name, a.shape[1] + k_extra, float(chi2), params or {}, a, x)

    def _projector(self, a):
        """Weighted residual-maker for columns a: returns a function mapping (..., n, q) -> projected (..., n, q)."""
        aw = a * self.sw[:, None]
        q, _ = np.linalg.qr(aw)

        def proj(c):
            if c.ndim == 2:
                cw = c * self.sw[:, None]
                return cw - q @ (q.T @ cw)
            cw = c * self.sw[None, :, None]
            return cw - np.einsum("np,kpq->knq", q, np.einsum("np,knq->kpq", q, cw))

        yw = self.y * self.sw
        return proj, yw - q @ (q.T @ yw)

    def _profile(self, base, blocks):
        """chi2 of base + each candidate block. blocks (K, n, q) -> chi2 (K,). `base` may be a cached projector."""
        proj, r = base if isinstance(base, tuple) else self._projector(base)
        c = proj(blocks)
        ctc = np.einsum("knp,knq->kpq", c, c)
        ctr = np.einsum("knp,n->kp", c, r)
        q = c.shape[-1]
        ctc = ctc + (1e-12 * np.trace(ctc, axis1=1, axis2=2)[:, None, None] + 1e-30) * np.eye(q)
        beta = np.linalg.solve(ctc, ctr[..., None])[..., 0]
        return float(r @ r) - np.einsum("kp,kp->k", ctr, beta)

    # ---------------------------------------------------------------- models

    def base(self, order=1, extra=()):
        cols = [M.base5(self.t, self.psi, self.f)]
        if order >= 2:
            cols.append(M.accel(self.t, self.psi, order))
        cols.extend(self.fixed)
        cols.extend(extra)
        return np.concatenate(cols, axis=1)

    def nuisance(self):
        blocks = [M.blend(self.psi)]
        if self.fixed:
            # VIM whose lever arm drifts linearly: a neighbour moving across the target's LSF changes both the flux
            # and the direction the photocentre is pulled
            blocks.append(self.fixed[0] * (self.t - self.t.mean())[:, None])
        return blocks

    def five(self, extra=()):
        return self.solve(self.base(1, extra), "5p")

    def accel(self, order=2, extra=()):
        return self.solve(self.base(order, extra), "7p" if order == 2 else "9p")

    def frequencies(self):
        f_min = 1.0 / P_MAX_YEARS
        f_max = M.DAYS_PER_YEAR / P_MIN_DAYS
        df = 1.0 / (OVERSAMPLE * self.span)
        return np.arange(f_min, f_max, df)

    def _kepler_blocks(self, period, ecc, tperi):
        return M.kepler(self.t[None, :], self.psi[None, :], period[:, None], ecc[:, None], tperi[:, None])

    def periodogram(self, base, freqs=None, chunk=4000):
        """Circular-orbit chi2 over a frequency grid (e = 0 is linear in phase, so frequency is the only grid)."""
        freqs = self.frequencies() if freqs is None else freqs
        pj = base if isinstance(base, tuple) else self._projector(base)
        out = np.empty(len(freqs))
        for i in range(0, len(freqs), chunk):
            p = 1.0 / freqs[i:i + chunk]
            out[i:i + chunk] = self._profile(pj, self._kepler_blocks(p, np.zeros_like(p), np.zeros_like(p)))
        return freqs, out

    def kepler(self, base=None, extra=(), name="K1", n_peaks=N_PEAKS, n_refine=N_REFINE):
        """Best single photocentre orbit on top of `base` (default: 5 parameters + extra nuisance blocks).

        Circular periodogram -> the n_peaks deepest local minima -> eccentricity x periastron grid at each ->
        Nelder-Mead on (log P, e, T_peri) from the n_refine best grid points."""
        base = self.base(1, extra) if base is None else base
        pj = self._projector(base)
        freqs, chi = self.periodogram(pj)
        loc = np.where((chi[1:-1] <= chi[:-2]) & (chi[1:-1] <= chi[2:]))[0] + 1
        peaks = loc[np.argsort(chi[loc])[:n_peaks]] if len(loc) else np.array([np.argmin(chi)])
        starts = []
        for j in peaks:
            p0 = 1.0 / freqs[j]
            pe = np.repeat(p0, len(ECC_GRID) * N_TPERI)
            ee = np.repeat(np.array(ECC_GRID), N_TPERI)
            tt = np.tile(np.arange(N_TPERI) / N_TPERI * p0, len(ECC_GRID))
            c2 = self._profile(pj, self._kepler_blocks(pe, ee, tt))
            i0 = int(np.argmin(c2))
            starts.append((c2[i0], p0, ee[i0], tt[i0] / p0))
        starts.sort()

        def unpack(z):
            per = np.exp(z[0])
            return per, (np.tanh(z[1]) + 1) / 2 * E_MAX, z[2] * per

        def obj(z):
            per, e, tp = unpack(z)
            return self._profile(pj, self._kepler_blocks(np.array([per]), np.array([e]), np.array([tp])))[0]

        best = None
        for _, p0, e0, ph in starts[:n_refine]:
            z0 = np.array([np.log(p0), np.arctanh(np.clip(2 * e0 / E_MAX - 1, -0.995, 0.995)), ph])
            res = minimize(obj, z0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-5, "maxiter": 400})
            if best is None or res.fun < best[0]:
                best = (res.fun, res.x)
        per, e, tp = (float(v) for v in unpack(best[1]))
        cols = np.concatenate([base, M.kepler(self.t, self.psi, per, e, tp)], axis=1)
        fit = self.solve(cols, name, k_extra=3, params={"period_yr": per, "ecc": e, "tperi_yr": tp})
        b, g, a, f = fit.x[-4:]
        fit.params.update(thiele_innes(a, b, f, g))
        fit.params["leverage"] = leverage(self, base, cols)
        return fit

    def kepler2(self, k1, extra=()):
        """A second orbit on top of a fitted first one, then both orbits' shapes refined together (the first orbit
        found alone is biased by the second; at high signal-to-noise that bias otherwise looks like structure)."""
        fit = self.kepler(base=k1.columns, name="K2")
        base = k1.columns[:, :-4]
        pj = self._projector(base)
        o1, o2 = k1.params, fit.params

        def enc(o):
            e = np.clip(2 * o["ecc"] / E_MAX - 1, -0.995, 0.995)
            return [np.log(o["period_yr"]), np.arctanh(e), o["tperi_yr"] / o["period_yr"]]

        def dec(z):
            per = np.exp(z[0])
            return per, (np.tanh(z[1]) + 1) / 2 * E_MAX, z[2] * per

        def blocks(z):
            p1, e1, t1 = dec(z[:3])
            p2, e2, t2 = dec(z[3:])
            return np.concatenate([M.kepler(self.t, self.psi, p1, e1, t1), M.kepler(self.t, self.psi, p2, e2, t2)],
                                  axis=1)

        def obj(z):
            return self._profile(pj, blocks(z)[None])[0]

        res = minimize(obj, np.array(enc(o1) + enc(o2)), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-4, "maxiter": 2000})
        cols = np.concatenate([base, blocks(res.x)], axis=1)
        joint = self.solve(cols, "K2", k_extra=6)
        if joint.chi2 < fit.chi2:
            fit = joint
            (p1, e1, t1), (p2, e2, t2) = dec(res.x[:3]), dec(res.x[3:])
            o1 = {"period_yr": p1, "ecc": e1, "tperi_yr": t1, **thiele_innes(*_abfg(joint.x[-8:-4]))}
            o2 = {"period_yr": p2, "ecc": e2, "tperi_yr": t2, **thiele_innes(*_abfg(joint.x[-4:]))}
        else:
            fit.k += 3  # carry the first orbit's 3 non-linear parameters
        fit.params = {"inner": o1, "outer": o2}
        return fit

    def lens(self, base=None, extra=()):
        """Astrometric microlensing on top of base: grid over t0, t_E, u0 with the 4 linear terms profiled."""
        base = self.base(1, extra) if base is None else base
        t0 = np.linspace(self.t.min() - 0.5, self.t.max() + 0.5, 60)
        te = np.geomspace(10 / M.DAYS_PER_YEAR, 3.0, 14)
        u0 = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 4.0])
        g0, ge, gu = (a.ravel() for a in np.meshgrid(t0, te, u0, indexing="ij"))
        blocks = M.lens(self.t[None, :], self.psi[None, :], g0[:, None], ge[:, None], gu[:, None])
        c2 = self._profile(base, blocks)
        # the lens must explain the photometry too: rank grid points by astrometric chi2 plus any photometric misfit
        top = np.argsort(c2)[:200]
        pen = np.array([max(self.lens_photometry(g0[j], ge[j], gu[j]), 0.0) for j in top])
        i = int(top[np.argmin(c2[top] + pen)])
        pj = self._projector(base)

        def obj(z):
            t0_, te_, u0_ = z[0], np.exp(z[1]), np.exp(z[2])
            a = self._profile(pj, M.lens(self.t, self.psi, t0_, te_, u0_)[None])[0]
            return a + max(self.lens_photometry(t0_, te_, u0_), 0.0)

        res = minimize(obj, np.array([g0[i], np.log(ge[i]), np.log(gu[i])]), method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 300})
        t0, te, u0 = res.x[0], float(np.exp(res.x[1])), float(np.exp(res.x[2]))
        cols = np.concatenate([base, M.lens(self.t, self.psi, t0, te, u0)], axis=1)
        fit = self.solve(cols, "lens", k_extra=3, params={"t0": float(t0), "tE_yr": te, "u0": u0})
        fit.params["phot_dchi2"] = self.lens_photometry(t0, te, u0)
        return fit

    def lens_photometry(self, t0, te, u0):
        """How much worse the measured fluxes fit the lens's own magnification than a constant (dark lens, all light
        from the source). > 25: the photometry rules this lens out. 0 when no flux is known."""
        if self.flux is None or self.flux_err is None:
            return 0.0
        u2 = ((self.t - t0) / te) ** 2 + u0 ** 2
        amp = (u2 + 2) / np.sqrt(u2 * (u2 + 4))
        far = amp < 1.001
        ref = np.median(self.flux[far]) if far.sum() >= 5 else np.median(self.flux)
        rel = self.flux / ref
        return float(np.sum(((rel - amp) ** 2 - (rel - 1) ** 2) / self.flux_err ** 2))

    def neighbour(self, base=None):
        """A point neighbour (rho, theta, b, relative motion) on top of base: grid over the static geometry, then
        Nelder-Mead over all five non-linear parameters with the amplitude profiled."""
        base = self.base(1) if base is None else base
        pj = self._projector(base)
        rho = np.geomspace(20, 1500, 24)
        # theta and theta + pi give the same pull up to sign (the amplitude is free); the chi2 is narrow in theta
        # (the pull matters only while the scan direction keeps the neighbour within ~2 LSF widths along scan)
        th = np.linspace(0, np.pi, 72, endpoint=False)
        bb = np.array([0.01, 0.03, 0.1, 0.3, 1.0])
        ss = np.array([35.0, 50.0])
        gr, gt, gb, gs = (a.ravel() for a in np.meshgrid(rho, th, bb, ss, indexing="ij"))
        c2 = np.concatenate([self._profile(pj, M.neighbour(self.t[None, :], self.psi[None, :], gr[k:k + 4000, None],
                                                           gt[k:k + 4000, None], gb[k:k + 4000, None],
                                                           lsf=gs[k:k + 4000, None]))
                             for k in range(0, len(gr), 4000)])
        i = int(np.argmin(c2))

        def unpack(z):
            return (np.exp(np.clip(z[0], np.log(10), np.log(2500))), z[1], 1 / (1 + np.exp(-np.clip(z[2], -30, 30))),
                    (z[3], z[4]), np.exp(np.clip(z[5], np.log(20), np.log(90))))

        def obj(z):
            r, t_, b, dmu, s_ = unpack(z)
            return self._profile(pj, M.neighbour(self.t, self.psi, r, t_, b, dmu, s_)[None])[0]

        z0 = np.array([np.log(gr[i]), gt[i], np.log(gb[i] / (1 - gb[i] + 1e-3)), 0.0, 0.0, np.log(gs[i])])
        res = minimize(obj, z0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 1200})
        r, t_, b, dmu, s_ = unpack(res.x if res.fun <= c2[i] else z0)
        cols = np.concatenate([base, M.neighbour(self.t, self.psi, r, t_, b, dmu, s_)], axis=1)
        return self.solve(cols, "nb", k_extra=6, params={"rho": r, "theta": t_, "b": b, "dmu": dmu, "lsf": s_})

    def changepoint(self, kind, base=None, extra=()):
        """Best step (position jump) or kick (proper-motion change) at a transit time, on top of base."""
        base = self.base(1, extra) if base is None else base
        ts = np.sort(self.t)
        cand = 0.5 * (ts[3:-4] + ts[4:-3])  # leave >= 4 transits on each side
        fn = M.step if kind == "step" else M.kick
        blocks = np.stack([fn(self.t, self.psi, tk) for tk in cand])
        c2 = self._profile(base, blocks)
        i = int(np.argmin(c2))
        cols = np.concatenate([base, fn(self.t, self.psi, cand[i])], axis=1)
        return self.solve(cols, kind, k_extra=1, params={"t_k": cand[i]})


def _abfg(x):
    """(B, G, A, F) column order of models.kepler -> thiele_innes(a, b, f, g) argument order."""
    b, g, a, f = x
    return a, b, f, g


def vim_index(src, cols):
    """Columns of `cols` that are the source's fixed VIM block."""
    if not src.fixed:
        return []
    v = src.fixed[0]
    return [j for j in range(cols.shape[1]) if any(np.array_equal(cols[:, j], v[:, k]) for k in range(v.shape[1]))]


def leverage(src, base, cols):
    """Share of the chi2 improvement over `base` carried by the two most improved transits.

    A high-eccentricity orbit can put its periastron on one or two outlying transits; such an 'orbit' is an outlier
    absorber, not evidence for a companion. Near 1: two transits carry the whole signal."""
    x, _ = M.wls(cols, src.y, src.s)
    xb, _ = M.wls(base, src.y, src.s)
    r0 = (src.y - base @ xb) / src.s
    r1 = (src.y - cols @ x) / src.s
    gain = r0 ** 2 - r1 ** 2
    tot = gain.sum()
    return float(np.sort(gain)[-2:].sum() / tot) if tot > 0 else 1.0


def thiele_innes(a, b, f, g):
    """Campbell elements from Thiele-Innes constants (mas): semi-major axis a0, inclination, Omega, omega."""
    u = (a ** 2 + b ** 2 + f ** 2 + g ** 2) / 2
    v = a * g - b * f
    a0 = np.sqrt(u + np.sqrt(max((u + v) * (u - v), 0.0)))
    cos_i = v / a0 ** 2
    return {"a0_mas": float(a0), "inc_deg": float(np.degrees(np.arccos(np.clip(cos_i, -1, 1))))}
