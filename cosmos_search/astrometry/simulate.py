"""DR4-like per-transit AL abscissae on real GOST cadence, for natural backgrounds and non-gravitational injections.

Cadence and geometry come from GOST predictions for real stars (`concurrent/gost_dr4.parquet`: times, scan angles;
GOST's scan angle equals DR4's scan_pos_angle to 0.003 deg on the pre-release sources), thinned to DR4's retention,
with parallax factors from `geometry`. Each simulated source borrows one real star's position, G and parallax.

Noise: per-transit AL sigma(G) fitted to the 12 DR4 pre-release sources after inflating the formal errors by their
intra-transit chi^2 (sigma^2 = a^2 + b^2 10^(0.4(G-15)) + c^2 10^(0.8(G-15)); 0.040 mas at G 12, 0.051 at 14, 0.58 at
19, 1.3 at 20). Pre-release residuals of single stars are Gaussian (kurtosis 3.1, none beyond 4 sigma in 631), but a
rare outlier tail is added anyway, and the sigma the fit is told can be wrong by a per-source factor, because DR4's
true calibration error is unknown.

Natural classes (everything gravity or ordinary light-centroid physics produces):
    single, binary (luminous companion), dark (WD/NS/BH companion), triple, blend (resolved-ish neighbour, scan-angle
    dependent pull, possibly moving relative to the target), vim (unresolved pair with a variable component, flux
    given to the fit), giant (convective photocentre jitter, red noise of ~6% of the angular radius), lens
    (astrometric microlensing by a dark lens).
Non-gravitational families (no force on a mass explains them):
    step (photocentre jump), kick (proper-motion change), thrust (acceleration switched on for a while), square
    (periodic square-wave displacement, a shape no orbit makes).
"""

import numpy as np

from . import geometry
from . import models as M

NOISE_ABC = (0.0372, 0.0557, 0.0117)  # mas; fitted to the pre-release, see module docstring
RETENTION = 0.85  # DR4 AGIS-used transits / GOST predictions on the pre-release sources (median)
OUTLIER_RATE = 0.003
OUTLIER_SCALE = 6.0
SIGMA_MISCAL = 0.08  # log-normal scatter of the true/assumed noise ratio per source
def flux_noise(g):
    """Fractional per-transit G flux error (~0.1% bright, ~3% at G 19)."""
    x = np.asarray(g, float) - 15.0
    return np.sqrt(0.001 ** 2 + (0.0015 * 10 ** (0.2 * x)) ** 2 + (0.0008 * 10 ** (0.4 * x)) ** 2)
LSF_SIGMA_MAS = 45.0  # AL line-spread function ~ 105 mas FWHM
MULTI_PEAK_MAS = 120.0
MULTI_PEAK_B = 0.06  # delta G < 3
WINDOW_MAS = 1000.0  # neighbours further than this (AL or AC) fall outside the window

NATURAL = {"single": 0.45, "binary": 0.28, "dark": 0.03, "triple": 0.06, "blend": 0.08, "vim": 0.04,
           "giant": 0.05, "lens": 0.01}
NONGRAV = ("step", "kick", "thrust", "square")
INSTRUMENT = ("fov_jump",)  # a calibration jump in one field of view only: Gaia's two FoVs see the same sky


def sigma_transit(g):
    a, b, c = NOISE_ABC
    x = np.asarray(g, float) - 15.0
    return np.sqrt(a ** 2 + b ** 2 * 10 ** (0.4 * x) + c ** 2 * 10 ** (0.8 * x))


class Cadence:
    """Real GOST transits + host star properties; draws a thinned cadence for a random star."""

    def __init__(self, gost, stars):
        self.stars = stars.set_index("source_id")
        self.by = {sid: g.sort_values("t") for sid, g in gost.groupby("source_id") if sid in self.stars.index}
        self.ids = np.array(sorted(self.by))
        self._xyz = {}

    def draw(self, rng, sid=None):
        sid = self.ids[rng.integers(len(self.ids))] if sid is None else sid
        g = self.by[sid]
        keep = rng.random(len(g)) < RETENTION
        g = g[keep]
        if sid not in self._xyz:
            self._xyz[sid] = (self.by[sid]["t"].to_numpy(), geometry.observer_xyz(self.by[sid]["t"].to_numpy()))
        tt, xyz = self._xyz[sid]
        xyz = xyz[np.searchsorted(tt, g["t"].to_numpy())]
        st = self.stars.loc[sid]
        psi = g["scan_angle"].to_numpy()
        f = geometry.parallax_factor_al(st["ra"], st["dec"], psi, xyz)
        t = (g["t"].to_numpy() + 2457000.0 - 2457936.875) / M.DAYS_PER_YEAR  # J2017.5 = JD 2457936.875
        self.last_fov = np.where(g["fov"].to_numpy() == "FoVP", 1, 2)
        return sid, st, t, psi, f


def thiele_innes(a0, inc, node, omega):
    ci = np.cos(inc)
    A = a0 * (np.cos(omega) * np.cos(node) - np.sin(omega) * np.sin(node) * ci)
    B = a0 * (np.cos(omega) * np.sin(node) + np.sin(omega) * np.cos(node) * ci)
    F = a0 * (-np.sin(omega) * np.cos(node) - np.cos(omega) * np.sin(node) * ci)
    G = a0 * (-np.sin(omega) * np.sin(node) + np.cos(omega) * np.cos(node) * ci)
    return A, B, F, G


def orbit_offsets(t, period, ecc, tperi, a0, inc, node, omega):
    x, y = M.kepler_xy(t, period, ecc, tperi)
    A, B, F, G = thiele_innes(a0, inc, node, omega)
    return B * x + G * y, A * x + F * y  # (dra*, ddec)


def al(dra, ddec, psi):
    return dra * np.sin(psi) + ddec * np.cos(psi)


def _mass_from_abs_g(mg):
    return float(np.clip(10 ** (-0.1 * (mg - 4.7)), 0.1, 3.0))


def _random_orbit(rng, period_yr, m1, m2, beta, plx, ecc=None):
    a_au = (max(m1 + m2, 1e-3) * period_yr ** 2) ** (1 / 3)
    a0 = a_au * abs(m2 / (m1 + m2) - beta) * plx
    if ecc is None:
        ecc = 0.0 if period_yr * M.DAYS_PER_YEAR < 10 else rng.uniform(0, 0.9)
    return dict(period=period_yr, ecc=ecc, tperi=rng.uniform(0, period_yr), a0=a0,
                inc=np.arccos(rng.uniform(-1, 1)), node=rng.uniform(0, 2 * np.pi), omega=rng.uniform(0, 2 * np.pi))


def _log_normal_period(rng, max_yr=1e4):
    while True:
        p = 10 ** rng.normal(5.03, 2.28) / M.DAYS_PER_YEAR  # Raghavan et al. 2010
        if 0.3 / M.DAYS_PER_YEAR < p < max_yr:
            return p


def _blend_pull(d, b, s=LSF_SIGMA_MAS):
    """AL centroid of a single-LSF fit to target + neighbour (flux ratio b <= 1) at AL offset d (mas).

    A least-squares fit with free amplitude maximises the cross-correlation, i.e. the peak of a two-Gaussian mixture of
    width sqrt(2) s at 0 and d; found on a grid between them and polished by Newton steps."""
    d = np.asarray(d, float)
    s2 = 2 * s * s
    grid = np.linspace(-0.1, 1.1, 121)[None, :] * d[:, None]
    cc = np.exp(-grid ** 2 / s2) + b * np.exp(-(grid - d[:, None]) ** 2 / s2)
    x = grid[np.arange(len(d)), np.argmax(cc, axis=1)]
    for _ in range(4):
        e0, e1 = np.exp(-x ** 2 / s2), b * np.exp(-(x - d) ** 2 / s2)
        g1 = -2 * (x * e0 + (x - d) * e1) / s2
        h = -2 * (e0 + e1) / s2 + 4 * (x ** 2 * e0 + (x - d) ** 2 * e1) / s2 ** 2
        x = np.where(h < 0, x - g1 / h, x)
    return x


def _ou(rng, t, sigma, tau):
    """Ornstein-Uhlenbeck path at sorted times t (years), stationary sd sigma, correlation time tau (years)."""
    out = np.empty(len(t))
    out[0] = rng.normal(0, sigma)
    for i in range(1, len(t)):
        r = np.exp(-(t[i] - t[i - 1]) / tau)
        out[i] = r * out[i - 1] + rng.normal(0, sigma * np.sqrt(1 - r * r))
    return out


def simulate(rng, cadence, kind, amp=None, sid=None):
    """One source. Returns (data dict for fit.Source, truth dict). amp: non-grav amplitude in per-transit sigmas."""
    sid, st, t, psi, f = cadence.draw(rng, sid)
    g = float(st["phot_g_mean_mag"])
    plx = float(st["parallax"]) if np.isfinite(st["parallax"]) else 0.3
    plx = max(abs(plx), 0.05)
    mg = g + 5 * np.log10(plx / 100.0)
    m1 = _mass_from_abs_g(mg)
    sig = sigma_transit(g) * np.ones(len(t))
    truth = {"kind": kind, "gost_id": int(sid), "g": g, "plx": plx, "m1": m1, "n": len(t)}
    pm = rng.normal(0, 8, 2)
    dra = rng.normal(0, 0.5) + pm[0] * t
    ddec = rng.normal(0, 0.5) + pm[1] * t
    w = al(dra, ddec, psi) + plx * f
    flux = None
    multi = np.zeros(len(t), bool)

    if kind in ("binary", "dark"):
        p = _log_normal_period(rng)
        if kind == "binary":
            q = rng.uniform(0.1, 1.0)
            m2, beta = q * m1, q ** 4 / (1 + q ** 4)
        else:
            m2 = rng.choice([0.6, 1.4, 8.0], p=[0.85, 0.1, 0.05]) * rng.uniform(0.8, 1.25)
            beta = 0.0
        o = _random_orbit(rng, p, m1, m2, beta, plx)
        w += al(*orbit_offsets(t, o["period"], o["ecc"], o["tperi"], o["a0"], o["inc"], o["node"], o["omega"]), psi)
        truth.update(m2=m2, beta=beta, **{f"o_{k}": v for k, v in o.items()})
    elif kind == "triple":
        p_in = _log_normal_period(rng, max_yr=8.0)
        p_out = p_in * 10 ** rng.uniform(0.7, 3.0)
        q_in, q_out = rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0)
        m2 = q_in * m1
        o_in = _random_orbit(rng, p_in, m1, m2, q_in ** 4 / (1 + q_in ** 4), plx)
        m3 = q_out * (m1 + m2)
        l12, l3 = 1 + q_in ** 4, (m3 / m1) ** 4
        o_out = _random_orbit(rng, p_out, m1 + m2, m3, l3 / (l12 + l3), plx)
        for o in (o_in, o_out):
            w += al(*orbit_offsets(t, o["period"], o["ecc"], o["tperi"], o["a0"], o["inc"], o["node"], o["omega"]),
                    psi)
        truth.update(p_in=p_in, p_out=p_out, a0_in=o_in["a0"], a0_out=o_out["a0"])
    elif kind == "blend":
        rho = 10 ** rng.uniform(np.log10(30), np.log10(1500))
        th = rng.uniform(0, 2 * np.pi)
        b = 10 ** rng.uniform(-2, 0)
        rel = rng.normal(0, 5, 2) if rng.random() < 0.5 else np.zeros(2)  # chance alignment or comoving
        ox = rho * np.sin(th) + rel[0] * t
        oy = rho * np.cos(th) + rel[1] * t
        d_al = ox * np.sin(psi) + oy * np.cos(psi)
        d_ac = ox * np.cos(psi) - oy * np.sin(psi)
        inside = (np.abs(d_al) < WINDOW_MAS) & (np.abs(d_ac) < WINDOW_MAS)
        x = _blend_pull(d_al, b)
        w += np.where(inside, x, 0.0)
        # the same single-LSF fit returns an amplitude that includes the neighbour's overlap with the fitted LSF:
        # flux and pull rise together, which is what lets VIM-type columns recognise a close or crossing neighbour
        flux = np.where(inside, np.exp(-x ** 2 / (4 * LSF_SIGMA_MAS ** 2))
                        + b * np.exp(-(x - d_al) ** 2 / (4 * LSF_SIGMA_MAS ** 2)), 1.0)
        # Gaia's image parameter determination sees a second peak when the neighbour is separated along scan by
        # more than ~one LSF width and not too faint (DR4 transit_proc_flags multi-source; DR3 ipd_frac_multi_peak)
        multi = inside & (np.abs(d_al) > MULTI_PEAK_MAS) & (b > MULTI_PEAK_B)
        truth.update(rho=rho, b=b, rel_pm=float(np.hypot(*rel)))
    elif kind == "vim":
        rho = rng.uniform(5, 60)
        th = rng.uniform(0, 2 * np.pi)
        f2 = 10 ** rng.uniform(-1.5, 0)
        amp_v = rng.uniform(0.05, 0.6)
        if rng.random() < 0.6:
            pv = 10 ** rng.uniform(-0.5, 2.7) / M.DAYS_PER_YEAR
            var = 1 + amp_v * np.sin(2 * np.pi * t / pv + rng.uniform(0, 2 * np.pi))
        else:
            var = np.exp(_ou(rng, t, amp_v, rng.uniform(0.02, 1.0)))
        F2 = f2 * np.clip(var, 0.02, None)
        beta = F2 / (1 + F2)
        w += al(rho * np.sin(th) * beta, rho * np.cos(th) * beta, psi)
        flux = 1 + F2
        truth.update(rho=rho, f2=f2, amp_v=amp_v)
    elif kind == "giant":
        radius_rsun = 10 ** rng.uniform(1, 2.5)
        theta = radius_rsun * 0.00465 * plx
        tau = rng.uniform(0.1, 1.0)
        jx, jy = _ou(rng, t, 0.06 * theta, tau), _ou(rng, t, 0.06 * theta, tau)
        w += al(jx, jy, psi)
        truth.update(radius_rsun=radius_rsun, theta_mas=theta, tau=tau)
    elif kind == "lens":
        te = 10 ** rng.uniform(np.log10(20), np.log10(300)) / M.DAYS_PER_YEAR
        t0 = rng.uniform(t.min(), t.max())
        u0 = 10 ** rng.uniform(-1, 0.5)
        th_e = 10 ** rng.uniform(np.log10(0.3), np.log10(5))
        phi = rng.uniform(0, 2 * np.pi)
        tau = (t - t0) / te
        den = tau ** 2 + u0 ** 2 + 2
        ex, ey = np.sin(phi), np.cos(phi)
        w += al(th_e * (tau * ex + u0 * ey) / den, th_e * (tau * ey - u0 * ex) / den, psi)
        u2 = tau ** 2 + u0 ** 2
        flux = (u2 + 2) / np.sqrt(u2 * (u2 + 4))  # a dark lens magnifies the source too
        truth.update(te=te, t0=t0, u0=u0, theta_e=th_e)
    elif kind in NONGRAV:
        a = amp * float(np.median(sig))
        phi = rng.uniform(0, 2 * np.pi)
        ex, ey = np.sin(phi), np.cos(phi)
        lo, hi = np.quantile(t, [0.1, 0.9])
        tk = rng.uniform(lo, hi)
        span = t.max() - t.min()
        if kind == "step":
            s = (t >= tk) * a
        elif kind == "kick":
            s = np.clip(t - tk, 0, None) / max(t.max() - tk, 1e-3) * a
        elif kind == "thrust":
            dur = rng.uniform(0.2, 0.6) * span
            t2 = min(tk + dur, t.max())
            tt = np.clip(t, tk, t2) - tk
            s = 0.5 * tt ** 2 + (t2 - tk) * np.clip(t - t2, 0, None)
            s = s / max(s.max(), 1e-9) * a
        else:  # square
            ps = 10 ** rng.uniform(np.log10(30), np.log10(1000)) / M.DAYS_PER_YEAR
            s = a * np.sign(np.sin(2 * np.pi * (t - tk) / ps))
            truth.update(p_sq=ps)
        w += al(s * ex, s * ey, psi)
        truth.update(amp=amp, amp_mas=a, t_k=tk)
    elif kind == "fov_jump":
        a = amp * float(np.median(sig))
        tk = rng.uniform(*np.quantile(t, [0.1, 0.9]))
        which = rng.integers(1, 3)
        w += a * ((cadence.last_fov == which) & (t >= tk))
        truth.update(amp=amp, amp_mas=a, t_k=tk, fov=int(which))
    elif kind != "single":
        raise ValueError(kind)

    flux = (np.ones(len(t)) if flux is None else flux) * (1 + rng.normal(0, flux_noise(g), len(t)))
    miscal = float(np.exp(rng.normal(0, SIGMA_MISCAL)))
    noise = rng.normal(0, 1, len(t)) * sig * miscal
    out = rng.random(len(t)) < OUTLIER_RATE
    noise[out] *= OUTLIER_SCALE
    truth.update(miscal=miscal, n_outliers=int(out.sum()))
    # what DR4 would let us estimate for the luminous star: mass from M_G, radius from mass (giants: their radius)
    r1 = truth.get("radius_rsun", float(m1 ** 0.8)) * float(np.exp(rng.normal(0, 0.2)))
    data = {"t": t, "psi": psi, "f_al": f, "w": w + noise, "sigma": sig, "flux": flux,
            "flux_err": flux_noise(g) * np.ones(len(t)), "fov": cadence.last_fov,
            "multi_source": multi, "m1": m1, "r1": r1}
    return data, truth
