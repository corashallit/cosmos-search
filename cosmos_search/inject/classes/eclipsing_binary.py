"""Synthetic eclipsing binaries, for measuring what the EB filter (cosmos_search.ebfilter) removes.

Real EB catalogues cannot be the truth for a filter that runs on the same photometry they were found in (DPAC's
eclipsing-binary table comes from the Gaia epochs; ZTF's catalogues from ZTF's), and they hold only the EBs that
were easy to find. Injected binaries have exact truth, cover the periods and eclipse shapes the queue will meet,
and go into real hosts' Gaia and ZTF epochs, so recall is measured on the real cadence and noise.

The model is two uniform disks on a Keplerian orbit: the star in front hides the overlap area of the one behind,
each star's surface brightness is a blackbody at its temperature in each band, a third light dilutes everything,
and an ellipsoidal term of amplitude ~ q (R/a)^3 sin^2 i rides on top. That is enough to set eclipse depth,
duration, timing and colour; limb darkening and reflection change shapes by less than the photometric noise at the
sampling we have.

The primary is the host itself when Gaia gives its radius and mass (vetting.physics); the companion is a main-
sequence star with mass ratio q ~ U(0.1, 1). Periods are log-uniform over 0.3-300 d, so recall can be read per
period bin and reweighted to any period distribution.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from cosmos_search.inject.lightcurve import wavelength
from cosmos_search.vetting import physics

G_SI = 6.674e-11
M_SUN = 1.989e30
R_SUN = 6.957e8
DAY = 86400.0
HC_K_UM = 14387.77  # h c / k in micrometre kelvin


def _planck(lam_um, temp):
    return 1.0 / (lam_um ** 5 * np.expm1(HC_K_UM / (lam_um * temp)))


def _overlap(r1, r2, d):
    """Area of the intersection of disks of radii r1, r2 with centres d apart (vectorised in d)."""
    d = np.asarray(d, float)
    out = np.zeros_like(d)
    full = d <= abs(r1 - r2)
    out[full] = math.pi * min(r1, r2) ** 2
    part = (d < r1 + r2) & ~full
    if part.any():
        x = d[part]
        a1 = r1 ** 2 * np.arccos(np.clip((x ** 2 + r1 ** 2 - r2 ** 2) / (2 * x * r1), -1, 1))
        a2 = r2 ** 2 * np.arccos(np.clip((x ** 2 + r2 ** 2 - r1 ** 2) / (2 * x * r2), -1, 1))
        a3 = 0.5 * np.sqrt(np.clip((-x + r1 + r2) * (x + r1 - r2) * (x - r1 + r2) * (x + r1 + r2), 0, None))
        out[part] = a1 + a2 - a3
    return out


def _kepler(m, e):
    """Eccentric anomaly for mean anomaly m (Newton's method)."""
    E = m + e * np.sin(m) if e < 0.8 else np.full_like(m, math.pi)
    for _ in range(50):
        d = (E - e * np.sin(E) - m) / (1 - e * np.cos(E))
        E -= d
        if np.max(np.abs(d)) < 1e-10:
            break
    return E


@dataclass
class EclipsingBinary:
    """Flux factor of a binary relative to its out-of-eclipse light. Radii in Rsun, masses in Msun, times in days."""

    period: float
    t_peri: float
    ecc: float
    omega: float            # argument of periastron, radians
    incl: float             # radians
    r1: float
    r2: float
    m1: float
    m2: float
    t1: float               # K
    t2: float               # K
    third_light: float = 0.0  # extra light as a fraction of the binary's
    kind: str = field(default="eclipsing_binary", init=False)

    def _geometry(self, t):
        a = (G_SI * (self.m1 + self.m2) * M_SUN * (self.period * DAY) ** 2 / (4 * math.pi ** 2)) ** (1 / 3) / R_SUN
        m = 2 * math.pi * ((np.asarray(t, float) - self.t_peri) / self.period % 1.0)
        E = _kepler(m, self.ecc)
        nu = 2 * np.arctan2(math.sqrt(1 + self.ecc) * np.sin(E / 2), math.sqrt(1 - self.ecc) * np.cos(E / 2))
        r = a * (1 - self.ecc * np.cos(E))
        theta = nu + self.omega
        sep = r * np.sqrt(1 - math.sin(self.incl) ** 2 * np.sin(theta) ** 2)
        front2 = np.sin(theta) > 0          # star 2 between us and star 1 (primary eclipse when star 1 is hotter)
        return a, theta, sep, front2

    def factor(self, t, band):
        lam = wavelength(band)
        s1, s2 = _planck(lam, self.t1), _planck(lam, self.t2)
        l1, l2 = s1 * self.r1 ** 2 * math.pi, s2 * self.r2 ** 2 * math.pi
        a, theta, sep, front2 = self._geometry(t)
        ov = _overlap(self.r1, self.r2, sep)
        lost = np.where(front2, s1 * ov, s2 * ov)
        q = self.m2 / self.m1
        sin2i = math.sin(self.incl) ** 2
        ell = (l1 * q * (self.r1 / a) ** 3 + l2 / q * (self.r2 / a) ** 3) * sin2i / (l1 + l2)
        k = (l1 + l2 - lost) / (l1 + l2)
        k = k * (1 - ell * np.cos(2 * (theta - math.pi / 2)))
        return (k + self.third_light) / (1 + self.third_light)


def host_star(source, rng):
    """(R, M, Teff) for the primary: the host's own when Gaia gives them, else a main-sequence draw."""
    sp = physics.stellar_params(source) if source else None
    bp_rp = (source or {}).get("bp_rp")
    if sp and bp_rp is not None and np.isfinite(bp_rp):
        r, m = sp
        return r, m, physics.teff_from_bp_rp(bp_rp)
    r = float(np.exp(rng.uniform(np.log(0.5), np.log(2.5))))
    m = r ** 1.25 if r < 1.5 else 1.0
    return r, m, 5772.0 * m ** 0.55


def sample(rng, source, t_min, t_max, p_range=(0.3, 300.0)):
    """One eclipsing binary around a host (source: dict as vetting.physics.stellar_params takes, or None)."""
    r1, m1, t1 = host_star(source, rng)
    for _ in range(1000):
        q = rng.uniform(0.1, 1.0)
        m2 = q * m1
        r2 = m2 ** 0.8
        t2 = t1 * q ** 0.55
        period = float(np.exp(rng.uniform(*np.log(p_range))))
        a = (G_SI * (m1 + m2) * M_SUN * (period * DAY) ** 2 / (4 * math.pi ** 2)) ** (1 / 3) / R_SUN
        ecc = 0.0 if period < 3 else float(rng.uniform(0, 0.6))
        if a * (1 - ecc) < 1.1 * (r1 + r2):
            continue
        omega = float(rng.uniform(0, 2 * math.pi))
        # Impact parameter at the primary conjunction (theta = pi/2), uniform over the eclipsing range.
        r_conj = a * (1 - ecc ** 2) / (1 + ecc * math.cos(math.pi / 2 - omega))
        b = rng.uniform(0, 0.98)
        incl = math.acos(min(1.0, b * (r1 + r2) / r_conj))
        # Phase: periastron time uniform over one period.
        t_peri = float(rng.uniform(t_min, t_min + period))
        third = 0.0 if rng.random() < 0.5 else float(rng.uniform(0, 1.0))
        return EclipsingBinary(period=period, t_peri=t_peri, ecc=ecc, omega=omega, incl=incl, r1=r1, r2=r2, m1=m1,
                               m2=m2, t1=t1, t2=t2, third_light=third)
    raise RuntimeError("no detached orbit found for this host")


def truth(eb):
    from dataclasses import asdict

    return asdict(eb)


def from_truth(d):
    return EclipsingBinary(**{k: v for k, v in d.items() if k != "kind"})
