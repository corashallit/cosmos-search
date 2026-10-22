"""Physical limits that decide whether a fitted motion could be gravitational, and what a non-gravitational one costs.

Units: AU, years, solar masses, so G = 4 pi^2. Angles in mas, parallax in mas.

Kepler bound. A photocentre orbit with semi-major axis a0 (mas) and period P (yr) at parallax plx has astrometric mass
function f_A = (a0/plx)^3 / P^2. For a companion of mass M2 and light fraction beta, a0/plx = a (M2/M - beta) <= a, so
f_A = M (M2/M - beta)^3 <= M2^3 / M^2 <= M. Any gravitational explanation needs total mass >= f_A, whatever the
companion is. f_A above ~100 Msun is beyond every known stellar remnant.

Periastron bound. With total mass at most m_cap, the relative orbit has a <= (m_cap P^2)^(1/3) and periastron
a (1 - e). If that is inside the luminous star even at m_cap, the orbit is impossible: the fit used eccentricity to
put a periastron spike on one or two outlying transits.

Acceleration bound. A constant acceleration g (mas/yr^2) without visible curvature over span T needs an orbit of
period P >~ 2T; a circular orbit with g = 4 pi^2 M / r^2 and r^3 = M P^2 gives M = g_phys^3 P^4 / (2 pi)^6, so
M_min = (g/plx)^3 (2T)^4 / (2 pi)^6.

Energy bound on non-gravitational motion of a star. A photon rocket of power L gives thrust L/c; to move a star of
mass M by an acceleration a it needs L = M a c. `luminosity_for_acceleration` shows that any acceleration Gaia DR4 can
measure for a solar-mass star needs ~10^4 L_sun of directed power. So a non-Keplerian *mass* motion of a star is
excluded; a non-Keplerian astrometric signal is a motion of the light centroid (occulters, blends, variability,
lensing, surface features, instrument), and the filter is designed around that.
"""

import numpy as np

M_CAP = 100.0  # Msun: heavier than any known stellar remnant
RSUN_AU = 0.00465047
AU_M = 1.495978707e11
YR_S = 3.15576e7
MSUN_KG = 1.98892e30
LSUN_W = 3.828e26
C_MS = 2.99792458e8


def mass_function(a0_mas, plx_mas, period_yr):
    return (a0_mas / plx_mas) ** 3 / period_yr ** 2


def periastron_rsun(period_yr, ecc, m_tot=M_CAP):
    return (m_tot * period_yr ** 2) ** (1 / 3) * (1 - ecc) / RSUN_AU


def acceleration_min_mass(g_mas_yr2, plx_mas, span_yr):
    return (g_mas_yr2 / plx_mas) ** 3 * (2 * span_yr) ** 4 / (2 * np.pi) ** 6


def main_sequence_radius(m1):
    return np.asarray(m1, float) ** 0.8


def kepler_verdict(period_yr, ecc, a0_mas, plx_mas, plx_err, r1_rsun, m_cap=M_CAP):
    """Is a fitted photocentre orbit physically possible? Returns (f_A, verdict) with verdict one of
    'ok', 'rp_inside_star', 'over_mass_cap', or 'unknown' (parallax not significant, so f_A is unbounded)."""
    rp = periastron_rsun(period_yr, ecc, m_cap)
    if rp < r1_rsun:
        return np.nan, "rp_inside_star"
    if not (plx_mas > 0 and plx_mas > 5 * plx_err):
        return np.nan, "unknown"
    f = mass_function(a0_mas, plx_mas, period_yr)
    return f, ("over_mass_cap" if f > m_cap else "ok")


def luminosity_for_acceleration(a_ms2, mass_msun=1.0):
    """Directed photon power (L_sun) to accelerate a star of the given mass at a (m/s^2)."""
    return mass_msun * MSUN_KG * a_ms2 * C_MS / LSUN_W


def acceleration_ms2(g_mas_yr2, plx_mas):
    """Physical acceleration of an astrometric acceleration g at parallax plx."""
    return g_mas_yr2 / plx_mas * AU_M / YR_S ** 2
