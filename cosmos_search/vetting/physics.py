"""Per-event physics: what a single occultation's duration says about any orbit that could produce it.

An occulter smaller than the star, crossing it, moves at most v ~ 2 R* / D for a dip of duration D. A
circular orbit at that speed has a = G M* / v^2, so P >= (pi / 4) G M* D^3 / R*^3: about 1 d for a 106.5 min
dip on the Sun and about 6 yr for a 1 d dip. An eccentric orbit seen near apastron moves slower than the
circular speed for its period, which relaxes the bound by up to ((1 - e) / (1 + e))^(3/2). Recurrences far
inside that floor fit no single orbiting body (docs/vetting.md §10; the DR4 plan's Gaia arm uses the same term).

Stellar radius and mass come from Gaia photometry and parallax with no extinction correction. That is crude
(+-20% in R* is typical for dwarfs, worse when reddened), so callers should demand a margin well beyond it.
"""

import numpy as np

G_MSUN = 1.32712440018e20  # m^3 s^-2
R_SUN = 6.957e8  # m
DAY = 86400.0
T_SUN = 5772.0
M_G_SUN = 4.67


def teff_from_bp_rp(bp_rp):
    """Dwarf colour-temperature relation (Mucciarelli & Bellazzini 2020), valid for BP-RP ~0.4-1.6."""
    c = float(np.clip(bp_rp, 0.39, 1.6))
    return 5040.0 / (0.4929 + 0.5092 * c - 0.0353 * c * c)


def stellar_params(source):
    """(R*, M*) in solar units from G, BP-RP and parallax, or None if the parallax is too poor (SNR < 5).

    Dwarfs (R < 1.5 Rsun) get M ~ R^1.25 (the inverse of R ~ M^0.8); larger stars get 1 Msun, which makes
    the period floor small, so evolved hosts carry little duration tension, as they should.
    """
    g, c = source.get("phot_g_mean_mag"), source.get("bp_rp")
    plx, err = source.get("parallax"), source.get("parallax_error")
    if None in (g, c, plx, err) or not (err > 0 and plx / err >= 5):
        return None
    m_g = g + 5 * np.log10(plx) - 10  # parallax in mas
    lum = 10 ** (-0.4 * (m_g - M_G_SUN))
    r = float(np.sqrt(lum) * (T_SUN / teff_from_bp_rp(c)) ** 2)
    m = r ** 1.25 if r < 1.5 else 1.0
    return r, float(m)


def min_period_days(duration_days, r_star, m_star, e_max=0.9):
    """Shortest orbital period (days) consistent with an occultation lasting duration_days."""
    d = duration_days * DAY
    p_circ = np.pi / 4 * G_MSUN * m_star * d ** 3 / (r_star * R_SUN) ** 3
    return float(p_circ * ((1 - e_max) / (1 + e_max)) ** 1.5 / DAY)
