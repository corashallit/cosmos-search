"""Sky geometry for matching: unit vectors, gnomonic offsets in mas, random shifts, HEALPix grouping.

Offsets are (xi, eta) on the tangent plane at a reference position, xi toward east (increasing RA) and
eta toward north, in milliarcseconds. At the scales we match on (< 1 deg) the gnomonic projection is exact
enough that offsets add linearly.
"""

import numpy as np

MAS_PER_RAD = 180.0 / np.pi * 3.6e6
MAS_PER_DEG = 3.6e6
MAS_PER_ARCSEC = 1e3


def unit_vectors(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    cd = np.cos(dec)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])


def chord(angle_mas):
    """Chord length on the unit sphere for an angle, for KD-tree radius queries on unit vectors."""
    return 2.0 * np.sin(np.asarray(angle_mas) / MAS_PER_RAD / 2.0)


def separation_mas(ra1, dec1, ra2, dec2):
    """Great-circle separation (Vincenty form, stable at all angles)."""
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    dra = ra2 - ra1
    num = np.hypot(np.cos(dec2) * np.sin(dra),
                   np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(dra))
    den = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    return np.arctan2(num, den) * MAS_PER_RAD


def tangent_offsets(ra0, dec0, ra, dec):
    """Gnomonic offsets (xi, eta) in mas of (ra, dec) about (ra0, dec0). All in degrees."""
    ra0, dec0, ra, dec = map(np.radians, (ra0, dec0, ra, dec))
    dra = ra - ra0
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(dra)
    xi = np.cos(dec) * np.sin(dra) / cosc
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(dra)) / cosc
    return xi * MAS_PER_RAD, eta * MAS_PER_RAD


def offset_position(ra0, dec0, xi_mas, eta_mas):
    """Inverse gnomonic: the (ra, dec) in degrees at tangent-plane offset (xi, eta) mas from (ra0, dec0)."""
    ra0, dec0 = np.radians(ra0), np.radians(dec0)
    xi, eta = np.asarray(xi_mas) / MAS_PER_RAD, np.asarray(eta_mas) / MAS_PER_RAD
    den = np.cos(dec0) - eta * np.sin(dec0)
    ra = ra0 + np.arctan2(xi, den)
    dec = np.arctan2(np.sin(dec0) + eta * np.cos(dec0), np.hypot(xi, den))
    return np.degrees(ra) % 360.0, np.degrees(dec)


def random_annulus(n, rmin_mas, rmax_mas, rng):
    """n offsets drawn uniformly by area from the annulus rmin <= r <= rmax."""
    r = np.sqrt(rng.uniform(rmin_mas**2, rmax_mas**2, n))
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    return r * np.sin(phi), r * np.cos(phi)


def healpix_groups(ra_deg, dec_deg, order):
    """Nested HEALPix index at `order` for each position (used to batch nearby targets into one fetch)."""
    import cdshealpix
    from astropy.coordinates import Latitude, Longitude
    import astropy.units as u

    ipix = cdshealpix.nested.lonlat_to_healpix(
        Longitude(np.asarray(ra_deg) * u.deg), Latitude(np.asarray(dec_deg) * u.deg), order)
    return np.asarray(ipix, dtype=np.int64)


def healpix_order_for_side(side_deg):
    """Finest HEALPix order whose pixels are still at least side_deg across (pixel side ~ 58.6 deg / 2^order)."""
    return int(max(0, min(29, np.floor(np.log2(58.6 / max(side_deg, 1e-9))))))
