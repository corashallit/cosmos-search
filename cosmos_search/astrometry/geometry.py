"""Observer geometry for simulated epoch astrometry: Gaia's barycentric position and AL parallax factors.

Gaia orbits L2, about 1.5e6 km (0.01 AU) beyond the Earth on the Sun-Earth line, so its barycentric position is
taken as the Earth-Moon barycentre's scaled by 1.01. Against the DR4 pre-release's own parallax_factor_al this is
good to ~2e-3 (see tests), far below any parallax error that matters here.

Parallax displaces a star by -plx times the observer's position projected on the sky; along a scan of position
angle psi (from north through east) the AL component is

    f_AL = f_ra* sin(psi) + f_dec cos(psi),   f_ra* = X sin(ra) - Y cos(ra),
    f_dec = X cos(ra) sin(dec) + Y sin(ra) sin(dec) - Z cos(dec).
"""

import numpy as np

L2_SCALE = 1.01


def observer_xyz(btjd):
    """Gaia's barycentric ICRS position in AU at the given BTJD (TCB treated as TDB; the difference is irrelevant)."""
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time

    t = Time(np.asarray(btjd, float) + 2457000.0, format="jd", scale="tdb")
    p = get_body_barycentric("earth-moon-barycenter", t).xyz.to("AU").value
    return L2_SCALE * p.T


def parallax_factor_al(ra_deg, dec_deg, psi, xyz):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    f_ra = x * np.sin(ra) - y * np.cos(ra)
    f_dec = x * np.cos(ra) * np.sin(dec) + y * np.sin(ra) * np.sin(dec) - z * np.cos(dec)
    return f_ra * np.sin(psi) + f_dec * np.cos(psi)
