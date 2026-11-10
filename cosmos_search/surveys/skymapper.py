"""SkyMapper Southern Survey DR4: per-image PSF photometry in u, v, g, r, i, z, 2014-03 to 2021-09, dec < +16.

Public TAP at api.skymapper.nci.org.au (no account). `dr4.master` has one row per object with its nearest Gaia
DR3 source; `dr4.photometry` has one row per detection on an image; `dr4.images` gives the image's date. The
Gaia-id column of `master` is not indexed (a query on it times out), so the object is found by position.

A star gets 4-20 visits per filter over seven years, from the Shallow Survey (5-40 s exposures, reaching stars
of ~9 mag) and the Main Survey (100 s). That is too sparse for a light curve and enough for a level in each year,
which is what a fade of several percent needs. DR4's zero points are fitted per image against Gaia-based
predictions for the field's stars, so one star's change does not move them.

Kept detections: Source Extractor flags below 4 (bits 1 and 2 mark neighbours and deblending; 4 and above are
saturation, truncation or failed apertures), no flagged pixels (`nimaflags` = 0: bad, saturated or cross-talk),
and a PSF magnitude with an error below MAX_ERR.
"""

import numpy as np

from cosmos_search.net import Tap

from . import MJD_TO_BTJD, empty, mag_to_flux, table

TAP_URL = "https://api.skymapper.nci.org.au/public/tap"
MATCH_ARCSEC = 1.5      # SkyMapper's seeing is ~2.5-3"; a Gaia position (at SkyMapper's epoch) is good to mas
SEARCH_ARCSEC = 5.0
MAX_ERR = 0.1           # mag
BANDS = {f: f"sm_{f}" for f in "uvgriz"}

_tap = None


def tap():
    global _tap
    if _tap is None:
        _tap = Tap(TAP_URL, job_slots=2)
    return _tap


def object_adql(ra, dec, radius_arcsec=SEARCH_ARCSEC):
    return ("SELECT object_id, raj2000, dej2000, gaia_dr3_id1, gaia_dr3_dist1, flags_psf FROM dr4.master "
            f"WHERE 1=CONTAINS(POINT('ICRS', raj2000, dej2000), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, "
            f"{radius_arcsec / 3600:.7f}))")


def photometry_adql(object_id):
    return ("SELECT p.object_id, p.image_id, p.filter, p.mag_psf, p.e_mag_psf, p.flags, p.nimaflags, p.chi2_psf, "
            "i.date, i.exp_time, i.image_type FROM dr4.photometry AS p JOIN dr4.images AS i "
            f"ON i.image_id = p.image_id WHERE p.object_id = {int(object_id)}")


def pick_object(objects, source_id, ra, dec):
    """The SkyMapper object for a Gaia source: the one whose nearest Gaia DR3 source is this one, else the nearest
    within MATCH_ARCSEC of (ra, dec); None when there is neither."""
    if not len(objects):
        return None
    sep = 3600 * np.hypot((objects["raj2000"].to_numpy(float) - ra) * np.cos(np.radians(dec)),
                          objects["dej2000"].to_numpy(float) - dec)
    mine = (objects["gaia_dr3_id1"].to_numpy() == int(source_id)) & (sep < SEARCH_ARCSEC)
    if mine.any():
        return int(objects["object_id"].to_numpy()[np.flatnonzero(mine)[np.argmin(sep[mine])]])
    return int(objects["object_id"].to_numpy()[np.argmin(sep)]) if sep.min() < MATCH_ARCSEC else None


def clean(rows, source_id):
    """Kept detections (module notes) as a COLUMNS table; one group per filter."""
    if not len(rows):
        return empty()
    ok = ((rows["flags"].to_numpy() < 4) & (rows["nimaflags"].to_numpy() == 0)
          & np.isfinite(rows["mag_psf"].to_numpy(float)) & (rows["e_mag_psf"].to_numpy(float) < MAX_ERR)
          & (rows["e_mag_psf"].to_numpy(float) > 0) & rows["filter"].astype(str).isin(list(BANDS)).to_numpy())
    r = rows[ok]
    if not len(r):
        return empty()
    band = r["filter"].astype(str).map(BANDS).to_numpy()
    mag, magerr = r["mag_psf"].to_numpy(float), r["e_mag_psf"].to_numpy(float)
    flux, err = mag_to_flux(mag, magerr)
    return table(source_id, "skymapper", band, band, r["date"].to_numpy(float) + MJD_TO_BTJD, flux, err, mag)


def fetch(source_id, ra, dec, cache=True):
    """SkyMapper DR4 epochs of the object at (ra, dec), the Gaia position at about 2017.5."""
    objects = tap().query(object_adql(ra, dec), sync=True, cache=cache).to_pandas()
    oid = pick_object(objects, source_id, ra, dec)
    if oid is None:
        return empty()
    return clean(tap().query(photometry_adql(oid), sync=True, cache=cache).to_pandas(), source_id)
