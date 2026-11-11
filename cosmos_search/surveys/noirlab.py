"""NOIRLab Source Catalog DR2: single-exposure photometry from DECam (CTIO 4 m), Mosaic-3 and 90Prime, 2012-09 to
2019-11, over about three quarters of the sky, most of it south of dec +30.

Public TAP at Astro Data Lab (datalab.noirlab.edu/tap, no account). `nsc_dr2.object` has one row per object and
`nsc_dr2.meas` one row per measurement on an exposure, keyed by `objectid`. The service takes a cone as
`'t' = q3c_radial_query(ra, dec, ra0, dec0, radius_deg)`; ADQL's CONTAINS/POINT fails there (2026-11-10).

The catalogue is every public exposure of those cameras, so coverage is uneven: a few exposures per filter for
most stars, hundreds in a few fields. Typical exposures saturate near 15.5-16 mag, so this is the survey for the
faint half of the list, where ASAS-SN and SkyMapper give out. Its last exposure is two months before DR4's, so it
is a second instrument over Gaia's own years and says nothing about the years since.

Kept measurements: Source Extractor flags below 4 (4 is saturation), `mag_auto` (the magnitude NSC's own means
are built from) with an error below MAX_ERR. One group per instrument and filter: the exposure name starts with
the instrument (c4d DECam, k4m Mosaic-3, ksb 90Prime).
"""

import numpy as np

from cosmos_search.net import Tap

from . import MJD_TO_BTJD, empty, mag_to_flux, table

TAP_URL = "https://datalab.noirlab.edu/tap"
MATCH_ARCSEC = 1.0
MAX_ERR = 0.1
BANDS = {"u": "decam_u", "g": "decam_g", "r": "decam_r", "i": "decam_i", "z": "decam_z", "Y": "decam_Y",
         "VR": "decam_VR"}

_tap = None


def tap():
    global _tap
    if _tap is None:
        _tap = Tap(TAP_URL, job_slots=2)
    return _tap


def object_adql(ra, dec, radius_arcsec=MATCH_ARCSEC):
    return ("SELECT id, ra, dec, ndet FROM nsc_dr2.object WHERE 't' = q3c_radial_query(ra, dec, "
            f"{ra:.7f}, {dec:.7f}, {radius_arcsec / 3600:.7f})")


def meas_adql(object_id):
    oid = str(object_id)
    if not oid.replace("_", "").isalnum():
        raise ValueError(f"unexpected NSC object id {oid!r}")
    return ("SELECT objectid, exposure, filter, mjd, mag_auto, magerr_auto, flags, fwhm FROM nsc_dr2.meas "
            f"WHERE objectid = '{oid}'")


def pick_object(objects, ra, dec):
    """The nearest NSC object to (ra, dec), or None."""
    if not len(objects):
        return None
    sep = np.hypot((objects["ra"].to_numpy(float) - ra) * np.cos(np.radians(dec)),
                   objects["dec"].to_numpy(float) - dec)
    return str(objects["id"].to_numpy()[np.argmin(sep)])


def clean(rows, source_id):
    """Kept measurements (module notes) as a COLUMNS table; one group per instrument and filter."""
    if not len(rows):
        return empty()
    filt = rows["filter"].astype(str).str.strip()
    ok = ((rows["flags"].to_numpy() < 4) & np.isfinite(rows["mag_auto"].to_numpy(float))
          & (rows["mag_auto"].to_numpy(float) < 50) & (rows["magerr_auto"].to_numpy(float) < MAX_ERR)
          & (rows["magerr_auto"].to_numpy(float) > 0) & filt.isin(list(BANDS)).to_numpy())
    r, filt = rows[ok], filt[ok]
    if not len(r):
        return empty()
    band = filt.map(BANDS).to_numpy()
    group = (r["exposure"].astype(str).str[:3] + "_" + filt).to_numpy()
    mag, magerr = r["mag_auto"].to_numpy(float), r["magerr_auto"].to_numpy(float)
    flux, err = mag_to_flux(mag, magerr)
    return table(source_id, "noirlab", band, group, r["mjd"].to_numpy(float) + MJD_TO_BTJD, flux, err, mag)


def fetch(source_id, ra, dec, cache=True):
    """NSC DR2 epochs of the object at (ra, dec), the Gaia position at about 2016.5."""
    oid = pick_object(tap().query(object_adql(ra, dec), sync=True, cache=cache).to_pandas(), ra, dec)
    if oid is None:
        return empty()
    return clean(tap().query(meas_adql(oid), sync=True, cache=cache).to_pandas(), source_id)
