"""MIT Quick-Look Pipeline (QLP) FFI light curves from MAST's public S3 bucket.

QLP covers stars with Tmag < 13.5, one file per (sector, TIC). We read SAP_FLUX, not the detrended
DET_FLUX: QLP's detrending is a spline with 0.3-2.5 d break spacing (KSPBSMIN/KSPBSMAX in the header),
which flattens exactly the day-long dips this arm is looking for. Our own baseline (dimming.detect) clips
one-sidedly instead.

The small- and large-aperture fluxes and the background and centroid series ride along as aux channels
for the nuisance tests in dimming.features. The aperture fluxes are QLP-detrended, so their depth ratios
are only meaningful for events shorter than about a day.
"""

import io

import numpy as np
from astropy.io import fits

from cosmos_search.dimming.lightcurve import LightCurve

from .http import STPUBDATA, get_bytes

PREFIX = "mast/hlsp/qlp"

# TESS quality bits (SPOC convention, which QLP follows for the spacecraft bits).
ATTITUDE_TWEAK, SAFE_MODE, COARSE_POINT, EARTH_POINT, ARGABRIGHTENING = 1, 2, 4, 8, 16
DESAT = 32               # reaction-wheel desaturation (momentum dump)
MANUAL_EXCLUDE = 128
STRAYLIGHT = 2048        # scattered light from Earth or Moon
QLP_BIT30 = 1 << 30      # QLP-internal low-quality flag (undocumented in the HLSP readme; kept and measured)

# Straylight and QLP's bit 30 mark the scattered-light stretches near orbit perigee, where the background
# rises by orders of magnitude and SAP flux swings by tens of percent. Searching them produced ~20 spurious
# "dips" per affected star in the first test, so they are dropped; the cost is ~10-15% of each sector's
# time, which the injection recovery maps include because windows there go unrecovered.
DROP = (ATTITUDE_TWEAK | SAFE_MODE | COARSE_POINT | EARTH_POINT | ARGABRIGHTENING | DESAT | MANUAL_EXCLUDE
        | STRAYLIGHT | QLP_BIT30)
REPORT = {"straylight": STRAYLIGHT, "qlp30": QLP_BIT30}


# QLP's delivery for sectors 74-79 is version 02 (checked against the bucket on 2026-10-21); the rest are 01.
V02_SECTORS = frozenset(range(74, 80))


def versions(sector):
    """File versions to try, most likely first."""
    return ("v02", "v01") if int(sector) in V02_SECTORS else ("v01", "v02")


def key(sector, tic, version=None):
    version = version or versions(sector)[0]
    tid = f"{int(tic):016d}"
    groups = "/".join(tid[i:i + 4] for i in range(0, 16, 4))
    return f"{PREFIX}/s{int(sector):04d}/{groups}/hlsp_qlp_tess_ffi_s{int(sector):04d}-{tid}_tess_{version}_llc.fits"


def url(sector, tic, version=None):
    return f"{STPUBDATA}/{key(sector, tic, version)}"


def fetch(sector, tic):
    """Raw LightCurve (uncleaned) or None if QLP has no light curve for this star in this sector."""
    for version in versions(sector):
        data = get_bytes(url(sector, tic, version), missing_ok=True)
        if data is not None:
            return read(data)
    return None


def read(data):
    with fits.open(io.BytesIO(data), memmap=False) as h:
        hdr, d = h[0].header, h[1].data
        q = np.asarray(d["QUALITY"], np.int64)
        meta = {
            "survey": "tess-qlp", "tic": int(hdr["TICID"]), "sector": int(hdr["SECTOR"]),
            "camera": int(hdr["CAMERA"]), "ccd": int(hdr["CCD"]), "tmag": float(hdr["TESSMAG"]),
            "ra": float(hdr["RA_OBJ"]), "dec": float(hdr["DEC_OBJ"]),
            "teff": _num(hdr.get("TEFF")), "radius": _num(hdr.get("RADIUS")),
            "mass": _num(hdr.get("MASS")),
            "cadence_s": float(h[1].header.get("TIMEDEL", np.nan)) * 86400,
        }
        t = np.asarray(d["TIME"], float)
        meta["desat_times"] = t[(q & DESAT) != 0].tolist()
        names = set(d.columns.names)

        def col(*options):
            # Sectors before ~56 name the detrended columns KSPSAP_* instead of DET_*.
            for o in options:
                if o in names:
                    return np.asarray(d[o], float)
            return None

        aux = {
            "bkg": col("SAP_BKG"), "x": col("SAP_X"), "y": col("SAP_Y"),
            "flux_sml": col("DET_FLUX_SML", "KSPSAP_FLUX_SML"),
            "flux_lag": col("DET_FLUX_LAG", "KSPSAP_FLUX_LAG"),
        }
        aux = {k: v for k, v in aux.items() if v is not None}
        flux = col("SAP_FLUX")
        err = col("DET_FLUX_ERR", "KSPSAP_FLUX_ERR")
        if err is None:
            err = np.full(len(t), np.nan)
    return LightCurve(t, flux, err, q, band="TESS", meta=meta, aux=aux)


def clean(lc):
    return lc.clean(bad_bits=DROP)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan
