"""TESS-Gaia Light Curves (TGLC; Han & Brandt 2023) from MAST's public S3 bucket.

TGLC fits Gaia-informed PSFs to FFIs and so reaches Tmag ~16 and deblends crowded fields. Files are keyed
by Gaia DR3 source_id and camera/CCD, which makes TGLC the natural bridge to the Gaia arm (T5): a Gaia
candidate's TESS light curve is one deterministic path away once its sector and CCD are known.

Used here as an independent second pipeline: a dip that appears in both QLP and TGLC is not a QLP
artefact, and TGLC's PSF flux against its aperture flux is another blend test.
"""

import io

import numpy as np
from astropy.io import fits

from cosmos_search.dimming.lightcurve import LightCurve

from .http import STPUBDATA, get_bytes

PREFIX = "mast/hlsp/tglc"


def key(sector, camera, ccd, gaia_id):
    # Directories are the id zero-padded to 21 digits, first 16 digits in groups of four:
    # 2025137660638874880 -> 0020/2513/7660/6388.
    padded = f"{int(gaia_id):021d}"[:16]
    groups = "/".join(padded[i:i + 4] for i in range(0, 16, 4))
    cc = f"cam{int(camera)}-ccd{int(ccd)}"
    s = f"s{int(sector):04d}"
    return f"{PREFIX}/{s}/{cc}/{groups}/hlsp_tglc_tess_ffi_gaiaid-{int(gaia_id)}-{s}-{cc}_tess_v1_llc.fits"


def url(sector, camera, ccd, gaia_id):
    return f"{STPUBDATA}/{key(sector, camera, ccd, gaia_id)}"


def fetch(sector, camera, ccd, gaia_id):
    data = get_bytes(url(sector, camera, ccd, gaia_id), missing_ok=True)
    return None if data is None else read(data)


def read(data):
    with fits.open(io.BytesIO(data), memmap=False) as h:
        hdr, d = h[0].header, h[1].data
        meta = {
            "survey": "tess-tglc", "gaia_dr3": int(hdr["GAIADR3"]), "tic": _int(hdr.get("TICID")),
            "sector": int(hdr["SECTOR"]), "camera": int(hdr["CAMERA"]), "ccd": int(hdr["CCD"]),
            "tmag": float(hdr["TESSMAG"]), "ra": float(hdr["RA_OBJ"]), "dec": float(hdr["DEC_OBJ"]),
            "gaia_g": float(hdr.get("GAIA_G", np.nan)), "gaia_bp": float(hdr.get("GAIA_BP", np.nan)),
            "gaia_rp": float(hdr.get("GAIA_RP", np.nan)),
        }
        flags = np.asarray(d["TESS_flags"], np.int64) | (np.asarray(d["TGLC_flags"], np.int64) << 16)
        flux = np.asarray(d["cal_aper_flux"], float)
        aux = {"bkg": np.asarray(d["background"], float), "flux_psf": np.asarray(d["cal_psf_flux"], float)}
        t = np.asarray(d["time"], float)
    # TGLC has no per-point errors; LightCurve.clean fills them from the point-to-point scatter.
    return LightCurve(t, flux, np.full(len(t), np.nan), flags, band="TESS", meta=meta, aux=aux)


def clean(lc):
    return lc.clean(bad_bits=~0)


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
