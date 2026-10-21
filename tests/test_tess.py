import io

import numpy as np
import pytest
from astropy.io import fits

from cosmos_search.tess import qlp, tglc


def test_qlp_key_layout():
    assert qlp.key(40, 7547487) == (
        "mast/hlsp/qlp/s0040/0000/0000/0754/7487/hlsp_qlp_tess_ffi_s0040-0000000007547487_tess_v01_llc.fits")


def test_tglc_key_layout():
    assert tglc.key(40, 1, 1, 2025137660638874880) == (
        "mast/hlsp/tglc/s0040/cam1-ccd1/0020/2513/7660/6388/"
        "hlsp_tglc_tess_ffi_gaiaid-2025137660638874880-s0040-cam1-ccd1_tess_v1_llc.fits")


def _fake_qlp(n=2000):
    t = 3000 + np.arange(n) * 200 / 86400
    q = np.zeros(n, np.int32)
    q[10] = qlp.DESAT
    q[20:40] = qlp.STRAYLIGHT
    cols = [fits.Column(name="TIME", format="D", array=t),
            fits.Column(name="SAP_FLUX", format="E", array=np.ones(n)),
            fits.Column(name="DET_FLUX_ERR", format="E", array=np.full(n, 1e-3)),
            fits.Column(name="QUALITY", format="J", array=q)]
    for c in ("SAP_BKG", "SAP_X", "SAP_Y", "DET_FLUX_SML", "DET_FLUX_LAG"):
        cols.append(fits.Column(name=c, format="E", array=np.ones(n)))
    hdr = fits.Header()
    for k, v in dict(TICID=123, SECTOR=70, CAMERA=1, CCD=2, TESSMAG=10.5, RA_OBJ=1.0, DEC_OBJ=2.0,
                     TEFF=5800.0, RADIUS=1.0, MASS=1.0).items():
        hdr[k] = v
    lc_hdr = fits.Header()
    lc_hdr["TIMEDEL"] = 200 / 86400
    buf = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(header=hdr), fits.BinTableHDU.from_columns(cols, header=lc_hdr)]).writeto(buf)
    return buf.getvalue()


def test_qlp_read_and_clean():
    lc = qlp.read(_fake_qlp())
    assert lc.meta["tic"] == 123 and lc.meta["sector"] == 70 and lc.band == "TESS"
    assert len(lc.meta["desat_times"]) == 1
    clean = qlp.clean(lc)
    assert len(clean) == len(lc) - 21          # momentum dump and scattered-light cadences dropped
    assert set(clean.aux) == {"bkg", "x", "y", "flux_sml", "flux_lag"}


@pytest.mark.network
def test_qlp_fetch_live():
    lc = qlp.fetch(90, 1533579)
    assert lc is not None and lc.meta["tic"] == 1533579 and len(lc) > 5000
    assert qlp.fetch(90, 1) is None
    assert qlp.fetch(78, 101644988) is not None   # a version-02 sector


def test_veto_reasons():
    import pandas as pd

    from cosmos_search.tess.analyse import veto_reasons

    base = dict(variant="real", sector=60, camera=1, ccd=1, t_start=10.0, t_end=10.5, edge_dist=2.0,
                explained_bkg=0.0, z_bkg=0.0, depth_ratio_sml=np.nan, depth_ratio_lag=np.nan, dump_dist=5.0)
    rows = [
        base,
        {**base, "explained_bkg": 0.9, "z_bkg": 4.0},   # background rose and explains the dip
        {**base, "explained_bkg": 0.9, "z_bkg": -4.0},  # background fell with the flux: star's own light
        {**base, "edge_dist": 0.0},                     # starts on a gap
        {**base, "edge_dist": 0.05},                    # near a gap but both sides observed
    ]
    cm = {(60, 1, 1): {"bins": {40, 41}, "frac": {40: 0.1, 41: 0.1}}}  # 10.0-10.5 d covers bins 40-42
    out = veto_reasons(pd.DataFrame(rows), cm, {})
    assert out == ["common_mode", "common_mode;systematics", "common_mode", "common_mode;one_sided", "common_mode"]
    out = veto_reasons(pd.DataFrame(rows), {}, {})
    assert out == ["", "systematics", "", "one_sided", ""]
