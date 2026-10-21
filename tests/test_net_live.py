"""Live checks against real archives. Skipped in CI (pytest -m "not network")."""

import os

import pytest

from cosmos_search.gaia import archive
from cosmos_search.net import get_client, s3_list

pytestmark = pytest.mark.network


def test_gaia_async_query_with_login():
    if not os.environ.get("GAIA_USER"):
        pytest.skip("GAIA_USER not set")
    t = archive.tap().query(
        "SELECT TOP 5 source_id, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE parallax > 100",
        cache=False,
    )
    assert len(t) == 5 and "source_id" in t.colnames


def test_list_gaia_epoch_photometry_bulk_files():
    entries = list(s3_list(archive.CDN_LIST_URL, "Gaia/gdr3/Photometry/epoch_photometry/EpochPhotometry_0000"))
    assert entries and entries[0].key.endswith(".csv.gz") and entries[0].size > 1e6


def test_list_public_tess_bucket():
    it = s3_list("https://stpubdata.s3.amazonaws.com", "mast/hlsp/qlp/", delimiter="/")
    sectors = [e.key for e in it if e.is_prefix]
    assert "mast/hlsp/qlp/s0040/" in sectors


def test_missing_key_returns_none():
    url = "https://stpubdata.s3.amazonaws.com/mast/hlsp/qlp/does-not-exist.fits"
    assert get_client().get_bytes(url, missing_ok=True) is None
