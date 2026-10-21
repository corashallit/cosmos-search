"""Concurrent-coverage tooling: GOST parsing, TESS sector ranking and binning, the caught-dip sampler, the gate."""

import numpy as np
import pandas as pd

from cosmos_search.dayone import concurrent, external
from cosmos_search.gaia import gost
from cosmos_search.vetting.candidate import LightCurve

GOST_CSV = b"""Target, ra[rad], dec[rad], ra[h:m:s], dec[d:m:s], ObservationTimeAtGaia[UTC], CcdRow[1-7], zetaFieldAngle[rad], scanAngle[rad], Fov[FovP=preceding/FovF=following], parallaxFactorAlongScan, parallaxFactorAcrossScan, ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]
42,3.21,-0.15,12:17:01.388,-08:49:15.406,2014-08-15T07:07:35.546,7,-0.0038,0.41,FoVF,0.08,0.73,2456884.793791831
42,3.21,-0.15,12:17:01.388,-08:49:15.406,2014-08-15T05:21:01.334,6,-0.0050,0.41,FoVP,0.08,0.73,2456884.719789873
"""


def test_gost_parse_sorts_and_converts_to_btjd():
    tr = gost.parse(GOST_CSV)
    assert list(tr["fov"]) == ["FoVP", "FoVF"]
    assert np.allclose(tr["t"], [-115.280210127, -115.206208169])
    assert tr["source_id"].dtype == np.int64


def test_bin_keeps_single_point_bins():
    # 30-min cadence (TESS sectors 1-26) binned at 30 min: one point per bin, all kept.
    t = np.arange(500) / 48.0
    f = 1 + 0.001 * np.random.default_rng(0).standard_normal(500)
    tb, fb, eb = external._bin(t, f, external.TESS_BIN_DAYS)
    assert len(tb) == 500
    assert np.allclose(eb, 0.001, rtol=0.2)
    # 10-min cadence: three points per bin, error shrinks by sqrt(3).
    t = np.arange(1500) / 144.0
    f = 1 + 0.001 * np.random.default_rng(1).standard_normal(1500)
    tb, fb, eb = external._bin(t, f, external.TESS_BIN_DAYS)
    assert 490 <= len(tb) <= 510
    assert np.allclose(np.median(eb), 0.001 / np.sqrt(3), rtol=0.2)


def test_rank_sectors_puts_the_dip_sector_first():
    secs = [(1, 1, 1), (7, 2, 2), (34, 2, 2), (61, 1, 1)]
    assert external.rank_sectors(secs, near=[1500.0])[0][0] == 7       # inside sector 7
    assert external.rank_sectors(secs, near=[2600.0])[0][0] == 34      # after DR4: nearest sector
    # No dip time: sectors inside DR4's span first, in numerical order, then the rest by distance.
    assert [s[0] for s in external.rank_sectors(secs)] == [1, 7, 34, 61]
    assert external.sector_gap(999, [1500.0]) == np.inf


def test_draw_caught_is_uniform_over_the_union_of_windows():
    rng = np.random.default_rng(3)
    tg = np.array([0.0, 0.05, 100.0])          # a transit pair and a lone transit
    h = 0.5
    t0, p = concurrent.draw_caught(tg, h, 20000, rng, span=(-1000.0, 1000.0))
    # Union: [-0.5, 0.55] (length 1.05) and [99.5, 100.5] (length 1.0).
    assert np.isclose(p, 2.05 / 2000.0)
    assert abs(np.mean(t0 > 50) - 1.0 / 2.05) < 0.02


def test_caught_windows_needs_five_sigma():
    assert concurrent.caught_windows(None, 0.01, 1.0, 0.04) is None
    assert np.isclose(concurrent.caught_windows(None, 0.001, 1.0, 0.1), 0.5 - 0.2 * 0.05)


def test_gate_one_look_then_testable_with_concurrent_epochs():
    rng = np.random.default_rng(5)
    # Gaia: transit pairs every 30 d; dip of 3 d at t0 = 300.02 covers one pair.
    tg = np.sort(np.concatenate([np.arange(0, 1500, 30.0), np.arange(0, 1500, 30.0) + 0.074]))
    t0, dur, dep, s = 300.02, 3.0, 0.05, 0.001
    fg = concurrent.trapezoid(tg, t0, dur, dep) + s * rng.standard_normal(len(tg))
    g = LightCurve("G", tg, fg, np.full(len(tg), s))
    assert concurrent.gate([g])[0] == "one_look"
    # Nightly ZTF-like epochs, two per night, through the dip.
    tz = np.sort(np.concatenate([np.arange(200, 400, 1.0) + 0.3, np.arange(200, 400, 1.0) + 0.35]))
    fz = concurrent.trapezoid(tz, t0, dur, dep) + 0.005 * rng.standard_normal(len(tz))
    z = LightCurve("ztf_r", tz, fz, np.full(len(tz), 0.005))
    status, det = concurrent.gate([g, z])
    assert status == "testable"
    assert "ztf_r" in det["bands"]
