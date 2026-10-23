import numpy as np
import pandas as pd
import pyarrow as pa

from cosmos_search.gaia import epoch
from cosmos_search.slow import drift, known

YEAR = 365.25


def lightcurves(n_stars, rng, drift_per_year=0.002, trend_star=None, trend_per_year=0.0, sigma=0.001):
    """G light curves of quiet stars (epoch.SCHEMA) sharing a linear calibration drift; one star may also trend."""
    rows = []
    for i in range(n_stars):
        t = np.sort(rng.uniform(1700, 1700 + 3 * YEAR, 60))
        f = 1 + drift_per_year * (t - 1700) / YEAR + rng.normal(0, sigma, t.size)
        if i == trend_star:
            f = f + trend_per_year * (t - 1700) / YEAR
        rows.append(pd.DataFrame({"source_id": i + 1, "band": "G", "t": t, "flux": 1e4 * f,
                                  "err": np.full(t.size, 1e4 * sigma), "flags": 0, "transit_id": 0,
                                  "fov": 1, "ccd_row": 1, "n_obs": 9}))
    df = pd.concat(rows, ignore_index=True)
    return pa.Table.from_pandas(df, schema=epoch.SCHEMA, preserve_index=False)


def aux(n_stars):
    return pd.DataFrame({"source_id": np.arange(1, n_stars + 1), "phot_g_mean_mag": np.full(n_stars, 14.5),
                         "bp_rp": np.full(n_stars, 1.0)})


def test_drift_removed_and_own_trend_kept():
    rng = np.random.default_rng(1)
    n = 400
    lc = lightcurves(n, rng, drift_per_year=0.002, trend_star=0, trend_per_year=-0.005)
    ep = drift.normalise(lc)
    cells = drift.star_cells(aux(n))
    ref = ep[ep["source_id"] != 1]
    table = drift.fit(ref, cells, min_stars=30)
    ep["frac_corr"] = ep["frac"] - drift.drift_at(ep, cells, table)
    raw = drift.trend(ep, "frac").set_index("source_id")
    cor = drift.trend(ep, "frac_corr").set_index("source_id")
    quiet = cor.index != 1
    # The shared drift (+0.2 %/yr) is in every raw slope and gone after the correction.
    assert abs(raw.loc[quiet, "slope"].median() - 0.002) < 3e-4
    assert abs(cor.loc[quiet, "slope"].median()) < 2e-4
    # The trending star keeps its own -0.5 %/yr.
    assert abs(cor.loc[1, "slope"] - (-0.005)) < 1e-3


def test_sparse_cells_fall_back_to_pooled_then_zero():
    rng = np.random.default_rng(2)
    lc = lightcurves(40, rng)
    ep = drift.normalise(lc)
    a = aux(40)
    a.loc[:19, "bp_rp"] = 0.5  # two colour bins of 20 stars each: too few alone, enough pooled
    cells = drift.star_cells(a)
    table = drift.fit(ep, cells, min_stars=30)
    assert (table["cbin"] == drift.POOLED).all()
    d = drift.drift_at(ep, cells, table)
    assert np.isfinite(d).all() and np.any(d != 0)
    assert np.all(drift.drift_at(ep, cells, table.iloc[0:0]) == 0)


def test_half_is_stable_and_balanced():
    sid = np.arange(1, 20001, dtype=np.int64) * 7919
    h = drift.half(sid)
    assert np.array_equal(h, drift.half(sid))
    assert 0.47 < h.mean() < 0.53


def test_vsx_kind():
    assert known.vsx_kind("RCB") == "rcb"
    assert known.vsx_kind("UXOR:+TTS/ROT:") == "uxor"
    assert known.vsx_kind("L|DYPER") == "dyper"
    assert known.vsx_kind("EA") is None


def test_fading_alerts_keeps_fades_in_dr4_span():
    text = ("#Name, Date, RaDeg, DecDeg, AlertMag, HistoricMag, HistoricStdDev, Class, Published, Comment, TNSid\n"
            'Gaia16aaa,2016-03-01 00:00:00,10.0,20.0,17.5,16.0,0.1,"unknown",2016-03-02,"star fades",\n'
            'Gaia16aab,2016-03-01 00:00:00,11.0,21.0,15.0,16.0,0.1,"CV",2016-03-02,"outburst",\n'
            'Gaia21aaa,2021-03-01 00:00:00,12.0,22.0,17.5,16.0,0.1,"unknown",2021-03-02,"later fade",\n')
    f = known.fading_alerts(text)
    assert list(f["name"]) == ["Gaia16aaa"]
    assert f["kind"].iloc[0] == "alert_fade"
    assert "dG 1.5" in f["note"].iloc[0]


def test_sky_pixel_and_offsets():
    sid = np.array([5 * 2**35 * 4**7, 5 * 2**35 * 4**7 + 1, 6 * 2**35 * 4**7], dtype=np.int64)
    assert list(drift.sky_pixel(sid, level=5)) == [5, 5, 6]
    tr = pd.DataFrame({"source_id": np.repeat(sid, 4), "band": 0, "slope": [0.01] * 8 + [0.0] * 4,
                       "slope_err": 0.005})
    off = drift.sky_offsets(tr, level=5, min_stars=5)
    assert list(off["pix"]) == [5] and off["offset_z"].iloc[0] == 2.0
