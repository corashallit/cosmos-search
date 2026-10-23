import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from cosmos_search.gaia import epoch
from cosmos_search.slow import checks, detect, drift, known, select

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


def epochs_frame(t, g, bp=None, rp=None, sid=1):
    parts = [pd.DataFrame({"source_id": sid, "band": np.int8(0), "t": t, "frac_corr": g})]
    for code, y in ((1, bp), (2, rp)):
        if y is not None:
            parts.append(pd.DataFrame({"source_id": sid, "band": np.int8(code), "t": t, "frac_corr": y}))
    return pd.concat(parts, ignore_index=True)


def gaia_times(rng, n_visits=40, span=1000.0):
    """Visits of two epochs ~0.07 d apart (the two fields of view) at random times."""
    tv = np.sort(rng.uniform(0, span, n_visits))
    return np.sort(np.r_[tv, tv + 0.074])


def test_visits_group_epochs():
    t = np.array([0.0, 0.07, 0.3, 10.0, 10.07, 30.0])
    tv, yv = detect.visits(t, np.array([1.0, 3.0, 2.0, 5.0, 7.0, 4.0]))
    assert np.allclose(yv, [2.0, 6.0, 4.0]) and len(tv) == 3


def test_noise_has_abbe_near_one_and_symmetric_episodes():
    rng = np.random.default_rng(3)
    rows = [detect.star_features(epochs_frame(t, rng.normal(0, 1e-3, t.size), sid=i))
            for i in range(300) for t in [gaia_times(rng)]]
    f = pd.concat(rows, ignore_index=True)
    assert 0.85 < f["abbe"].median() < 1.15
    assert abs(f["s_dim"].median() - f["s_bright"].median()) < 0.3
    assert f["sigma_visit"].median() == pytest.approx(1e-3 / np.sqrt(2), rel=0.15)


def test_long_dip_found_with_colour():
    rng = np.random.default_rng(4)
    t = gaia_times(rng)
    sig = 1e-3
    inside = (t > 400) & (t < 600)
    g = rng.normal(0, sig, t.size) - 0.01 * inside
    bp = rng.normal(0, sig, t.size) - 0.015 * inside
    rp = rng.normal(0, sig, t.size) - 0.007 * inside
    f = detect.star_features(epochs_frame(t, g, bp, rp)).iloc[0]
    assert f["s_dim"] > 20 and f["s_bright"] < 5
    assert 380 <= f["dim_t0"] and f["dim_t1"] <= 620
    assert f["dim_depth"] == pytest.approx(0.01, rel=0.2)
    assert f["depth_bp"] > f["depth_rp"] > 0
    assert f["abbe"] < 0.3


def test_short_dip_does_not_count_as_episode():
    # One deep visit is a short dip, not a long episode: runs need MIN_EPISODE_VISITS and MIN_EPISODE_DAYS.
    rng = np.random.default_rng(5)
    t = gaia_times(rng)
    g = rng.normal(0, 1e-3, t.size)
    g[20:22] -= 0.05
    f = detect.star_features(epochs_frame(t, g)).iloc[0]
    # Counted with the deep visit, the best run would score ~40 sigma.
    assert f["s_dim"] < 6


def test_alert_lightcurve_and_epochs():
    text = ("Gaia16zzz\n#Date,JD(TCB),averagemag\n"
            "2014-09-02 14:50:24,2456903.11833,untrusted\n"
            "2015-01-01 00:00:00,2457023.5,15.00\n"
            "2016-01-01 00:00:00,2457388.5,15.75\n"
            "2021-01-01 00:00:00,2459215.5,16.00\n")
    lc = known.alert_lightcurve(text, "Gaia16zzz")
    assert list(lc["mag"]) == [15.0, 15.75, 16.0]
    assert lc["t"].iloc[0] == pytest.approx(2457023.5 - known.BTJD_T0_JD)
    ep = known.alert_epochs(lc)
    assert len(ep) == 2  # 2021 is after DR4's span
    assert ep["frac_corr"].min() < 0 < ep["frac_corr"].max() or 0 in list(ep["frac_corr"])
    assert (ep["band"] == 0).all()


@pytest.mark.parametrize("span", [27.0, 1000.0, 2000.0])
def test_slow_families_sample_and_fade(span):
    from cosmos_search.inject import lightcurve as lc

    rng = np.random.default_rng(9)
    t = np.linspace(0.0, span, 500)
    assert not set(lc.SLOW_FAMILIES) - {"secular_fade"} & set(lc.FAMILIES)
    for fam in lc.SLOW_FAMILIES:
        for _ in range(20):
            sig = lc.sample(rng, fam, 0.0, span)
            for band in ("G", "BP", "RP"):
                k = sig.factor(t, band)
                assert np.all(np.isfinite(k)) and np.all(k > 0)
                # A steady fade is centred on the span (brighter before the middle); the others only dim.
                assert fam == "secular_fade" or np.all(k <= 1 + 1e-12)
            if fam == "secular_fade":
                assert sig.factor(t[-1], "G") < sig.factor(t[0], "G")


def test_archive_score_separates_slow_from_fast():
    rng = np.random.default_rng(10)
    rows, n_each = [], 200
    for i in range(3 * n_each):
        t = gaia_times(rng, n_visits=40)
        y = rng.normal(0, 1e-3, t.size)
        if i >= n_each and i < 2 * n_each:
            y += -0.004 * (t > 500)  # slow: one step down halfway, held for the rest
        elif i >= 2 * n_each:
            y += rng.normal(0, 4e-3, t.size)  # fast: extra scatter every epoch
        rows.append(pd.DataFrame({"source_id": i, "band": np.int8(0), "t": t, "frac": y, "err_frac": 1e-3}))
    s = select.epoch_summary(pd.concat(rows, ignore_index=True))
    s["slow_z"] = select.slow_z(s["n"], s["uwv"], s["abbe"])
    noise, slow, fast = (s["slow_z"].iloc[k * n_each:(k + 1) * n_each] for k in range(3))
    assert 0.8 < s["uwv"].iloc[:n_each].median() < 1.2 and 0.85 < s["abbe"].iloc[:n_each].median() < 1.15
    assert slow.median() > np.quantile(noise, 0.99)
    assert np.quantile(fast, 0.9) < slow.median()


def test_adql_one_term_per_bin():
    cuts = pd.DataFrame({"gbin": [0, 1], "slow_z_min": [5.0, 6.5]})
    q = select.adql(cuts, source_id_range=(0, 2**35 - 1))
    assert "(phot_g_mean_mag >= 0 AND phot_g_mean_mag < 12 AND" in q and "> 6.5)" in q
    assert "BETWEEN 0 AND 34359738367" in q and "CASE" not in q
    assert "random_index_as < 1000000" in select.sample_adql(1e6)


def test_checks_colour_and_wise():
    c = checks.colour(0.017, 0.001, 0.010, 0.001)
    assert c["z_grey"] > 3 and abs(c["z_dust"]) < 1
    assert np.isnan(checks.colour(0.01, 0.001, np.nan, 0.001)["ratio"])
    rng = np.random.default_rng(12)
    mjd = np.concatenate([v + rng.uniform(0, 2, 12) for v in np.arange(56700, 60300, 182.6)])
    flux = 1 + rng.normal(0, 0.003, mjd.size) - 0.02 * ((mjd > 57500) & (mjd < 58200))
    lc = pd.DataFrame({"mjd": mjd, "band": "W1", "flux": flux, "err": 0.003})
    w = checks.wise_change(lc, 57500, 58200)
    assert w["W1"][0] == pytest.approx(-0.02, abs=0.004) and w["W1"][2] >= 3
    assert np.isnan(w["W2"][0])
    p = checks.wise_predictions(0.01, 5800)
    assert p["W1"]["grey"] == -0.01 and p["W2"]["dust_shell_1000K"] > p["W2"]["dust_shell_300K"] > 0


def test_checks_after():
    t = np.r_[np.linspace(1200, 1850, 50), np.linspace(1900, 3000, 50)]  # BTJD, either side of 2020-01-20
    z = pd.DataFrame({"band": "ztf_g", "t": t, "flux": np.where(t < 1868.5, 1.0, 0.97)})
    a = checks.after(z)["ztf_g"]
    assert a["change"] == pytest.approx(-0.03) and a["n_pre"] == 50


def test_wise_slope():
    rng = np.random.default_rng(13)
    mjd = np.concatenate([v + rng.uniform(0, 2, 12) for v in np.arange(56700, 58000, 182.6)])
    lc = pd.DataFrame({"mjd": mjd, "band": "W1", "flux": 1 - 0.01 * (mjd - 56700) / 365.25
                       + rng.normal(0, 0.002, mjd.size), "err": 0.002})
    s, e, n = checks.wise_slope(lc, 56600, 58100)["W1"]
    assert s == pytest.approx(-0.01, abs=0.002) and n >= 6 and e < 0.002


def test_outburst_is_not_a_trend():
    # One bright outburst at the start: the median-based visit noise ignores it, the trend error must not.
    rng = np.random.default_rng(14)
    t = gaia_times(rng)
    g = rng.normal(0, 1e-3, t.size)
    g[t < 60] += 0.5
    f = detect.star_features(epochs_frame(t, g)).iloc[0]
    assert abs(f["slope_z"]) < 5
    slow = detect.star_features(epochs_frame(t, rng.normal(0, 1e-3, t.size) - 0.004 * t / 365.25)).iloc[0]
    assert slow["slope_z"] < -10


def test_rule_slow_v1_flags_and_scores():
    import pandas as pd

    from cosmos_search.slow import rule

    thr, table = rule.load()
    assert set(thr) == {"G<19", "G>=19"} and len(table)
    stats = pd.DataFrame({"source_id": [1, 2, 3, 4], "gbin": [5, 5, 9, 9], "n_visits": 30, "sigma_visit": 0.01,
                          "abbe": 0.5, "s_dim": [8.0, 1.0, 8.0, 1.0], "s_bright": 0.0,
                          "slope_z": [0.0, -5.0, 0.0, -9.0]})
    f = rule.apply(stats, thr)
    assert f["flagged"].tolist() == [True, True, False, True]
    assert f["by"].tolist() == ["episode", "trend", "", "trend"]

    rng = np.random.default_rng(0)
    rows = [{"id": i, "group": "natural", "family": "", "gbin": 5, "flagged": i < 10, "by": "", "snr": np.nan}
            for i in range(5000)]
    for fam in ("staircase_fade", "slow_cycle", "accelerating_fade", "fade_with_dips"):
        rows += [{"id": 0, "group": "held_out", "family": fam, "gbin": 5, "flagged": rng.random() < 0.85,
                  "by": "trend", "snr": 30.0} for _ in range(150)]
    for fam, r in rule.KNOWN_S3.items():
        rows += [{"id": 0, "group": "known", "family": fam, "gbin": 5, "flagged": rng.random() < r, "by": "episode",
                  "snr": 30.0} for _ in range(100)]
    s = rule.score(pd.DataFrame(rows))
    assert s["criteria"]["P1_false_alarm"]["pass"] and s["criteria"]["P2_held_out_recall"]["pass"]
    rows = [r | {"flagged": False} if r["family"] == "slow_cycle" else r for r in rows]
    s = rule.score(pd.DataFrame(rows))
    assert not s["criteria"]["P2_held_out_recall"]["pass"] and not s["pass"]
