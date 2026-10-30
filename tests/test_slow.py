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


def test_time_order_reads_late_brightening_as_rise():
    # Brighter for good from t = 400 (60% of visits bright): against the median the first 40% look like a dimming
    # episode; read in time order the star rose, so it scores in s_rise and not s_fade.
    rng = np.random.default_rng(6)
    t = gaia_times(rng)
    g = rng.normal(0, 1e-3, t.size) + 0.01 * (t > 400)
    f = detect.star_features(epochs_frame(t, g)).iloc[0]
    assert f["s_dim"] > 20
    assert f["s_rise"] > 20 and f["s_fade"] < 5


def test_time_order_finds_early_drop_as_fade():
    # Fainter for good from t = 400: the bright early visits are the minority, so the median sits at the faint
    # level and s_dim sees only a brightening. In time order it is a fade ("drop"), with its window after the drop.
    rng = np.random.default_rng(7)
    t = gaia_times(rng)
    g = rng.normal(0, 1e-3, t.size) - 0.01 * (t > 400)
    f = detect.star_features(epochs_frame(t, g, 1.5 * g, 0.7 * g)).iloc[0]
    assert f["s_dim"] < 5 and f["s_fade"] > 20 and f["s_rise"] < 5
    tv, _ = detect.visits(t, t)
    assert f["fade_kind"] == "drop" and f["dim_t0"] == pytest.approx(tv[tv > 400][0])
    assert f["dim_depth"] == pytest.approx(0.01, rel=0.2)
    assert f["depth_bp"] > f["depth_rp"] > 0


def test_time_order_keeps_closed_dip_and_late_fade():
    rng = np.random.default_rng(8)
    t = gaia_times(rng)
    noise = rng.normal(0, 1e-3, t.size)
    dip = detect.star_features(epochs_frame(t, noise - 0.01 * ((t > 400) & (t < 600)))).iloc[0]
    late = detect.star_features(epochs_frame(t, noise - 0.01 * (t > 700))).iloc[0]
    assert dip["fade_kind"] == "dip" and dip["s_fade"] == pytest.approx(dip["s_dim"])
    assert late["fade_kind"] == "fade" and late["s_fade"] > 20 and late["s_rise"] < 5


def test_time_order_is_a_mirror_on_noise():
    rng = np.random.default_rng(10)
    f = pd.concat([detect.star_features(epochs_frame(t, rng.normal(0, 1e-3, t.size), sid=i))
                   for i in range(300) for t in [gaia_times(rng)]], ignore_index=True)
    assert abs(f["s_fade"].median() - f["s_rise"].median()) < 0.3
    t = gaia_times(rng)
    y = rng.normal(0, 1e-3, t.size) + 0.004 * (t > 600)
    a = detect.star_features(epochs_frame(t, y)).iloc[0]
    b = detect.star_features(epochs_frame(t, -y)).iloc[0]
    assert a["s_fade"] == pytest.approx(b["s_rise"]) and a["s_rise"] == pytest.approx(b["s_fade"])


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


def test_period_search_finds_repeated_dips_not_one_episode():
    from cosmos_search.inject import lightcurve as inj
    from cosmos_search.slow import detect, period

    rng = np.random.default_rng(3)
    tv = np.sort(rng.uniform(0, 1000, 40))
    noise = rng.normal(0, 0.002, 40)
    dips = inj.Composite(inj._periodic_dips(rng, 0, 1000, 220.0, 0.02, 60.0, 10.0, 10.0))
    y = noise + dips.factor(tv, "G") - 1
    r = period.star_scores(tv, y, detect.visit_sigma(y))
    assert r["s_rep"] > 8 and r["rep_events"] >= 2 and abs(r["rep_period"] - 220.0) < 15
    assert r["s_rep_bright"] < r["s_rep"]
    one = inj.Dip(t0=500.0, covering=0.02, ingress=20.0, egress=20.0, flat=200.0)
    y1 = noise + one.factor(tv, "G") - 1
    r1 = period.star_scores(tv, y1, detect.visit_sigma(y1))
    assert not r1["s_rep"] > 5 and r1["s_rep_any"] > r["s_rep"]  # found as one episode, not as repeating
    rn = period.star_scores(tv, noise, detect.visit_sigma(noise))
    assert rn["s_rep_any"] < 6


def test_period_window_counts_wrapped_event_once():
    from cosmos_search.slow import period

    # Two visits either side of phase 0 in the same cycle form one event; a window holding only them is invalid.
    tv = np.array([99.0, 101.0, 150.0, 199.5, 250.0, 330.0])
    r = period._scan(tv, np.zeros(6), 1.0, np.array([0.01]), 3)
    assert np.isfinite(r["dim"]).all()
    assert period.events(np.argsort(tv), 0, 2, tv, np.zeros(6), 1.0)[0] == 1
    # Two separate faint events with baseline between them pass both conditions; one event does not.
    dev = np.array([-5.0, 0.0, 0.0, -5.0, 0.0, 0.0])
    n_ev, z_between, z_rest = period.events(np.array([0, 3, 1, 2, 4, 5]), 0, 2, tv, dev, 1.0)
    assert n_ev == 2 and z_between == 0 and z_rest == 5


def test_rule_slow_v2_adds_repeated_dimming():
    import pandas as pd

    from cosmos_search.slow import rule_v2

    thr, thr_p, table = rule_v2.load()
    assert set(thr_p) == {"G<19", "G>=19"} and len(table)
    stats = pd.DataFrame({"source_id": [1, 2, 3], "gbin": [5, 5, 5], "n_visits": 30, "sigma_visit": 0.01,
                          "abbe": 0.5, "s_dim": [8.0, 1.0, 1.0], "s_bright": 0.0, "slope_z": 0.0,
                          "s_rep": [9.0, 9.0, 1.0], "rep_period": 200.0})
    f = rule_v2.apply(stats, thr, thr_p)
    assert f["flagged"].tolist() == [True, True, False]
    assert f["by"].tolist() == ["episode+repeat", "repeat", ""]


def test_band_z_measures_bp_rp_at_the_window_times():
    from cosmos_search.slow import period

    tb = np.arange(0.0, 400.0, 20.0)
    yb = np.zeros(len(tb))
    yb[[3, 4, 12, 13]] = -0.01  # faint at the window's visits
    bands = [(tb, yb, 0.002), (tb + 0.5, yb, 0.002)]
    z = period.band_z(tb[[3, 4, 12, 13]] + 0.2, bands)
    assert z > 10
    assert period.band_z(tb[[3, 4, 12, 13]] + 0.2, bands, -1.0) == -z
    assert np.isnan(period.band_z(np.array([1000.0]), bands))  # no BP or RP visit within a visit gap


def test_compiled_period_search_matches_numpy_exactly():
    from cosmos_search.inject import lightcurve as inj
    from cosmos_search.slow import detect, period

    rng = np.random.default_rng(11)
    for n, span in ((15, 1000.0), (40, 2000.0), (150, 2000.0)):  # DR3-, DR4-like and > 128 visits (pairwise split)
        tv = np.sort(rng.uniform(0, span, n))
        dips = inj.Composite(inj._periodic_dips(rng, 0, span, 230.0, 0.01, 50.0, 10.0, 10.0))
        y = rng.normal(0, 0.002, n) + dips.factor(tv, "G") - 1
        sig = detect.visit_sigma(y)
        freqs = period.freq_grid(tv[-1] - tv[0])
        a = period.scan(tv, y, sig, freqs)
        b = period.scan(tv, y, sig, freqs, compiled=False)
        for k in ("dim", "length", "start", "bright", "b_length", "b_start", "order"):
            assert np.array_equal(a[k], b[k]), k
        dev = y - np.median(y)
        for i in np.argsort(a["dim"])[::-1][:20]:
            e1 = period.events(a["order"][i], a["start"][i], a["length"][i], tv, dev, sig)
            e2 = period.events(a["order"][i], a["start"][i], a["length"][i], tv, dev, sig, compiled=False)
            assert e1[0] == e2[0] and np.array_equal(e1[1:], e2[1:], equal_nan=True)
    x = rng.normal(size=300)
    assert all(period._pairwise_sum(x[:m]) == np.sum(x[:m]) for m in range(300))


def test_search_flags_fades_and_mirrors_rises():
    from cosmos_search.slow import search

    stats = pd.DataFrame({
        "source_id": [10, 11, 12, 13], "gbin": [5, 5, 5, 9], "n_visits": 30, "sigma_visit": 0.01, "abbe": 0.5,
        "s_fade": [9.0, 1.0, 1.0, 9.0], "s_rise": [1.0, 9.0, 1.0, 1.0], "fade_kind": "dip",
        "rise_kind": ["bump", "jump", "bump", "bump"], "slope_z": [0.0, 0.0, 5.0, 0.0],
        "s_rep": np.nan, "s_rep_bright": np.nan, "dim_t0": [100.0, np.nan, np.nan, 100.0],
        "rise_t0": [np.nan, 200.0, np.nan, np.nan]})
    f = search.flag(stats)
    assert f["flagged"].tolist() == [True, False, False, False]  # 13 is fainter than 19, where s_fade needs > 17
    assert f["early"].tolist() == [False, True, False, False]
    assert f["rising"].tolist() == [False, True, True, False]
    assert f["strength"].iloc[0] == pytest.approx(9.0 / 6.04, rel=0.01)
    assert f["rise_strength"].iloc[1] == pytest.approx(9.0 / 6.04, rel=0.01)
    s = search.summary(f)
    assert s["G<19"]["fading"] == 1 and s["G<19"]["rising"] == 2 and s["G>=19"]["fading"] == 0
    assert s["G<19"]["above_strength"]["1.25"] == {"fading": 1, "rising": 2}  # star 12 rises at 5 / 3.94
    assert s["G<19"]["above_strength"]["1.5"] == {"fading": 0, "rising": 0}
    assert s["G<19"]["fade_set_aside_shared_date"] == 0 and s["G<19"]["rise_set_aside_shared_date"] == 0
    assert search.keep_mask(stats, *search.rule_v3.load()[:2]).tolist() == [True, True, True, False]


def test_period_search_ignores_one_extreme_bright_visit():
    from cosmos_search.slow import detect, period

    rng = np.random.default_rng(5)
    tv = np.sort(rng.uniform(0, 1000, 20))
    y = rng.normal(0, 0.005, 20)
    y[-2] = 9.0  # one visit at 10x the star's flux, near the end
    sig = detect.visit_sigma(y)
    assert period.star_scores(tv, y, sig, clip=None)["s_rep"] > 50  # the unclipped line fit makes it "repeat"
    assert not period.star_scores(tv, y, sig)["s_rep"] > 5
    quiet = rng.normal(0, 0.005, 20)  # nothing beyond the clip: the same numbers as without it
    a, b = period.star_scores(tv, quiet, detect.visit_sigma(quiet)), period.star_scores(
        tv, quiet, detect.visit_sigma(quiet), clip=None)
    assert all(a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])) for k in a)


def test_motion_marks_only_stars_without_measurable_motion_as_still():
    from cosmos_search.slow import search

    g = pd.DataFrame({
        # a quasar, a nearby star, a star with only proper motion, a two-parameter solution, a correlated pair
        "parallax": [0.05, 5.0, 0.1, np.nan, 0.3],
        "parallax_error": [0.2, 0.1, 0.2, np.nan, 0.1],
        "pmra": [-0.1, 20.0, 0.0, np.nan, 0.3],
        "pmra_error": [0.2, 0.1, 0.1, np.nan, 0.1],
        "pmdec": [0.2, -3.0, 2.0, np.nan, 0.0],
        "pmdec_error": [0.2, 0.1, 0.1, np.nan, 0.1],
        "pmra_pmdec_corr": [0.1, 0.0, 0.0, np.nan, 0.0],
        "parallax_pmra_corr": [0.0, 0.0, 0.0, np.nan, 0.95],
        "parallax_pmdec_corr": [0.0, 0.0, 0.0, np.nan, 0.0],
    })
    m = search.motion(g)
    assert m["still"].tolist() == [True, False, False, False, True]
    assert np.isnan(m["chi2_motion"].iloc[3])
    # Parallax and pmra 3 sigma each, but 0.95 correlated: together barely beyond the errors (chi2 ~9.2, not 18).
    assert 8 < m["chi2_motion"].iloc[4] < 10


def test_band_agreement_passes_any_mix_and_flags_g_alone():
    from cosmos_search.slow import checks

    one = np.full(6, 0.001)
    # grey, dust (BP deeper), a cooler star with RP up, G alone, BP/RP alone, BP missing
    d_g = np.array([0.05, 0.05, 0.02, 0.05, 0.0, 0.05])
    d_bp = np.array([0.05, 0.07, 0.04, 0.0, 0.05, np.nan])
    d_rp = np.array([0.05, 0.04, -0.01, 0.0, 0.05, 0.05])
    z = checks.band_agreement(d_g, one, d_bp, one, d_rp, one)
    assert z[:3].tolist() == [0.0, 0.0, 0.0]
    assert z[3] > 30 and z[4] < -30 and np.isnan(z[5])
    rows = pd.DataFrame({"strength_by": ["episode", "trend", "trend", "repeat"],
                         "dim_depth": [0.05, np.nan, np.nan, np.nan], "dim_depth_err": [0.001] * 4,
                         "depth_bp": [0.0] * 4, "depth_bp_err": [0.001] * 4, "depth_rp": [0.0] * 4,
                         "depth_rp_err": [0.001] * 4,
                         "slope": [np.nan, -0.02, -0.02, np.nan], "slope_z": [np.nan, -20.0, -20.0, np.nan],
                         "slope_bp": [np.nan, 0.0, -0.02, np.nan], "slope_bp_err": [0.001] * 4,
                         "slope_rp": [np.nan, 0.0, -0.02, np.nan], "slope_rp_err": [0.001] * 4,
                         "rep_depth": [np.nan, np.nan, np.nan, 0.0], "rep_depth_err": [0.001] * 4,
                         "rep_depth_bp": [np.nan, np.nan, np.nan, 0.03], "rep_depth_bp_err": [0.001] * 4,
                         "rep_depth_rp": [np.nan, np.nan, np.nan, 0.03], "rep_depth_rp_err": [0.001] * 4})
    v = checks.band_check(rows, blend=[0.0, 0.5, 0.0, 0.0])["band_verdict"].tolist()
    # G alone on an unblended star; the same on a blended one is untested (dilution); a grey trend; BP/RP alone
    assert v == ["g_only", "untested", "agree", "bp_rp_only"]


def test_eclipse_check_explains_a_binary_seen_by_gaia_and_ztf_and_not_one_long_dip():
    import math

    from cosmos_search.inject.classes.eclipsing_binary import EclipsingBinary

    rng = np.random.default_rng(5)
    eb = EclipsingBinary(period=2.7, t_peri=0.3, ecc=0.0, omega=math.pi / 2, incl=math.radians(89.0), r1=1.0,
                         r2=0.8, m1=1.0, m2=0.8, t1=5800.0, t2=5000.0)
    src = {"phot_g_mean_mag": 9.7, "bp_rp": 0.82, "parallax": 10.0, "parallax_error": 0.02}
    tg = np.sort(np.r_[rng.uniform(-1000, 0, 70), 0.3 + 2.7 * np.array([-300, -200, -100])])
    nights = np.sort(rng.choice(1200, 400, replace=False)) + 500.0
    tz = nights + rng.uniform(0.15, 0.35, 400)

    def obs(k, t, band):
        return pd.DataFrame({"band": band, "t": t, "flux": k * (1 + 0.005 * rng.standard_normal(len(t))),
                             "err": 0.005})

    v = checks.eclipse(obs(eb.factor(tg, "G"), tg, "G"), obs(eb.factor(tz, "ztf_r"), tz, "ztf_r"), src)
    assert v["eb_explained"]
    assert min(abs(v["eb_period"] / eb.period - 1), abs(v["eb_period"] / eb.period - 0.5)) < 1e-3
    dip = lambda t: 1 - 0.05 * ((t > -600) & (t < -450))  # noqa: E731
    v = checks.eclipse(obs(dip(tg), tg, "G"), obs(np.ones(len(tz)), tz, "ztf_r"), src)
    assert not v["eb_explained"]


def test_detector_measures_band_slopes_and_repeat_band_depths():
    from cosmos_search.slow import detect, period

    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 1000, 60))
    rows = []
    for code, slope in ((0, -0.02), (1, 0.0), (2, -0.02)):
        y = slope * (t - t.mean()) / 365.25 + rng.normal(0, 0.001, len(t))
        rows.append(pd.DataFrame({"source_id": 7, "band": code, "t": t, "frac_corr": y}))
    f = detect.star_features(pd.concat(rows, ignore_index=True)).iloc[0]
    assert f["slope_bp"] == pytest.approx(0.0, abs=0.002) and f["slope_rp"] == pytest.approx(-0.02, abs=0.002)
    assert 0 < f["slope_bp_err"] < 0.002
    tv = np.arange(0, 1000, 20.0)
    yv = np.where((tv % 100) < 25, -0.05, 0.0) + rng.normal(0, 0.002, len(tv))   # two visits in five, every 100 d
    bands = [(tv, yv * 0.5, 0.002), (tv, yv, 0.002)]
    out = period.star_scores(tv, yv, 0.002, bands=bands, band_names=["bp", "rp"])
    assert out["rep_depth"] == pytest.approx(0.05, abs=0.01)
    assert out["rep_depth_bp"] == pytest.approx(0.025, abs=0.005) and out["rep_depth_rp"] == pytest.approx(0.05, abs=0.01)
