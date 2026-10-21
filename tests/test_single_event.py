import numpy as np

from cosmos_search.vetting import single_event
from cosmos_search.vetting.candidate import LightCurve

SUN_LIKE = {"phot_g_mean_mag": 12.0, "bp_rp": 0.82, "parallax": 3.0, "parallax_error": 0.03}


def _gaia_times(rng, n_visits=30, span=1000.0):
    t = []
    for x in np.sort(rng.uniform(0, span, n_visits)):
        t += [x, x + 0.074]
    return np.array(t)


def _nightly(rng, start, end, frac=0.35):
    n = np.arange(start, end)
    n = n[rng.random(len(n)) < frac]
    return n + rng.uniform(0.1, 0.4, len(n))


def _lcs(k_fn, rng, tg, tz, s=0.002, sz=0.01):
    out = [LightCurve(b, tg, k_fn(tg) * (1 + e * rng.standard_normal(len(tg))), np.full(len(tg), e))
           for b, e in (("G", s), ("BP", 2 * s), ("RP", 2 * s))]
    out.append(LightCurve("ztf_r", tz, k_fn(tz) * (1 + sz * rng.standard_normal(len(tz))), np.full(len(tz), sz)))
    return out


def test_overlap_fraction():
    assert np.isclose(single_event.overlap_fraction(0.0, 0.1), 0.01)
    assert single_event.overlap_fraction(1.2, 0.1) == 0.0
    half = single_event.overlap_fraction(1e3, 1e3)        # a huge occulter whose edge passes the centre
    assert 0.45 < half < 0.55
    a, b = single_event.overlap_fraction(np.array([0.95, 1.05]), 0.1)
    assert 0 < b < a < 0.01


def test_no_test_without_exactly_one_event():
    rng = np.random.default_rng(0)
    tg, tz = _gaia_times(rng), _nightly(rng, 1300, 2500)
    assert single_event.test(_lcs(lambda t: np.ones_like(t), rng, tg, tz), SUN_LIKE) is None
    two = tg[[10, 40]]
    k = lambda t: 1 - 0.1 * np.any(np.abs(t[:, None] - two[None, :]) < 0.05, axis=1)
    assert single_event.test(_lcs(k, rng, tg, tz), SUN_LIKE) is None


def test_long_one_off_dip_beats_eclipses():
    """A 0.1-deep dip seen on Gaia epochs days apart: no eclipse on a Sun-like star lasts that long at a period
    common enough to be caught once and never again, so the one-off reading wins by a wide margin."""
    rng = np.random.default_rng(1)
    tg, tz = _gaia_times(rng), _nightly(rng, 1300, 2500)
    i = 20
    tg = np.sort(np.r_[tg, tg[i] + 2.0, tg[i] + 2.074, tg[i] + 4.0, tg[i] + 4.074])
    t0, t1 = tg[i] - 0.01, tg[i] + 4.1
    k = lambda t: 1 - 0.1 * ((t >= t0) & (t <= t1))
    r = single_event.test(_lcs(k, rng, tg, tz), SUN_LIKE, n_draws=20000, rng=np.random.default_rng(2))
    assert r is not None and r["n_in_dip"] >= 4
    assert r["duration_min_days"] > 4.0
    assert r["log10_oneoff_vs_eb"] > single_event.B_SHORTLIST
    assert r["single_dip_flag"]


def test_one_epoch_eclipse_is_not_flagged():
    """A long-period binary caught on one Gaia epoch: an eclipse explains it and the flag stays off."""
    rng = np.random.default_rng(3)
    tg, tz = _gaia_times(rng), _nightly(rng, 1300, 2500)
    i = 21
    k = lambda t: 1 - 0.2 * (np.abs(t - tg[i]) < 0.02)
    r = single_event.test(_lcs(k, rng, tg, tz), SUN_LIKE, n_draws=8000, rng=np.random.default_rng(4))
    assert r is not None and r["n_in_dip"] == 1
    assert not r["single_dip_flag"]


def test_colour_ratio_of_a_dust_dip():
    rng = np.random.default_rng(5)
    tg, tz = _gaia_times(rng), _nightly(rng, 1300, 2500)
    i = 30
    win = lambda t: (np.abs(t - tg[i] - 0.037) < 0.1)
    lcs = []
    for band, depth in (("G", 0.1), ("BP", 0.13), ("RP", 0.075)):
        f = 1 - depth * win(tg) + 0.001 * rng.standard_normal(len(tg))
        lcs.append(LightCurve(band, tg, f, np.full(len(tg), 0.001)))
    lcs.append(LightCurve("ztf_r", tz, 1 + 0.01 * rng.standard_normal(len(tz)), np.full(len(tz), 0.01)))
    r = single_event.test(lcs, SUN_LIKE, n_draws=4000)
    assert abs(r["colour_ratio_rp_bp"] - 0.075 / 0.13) < 0.05


def test_speed_bound_flags_an_impossibly_fast_change_on_a_giant():
    rng = np.random.default_rng(6)
    tg, tz = _gaia_times(rng), _nightly(rng, 1300, 2500)
    giant = {"phot_g_mean_mag": 9.0, "bp_rp": 1.5, "parallax": 1.0, "parallax_error": 0.02}
    i = 12
    k = lambda t: 1 - 0.3 * (np.abs(t - tg[i]) < 0.01)      # in one field of view, gone 106 min later
    r = single_event.test(_lcs(k, rng, tg, tz), giant, n_draws=4000)
    assert r["v_required_km_s"] > r["v_escape_km_s"]


def test_opposite_colour_change_excludes_occultation():
    """BP dims while RP brightens across a two-epoch event: no occulter does that, so the flag can fire on colour."""
    rng = np.random.default_rng(7)
    tg, tz = _gaia_times(rng), _nightly(rng, 1300, 2500)
    i = 14
    win = lambda t: np.abs(t - tg[i] - 0.037) < 0.06
    lcs = []
    for band, depth in (("G", 0.03), ("BP", 0.08), ("RP", -0.03)):
        lcs.append(LightCurve(band, tg, 1 - depth * win(tg) + 0.002 * rng.standard_normal(len(tg)),
                              np.full(len(tg), 0.002)))
    lcs.append(LightCurve("ztf_r", tz, 1 + 0.01 * rng.standard_normal(len(tz)), np.full(len(tz), 0.01)))
    r = single_event.test(lcs, SUN_LIKE, n_draws=4000)
    assert r is not None and r["colour_excludes_occultation"]
    assert r["colour_ratio_rp_bp"] < 0
