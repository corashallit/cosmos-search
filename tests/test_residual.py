import math

import numpy as np

from cosmos_search.ebfilter import search as S
from cosmos_search.inject import lightcurve as lcinj
from cosmos_search.inject.classes.eclipsing_binary import EclipsingBinary
from cosmos_search.residual import model as M
from cosmos_search.vetting.candidate import LightCurve

SUN_AT_100PC = {"phot_g_mean_mag": 9.7, "bp_rp": 0.82, "parallax": 10.0, "parallax_error": 0.02}


def _cadence(rng, n_nights=400, span=1200.0):
    nights = np.sort(rng.choice(int(span), n_nights, replace=False)).astype(float)
    t = np.concatenate([nights + rng.uniform(0.15, 0.35, n_nights), nights[::3] + rng.uniform(0.36, 0.45)])
    return np.sort(t)


def _eb(period=2.7, t_peri=0.3):
    return EclipsingBinary(period=period, t_peri=t_peri, ecc=0.0, omega=math.pi / 2, incl=math.radians(89.0),
                           r1=1.0, r2=0.8, m1=1.0, m2=0.8, t1=5800.0, t2=5000.0)


def _mid_primary(eb, near):
    tt = np.linspace(near, near + eb.period, 200001)
    return float(tt[np.argmin(eb.factor(tt, "G"))])


def _star(rng, extra=None, noise=0.005):
    """A ZTF-like r band and a Gaia-like G band (60 epochs, three of them at mid-primary eclipse, so the star's
    G-to-ZTF depth ratio can be measured) of one EB, times an optional extra signal factor. Both bands get the
    r-band eclipse depths: colour is the calibration's business, not the logic's."""
    eb = _eb()
    t_z = _cadence(rng)
    t_g = np.sort(np.r_[rng.uniform(0, 1200, 57), [_mid_primary(eb, x) for x in (200.0, 600.0, 1000.0)]])
    out = []
    for band, t in (("ztf_r", t_z), ("G", t_g)):
        k = eb.factor(t, "ztf_r") * (extra(t) if extra else 1.0)
        out.append(LightCurve(band, t, k * (1 + noise * rng.standard_normal(len(t))), np.full(len(t), noise)))
    return eb, out


def _force_g_epoch(rng, eb, lcs, depth_factor):
    """Put a G epoch at mid-eclipse of a cycle with no ZTF epoch nearby, multiplied by depth_factor."""
    g = lcs[1]
    t_min = _mid_primary(eb, eb.t_peri + eb.period * 150)
    f = eb.factor(np.array([t_min]), "ztf_r")[0] * depth_factor
    g.time = np.r_[g.time, t_min]
    g.flux = np.r_[g.flux, f]
    g.flux_err = np.r_[g.flux_err, 0.005]
    o = np.argsort(g.time)
    g.time, g.flux, g.flux_err = g.time[o], g.flux[o], g.flux_err[o]
    return t_min


def test_plain_eb_is_demoted_and_leaves_nothing():
    rng = np.random.default_rng(11)
    eb, lcs = _star(rng)
    t_target = _force_g_epoch(rng, eb, lcs, 1.0)
    v = M.run(lcs, SUN_AT_100PC, t_target)
    assert v["baseline"] == "eclipse" and v["eb_explained"]
    assert v["target_in_window"] and not v["target_residual"]
    assert v["n_residual_events_elsewhere"] == 0
    assert v["demote"] and not v["eb_residual"]


def test_extra_dip_inside_an_eclipse_is_a_residual():
    """The EB filter's weak spot: a dip that lands on an eclipse. The sum is deeper than the eclipse at that phase."""
    rng = np.random.default_rng(12)
    eb, lcs = _star(rng)
    t_target = _force_g_epoch(rng, eb, lcs, 0.6)              # 40% extra dimming at mid-eclipse
    v = M.run(lcs, SUN_AT_100PC, t_target)
    assert v["baseline"] == "eclipse"
    assert v["target_in_window"] and v["target_residual"]
    assert not v["demote"] and v["eb_residual"]


def test_aperiodic_train_on_an_eb_survives_in_the_residuals():
    rng = np.random.default_rng(13)
    sig = None
    for _ in range(50):
        sig = lcinj.sample(rng, "dip_train_poisson", 0.0, 1200.0)
        tt = np.linspace(0, 1200, 20001)
        if np.sum(np.diff((sig.factor(tt, "ztf_r") < 0.9).astype(int)) == 1) >= 8:
            break
    _, lcs = _star(rng, extra=lambda t: sig.factor(t, "ztf_r"))
    raw = S.labels(lcs)
    assert (raw[1] == 1).sum() > 0
    v = M.run(lcs, SUN_AT_100PC, None)
    assert v["baseline"] == "eclipse"
    assert v["n_residual_events"] >= 4
    assert not v["demote"]


def test_constant_baseline_keeps_every_dip():
    rng = np.random.default_rng(14)
    t = _cadence(rng)
    sig = lcinj.sample(rng, "grey_dip", float(t[0]), float(t[-1]))
    lc = LightCurve("ztf_r", t, sig.factor(t, "ztf_r") * (1 + 0.005 * rng.standard_normal(len(t))),
                    np.full(len(t), 0.005))
    v = M.run([lc], SUN_AT_100PC, None)
    assert v["baseline"] == "constant"
    assert v["n_residual_dip_epochs"] == int((S.labels([lc])[1] == 1).sum())
    assert not v["demote"] and not v["eb_residual"]


def test_residual_lightcurves_drop_explained_epochs_and_their_bp_rp():
    t = np.arange(20.0)
    lcs = [LightCurve("G", t, np.ones(20), np.full(20, 0.01)),
           LightCurve("BP", t + 0.001, np.ones(20), np.full(20, 0.01)),
           LightCurve("ztf_r", t + 0.5, np.ones(20), np.full(20, 0.01))]
    idx = np.array([[0, i] for i in range(20)] + [[2, i] for i in range(20)])
    tt = np.r_[t, t + 0.5]
    drop = np.zeros(40, bool)
    drop[[3, 25]] = True
    excess = np.zeros(40)
    excess[7] = 0.2
    out = M.residual_lightcurves(lcs, idx, tt, drop, excess)
    assert len(out[0].time) == 19 and 3.0 not in out[0].time
    assert len(out[1].time) == 19                                   # BP follows G's transit
    assert len(out[2].time) == 19 and 5.5 not in out[2].time
    assert abs(out[0].flux[list(out[0].time).index(7.0)] - 1.2) < 1e-12


def test_requeue_demotes_to_controls_and_audits_them():
    import pandas as pd

    from cosmos_search.dayone import rank as R

    n = 40
    ranked = pd.DataFrame({"source_id": np.arange(n), "tier": [1] * 30 + [2] * 6 + [3] * 4,
                           "w": np.linspace(10, 1, n), "fov": False, "grey": False, "tension": False,
                           "bright": False})
    ranked.loc[[5, 6], "grey"] = True
    verdicts = pd.DataFrame({"source_id": np.arange(36), "demote": False, "eb_explained": False,
                             "eb_residual": False})
    verdicts.loc[[0, 1, 2], "demote"] = True
    verdicts.loc[[0, 1, 2, 3], "eb_explained"] = True
    verdicts.loc[[10], "eb_residual"] = True
    v1 = R.apply_residual(ranked, verdicts, flag_eb_residual=True)
    assert set(v1[v1.explained_by == "residual"].source_id) == {0, 1, 2}
    assert (v1[v1.source_id.isin([0, 1, 2])].tier == 3).all()
    assert v1[v1.tier < 3].source_id.iloc[:3].tolist() == [5, 6, 10]      # grey x2, then eb_residual
    filt = R.apply_residual(ranked, verdicts, rule="filter")
    assert set(filt[filt.explained_by == "ebfilter"].source_id) == {0, 1, 2, 3}
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        q = R.queue_v1(v1, n_vet=10, n_audit=8, n_controls=3, rng=rng)
        assert (q[q.role == "ranked"].tier < 3).all() and q.source_id.is_unique
        a = q[q.role == "audit"]
        assert ((a.inclusion > 0) & (a.inclusion <= 1)).all()
        seen |= set(a.source_id) & {0, 1, 2}
    assert seen == {0, 1, 2}                                               # demoted stars stay reachable


def test_active_stars_go_to_tier_2_and_into_the_controls():
    import pandas as pd

    from cosmos_search.dayone import rank as R

    n = 30
    ranked = pd.DataFrame({"source_id": np.arange(n), "tier": [1] * 25 + [3] * 5, "w": np.linspace(10, 1, n),
                           "fov": False, "grey": False, "tension": False, "bright": False,
                           "best_class_name": ["RS", "SOLAR_LIKE", "AGN"] + [None] * 27})
    verdicts = pd.DataFrame({"source_id": np.arange(25), "demote": False, "eb_explained": False,
                             "eb_residual": False, "residual_aperiodic": False})
    verdicts.loc[[0, 7], "residual_aperiodic"] = True
    v1 = R.apply_residual(ranked, verdicts)
    assert (v1.set_index("source_id").loc[[0, 1, 2], "tier"] == 2).all()
    assert v1[v1.tier == 1].source_id.iloc[0] == 7                         # aperiodic ranks, it does not bypass
    q = R.queue_v1(v1, 5, 2, 2, np.random.default_rng(1), n_active_controls=3)
    c = q[q.role == "control"].set_index("source_id")
    assert {0, 1, 2} <= set(c.index)
    assert c.loc[2, "control_label"] == "extragalactic" and c.loc[0, "control_label"] == "rotation_or_spots"


def _gaia_cadence(rng, n_transits=40, span=1000.0):
    """Gaia-like G epochs: transits at random times, each seen by both fields of view 106.5 min apart."""
    t0 = np.sort(rng.uniform(0, span, n_transits))
    return np.sort(np.r_[t0, t0 + 106.5 / 1440])


def test_activity_score_separates_spots_from_a_flat_star_with_dips():
    from cosmos_search.residual import activity as A

    rng = np.random.default_rng(21)
    t = _gaia_cadence(rng)
    noise = 0.003 * rng.standard_normal(len(t))
    spots = 1 + 0.02 * np.sin(2 * np.pi * t / 3.7) + noise
    flat = 1 + noise
    dips = flat.copy()
    dips[rng.choice(len(t), 6, replace=False)] *= 0.9                      # 10% dips on six epochs
    a_spots, a_flat, a_dips = (A.score(t, f)[0] for f in (spots, flat, dips))
    assert a_spots > A.THRESHOLD > a_flat
    assert a_dips < A.THRESHOLD and abs(a_dips - a_flat) < 0.15             # dips leave the ceiling alone
    assert np.isnan(A.score(np.arange(30.0), flat[:30])[0])                # no close pairs: no score


def test_activity_cal_rates_and_choice():
    import pandas as pd

    from cosmos_search.residual import activity as A

    real = pd.DataFrame({"kind": "real_class", "family": "RS", "activity": [0.9, 0.8, 0.1, 0.05],
                         "aperiodic": [True, True, True, False]})
    inj = pd.DataFrame({"kind": "anomaly", "family": "grey_dip", "activity": [0.1, 0.2, 0.5, 0.1],
                        "activity_host": [0.1, 0.2, 0.1, 0.1], "aperiodic": [True, True, False, True]})
    df = pd.concat([real, inj], ignore_index=True)
    r = A.rates(df, 0.35)
    assert r["capture"] == 2 / 3 and r["cost"] == 0 and r["flips"] == 0.25
    th, _ = A.choose(df)
    assert th == 0.6                                                       # J ties up to 0.60: the higher one
