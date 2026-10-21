import numpy as np
import pytest

from cosmos_search.dimming.detect import find_events, scan
from cosmos_search.dimming.features import edge_kinematics, event_diagnostics, extract
from cosmos_search.dimming.lightcurve import LightCurve
from cosmos_search.dimming.population import AnomalyModel, common_mode_epochs, in_common_mode
from cosmos_search.inject.classes import dimming as fam
from cosmos_search.inject.lightcurve import Dip


def tess_like(rng, sigma=1e-3, span=27.0, cadence_s=200.0, gap=(13.0, 14.0)):
    t = np.arange(0, span, cadence_s / 86400) + 3000.0
    t = t[(t - 3000 < gap[0]) | (t - 3000 > gap[1])]
    f = 1 + rng.normal(0, sigma, len(t))
    return LightCurve(t, f, np.full(len(t), sigma), band="TESS", meta={"radius": 1.0, "mass": 1.0})


def test_finds_a_day_long_dip_and_not_noise():
    rng = np.random.default_rng(1)
    lc = tess_like(rng)
    assert find_events(lc, threshold=5.0) == []
    dip = Dip(t0=3007.0, covering=0.01, ingress=0.3, egress=0.6, shape="gaussian")
    k = dip.factor(lc.t, "TESS")
    ev = find_events(lc.with_flux(lc.flux * k), threshold=5.0)
    assert len(ev) == 1
    assert abs(ev[0].t_peak - 3007.0) < 0.3
    assert ev[0].depth > 0.004


def test_mirror_turns_dips_into_bursts():
    rng = np.random.default_rng(2)
    lc = tess_like(rng)
    k = Dip(t0=3020.0, covering=0.02, ingress=0.1, egress=0.1, flat=0.2).factor(lc.t, "TESS")
    dipped = lc.with_flux(lc.flux * k)
    tr = scan(dipped.inverted())
    assert find_events(dipped.inverted(), tr, sign=-1) == []
    assert len(find_events(dipped.inverted(), tr, sign=+1)) == 1


def test_baseline_bridges_long_dip():
    # One-sided clipping keeps a 3-day 5% dip from being absorbed into the trend.
    rng = np.random.default_rng(3)
    lc = tess_like(rng)
    k = Dip(t0=3004.0, covering=0.05, ingress=0.5, egress=0.5, flat=2.0).factor(lc.t, "TESS")
    ev = find_events(lc.with_flux(lc.flux * k), threshold=5.0)
    assert ev and max(e.depth for e in ev) > 0.03


def test_edge_kinematics_flags_unresolved_steps_only():
    rng = np.random.default_rng(4)
    lc = tess_like(rng)
    smooth = lc.with_flux(lc.flux * Dip(t0=3006.0, covering=0.1, ingress=0.2, egress=0.2, flat=0.2).factor(lc.t, "TESS"))
    e = max(find_events(smooth), key=lambda x: x.snr)
    assert not edge_kinematics(smooth, e)["fast_edge"]
    jump = lc.flux.copy()
    sel = (lc.t > 3006.0) & (lc.t < 3006.5)
    jump[sel] *= 0.9
    stepped = lc.with_flux(jump)
    e = max(find_events(stepped), key=lambda x: x.snr)
    k = edge_kinematics(stepped, e)
    assert k["fast_edge"] and k["kinematic_limit"] < 0.05


def test_background_coupled_dip_is_explained_injected_is_not():
    rng = np.random.default_rng(5)
    lc = tess_like(rng)
    bkg = rng.normal(0, 1, len(lc.t)) + 0.3 * np.sin(lc.t)
    sel = (lc.t > 3008) & (lc.t < 3008.5)
    bkg[sel] += 20
    coupled = lc.with_flux(lc.flux - 0.0005 * (bkg - bkg.mean()))
    coupled.aux = {"bkg": bkg}
    e = max(find_events(coupled), key=lambda x: x.snr)
    assert event_diagnostics(coupled, e)["explained_total"] > 0.7

    clean_bkg = rng.normal(0, 1, len(lc.t))
    injected = lc.with_flux(lc.flux * Dip(t0=3008.0, covering=0.01, ingress=0.1, egress=0.1, flat=0.3).factor(lc.t, "TESS"))
    injected.aux = {"bkg": clean_bkg}
    e = max(find_events(injected), key=lambda x: x.snr)
    assert abs(event_diagnostics(injected, e)["explained_total"]) < 0.3


def test_extract_runs_and_reports_shape_features():
    rng = np.random.default_rng(6)
    lc = tess_like(rng)
    f, dips, bursts = extract(lc)
    for key in ("min_stat_3h", "log_beta_28.8h", "qskew_10", "Q", "bimodality", "dip_snr"):
        assert key in f
    assert f["n_dips"] == 0


def test_common_mode_flags_shared_epochs():
    events = [("g", i, 3005.0, 3005.2) for i in range(15)]
    events += [("g", i, 3000.0 + i, 3000.1 + i) for i in range(15, 25)]
    cm = common_mode_epochs(events, {"g": 100})
    assert in_common_mode(cm, "g", 3005.05, 3005.1) > 0.1
    assert in_common_mode(cm, "g", 3016.0, 3016.05) == 0.0


def test_anomaly_model_ranks_outlier_above_population():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(500, 5))
    strata = np.zeros(500, int)
    m = AnomalyModel(list("abcde")).fit(X, strata)
    s = m.score(np.vstack([X[:5], np.full((1, 5), 8.0)]), np.zeros(6, int))
    assert s[-1] > 1.0 > np.median(s[:5])


@pytest.mark.parametrize("family", fam.FAMILIES)
@pytest.mark.parametrize("span", [27.0, 2000.0])
def test_injection_families_sample_on_both_baselines(family, span):
    rng = np.random.default_rng(8)
    t = np.linspace(3000, 3000 + span, 4000)
    for _ in range(5):
        sig = fam.sample(rng, family, t[0], t[-1])
        k = sig.factor(t, "TESS")
        assert np.all(np.isfinite(k)) and np.all(k >= 0)
        for a, b in fam.windows(sig, t[0], t[-1]):
            assert b >= t[0] and a <= t[-1]


def test_levels_factor_is_piecewise():
    lv = fam.Levels([0.0, 10.0, 20.0], [1.0, 0.98, 0.99])
    assert np.allclose(lv.factor(np.array([5.0, 15.0, 25.0]), "TESS"), [1.0, 0.98, 0.99])
