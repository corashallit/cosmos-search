import math

import numpy as np
import pytest

from cosmos_search.ebfilter import decide as D
from cosmos_search.ebfilter import search as S
from cosmos_search.inject import lightcurve as lcinj
from cosmos_search.inject.classes.eclipsing_binary import EclipsingBinary, from_truth, sample, truth
from cosmos_search.vetting.candidate import LightCurve

SUN_AT_100PC = {"phot_g_mean_mag": 9.7, "bp_rp": 0.82, "parallax": 10.0, "parallax_error": 0.02}


def _cadence(rng, n_nights=400, span=1200.0):
    """ZTF-like: one or two epochs a night at a random hour, on random nights."""
    nights = np.sort(rng.choice(int(span), n_nights, replace=False)).astype(float)
    t = np.concatenate([nights + rng.uniform(0.15, 0.35, n_nights), nights[::3] + rng.uniform(0.36, 0.45)])
    return np.sort(t)


def _lc(t, k, rng, band="ztf_r", noise=0.005):
    f = k * (1 + noise * rng.standard_normal(len(t)))
    return LightCurve(band, t, f, np.full(len(t), noise))


def _eb(period=2.7, t_peri=0.3):
    return EclipsingBinary(period=period, t_peri=t_peri, ecc=0.0, omega=math.pi / 2, incl=math.radians(89.0),
                           r1=1.0, r2=0.8, m1=1.0, m2=0.8, t1=5800.0, t2=5000.0)


def test_eb_model_depth_duration_and_truth_roundtrip():
    eb = EclipsingBinary(period=2.0, t_peri=0.0, ecc=0.0, omega=math.pi / 2, incl=math.pi / 2, r1=1, r2=1, m1=1,
                         m2=1, t1=5772, t2=5772)
    t = np.linspace(-0.5, 0.5, 100001)
    k = eb.factor(t, "G")
    assert abs(k.min() - 0.5) < 0.01                         # twins, central: half the light
    dur = np.ptp(t[k < 0.99])
    assert dur <= D.max_duration_days(2.0, 1.0, 2.0, e_max=0.0) + 1e-3
    assert dur > 0.8 * D.max_duration_days(2.0, 1.0, 2.0, e_max=0.0)
    assert from_truth(truth(eb)) == eb
    rng = np.random.default_rng(1)
    for _ in range(20):
        e = sample(rng, SUN_AT_100PC, 0.0, 1000.0)
        tt = np.linspace(0, e.period, 50001) + e.t_peri
        assert e.factor(tt, "G").min() < 0.999             # every draw eclipses


def test_null_bits_counts_runs():
    lab = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0])
    # K = 3 dip epochs in E = 2 runs among N = 10: gamma(3) = 5 bits + log2(3) + log2 C(8, 2) + log2 C(2, 1)
    expect = 5 + math.log2(3) + math.log2(28) + 1
    assert abs(S.null_bits(lab) - expect) < 1e-9


def test_finds_eclipsing_binary_and_explains_its_dip():
    rng = np.random.default_rng(2)
    t = _cadence(rng)
    eb = _eb()
    lc = _lc(t, eb.factor(t, "ztf_r"), rng)
    lab_t = S.labels([lc])[0][S.labels([lc])[1] == 1]
    target = float(lab_t[0])
    v = D.decide([lc], SUN_AT_100PC, target)
    assert v["explained"] and v["target_covered"]
    assert v["delta_bits"] > 20
    ratio = v["period"] / eb.period
    assert min(abs(ratio - 1), abs(ratio - 0.5)) < 1e-3     # the period, or half of it for similar eclipses


def test_random_dips_are_not_explained():
    rng = np.random.default_rng(3)
    t = _cadence(rng)
    hits = 0
    for _ in range(4):
        sig = lcinj.sample(rng, "dip_train_poisson", float(t[0]), float(t[-1]))
        lc = _lc(t, sig.factor(t, "ztf_r"), rng)
        v = D.decide([lc], SUN_AT_100PC)
        hits += v["explained"]
        assert v["delta_bits"] < D.DELTA_BITS
    assert hits == 0


def test_null_bound_holds_on_shuffled_dips():
    """P(delta >= k) <= 2^-k when dip runs sit at random among the epochs: shuffle an EB's labels."""
    rng = np.random.default_rng(4)
    t = _cadence(rng, n_nights=150, span=400.0)
    lc = _lc(t, _eb(period=1.9).factor(t, "ztf_r"), rng)
    tt, lab, bi, z, names, frac = S.labels([lc])
    deltas = []
    for _ in range(12):
        perm = lab.copy()
        rng.shuffle(perm)
        deltas.append(S.search(tt, perm, bi, names)["delta_bits"])
    assert np.mean(np.array(deltas) >= 3) <= 0.125 + 0.25   # bound 2^-3, with room for 12 draws
    assert max(deltas) < 10


def test_extra_dip_on_an_eb_is_not_explained():
    rng = np.random.default_rng(5)
    t = _cadence(rng)
    eb = _eb()
    k = eb.factor(t, "ztf_r")
    # A deep dip at a phase between the eclipses.
    ph = ((t - eb.t_peri) / eb.period) % 1
    j = int(np.flatnonzero((np.abs(ph - 0.0) > 0.15) & (np.abs(ph - 0.5) > 0.15))[50])
    k[j] *= 0.5
    lc = _lc(t, k, rng)
    v = D.decide([lc], SUN_AT_100PC, float(t[j]))
    assert not v["explained"]
    assert v["delta_bits"] > 20                             # the binary itself is still found


def test_strict_period_with_impossible_duration_is_flagged():
    rng = np.random.default_rng(6)
    t = _cadence(rng)
    # 30 d period, 4 d flat-bottomed dips around a Sun: no stellar companion eclipses that long at 30 d.
    sig = lcinj.Composite([lcinj.Dip(t0=float(t0), covering=0.2, ingress=0.2, egress=0.2, flat=3.6)
                           for t0 in np.arange(5.0, t[-1], 30.0)])
    lc = _lc(t, sig.factor(t, "ztf_r"), rng)
    tt, lab = S.labels([lc])[:2]
    v = D.decide([lc], SUN_AT_100PC, float(tt[lab == 1][0]))
    assert v["delta_bits"] > D.DELTA_BITS
    assert not v["explained"] and v["periodic_unphysical"]


def test_max_duration_grows_with_period_and_eccentricity():
    a = D.max_duration_days(1.0, 1.0, 1.0, e_max=0.0)
    assert D.max_duration_days(10.0, 1.0, 1.0, e_max=0.0) > a
    assert D.max_duration_days(1.0, 1.0, 1.0, e_max=0.9) == pytest.approx(a * math.sqrt(1.9 / 0.1))


def test_cli_verdicts_on_candidate_dicts():
    from cosmos_search.ebfilter.__main__ import verdicts

    rng = np.random.default_rng(7)
    t = _cadence(rng)
    eb = _eb()
    tg = np.sort(rng.uniform(t[0], t[-1], 60))
    tg = np.r_[tg, t[np.argmin(eb.factor(t, "G"))] + 1e-3]       # Gaia caught one eclipse
    g = _lc(np.sort(tg), eb.factor(np.sort(tg), "G"), rng, band="G")
    z = _lc(t, eb.factor(t, "ztf_r"), rng)
    cand = {"candidate_id": "gaia_dr3:42", "source": SUN_AT_100PC,
            "lightcurves": [dict(g.to_dict(), meta={"flags": [0] * len(g.time)}), z.to_dict()]}
    out = verdicts([cand])
    assert out.source_id.iloc[0] == 42
    assert bool(out.explained.iloc[0]) and bool(out.target_covered.iloc[0])
