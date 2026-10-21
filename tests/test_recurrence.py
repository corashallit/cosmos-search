import numpy as np

from cosmos_search.vetting import recurrence
from cosmos_search.vetting.candidate import LightCurve


def _nightly(rng, span=600.0):
    nights = np.arange(0.0, span)
    nights = nights[rng.random(len(nights)) < 0.7]
    return np.sort(nights + rng.uniform(0.1, 0.4, len(nights)))


def _lc(t, k, rng, band="ztf_r", s=0.01):
    return LightCurve(band, t, k * (1 + s * rng.standard_normal(len(t))), np.full(len(t), s))


def test_strict_period_aligns_every_eclipse():
    rng = np.random.default_rng(3)
    t = _nightly(rng)
    p, dur = 2.37, 0.3
    k = 1 - 0.15 * (np.abs(((t - 0.2) / p + 0.5) % 1 - 0.5) * p < dur / 2)
    r = recurrence.test([_lc(t, k, rng)])
    assert r["testable"] and r["n_events"] >= 10
    assert r["aligned_fraction"] == 1.0 and r["purity"] >= 0.8
    assert abs(r["period"] - p) / p < 1e-3


def test_aperiodic_dips_do_not_align():
    rng = np.random.default_rng(4)
    t = _nightly(rng)
    centres = np.cumsum(rng.exponential(12.0, 80))
    k = np.ones_like(t)
    for c in centres:
        k[np.abs(t - c) < 0.4] = 0.85
    r = recurrence.test([_lc(t, k, rng)])
    assert r["testable"] and r["n_events"] >= 10
    assert r["aligned_fraction"] < 0.5


def test_gaia_colour_bands_are_not_counted_twice():
    rng = np.random.default_rng(5)
    t = _nightly(rng, 200)
    k = np.where(np.isin(np.arange(len(t)), [10, 50, 90]), 0.8, 1.0)
    lcs = [_lc(t, k, rng, b) for b in ("G", "BP", "RP")]
    assert recurrence.test(lcs)["n_events"] == 3
