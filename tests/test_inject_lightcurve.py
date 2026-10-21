import json

import numpy as np
import pytest

from cosmos_search.inject import lightcurve as lc


T = np.linspace(0, 2000, 4001)


def test_grey_dip_is_achromatic_and_local():
    dip = lc.Dip(t0=100, covering=0.2, ingress=1, egress=1, flat=2)
    for band in ("BP", "G", "RP", "TESS"):
        k = dip.factor(T, band)
        assert k.min() == pytest.approx(0.8)
        assert np.all(k[(T < 99) | (T > 105)] == 1.0)


def test_dust_dip_reddens_and_blue_dip_blues():
    dust = lc.Dip(t0=100, covering=0.5, tau_ref=0.5, beta=2.0, ingress=1, egress=1)
    blue = lc.Dip(t0=100, covering=0.5, tau_ref=0.5, beta=-2.0, ingress=1, egress=1)
    assert dust.factor(T, "BP").min() < dust.factor(T, "RP").min()
    assert blue.factor(T, "BP").min() > blue.factor(T, "RP").min()


def test_inject_scales_flux_and_errors():
    flux, err = np.full(T.size, 1000.0), np.full(T.size, 10.0)
    f2, e2 = lc.inject(T, flux, err, "G", lc.Step(t_step=1000, fraction=-0.1))
    assert f2[T < 1000].tolist() == flux[T < 1000].tolist()
    assert f2[T >= 1000] == pytest.approx(900.0)
    assert e2[T >= 1000] == pytest.approx(9.0)


def test_flare_brightens_more_in_blue():
    fl = lc.Flare(t0=500, amplitude=0.1)
    assert fl.factor(T, "BP").max() > fl.factor(T, "RP").max() > 1.0


def test_primes_spacing_is_not_periodic():
    times = lc.event_times(np.random.default_rng(0), "primes", 0, 2000, unit=1.0)
    gaps = np.diff(times)
    assert len(times) > 5
    assert np.allclose(gaps, lc.PRIMES[1:len(times)])


@pytest.mark.parametrize("family", lc.FAMILIES)
def test_every_family_samples_and_serialises(family):
    rng = np.random.default_rng(1)
    sig = lc.sample(rng, family, 0.0, 2000.0)
    for band in ("BP", "G", "RP", "TESS"):
        k = sig.factor(T, band)
        assert k.shape == T.shape and np.all(np.isfinite(k)) and np.all(k > 0)
    json.dumps(lc.truth(sig))


def test_unknown_band_is_an_error():
    with pytest.raises(ValueError):
        lc.Dip(t0=0, covering=0.1).factor(T, "K")


@pytest.mark.parametrize("span", [27.0, 2000.0])
@pytest.mark.parametrize("family", [f for f in lc.FAMILIES if f.startswith("dip_train_")])
def test_dip_trains_fit_short_and_long_spans(family, span):
    rng = np.random.default_rng(2)
    counts = [len(lc.sample(rng, family, 0.0, span).parts) for _ in range(200)]
    assert np.median(counts) >= 3
