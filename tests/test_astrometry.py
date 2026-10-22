"""Astrometric arm: geometry against real DR4 pre-release epochs, orbit recovery, plausibility, simulator + cascade."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cosmos_search.astrometry import (
    detect,
    dips,
    fit,
    geometry,
    kepler_bound,
    models,
    physics,
    simulate,
)

DATA = Path(__file__).parent / "fixtures" / "dr4_prerelease_transits.csv"  # Gaia DR4 pre-release (ESA, CC BY-SA 3.0 IGO)
BH3 = 4318465066420528000
SINGLE = 4181040337841125632  # HD 183633, parallax ~1.0 mas


@pytest.fixture(scope="module")
def pre():
    return pd.read_csv(DATA)


def _source(g):
    k = max(1.0, np.sqrt(g.chi2_intra.sum() / g.dof_intra.sum()))
    return fit.Source(g.t, g.psi, g.f_al, g.w, np.sqrt((g.sigma_w * k) ** 2 + 0.02 ** 2))


def test_parallax_factor_matches_dr4(pre):
    xyz = geometry.observer_xyz(pre.btjd.to_numpy())
    f = geometry.parallax_factor_al(pre.ra0.to_numpy(), pre.dec0.to_numpy(), pre.psi.to_numpy(), xyz)
    assert np.abs(f - pre.f_al.to_numpy()).max() < 3e-3


def test_single_star_parallax(pre):
    src = _source(pre[pre.source_id == SINGLE])
    f5 = src.five()
    assert abs(f5.x[2] - 1.0) < 0.05
    assert f5.chi2 / (src.n - 5) < 1.5


def test_bh3_orbit_and_mass_function(pre):
    src = _source(pre[pre.source_id == BH3])
    k1 = src.kepler()
    p = k1.params
    assert abs(p["period_yr"] * 365.25 / 4253 - 1) < 0.05  # Gaia Collaboration 2024: 4253 +- 98 d
    assert abs(p["ecc"] - 0.73) < 0.05
    fa = physics.mass_function(p["a0_mas"], k1.x[2], p["period_yr"])
    assert 25 < fa < 40  # M_BH^3 / M_tot^2 ~ 31 Msun
    assert k1.chi2 / (src.n - k1.k) < 2


def test_kepler_verdicts():
    # a 0.6-d, e = 0.97 'orbit' puts periastron inside a Sun-like star even at 100 Msun
    assert physics.kepler_verdict(0.6 / 365.25, 0.97, 0.1, 1.0, 0.01, 1.0)[1] == "rp_inside_star"
    assert physics.kepler_verdict(1.0, 0.1, 30.0, 1.0, 0.01, 1.0)[1] == "over_mass_cap"
    assert physics.kepler_verdict(1.0, 0.1, 1.0, 1.0, 0.01, 1.0)[1] == "ok"
    assert physics.kepler_verdict(1.0, 0.1, 1.0, 0.03, 0.01, 1.0)[1] == "unknown"


def test_energy_bound():
    a = physics.acceleration_ms2(0.04, 100.0)  # a 5-sigma DR4 acceleration of a G = 12 star at 10 pc
    assert physics.luminosity_for_acceleration(a) > 1e4


def test_kepler_bound_flags_aliases():
    df = pd.DataFrame({
        "source_id": [1, 2], "nss_solution_type": ["Orbital", "OrbitalAlternative"],
        "period": [185.6, 15.78], "a_thiele_innes": [1.0, 1.0], "b_thiele_innes": [2.0, 0.5],
        "f_thiele_innes": [-2.0, -0.5], "g_thiele_innes": [1.0, 1.0], "parallax": [2.1, 0.5],
        "parallax_error": [0.02, 0.05], "goodness_of_fit": [0.3, 1.0], "phot_g_mean_mag": [13.8, 15.0],
        "ipd_frac_multi_peak": [0, 0], "ipd_frac_odd_win": [0, 0], "mass_flame": [0.93, np.nan]})
    o = kepler_bound.orbits(df)
    assert o.loc[1, "prec_alias"] and o.loc[1, "alt_short"]
    assert o.loc[0, "year_alias"] and not o.loc[0, "prec_alias"]
    assert (o["f_A"] > 0).all()


def test_neighbour_template_is_odd_in_scan_direction():
    psi = np.linspace(0, 2 * np.pi, 50)
    a = models.neighbour(np.zeros(50), psi, 100.0, 0.3, 0.5)[:, 0]
    b = models.neighbour(np.zeros(50), psi + np.pi, 100.0, 0.3, 0.5)[:, 0]
    assert np.allclose(a, -b)


def test_dip_localisation_on_target():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(-2.5, 2.5, 90))
    psi = rng.uniform(0, 2 * np.pi, 90)
    f = rng.normal(0, 0.7, 90)
    flux = np.where(np.abs(t) < 0.03, 0.7, 1.0) * (1 + rng.normal(0, 0.001, 90))
    sig = np.full(90, 0.05)
    w = 2.0 * f + rng.normal(0, 0.05, 90)
    r = dips.localise(t, psi, f, w, sig, flux)
    assert r["verdict"] == "on_target"
    # same dips on a neighbour 20 mas away carrying a third of the light
    x = 20.0 * (0.5 * np.where(np.abs(t) < 0.03, 0.4, 1.0)) / (1 + 0.5 * np.where(np.abs(t) < 0.03, 0.4, 1.0))
    flux2 = 1 + 0.5 * np.where(np.abs(t) < 0.03, 0.4, 1.0)
    w2 = 2.0 * f + x * np.sin(psi) + rng.normal(0, 0.05, 90)
    assert dips.localise(t, psi, f, w2, sig, flux2)["verdict"] == "neighbour"


def test_simulate_and_cascade_runs():
    rng = np.random.default_rng(1)
    gost = pd.DataFrame({"source_id": 1, "t": np.sort(rng.uniform(-90, 1900, 90)),
                         "fov": "FoVP", "ccd_row": 1, "scan_angle": rng.uniform(-np.pi, np.pi, 90)})
    stars = pd.DataFrame({"source_id": [1], "ra": [10.0], "dec": [-20.0], "phot_g_mean_mag": [14.0],
                          "parallax": [2.0]})
    cad = simulate.Cadence(gost, stars)
    data, truth = simulate.simulate(rng, cad, "single")
    out = detect.cascade(data)
    assert out["best"] in ("5p", "5p+N", "7p", "K1", "lens", "nb")
    assert np.isfinite(out["dbic_cp"])
    data, truth = simulate.simulate(rng, cad, "square", amp=40.0)
    out = detect.cascade(data)
    assert max(out["dbic_cp"], out["dbic_implausible"]) > 20


def test_fov_pair_test_sees_a_fov_jump_but_not_one_sky_flip():
    rng = np.random.default_rng(2)
    t = np.sort(np.concatenate([u := rng.uniform(-2.5, 2.5, 40), u + 106.5 / 1440 / 365.25]))
    fov = np.tile([1, 2], 40)
    sig = np.ones(80)
    r = rng.normal(0, 1, 80)
    flip = r.copy()
    flip[41] += 30.0  # an abrupt change on the sky between the two transits of one pair
    jump = r + 3.0 * ((fov == 2) & (t > 0))
    assert detect.fov_pair_z(t, fov, flip, sig) < 4
    assert detect.fov_pair_z(t, fov, jump, sig) > 6


def test_vim_only_when_the_flux_varies():
    rng = np.random.default_rng(3)
    t = np.linspace(-2.5, 2.5, 80)
    psi = rng.uniform(0, 2 * np.pi, 80)
    zero = np.zeros(80)
    noise = 1 + rng.normal(0, 0.003, 80)
    assert not fit.Source(t, psi, zero, zero, np.ones(80), noise, 0.003).fixed
    var = noise * (1 + 0.1 * np.sin(t * 7))
    assert fit.Source(t, psi, zero, zero, np.ones(80), var, 0.003).fixed
