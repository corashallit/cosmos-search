import math

import numpy as np

from cosmos_search.vetting import context_tests as ct
from cosmos_search.vetting.candidate import LightCurve


def _nightly(rng, span=600.0):
    nights = np.arange(0.0, span)
    nights = nights[rng.random(len(nights)) < 0.7]
    return np.sort(nights + rng.uniform(0.1, 0.4, len(nights)))


def _lc(t, k, rng, band="ztf_r", s=0.01):
    return LightCurve(band, t, k * (1 + s * rng.standard_normal(len(t))), np.full(len(t), s))


def _eclipses(t, p=2.37, t0=0.2, dur=0.3):
    return 1 - 0.15 * (np.abs(((t - t0) / p + 0.5) % 1 - 0.5) * p < dur / 2)


def _random_dips(t, rng):
    k = np.ones_like(t)
    for c in np.cumsum(rng.exponential(12.0, 80)):
        k[np.abs(t - c) < 0.4] = 0.85
    return k


def test_strict_period_is_significant_for_eclipses_and_not_for_random_dips():
    rng = np.random.default_rng(3)
    t = _nightly(rng)
    eb = ct.strict_period([_lc(t, _eclipses(t), rng)])
    assert eb["delta_bits"] > 20 and eb["p_random_upper"] < 1e-6
    assert abs(eb["period"] - 2.37) / 2.37 < 1e-2
    rnd = ct.strict_period([_lc(t, _random_dips(t, rng), rng)])
    assert rnd["delta_bits"] < eb["delta_bits"] / 2


def test_known_variable_skips_gaia_named_and_poor_matches():
    assert ct.known_variable({"vsx_Name": "WY Cen", "vsx_Type": "EA", "vsx_Period": 76.5}) == \
        {"type": "EA", "period_d": 76.5}
    assert ct.known_variable({"vsx_Name": "Gaia DR3 123", "vsx_Type": "EA", "vsx_Period": 1.0}) is None
    assert ct.known_variable({"vsx_Name": "V1 Cen", "vsx_Type": "EA", "vsx_Period": 1.0, "vsx_p_cal": 0.2}) is None
    assert ct.known_variable({"vsx_Name": float("nan")}) is None
    assert ct.known_variable({"vsx_Name": "V2 Cen", "vsx_Type": None, "vsx_Period": float("nan")}) == \
        {"type": "VAR", "period_d": None}


def test_catalogue_ephemeris_folds_eclipses_into_windows_and_random_dips_out():
    rng = np.random.default_rng(5)
    t = _nightly(rng)
    sun = {"phot_g_mean_mag": 10.0, "bp_rp": 0.82, "parallax": 10.0, "parallax_error": 0.1}
    eb = ct.catalogue_ephemeris([_lc(t, _eclipses(t), rng)], 2.37, sun)
    assert eb["n_events"] >= 10 and eb["n_events_on_ephemeris"] == eb["n_events"]
    assert eb["chance_fraction"] < 0.35
    rnd = ct.catalogue_ephemeris([_lc(t, _random_dips(t, rng), rng)], 2.37, sun)
    assert rnd["n_events_on_ephemeris"] < 0.5 * rnd["n_events"]
    assert ct.catalogue_ephemeris([_lc(t, _eclipses(t), rng)], None) is None


def _orbit(P=40.0, e=0.3, w=60.0, tp=-389.0 + 5.0, **kw):
    return {"nss_solution_type": "SB1", "period": P, "eccentricity": e, "arg_periastron": w, "t_periastron": tp,
            "period_error": 0.0, "t_periastron_error": 0.0, **kw}


def test_conjunctions_are_where_true_anomaly_plus_omega_is_90_or_270():
    orb = _orbit()
    P, e, w = orb["period"], orb["eccentricity"], math.radians(orb["arg_periastron"])
    tp = orb["t_periastron"] + ct.GAIA_REF_BTJD
    for tc, _ in ct.conjunctions(orb, 0.0, 200.0):
        M = 2 * math.pi * (tc - tp) / P
        E = M
        for _ in range(50):
            E = M + e * math.sin(E)
        nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))
        x = math.degrees(nu + w) % 180.0
        assert min(abs(x - 90.0), abs(x - 270.0 + 180.0)) < 1e-6


def test_orbit_conjunction_counts_dips_at_conjunction_and_ignores_photometric_solutions():
    rng = np.random.default_rng(6)
    t = np.sort(rng.uniform(0.0, 600.0, 900))
    orb = _orbit()
    tc = np.array([c[0] for c in ct.conjunctions(orb, 0.0, 600.0)])
    k_at = np.where(np.min(np.abs(t[:, None] - tc[None, :]), axis=1) < 0.2, 0.8, 1.0)
    r = ct.orbit_conjunction([_lc(t, k_at, rng)], orb)
    assert r["n_events"] >= 5 and r["n_events_at_conjunction"] == r["n_events"]
    mid = tc[:-1] + np.diff(tc) / 2
    k_off = np.where(np.min(np.abs(t[:, None] - mid[None, :]), axis=1) < 0.2, 0.8, 1.0)
    r = ct.orbit_conjunction([_lc(t, k_off, rng)], orb)
    assert r["n_events_at_conjunction"] <= 1
    assert ct.orbit_conjunction([_lc(t, k_at, rng)], {**orb, "nss_solution_type": "EclipsingBinary"}) is None
    assert ct.orbit_conjunction([_lc(t, k_at, rng)], {**orb, "t_periastron": float("nan")}) is None
    circ = ct.orbit_conjunction([_lc(t, k_at, rng)], {**orb, "eccentricity": None, "arg_periastron": None})
    assert circ is not None
