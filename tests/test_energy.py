import numpy as np
import pandas as pd
import pytest

from cosmos_search.energy import budget, dimming, inject, run
from cosmos_search.energy.__main__ import delta_from_ranked
from cosmos_search.energy.photometry import design, from_context
from cosmos_search.energy.photosphere import Locus

# Photospheric colours of the synthetic stars: G-W1, K-W1, W1-W2, W1-W3, W3-W4.
COL = {"G1": 1.6, "K1": 0.05, "12": -0.03, "13": 0.02, "34": 0.05}


def stars(n, rng, w1=(8.3, 13.0), w3_bright=False):
    w1m = rng.uniform(*w1, n)
    if w3_bright:
        w1m = rng.uniform(4.0, 5.4, n) + COL["13"]
    noise = lambda s: rng.normal(0, s, n)
    df = pd.DataFrame({
        "source_id": rng.integers(1, 2**62, n), "bp_rp": rng.uniform(0.7, 1.1, n), "parallax": np.full(n, 5.0),
        "parallax_error": np.full(n, 0.05), "ruwe": np.full(n, 1.0),
        "phot_g_mean_mag": w1m + COL["G1"] + noise(0.02), "phot_g_mean_flux_over_error": np.full(n, 1000.0),
        "phot_g_n_obs": np.full(n, 300),
        "ks_m": w1m + COL["K1"] + noise(0.02), "ks_msigcom": np.full(n, 0.02),
        "w1mpro": w1m, "w1mpro_error": np.full(n, 0.02),
        "w2mpro": w1m - COL["12"] + noise(0.02), "w2mpro_error": np.full(n, 0.02),
        "w3mpro": w1m - COL["13"] + noise(0.03), "w3mpro_error": np.full(n, 0.03),
        "w_ph_qual": "AAAA", "w_cc_flags": "0000", "w_ext_flag": 0, "w_var_flag": "nnnn", "w_mates": 0,
        "t_ph_qual": "AAA", "t_mates": 0,
    })
    df["w4mpro"] = df.w3mpro - COL["34"] + noise(0.05)
    df["w4mpro_error"] = 0.05
    return df


@pytest.fixture(scope="module")
def locus():
    rng = np.random.default_rng(1)
    gen = stars(3000, rng)
    mid = stars(2000, rng, w1=(8.3, 9.7))
    bright = stars(2000, rng, w3_bright=True)
    return Locus(k=32).fit({"general": gen, "mid": mid, "bright": bright})


def test_excess_per_f_matches_blackbody_numbers():
    e = budget.excess_per_f([5800.0], [150.0, 300.0], 0.0)[0]
    w3, w4 = budget.BANDS.index("W3"), budget.BANDS.index("W4")
    # Monochromatic estimates: W4 at 150 K ~3400 per unit f, W3 at 300 K ~390; band averaging adds ~10%.
    assert 3000 < e[0, w4] < 4500 and 350 < e[1, w3] < 500
    assert e[0, budget.BANDS.index("G")] < 1e-6


def test_thin_floor_is_tiny_and_scales_with_t_squared():
    f = budget.f_thin(np.array([0.01]), np.array([300.0, 600.0]), np.array([5800.0]))[0]
    assert f[0] == pytest.approx(np.pi / 8 * 0.01 * (300 / 5800) ** 2)
    assert f[1] / f[0] == pytest.approx(4.0)


def test_kinematic_floor_is_cold_for_gaia_brackets():
    assert budget.t_floor_kinematic(0.5, 0.1, 1.0, 1.0, 5772) < 50
    assert np.isnan(budget.t_floor_kinematic(0.5, 0.0, 1.0, 1.0, 5772))


def test_mean_deficit_lower_bound():
    t = np.repeat(np.arange(40) * 30.0, 2) + np.tile([0, 0.074], 40)
    f = np.ones_like(t) + np.random.default_rng(0).normal(0, 0.002, len(t))
    f[10:12] = 0.8
    d = dimming.mean_deficit(t, f)
    assert d["k_groups"] == 1 and d["n_groups"] == 40
    assert d["delta"] == pytest.approx(0.2 / 40, rel=0.05)
    assert 0 < d["delta_lo"] < d["delta"] / 10
    assert dimming.mean_deficit(t, np.ones_like(t))["delta_lo"] == 0.0


def test_photosphere_only_vs_injected_excess(locus):
    rng = np.random.default_rng(2)
    host = stars(60, rng, w1=(9.0, 11.0))
    res, des = run.evaluate(host, locus)
    assert (res.g5_z_excess < 5).mean() > 0.9
    comps = inject.draw("rcb_shell", len(host), rng)
    res2, _ = run.evaluate(inject.apply(host, des, comps), locus)
    assert (res2.g5_z_excess >= 5).mean() > 0.9
    assert ((res2.g5_t_best > 300) & (res2.g5_t_best < 1800)).mean() > 0.8


def test_contaminated_bands_cannot_make_an_excess(locus):
    rng = np.random.default_rng(3)
    host = stars(30, rng, w1=(9.0, 11.0))
    _, des = run.evaluate(host, locus)
    # A 300 K emitter shows only in WISE (a hot one would also raise K over G, which is a real excess).
    inj = inject.apply(host, des, [[(300.0, 0.05, 0.0)] for _ in range(len(host))])
    inj["w_cc_flags"] = "DDDD"
    res, d = run.evaluate(inj, locus)
    for b in ("W1", "W2", "W3", "W4"):     # bound, or (if the injection saturated it) not used at all
        assert not (d[f"use_{b}"] & ~d[f"bound_{b}"]).any()
    assert (res.g5_z_excess < 5).all()


def test_shell_deficit_found_without_ir_and_not_with_it(locus):
    rng = np.random.default_rng(4)
    host = stars(40, rng, w1=(9.0, 10.0))
    delta = np.full(len(host), 0.1)
    res, des = run.evaluate(host, locus, delta=delta)
    assert res.g5_deficit_shell.mean() > 0.8          # no IR: a 10% shell at 150-1500 K would have shown
    assert not res.g5_deficit_thin.any()               # an orbit-confined occulter never shows
    comps = [[(400.0, 0.1, 0.0)] for _ in range(len(host))]
    res2, _ = run.evaluate(inject.apply(host, des, comps), locus, delta=delta)
    assert not res2.g5_deficit_shell.any()


def test_upper_limits_bound_the_fit(locus):
    rng = np.random.default_rng(5)
    host = stars(20, rng, w1=(9.0, 10.0))
    host["w4mpro"] = host.w3mpro - 0.05 - 1.0     # W4 'U': 95% limit 1 mag brighter than the photosphere
    host["w4mpro_error"] = np.nan
    host["w_ph_qual"] = "AAAU"
    res, d = run.evaluate(host, locus)
    assert (d.d_W4 == 0).all() and d.use_W4.all()
    assert (res.g5_z_excess < 5).all()


def test_from_context_renames_and_drops_weak_matches():
    wide = pd.DataFrame({"source_id": [1, 2], "2mass_k_m": [10.0, 11.0], "2mass_k_msigcom": [0.02, 0.02],
                         "2mass_ph_qual": ["AAA", "AAA"], "2mass_p_cal": [0.99, 0.5],
                         "allwise_w1mpro": [9.9, 10.9], "allwise_w1sigmpro": [0.02, 0.02],
                         "allwise_ph_qual": ["AUUU", "AUUU"], "allwise_p_cal": [0.99, 0.99],
                         "allwise_multiplicity": [1.0, 2.0]})
    df = from_context(wide)
    assert df.loc[0, "ks_m"] == 10.0 and np.isnan(df.loc[1, "ks_m"])
    assert list(df.w_mates) == [0, 1] and df.loc[0, "w1mpro"] == 9.9


def test_delta_from_ranked():
    ranked = pd.DataFrame({"source_id": [1, 2], "depth_frac": [-0.2, -0.4], "n_episodes": [1, 4]})
    phot = pd.DataFrame({"source_id": [1, 2], "matched_transits": [40, 40]})
    d = delta_from_ranked(ranked, phot)
    assert 0 < d[1] < 0.1 * 1 / 20 and d[2] > d[1]


def test_ir_holdout_hides_the_energy_budget():
    from cosmos_search.vetting.candidate import Candidate
    from cosmos_search.vetting.packet import build

    t = np.arange(60) * 10.0
    lc = {"band": "G", "time": t, "flux": np.ones(60), "flux_err": np.full(60, 0.01)}
    ctx = {"ir_excess_sigma": 7.0, "W3": 9.0, "waste_heat_max_fraction_300K": 0.01, "tess_contamination": 0.1}
    cand = Candidate.from_dict({"candidate_id": "gaia_dr3:1", "arm": "gaia", "source": {}, "lightcurves": [lc],
                                "context": ctx})
    packet, key = build(cand, holdout=("ir",))
    assert set(packet["context"]) == {"tess_contamination"}
    assert key["truths"]["ir_excess"] is True
