import numpy as np
import pandas as pd

from cosmos_search.dayone import rank, vet
from cosmos_search.gaia.epoch import gaia_time_to_btjd


def _eb(ref_btjd=100.0, freq=0.1, dur=0.2, depth=0.5):
    # reference_time is on Gaia's clock; pick it so that it maps to ref_btjd.
    gaia_t = ref_btjd + 2457000.0 - 2455197.5
    gaia_t += ref_btjd - gaia_time_to_btjd(gaia_t)
    return pd.DataFrame({"reference_time": [gaia_t], "frequency": [freq], "derived_primary_ecl_phase": [0.0],
                         "derived_primary_ecl_duration": [dur], "derived_primary_ecl_depth": [depth],
                         "derived_secondary_ecl_phase": [np.nan], "derived_secondary_ecl_duration": [np.nan],
                         "derived_secondary_ecl_depth": [np.nan]})


def test_in_eclipse_uses_the_core_of_the_eclipse_and_its_depth():
    eb = _eb()
    assert rank.in_eclipse([100.0 + 10 * 3], eb)[0]            # exactly at phase 0, three cycles on
    assert rank.in_eclipse([100.0 + 0.04 / 0.1], eb)[0]        # |dphi| = 0.04 <= 0.25 * 0.2
    assert not rank.in_eclipse([100.0 + 0.08 / 0.1], eb)[0]    # inside the full width, outside the core
    assert not rank.in_eclipse([100.0], eb, depth_mag=[2.0])[0]  # far deeper than the model's eclipse


def test_episodes_and_duration_bound():
    t = np.array([0.0, 0.05, 10.0, 10.074, 10.6, 50.0])
    z = np.array([-5.0, -4.0, 0.0, -6.0, -3.5, -4.0])
    fov = np.array([1, 2, 1, 2, 1, 1])
    eps = rank.episodes(t, z, fov)
    assert [e[2] for e in eps] == [2, 2, 1]
    assert np.isclose(eps[0][1], rank.FOV_PAIR_DAYS) and np.isclose(eps[1][1], 0.526)
    assert eps[2][1] == 0.0


def test_rank_tiers_and_queue_roles():
    n = 40
    rng = np.random.default_rng(0)
    sel = pd.DataFrame({"source_id": np.arange(n), "channel": ["fov"] * 20 + ["single"] * 20,
                        "w": rng.uniform(5, 50, n), "q": 0.05, "g_mag": 15.0})
    ctx = pd.DataFrame({"source_id": np.arange(n), "phot_g_mean_mag": 15.0,
                        "best_class_name": ["ECL"] * 10 + ["SOLAR_LIKE"] * 30})
    eb = pd.concat([_eb()] * 10 + [_eb().iloc[:0]] * 0, ignore_index=True)
    for c in eb.columns:
        ctx[c] = np.nan
        ctx.loc[:9, c] = eb[c].to_numpy()
    phys = pd.DataFrame({"source_id": np.arange(n), "t_deepest": 100.0, "depth_frac": -0.3,
                         "grey": False, "tension": [True] + [False] * (n - 1)})
    r = rank.rank(sel, ctx, phys)
    assert (r.loc[r.source_id < 10, "tier"] == 3).all() and (r.loc[r.source_id >= 10, "tier"] == 1).all()
    assert r.iloc[0]["channel"] == "fov" and r.iloc[0]["tier"] == 1
    q = rank.queue(r, 5, 4, 3, np.random.default_rng(1))
    assert (q.role == "ranked").sum() == 5 and (q.role == "control").sum() == 3
    assert set(q.loc[q.role == "control", "source_id"]) <= set(range(10))


def test_coincidence_excludes_the_star_itself():
    ranked = pd.DataFrame({"source_id": [1, 2, 3], "t_deepest": [10.0, 10.01, 30.0]})
    at = vet.coincidence_at(ranked)
    r = at(10.0, "gaia_dr3:1")
    assert r["k"] == 1 and r["n"] == 1
    assert at(30.0, "gaia_dr3:3")["n"] == 0


def test_score_v1_counts_rare_astrophysics_as_interesting():
    from cosmos_search.dayone.vet import interesting_score, score_v1

    assert interesting_score({"disk_occultation": 0.3, "unexplained": 0.1, "eclipsing_binary": 0.6}) == 0.4
    assert interesting_score({"yso_dipper": 0.9, "unexplained": 0.1}) == 0.1
    table = pd.DataFrame([
        {"kind": "decoy", "injected_family": "grey_dip", "ok": True, "s_interesting": 0.7, "p_unexplained": 0.1},
        {"kind": "decoy", "injected_family": "dip_train_primes", "ok": True, "s_interesting": 0.2, "p_unexplained": 0.2},
        {"kind": "decoy", "injected_family": "g_only", "ok": True, "s_interesting": 0.5, "p_unexplained": 0.5},
        {"kind": "control", "injected_family": None, "ok": True, "s_interesting": 0.1, "p_unexplained": 0.05},
        {"kind": "real", "injected_family": None, "ok": True, "s_interesting": 0.4, "p_unexplained": 0.1},
    ])
    rep = score_v1(table, threshold=0.35)
    assert rep["target_decoy_recall"]["k"] == 1 and rep["target_decoy_recall"]["n"] == 2
    assert rep["other_decoys_flagged"]["k"] == 1
    assert rep["controls_flagged"]["k"] == 0
    assert len(rep["shortlist"]) == 1
    assert rep["auc_s_target_vs_controls"] == 1.0


def test_keep_table_keeps_g3_selections_and_matches_dr3_g3_statistics(tmp_path):
    import importlib.util
    import json

    from cosmos_search.dayone import keep
    from cosmos_search.gaia import features

    spec = importlib.util.spec_from_file_location("dr3_g3", "scripts/dr3_g3.py")
    g3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g3)
    rng = np.random.default_rng(1)
    n = 400
    f = pd.DataFrame({"source_id": np.arange(n, dtype=np.int64), "n_g": rng.integers(5, 80, n),
                      "med_flux_g": 10 ** ((25.6874 - rng.uniform(8, 20, n)) / 2.5),
                      "min_z_rob": -rng.exponential(5, n), "max_z_rob": rng.exponential(5, n),
                      "min_partner_z": -rng.exponential(5, n), "max_partner_z": rng.exponential(5, n)})
    absw, g_bin = features.g3_abs_w(f)
    for name, dim, bright in g3.statistics(f):
        if name in absw:
            fin = np.isfinite(dim)
            assert np.allclose(absw[name][fin], np.fmax(dim, bright)[fin])
    assert (g_bin == g3.strata(f)[0]).all()
    _, q = g3.run(f)
    t = keep.table(q, level=0.5, margin=1.0)
    path = tmp_path / "t.json"
    path.write_text(json.dumps(t))
    kept = features.keep_light_curve(f, controls=0, table=path)
    assert kept[keep.selected(q, 0.5)].all()
    assert kept.mean() < 1


def test_wise_upper_limits_are_named_as_limits():
    # WY Cen (Gaia DR3 5862265011945308544): W3 and W4 are AllWISE upper limits, W1/W2 are detections.
    row = pd.Series({"allwise_w1sigmpro": 0.024, "allwise_w2sigmpro": 0.021, "allwise_w3sigmpro": np.nan,
                     "allwise_w4sigmpro": np.nan})
    ctx = {"W1": 9.857, "W2": 9.944, "W3": 10.029, "W4": 7.462, "Ks": 10.038}
    out = vet.wise_limits(ctx, row)
    assert out == {"W1": 9.857, "W2": 9.944, "W3_upper_limit": 10.029, "W4_upper_limit": 7.462, "Ks": 10.038}
    # no sigma columns (older context tables) or no WISE match: unchanged
    assert vet.wise_limits(ctx, pd.Series({"x": 1.0})) == ctx
    assert list(vet.wise_limits({"W4": float("nan")}, row)) == ["W4"]
