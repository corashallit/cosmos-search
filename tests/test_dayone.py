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
