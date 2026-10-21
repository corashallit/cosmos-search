import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("dr3_g3", Path(__file__).resolve().parents[1] / "scripts" / "dr3_g3.py")
g3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g3)


def fake_features(rng, n=20000, n_real=200):
    n_g = rng.integers(5, 150, n)
    x = rng.standard_t(4, (n, 4))
    f = pd.DataFrame({
        "source_id": np.arange(n, dtype=np.int64), "n_g": n_g, "med_flux_g": 10 ** ((25.6874 - rng.uniform(10, 20, n)) / 2.5),
        "min_z_rob": -np.abs(x[:, 0]) - 2, "max_z_rob": np.abs(x[:, 1]) + 2,
        "min_partner_z": x[:, 2], "max_partner_z": x[:, 3],
        "min_frac": -0.01 * np.ones(n), "min_frac_bp": -0.01 * np.ones(n), "min_frac_rp": -0.01 * np.ones(n),
        "n_low_episodes": np.ones(n, int), "n_rej_low3": np.zeros(n, int), "slope_g": np.zeros(n),
        "control": np.zeros(n, bool), "bp_rp": np.ones(n), "ruwe": np.ones(n), "ipd_frac_multi_peak": np.zeros(n),
    })
    # Real dimmers: deep in both fields of view.
    f.loc[: n_real - 1, ["min_z_rob", "min_partner_z"]] = -40.0
    f.loc[: n_real - 1, "n_g"] = 60
    return f


def test_g3_finds_fov_confirmed_dimmers_and_reports_symmetry():
    f = fake_features(np.random.default_rng(1))
    sym, q = g3.run(f)
    assert set(sym["statistic"]) == {"single", "fov"}
    found = q["q_fov"] <= 0.01
    # knockoff+ cannot report q below 1 / (number of discoveries), so 200 real dimmers are needed for q <= 0.01.
    assert found[:200].all()
    assert found[200:].sum() <= 2
    # Sources with too few transits get no statistic.
    assert np.isnan(q.loc[f["n_g"] < 10, "w_fov"]).all()
    text = g3.summary(sym, q, f)
    assert "FoV-confirmed dimmers at q ≤ 0.01" in text
