import json

import numpy as np
import pandas as pd
import pytest

from cosmos_search.inject import lightcurve as inj
from cosmos_search.slow import checks, detect, vet
from cosmos_search.vetting import packet


def test_ratios_follow_each_explanation():
    grey = vet.ratios("grey")
    assert all(v == 1.0 for v in grey.values())
    dust = vet.ratios("dust", beta=1.5)
    assert dust["BP"] / dust["RP"] == pytest.approx((0.51 / 0.78) ** -1.5)
    assert 0.03 < dust["W1"] < 0.12          # near WISE_OVER_G_DUST
    shell = vet.ratios("reemit", teff=5800, beta=1.5, t_shell=300)
    assert shell["BP"] == dust["BP"] and shell["W2"] < 0 and shell["W1"] < 0   # the infrared brightens
    hot = vet.ratios("temperature", teff=5800)
    # A cooling Sun-like star: BP dims more than RP, W1/W2 dim far less than G (Rayleigh-Jeans side).
    assert 1.3 < hot["BP"] / hot["RP"] < 1.6 and 0.2 < hot["W1"] < 0.5 and hot["G"] == 1.0
    g = vet.ratios("g_only")
    assert g["G"] == 1.0 and all(v == 0.0 for k, v in g.items() if k != "G")
    with pytest.raises(ValueError):
        vet.ratios("violet")


def test_band_scaled_carries_g_change():
    base = inj.Dip(t0=100.0, covering=0.1, ingress=10, egress=10, flat=50)
    s = inj.BandScaled(base, {"G": 1.0, "BP": 2.0, "W1": -1.0}, default=0.0)
    t = np.array([0.0, 140.0])
    assert s.factor(t, "G") == pytest.approx([1.0, 0.9])
    assert s.factor(t, "BP") == pytest.approx([1.0, 0.8])
    assert s.factor(t, "W1") == pytest.approx([1.0, 1.1])
    assert s.factor(t, "RP") == pytest.approx([1.0, 1.0])
    tr = inj.truth(s)
    assert tr["kind"] == "band_scaled" and tr["base"]["covering"] == 0.1
    json.dumps(tr)


def _epochs(rng, sig=None, n_visits=40, sigma=0.002):
    """Drift-corrected epochs (band codes 0/1/2) of a quiet star, two epochs per visit, with a signal."""
    tv = np.sort(rng.uniform(0, 1000, n_visits))
    t = np.repeat(tv, 2) + np.tile([0.0, 0.074], n_visits)
    rows = []
    for code, name in ((0, "G"), (1, "BP"), (2, "RP")):
        k = sig.factor(t, name) if sig is not None else np.ones_like(t)
        rows.append(pd.DataFrame({"source_id": 7, "band": code, "t": t,
                                  "frac_corr": k - 1 + rng.normal(0, sigma, t.size),
                                  "err_frac": sigma, "fov": np.tile([1, 2], n_visits)}))
    return pd.concat(rows, ignore_index=True)


def test_band_changes_of_an_episode_and_a_trend():
    rng = np.random.default_rng(3)
    dip = inj.BandScaled(inj.Dip(t0=400.0, covering=0.03, ingress=30, egress=30, flat=200), vet.ratios("dust", beta=2.0))
    ep = _epochs(rng, dip)
    row = detect.star_features(ep).iloc[0].to_dict()
    row["by"] = "episode"
    ch = vet.band_changes(ep, row)
    assert ch["G"][0] > 0.015 and ch["BP"][0] / ch["RP"][0] == pytest.approx((0.51 / 0.78) ** -2.0, rel=0.25)
    trend = inj.BandScaled(inj.SecularTrend(rate_per_year=-0.02, t_ref=500.0), vet.ratios("grey"))
    ep = _epochs(rng, trend)
    row = detect.star_features(ep).iloc[0].to_dict()
    row["by"] = "trend"
    t0, t1, kind = vet.change_window(row)
    assert kind == "trend"
    ch = vet.band_changes(ep, row)
    expected = 0.02 * (t1 - t0) / 365.25
    for b in ("G", "BP", "RP"):
        assert ch[b][0] == pytest.approx(expected, rel=0.1) and ch[b][1] > 0


def test_neowise_visits_and_ztf_normalised_inject_before_normalising():
    rng = np.random.default_rng(4)
    mjd = np.concatenate([m + rng.uniform(0, 2, 12) for m in np.arange(57000.0, 60000.0, 180.0)])
    clean = pd.DataFrame({"mjd": np.tile(mjd, 2), "band": np.repeat(["W1", "W2"], mjd.size),
                          "flux": 1 + rng.normal(0, 0.01, 2 * mjd.size), "err": 0.01})
    sig = inj.BandScaled(inj.Step(t_step=1500.0, fraction=-0.2), vet.ratios("grey"))
    v = vet.neowise_visits(clean, sig)
    w1 = v[v["band"] == "W1"]
    assert len(w1) == len(np.arange(57000.0, 60000.0, 180.0))
    late = w1["t"] > 1500.0 + 10
    assert np.median(w1["flux"]) == pytest.approx(1.0) and w1.loc[late, "flux"].mean() < w1.loc[~late, "flux"].mean() - 0.15
    raw = pd.DataFrame({"oid": np.repeat([1, 2], 50), "band": "ztf_r", "t": np.tile(np.linspace(1200, 3000, 50), 2),
                        "mag": np.r_[np.full(50, 15.0), np.full(50, 15.3)], "magerr": 0.01})
    z = vet.ztf_normalised(raw, sig)
    for oid in (1, 2):
        assert np.median(z["flux"][raw["oid"] == oid]) == pytest.approx(1.0)
    assert z["flux"][(raw["t"] > 1500) & (raw["oid"] == 1)].mean() < z["flux"][(raw["t"] < 1500) & (raw["oid"] == 1)].mean()


def test_candidate_builds_a_clean_packet():
    rng = np.random.default_rng(5)
    sig = inj.BandScaled(inj.Dip(t0=300.0, covering=0.05, ingress=30, egress=30, flat=150), vet.ratios("grey"))
    ep = _epochs(rng, sig)
    row = detect.star_features(ep).iloc[0].to_dict()
    row.update(by="episode", s_rep=np.nan, rep_period=np.nan)
    mjd = np.concatenate([m + rng.uniform(0, 2, 12) for m in np.arange(56900.0, 60000.0, 180.0)])
    clean = pd.DataFrame({"mjd": np.tile(mjd, 2), "band": np.repeat(["W1", "W2"], mjd.size),
                          "flux": 1 + rng.normal(0, 0.01, 2 * mjd.size), "err": 0.01})
    ztf = pd.DataFrame({"band": "ztf_g", "t": np.linspace(1300, 3000, 200), "flux": 1 + rng.normal(0, 0.01, 200),
                        "err": 0.01})
    ch = vet.band_changes(ep, row)
    tests = vet.tests(row, ch, clean, ztf, 0.8)
    assert abs(tests["colour_of_change"]["z_grey"]) < 4
    assert "W1_change" in tests["neowise_change"] and "W1_pred_star_temperature" in tests["neowise_change"]
    cand = vet.candidate("gaia_dr3:7", row, ep, {"ra": 10.0, "dec": 20.0, "phot_g_mean_mag": 14.2, "bp_rp": 0.8,
                                                 "parallax": 2.0, "parallax_error": 0.05},
                         {"W1": 12.0}, neowise=vet.neowise_visits(clean), ztf=ztf, tests_=tests)
    pk, key = packet.build(cand, holdout=(), pipeline_flags=False, pipeline_tests=False)
    assert pk["bands"] == ["G", "BP", "RP", "ztf_g", "W1", "W2"]
    assert len(pk["dips"]) == 1 and pk["dips"][0]["t_start"] >= 0
    assert set(pk["tests"]) == {"slow_change", "colour_of_change", "neowise_change", "ztf_after"}
    assert pk["tests"]["slow_change"]["n_visits"] == row["n_visits"]   # numpy integers kept
    assert key["pipeline"] == {} and "ra" not in pk["source"]


def test_snr_matches_the_detector_definition():
    sig = inj.BandScaled(inj.Step(t_step=50.0, fraction=-0.01), vet.ratios("grey"))
    tv = np.arange(0.0, 100.0, 10.0)
    assert vet.snr(sig, tv, 0.001) == pytest.approx(0.01 * np.sqrt(5) / 0.001)
    assert checks.DUST_BP_RP == 1.7
