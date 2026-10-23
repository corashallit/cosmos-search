import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from cosmos_search.gaia import epoch
from cosmos_search.slow import drift, release, sky


def patch_epochs(rng, n_stars=40, n_visits=30, shared=None, sigma=1e-3, patch=7, sid0=1):
    """Stars in one patch observed at the same visits; `shared` adds the same offset to every star per visit."""
    tv = np.linspace(0, 1000, n_visits) + rng.uniform(0, 10, n_visits)
    shared = np.zeros(n_visits) if shared is None else shared
    rows = []
    for k in range(n_stars):
        t = np.r_[tv, tv + 0.074]
        y = np.r_[shared, shared] + rng.normal(0, sigma, t.size)
        rows.append(pd.DataFrame({"source_id": sid0 + k, "band": np.int8(0), "t": t, "frac_corr": y, "patch": patch}))
    return pd.concat(rows, ignore_index=True), tv


def test_sky_offsets_recover_shared_visit_offsets():
    rng = np.random.default_rng(1)
    shared = rng.normal(0, 3e-3, 30)
    ep, tv = patch_epochs(rng, shared=shared)
    off = sky.offsets(ep)
    assert len(off) == 30 and off.attrs["tau"][0] == pytest.approx(3e-3, rel=0.4)
    got = off.sort_values("t_lo")["offset"].to_numpy()
    assert np.corrcoef(got, shared)[0, 1] > 0.98
    target = ep[ep["source_id"] == 1].assign(source_id=999)
    corr = target["frac_corr"].to_numpy() - sky.offset_at(target, off)
    assert corr.std() < 0.5 * target["frac_corr"].std()


def test_sky_offsets_shrink_to_nothing_without_a_shared_term():
    rng = np.random.default_rng(2)
    ep, _ = patch_epochs(rng)
    off = sky.offsets(ep)
    assert np.abs(off["offset"]).max() < 2e-4


def test_sky_offset_ignores_comparison_star_with_own_event():
    rng = np.random.default_rng(3)
    ep, tv = patch_epochs(rng, n_stars=10)
    ev = (ep["source_id"] == 2) & (ep["t"] > tv[10]) & (ep["t"] < tv[15])
    ep.loc[ev, "frac_corr"] -= 0.05
    assert np.abs(sky.offsets(ep)["raw"]).max() < 2e-3


def test_correct_by_halves_never_uses_a_star_on_itself():
    rng = np.random.default_rng(4)
    ep, _ = patch_epochs(rng, n_stars=30, sid0=10**15)
    ep = ep.drop(columns="patch")
    sid = ep["source_id"].unique()
    # One star with a deep long dip, as a comparison star of its own half only.
    m = (ep["source_id"] == sid[0]) & (ep["t"] > 300) & (ep["t"] < 600)
    ep.loc[m, "frac_corr"] -= 0.004
    off, tables = sky.correct_by_halves(ep, sid, level=5)
    own = drift.half(np.array([sid[0]]))[0]
    in_dip = off[m.to_numpy()]
    assert np.abs(in_dip).max() < 1e-3  # its own dip did not leak into its correction


def test_quiet_ids_and_cross_corrected():
    rng = np.random.default_rng(5)
    rows = []
    for k in range(200):
        t = np.sort(rng.uniform(0, 1000, 40))
        noisy = k % 10 == 0
        y = rng.normal(0, 3e-3 if noisy else 1e-3, t.size)
        rows.append(pd.DataFrame({"source_id": 10**12 + k, "band": np.int8(0), "t": t, "frac": y,
                                  "err_frac": 1e-3, "fov": 1}))
    ep = pd.concat(rows, ignore_index=True)
    q = release.quiet_ids(ep)
    assert len(q) == 180 and not np.isin(10**12, q)
    cells = pd.DataFrame({"gbin": np.int8(5), "cbin": np.int8(1)}, index=pd.Index(ep["source_id"].unique(), name="source_id"))
    out, table = release.cross_corrected(ep, cells, q)
    assert "frac_corr" in out and len(out) == len(ep) and len(table) > 0


def test_thresholds_and_stars_needed():
    assert release.stars_needed(0.001, 0.2) == 25000
    rng = np.random.default_rng(6)
    st = pd.DataFrame({"gbin": np.int8(5), "s_rise": rng.normal(0, 1, 20000), "slope_z": rng.normal(0, 1, 20000)})
    thr = release.thresholds(st, {"s": "s_rise", "z": "slope_z"}, n_boot=50)
    v = thr["G<19"]["s"]
    assert v["value"] == pytest.approx(3.09, abs=0.3) and v["beyond"] == 20 and v["lo"] < v["value"] < v["hi"]


def test_patch_ranges_are_whole_pixels():
    r = release.patch_ranges(5, level=9, seed=1)
    pix = drift.sky_pixel(np.array([x for rr in r for x in rr]), 9)
    assert (pix[0::2] == pix[1::2]).all() and len(set(pix)) == 5


def test_mirror_rows_to_table():
    rows = Table({
        "source_id": np.array([5, 6], np.int64),
        "n_transits": [2, 1],
        "transit_id": [np.array([10, 11]), np.array([12])],
        "g_transit_time": [np.array([1700.0, 1701.0]), np.array([1702.0])],
        "g_transit_flux": [np.array([100.0, 101.0]), np.array([50.0])],
        "g_transit_flux_error": [np.array([1.0, 1.0]), np.array([1.0])],
        "g_transit_n_obs": [np.array([9, 9]), np.array([8])],
        "bp_obs_time": [np.array([1700.0, 1701.0]), np.array([1702.0])],
        "bp_flux": [np.ma.masked_array([10.0, 11.0], [False, True]), np.array([5.0])],
        "bp_flux_error": [np.array([1.0, 1.0]), np.array([1.0])],
        "rp_obs_time": [np.array([1700.0, 1701.0]), np.array([1702.0])],
        "rp_flux": [np.array([20.0, 21.0]), np.array([7.0])],
        "rp_flux_error": [np.array([1.0, 1.0]), np.array([1.0])],
        **{name: [np.array([False, True]), np.array([False])] for name in epoch._BOOL},
    }, dtype=[np.int64, np.int64] + [object] * (11 + len(epoch._BOOL)))
    lc = epoch.mirror_rows_to_table(rows).to_pandas()
    assert len(lc) == 8  # 3 G + 2 BP (one masked) + 3 RP
    g5 = lc[(lc["source_id"] == 5) & (lc["band"] == "G")]
    assert list(g5["flux"]) == [100.0, 101.0] and list(g5["flags"] & epoch.FLAG_VARI_REJECT) == [0, 1]
    assert "source_id IN (1, 2)" in epoch.mirror_adql(source_ids=[1, 2])
