import numpy as np
import pytest

from cosmos_search.gaia import epoch, features

HEADER = (
    "solution_id,source_id,n_transits,transit_id,g_transit_time,g_transit_flux,g_transit_flux_error,"
    "g_transit_flux_over_error,g_transit_mag,g_transit_n_obs,bp_obs_time,bp_flux,bp_flux_error,"
    "bp_flux_over_error,bp_mag,rp_obs_time,rp_flux,rp_flux_error,rp_flux_over_error,rp_mag,"
    "photometry_flag_noisy_data,photometry_flag_bp_reject,photometry_flag_rp_reject,"
    "variability_flag_g_reject,variability_flag_bp_reject,variability_flag_rp_reject"
)


def tid(obmt_units, fov, row, pix=100):
    return (obmt_units << 17) | ((fov - 1) << 15) | (row << 12) | pix


def arr(values, fmt="{}"):
    return '"[' + ",".join(fmt.format(v) for v in values) + ']"'


def bools(values):
    return '"[' + ",".join(f'""{bool(v)}""' for v in values) + ']"'


def synth_csv(sources, seed=0):
    """Bulk-file-shaped CSV. sources: list of (source_id, g_flux array, extra kwargs)."""
    rng = np.random.default_rng(seed)
    lines = ["# %ECSV 1.0", "# ---", HEADER]
    for sid, g, kw in sources:
        n = len(g)
        t = 1700 + np.sort(rng.uniform(0, 900, n // 2)).repeat(2) + np.tile([0, 0.074], n // 2)
        fov = np.tile([1, 2], n // 2)
        tids = [tid(10_000_000 + i * 1000, f, 1 + i % 7) for i, f in enumerate(fov)]
        err = np.full(n, 10.0)
        bp = kw.get("bp", g * 0.5)
        rp = kw.get("rp", g * 0.8)
        z = [False] * n
        cells = [
            "1", str(sid), str(n), arr(tids), arr(t), arr(g), arr(err), arr(g / err), arr(np.zeros(n)),
            arr([9] * n), arr(t + 0.0005), arr(bp), arr(err), arr(bp / err), arr(np.zeros(n)),
            arr(t + 0.0006), arr(rp), arr(err), arr(rp / err), arr(np.zeros(n)),
            bools(z), bools(z), bools(z), bools(kw.get("g_reject", z)), bools(z), bools(z),
        ]
        lines.append(",".join(cells))
    return ("\n".join(lines) + "\n").encode()


def test_decode_transit_id_round_trips():
    t = np.array([tid(123456, 1, 3, 42), tid(999, 2, 7, 4095)], dtype=np.int64)
    obmt, fov, row, pix = epoch.decode_transit_id(t)
    assert list(obmt) == [123456 * 204800, 999 * 204800]
    assert list(fov) == [1, 2] and list(row) == [3, 7] and list(pix) == [42, 4095]


def test_gaia_time_to_btjd():
    # BJD_TCB 2457000.0 is ~18.6 s later in TCB than in TDB.
    assert epoch.gaia_time_to_btjd(2457000.0 - 2455197.5) == pytest.approx(-2.157e-4, abs=2e-6)


def test_read_epoch_csv_long_format_and_nan_dropping():
    g = np.full(20, 1000.0)
    bp = g * 0.5
    bp[3] = np.nan
    lc = epoch.read_epoch_csv(synth_csv([(42, g, {"bp": bp}), (7, g, {})]))
    assert lc.schema.equals(epoch.SCHEMA)
    sid, band = lc["source_id"].to_numpy(), lc["band"].to_pylist()
    assert list(np.unique(sid)) == [7, 42]
    assert band.count("G") == 40 and band.count("BP") == 39 and band.count("RP") == 40
    assert set(lc["fov"].to_numpy()) == {1, 2}
    assert np.all(np.diff(sid) >= 0)


def test_read_epoch_csv_rejects_length_mismatch():
    bad = synth_csv([(1, np.full(4, 100.0), {})]).replace(b",4,", b",5,", 1)
    with pytest.raises(ValueError):
        epoch.read_epoch_csv(bad)


def test_features_find_a_grey_dip_confirmed_by_partner_and_colour():
    rng = np.random.default_rng(1)
    quiet = 1000 + rng.uniform(-10, 10, 40)
    dip = quiet.copy()
    dip[10:12] *= 0.8  # both FoV transits of one pair dim by 20%
    bp, rp = dip * 0.5, dip * 0.8
    lc = epoch.read_epoch_csv(synth_csv([(1, quiet, {}), (2, dip, {"bp": bp, "rp": rp})]))
    f = features.source_features(lc).set_index("source_id")
    assert f.loc[2, "min_z_rob"] < -10 and f.loc[1, "min_z_rob"] > -5
    assert f.loc[2, "min_partner_z"] < -10
    assert f.loc[2, "min_frac_bp"] == pytest.approx(-0.2, abs=0.03)
    assert f.loc[2, "min_frac_rp"] == pytest.approx(-0.2, abs=0.03)
    assert f.loc[2, "partner_agree_low"] == 1.0
    assert f.loc[2, "n_low_episodes"] == 1
    assert features.keep_light_curve(f.reset_index())[1]


def test_rejected_faint_points_are_counted_not_used_for_scale():
    rng = np.random.default_rng(2)
    g = 1000 + rng.normal(0, 10, 30)
    g[5] = 500
    reject = [i == 5 for i in range(30)]
    f = features.source_features(epoch.read_epoch_csv(synth_csv([(3, g, {"g_reject": reject})])))
    assert f.loc[0, "n_rej"] == 1 and f.loc[0, "n_rej_low3"] == 1
    assert f.loc[0, "min_z_rob"] > -5


def test_control_mask_is_deterministic_and_near_one_percent():
    ids = np.arange(1, 2_000_001, dtype=np.int64) * 34359738368
    m = features.control_mask(ids)
    assert np.array_equal(m, features.control_mask(ids))
    assert 0.009 < m.mean() < 0.011


def test_stage0_removes_g_only_faults_but_keeps_grey_dips():
    rng = np.random.default_rng(7)
    g = 10_000 + rng.uniform(-20, 20, 40)
    grey = g.copy()
    grey[10] *= 0.6
    fault = g.copy()
    fault[10] *= 0.6
    flat = 10_000 + rng.uniform(-20, 20, 40)
    lc = epoch.read_epoch_csv(synth_csv([(1, grey, {}), (2, fault, {"bp": flat * 0.5, "rp": flat * 0.8})]))
    f = features.source_features(lc).set_index("source_id")
    assert f.loc[1, "min_z_rob"] < -50 and f.loc[1, "stage0_n_band_inconsistent"] == 0
    assert f.loc[2, "min_z_raw"] < -50 and f.loc[2, "stage0_n_band_inconsistent"] == 1
    assert f.loc[2, "min_z_rob"] > -5
    raw = features.source_features(lc, clean=False).set_index("source_id")
    assert raw.loc[2, "min_z_rob"] < -50 and "stage0_n_consistent" not in raw
