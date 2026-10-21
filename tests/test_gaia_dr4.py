import numpy as np
from astropy.table import Table

from cosmos_search.gaia import dr4, epoch, features


def tid(i, fov):
    return ((10_000_000 + i * 1000) << 17) | ((fov - 1) << 15) | ((1 + i % 7) << 12) | 100


def dr4_rows(n=40, dip_at=None):
    t = 1800 + np.sort(np.random.default_rng(0).uniform(0, 1900, n // 2)).repeat(2) + np.tile([0, 0.074], n // 2)
    g = 1000 + np.random.default_rng(1).uniform(-10, 10, n)
    if dip_at is not None:
        g[dip_at:dip_at + 2] *= 0.7
    phot = [np.zeros(12, np.int16) for _ in range(n)]
    phot[3] = phot[3].copy()
    phot[3][1] = 0x1  # saturation correction on AF1
    phot[5] = phot[5].copy()
    phot[5][10] = 0x2000  # BP unrecoverable
    proc = np.zeros(n, np.int16)
    proc[7] = 0x0002  # multiple-source IPD
    return Table({
        "source_id": np.full(n, 99, np.int64), "transit_id": [tid(i, 1 + i % 2) for i in range(n)],
        "g_n_obs": np.full(n, 9), "g_obs_time": t, "g_flux": g, "g_flux_error": np.full(n, 10.0),
        "bp_obs_time": t + 0.0005, "bp_flux": g * 0.5, "bp_flux_error": np.full(n, 10.0),
        "rp_obs_time": t + 0.0006, "rp_flux": g * 0.8, "rp_flux_error": np.full(n, 10.0),
        "transit_proc_flags": proc, "phot_ccd_proc_flags": phot,
    })


def test_dr4_rows_map_to_shared_schema_with_flags():
    lc = dr4.epoch_photometry_to_lc(dr4_rows())
    assert lc.schema.equals(epoch.SCHEMA)
    band = np.asarray(lc["band"].combine_chunks().indices)
    flags, tids = lc["flags"].to_numpy(), lc["transit_id"].to_numpy()
    t3, t5, t7 = tid(3, 2), tid(5, 2), tid(7, 2)
    assert np.all(flags[tids == t3] & dr4.FLAG_SATURATION)
    assert flags[(tids == t5) & (band == 1)][0] & epoch.FLAG_PHOT_REJECT
    assert not flags[(tids == t5) & (band == 0)][0] & epoch.FLAG_PHOT_REJECT
    assert np.all(flags[tids == t7] & dr4.FLAG_MULTI_SOURCE)


def test_dr3_features_run_unchanged_on_dr4_rows():
    f = features.source_features(dr4.epoch_photometry_to_lc(dr4_rows(dip_at=10)))
    assert f.loc[0, "min_z_rob"] < -10 and f.loc[0, "min_partner_z"] < -10


def test_ccd_consistency_separates_coherent_dip_from_single_ccd_hit():
    flux = np.full((2, 9), 1000.0)
    flux[0] *= 0.7  # whole transit dims: coherent
    flux[1, 4] = 700.0  # one CCD hit
    t = np.tile(np.arange(9) * 4.85 / 86400, (2, 1)) + 2000
    table = Table({"source_id": [1, 1], "transit_id": [11, 12], "af_obs_time": list(t),
                   "af_flux": list(flux), "af_flux_error": list(np.full((2, 9), 5.0))})
    c = dr4.ccd_consistency(dr4.epoch_photometry_ccd_to_long(table)).set_index("transit_id")
    assert c.loc[11, "chi2_red"] < 1e-6 and c.loc[12, "chi2_red"] > 100
    assert c.loc[11, "n_ccd"] == 9


def test_ccd_at_extremes_tells_coherent_dip_from_one_ccd_spike():
    import pandas as pd

    rng = np.random.default_rng(4)
    n = 30
    flux = 1000 + rng.normal(0, 5, (n, 9))
    flux[7] *= 0.7  # transit 7: all nine CCDs dim (occultation)
    flux[12, 3] += 400  # transit 12: one CCD hit (cosmic ray)
    t = 2000 + np.arange(n)[:, None] * 0.3 + np.arange(9)[None, :] * 4.85 / 86400
    table = Table({"source_id": np.full(n, 5, np.int64), "transit_id": np.arange(n, dtype=np.int64) + 100,
                   "af_obs_time": list(t), "af_flux": list(flux), "af_flux_error": list(np.full((n, 9), 5.0))})
    feat = pd.DataFrame({"source_id": [5, 6], "tid_min": [107, 999], "tid_max": [112, 999]})
    r = dr4.ccd_at_extremes(feat, dr4.epoch_photometry_ccd_to_long(table)).set_index("source_id")
    assert r.loc[5, "frac_ccd_low_min"] == 1.0 and r.loc[5, "n_ccd_min"] == 9
    assert r.loc[5, "frac_ccd_high_max"] == 1 / 9 and r.loc[5, "chi2_ccd_max"] > 100
    assert np.isnan(r.loc[6, "frac_ccd_low_min"])


def test_features_carry_extreme_transit_ids():
    rows = dr4_rows(dip_at=10)
    f = features.source_features(dr4.epoch_photometry_to_lc(rows), clean=False)
    assert f.loc[0, "tid_min"] in set(rows["transit_id"][10:12])
