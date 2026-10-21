import numpy as np
import pandas as pd
import pytest

from cosmos_search.xmatch import Astrometry, ErrorModel, MemorySource, crossmatch
from cosmos_search.xmatch.astrometry import parallax_factors
from cosmos_search.xmatch.geometry import offset_position, separation_mas, tangent_offsets
from cosmos_search.xmatch.match import unit_posteriors
from cosmos_search.xmatch.simulate import sim_field, sim_survey
from cosmos_search.xmatch.surveys import SURVEYS


def _star(**kw):
    base = dict(source_id=1, ra=269.4485, dec=4.7398, ref_epoch=2016.0, parallax=546.9759, pmra=-801.551,
                pmdec=10362.394, ra_error=0.02, dec_error=0.02, parallax_error=0.04, pmra_error=0.03,
                pmdec_error=0.03)
    base.update(kw)
    return pd.DataFrame([base])


def test_linear_tangent_plane_motion_matches_astropy():
    # With zero radial velocity, uniform space motion is exactly linear in gnomonic coordinates.
    import astropy.units as u
    from astropy.coordinates import Distance, SkyCoord
    from astropy.time import Time

    t = _star()
    a = Astrometry.from_table(t)
    off, _ = a.predict(1999.25, "none")
    ra, dec = offset_position(a.ra, a.dec, off[:, 0], off[:, 1])
    c = SkyCoord(ra=t.ra[0] * u.deg, dec=t.dec[0] * u.deg, distance=Distance(parallax=t.parallax[0] * u.mas),
                 pm_ra_cosdec=t.pmra[0] * u.mas / u.yr, pm_dec=t.pmdec[0] * u.mas / u.yr,
                 radial_velocity=0 * u.km / u.s, obstime=Time(2016.0, format="jyear"))
    c2 = c.apply_space_motion(new_obstime=Time(1999.25, format="jyear"))
    assert separation_mas(ra[0], dec[0], c2.ra.deg, c2.dec.deg) < 0.5


def test_parallax_factor_amplitude_at_ecliptic_pole():
    # Seen from the ecliptic pole, the parallactic ellipse is a circle of radius ~ Earth's distance in au.
    t = np.linspace(2020.0, 2021.0, 24)
    fx, fy = parallax_factors(np.full_like(t, 270.0), np.full_like(t, 66.5607), t)
    r = np.hypot(fx, fy)
    assert np.all((r > 0.98) & (r < 1.02))


def test_propagated_covariance_grows_with_time():
    a = Astrometry.from_table(_star(pmra_error=1.0, pmdec_error=2.0))
    _, c0 = a.predict(2016.0, "none")
    _, c1 = a.predict(2000.0, "none")
    assert c1[0, 0, 0] == pytest.approx(0.02**2 + 16**2 * 1.0, rel=1e-6)
    assert c1[0, 1, 1] == pytest.approx(0.02**2 + 16**2 * 4.0, rel=1e-6)
    assert c0[0, 0, 0] == pytest.approx(0.02**2)


def test_unsolved_sources_get_the_local_motion_prior():
    t = pd.concat([_star(source_id=i, pmra=5.0 + i % 3, pmdec=-2.0, phot_g_mean_mag=19.5) for i in range(40)]
                  + [_star(source_id=99, pmra=np.nan, pmdec=np.nan, parallax=np.nan)], ignore_index=True)
    a = Astrometry.from_table(t)
    assert not a.solved[-1]
    assert a.mean[-1, 3] == pytest.approx(6.0)
    assert a.cov[-1, 3, 3] >= 1.0


def test_tangent_offsets_round_trip():
    ra, dec = offset_position(10.0, -70.0, 1234.0, -5678.0)
    xi, eta = tangent_offsets(10.0, -70.0, ra, dec)
    assert xi == pytest.approx(1234.0, abs=1e-6) and eta == pytest.approx(-5678.0, abs=1e-6)


def test_error_ellipse_orientation():
    rows = pd.DataFrame({"err_maj": [0.2, 0.2], "err_min": [0.1, 0.1], "err_ang": [0.0, 90.0]})
    c = SURVEYS["2mass"].covariance(rows)
    assert np.allclose(c[0], [[100.0**2, 0], [0, 200.0**2]])  # major axis north
    assert np.allclose(c[1], [[200.0**2, 0], [0, 100.0**2]], atol=1e-6)  # major axis east


def test_wise_cosigma_sign():
    rows = pd.DataFrame({"sigra": [0.1], "sigdec": [0.2], "sigradec": [-0.05]})
    c = SURVEYS["allwise"].covariance(rows)
    assert c[0, 0, 1] == pytest.approx(-(50.0**2))


def test_unit_posterior_closed_form():
    llr = np.log(np.array([4.0, 1.0]))
    p, p_none = unit_posteriors(llr, np.array([0, 0]), np.array([0.5]))
    den = 0.5 + 0.5 * 5.0
    assert p == pytest.approx([2.0 / den, 0.5 / den])
    assert p_none == pytest.approx([0.5 / den])


def _accuracy(res, truth):
    b = res.best
    sid = b["sim_id"].astype(float).fillna(-1).to_numpy().astype(int)
    src = (b.source_id - 10**15).to_numpy()
    return b, sid == truth[src, 0], truth[src, 0] >= 0


@pytest.mark.parametrize("bg,true_scale,true_floor", [(0.5, 1.3, 60.0), (30.0, 1.1, 150.0)])
def test_synthetic_field_calibration_and_false_matches(bg, true_scale, true_floor):
    sv = sim_survey(floor_mas=100.0)
    gaia, rows, truth = sim_field(4000, survey=sv, seed=3, bg_per_arcmin2=bg, true_scale=true_scale,
                                  true_floor_mas=true_floor, sigma_cat_mas=(200.0, 800.0) if bg > 1 else (80.0, 300.0))
    res = crossmatch(gaia, sv, source=MemorySource({"sim": rows, "gaia_dr3": gaia}), n_shifts=3)
    assert res.report["calibrated"]
    assert res.model.scale == pytest.approx(true_scale, rel=0.1)
    assert res.model.floor_mas == pytest.approx(true_floor, rel=0.35)
    assert np.median(res.prior) == pytest.approx(0.7, abs=0.05)
    chk = res.report["chi2_check"]
    assert chk["mean"] == pytest.approx(2.0, abs=0.15)
    pc = res.report["p_chance_check"]  # per-source chance probability agrees with the shifted sky
    assert pc["mean_p_chance"] == pytest.approx(pc["fmr_shift"], rel=0.3, abs=0.002)

    b, ok, _ = _accuracy(res, truth)
    for row in res.report["thresholds"]:
        T = row["threshold"]
        true_false = int(((b.p_best >= T) & ~ok).sum())
        tol = 5 + 3 * np.sqrt(true_false)
        assert abs(row["false_shift"] - true_false) < tol, (T, row, true_false)
        assert abs(row["false_model"] - true_false) < tol, (T, row, true_false)

    # Posteriors mean what they say.
    m = res.matches
    hit = m.sim_id.to_numpy() == truth[(m.source_id - 10**15).to_numpy(), 0]
    for lo, hi in [(0.05, 0.5), (0.5, 0.9), (0.9, 0.99)]:
        s = ((m.p_match >= lo) & (m.p_match < hi)).to_numpy()
        if s.sum() >= 50:
            assert hit[s].mean() == pytest.approx(m.p_match[s].mean(), abs=0.06)


def test_high_proper_motion_needs_propagation():
    sv = sim_survey(parallax_mode="exact", per_row_epoch=True, epoch=1999.0)
    gaia, rows, truth = sim_field(3000, survey=sv, seed=4, frac_high_pm=0.1)
    res = crossmatch(gaia, sv, source=MemorySource({"sim": rows, "gaia_dr3": gaia}), n_shifts=2)
    b, ok, has = _accuracy(res, truth)
    missed = (has & ~((b.p_best >= 0.9) & ok)).sum()
    still = gaia.copy()
    still[["pmra", "pmdec", "parallax"]] = 0.0
    res0 = crossmatch(still, sv, source=MemorySource({"sim": rows, "gaia_dr3": gaia}), n_shifts=2)
    b0, ok0, has0 = _accuracy(res0, truth)
    missed0 = (has0 & ~((b0.p_best >= 0.9) & ok0)).sum()
    assert missed < 20
    assert missed0 > 150


def test_grouped_survey_matches_each_group():
    sv = sim_survey(groups=True)
    gaia, rows, truth = sim_field(1500, survey=sv, seed=5, n_groups=3, bg_per_arcmin2=1.0)
    res = crossmatch(gaia, sv, source=MemorySource({"sim": rows, "gaia_dr3": gaia}), n_shifts=2)
    acc = res.best[res.best.p_best >= 0.9]
    per_target = acc.groupby("source_id").size()
    # Each target with counterparts in several groups gets one accepted match per group.
    n_true = (truth >= 0).sum(axis=1)
    src = per_target.index.to_numpy() - 10**15
    assert np.mean(per_target.to_numpy() == n_true[src]) > 0.97


def test_small_batches_skip_calibration():
    sv = sim_survey()
    gaia, rows, _ = sim_field(50, survey=sv, seed=6)
    res = crossmatch(gaia, sv, source=MemorySource({"sim": rows, "gaia_dr3": gaia}))
    assert not res.report["calibrated"]
    assert res.model == ErrorModel(floor_mas=sv.floor_mas)


def test_hats_reads_refuse_blind_runs(monkeypatch):
    from cosmos_search.net import BlindModeError
    from cosmos_search.xmatch import HatsSource

    monkeypatch.setenv("COSMOS_BLIND", "1")
    with pytest.raises(BlindModeError):
        HatsSource().fetch(SURVEYS["2mass"], 10.0, 10.0, 0.01)


def test_previous_release_pick_flags_splits():
    from cosmos_search.xmatch.ids import pick_closest

    d = pd.DataFrame({"dr3_source_id": [1, 1, 2, 3], "dr2_source_id": [10, 11, 12, 12],
                      "angular_distance": [5.0, 1.0, 2.0, 3.0], "magnitude_difference": [0.1, 0.0, 0.2, 0.3]})
    p = pick_closest(d).set_index("dr3_source_id")
    assert p.loc[1, "dr2_source_id"] == 11 and p.loc[1, "n_old"] == 2
    assert p.loc[2, "n_new"] == 2 and p.loc[3, "n_new"] == 2


def test_check_ids_counts_agreement():
    from cosmos_search.xmatch.ids import check_ids

    m = pd.DataFrame({"source_id": [1, 2, 3, 4], "rank": 1, "p_match": [0.99, 0.99, 0.99, 0.2],
                      "GAIA": [10, 99, np.nan, 13]})
    out, summ = check_ids(m, "GAIA", pd.Series({1: 10, 2: 12, 3: 11, 4: 13}))
    assert summ == {"matches": 3, "comparable": 2, "agree": 1, "disagree": 1}


def test_beam_contamination_counts_overlap_weighted_flux():
    from cosmos_search.xmatch.blend import beam_contamination

    ra, dec = offset_position(50.0, 10.0, np.array([0.0, 3000.0, 20000.0]), np.zeros(3))
    g = pd.DataFrame({"source_id": [1, 2, 3], "ra": ra, "dec": dec, "phot_rp_mean_mag": [15.0, 15.0, 10.0]})
    c = beam_contamination(g.iloc[:1], 6.0, MemorySource(g)).iloc[0]
    sigma = 6000.0 / 2.3548
    assert c.n_beam == 1 and c.n_near == 1
    assert c.contamination == pytest.approx(np.exp(-(3000.0**2) / (4 * sigma**2)), rel=1e-6)
    assert c.nearest_arcsec == pytest.approx(3.0, rel=1e-6)


def test_tail_component_keeps_the_floor_honest():
    sv = sim_survey(floor_mas=100.0)
    gaia, rows, truth = sim_field(4000, survey=sv, seed=7, bg_per_arcmin2=5.0, true_scale=1.0,
                                  true_floor_mas=100.0, tail_frac=0.1, tail_mas=1500.0)
    res = crossmatch(gaia, sv, source=MemorySource({"sim": rows, "gaia_dr3": gaia}), n_shifts=3)
    assert res.model.floor_mas == pytest.approx(100.0, rel=0.35)
    assert res.model.tail_frac == pytest.approx(0.1, abs=0.04)
    assert res.model.tail_mas == pytest.approx(1500.0, rel=0.3)
    b, ok, _ = _accuracy(res, truth)
    row = {r["threshold"]: r for r in res.report["thresholds"]}[0.9]
    true_false = int(((b.p_best >= 0.9) & ~ok).sum())
    assert abs(row["false_model"] - true_false) < 5 + 3 * np.sqrt(true_false)


def test_context_joins_surveys_into_one_row_per_target():
    from cosmos_search.xmatch.context import context

    sv = sim_survey()
    gaia, rows, truth = sim_field(400, survey=sv, seed=8)
    gaia["phot_rp_mean_mag"] = gaia.phot_g_mean_mag - 0.5
    import cosmos_search.xmatch.surveys as S
    S.SURVEYS["sim"] = sv
    try:
        src = MemorySource({"sim": rows, "gaia_dr3": gaia})
        wide, res = context(gaia, ["sim"], source=src, beams={"w1": 6.1})
    finally:
        del S.SURVEYS["sim"]
    assert len(wide) == len(gaia) and wide.source_id.is_unique
    assert {"sim_id", "sim_p", "sim_p_none", "sim_n_local", "blend_w1_contamination"} <= set(wide.columns)
    got = wide.sim_id.notna()
    assert got.mean() == pytest.approx(0.7, abs=0.06)


def test_competition_from_gaia_neighbours_halves_false_matches_and_stays_calibrated():
    sv = sim_survey(floor_mas=150.0, search_radius_mas=4000.0, shift_annulus_arcsec=(30.0, 60.0))
    gaia, rows, truth = sim_field(20000, survey=sv, seed=11, radius_deg=0.25, bg_per_arcmin2=5.0,
                                  sigma_cat_mas=(300.0, 1000.0), true_scale=1.0, true_floor_mas=150.0,
                                  p_counterpart=0.5)
    sel = np.random.default_rng(0).uniform(size=len(gaia)) < 0.3
    targets, tr = gaia[sel].reset_index(drop=True), truth[sel][:, 0]
    pos = {s: i for i, s in enumerate(targets.source_id)}
    src = MemorySource({"sim": rows, "gaia_dr3": gaia})
    out = {}
    for comp in (False, True):
        res = crossmatch(targets, sv, source=src, n_shifts=3, compete=comp)
        b = res.best
        sid = b["sim_id"].astype(float).fillna(-1).to_numpy().astype(int)
        ok = sid == tr[np.array([pos[s] for s in b.source_id])]
        acc = (b.p_best >= 0.9).to_numpy()
        row = {r["threshold"]: r for r in res.report["thresholds"]}[0.9]
        out[comp] = (acc.sum(), (acc & ~ok).sum(), row["false_model"], row["false_shift"])
    n0, f0, _, _ = out[False]
    n1, f1, fm1, fs1 = out[True]
    assert n1 > n0 and f1 < 0.7 * f0
    assert abs(fm1 - f1) < 5 + 3 * np.sqrt(f1) and abs(fs1 - f1) < 5 + 3 * np.sqrt(f1)
