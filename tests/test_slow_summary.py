import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from cosmos_search.gaia import epoch
from cosmos_search.slow import detect, drift, kinds, period, rule_v3, search, summary

LEVEL6 = 2 ** 35 * 4 ** 6      # source_ids per level-6 HEALPix pixel


def shapes(rng, t, kind):
    """Fractional flux change of one star at times t."""
    x = (t - t.min()) / (t.max() - t.min())
    if kind == "fade":
        return -0.03 * x
    if kind == "rise":
        return 0.03 * x
    if kind == "dip":
        return -0.04 * ((x > 0.4) & (x < 0.6))
    if kind == "drop":
        return -0.04 * (x > 0.7)
    if kind == "jump":
        return 0.04 * (x > 0.7)
    if kind == "repeat":
        return -0.05 * ((t % 90.0) < 20.0)
    return np.zeros(len(t))


def lightcurves(rng, kinds_=("noise",) * 6, n_visits=45, span=1000.0, sigma=0.002, flux=3e4, sid0=1, bands=True,
                drop=0.1):
    """Light curves (epoch.SCHEMA) of stars with the given shapes: two G epochs a visit, BP and RP on most transits,
    one star in ten with few visits."""
    rows = []
    for i, kind in enumerate(kinds_):
        n = 4 if kind == "short" else n_visits
        tv = np.sort(rng.uniform(1700.0, 1700.0 + span, n))
        t = np.sort(np.r_[tv, tv + 0.074])
        tid = np.arange(len(t)) + 1000 * i
        y = shapes(rng, t, kind)
        for band, scale in (("G", 1.0), ("BP", 0.4), ("RP", 0.6)):
            if band != "G" and not bands:
                continue
            keep = np.ones(len(t), bool) if band == "G" else rng.random(len(t)) > drop
            f = flux * scale * (1 + y[keep] + rng.normal(0, sigma, keep.sum()))
            rows.append(pd.DataFrame({"source_id": sid0 + i, "band": band, "t": t[keep], "flux": f,
                                      "err": np.full(keep.sum(), flux * scale * sigma), "flags": 0,
                                      "transit_id": tid[keep], "fov": 1, "ccd_row": 1, "n_obs": 9}))
    df = pd.concat(rows, ignore_index=True)
    return pa.Table.from_pandas(df, schema=epoch.SCHEMA, preserve_index=False)


KINDS = ("noise", "fade", "rise", "dip", "drop", "jump", "repeat", "short", "noise", "noise", "fade", "dip")


def test_summary_equals_the_detector_on_every_star():
    rng = np.random.default_rng(1)
    lc = lightcurves(rng, KINDS * 4)
    ep, _ = summary.corrected_epochs(lc)
    s = summary.star_summary(lc).set_index("source_id")
    d = detect.star_features(ep).set_index("source_id")
    assert len(s) == len(KINDS) * 4 and len(d) == len(s) - 4          # the four short stars have no scores
    short = s.index.difference(d.index)
    assert s.loc[short, "sigma_visit"].isna().all() and (s.loc[short, "n_visits"] == 4).all()
    assert np.isfinite(s.loc[short, "g_mag"]).all()
    s = s.loc[d.index]
    for c in ("n_visits", "t_first", "t_last", "sigma_visit", "abbe", "s_fade", "s_rise", "dim_t0", "dim_t1",
              "rise_t0", "dim_depth"):
        np.testing.assert_allclose(s[c].to_numpy(float), d[c].to_numpy(float), rtol=1e-12, atol=1e-15, err_msg=c)
    for c in ("slope", "slope_z", "slope_err", "slope_g", "slope_g_err", "slope_bp", "slope_bp_err", "slope_rp",
              "slope_rp_err"):
        np.testing.assert_allclose(s[c].to_numpy(float), d[c].to_numpy(float), rtol=1e-9, atol=1e-13, err_msg=c)
    assert (s["fade_kind"] == d["fade_kind"]).all() and (s["rise_kind"] == d["rise_kind"]).all()
    assert set(s["fade_kind"]) >= {"dip", "fade", "drop"} and set(s["rise_kind"]) >= {"bump", "rise", "jump"}


def test_repeat_bound_is_never_below_the_period_search():
    rng = np.random.default_rng(2)
    lc = lightcurves(rng, KINDS * 4, n_visits=40)
    ep, _ = summary.corrected_epochs(lc)
    s = summary.star_summary(lc).set_index("source_id")
    p = period.star_features(ep).set_index("source_id")
    s = s.loc[p.index]
    assert len(p) > 40
    assert p["s_rep_any"].notna().all() and p["s_rep"].notna().sum() >= 8
    assert (p["s_rep_any"].to_numpy() <= s["rep_bound"].to_numpy()).all()          # the best window of any period
    assert (p["s_rep_bright"].to_numpy() <= s["rep_bound_bright"].to_numpy()).all()
    assert not (p["s_rep"].to_numpy() > s["rep_bound"].to_numpy()).any()           # NaN where no period qualifies
    # the bound is tight enough to be useful: within a factor of three of the score for repeated dips
    rep = s.index[(s.index - 1) % len(KINDS) == KINDS.index("repeat")]
    assert (p.loc[rep, "s_rep"] > 7).all() and (s.loc[rep, "rep_bound"] < 3 * p.loc[rep, "s_rep"]).all()


def test_magnitudes_and_cells_come_from_the_light_curve():
    rng = np.random.default_rng(3)
    lc = lightcurves(rng, ("noise",) * 3, flux=10 ** ((25.6874 - 15.2) / 2.5))
    s = summary.star_summary(lc)
    assert s["g_mag"].to_numpy() == pytest.approx(15.2, abs=0.01)
    assert s["bp_rp"].to_numpy() == pytest.approx(25.3385 - 24.7479 - 2.5 * np.log10(0.4 / 0.6), abs=0.01)
    assert (s["gbin"] == np.digitize(15.2, drift.G_EDGES[1:-1])).all()
    cells = pd.DataFrame({"gbin": np.int8(2), "cbin": np.int8(0)}, index=pd.Index([1], name="source_id"))
    s2 = summary.star_summary(lc, cells=cells)
    assert s2["gbin"].tolist() == [2, s["gbin"].iloc[1], s["gbin"].iloc[2]]   # a catalogue's cell where it has one


def test_drift_table_is_applied():
    rng = np.random.default_rng(4)
    lc = lightcurves(rng, ("noise",) * 60, sigma=0.001)
    ep, stars = summary.corrected_epochs(lc)
    cells = stars[["gbin", "cbin"]]
    shifted = ep.assign(frac=ep["frac"] + 0.01 * (ep["t"] - 2200.0) / 365.25)     # a shared +1% a year
    table = drift.fit(shifted, cells, min_stars=10)
    plain = summary.star_summary(lc)
    shifted_lc = lc.set_column(lc.schema.get_field_index("flux"), "flux", pa.array(
        np.asarray(lc["flux"]) * (1 + 0.01 * (np.asarray(lc["t"]) - 1700.0) / 365.25)))
    with_table = summary.star_summary(shifted_lc, table)
    assert np.median(summary.star_summary(shifted_lc)["slope"]) == pytest.approx(0.01, abs=0.001)
    assert np.median(with_table["slope"]) == pytest.approx(0.0, abs=0.0005)
    assert np.median(plain["slope"]) == pytest.approx(0.0, abs=0.0005)


def test_stored_summary_round_trips(tmp_path):
    rng = np.random.default_rng(5)
    s = summary.star_summary(lightcurves(rng, KINDS * 2))
    summary.write(s.assign(full=True, ref=True), tmp_path / "s.parquet", extra=("full", "ref"))
    back = summary.from_store(pd.read_parquet(tmp_path / "s.parquet"))
    assert (back["gbin"] == s["gbin"]).all() and (back["cbin"] == s["cbin"]).all()
    for c in ("slope", "slope_err", "s_fade", "s_rise", "sigma_visit", "abbe", "g_mag", "slope_bp"):
        np.testing.assert_allclose(back[c].to_numpy(float), s[c].to_numpy(float), rtol=2e-6, equal_nan=True, err_msg=c)
    np.testing.assert_allclose(back["slope_z"], s["slope_z"], rtol=1e-5, equal_nan=True)
    strong = s["s_fade"].to_numpy() >= summary.STORE_MIN_SCORE
    assert strong.any() and (~strong).any()
    assert (back.loc[strong, "fade_kind"] == s.loc[strong, "fade_kind"]).all()
    np.testing.assert_allclose(back.loc[strong, "dim_t0"], s.loc[strong, "dim_t0"], rtol=1e-6)
    assert back.loc[~strong, "dim_t0"].isna().all() and (back.loc[~strong, "fade_kind"] == "").all()
    np.testing.assert_allclose(back["rep_bound"], np.fmax(s["rep_bound"], s["rep_bound_bright"]), rtol=2e-6,
                               equal_nan=True)
    assert back["full"].all() and back["n_visits"].dtype == np.int16


def stars_frame(rng, n, pixel=0, g=(13.0, 13.4), offset=0.0, scatter=0.002, err=0.002):
    """A summary-like frame of n stars in one level-6 pixel."""
    sid = pixel * LEVEL6 + rng.integers(1, LEVEL6, n)
    slope = offset + rng.normal(0, scatter, n)
    return pd.DataFrame({"source_id": sid, "g_mag": rng.uniform(*g, n), "slope": slope, "slope_err": err,
                         "slope_z": slope / err})


def test_reference_removes_an_offset_shared_by_neighbours_of_similar_brightness():
    rng = np.random.default_rng(6)
    bright = stars_frame(rng, 400, pixel=3, g=(12.0, 12.49), offset=0.002)       # +0.2% a year, one error
    faint = stars_frame(rng, 400, pixel=3, g=(16.0, 16.49), offset=0.0)
    other = stars_frame(rng, 400, pixel=9, g=(12.0, 12.49), offset=-0.001)
    s = pd.concat([bright, faint, other], ignore_index=True)
    ref = summary.reference_table(s)
    out = summary.apply_reference(s, ref)
    assert (out["ref_level"] == 6).all() and (out["n_ref"] == 400).all()
    b = out.iloc[:400]
    assert np.median(b["slope_z_raw"]) == pytest.approx(1.0, abs=0.15)
    assert np.median(b["slope_z"]) == pytest.approx(0.0, abs=1e-9)
    assert b["slope_ref"].iloc[0] == pytest.approx(0.002, abs=0.0003)
    assert out.iloc[800:]["slope_ref"].iloc[0] == pytest.approx(-0.001, abs=0.0003)
    # the median's error is added to the star's own
    assert 0 < b["slope_ref_err"].iloc[0] < 0.0003
    z = (b["slope"] - b["slope_ref"]) / np.hypot(b["slope_err"], b["slope_ref_err"])
    np.testing.assert_allclose(b["slope_z"], z)


def test_reference_falls_back_to_coarser_pixels_and_keeps_a_fake_out():
    rng = np.random.default_rng(7)
    # 8 level-6 pixels of one level-5 pixel with 20 stars each: only the level-5 cell reaches 50
    s = pd.concat([stars_frame(rng, 20, pixel=k, offset=0.001) for k in range(4, 8)]
                  + [stars_frame(rng, 20, pixel=k, offset=0.001) for k in range(4, 8)], ignore_index=True)
    ref = summary.reference_table(s)
    out = summary.apply_reference(s, ref)
    assert (out["ref_level"] == 5).all() and (out["n_ref"] == 160).all()
    # a fake made from one of these stars, 3% a year fainter, is measured against the real stars' table
    fake = s.iloc[[0]].assign(slope=s["slope"].iloc[0] - 0.03, g_mag=20.9)
    got = summary.apply_reference(fake, ref, g_mag=s["g_mag"].iloc[[0]].to_numpy())
    assert got["slope_ref"].iloc[0] == out["slope_ref"].iloc[0]
    assert got["slope_z"].iloc[0] < -10
    # no cell at all: the unreferenced significance is kept
    lone = stars_frame(rng, 5, pixel=900)
    none = summary.apply_reference(lone, summary.reference_table(lone))
    assert (none["n_ref"] == 0).all() and (none["slope_z"] == none["slope_z_raw"]).all()


def test_cell_sample_is_bounded_and_repeatable():
    rng = np.random.default_rng(8)
    s = pd.concat([stars_frame(rng, 500, pixel=1), stars_frame(rng, 30, pixel=70)], ignore_index=True)
    a = summary.cell_sample(s, level=6, per_cell=100)
    b = summary.cell_sample(s.sample(frac=1.0, random_state=1), level=6, per_cell=100)
    assert len(a) == 130 and set(a["source_id"]) == set(b["source_id"])
    again = summary.cell_sample(a, level=6, per_cell=100)
    assert set(again["source_id"]) == set(a["source_id"])                # sampling a sample changes nothing


def test_score_all_flags_the_stars_the_detector_on_every_star_flags():
    rng = np.random.default_rng(9)
    lc = lightcurves(rng, KINDS * 8 + ("noise",) * 60, flux=10 ** ((25.6874 - 14.2) / 2.5))
    thr, thr_p, table = rule_v3.load()
    summ, stats, keep = search.score_all(lc, table, thr, thr_p)
    assert len(summ) == len(KINDS) * 8 + 60
    assert summ["full"].sum() == len(stats) < len(summ)                   # the quiet majority needs no more
    ref = summary.reference_table(summ)
    new = search.merge(summary.apply_reference(summ, ref), stats)
    cells = summ.set_index("source_id")[["gbin", "cbin"]]
    old = search.score(lc, cells, table).merge(cells.reset_index(), on="source_id")
    old = summary.apply_reference(old.merge(summ[["source_id", "g_mag"]], on="source_id"), ref)
    shared = rule_v3.shared(old)
    f_old = search.flag(old, shared_flags=shared).set_index("id")
    f_new = search.flag(new[new["sigma_visit"].notna()].reset_index(drop=True)).set_index("id").loc[f_old.index]
    for c in ("flagged", "early", "rising", "rising_rep", "shared_fade", "shared_rise"):
        assert (f_new[c].to_numpy() == f_old[c].to_numpy()).all(), c
    assert f_old["flagged"].sum() >= 24 and f_old["rising"].sum() >= 16
    assert (f_new.loc[f_old["flagged"], "by"] == f_old.loc[f_old["flagged"], "by"]).all()
    # light curves are kept for the fading stars, not for the brightening ones
    fading = f_old.index[f_old["flagged"]]
    assert set(fading) <= set(keep)
    rising_only = f_old.index[f_old["rising"] & ~f_old["flagged"] & (f_old["strength"] < 0.5)]
    assert len(rising_only) and not set(rising_only) & set(keep)


def test_period_search_runs_only_where_the_bound_allows():
    rng = np.random.default_rng(10)
    lc = lightcurves(rng, ("noise",) * 80 + ("repeat",) * 4, flux=10 ** ((25.6874 - 14.2) / 2.5))
    thr, thr_p, table = rule_v3.load()
    summ, stats, _ = search.score_all(lc, table, thr, thr_p)
    n_searched = int(stats["s_rep_any"].notna().sum())
    assert 4 <= n_searched < 30
    rep = stats.set_index("source_id").loc[[81, 82, 83, 84], "s_rep"]
    assert (rep > thr_p["G<19"]).all()
    faint = lightcurves(rng, ("repeat",) * 4, flux=10 ** ((25.6874 - 17.0) / 2.5))
    _, st, _ = search.score_all(faint, table, thr, thr_p)
    assert "s_rep" not in st or st["s_rep"].isna().all()                  # fainter than PERIOD_G_MAX: not searched
    _, st, _ = search.score_all(faint, table, thr, thr_p, period_g_max=18.0)
    assert (st["s_rep"] > thr_p["G<19"]).all()


def test_star_kind_places_stars_on_the_colour_magnitude_diagram():
    g = pd.DataFrame({
        "phot_g_mean_mag": [12.0, 8.0, 15.0, 12.0, 17.0, 18.0, 12.0, 12.0, 19.5],
        "parallax":        [10.0, 10.0, 10.0, 1.0, 20.0, 0.5, np.nan, 10.0, 0.01],
        "parallax_error":  [0.02, 0.02, 0.02, 0.02, 0.05, 0.3, np.nan, 0.02, 0.5],
        "bp_rp":           [0.8, 0.2, 2.5, 1.2, 0.3, 1.0, 1.0, np.nan, 0.6]})
    still = np.array([False] * 8 + [True])
    k = kinds.star_kind(g, still)
    assert k.tolist() == ["FGK dwarf", "hot dwarf", "M dwarf", "giant or subgiant", "white dwarf or subluminous",
                          "parallax too uncertain", "no astrometry", "no colour", "no measurable motion"]
    assert kinds.absolute_magnitude(12.0, 10.0) == pytest.approx(7.0)
    assert set(k) <= set(kinds.READ_ORDER)


def test_rank_measures_fades_against_the_same_kind_of_stars_brightening():
    rng = np.random.default_rng(11)
    n = 400
    # dwarfs: fades and brightenings equally common, plus three clear fades; giants: as many strong either way
    dwarf_f, dwarf_r = rng.exponential(0.5, n), rng.exponential(0.5, n)
    dwarf_f[:3] = [9.0, 8.0, 7.0]
    dwarf_r[:3] = 0.1
    giant_f, giant_r = rng.exponential(3.0, n), rng.exponential(3.0, n)
    giant_r[:5] = 60.0                                        # and their strongest changes are brightenings
    stars = pd.DataFrame({"kind": ["FGK dwarf"] * n + ["giant or subgiant"] * n,
                          "strength": np.r_[dwarf_f, giant_f], "rise_strength": np.r_[dwarf_r, giant_r],
                          "phot_g_mean_mag": rng.uniform(10, 18, 2 * n)})
    r = kinds.rank(stars)
    assert (r["q_kind"].iloc[:3] < 0.5).all()                 # 3 fades against no brightening that strong
    assert r["n_fade_kind"].iloc[2] == 3 and r["n_rise_kind"].iloc[2] == 0
    assert r["q_kind"].iloc[n:].min() > 0.5                   # giants: nothing stands clear of their brightenings
    order = r["read_order"].to_numpy()
    assert sorted(order[:3].tolist()) == [0, 1, 2]            # the three dwarfs come first
    assert order[:n].max() < order[n:].min()                  # every dwarf before any giant
    assert sorted(order.tolist()) == list(range(2 * n))
    classified = np.zeros(2 * n, bool)
    classified[0] = True
    r2 = kinds.rank(stars, classified=classified)
    assert r2["read_order"].iloc[0] > r2["read_order"].iloc[3:n].max()   # classified: after its kind's others
    c = kinds.counts(stars)
    assert c["FGK dwarf"]["5.0"]["fading"] == 3 and c["FGK dwarf"]["5.0"]["rising"] == 0
