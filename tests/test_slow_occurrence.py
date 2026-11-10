import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_search.gaia import epoch
from cosmos_search.inject import lightcurve
from cosmos_search.slow import detect, drift, occurrence as oc, recall

YEAR = 365.25
REPO = Path(__file__).resolve().parents[1]


# -- the limit ----------------------------------------------------------------------------------------------------

def test_no_counts_gives_the_poisson_limit():
    assert oc.single_limit(0, 0) == pytest.approx(3.0, abs=0.01)
    # Stars that brighten and none that fade: still 3.0, never below.
    assert oc.single_limit(0, 12) == pytest.approx(3.0, abs=0.01)


def test_no_brightening_stars_counts_every_fading_star():
    for n in (1, 4, 20):
        assert oc.single_limit(n, 0) == pytest.approx(float(oc.poisson_upper(n)), rel=2e-3)


def test_brightening_stars_are_subtracted():
    # 200 fade and 200 brighten: the limit is about 1.645 sqrt(400) plus the truncation at zero, far below 200.
    lim = oc.single_limit(200, 200)
    assert 30 < lim < 50
    assert oc.single_limit(240, 200) > lim > oc.single_limit(170, 200)


@pytest.mark.parametrize("b, s", [(0.3, 4.0), (3.0, 10.0), (40.0, 25.0), (400.0, 80.0)])
def test_single_limit_covers(b, s):
    rng = np.random.default_rng(int(10 * b + s))
    toys = 400
    miss = sum(oc.single_limit(int(rng.poisson(b + s)), int(rng.poisson(b)), grid=801) < s for _ in range(toys))
    # 95% nominal; 400 draws give +-1.1%, so a true 95% fails this about once in 10^4.
    assert miss / toys < 0.09


def _synthetic_cells(speed=1.0):
    """Chance fades per (G range, significance bin) of the frozen grid and the stars a fade would put in each:
    bright stars far out in significance, faint ones near the threshold, chance fades falling with significance
    as a power law (about what DR3 shows). speed: how far out the fade puts its stars (1: bright stars at 60
    errors, where no star is by chance; 0.1: at 6, among the chance fades)."""
    from scipy import stats

    cfg = oc.Config()
    edges = np.r_[cfg.z_cuts, np.inf]
    tail = (edges[:-1] / 3.0) ** -3.5 - (edges[1:] / 3.0) ** -3.5          # share of chance fades in each bin
    b = np.outer([1, 2, 4, 8, 12, 10, 3], 12.5 * tail)
    stars, typical = [100, 200, 400, 700, 900, 600, 100], [60, 50, 35, 20, 10, 5, 3.5]
    a = np.array([n * np.diff(stats.norm.cdf(np.log(edges / (speed * z0)) / 0.5)) for n, z0 in zip(stars, typical)])
    return b, a


@pytest.mark.parametrize("speed, scale, signal, asymmetry", [
    (1.0, 1.0, 12.0, 1.0), (1.0, 0.0, 20.0, 1.0), (0.3, 1.0, 50.0, 1.0), (0.3, 20.0, 80.0, 0.6),
    (0.1, 20.0, 80.0, 1.5), (0.1, 20.0, 300.0, 0.6)])
def test_region_choice_and_limit_cover_together(speed, scale, signal, asymmetry):
    """The region is chosen on the brightening counts and the limit computed in it from the fading counts: the
    pair must cover, also where stars fade by chance less or more often than they brighten."""
    b, a = _synthetic_cells(speed)
    b = b * scale
    f = signal / a.sum()
    rng = np.random.default_rng(int(7 + signal + 10 * asymmetry))
    toys, miss = 600, 0
    det = oc.regions(a)
    for _ in range(toys):
        n, m = oc.regions(rng.poisson(asymmetry * b + f * a)), oc.regions(rng.poisson(b))
        k, j = oc.choose_region(m, det)
        miss += float(oc.poisson_upper(n[k, j])) / det[k, j] < f
    # 95% nominal: 600 draws give +-0.9%.
    assert miss / toys < 0.075


@pytest.mark.parametrize("speed, scale, signal", [(1.0, 1.0, 0.0), (0.3, 1.0, 50.0), (0.1, 20.0, 80.0)])
def test_subtracted_limit_covers_when_chance_fades_mirror_chance_brightenings(speed, scale, signal):
    """With perfect symmetry the subtracted limit covers and, with no real fades, the excess test fires no more
    often than its level."""
    b, a = _synthetic_cells(speed)
    b = b * scale
    f = signal / a.sum()
    rng = np.random.default_rng(int(7 + signal))
    toys, miss, excess = 400, 0, 0
    det = oc.regions(a)
    for _ in range(toys):
        n, m = oc.regions(rng.poisson(b + f * a)).astype(int), oc.regions(rng.poisson(b)).astype(int)
        k, j = oc.choose_mirror_region(n + m, det)
        miss += oc.single_limit(int(n[k, j]), int(m[k, j]), grid=801) / det[k, j] < f
        excess += oc.fade_excess_p(int(n[k, j]), int(m[k, j])) < 0.05
    # 95% nominal: 400 draws give +-1.1%.
    assert miss / toys < 0.085
    if signal == 0:
        assert excess / toys < 0.085


def test_subtracted_limit_fails_without_symmetry():
    """Why it is not the limit: with chance fades at 0.6 of the chance brightenings it sits below the truth."""
    b, a = _synthetic_cells(0.1)
    b, f = b * 20.0, 80.0 / a.sum()
    rng = np.random.default_rng(3)
    det, miss = oc.regions(a), 0
    for _ in range(100):
        n, m = oc.regions(rng.poisson(0.6 * b + f * a)).astype(int), oc.regions(rng.poisson(b)).astype(int)
        k, j = oc.choose_mirror_region(n + m, det)
        miss += oc.single_limit(int(n[k, j]), int(m[k, j]), grid=501) / det[k, j] < f
    assert miss > 40


def test_region_choices_do_not_look_at_the_fading_counts():
    b, a = _synthetic_cells()
    rng = np.random.default_rng(2)
    n, m, det = oc.regions(rng.poisson(b)).astype(int), oc.regions(rng.poisson(b)).astype(int), oc.regions(a)
    where = oc.choose_region(m, det)
    assert where is not None and det[where] > 0
    assert oc.choose_region(m, np.zeros_like(det)) is None
    # With no star past any cut the widest region wins: it holds the most detectable stars.
    assert oc.choose_region(np.zeros_like(m), det) == (det.shape[0] - 1, 0)
    # The subtracted limit's region: swapping fading and brightening changes nothing.
    assert oc.choose_mirror_region(n + m, det) == oc.choose_mirror_region(m + n, det)
    assert oc.choose_mirror_region(n + m, det) is not None


def test_large_counts_use_the_normal_form_and_err_high():
    exact = oc.single_limit(500, 500)
    assert oc.single_limit(501, 500) == pytest.approx(exact + 1.0, abs=1.0)
    assert oc.single_limit(10 ** 6, 10 ** 6) == pytest.approx(1.96 * np.sqrt(2e6), rel=0.01)


def test_fade_excess_p_is_the_binomial_tail():
    assert oc.fade_excess_p(0, 0) == 1.0
    assert oc.fade_excess_p(10, 0) == pytest.approx(2.0 ** -10)
    assert oc.fade_excess_p(5, 5) == pytest.approx(0.623, abs=1e-3)
    assert oc.fade_excess_p(3, 30) > 0.99


def test_expected_limits_bracket_an_even_split():
    lo, med, hi = oc.expected_limits(40, 60)
    assert lo < med < hi
    assert med == pytest.approx(oc.single_limit(50, 50))
    assert oc.expected_limits(0, 0) == (pytest.approx(3.0, abs=0.01),) * 3


def test_reviewed_limit():
    assert oc.reviewed_limit(0, 10_000, review_recall=0.88) == pytest.approx(3.0 / 8800, rel=2e-3)
    assert oc.reviewed_limit(2, 10_000) == pytest.approx(float(oc.poisson_upper(2)) / 10_000)


# -- the stars ----------------------------------------------------------------------------------------------------

def test_fgk_dwarf_places_stars_by_parallax_and_colour():
    aux = pd.DataFrame({
        #                 Sun at 100 pc, K giant, white dwarf, M dwarf, Sun with poor parallax, A star, no parallax
        "phot_g_mean_mag": [9.67, 8.0, 17.0, 15.0, 17.0, 7.0, 15.0],
        "bp_rp":           [0.82, 1.2, 0.3, 2.5, 0.82, 0.1, 0.9],
        "parallax":        [10.0, 2.0, 10.0, 10.0, 0.4, 5.0, np.nan],
        "parallax_error":  [0.02, 0.02, 0.1, 0.05, 0.2, 0.02, np.nan]})
    assert oc.fgk_dwarf(aux).tolist() == [True, False, False, False, False, False, False]
    assert oc.absolute_magnitude(9.67, 10.0) == pytest.approx(4.67)


def test_frozen_config_is_the_code_default():
    assert oc.Config.load() == oc.Config()
    assert oc.Config.load(oc.CONFIG_DIR / "steady-fades-v1.json") == oc.Config()


def test_strata_and_bins():
    cfg = oc.Config()
    assert oc.stratum([5.0, 12.99, 13.0, 18.99, 19.0, np.nan], cfg).tolist() == [0, 0, 1, 6, -1, -1]
    assert oc.zbin([2.9, 3.0, 9.9, 30.0, np.nan], cfg).tolist() == [-1, 0, 4, 10, -1]
    n, m, stars = oc.cell_counts(np.array([0, 0, 3, 3, -1]), np.array([-3.5, 3.5, -30.0, 0.0, -50.0]), cfg)
    assert n[0, 0] == 1 and m[0, 0] == 1 and n[3, 10] == 1 and n.sum() == 2 and m.sum() == 1
    assert stars.tolist() == [2, 0, 0, 2, 0, 0, 0]
    r = oc.regions(n)
    assert r[0, 0] == 1 and r[3, 0] == 2 and r[3, 10] == 1 and r[6, 5] == 1 and r[2, 1] == 0 and r[3, 11] == 0


def test_efficiency_is_the_share_past_the_cut_and_the_net_share_for_the_subtraction():
    cfg = oc.Config()
    # Four hosts in one G range: one pushed past -5, one that brightened past +5 anyway, one missed, one unscored.
    share, net, hosts = oc.efficiency(np.zeros(4, int), np.array([-5.5, 5.5, -1.0, np.nan]), cfg)
    assert hosts[0] == 4
    assert share[0].sum() == pytest.approx(0.25)
    assert net[0].sum() == pytest.approx(0.0)        # +1/4 fading and -1/4 brightening in the same bin
    share, net, _ = oc.efficiency(np.zeros(4, int), np.array([-5.5, -7.0, -1.0, np.nan]), cfg)
    assert share[0].sum() == net[0].sum() == pytest.approx(0.5)


def test_brightness_reference_takes_out_an_offset_that_depends_on_g():
    rng = np.random.default_rng(3)
    n = 20_000
    g = rng.uniform(11, 19, n)
    err = 0.0005 * 10 ** (0.2 * (g - 13).clip(0))
    offset = np.where(g < 13, 8e-4, np.where(g < 15, 3e-4, 0.0))
    stars = pd.DataFrame({"source_id": np.arange(n), "phot_g_mean_mag": g, "slope": offset + rng.normal(0, err)})
    stars["slope_z"] = stars["slope"] / err
    ref = oc.brightness_reference(g, stars["slope"].to_numpy())
    z = oc.trend_z(stars, g, ref)
    for lo, hi in ((11, 13), (13, 15), (15, 19)):
        k = (g >= lo) & (g < hi)
        assert abs(np.median(z[k])) < 0.06
        assert np.std(z[k]) == pytest.approx(1.0, abs=0.05)
    assert np.median(stars["slope_z"][g < 13]) > 1.0     # before the reference: +1.6 errors
    # A table that already carries a neighbour reference keeps its own significance.
    assert oc.trend_z(stars.assign(slope_ref=0.0), g, ref).tolist() == stars["slope_z"].tolist()


# -- recall -------------------------------------------------------------------------------------------------------

def lightcurves(n_stars, rng, sigma=0.001, n_epochs=60, span=3 * YEAR):
    rows = []
    for i in range(n_stars):
        t = np.sort(rng.uniform(1700, 1700 + span, n_epochs))
        for band in ("G", "BP", "RP"):
            f = 1 + rng.normal(0, sigma, t.size)
            rows.append(pd.DataFrame({"source_id": i + 1, "band": band, "t": t, "flux": 1e4 * f,
                                      "err": np.full(t.size, 1e4 * sigma), "flags": 0,
                                      "transit_id": np.arange(t.size), "fov": 1, "ccd_row": 1, "n_obs": 9}))
    return pa.Table.from_pandas(pd.concat(rows, ignore_index=True), schema=epoch.SCHEMA, preserve_index=False)


def detector(lc):
    """The detector on uncorrected fluxes, standing in for search.score."""
    ep = drift.normalise(lc)
    return detect.star_features(ep.assign(frac_corr=ep["frac"]))


def test_injected_fade_is_the_shared_injector_from_each_stars_first_epoch():
    rng = np.random.default_rng(5)
    lc = lightcurves(3, rng)
    out = recall.inject_steady_fade(lc, 0.02)
    assert out.schema == lc.schema and out.num_rows == lc.num_rows
    d0, d1 = lc.to_pandas(), out.to_pandas()
    for sid, d in d0.groupby("source_id"):
        sig = lightcurve.SecularTrend(rate_per_year=-0.02, t_ref=d["t"].min())
        flux, err = lightcurve.inject(d["t"].to_numpy(), d["flux"].to_numpy(), d["err"].to_numpy(), "G", sig)
        got = d1[d1["source_id"] == sid]
        assert np.allclose(got["flux"], flux) and np.allclose(got["err"], err)
    with pytest.raises(ValueError):
        recall.inject_steady_fade(lc, 0.5)   # reaches zero flux within three years


def test_injected_scores_recover_the_rate_and_keep_every_host():
    rng = np.random.default_rng(6)
    lc = lightcurves(30, rng)
    out = recall.injected_scores(lc, (0.0, 0.01), detector, columns=["slope", "slope_z"])
    assert len(out) == 60 and set(out.columns) == {"source_id", "rate", "slope", "slope_z"}
    quiet, faded = out[out["rate"] == 0.0], out[out["rate"] == 0.01]
    assert abs(quiet["slope"].median()) < 3e-4
    # 1% of the starting flux a year is slightly more than 1% of the median flux a year.
    assert faded["slope"].median() == pytest.approx(-0.01 / (1 - 0.015), rel=0.03)
    assert (faded["slope_z"] < -10).all()
    # A scorer that returns nothing for some hosts: they keep their rows, unscored.
    some = recall.injected_scores(lc, (0.01,), lambda t: detector(t).iloc[:10])
    assert len(some) == 30 and some["slope"].isna().sum() == 20


# -- end to end ---------------------------------------------------------------------------------------------------

def population(n, rng, faders=0, rate=0.02):
    """Trends of n FGK dwarfs with errors that grow towards faint stars, a heavy-tailed symmetric scatter standing in
    for real variability, a brightness-dependent offset, and `faders` stars fading at `rate`."""
    g = rng.uniform(11.5, 18.9, n)
    err = 0.0006 * 10 ** (0.25 * (g - 15).clip(0))
    own = rng.normal(0, 1, n) * np.where(rng.random(n) < 0.03, 4.0, 1.0)
    slope = np.where(g < 13, 6e-4, 0.0) + own * err
    slope[:faders] -= rate
    return pd.DataFrame({"source_id": np.arange(n), "phot_g_mean_mag": g, "slope": slope, "slope_z": slope / err,
                         "n_visits": 40, "t_first": 1700.0, "t_last": 1700 + 5.5 * YEAR})


def injected(stars, cfg):
    """Each star again at each rate with the fade added to its trend (the error unchanged)."""
    err = oc.slope_error(stars)
    return pd.concat([stars[["source_id"]].assign(rate=r, slope=stars["slope"] - r, slope_z=(stars["slope"] - r) / err)
                      for r in cfg.rates], ignore_index=True)


def test_limit_curve_without_faders():
    cfg = oc.Config()
    stars = population(30_000, np.random.default_rng(11))
    curve, cells, sym = oc.limit_curve(stars, injected(stars, cfg), cfg)
    assert curve["rate"].tolist() == list(cfg.rates)
    # 10% a year: every star would pass a cut that almost no star reaches by chance.
    top = curve.iloc[-1]
    assert top["recall"] > 0.99 and top["fading"] <= 2 and top["brightening"] == 0
    assert top["limit"] == pytest.approx(float(oc.poisson_upper(top["fading"])) / 30_000, rel=0.02)
    # Slower fades are caught in fewer stars and among more chance fades: the limit weakens.
    assert np.all(np.diff(curve["limit"]) <= 0)
    assert curve["limit"].iloc[0] > 30 * top["limit"]
    assert (curve["excess_p"] > 0.01).all()
    # A symmetric population: subtracting the chance fades helps where there are any, and only there.
    assert (curve["mirror_limit"] <= curve["limit"] * 1.3).all()
    assert curve["mirror_limit"].iloc[0] < 0.6 * curve["limit"].iloc[0]
    assert ((curve["mirror_expected_lo"] <= curve["mirror_expected"])
            & (curve["mirror_expected"] <= curve["mirror_expected_hi"])).all()
    assert abs(sym["median_z"]).max() < 0.1 and sym["median_z_raw"].iloc[0] > 0.5
    assert cells.groupby("rate")["chosen"].sum().eq(1).all() and cells.groupby("rate")["chosen_mirror"].sum().eq(1).all()
    assert (cells["recall"] >= cells["recall_net"]).all()


def test_limit_curve_does_not_hide_real_faders():
    cfg = oc.Config()
    stars = population(30_000, np.random.default_rng(12), faders=60, rate=0.02)
    curve, _, _ = oc.limit_curve(stars, injected(stars, cfg), cfg)
    row = curve[np.isclose(curve["rate"], 0.02)].iloc[0]
    assert row["excess_p"] < 1e-6 and row["fading"] >= 30 and row["brightening"] <= 3
    assert row["limit"] > row["expected"] * 3      # far more fading stars than the brightening ones would suggest
    for col in ("limit", "mirror_limit"):          # both sit above the true share of faders
        assert 60 / 30_000 < row[col] < 3 * 60 / 30_000


def test_injected_stars_take_their_hosts_neighbour_reference():
    cfg = oc.Config()
    stars = population(5_000, np.random.default_rng(23))
    err = oc.slope_error(stars)
    # A per-star table that carries its own reference (the bulk-file summary): slope_z is already against it.
    ref, ref_err = np.where(stars["phot_g_mean_mag"] < 13, 6e-4, 0.0), 0.5 * err
    table = stars.assign(slope_err=err, slope_ref=ref, slope_ref_err=ref_err, slope_z_raw=stars["slope_z"],
                         slope_z=(stars["slope"] - ref) / np.hypot(err, ref_err))
    inj = injected(stars, cfg).assign(slope_err=np.tile(err, len(cfg.rates)))
    (z, z_raw, _), (inj_z, _) = oc.significances(table, inj, cfg)
    assert np.allclose(z, table["slope_z"]) and np.allclose(z_raw, stars["slope_z"])
    first = np.isclose(inj["rate"], cfg.rates[0])
    assert np.allclose(inj_z[first], (stars["slope"] - cfg.rates[0] - ref) / np.hypot(err, ref_err))
    # Injections scored against a reference themselves keep their own significance.
    own = inj.assign(slope_ref=0.0, slope_z=1.0)
    assert (oc.significances(table, own, cfg)[1][0] == 1.0).all()


def test_sky_dispersion_is_near_one_for_independent_stars_and_large_for_grouped_ones():
    rng = np.random.default_rng(21)
    pix = rng.integers(0, 40, 2000)
    ratio, p = oc.sky_dispersion(pix, rng.random(2000) < 0.5)
    assert 0.6 < ratio < 1.5 and p > 0.01
    # Each patch of sky leans one way: three stars in four share their pixel's direction.
    lean = rng.random(40) < 0.5
    grouped = np.where(rng.random(2000) < 0.75, lean[pix], ~lean[pix])
    ratio, p = oc.sky_dispersion(pix, grouped)
    assert ratio > 5 and p < 0.01
    assert np.isnan(oc.sky_dispersion(pix[:0], grouped[:0])[0])
    assert np.isnan(oc.sky_dispersion(pix, np.ones(2000, bool))[0])


def test_plant_swaps_in_the_injected_scores_of_a_share_of_stars():
    cfg = oc.Config()
    stars = population(20_000, np.random.default_rng(22))
    inj = injected(stars, cfg)
    out, k = oc.plant(stars, inj, 0.02, 0.01, np.random.default_rng(0))
    assert 150 < k < 250 and len(out) == len(stars)
    moved = ~np.isclose(out.set_index("source_id")["slope"], stars.set_index("source_id")["slope"])
    assert moved.sum() == k
    assert np.allclose((stars.set_index("source_id")["slope"] - out.set_index("source_id")["slope"])[moved], 0.02)
    # The limit stays above the planted share and the excess test sees it.
    curve, _, _ = oc.limit_curve(out, inj, cfg, rates=(0.02,))
    row = curve.iloc[0]
    assert row["limit"] > k / len(stars) and row["excess_p"] < 1e-6
    assert oc.plant(stars, inj, 0.02, 0.0, np.random.default_rng(0))[1] == 0


def test_script_inject_and_limit(tmp_path):
    spec = importlib.util.spec_from_file_location("slow_occurrence", REPO / "scripts" / "slow_occurrence.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    rng = np.random.default_rng(13)
    n = 60
    lc = lightcurves(n, rng)
    (tmp_path / "lc").mkdir()
    pq.write_table(lc, tmp_path / "lc" / "a.parquet")
    aux = pd.DataFrame({"source_id": np.arange(1, n + 1), "phot_g_mean_mag": 14.5, "bp_rp": 0.9,
                        "parallax": 2.5, "parallax_error": 0.03})
    aux.loc[:9, "parallax"] = 0.01           # ten giants by parallax: not hosts, not searched
    aux.to_parquet(tmp_path / "aux.parquet", index=False)
    # A rule directory with an empty drift table: nothing to correct in synthetic light curves.
    rule = tmp_path / "rule"
    rule.mkdir()
    for name in ("thresholds.json", "period_thresholds.json"):
        (rule / name).write_text((script.rule_v3.RULE_DIR / name).read_text())
    pd.DataFrame({"band": pd.Series(dtype="int8"), "gbin": pd.Series(dtype="int8"), "cbin": pd.Series(dtype="int8"),
                  "tbin": pd.Series(dtype="int32"), "drift": pd.Series(dtype="float64"),
                  "n_stars": pd.Series(dtype="int64")}).to_parquet(rule / "drift_table.parquet", index=False)
    out = tmp_path / "occ"
    assert script.main(["inject", "--lightcurves", str(tmp_path / "lc"), "--aux", str(tmp_path / "aux.parquet"),
                        "--rule", str(rule), "--out", str(out), "--workers", "1"]) == 0
    inj = pd.read_parquet(out / "injections" / "a.parquet")
    assert inj["source_id"].nunique() == 50 and len(inj) == 50 * len(oc.Config().rates)
    # The real stars' scores, as slow_search.py score writes them.
    from cosmos_search.slow import search
    stats = search.score(lc, drift.star_cells(aux), pd.read_parquet(rule / "drift_table.parquet"), period_search=False)
    (tmp_path / "stats").mkdir()
    stats.to_parquet(tmp_path / "stats" / "a.parquet", index=False)
    assert script.main(["limit", "--stars", str(tmp_path / "stats"), "--aux", str(tmp_path / "aux.parquet"),
                        "--injections", str(out / "injections"), "--out", str(out / "limit")]) == 0
    rec = json.loads((out / "limit" / "limit.json").read_text())
    assert rec["inputs"]["fgk_dwarfs_searched"] == 50 and rec["inputs"]["stars_scored"] == 60
    fast = [r for r in rec["curve"] if r["rate"] >= 0.01]
    assert all(r["recall"] > 0.9 and r["limit"] == pytest.approx(3.0 / (50 * r["recall"]), rel=0.02) for r in fast)
    # Planted fades in the same stars: one star in five given its own injected scores at 2% a year.
    assert script.main(["planted", "--stars", str(tmp_path / "stats"), "--aux", str(tmp_path / "aux.parquet"),
                        "--injections", str(out / "injections"), "--out", str(out / "planted"), "--rates", "0.02",
                        "--shares", "0", "0.2", "--trials", "3"]) == 0
    check = pd.read_csv(out / "planted" / "planted.csv")
    assert check["trials"].tolist() == [3, 3] and check["excess_p_below_0.05"].tolist() == [0, 3]
    assert check["limit_below_planted"].tolist() == [0, 0]


def test_limit_counts_a_referenced_table_file_by_file(tmp_path):
    """A per-star table with its own neighbour reference and kind of star, as the bulk-file pass writes it, in
    two files: the limit step tallies them one at a time and gets what the in-memory route gets."""
    spec = importlib.util.spec_from_file_location("slow_occurrence", REPO / "scripts" / "slow_occurrence.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    cfg = oc.Config()
    stars = population(6_000, np.random.default_rng(31), faders=20, rate=0.02)
    err = oc.slope_error(stars)
    ref, ref_err = np.where(stars["phot_g_mean_mag"] < 13, 6e-4, 0.0), 0.3 * err
    table = stars.assign(slope_err=err, slope_ref=ref, slope_ref_err=ref_err, slope_z_raw=stars["slope_z"],
                         slope_z=(stars["slope"] - ref) / np.hypot(err, ref_err),
                         kind=np.where(np.arange(len(stars)) % 12 == 5, "giant", "FGK dwarf"))
    inj = injected(stars, cfg).assign(slope_err=np.tile(err, len(cfg.rates)))
    (tmp_path / "summary").mkdir()
    (tmp_path / "inj").mkdir()
    on_disk = table.rename(columns={"phot_g_mean_mag": "g_mag"})
    on_disk.iloc[:2_500].to_parquet(tmp_path / "summary" / "a.parquet", index=False)
    on_disk.iloc[2_500:].to_parquet(tmp_path / "summary" / "b.parquet", index=False)
    inj.to_parquet(tmp_path / "inj" / "a.parquet", index=False)
    assert script.main(["limit", "--stars", str(tmp_path / "summary"), "--kind", "FGK dwarf",
                        "--injections", str(tmp_path / "inj"), "--out", str(tmp_path / "limit")]) == 0
    rec = json.loads((tmp_path / "limit" / "limit.json").read_text())
    dwarfs = table[table["kind"] == "FGK dwarf"].drop(columns="kind")
    assert rec["inputs"]["neighbour_reference"] and rec["inputs"]["stars_scored"] == 6_000
    assert rec["inputs"]["fgk_dwarfs_searched"] == len(dwarfs) == rec["inputs"]["injection_hosts"]
    curve, _, sym = oc.limit_curve(dwarfs, inj, cfg)
    got = pd.DataFrame(rec["curve"])
    for col in ("limit", "mirror_limit", "fading", "brightening", "recall", "excess_p", "g_max", "z_min"):
        assert np.allclose(got[col], curve[col]), col
    assert pd.DataFrame(rec["symmetry"])["stars"].tolist() == sym["stars"].tolist()
    past = pd.read_parquet(tmp_path / "limit" / "past_cut.parquet")
    assert (past["z"].abs() >= cfg.z_cuts[0]).all() and past["z"].is_monotonic_increasing
    assert len(past) == sym["fading"].sum() + sym["brightening"].sum()
    # Planted fades in a table with its own reference are measured against it too.
    out, k = oc.plant(dwarfs, inj, 0.02, 0.05, np.random.default_rng(1))
    moved = out.set_index("source_id").loc[lambda d: ~np.isclose(d["slope"], dwarfs.set_index("source_id")["slope"])]
    assert len(moved) == k > 100
    assert np.allclose(moved["slope_z"], (moved["slope"] - moved["slope_ref"])
                       / np.hypot(moved["slope_err"], moved["slope_ref_err"]))
    assert np.allclose(moved["slope_z_raw"], moved["slope"] / moved["slope_err"])


def test_tallies_add_up():
    cfg = oc.Config()
    rng = np.random.default_rng(32)
    strat, z = rng.integers(-1, cfg.n_strata, 4_000), rng.normal(0.3, 2.0, 4_000)
    whole = oc.tally(strat, z, z + 0.5, cfg)
    parts = oc.tally(strat[:1_500], z[:1_500], z[:1_500] + 0.5, cfg) + oc.tally(strat[1_500:], z[1_500:], z[1_500:] + 0.5, cfg)
    for name in ("fading", "brightening", "stars", "side_fading", "side_brightening", "hist", "hist_raw"):
        assert np.array_equal(getattr(whole, name), getattr(parts, name)), name
    sym = oc.symmetry(whole, cfg)
    for s in range(cfg.n_strata):
        assert sym["median_z"][s] == pytest.approx(np.median(z[strat == s]), abs=0.05)
        assert sym["median_z_raw"][s] == pytest.approx(np.median(z[strat == s]) + 0.5, abs=0.05)
        assert sym["side_fading"][s] == np.sum((strat == s) & (-z >= 2) & (-z < 3))
