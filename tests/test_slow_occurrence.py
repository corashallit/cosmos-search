import importlib.util
import json
import sys
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
    """The detector on uncorrected fluxes, standing in for the search's scoring of a light-curve table."""
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


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod            # so that worker processes can find the script's functions
    spec.loader.exec_module(mod)
    return mod


def test_script_pass_inject_and_limit(tmp_path):
    """The steps as they run on a release: the search's pass and flag steps give the per-star table and the
    neighbour reference, the inject step scores faded copies of the FGK dwarfs with the pass's own function, and
    the limit and planted steps read all three."""
    script, search_script = _script("slow_occurrence"), _script("slow_search")
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
    run, lists, out = tmp_path / "run", tmp_path / "lists", tmp_path / "occ"
    assert search_script.main(["pass", "--out", str(run), "--lightcurves", str(tmp_path / "lc"), "--rule", str(rule),
                               "--no-period", "--workers", "1"]) == 0
    assert search_script.main(["flag", "--run", str(run), "--rule", str(rule), "--out", str(lists)]) == 0
    stored = pd.read_parquet(run / "summary" / "a.parquet")
    assert len(stored) == n and "slope_z" not in stored and "slope_ref" not in stored and stored["ref"].all()
    assert script.main(["inject", "--lightcurves", str(tmp_path / "lc"), "--aux", str(tmp_path / "aux.parquet"),
                        "--rule", str(rule), "--out", str(out), "--workers", "1"]) == 0
    inj = pd.read_parquet(out / "injections" / "a.parquet")
    assert inj["source_id"].nunique() == 50 and len(inj) == 50 * len(oc.Config().rates)
    # An injected star is scored by the pass's own function: without a fade it would have its host's row.
    host = stored.set_index("source_id").loc[inj["source_id"]]
    slow = inj["rate"].to_numpy() == 0.001
    assert np.allclose(inj["slope"][slow], host["slope"][slow].to_numpy() - 0.001, atol=2e-5)
    assert np.allclose(inj["g_mag"][slow], host["g_mag"][slow].to_numpy(), atol=0.01)
    args = ["--stars", str(run / "summary"), "--reference", str(lists / "reference.parquet"),
            "--aux", str(tmp_path / "aux.parquet"), "--injections", str(out / "injections")]
    assert script.main(["limit", *args, "--out", str(out / "limit")]) == 0
    rec = json.loads((out / "limit" / "limit.json").read_text())
    assert rec["inputs"]["fgk_dwarfs_searched"] == 50 and rec["inputs"]["stars_scored"] == 60
    assert rec["inputs"]["neighbour_reference"] and rec["inputs"]["injection_hosts"] == 50
    fast = [r for r in rec["curve"] if r["rate"] >= 0.01]
    assert all(r["recall"] > 0.9 and r["limit"] == pytest.approx(3.0 / (50 * r["recall"]), rel=0.02) for r in fast)
    # Stars and injected stars are measured against the same reference table, each by its own magnitude.
    reference = pd.read_parquet(lists / "reference.parquet")
    stars, _, offset = script.read_stars(run / "summary", str(tmp_path / "aux.parquet"), oc.Config(), reference)
    inj_ref = script.read_injections(out / "injections", reference, offset)
    for d in (stars, inj_ref[np.isfinite(inj_ref["slope"])]):
        assert np.allclose(d["slope_z"], (d["slope"] - d["slope_ref"]) / np.hypot(d["slope_err"], d["slope_ref_err"]))
        assert np.allclose(d["slope_z_raw"], d["slope"] / d["slope_err"])
    assert stars["slope_ref"].nunique() == 1 and stars["slope_ref_err"].iloc[0] > 0
    assert not offset.any() and rec["population_offset"][0]["stars"] == 0     # 50 stars: too few for an own median
    # Planted fades in the same stars: one star in five given its own injected scores at 2% a year.
    assert script.main(["planted", *args, "--out", str(out / "planted"), "--rates", "0.02",
                        "--shares", "0", "0.2", "--trials", "3"]) == 0
    check = pd.read_csv(out / "planted" / "planted.csv")
    assert check["trials"].tolist() == [3, 3] and check["excess_p_below_0.05"].tolist() == [0, 3]
    assert check["limit_below_planted"].tolist() == [0, 0]
    # As on the whole sky: the catalogue in sky slices, and hosts only from the pass's random share of stars
    # (its lightcurves/ directory also holds the stars near a threshold, which are not a fair sample).
    (tmp_path / "sky").mkdir()
    aux.to_parquet(tmp_path / "sky" / "aux_L4_0000000.parquet", index=False)
    assert script.main(["inject", "--lightcurves", str(tmp_path / "lc"), "--aux", str(tmp_path / "sky"), "--rule",
                        str(rule), "--out", str(tmp_path / "occ2"), "--workers", "1", "--random-share", "0.5"]) == 0
    part = pd.read_parquet(tmp_path / "occ2" / "injections" / "a.parquet")
    dwarfs = np.arange(11, n + 1)
    assert set(part["source_id"]) == set(dwarfs[script.summary.random_share(dwarfs, 0.5)]) and 10 < part["source_id"].nunique() < 40
    assert part.equals(inj[inj["source_id"].isin(part["source_id"])].reset_index(drop=True))


def test_catalogue_in_sky_slices(tmp_path):
    """The whole sky's catalogue rows come one HEALPix pixel per query and per file. A query the service refuses
    is asked again in quarters, and a file of stars reads only the slices its source_ids fall in."""
    script = _script("slow_occurrence")
    cfg = oc.Config()
    where = script.sky_where(cfg)
    assert "phot_g_mean_mag < 19" in where and "parallax >= 5 * parallax_error" in where and "bp_rp < 1.8" in where
    width = 2 ** 35 * 4 ** (12 - 2)                       # source_ids per level-2 pixel
    rng = np.random.default_rng(51)
    ids = np.sort(rng.choice(np.arange(3 * width, 5 * width, 2 ** 31), 3_000, replace=False))
    cat = pd.DataFrame({"source_id": ids, "phot_g_mean_mag": rng.uniform(12, 19, len(ids)), "bp_rp": 0.9,
                        "parallax": 2.0, "parallax_error": 0.1})
    calls = []

    def query(lo, hi):
        calls.append((lo, hi))
        d = cat[(cat["source_id"] >= lo) & (cat["source_id"] <= hi)]
        if len(d) > 400:
            raise RuntimeError("too many rows for one query")
        return d.rename(columns=str.upper).sample(frac=1.0, random_state=0)     # any order, any case

    for pix in (3, 4):
        d = script.sky_rows(query, pix * width, (pix + 1) * width - 1)
        want = cat[(cat["source_id"] // width) == pix].reset_index(drop=True)
        assert d.equals(want) and d["source_id"].is_monotonic_increasing
        d.to_parquet(tmp_path / f"aux_L2_{pix:07d}.parquet", index=False)
    assert len(calls) == 10 and max(hi - lo for lo, hi in calls) == width - 1     # each pixel: refused, then 4 quarters
    rows = script.aux_rows(tmp_path, ids[1_000], ids[2_000])            # spans both slices
    assert rows.index.tolist() == ids[1_000:2_001].tolist() and list(rows.columns) == script.AUX_COLUMNS[1:]
    assert script.aux_rows(tmp_path, ids[10], ids[20]).index.tolist() == ids[10:21].tolist()
    assert not len(script.aux_rows(tmp_path, 7 * width, 8 * width - 1))
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        script.aux_rows(tmp_path / "empty", 0, 1)


def test_limit_counts_a_referenced_table_file_by_file(tmp_path):
    """A per-star table with its own neighbour reference, in two files: the limit step tallies them one at a
    time and gets what the in-memory route gets. Giants by parallax and rows marked as not a fair sample of
    their magnitude (`ref` false) are left out."""
    script = _script("slow_occurrence")
    cfg = oc.Config()
    stars = population(6_000, np.random.default_rng(31), faders=20, rate=0.02)
    err = oc.slope_error(stars)
    ref, ref_err = np.where(stars["phot_g_mean_mag"] < 13, 6e-4, 0.0), 0.3 * err
    k = np.arange(len(stars))
    table = stars.assign(slope_err=err, slope_ref=ref, slope_ref_err=ref_err, slope_z_raw=stars["slope_z"],
                         slope_z=(stars["slope"] - ref) / np.hypot(err, ref_err), ref=k % 20 != 7)
    # Catalogue rows: a main-sequence parallax for each star's G at BP-RP 0.9, a giant's for one star in twelve.
    aux = pd.DataFrame({"source_id": stars["source_id"], "phot_g_mean_mag": stars["phot_g_mean_mag"], "bp_rp": 0.9,
                        "parallax": 10 ** ((5.9 - stars["phot_g_mean_mag"] + 10) / 5) * np.where(k % 12 == 5, 0.01, 1.0)})
    aux["parallax_error"] = aux["parallax"] / 50
    aux.to_parquet(tmp_path / "aux.parquet", index=False)
    inj = injected(stars, cfg).assign(slope_err=np.tile(err, len(cfg.rates)))
    (tmp_path / "summary").mkdir()
    (tmp_path / "inj").mkdir()
    on_disk = table.drop(columns="phot_g_mean_mag")
    on_disk.iloc[:2_500].to_parquet(tmp_path / "summary" / "a.parquet", index=False)
    on_disk.iloc[2_500:].to_parquet(tmp_path / "summary" / "b.parquet", index=False)
    inj.to_parquet(tmp_path / "inj" / "a.parquet", index=False)
    assert script.main(["limit", "--stars", str(tmp_path / "summary"), "--aux", str(tmp_path / "aux.parquet"),
                        "--injections", str(tmp_path / "inj"), "--out", str(tmp_path / "limit")]) == 0
    rec = json.loads((tmp_path / "limit" / "limit.json").read_text())
    dwarfs = table[(k % 12 != 5) & (k % 20 != 7)].drop(columns="ref")
    assert 5_000 < len(dwarfs) < 5_300
    assert rec["inputs"]["neighbour_reference"] and rec["inputs"]["stars_scored"] == 6_000
    assert rec["inputs"]["fgk_dwarfs_searched"] == len(dwarfs) == rec["inputs"]["injection_hosts"]
    curve, _, sym = oc.limit_curve(dwarfs, inj, cfg)
    got = pd.DataFrame(rec["curve"])
    for col in ("limit", "mirror_limit", "fading", "brightening", "recall", "excess_p", "g_max", "z_min"):
        assert np.allclose(got[col], curve[col]), col
    assert pd.DataFrame(rec["symmetry"])["stars"].tolist() == sym["stars"].tolist()
    # The same from a catalogue kept in sky slices.
    (tmp_path / "sky").mkdir()
    aux.to_parquet(tmp_path / "sky" / "aux_L4_0000000.parquet", index=False)
    assert script.main(["limit", "--stars", str(tmp_path / "summary"), "--aux", str(tmp_path / "sky"),
                        "--injections", str(tmp_path / "inj"), "--out", str(tmp_path / "limit_sky")]) == 0
    assert json.loads((tmp_path / "limit_sky" / "limit.json").read_text())["curve"] == rec["curve"]
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


def test_searched_stars_are_centred_on_their_own_median():
    """Against a neighbour reference made from all stars, the searched stars can sit off zero (the leftover
    instrument trend depends on colour). Their own median per G range is taken out, from histograms that add up
    file by file, and injected stars get the same shift."""
    rng = np.random.default_rng(41)
    n = 40_000
    g = rng.uniform(12.0, 19.0, n)
    err = 3e-4 * 10 ** (0.2 * (g - 13))
    shift = np.where(g >= 18, -0.3, np.where(g >= 17, -0.1, 0.0)) * err          # in trend, per year
    stars = pd.DataFrame({"source_id": np.arange(n), "g_mag": g, "slope": rng.normal(0, err) + 2e-4 + shift,
                          "slope_err": err, "slope_ref": 2e-4, "slope_ref_err": 0.1 * err})
    stars["slope_z"] = (stars["slope"] - stars["slope_ref"]) / np.hypot(err, 0.1 * err)
    whole = oc.offset_hist(stars)
    assert np.array_equal(whole, oc.offset_hist(stars.iloc[:15_000]) + oc.offset_hist(stars.iloc[15_000:]))
    assert whole.sum() == n
    off = oc.offsets(whole)
    out = oc.centre(stars, off)
    for lo in np.arange(12.0, 19.0, 0.5):
        k = (g >= lo) & (g < lo + 0.5)
        assert off[oc.offset_range(lo + 0.1)] == pytest.approx(np.median((stars["slope"] - 2e-4)[k]), abs=3e-6)
        assert abs(np.median(out["slope_z"][k])) < 0.01 < 0.05 + abs(np.median(stars["slope_z"][k]))
    assert np.median(stars["slope_z"][g >= 18]) < -0.2
    assert np.allclose(out["slope_z"], (out["slope"] - out["slope_ref"]) / np.hypot(err, 0.1 * err))
    # Ranges with too few stars take their neighbours' median; a table with too few stars is left alone.
    few = stars[(g < 12.2) | (g >= 13)]
    assert oc.offsets(oc.offset_hist(few))[oc.offset_range(12.1)] == pytest.approx(0.0, abs=3e-5)
    assert not oc.offsets(oc.offset_hist(stars.iloc[:300])).any()
    # An injected star is moved by its own G range's offset, whatever its trend.
    inj = stars.iloc[:5].assign(slope=-0.02)
    assert np.allclose(oc.centre(inj, off)["slope_ref"], 2e-4 + off[oc.offset_range(inj["g_mag"])])


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
