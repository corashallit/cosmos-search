import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from cosmos_search.viz import astrometry, colour, data, html, lightcurve, population, sheet, style  # noqa: E402

RNG = np.random.default_rng(7)
THR = {"G<19": {"s": 5.6, "z": 3.8, "rep": 7.0}, "G>=19": {"s": 6.4, "z": 3.9, "rep": 7.7}}


def _epochs(n=60, span=1000.0, depth=0.0, t0=None, t1=None, noise=0.002, scale=1.0):
    """Synthetic Gaia epochs in G, BP and RP: pairs 106 minutes apart, a box dip of `depth` between t0 and t1."""
    t = np.sort(RNG.uniform(0, span, n))
    t = np.concatenate([t, t + 106 / 1440])
    rows = []
    for band, k in ((0, 1.0), (1, 1.3), (2, 0.8)):
        f = RNG.normal(0, noise, len(t))
        if depth:
            f[(t >= t0) & (t <= t1)] -= depth * k * scale
        rows.append(pd.DataFrame({"source_id": 1, "band": band, "t": t, "frac": f, "err_frac": noise, "fov": 1}))
    return pd.concat(rows, ignore_index=True)


def _pop(n=4000):
    g = RNG.uniform(12, 20.5, n)
    pop = pd.DataFrame({
        "source_id": np.arange(n), "phot_g_mean_mag": g, "bp_rp": RNG.uniform(0.3, 2.5, n),
        "n_visits": 30, "sigma_visit": 0.001 * 10 ** (0.2 * (g - 12)), "abbe": RNG.uniform(0.05, 1.6, n),
        "s_fade": RNG.exponential(1.5, n), "s_rise_ns": RNG.exponential(1.4, n), "slope_z": RNG.normal(0, 1.2, n),
        "s_rep": RNG.exponential(1.5, n), "s_rep_bright": RNG.exponential(1.5, n),
        "dim_depth": 10 ** RNG.uniform(-3, -0.5, n), "min_frac": -(10 ** RNG.uniform(-3, -0.3, n)),
        "depth_bp": 10 ** RNG.uniform(-2, -1, n), "depth_rp": 10 ** RNG.uniform(-2.1, -1.1, n),
        "depth_bp_err": 1e-3, "depth_rp_err": 1e-3,
        "best_class_name": RNG.choice(["ECL", "AGN", "LPV", "RS", "YSO", None], n),
    })
    return pd.concat([pop, data.strengths(pop, THR)], axis=1)


def _svg_text(paths, theme="light"):
    return paths[theme]["svg"].read_text()


def test_tokens_cover_both_themes_and_ramp_runs_from_page_to_blue():
    assert set(style.TOKENS["light"]) == set(style.TOKENS["dark"])
    for name in style.THEMES:
        with style.theme(name):
            r = style.ramp(5)
            assert r[0] != r[-1]
            assert style.mix("blue", "blue", 0.3).lower() == style.c("blue").lower()


def test_plain_numbers_for_axes_and_labels():
    assert style._plain_number(1000) == "1k" and style._plain_number(2e6) == "2M"
    assert style._plain_number(0.01) == "0.01" and style._plain_number(1e-6) == "10⁻⁶"
    assert style.number(17400.4) == "17,400" and style.number(0.8241) == "0.824"


def test_year_and_btjd_round_trip():
    assert style.year(0.0) == pytest.approx(2014.935, abs=1e-3)   # BTJD 0 = JD 2457000 = 2014-12-08 12h
    assert style.btjd(style.year(1234.5)) == pytest.approx(1234.5)


def test_strengths_take_the_largest_score_over_its_threshold():
    pop = pd.DataFrame({"phot_g_mean_mag": [15.0, 20.0], "s_fade": [11.2, 0.0], "slope_z": [-1.0, 7.8],
                        "s_rep": [0.0, 0.0], "s_rise_ns": [0.0, 3.2], "s_rep_bright": [14.0, 0.0]})
    st = data.strengths(pop, THR)
    assert st["fade_strength"].tolist() == pytest.approx([2.0, 0.0])
    assert st["rise_strength"].tolist() == pytest.approx([2.0, 2.0])
    assert st["group"].tolist() == ["G<19", "G>=19"]
    assert data.survival([1, 2, 3, 4], [0, 2.5, 5]).tolist() == [4, 2, 0]


def test_typical_noise_prefers_quiet_unclassified_stars():
    pop = _pop()
    pop.loc[pop["best_class_name"].notna(), "sigma_visit"] *= 5
    quiet = data.typical_noise(pop, 15.0)
    allstars = data.typical_noise(pop, 15.0, quiet=False)
    assert quiet < allstars


def test_slow_change_renders_both_themes_with_hover_titles(tmp_path):
    ep = _epochs(depth=0.03, t0=400, t1=650)
    change = {"kind": "dip", "t0": 400.0, "t1": 650.0, "depth": 0.03, "slope": np.nan}
    paths = style.render(lightcurve.slow_change, tmp_path / "lc", ep, change, noise=0.002, name="test star")
    for theme in style.THEMES:
        assert paths[theme]["png"].stat().st_size > 10_000
        svg = _svg_text(paths, theme)
        assert "dimming episode" in svg and "fainter" in svg
        assert svg.count("<title>") > 20        # one hover title per plotted visit
    assert style.TOKENS["light"]["box"] in _svg_text(paths, "light")
    assert style.TOKENS["dark"]["box"] in _svg_text(paths, "dark")


def test_large_changes_switch_to_magnitudes_and_label_in_magnitudes(tmp_path):
    ep = _epochs(depth=0.95, t0=300, t1=1000, scale=1.0)
    assert lightcurve.choose_scale(data.visits(ep, "G")["y"]) == "mag"
    change = {"kind": "drop", "t0": 300.0, "t1": 1000.0, "depth": 0.17, "slope": np.nan}
    paths = style.render(lightcurve.slow_change, tmp_path / "big", ep, change, themes=("light",))
    svg = _svg_text(paths)
    assert "change (mag)" in svg
    m = re.search(r"by (\d+\.\d) mag", svg)
    assert m and float(m.group(1)) > 2.5      # read off the visits (95% = 3.3 mag), not the 17% passed in


def test_slow_candidate_sheet_and_page(tmp_path):
    pop = _pop()
    fade = pop.loc[pop["fade_strength"] >= 1, "fade_strength"].to_numpy()
    rise = pop.loc[pop["rise_strength"] >= 1, "rise_strength"].to_numpy()
    hr = pd.DataFrame({"phot_g_mean_mag": RNG.uniform(8, 18, 3000), "bp_rp": RNG.uniform(0, 3, 3000),
                       "parallax": RNG.uniform(1, 20, 3000)})
    ctx = sheet.Context(pop=pop, lists={"G<19": (fade, rise, len(pop))}, hr=hr)
    row = pd.Series({"id": 4321, "phot_g_mean_mag": 15.2, "bp_rp": 1.0, "abbe": 0.2, "dim_depth": 0.03,
                     "fade_kind": "dip", "dim_t0": 400.0, "dim_t1": 650.0, "by": "episode", "slope": np.nan,
                     "strength": 3.1, "depth_bp": 0.031, "depth_bp_err": 0.002, "depth_rp": 0.029,
                     "depth_rp_err": 0.002, "t_first": 0.0, "t_last": 1000.0})
    checks = {"W1_change": -0.002, "W1_err": 0.004, "W2_change": 0.001, "W2_err": 0.006,
              "W1_pred_grey": -0.03, "W2_pred_grey": -0.03, "W1_pred_dust_in_front": -0.002,
              "W2_pred_dust_in_front": -0.001}
    figs = sheet.slow_candidate(ctx, row, _epochs(depth=0.03, t0=400, t1=650), tmp_path, checks=checks,
                                astro={"parallax": 2.0, "parallax_error": 0.05}, sid=4321)
    assert [f["key"] for f in figs] == ["lightcurve", "rare", "classes", "hr", "colour", "infrared"]
    assert all(f["caption"] and f["paths"]["dark"]["svg"].exists() for f in figs)
    out = html.page("Test page", [{"heading": "A star", "text": "<p>hello</p>", "figures": figs}],
                    tmp_path / "index.html", png_links=True)
    doc = out.read_text()
    assert doc.count('<figure class="viz"') == len(figs)
    assert 'id="theme-toggle"' in doc and "prefers-color-scheme: dark" in doc
    assert "<?xml" not in doc
    ids = re.findall(r'\bid="([^"]+)"', doc)
    assert len(ids) == len(set(ids))            # every inlined SVG's ids are prefixed apart
    assert "4321-rare-light.png" in doc


def test_infrared_needs_neowise():
    with style.theme("light"):
        assert colour.infrared({"dim_depth": 0.02}) is None


def test_hr_diagram_with_an_uncertain_parallax(tmp_path):
    hr = pd.DataFrame({"phot_g_mean_mag": RNG.uniform(8, 18, 2000), "bp_rp": RNG.uniform(0, 3, 2000),
                       "parallax": RNG.uniform(1, 20, 2000)})
    paths = style.render(population.hr_diagram, tmp_path / "hr", hr,
                         {"phot_g_mean_mag": 19.6, "bp_rp": 1.2, "parallax": -0.3, "parallax_error": 0.24},
                         themes=("light",))
    assert "too uncertain to place it" in _svg_text(paths)


def test_short_dip_and_its_depth(tmp_path):
    ep = _epochs(depth=0.4, t0=500.0, t1=500.1)
    pop = _pop()
    ctx = sheet.Context(pop=pop)
    row = {"source_id": 5885363968852633856, "phot_g_mean_mag": 15.0, "min_frac": -0.4}
    t_dip = float(ep.loc[ep["frac"].idxmin(), "t"])
    figs = sheet.short_dip_candidate(ctx, row, ep, t_dip, tmp_path, sid=5885363968852633856)
    assert [f["key"] for f in figs] == ["dip", "dipdepth"]
    assert figs[0]["paths"]["light"]["png"].name == "5885363968852633856-dip-light.png"


def test_kepler_bound_marks_known_and_candidate(tmp_path):
    n = 3000
    orb = pd.DataFrame({"source_id": np.arange(n), "period": 10 ** RNG.uniform(0, 3.5, n),
                        "f_A": 10 ** RNG.uniform(-6, 0, n), "m1": 1.0})
    orb.loc[0, ["source_id", "period", "f_A"]] = [list(astrometry.KNOWN)[0], 185.6, 11.1]
    paths = style.render(astrometry.kepler_bound, tmp_path / "kb", orb, {"period": 4250.0, "f_A": 33.0,
                                                                          "label": "Gaia BH3"}, themes=("dark",))
    svg = _svg_text(paths, "dark")
    assert "Gaia BH1" in svg and "Gaia BH3" in svg


def test_artifact_fragment_has_no_skeleton_and_does_not_persist_the_theme(tmp_path):
    out = html.page("Test page", [{"heading": "A star", "text": "<p>hello</p>", "figures": []}],
                    tmp_path / "artifact.html", fragment=True)
    doc = out.read_text()
    assert doc.startswith("<title>Test page</title>")
    assert "<!doctype" not in doc and "<body" not in doc and "</html>" not in doc
    assert 'name="claude:viewport-fit-sides"' in doc and "if (false)" in doc
    assert "if (true)" in html.page("T", [], tmp_path / "index.html").read_text()
