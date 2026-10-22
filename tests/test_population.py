import numpy as np

from cosmos_search.population import events, split


def _star(sid, t, rng, dips=(), noise=0.01):
    f = 1000 * (1 + noise * rng.standard_normal(len(t)))
    for t0, t1, d in dips:
        f[(t >= t0) & (t <= t1)] *= 1 - d
    return {"source_id": np.full(len(t), sid, np.int64), "t": t, "flux": f,
            "err_rel": np.full(len(t), noise, np.float32), "flags": np.zeros(len(t), np.int8)}


def _cadence(rng, n_visits=20):
    starts = np.sort(rng.uniform(0, 1000, n_visits))
    starts = starts[np.r_[True, np.diff(starts) > 2]]
    return np.sort(np.concatenate([[s, s + 0.074] for s in starts])), starts


def _cat(stars):
    return {k: np.concatenate([s[k] for s in stars]) for k in stars[0]}


def test_split_is_by_day_and_balanced():
    days = np.arange(-200, 2000)
    h = split.half_of_day(days)
    assert 0.45 < h.mean() < 0.55
    assert np.all(split.half_of_time(days + 0.3) == h)


def test_single_one_visit_dip_and_mirror():
    rng = np.random.default_rng(1)
    t, starts = _cadence(rng)
    s = starts[7]
    quiet = _star(1, t, rng)
    dip = _star(2, t, rng, dips=[(s - 0.1, s + 0.2, 0.2)])
    two = _star(3, t, rng, dips=[(s - 0.1, s + 0.2, 0.2), (starts[12] - 0.1, starts[12] + 0.2, 0.2)])
    one_epoch = _star(4, t, rng, dips=[(s - 0.01, s + 0.01, 0.2)])
    bright = _star(5, t, rng, dips=[(s - 0.1, s + 0.2, -0.2)])
    rows, stars = events.prepare(_cat([quiet, dip, two, one_epoch, bright]))
    ev = events.find(rows, stars)
    got = {(int(r.source_id), r.cls) for r in ev.itertuples()}
    assert got == {(2, "SOV"), (4, "SOV1"), (5, "MIR")}
    e = ev[ev.source_id == 2].iloc[0]
    assert abs(e.depth - 0.2) < 0.05 and e.n_in == 2 and e.half == split.half_of_time(s)


def test_unbracketed_first_visit_is_not_an_event():
    rng = np.random.default_rng(2)
    t, starts = _cadence(rng)
    rows, stars = events.prepare(_cat([_star(1, t, rng, dips=[(starts[0] - 0.1, starts[0] + 0.2, 0.3)])]))
    assert events.find(rows, stars).empty


def test_generic_exposure_matches_analytic(monkeypatch):
    from cosmos_search.population import exposure
    rng = np.random.default_rng(3)
    starts = np.arange(30) * 30.0 + 0.3
    t = np.sort(np.concatenate([starts, starts + 0.074]))
    rows, stars = events.prepare(_cat([_star(1, t, rng, noise=0.001)]))
    monkeypatch.setattr(exposure, "GEN_T", (0.5, 0.5 + 1e-9))
    monkeypatch.setattr(exposure, "GEN_DEPTH", (0.3, 0.3 + 1e-9))
    e = exposure.generic(rows, stars, n_draws=4000, rng=np.random.default_rng(4))
    # Both epochs of an interior visit in the dip's flat part: start within (T - w) of the pair, ingress <= T/2.
    halves = split.half_of_time(starts[1:-1])
    for h in (0, 1):
        n = np.sum(halves == h)
        lo, hi = exposure.GEN_RATE * n * (0.5 - 0.074) * 0.4, exposure.GEN_RATE * n * (0.5 + 0.074)
        assert lo < e[0, h] < hi


def test_glm_recovers_a_covariate_effect():
    import pandas as pd
    from cosmos_search.population import glm
    rng = np.random.default_rng(5)
    n = 200_000
    stars = pd.DataFrame({c: rng.normal(size=n) for c in ("phot_g_mean_mag", "bp_rp", "m_g", "log_excess",
                                                         "log_eb_share", "ipd_frac_multi_peak", "ruwe",
                                                         "blend_frac", "n_visits", "abs_ecl_lat")})
    stars["phot_g_mean_mag"] = rng.uniform(10, 18, n)
    stars["abs_ecl_lat"] = rng.uniform(0, 90, n)
    codes, _ = glm.covariate_codes(stars)
    e = rng.uniform(0.005, 0.02, n)
    lam = e * np.exp(np.log(3.0) * (stars.phot_g_mean_mag < 13))
    y = rng.poisson(lam)
    m = glm.fit(codes, y, e, ["g"])
    pred = m["predict"](codes, e)
    bright = stars.phot_g_mean_mag.to_numpy() < 13
    assert abs(pred[bright].sum() / pred[~bright].sum() - lam[bright].sum() / lam[~bright].sum()) < 0.15 * 3
    stat, df, p = glm.lrt(codes, y, e, ["g"], "ecl")
    assert p > 1e-3


def test_scan_finds_an_injected_cell_excess():
    from cosmos_search.population import scan
    rng = np.random.default_rng(6)
    win = scan.Windows()
    lam = np.full((12 * 64 * 64, scan.N_SHELLS + 1), 0.02)
    obs = rng.poisson(lam).astype(float)
    import healpy as hp
    pix = hp.query_disc(64, hp.ang2vec(120.0, -30.0, lonlat=True), np.radians(3.0), nest=True)
    obs[pix, 2] += rng.poisson(1.0, len(pix))
    null = scan.null_max(win, lam, n=39, rng=rng)
    got, top = scan.candidates(win, obs, lam, null)
    assert got and got[0]["distance"] == "500-1000 pc"
    assert hp.rotator.angdist([got[0]["ra"], got[0]["dec"]], [120.0, -30.0], lonlat=True)[0] < np.radians(3)


def test_population_columns_geometry():
    import pandas as pd
    from cosmos_search.population import analysis, glm
    # Galactic centre at 8.2 kpc, the Sun's neighbourhood towards the pole, a fast star at 1 kpc
    df = pd.DataFrame({"source_id": np.array([0, 1 << 47, 2 << 47], np.int64), "l": [0.0, 0.0, 90.0],
                       "b": [0.0, 90.0, 0.0], "r_med_photogeo": [8200.0, 2000.0, 1000.0],
                       "pmra": [0.0, 0.0, 30.0], "pmdec": [0.0, 0.0, 40.0]})
    dens = pd.Series([1000.0, 10.0, 0.0], index=[0, 1, 2])
    analysis.population_columns(df, dens)
    assert df.r_gal[0] < 1e-6 and abs(df.abs_z[0] - 0.02) < 1e-9
    assert abs(df.r_gal[1] - 8.2) < 1e-6 and abs(df.abs_z[1] - 2.02) < 1e-9
    assert abs(df.v_tan[2] - 4.74047 * 50) < 1e-6
    assert abs(df.log_dens[0] - np.log10(1000 / analysis.PIXEL_AREA_DEG2)) < 1e-9
    codes, _ = glm.covariate_codes(df.assign(**{c: 1.0 for c, _ in [glm.BINS[k] for k in glm.MAIN if k != "eb"]},
                                             log_eb_share=0.0, abs_ecl_lat=10.0))
    assert set(glm.DR4) <= set(codes.columns)
    assert codes.gal[0] != codes.gal[1]


def test_monitor_dispersion_only_raises_phi():
    import pandas as pd
    from cosmos_search.population import analysis, scan
    rng = np.random.default_rng(8)
    n = 60000
    px = rng.integers(0, 12 * 64 * 64, n)
    stars = pd.DataFrame({"pixel": px, "shell": np.full(n, 3), "phot_g_mean_mag": rng.uniform(12, 17, n),
                          "bp_rp": np.full(n, 1.0), "m_g": np.full(n, 4.0)})
    e = np.full(n, 1.0)
    # the monitor's rate varies by sky region (an artefact field); the main class is Poisson
    patch = (px // 4096) % 2 == 0
    y_mon = rng.poisson(np.where(patch, 0.4, 0.1) * rng.gamma(2.0, 0.5, n)).astype(float)
    win = scan.Windows()
    phi_mon = analysis.monitor_dispersion(stars, y_mon, e, ["g"], win)
    assert phi_mon.max() > 1.5 and phi_mon.min() >= 1.0


def test_dispersion_curve_tracks_relative_scatter():
    from cosmos_search.population import scan
    rng = np.random.default_rng(9)
    win = scan.Windows()
    # cell rates vary by a smooth multiplicative field: relative scatter, so window phi grows with E
    npix = 12 * 64 * 64
    base = np.exp(rng.uniform(np.log(0.002), np.log(0.2), npix))
    field = np.exp(0.3 * rng.standard_normal(npix // 64)).repeat(64)
    lam = np.zeros((npix, scan.N_SHELLS + 1)); lam[:, -1] = base
    obs = rng.poisson(lam * field[:, None]).astype(float)
    curves = scan.dispersion_curve(win, obs, lam)
    em, ph = curves[(3, len(win.shells) - 1)]
    assert np.all(np.diff(ph) >= 0) and ph[-1] > ph[0]
    phi = scan.phi_from_curves(win, lam, curves)
    assert phi.min() >= 1.0


def test_scan_coincidence_groups_one_scan():
    from cosmos_search.population import analysis
    t = [10.0, 10.02, 10.05, 50.0, 90.0, 90.01]
    ra = [100.0, 100.5, 101.0, 100.0, 100.0, 110.0]       # last pair: same time, 10 degrees apart
    dec = [-20.0] * 6
    assert analysis.scan_coincidence(t, ra, dec) == (6, 3, 4)
