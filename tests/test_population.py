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
