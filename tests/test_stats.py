import numpy as np
import pytest

from cosmos_search.inject.recovery import completeness, wilson
from cosmos_search.stats import (
    TailSketch, audit_inclusion, bh, cauchy_combine, empirical_p, gpd_fit, horvitz_thompson,
    poisson_sample, storey_pi0, stratified_p, tail_p,
)


def test_empirical_p_is_valid_and_never_zero():
    rng = np.random.default_rng(1)
    null = rng.normal(size=999)
    p = empirical_p(rng.normal(size=20000), null)
    assert p.min() >= 1 / 1000
    # Under the null, P(p <= a) <= a, up to the sampling error of one 999-draw null.
    for a in (0.01, 0.05, 0.2):
        assert np.mean(p <= a) <= a + 3 * np.sqrt(a * (1 - a) / 999)
    assert empirical_p(1e9, null) == pytest.approx(1 / 1000)


@pytest.mark.parametrize("xi_true", [0.0, 1 / 3])
def test_gpd_fit_recovers_shape(xi_true):
    rng = np.random.default_rng(2)
    u = rng.random(20000)
    y = -np.log(u) if xi_true == 0 else (u ** -xi_true - 1) / xi_true
    xi, sigma = gpd_fit(y)
    assert xi == pytest.approx(xi_true, abs=0.05)
    assert sigma == pytest.approx(1.0, rel=0.1)


@pytest.mark.parametrize("dist", ["exponential", "pareto3"])
def test_tail_p_extrapolates_beyond_the_null(dist):
    # Fit on 10^5 draws, evaluate 2-3 and 6 decades past the fitted tail, over many nulls.
    n, ratios = 100_000, []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        if dist == "exponential":
            null, x = rng.exponential(size=n), np.array([12.0, 20.0])
            exact = np.exp(-x)
        else:
            null, x = rng.random(n) ** (-1 / 3), np.array([100.0, 1000.0])
            exact = x**-3.0
        p, extrapolated = tail_p(x, null)
        assert extrapolated.all() and p[1] < 1 / n
        ratios.append(p / exact)
    ratios = np.array(ratios)
    # Near: within a factor of a few. Far: rarely too small, and at worst ~10^2 too large.
    assert np.median(ratios[:, 0]) == pytest.approx(1, abs=2)
    assert np.mean(ratios < 0.3) <= 0.15
    assert ratios.max() < 500


def test_tail_p_conservative_never_hits_zero_for_short_tails():
    rng = np.random.default_rng(4)
    null = rng.random(50_000)  # bounded: an unclamped fit puts the endpoint near 1
    p, _ = tail_p(np.array([1.5]), null)
    assert p[0] > 0
    p_free, _ = tail_p(np.array([1.5]), null, conservative=False)
    assert p_free[0] == 0


def test_stratified_p_uses_own_stratum_and_flags_thin_ones():
    rng = np.random.default_rng(5)
    null = np.concatenate([rng.normal(0, 1, 5000), rng.normal(0, 5, 5000), rng.normal(0, 1, 50)])
    null_strata = np.repeat(["quiet", "noisy", "thin"], [5000, 5000, 50])
    stat = np.array([4.0, 4.0, 4.0])
    p, _ = stratified_p(stat, np.array(["quiet", "noisy", "thin"]), null, null_strata)
    assert p[0] < 1e-3 < 0.1 < p[1]
    assert np.isnan(p[2])


def test_mirror_null_calibrates_a_one_sided_statistic():
    # Symmetric noise with heavy tails: dimming vs brightening statistics share a null.
    rng = np.random.default_rng(6)
    n = 500_000
    noise = rng.standard_t(3, size=(n, 2))
    dim, bright = noise[:, 0], noise[:, 1]
    p, _ = tail_p(dim, bright)
    for a in (1e-4, 1e-3, 1e-2):
        # Counting noise from both the statistics and the null: relative sd ~ sqrt(2 / (a n)).
        assert np.mean(p <= a) == pytest.approx(a, rel=3 * np.sqrt(2 / (a * n)))


def test_cauchy_combine_is_valid_and_handles_tiny_p():
    rng = np.random.default_rng(7)
    # Strongly correlated inputs: the same uniform repeated with jitter.
    base = rng.random(50_000)
    p = np.clip(np.column_stack([base, base, rng.random(50_000)]), 1e-300, 1)
    combined = cauchy_combine(p)
    assert np.mean(combined <= 0.01) == pytest.approx(0.01, rel=0.3)
    tiny = cauchy_combine(np.array([1e-30, 0.5]))
    assert tiny == pytest.approx(2e-30, rel=1e-3)


def test_bh_and_storey():
    rng = np.random.default_rng(8)
    p = np.concatenate([rng.random(9900), rng.random(100) * 1e-6])
    rejected, q = bh(p, 0.05)
    assert rejected[-100:].all()
    assert rejected[:9900].sum() <= 15
    assert (q >= p * (1 - 1e-12)).all() and (q <= 1).all()
    assert storey_pi0(p) == pytest.approx(0.99, abs=0.03)


def test_audit_inclusion_and_horvitz_thompson_are_unbiased():
    rng = np.random.default_rng(9)
    n = 20_000
    scores = rng.normal(size=n)
    pi = audit_inclusion(scores, 500, floor=0.2)
    assert pi.sum() == pytest.approx(500)
    assert pi.max() <= 1 and pi.min() >= 0.2 * 500 / n * 0.99
    assert np.corrcoef(scores, pi)[0, 1] > 0.9
    # Missed real objects concentrate at high scores; the estimate must still be unbiased.
    real = rng.random(n) < 0.002 + 0.02 * (scores > 2)
    draws = []
    for _ in range(2000):
        take = poisson_sample(rng, pi)
        draws.append(horvitz_thompson(real[take], pi[take]))
    est, se = np.array(draws).T
    assert abs(est.mean() - real.sum()) < 4 * est.std() / np.sqrt(len(est))
    # The reported standard error matches the spread of the estimates.
    assert np.sqrt(np.mean(se**2)) == pytest.approx(est.std(), rel=0.15)


def test_completeness_map():
    rng = np.random.default_rng(10)
    depth = 10 ** rng.uniform(-3, 0, 5000)
    rec = rng.random(5000) < np.clip(depth / 0.01, 0, 1)
    c = completeness({"depth": depth}, rec, {"depth": np.logspace(-3, 0, 7)})
    f = c.fraction
    assert f[0] < 0.4 and f[-1] > 0.98
    assert ((c.lo <= f) & (f <= c.hi)).all()
    assert c.n.sum() == 5000
    assert c.rows()[0]["depth_lo"] == pytest.approx(1e-3)
    lo, hi = wilson(0, 0)
    assert (lo, hi) == (0, 1)


def _stream(rng, n_files, per_file, max_exact, edges):
    sk, everything = TailSketch(edges, max_exact), []
    for _ in range(n_files):
        v = rng.standard_t(3, per_file)
        everything.append(v)
        sk.add(v)
    return sk, np.concatenate(everything)


def test_tail_sketch_is_exact_in_the_tail_and_bounded():
    rng = np.random.default_rng(11)
    edges = np.linspace(-20, 20, 4001)
    sk, null = _stream(rng, 50, 20_000, 5_000, edges)
    assert sk.n == null.size == 1_000_000
    assert sk.exact.size <= 5_000
    s = np.array([sk.u, sk.u + 1, 8.0, 15.0])
    s = s[s >= sk.u]
    p, _ = sk.p(s, min_exceed=1)
    assert np.allclose(p, empirical_p(s, null))
    # Below u: counts whole bins, so never more significant than the exact empirical p.
    low = np.array([-1.0, 0.0, 0.5, 1.0])
    assert (sk.p(low)[0] >= empirical_p(low, null)).all()
    assert np.allclose(sk.p(low)[0], empirical_p(low, null), rtol=1e-2)


def test_tail_sketch_extrapolates_like_tail_p():
    rng = np.random.default_rng(12)
    edges = np.linspace(-20, 20, 4001)
    sk, null = _stream(rng, 10, 20_000, 10_000, edges)
    s = np.array([60.0, 200.0])
    p_sketch, far = sk.p(s)
    p_full, _ = tail_p(s, null)
    assert far.all()
    assert np.allclose(p_sketch, p_full)


def test_tail_sketch_merge_matches_one_pass_and_roundtrips():
    import json

    edges = np.linspace(-20, 20, 401)
    a, null_a = _stream(np.random.default_rng(13), 5, 10_000, 2_000, edges)
    b, null_b = _stream(np.random.default_rng(14), 5, 10_000, 3_000, edges)
    b = TailSketch.from_dict(json.loads(json.dumps(b.to_dict())))
    merged = a.merge(b)
    null = np.concatenate([null_a, null_b])
    assert merged.n == null.size
    s = np.array([merged.u, 6.0, 9.0])
    assert np.allclose(merged.p(s, min_exceed=1)[0], empirical_p(s, null))
    with pytest.raises(ValueError):
        merged.merge(TailSketch(np.linspace(0, 1, 3)))


def _extremes(rng, n_src, n_pts, n_real=0, depth=25.0):
    """Min and max of n_pts heavy-tailed noise points per source; the first n_real get one deep dip."""
    x = rng.standard_t(4, size=(n_src, n_pts))
    x[:n_real, 0] -= depth
    return -x.min(axis=1), x.max(axis=1)


def test_knockoff_q_controls_fdr_and_finds_dimmers():
    from cosmos_search.stats import knockoff_q

    rng = np.random.default_rng(15)
    fdp, power = [], []
    for _ in range(30):
        # Epoch counts vary between sources, which breaks cross-source exchangeability but not competition.
        dims, brights = [], []
        for n_pts, n_real in ((15, 40), (150, 40)):
            d, b = _extremes(rng, 5000, n_pts, n_real)
            dims.append(d), brights.append(b)
        dim, bright = np.concatenate(dims), np.concatenate(brights)
        real = np.zeros(dim.size, bool)
        real[:40] = real[5000:5040] = True
        found = knockoff_q(dim, bright) <= 0.1
        fdp.append((found & ~real).sum() / max(found.sum(), 1))
        power.append((found & real).sum() / real.sum())
    assert np.mean(fdp) <= 0.1 + 0.03
    assert np.mean(power) > 0.5


def test_knockoff_q_on_pure_null_finds_almost_nothing():
    from cosmos_search.stats import knockoff_q

    rng = np.random.default_rng(16)
    dim, bright = _extremes(rng, 20000, 60)
    q = knockoff_q(dim, bright)
    assert (q <= 0.1).sum() <= 5
    assert ((q >= 0) & (q <= 1)).all()


def test_tail_ratio_is_one_for_symmetric_noise_and_flags_dimmers():
    from cosmos_search.stats import tail_ratio

    rng = np.random.default_rng(17)
    dim, bright = _extremes(rng, 50000, 60, n_real=500)
    strata = np.where(np.arange(dim.size) < 25000, "a", "b")
    rows = tail_ratio(dim, bright, strata, [3.0, 20.0])
    by = {(r["stratum"], r["threshold"]): r for r in rows}
    assert by[("b", 3.0)]["ratio_lo"] < 1 < by[("b", 3.0)]["ratio_hi"] * 1.05
    assert by[("a", 20.0)]["ratio_lo"] > 3
