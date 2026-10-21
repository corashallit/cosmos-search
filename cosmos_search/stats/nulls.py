"""Upper-tail p-values against empirical nulls, with a generalised Pareto tail beyond them.

Convention: larger statistic = more anomalous, p = P(null >= s).

An empirical null of n draws cannot resolve p below 1/(n + 1). The search needs far smaller
p-values (statistics.md §1), so beyond the last few exceedances we fit a generalised Pareto
distribution (GPD) to the null's upper tail. Pickands–Balkema–de Haan says excesses over a high
threshold are GPD for essentially any continuous null, so this assumes far less than a Gaussian
or chi-square tail does. The fit uses probability-weighted moments (Hosking & Wallis 1987),
which are closed-form and stable for the shape range we expect (xi < 0.5).

The mirror null (statistics.md §2) has as many draws as there are sources, so within wide
strata the empirical part reaches p ~ 1/N directly and the GPD only covers the last decade or so.
"""

import numpy as np


def empirical_p(stat, null):
    """(1 + #{null >= s}) / (1 + n): exact-valid for exchangeable draws, never zero."""
    null = np.sort(np.asarray(null, dtype=float))
    null = null[np.isfinite(null)]
    s = np.asarray(stat, dtype=float)
    exceed = null.size - np.searchsorted(null, s, side="left")
    return (1.0 + exceed) / (1.0 + null.size)


def gpd_fit(excess):
    """Probability-weighted-moment fit of a GPD to excesses over a threshold. Returns (xi, sigma)."""
    y = np.sort(np.asarray(excess, dtype=float))
    m = y.size
    if m < 10:
        raise ValueError(f"need at least 10 excesses for a GPD fit, got {m}")
    a0 = y.mean()
    a1 = np.sum(y * (m - 1 - np.arange(m)) / (m - 1)) / m
    k = a0 / (a0 - 2 * a1) - 2  # Hosking's k = -xi
    sigma = 2 * a0 * a1 / (a0 - 2 * a1)
    return -k, sigma


def _xi_upper(excess, q, n_boot=200):
    rng = np.random.default_rng(0)
    m = excess.size
    boots = [gpd_fit(excess[rng.integers(0, m, m)])[0] for _ in range(n_boot)]
    return float(np.quantile(boots, q))


def gpd_sf(y, xi, sigma):
    """Survival function of the GPD at excess y >= 0."""
    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    if abs(xi) < 1e-9:
        return np.exp(-y / sigma)
    z = 1 + xi * y / sigma
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(z > 0, np.abs(z) ** (-1 / xi), 0.0)
    return out


def tail_p(stat, null, min_exceed=10, tail_fraction=0.01, min_tail=100, conservative=True, xi_quantile=0.9):
    """Empirical p where the null has at least min_exceed exceedances, GPD tail beyond.

    The GPD is fitted to the top tail_fraction of the null (at least min_tail points).
    conservative=True clamps the shape at xi >= 0, so a short-tailed fit can never claim a
    finite upper endpoint the null has not shown, and replaces the point estimate of xi with its
    xi_quantile bootstrap quantile. Shape noise (sd ~ 1/sqrt(m)) compounds with distance: several
    decades past the fitted tail a point estimate is often an order of magnitude too small. With
    xi_quantile = 0.9 about 1 null in 10 still gives p too small there, and the typical cost is
    p a few to ~10^2 times too large. The bootstrap seed is fixed so p-values are reproducible.
    Returns (p, extrapolated) where extrapolated marks p-values that came from the GPD.
    """
    null = np.sort(np.asarray(null, dtype=float))
    null = null[np.isfinite(null)]
    s = np.atleast_1d(np.asarray(stat, dtype=float))
    n = null.size
    exceed = n - np.searchsorted(null, s, side="left")
    p = (1.0 + exceed) / (1.0 + n)
    far = exceed < min_exceed
    if far.any():
        m = max(min_tail, int(np.ceil(tail_fraction * n)))
        if m >= n:
            raise ValueError(f"null of {n} draws is too small for a {m}-point tail fit")
        # Below min_exceed exceedances the empirical count is mostly noise; the fitted tail is smoother.
        p[far] = gpd_tail_p(s[far], null[n - m - 1:], n, conservative, xi_quantile)
    return p.reshape(np.shape(stat)), far.reshape(np.shape(stat))


def gpd_tail_p(stat, top, n, conservative=True, xi_quantile=0.9):
    """Tail p-values from the m + 1 largest of n null draws, sorted ascending.

    The smallest of them is the threshold u and the other m are the excesses' source, so the
    fitted tail is P(X > u) = m / n times the GPD survival function. See tail_p for the
    conservative option.
    """
    top = np.asarray(top, dtype=float)
    u, m = top[0], top.size - 1
    excess = top[1:] - u
    xi, sigma = gpd_fit(excess)
    if conservative:
        xi = max(_xi_upper(excess, xi_quantile), 0.0)
        # Refit the scale at that shape so the tail still matches the excesses' mean.
        sigma = max(np.mean(excess) * (1 - xi), 1e-300) if xi < 1 else sigma
    return (m / n) * gpd_sf(np.asarray(stat, dtype=float) - u, xi, sigma)


def stratified_p(stat, strata, null, null_strata, min_null=1000, **tail_kw):
    """tail_p within each stratum. Sources whose stratum has fewer than min_null null draws get NaN.

    Strata are any hashable labels (e.g. tuples of magnitude, colour and crowding bins). Making
    strata coarse enough to hold min_null draws is the caller's choice; NaN makes the failure
    visible instead of silently borrowing a null from somewhere else.
    """
    stat = np.asarray(stat, dtype=float)
    strata = np.asarray(strata)
    null = np.asarray(null, dtype=float)
    null_strata = np.asarray(null_strata)
    p = np.full(stat.shape, np.nan)
    extrapolated = np.zeros(stat.shape, dtype=bool)
    for label in np.unique(strata):
        here = strata == label
        ref = null[null_strata == label]
        if ref.size < min_null:
            continue
        p[here], extrapolated[here] = tail_p(stat[here], ref, **tail_kw)
    return p, extrapolated
