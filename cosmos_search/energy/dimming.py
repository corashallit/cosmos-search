"""How much starlight is missing on average: the <delta> the energy budget needs, with a lower bound.

The energy budget scales with the time-averaged dimming, not the depth of any one dip. Gaia samples a star in visit
groups (two fields of view 106.5 min apart, successive scans 6 h apart, then weeks of nothing), so epochs within a
group are not independent looks: one dip lasting a week can fill a whole group. The estimate therefore averages per
group first:

    u_g   = mean over the group's epochs of max(0, 1 - flux / baseline), counting only epochs DIP_SIGMA below
    <delta> = mean of u_g over groups

and its lower bound replaces the fraction of groups that are dimmed (k of N) by its one-sided Clopper-Pearson lower
limit, keeping the mean u_g of the dimmed groups. With one dimmed group in forty that lower limit is ~20x below the
point estimate, which is the honest size of what one look can say about a duty cycle.
"""

import numpy as np
from scipy.stats import beta as beta_dist

GROUP_GAP_DAYS = 2.0
DIP_SIGMA = 3.0
CONF = 0.95


def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) > 1 else float("nan")


def groups(t, gap=GROUP_GAP_DAYS):
    t = np.asarray(t, float)
    order = np.argsort(t)
    g = np.zeros(len(t), np.int64)
    g[order[1:]] = np.cumsum(np.diff(t[order]) > gap)
    return g


def mean_deficit(t, flux, err=None, conf=CONF):
    """dict(delta, delta_lo, n_groups, k_groups, depth_mean) for one band's light curve (flux normalised or not)."""
    t, f = np.asarray(t, float), np.asarray(flux, float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if len(f) < 5:
        return {"delta": np.nan, "delta_lo": np.nan, "n_groups": 0, "k_groups": 0, "depth_mean": np.nan}
    base = np.median(f)
    s = robust_sigma(f)
    if err is not None:
        e = np.asarray(err, float)[ok]
        s = max(s, float(np.nanmedian(e)))
    deficit = np.clip(1 - f / base, 0, None) * (f < base - DIP_SIGMA * s)
    g = groups(t)
    u = np.bincount(g, weights=deficit) / np.bincount(g)
    n, k = len(u), int((u > 0).sum())
    delta = float(u.mean())
    if k == 0:
        return {"delta": 0.0, "delta_lo": 0.0, "n_groups": n, "k_groups": 0, "depth_mean": 0.0}
    lo = float(beta_dist.ppf(1 - conf, k, n - k + 1))
    depth = float(u[u > 0].mean())
    return {"delta": delta, "delta_lo": depth * lo, "n_groups": n, "k_groups": k, "depth_mean": depth}
