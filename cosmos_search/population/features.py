"""Dip-blind light-curve features for the null (docs/population.md §14).

On DR3 the window overdispersion is mostly static and lives in deep dips on faint stars that DPAC classifies as
eclipsing binaries. DPAC's class reads the whole light curve, dip included, so it cannot enter the null. These
features read every epoch except the visit holding the star's deepest epoch, the same rule for every star with or
without an event. A single one-visit dip sits in that visit, so nothing here sees it; a binary's other eclipses,
an active star's spots or a pulsator's cycle are all still visible.

- f_low: share of outside epochs at z >= 2.5 (faint points, as eclipses below threshold make).
- f_high: share at z <= -2.5 (bright points).
- bowley: quartile skewness of the outside fluxes; dips skew it negative in flux, positive in z.
"""

import numpy as np
import pandas as pd

Z_PT = 2.5


def _group_quantile(gid, x, ng, q):
    """q-quantile of x per group (nearest rank), NaN for empty groups."""
    order = np.lexsort((x, gid))
    gs, xs = gid[order], x[order]
    start = np.searchsorted(gs, np.arange(ng))
    n = np.bincount(gs, minlength=ng)
    k = start + np.clip(np.floor(q * (n - 1)).astype(int), 0, None)
    out = np.full(ng, np.nan)
    ok = n > 0
    out[ok] = xs[k[ok]]
    return out


def outside_deepest(rows, stars):
    """Per star (aligned with stars): n_out, f_low, f_high, bowley, from epochs outside the deepest visit."""
    gid, vid, z = rows["gid"], rows["vid"], rows["z"]
    ng = len(stars)
    zz = np.where(np.isfinite(z), z, -np.inf)
    order = np.lexsort((-zz, gid))
    first = np.searchsorted(gid[order], np.arange(ng))
    deep_vid = np.full(ng, -1)
    has = first < len(order)
    deep_vid[has] = vid[order[np.minimum(first[has], len(order) - 1)]]
    out = (vid != deep_vid[gid]) & np.isfinite(z)
    g, zo = gid[out], z[out]
    n = np.bincount(g, minlength=ng).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_low = np.bincount(g, weights=(zo >= Z_PT).astype(float), minlength=ng) / n
        f_high = np.bincount(g, weights=(zo <= -Z_PT).astype(float), minlength=ng) / n
        q1, q2, q3 = (_group_quantile(g, zo, ng, q) for q in (0.25, 0.5, 0.75))
        bowley = (q3 + q1 - 2 * q2) / (q3 - q1)
    return pd.DataFrame({"source_id": stars.source_id.to_numpy(), "n_out": n, "f_low": f_low, "f_high": f_high,
                         "bowley": bowley})
