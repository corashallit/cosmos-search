"""Per-star statistics for slow changes: trend, smoothness and the strongest long dimming (or brightening) episode.

Works on drift-corrected epochs (`drift.normalise` plus a `frac_corr` column). Epochs are first combined into
visits (groups less than VISIT_GAP_DAYS apart: Gaia's two fields of view and consecutive scans), because a slow
change is a statement about visits, not about the epochs inside one.

Noise is measured on each star's own visit-to-visit differences, sigma_visit = 1.4826 MAD(diff) / sqrt(2). That
captures photon noise and any variability faster than the gap between visits, and is nearly blind to changes
slower than that. Every significance below is in units of that sigma, so an active or pulsating star does not
pass simply for being variable; it passes only if its slow component stands out above its own fast scatter.

- `slope_z`: OLS trend over visit means, in units of its error from the larger of sigma_visit and the scatter
  about the fitted line. sigma_visit alone (a median) misses rare large jumps: a cataclysmic variable's outburst
  early in the data, or a long-period variable sampled at a few extreme phases, then reads as a steep fade.
- `abbe`: von Neumann ratio / 2 of the visit means, ~1 for white noise and ~0 for a smooth change. The same
  statistic DR4 publishes for nearly every source (`all_source_photometry.phot_{g,bp,rp}_abbe_value`), so a
  whole-sky selection on TAP can be calibrated against it.
- `s_dim`: the largest deficit of any run of consecutive visits lasting at least MIN_EPISODE_DAYS, against the
  star's median, as (median - run mean) * sqrt(n_run) / sigma_visit, with the run's deepest visit left out (so
  n_run counts the others) and no single visit can make an episode. `s_bright` is the same for excesses: its
  distribution is the false-alarm estimate for `s_dim` (statistics.md §2). Runs are capped at
  MAX_EPISODE_FRACTION of the visits so the median stays a baseline.
- `depth_bp`, `depth_rp` (and errors): the mean BP and RP deficits over the G episode's time window, for the
  colour of the change. Grey bodies give equal fractional depths; dust gives depth_bp > depth_rp.
"""

import numpy as np
import pandas as pd

VISIT_GAP_DAYS = 2.0
MIN_EPISODE_DAYS = 20.0
MIN_EPISODE_VISITS = 3
MAX_EPISODE_FRACTION = 0.6
MIN_VISITS = 8
SIGMA_FLOOR = 1e-5


def visits(t, y):
    """Visit mean times and values for epochs sorted by time (gaps > VISIT_GAP_DAYS start a new visit)."""
    new = np.r_[True, np.diff(t) > VISIT_GAP_DAYS]
    idx = np.cumsum(new) - 1
    n = np.bincount(idx).astype(float)
    return np.bincount(idx, t) / n, np.bincount(idx, y) / n


def visit_sigma(yv):
    """Robust scatter of one visit from first differences of visit means. (Second differences, which cancel a
    change close to linear over three visits, found 215 rather than 211 of 358 fading alerts on 2026-10-23: not
    worth their noisier estimate.)"""
    d = np.diff(yv)
    return max(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2), SIGMA_FLOOR)


def abbe(yv):
    v = np.var(yv, ddof=1)
    return float(np.sum(np.diff(yv) ** 2) / (len(yv) - 1) / (2 * v)) if v > 0 else np.nan


def episodes(tv, yv, sigma):
    """Strongest dimming and brightening runs of consecutive visits.

    Each run's most extreme visit (deepest for dimming, brightest for brightening) is left out of its sum, so no
    single visit can make an episode: one deep visit is a short dip, which the short-dip search covers.
    Returns (s_dim, i, j, s_bright, i, j) with runs as visit index ranges [i, j)."""
    n = len(yv)
    dev = yv - np.median(yv)
    c = np.r_[0.0, np.cumsum(dev)]
    i, j = np.triu_indices(n + 1, k=MIN_EPISODE_VISITS)
    length = j - i
    keep = (length <= MAX_EPISODE_FRACTION * n) & (tv[j - 1] - tv[i] >= MIN_EPISODE_DAYS)
    i, j, length = i[keep], j[keep], length[keep]
    if not len(i):
        return np.nan, -1, -1, np.nan, -1, -1
    # lo[a, b] = min(dev[a:b + 1]), hi likewise, for b >= a.
    later = np.arange(n)[None, :] >= np.arange(n)[:, None]
    lo = np.minimum.accumulate(np.where(later, dev[None, :], np.inf), axis=1)
    hi = np.maximum.accumulate(np.where(later, dev[None, :], -np.inf), axis=1)
    run = c[j] - c[i]
    norm = np.sqrt(length - 1) * sigma
    dim = -(run - lo[i, j - 1]) / norm  # positive = fainter than the median
    bright = (run - hi[i, j - 1]) / norm
    a, b = int(np.argmax(dim)), int(np.argmax(bright))
    return float(dim[a]), int(i[a]), int(j[a]), float(bright[b]), int(i[b]), int(j[b])


def _band_depth(t, y, t0, t1):
    """Mean deficit of one band's visits inside [t0, t1] against that band's median, and its error."""
    if len(t) < MIN_VISITS:
        return np.nan, np.nan
    tv, yv = visits(t, y)
    inside = (tv >= t0) & (tv <= t1)
    if inside.sum() < 1:
        return np.nan, np.nan
    sig = visit_sigma(yv)
    return float(np.median(yv) - yv[inside].mean()), float(sig / np.sqrt(inside.sum()))


def star_features(epochs, col="frac_corr"):
    """One row per star with slow-change statistics (G for detection, BP/RP for the episode's colour)."""
    e = epochs.sort_values(["source_id", "band", "t"], kind="stable")
    sid = e["source_id"].to_numpy()
    band = e["band"].to_numpy()
    t = e["t"].to_numpy()
    y = e[col].to_numpy()
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    ends = np.r_[starts[1:], len(sid)]
    rows = []
    for a, b in zip(starts, ends):
        bb = band[a:b]
        g = slice(a, a + int(np.sum(bb == 0)))  # G rows come first within a star (band 0)
        tg, yg = t[g], y[g]
        if len(tg) < MIN_VISITS:
            continue
        tv, yv = visits(tg, yg)
        if len(tv) < MIN_VISITS:
            continue
        sig = visit_sigma(yv)
        x = (tv - tv.mean()) / 365.25
        sxx = np.sum(x * x)
        slope = float(np.sum(x * (yv - yv.mean())) / sxx) if sxx > 0 else np.nan
        resid = yv - yv.mean() - slope * x
        sig_trend = max(sig, float(np.sqrt(np.sum(resid ** 2) / (len(tv) - 2))))
        s_dim, di, dj, s_br, _, _ = episodes(tv, yv, sig)
        row = {"source_id": int(sid[a]), "n_visits": len(tv), "t_first": tv[0], "t_last": tv[-1],
               "sigma_visit": sig, "abbe": abbe(yv),
               "slope": slope, "slope_z": slope / (sig_trend / np.sqrt(sxx)) if sxx > 0 else np.nan,
               "s_dim": s_dim, "s_bright": s_br, "dim_depth": np.nan, "dim_t0": np.nan, "dim_t1": np.nan,
               "dim_visits": 0, "depth_bp": np.nan, "depth_bp_err": np.nan, "depth_rp": np.nan,
               "depth_rp_err": np.nan}
        if di >= 0:
            t0, t1 = tv[di], tv[dj - 1]
            row.update(dim_depth=float(np.median(yv) - yv[di:dj].mean()), dim_t0=t0, dim_t1=t1,
                       dim_visits=dj - di)
            for code, name in ((1, "bp"), (2, "rp")):
                m = np.flatnonzero(bb == code) + a
                d, de = _band_depth(t[m], y[m], t0, t1)
                row[f"depth_{name}"], row[f"depth_{name}_err"] = d, de
        rows.append(row)
    return pd.DataFrame(rows)
