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
  colour of the change. Grey bodies give equal fractional depths; dust gives depth_bp > depth_rp. `depth_g` is
  G's on the same footing, and `slope_g`, `slope_bp`, `slope_rp` (and errors) each band's trend over its own
  visits, so G's change can be checked against BP's and RP's (`checks.band_agreement`). All of these use only
  transits measured in all three bands: DPAC rejects 2-5% of epochs in one band and not the others (no more often
  in a dip than outside one, on the DR3 trial), and for a dip seen in one or two transits a missing BP or RP epoch
  alone made eclipses read as G-only changes (2026-10-30). `dim_depth_err` is the error of `dim_depth` (all G
  epochs).

Which level is the star's normal one. `s_dim` measures runs against the median, so it takes the level the star
spends most of its visits at as normal. A star that brightens for good partway through, and spends more than half
its visits bright, then shows its earlier visits as a dimming episode: on DR3, 16% of late-starting brightenings
were flagged as fades this way (slow-blind-1). `s_fade` and `s_rise` read the light curve in time order instead:
a run of faint visits that starts the light curve, or follows visits that are themselves faint (by more than half
the run's depth), is a star that got brighter, so it scores in `s_rise`; a run of bright visits that starts the
light curve is a star that got fainter afterwards, so it scores in `s_fade`. Closed runs (normal visits before
them) score as before. The two are mirror images, so `s_rise` stays the false-alarm estimate for `s_fade`. The
cost is a dimming episode already under way at the first visit, which reads as a rise; DR4's longer span makes
those rarer. `fade_kind` says which run made `s_fade`: "dip" (normal before and after), "fade" (normal before,
faint to the last visit) or "drop" (bright first, fainter for the rest); `rise_kind` likewise for `s_rise` ("bump",
"rise", "jump"), with its bright window `rise_t0`-`rise_t1`. The episode window, `dim_depth` and the
band depths describe that run, with the normal level taken from the visits outside the window. `s_dim` and
`s_bright` stay as they were for the frozen rules slow-v1 and slow-v2.
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


def _runs(tv, yv, sigma):
    """Every candidate run of consecutive visits with its dimming and brightening scores (deepest or brightest
    visit left out) and whether it is open at the start (see the module docstring). None if no run qualifies."""
    n = len(yv)
    dev = yv - np.median(yv)
    c = np.r_[0.0, np.cumsum(dev)]
    i, j = np.triu_indices(n + 1, k=MIN_EPISODE_VISITS)
    length = j - i
    keep = (length <= MAX_EPISODE_FRACTION * n) & (tv[j - 1] - tv[i] >= MIN_EPISODE_DAYS)
    i, j, length = i[keep], j[keep], length[keep]
    if not len(i):
        return None
    # lo[a, b] = min(dev[a:b + 1]), hi likewise, for b >= a.
    later = np.arange(n)[None, :] >= np.arange(n)[:, None]
    lo = np.minimum.accumulate(np.where(later, dev[None, :], np.inf), axis=1)
    hi = np.maximum.accumulate(np.where(later, dev[None, :], -np.inf), axis=1)
    run = c[j] - c[i]
    norm = np.sqrt(length - 1) * sigma
    dim = -(run - lo[i, j - 1]) / norm  # positive = fainter than the median
    bright = (run - hi[i, j - 1]) / norm
    # Open at the start: the run begins the light curve, or the visits before it sit on the run's side of the
    # median by more than half the run's mean offset (the same level continues back to the first visit).
    with np.errstate(invalid="ignore", divide="ignore"):
        lead = np.where(i > 0, c[i] / np.maximum(i, 1), np.nan)
    half = run / length / 2
    open_lo = (i == 0) | (lead < half)
    open_hi = (i == 0) | (lead > half)
    return i, j, dim, bright, open_lo, open_hi


def episodes(tv, yv, sigma):
    """Strongest dimming and brightening runs of consecutive visits, against the median.

    Each run's most extreme visit (deepest for dimming, brightest for brightening) is left out of its sum, so no
    single visit can make an episode: one deep visit is a short dip, which the short-dip search covers.
    Returns (s_dim, i, j, s_bright, i, j) with runs as visit index ranges [i, j)."""
    r = _runs(tv, yv, sigma)
    if r is None:
        return np.nan, -1, -1, np.nan, -1, -1
    i, j, dim, bright = r[:4]
    a, b = int(np.argmax(dim)), int(np.argmax(bright))
    return float(dim[a]), int(i[a]), int(j[a]), float(bright[b]), int(i[b]), int(j[b])


def _fade(tv, yv, sigma):
    """Strongest getting-fainter run in time order: (score, kind, t0, t1) with [t0, t1] the faint window."""
    r = _runs(tv, yv, sigma)
    if r is None:
        return np.nan, "", np.nan, np.nan
    i, j, dim, bright, open_lo, open_hi = r
    # A run open at the start that also reaches the last visit shows no change in time either way: neither.
    to_end = j == len(yv)
    fade = np.maximum(np.where(open_lo, -np.inf, dim), np.where(open_hi & ~to_end, bright, -np.inf))
    a = int(np.argmax(fade))
    n = len(yv)
    if open_hi[a] and not to_end[a] and bright[a] >= dim[a]:
        return float(fade[a]), "drop", float(tv[j[a]]), float(tv[-1])
    return float(fade[a]), ("dip" if j[a] < n else "fade"), float(tv[i[a]]), float(tv[j[a] - 1])


def ordered_episodes(tv, yv, sigma):
    """The runs read in time order (module docstring): the strongest getting-fainter run and, as its mirror, the
    strongest getting-brighter run (the same on the flipped light curve).

    Returns ((s_fade, kind, t0, t1), (s_rise, kind, t0, t1)): kinds "dip", "fade" or "drop" ("bump", "rise",
    "jump" for brightening), windows the faint (bright) visits."""
    up = _fade(tv, -yv, sigma)
    names = {"dip": "bump", "fade": "rise", "drop": "jump", "": ""}
    return _fade(tv, yv, sigma), (up[0], names[up[1]], up[2], up[3])


def _trend(tv, yv, sig):
    """OLS slope per year of visit means and its error from the larger of sig and the scatter about the line."""
    x = (tv - tv.mean()) / 365.25
    sxx = np.sum(x * x)
    if not sxx > 0 or len(tv) < 3:
        return np.nan, np.nan
    slope = float(np.sum(x * (yv - yv.mean())) / sxx)
    resid = yv - yv.mean() - slope * x
    return slope, float(max(sig, np.sqrt(np.sum(resid ** 2) / (len(tv) - 2))) / np.sqrt(sxx))


def _band_slope(t, y):
    """One band's trend (`_trend`) over its own visits, with that band's visit scatter."""
    if len(t) < MIN_VISITS:
        return np.nan, np.nan
    tv, yv = visits(t, y)
    return _trend(tv, yv, visit_sigma(yv)) if len(tv) >= MIN_VISITS else (np.nan, np.nan)


def _band_depth(t, y, t0, t1):
    """Mean deficit of one band's visits inside [t0, t1] against the median of its visits outside, and its
    error. t0 and t1 are G visit times; a BP or RP visit's mean time can differ from G's by minutes (an epoch kept
    in one band and not the other), so the window is widened by half the visit gap. Without that, the DR3 trial's
    one-visit eclipses lost their first or last visit in BP and RP and read as G-only changes (2026-10-30)."""
    if len(t) < MIN_VISITS:
        return np.nan, np.nan
    tv, yv = visits(t, y)
    inside = (tv >= t0 - VISIT_GAP_DAYS / 2) & (tv <= t1 + VISIT_GAP_DAYS / 2)
    if inside.sum() < 1 or (~inside).sum() < 1:
        return np.nan, np.nan
    sig = visit_sigma(yv)
    return (float(np.median(yv[~inside]) - yv[inside].mean()),
            float(sig * np.sqrt(1 / inside.sum() + np.pi / 2 / (~inside).sum())))


def same_transits(bands, transit_ids, offset=0):
    """Row positions (plus offset) of each band, {"g", "bp", "rp"}, keeping only transits that all three bands
    have; every row of each band when transit ids are not given."""
    rows = {name: np.flatnonzero(bands == code) for code, name in ((0, "g"), (1, "bp"), (2, "rp"))}
    if transit_ids is not None:
        common = np.intersect1d(np.intersect1d(transit_ids[rows["g"]], transit_ids[rows["bp"]]),
                                transit_ids[rows["rp"]])
        rows = {k: r[np.isin(transit_ids[r], common)] for k, r in rows.items()}
    return {k: r + offset for k, r in rows.items()}


def star_features(epochs, col="frac_corr"):
    """One row per star with slow-change statistics (G for detection, BP/RP for the episode's colour)."""
    e = epochs.sort_values(["source_id", "band", "t"], kind="stable")
    sid = e["source_id"].to_numpy()
    band = e["band"].to_numpy()
    t = e["t"].to_numpy()
    y = e[col].to_numpy()
    tid = e["transit_id"].to_numpy() if "transit_id" in e else None
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
        s_dim, _, _, s_br, _, _ = episodes(tv, yv, sig)
        (s_fade, kind, t0, t1), (s_rise, rkind, r0, r1) = ordered_episodes(tv, yv, sig)
        row = {"source_id": int(sid[a]), "n_visits": len(tv), "t_first": tv[0], "t_last": tv[-1],
               "sigma_visit": sig, "abbe": abbe(yv),
               "slope": slope, "slope_z": slope / (sig_trend / np.sqrt(sxx)) if sxx > 0 else np.nan,
               "s_dim": s_dim, "s_bright": s_br, "s_fade": s_fade, "s_rise": s_rise, "fade_kind": kind,
               "rise_kind": rkind, "rise_t0": r0, "rise_t1": r1,
               "dim_depth": np.nan, "dim_depth_err": np.nan, "dim_t0": np.nan, "dim_t1": np.nan, "dim_visits": 0,
               **{f"depth_{n}{x}": np.nan for n in ("g", "bp", "rp") for x in ("", "_err")}}
        band_rows = same_transits(bb, tid[a:b] if tid is not None else None, a)
        for name, m in band_rows.items():
            row[f"slope_{name}"], row[f"slope_{name}_err"] = _band_slope(t[m], y[m])
        if kind:
            inside = (tv >= t0) & (tv <= t1)
            if (~inside).any():
                row.update(dim_depth=float(np.median(yv[~inside]) - yv[inside].mean()),
                           dim_depth_err=float(sig * np.sqrt(1 / inside.sum() + np.pi / 2 / (~inside).sum())))
            row.update(dim_t0=t0, dim_t1=t1, dim_visits=int(inside.sum()))
            for name, m in band_rows.items():
                d, de = _band_depth(t[m], y[m], t0, t1)
                row[f"depth_{name}"], row[f"depth_{name}_err"] = d, de
        rows.append(row)
    return pd.DataFrame(rows)
