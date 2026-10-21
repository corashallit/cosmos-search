"""Find localised flux deficits (and, symmetrically, excesses) at several time scales.

The statistic is deliberately simple and shape-agnostic: a moving mean of the residual over a window of
duration d, divided by its own empirically measured scatter. Dividing by the measured scatter of the
moving mean, not by sigma/sqrt(n), absorbs red noise and stellar variability at that scale (the "beta
factor" of Pont et al. 2006, estimated robustly), so a star that wobbles by 1% on day scales needs a deeper
dip to be flagged than a quiet one. That is the right null for "is this unusual for this star?".

The baseline is fitted with asymmetric clipping: points well below the trend are excluded and the trend
bridges them. Occultation only removes light, so a baseline that ignores deficits but follows excesses
keeps long dips from being absorbed into the trend.
"""

from dataclasses import dataclass, asdict

import numpy as np

from .lightcurve import robust_std, segments

# Window durations searched, in days: ~1 h to ~4 d. Boyajian's star's dips lasted ~0.5 to several days;
# the shortest scale catches sharp-edged or small-body events.
DEFAULT_SCALES = (0.04, 0.125, 0.4, 1.2, 3.6)


@dataclass
class Event:
    sign: int            # -1 deficit (dip), +1 excess
    scale: float         # window duration that gave the peak statistic, d
    t_peak: float        # time of extreme moving mean
    snr: float           # |statistic| at peak, in units of the empirical scatter at that scale
    depth: float         # fractional deficit (or excess) of the moving mean at peak
    t_start: float       # span where the scale-d moving mean stays beyond half the peak
    t_end: float
    eqw: float           # equivalent width: integral of |residual| over the span, days
    asym: float          # (egress - ingress) / (egress + ingress); >0 means slow recovery
    n_points: int
    segment: int         # index of the gap-free segment it falls in
    edge_dist: float     # distance from the event span to the nearest segment edge, d

    def as_dict(self):
        return asdict(self)


def baseline(t, flux, window, k_lo=2.0, k_hi=4.0, n_iter=5, max_gap=0.5):
    """Robust trend with one-sided clipping, computed per gap-free segment.

    Bins of width window/4 take a median of unclipped points; the trend at each bin is the median of bin
    medians within +-window/2; it is linearly interpolated to every point, bridging empty bins.
    """
    out = np.full(len(t), np.nan)
    for sl in segments(t, max_gap):
        tt, ff = t[sl], flux[sl]
        keep = np.ones(len(tt), bool)
        trend = np.full(len(tt), np.median(ff))
        for _ in range(n_iter):
            trend = _binned_trend(tt, ff, keep, window)
            r = ff - trend
            s = robust_std(r[keep])
            if not np.isfinite(s) or s == 0:
                break
            new_keep = (r > -k_lo * s) & (r < k_hi * s)
            # Grow the clipped regions by a quarter window so dip wings don't drag the trend down.
            new_keep = ~_dilate(tt, ~new_keep, window / 4)
            if new_keep.sum() < max(5, 0.2 * len(tt)) or np.array_equal(new_keep, keep):
                break
            keep = new_keep
        out[sl] = trend
    return out


def _binned_trend(t, f, keep, window):
    width = window / 4
    if t[-1] - t[0] < width:
        m = np.median(f[keep]) if keep.any() else np.median(f)
        return np.full(len(t), m)
    edges = np.arange(t[0], t[-1] + width, width)
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 1)
    nb = len(edges)
    centres = edges + width / 2
    med = np.full(nb, np.nan)
    order = np.argsort(idx[keep], kind="stable")
    kidx, kf = idx[keep][order], f[keep][order]
    if len(kidx):
        starts = np.searchsorted(kidx, np.arange(nb), side="left")
        stops = np.searchsorted(kidx, np.arange(nb), side="right")
        for b in np.flatnonzero(stops > starts):
            med[b] = np.median(kf[starts[b]:stops[b]])
    good = np.isfinite(med)
    if not good.any():
        return np.full(len(t), np.median(f))
    # Smooth bin medians over +-window/2 (two bins each side).
    sm = np.full(nb, np.nan)
    gb = np.flatnonzero(good)
    for b in gb:
        lo, hi = np.searchsorted(gb, [b - 2, b + 3])
        sm[b] = np.median(med[gb[lo:hi]])
    return np.interp(t, centres[good], sm[good])


def _dilate(t, mask, radius):
    if not mask.any():
        return mask
    tm = t[mask]
    lo = np.searchsorted(tm, t - radius, side="left")
    hi = np.searchsorted(tm, t + radius, side="right")
    return hi > lo


def moving_mean(t, r, width):
    """Mean of r over |t_j - t_i| <= width/2, and the count, at every point. Irregular sampling is fine."""
    c = np.concatenate([[0.0], np.cumsum(r)])
    lo = np.searchsorted(t, t - width / 2, side="left")
    hi = np.searchsorted(t, t + width / 2, side="right")
    n = hi - lo
    return (c[hi] - c[lo]) / np.maximum(n, 1), n


def local_noise(t, r, window, clip=20.0):
    """Per-point white-noise level: moving RMS of point-to-point differences over `window` days.

    Noise is not stationary (scattered light, pointing jitter, crowding change within a sector), and a
    single global sigma turns every noisy stretch into a string of "events". Differences are clipped at
    `clip` times the global level so one sharp dip edge does not inflate its own noise estimate much.
    The local level is floored at the global level.
    """
    if len(r) < 3:
        return np.full(len(r), robust_std(r) if len(r) else np.nan)
    d = np.diff(r) / np.sqrt(2)
    g = robust_std(d)
    if not np.isfinite(g) or g == 0:
        g = robust_std(r) or 1.0
    d = np.clip(np.abs(d), 0, clip * g)
    # Assign each difference to the later point; the first point copies the second.
    d2 = np.concatenate([[d[0]], d]) ** 2
    ms, _ = moving_mean(t, d2, window)
    # RMS of a Gaussian equals its sigma; MAD-based g is the robust equivalent for the floor.
    return np.maximum(np.sqrt(ms), g)


@dataclass
class ScaleTrace:
    scale: float
    resid: np.ndarray     # flux / baseline - 1
    mean: np.ndarray      # moving mean of resid at this scale
    stat: np.ndarray      # mean / empirical scatter of the white-noise z; signed
    beta: float           # red-noise inflation at this scale (>= 1)


def scan(lc, scales=DEFAULT_SCALES, window_factor=4.0, min_window=1.0, max_gap=0.5, min_points=3,
         noise_window=0.5):
    """Per-scale residuals and statistics for a cleaned LightCurve."""
    traces = []
    t, f = lc.t, lc.flux
    for d in scales:
        w = max(window_factor * d, min_window)
        base = baseline(t, f, w, max_gap=max_gap)
        r = f / base - 1.0
        sig = local_noise(t, r, noise_window)
        m, n = moving_mean(t, r, d)
        z = m / (sig / np.sqrt(np.maximum(n, 1)))
        z[n < min_points] = 0.0
        beta = max(1.0, robust_std(z[n >= min_points])) if (n >= min_points).sum() > 10 else 1.0
        traces.append(ScaleTrace(d, r, m, z / beta, beta))
    return traces


def find_events(lc, traces=None, threshold=5.0, sign=-1, max_gap=0.5, **scan_kw):
    """Events whose |statistic| exceeds `threshold` in the direction `sign`, merged across scales."""
    if traces is None:
        traces = scan(lc, max_gap=max_gap, **scan_kw)
    t = lc.t
    segs = segments(t, max_gap)
    seg_id = np.zeros(len(t), int)
    for i, sl in enumerate(segs):
        seg_id[sl] = i
    found = []
    for tr in traces:
        s = sign * tr.stat
        hot = s > threshold
        if not hot.any():
            continue
        # Group hot points into runs: consecutive hot indices without a time gap wider than the scale.
        idx = np.flatnonzero(hot)
        splits = np.flatnonzero((np.diff(idx) > 1) | (np.diff(t[idx]) > max(tr.scale, max_gap))) + 1
        for run in np.split(idx, splits):
            k = run[np.argmax(s[run])]
            found.append(_measure(t, tr, k, sign, seg_id, segs))
    return _merge(found)


def _measure(t, tr, k, sign, seg_id, segs):
    m = sign * tr.mean
    peak = m[k]
    sl = segs[seg_id[k]]
    # Walk out from the peak while the moving mean stays above half the peak, inside the segment.
    a = k
    while a > sl.start and m[a - 1] > peak / 2:
        a -= 1
    b = k
    while b < sl.stop - 1 and m[b + 1] > peak / 2:
        b += 1
    t0, t1 = t[a], t[b]
    inside = slice(a, b + 1)
    rr = sign * tr.resid[inside]
    tt = t[inside]
    eqw = float(np.trapezoid(np.clip(rr, 0, None), tt)) if len(tt) > 1 else float(max(rr[0], 0) * tr.scale)
    # Ingress/egress from the unsmoothed residual's extreme inside the span.
    j = int(np.argmax(rr)) if len(rr) else 0
    tmin = tt[j] if len(tt) else t[k]
    ing, egr = tmin - t0, t1 - tmin
    asym = (egr - ing) / (egr + ing) if (egr + ing) > 0 else 0.0
    seg_t0, seg_t1 = t[sl.start], t[sl.stop - 1]
    return Event(
        sign=sign, scale=tr.scale, t_peak=float(t[k]), snr=float(sign * tr.stat[k]), depth=float(peak),
        t_start=float(t0), t_end=float(t1), eqw=eqw, asym=float(asym), n_points=int(b - a + 1),
        segment=int(seg_id[k]), edge_dist=float(min(t0 - seg_t0, seg_t1 - t1)),
    )


def _merge(events):
    """Overlapping events from different scales are one event; keep the highest-SNR description."""
    events = sorted(events, key=lambda e: -e.snr)
    kept = []
    for e in events:
        if any(e.t_start <= k.t_end and k.t_start <= e.t_end for k in kept):
            continue
        kept.append(e)
    return sorted(kept, key=lambda e: e.t_peak)
