"""Per-light-curve features: what the detector found, and how this star varies in general.

Two kinds, kept apart on purpose:

- Event features describe the strongest deficit and excess (depth, duration, shape, and whether it
  coincides with anything in a nuisance channel such as background or centroid).
- Shape-agnostic features summarise the variability as a whole in noise-normalised units, so stars of
  different brightness are comparable: the extreme statistic and red-noise inflation at every scale,
  quantile skew, tail ratios, quasi-periodicity, smoothness, and multimodality of the flux distribution.
  These feed the population anomaly score (population.py), which asks "how unusual is this light curve
  relative to the others" without saying what to look for.
"""

import numpy as np

from .detect import DEFAULT_SCALES, find_events, scan
from .lightcurve import robust_std, segments

# Auxiliary series whose names start with this prefix are alternative photometry of the same star
# (other apertures, other pipelines); the rest are nuisance channels (background, centroids).
ALT_FLUX_PREFIX = "flux_"


def extract(lc, traces=None, scales=DEFAULT_SCALES, threshold=5.0, max_gap=0.5, flag_bits=None, **scan_kw):
    """Features dict and event lists for a cleaned LightCurve.

    traces: output of detect.scan, if already computed. flag_bits: optional {name: bitmask} of flags to
    report as the fraction of flagged points inside the best dip (the points are kept in the light curve so
    events in flagged spans are still visible).
    """
    if traces is None:
        traces = scan(lc, scales=scales, max_gap=max_gap, **scan_kw)
    dips = find_events(lc, traces, threshold=threshold, sign=-1, max_gap=max_gap)
    bursts = find_events(lc, traces, threshold=threshold, sign=+1, max_gap=max_gap)
    f = {"n_points": len(lc), "span": float(lc.t[-1] - lc.t[0]) if len(lc) else 0.0}

    # Noise-normalised variability spectrum: one number per scale for each of these.
    for tr in traces:
        key = _scale_key(tr.scale)
        f[f"min_stat_{key}"] = float(np.min(tr.stat)) if len(tr.stat) else 0.0
        f[f"max_stat_{key}"] = float(np.max(tr.stat)) if len(tr.stat) else 0.0
        f[f"log_beta_{key}"] = float(np.log10(tr.beta))
        neg = (tr.stat < -4).sum()
        pos = (tr.stat > 4).sum()
        f[f"tail_ratio_{key}"] = float(np.log10((neg + 1) / (pos + 1)))

    r = traces[min(2, len(traces) - 1)].resid  # residual from the ~1.6 d baseline
    sig = robust_std(np.diff(r)) / np.sqrt(2)
    f["sigma_white"] = float(sig)
    f["rms"] = float(robust_std(r))
    f["rms_over_white"] = float(f["rms"] / sig) if sig > 0 else np.nan
    f.update(_shape_stats(lc.flux, sig))
    f.update(_periodicity(lc, sig))
    f["n_dips"] = len(dips)
    f["n_bursts"] = len(bursts)
    f["dip_periodicity"] = _event_periodicity(dips)
    f["dip_self_similarity"] = _self_similarity(lc, dips)
    _add_event(f, "dip", dips, lc, traces, flag_bits)
    _add_event(f, "burst", bursts, lc, traces, None)
    f["n_segments"] = len(segments(lc.t, max_gap))
    return f, dips, bursts


def _scale_key(d):
    return f"{d * 24:g}h"


def _shape_stats(flux, sig):
    x = flux[np.isfinite(flux)]
    if len(x) < 10:
        return {}
    q = np.percentile(x, [1, 10, 50, 90, 99])
    spread = q[3] - q[1]
    out = {
        # Quantile skew, robust to outliers: negative when the low tail is longer (dimming).
        "qskew_10": float((q[3] + q[1] - 2 * q[2]) / spread) if spread > 0 else 0.0,
        "qskew_1": float((q[4] + q[0] - 2 * q[2]) / (q[4] - q[0])) if q[4] > q[0] else 0.0,
        "low_tail_sigma": float((q[2] - q[0]) / sig) if sig > 0 else np.nan,
        "high_tail_sigma": float((q[4] - q[2]) / sig) if sig > 0 else np.nan,
    }
    # Cody et al. (2014) flux asymmetry M, sign flipped so that dips give M > 0 as in their magnitude form.
    lo, hi = np.percentile(x, [10, 90])
    ext = x[(x <= lo) | (x >= hi)]
    rms = np.std(x)
    out["cody_M"] = float((np.median(x) - np.mean(ext)) / rms) if rms > 0 else 0.0
    # Sarle's bimodality coefficient: > 5/9 suggests discrete levels.
    n = len(x)
    m = x - x.mean()
    s2 = np.mean(m ** 2)
    if s2 > 0 and n > 3:
        g = np.mean(m ** 3) / s2 ** 1.5
        k = np.mean(m ** 4) / s2 ** 2 - 3
        out["bimodality"] = float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))
    # von Neumann ratio: small when variability is smooth relative to point-to-point noise.
    out["von_neumann"] = float(np.mean(np.diff(x) ** 2) / s2) if s2 > 0 else np.nan
    return out


def _periodicity(lc, sig, n_bins=50):
    """Lomb-Scargle peak, and Cody et al.'s Q: variance left after removing the folded mean shape.

    Q ~ 0 for strictly periodic variability (EBs, spots), ~ 1 for aperiodic (bursters, dippers).
    """
    from astropy.timeseries import LombScargle

    t, y = lc.t, lc.flux
    span = t[-1] - t[0]
    if len(t) < 50 or span < 1:
        return {"ls_power": np.nan, "ls_period": np.nan, "Q": np.nan}
    fmin, fmax = 1 / min(span / 2, 15.0), 1 / 0.05
    tb, yb = _bin(t, y, 0.02)
    freq = np.linspace(fmin, fmax, 5000)
    power = LombScargle(tb, yb).power(freq)
    k = int(np.argmax(power))
    p = 1 / freq[k]
    # Fold at P and 2P (EBs often peak at P/2) and keep whichever leaves less variance.
    best_q = np.inf
    for period in (p, 2 * p):
        ph = ((t - t[0]) / period) % 1
        idx = np.minimum((ph * n_bins).astype(int), n_bins - 1)
        prof = np.array([np.median(y[idx == b]) if (idx == b).any() else np.nan for b in range(n_bins)])
        prof = np.where(np.isfinite(prof), prof, np.nanmedian(prof))
        resid = y - prof[idx]
        raw, left = np.var(y) - sig ** 2, np.var(resid) - sig ** 2
        q = left / raw if raw > 0 else np.nan
        if np.isfinite(q) and q < best_q:
            best_q = q
    return {"ls_power": float(power[k]), "ls_period": float(p), "Q": float(np.clip(best_q, 0, 2))}


def _bin(t, y, width):
    idx = np.floor((t - t[0]) / width).astype(int)
    _, inv, counts = np.unique(idx, return_inverse=True, return_counts=True)
    return np.bincount(inv, t) / counts, np.bincount(inv, y) / counts


def _event_periodicity(events, tol=0.05):
    """Fraction of dip pairs explained by one period (with harmonics). EBs score ~1, aperiodic dippers low."""
    if len(events) < 3:
        return np.nan
    tp = np.array([e.t_peak for e in events])
    gaps = np.diff(tp)
    p0 = np.min(gaps)
    best = 0.0
    for p in (p0, p0 / 2):
        ratio = (tp - tp[0]) / p
        frac = np.mean(np.abs(ratio - np.round(ratio)) < tol + 0.5 * np.mean([e.t_end - e.t_start for e in events]) / p)
        best = max(best, frac)
    return float(best)


def _self_similarity(lc, events, n=32):
    """Highest correlation between profiles of two distinct dips, resampled onto a common grid.

    Natural dust dips vary in shape; an identical complex shape repeating at irregular times would be odd.
    """
    if len(events) < 2:
        return np.nan
    profiles = []
    for e in events[:12]:
        half = max(e.t_end - e.t_start, e.scale)
        grid = np.linspace(e.t_peak - half, e.t_peak + half, n)
        sel = (lc.t >= grid[0]) & (lc.t <= grid[-1])
        if sel.sum() < 5:
            continue
        p = np.interp(grid, lc.t[sel], lc.flux[sel])
        p = p - p.mean()
        nrm = np.linalg.norm(p)
        if nrm > 0:
            profiles.append(p / nrm)
    if len(profiles) < 2:
        return np.nan
    P = np.array(profiles)
    c = P @ P.T
    np.fill_diagonal(c, -1)
    return float(c.max())


def _add_event(f, name, events, lc, traces, flag_bits):
    if not events:
        f[f"{name}_snr"] = 0.0
        return
    e = max(events, key=lambda ev: ev.snr)
    for k in ("snr", "depth", "scale", "eqw", "asym", "edge_dist", "t_peak"):
        f[f"{name}_{k}"] = float(getattr(e, k))
    f[f"{name}_duration"] = float(e.t_end - e.t_start)
    # Edge sharpness: steepest flux change across the event, in units of white noise per cadence.
    sel = (lc.t >= e.t_start - e.scale) & (lc.t <= e.t_end + e.scale)
    if sel.sum() > 3:
        tr = min(traces, key=lambda x: abs(x.scale - e.scale))
        d = np.abs(np.diff(tr.resid[sel]))
        sw = robust_std(np.diff(tr.resid)) / np.sqrt(2)
        f[f"{name}_max_step_sigma"] = float(np.max(d) / sw) if sw > 0 else np.nan
    if name == "dip":
        f.update({f"dip_{k}": v for k, v in event_diagnostics(lc, e, flag_bits).items()})


def event_diagnostics(lc, e, flag_bits=None):
    """Nuisance-channel tests, flagged-point fractions and distance to the nearest momentum dump (if the
    survey lists them in meta["desat_times"]) for one event."""
    out = _nuisance_tests(lc, e)
    out.update(edge_kinematics(lc, e))
    inside = (lc.t >= e.t_start) & (lc.t <= e.t_end)
    for fname, bits in (flag_bits or {}).items():
        out[f"flagfrac_{fname}"] = float(np.mean((lc.flags[inside] & bits) != 0)) if inside.any() else 0.0
    dumps = np.asarray(lc.meta.get("desat_times", []), float)
    if len(dumps):
        mid = 0.5 * (e.t_start + e.t_end)
        half = 0.5 * (e.t_end - e.t_start)
        out["dump_dist"] = float(max(0.0, np.min(np.abs(dumps - mid)) - half))
    return out


V_ESC_SUN_KM_S = 617.7
R_SUN_KM = 6.957e5


def edge_kinematics(lc, e, n_sigma=5.0):
    """Is the event's ingress or egress too fast for an occultation of this star?

    An occulter of radius r moving at speed v across the disk changes the flux over 2r/v and removes at
    most (r/R*)^2 of the light. So a flux change of fractional size D completed within one cadence dt
    needs r >= R* sqrt(D) and hence v >= 2 R* sqrt(D) / dt. Taking twice the star's surface escape speed
    as a generous ceiling for anything near it, the deepest change an occultation can make in one cadence
    is D_max = (v_max dt / 2 R*)^2. For a Sun-like star and 200 s cadence that is ~3%.

    Steps larger than that cannot be occultations: flares (which become "dips" in the mirror null),
    instrumental jumps, or something that changes the star's own output. They are reported separately
    rather than mixed into the occultation channel, so the mirror null is not dominated by flares.

    Returns edge_frac (largest single-cadence change across the event as a fraction of its depth),
    edge_sigma (that change in units of the difference noise), depth_peak, kinematic_limit and fast_edge.
    """
    t, f = lc.t, lc.flux
    if len(t) < 10:
        return {}
    dt = float(np.median(np.diff(t)))
    pad = max(e.scale, 3 * dt)
    sel = (t >= e.t_start - pad) & (t <= e.t_end + pad)
    around = (t >= e.t_start - 3 * pad - 0.5) & (t <= e.t_end + 3 * pad + 0.5) & ~sel
    if sel.sum() < 4 or around.sum() < 5:
        return {}
    base = np.median(f[around])
    ff = f[sel]
    sm = np.convolve(ff, np.ones(3) / 3, mode="same") if len(ff) >= 3 else ff
    depth_peak = float(base - np.min(sm[1:-1] if len(sm) > 2 else sm))
    tt = t[sel]
    # Only consecutive cadences count as a single-cadence step. A 3-point running median first, so one bad
    # point (cosmic ray, hot pixel) does not register as two steps, while a genuine step edge survives.
    contiguous = np.diff(tt) < 1.5 * dt
    fm = _median3(ff)
    steps = np.abs(np.diff(fm))[contiguous]
    sig_d = robust_std(np.diff(f))
    if not len(steps) or depth_peak <= 0 or not np.isfinite(sig_d) or sig_d == 0:
        return {"depth_peak": depth_peak}
    big = float(np.max(steps))
    radius = lc.meta.get("radius")
    mass = lc.meta.get("mass")
    radius = radius if radius and np.isfinite(radius) and radius > 0 else 1.0
    mass = mass if mass and np.isfinite(mass) and mass > 0 else 1.0
    v_max = 2 * V_ESC_SUN_KM_S * np.sqrt(mass / radius)
    limit = float(min(1.0, (v_max * dt * 86400 / (2 * radius * R_SUN_KM)) ** 2))
    edge_sigma = big / sig_d
    return {
        "depth_peak": depth_peak, "edge_frac": big / depth_peak, "edge_sigma": float(edge_sigma),
        "kinematic_limit": limit,
        "fast_edge": bool(edge_sigma > n_sigma and big > limit),
    }


def _median3(x):
    if len(x) < 3:
        return x
    m = np.median(np.column_stack([x[:-2], x[1:-1], x[2:]]), axis=1)
    return np.concatenate([[x[0]], m, [x[-1]]])


def _nuisance_tests(lc, e, n_null=64, seed=0):
    """Does the dip coincide with structure in another channel?

    Nuisance channels (background, centroids): the in-event mean minus the mean of the surrounding
    stretch, compared with the same difference measured at n_null random placements of the same window
    elsewhere in the light curve. This placement null carries the channel's own red noise and trends, which
    a white-noise z-score does not (the first pilot vetoed half of all injected dips on centroid "shifts"
    that were ordinary pointing drift). A dip with an unusual background jump or centroid shift is likely
    systematic or from a neighbour.

    The same placement statistic on the flux itself is reported as `placement_z`: how unusual this dip is
    relative to every other stretch of the same light curve, with no noise model at all.

    Alternative photometry: ratio of fractional depth (alt / primary). A true on-target dip has the same
    fractional depth in every aperture; contamination does not.
    """
    out = {}
    t = lc.t
    dur = max(e.t_end - e.t_start, np.median(np.diff(t)) * 3 if len(t) > 3 else 0.0)
    pad = max(3 * dur, 1.0)
    inside = (t >= e.t_start) & (t <= e.t_end)
    around = (t >= e.t_start - pad) & (t <= e.t_end + pad) & ~inside
    if inside.sum() < 2 or around.sum() < 5:
        return out
    rng = np.random.default_rng(seed)
    lo, hi = t[0] + pad + dur / 2, t[-1] - pad - dur / 2
    centres = rng.uniform(lo, hi, 4 * n_null) if hi > lo else np.array([])
    mid = 0.5 * (e.t_start + e.t_end)
    centres = centres[np.abs(centres - mid) > dur + pad][:n_null]

    def placement_z(x):
        ok = np.isfinite(x)
        tt, xx = t[ok], x[ok]
        if len(tt) < 10:
            return np.nan
        delta = _window_delta(tt, xx, np.array([mid]), dur / 2, pad)[0]
        null = _window_delta(tt, xx, centres, dur / 2, pad)
        null = null[np.isfinite(null)]
        s = robust_std(null)
        if len(null) < 16 or not np.isfinite(s) or s == 0 or not np.isfinite(delta):
            return np.nan
        return float((delta - np.median(null)) / s)

    out["placement_z"] = placement_z(lc.flux)
    base = np.median(lc.flux[around])
    depth = base - np.median(lc.flux[inside])
    nuisance = {}
    for name, series in lc.aux.items():
        x = np.asarray(series, float)
        if name.startswith(ALT_FLUX_PREFIX):
            if e.t_end - e.t_start > ALT_FLUX_MAX_DURATION:
                continue
            ab = np.nanmedian(x[around])
            alt_depth = (ab - np.nanmedian(x[inside])) / ab if ab else np.nan
            out[f"depth_ratio_{name[len(ALT_FLUX_PREFIX):]}"] = float(alt_depth / (depth / base)) if depth else np.nan
        else:
            out[f"z_{name}"] = placement_z(x)
            nuisance[name] = x
    if nuisance:
        out.update(_explained(lc.flux, nuisance, inside, around, depth))
    return out


# Alternative-aperture fluxes may be detrended by their pipeline (QLP's spline breaks at 0.3-2.5 d), so
# their depth ratio is only reported for events well shorter than the shortest break. At 0.3 d the spline
# already ate part of real dips (known YSO dippers failed the ratio test), which injections multiplied in
# after detrending could not show.
ALT_FLUX_MAX_DURATION = 0.1


def _explained(flux, channels, inside, around, depth):
    """Fraction of the dip that the nuisance channels can explain through their coupling to flux.

    Fit flux = a + sum_k b_k x_k on the surrounding stretch only (the dip excluded), then predict the
    in-event flux change from the channels' in-event change. A scattered-light dip is mostly explained by
    the background term; a pointing excursion by the centroid terms; an occultation by nothing. This is a
    physical question ("could the systematics have done this?") rather than a significance cut, and an
    injected dip, which has no coupling by construction, should score near zero.
    """
    names = [k for k, x in channels.items() if np.isfinite(x[around]).sum() > 10 and np.isfinite(x[inside]).all()]
    if not names or depth == 0:
        return {}
    X = np.column_stack([channels[k] for k in names])
    Xa, ya = X[around], flux[around]
    ok = np.isfinite(Xa).all(axis=1) & np.isfinite(ya)
    Xa, ya = Xa[ok], ya[ok]
    mu, sd = Xa.mean(axis=0), Xa.std(axis=0)
    sd[sd == 0] = 1.0
    A = np.column_stack([np.ones(len(Xa)), (Xa - mu) / sd])
    coef, *_ = np.linalg.lstsq(A, ya, rcond=None)
    dx = ((X[inside] - mu) / sd).mean(axis=0) - ((Xa - mu) / sd).mean(axis=0)
    parts = coef[1:] * dx
    out = {f"explained_{k}": float(-p / depth) for k, p in zip(names, parts)}
    out["explained_total"] = float(-parts.sum() / depth)
    return out


def _window_delta(t, x, centres, half, pad):
    """mean(x within +-half of each centre) - mean(x in the flanking pad on both sides)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    def window_sum(a, b):
        i, j = np.searchsorted(t, a, side="left"), np.searchsorted(t, b, side="right")
        return c[j] - c[i], j - i
    s_in, n_in = window_sum(centres - half, centres + half)
    s_all, n_all = window_sum(centres - half - pad, centres + half + pad)
    s_out, n_out = s_all - s_in, n_all - n_in
    with np.errstate(invalid="ignore", divide="ignore"):
        d = s_in / n_in - s_out / n_out
    d[(n_in < 2) | (n_out < 5)] = np.nan
    return d
