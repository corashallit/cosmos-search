"""Slow period search: dimming that repeats, from about 20 days to two-thirds of the light curve's span.

The detector in `detect` finds the one strongest run of consecutive faint visits. A change that repeats (a body or
swarm on an orbit of months to years, a slow cycle) spreads its faint visits over several cycles, so no single run
in time stands out and the median sits partway down. Folding at the right period gathers them into one run in
phase. The score is the episode score of `detect` computed on the folded light curve:

    s_rep = max over trial periods and phase windows of (median - window mean) * sqrt(k - 1) / sigma_visit

over windows of k >= MIN_EPISODE_VISITS visits contiguous in phase, at most MAX_EPISODE_FRACTION of the visits,
with the window's deepest visit left out so that no single visit can make it. Visits are first detrended with
their own straight line (the trend is `detect`'s slope_z). `s_rep_bright` is the same for repeated brightening and
gives the false-alarm estimate, as `s_bright` does for `s_dim`.

A window counts only if its visits come from at least two cycles. Even so, one long dip folded at a period shorter
than itself also fills a window from two cycles, so `s_rep` takes the best of the top periods whose window visits
form at least two groups in time, whose visits in between are not fainter than BETWEEN_Z sigma, and whose window
without its strongest event is still REST_Z sigma faint (`events`): a repeating change returns to the baseline
between events and shows more than once, one long episode does not. `s_rep_any` is the best window without
that condition. Periods start at P_MIN_DAYS because Gaia's visits come weeks apart and a
faster change folds onto aliases; changes that fast are the domain of the eclipsing-binary filter (`ebfilter`).
"""

import numpy as np
import pandas as pd

from cosmos_search.slow import detect

P_MIN_DAYS = 20.0
MIN_CYCLES = 1.5          # longest trial period: span / MIN_CYCLES
PHASE_DRIFT = 0.02        # frequency step: a window drifts by at most this phase across the span
TOP_PERIODS = 20          # best periods checked for separate events
BETWEEN_Z = 3.0           # visits between events may not be fainter than this (in sigma of their mean)
REST_Z = 3.0              # the window without its strongest event must still be this faint
FAMILIES = ("repeat_long_dip", "slow_cycle")  # what calibration injects (inject.lightcurve.PERIODIC_FAMILIES)


def freq_grid(span, p_min=P_MIN_DAYS, min_cycles=MIN_CYCLES, drift=PHASE_DRIFT):
    f_lo, f_hi = min_cycles / span, 1.0 / p_min
    if f_lo >= f_hi:
        return np.array([])
    return np.arange(f_lo, f_hi, drift / span)


def _window_extremes(v, kmax):
    """For a (m, 2n) array (folded values repeated once, for windows that wrap), running sums, minima and maxima
    of the windows starting at each of the first n columns for lengths 1..kmax: three (kmax, m, n) arrays."""
    m, n2 = v.shape
    n = n2 // 2
    s = np.empty((kmax, m, n))
    lo = np.empty((kmax, m, n))
    hi = np.empty((kmax, m, n))
    s[0], lo[0], hi[0] = v[:, :n], v[:, :n], v[:, :n]
    for k in range(1, kmax):
        x = v[:, k:k + n]
        s[k] = s[k - 1] + x
        lo[k] = np.minimum(lo[k - 1], x)
        hi[k] = np.maximum(hi[k - 1], x)
    return s, lo, hi


def scan(tv, yv, sigma, freqs, chunk=1000):
    """Best dimming and brightening window per trial frequency.

    Returns a dict of arrays over freqs: dim, bright (scores), and for the dimming window its start (index into the
    folded order), length and phase order, so the window's visits can be recovered."""
    n = len(yv)
    kmax = int(np.floor(detect.MAX_EPISODE_FRACTION * n))
    if kmax < detect.MIN_EPISODE_VISITS or not len(freqs):
        return None
    parts = [_scan(tv, yv - np.median(yv), sigma, freqs[i:i + chunk], kmax) for i in range(0, len(freqs), chunk)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _scan(tv, dev, sigma, freqs, kmax):
    n = len(dev)
    ph = np.outer(freqs, tv) % 1.0
    order = np.argsort(ph, axis=1)
    v = np.take(dev, order)
    cyc = np.floor(np.take(tv, order) * freqs[:, None])  # cycle number of each folded visit
    v2 = np.concatenate([v, v], axis=1)
    # A window that wraps past phase 1 continues the same event: counted from the window's start, a visit after
    # the wrap belongs to the cycle before its own number.
    c2 = np.concatenate([cyc, cyc - 1], axis=1)
    s, lo, hi = _window_extremes(v2, kmax)
    _, clo, chi = _window_extremes(c2, kmax)
    k = np.arange(1, kmax + 1)[:, None, None]
    valid = (k >= detect.MIN_EPISODE_VISITS) & (chi > clo)  # visits from at least two cycles
    km1 = np.maximum(k - 1, 1)
    norm = np.sqrt(km1) * sigma
    dim = np.where(valid, -(s - lo) / norm, -np.inf)
    bright = np.where(valid, (s - hi) / norm, -np.inf)
    flat = dim.reshape(kmax, len(freqs), n).transpose(1, 0, 2).reshape(len(freqs), -1)
    best = np.argmax(flat, axis=1)
    return {"dim": flat[np.arange(len(freqs)), best], "length": best // n + 1, "start": best % n, "order": order,
            "bright": bright.max(axis=(0, 2))}


def events(order_row, start, length, tv, dev, sigma):
    """For one window, in time order: the number of events (groups of window visits separated by visits outside
    the window); z_between, how faint the outside visits between the first and last event are, (mean deviation) *
    sqrt(n) / sigma; and z_rest, the deficit of the window's visits once the strongest event is taken out, -(mean
    deviation) * sqrt(n) / sigma. A repeating dimming returns to the baseline between events (z_between near 0 or
    above) and still shows with its strongest event removed (z_rest well above 0). One long episode fails one or
    the other: folded at a shorter period, the visits between are inside the episode; folded at a longer one, the
    window's other "events" are baseline visits that happen to share its phase."""
    n = len(tv)
    inside = np.zeros(n, bool)
    inside[order_row[(start + np.arange(length)) % n]] = True
    by_time = np.argsort(tv)
    ins = inside[by_time]
    grp = np.cumsum(ins & ~np.r_[False, ins[:-1]])
    n_ev = int(grp[-1])
    first, last = np.flatnonzero(ins)[[0, -1]]
    between = by_time[first:last + 1][~ins[first:last + 1]]
    z_between = float(dev[between].mean() * np.sqrt(len(between)) / sigma) if len(between) else np.nan
    d = dev[by_time][ins]
    g = grp[ins]
    sums = np.bincount(g, d)[1:]
    rest = g != (np.argmin(sums) + 1)
    z_rest = float(-d[rest].sum() / (sigma * np.sqrt(rest.sum()))) if rest.any() else np.nan
    return n_ev, z_between, z_rest


def _detrend(tv, yv):
    x = tv - tv.mean()
    sxx = np.sum(x * x)
    return yv - yv.mean() - (np.sum(x * (yv - yv.mean())) / sxx * x if sxx > 0 else 0.0)


def star_scores(tv, yv, sigma, top=TOP_PERIODS):
    """s_rep (best window among the top periods that passes the repeat conditions of `events`), its period, depth,
    length and events, and
    s_rep_bright (best repeated brightening, no event condition: the false-alarm estimate is deliberately
    conservative)."""
    out = {"s_rep": np.nan, "rep_period": np.nan, "rep_depth": np.nan, "rep_visits": 0, "rep_events": 0,
           "rep_between_z": np.nan, "rep_rest_z": np.nan,
           "s_rep_any": np.nan, "s_rep_bright": np.nan}
    span = tv[-1] - tv[0]
    freqs = freq_grid(span)
    y = _detrend(tv, yv)
    r = scan(tv, y, sigma, freqs)
    if r is None:
        return out
    out["s_rep_any"], out["s_rep_bright"] = float(r["dim"].max()), float(r["bright"].max())
    dev = y - np.median(y)
    for i in np.argsort(r["dim"])[::-1][:top]:
        if not np.isfinite(r["dim"][i]):
            break
        ne, z, zr = events(r["order"][i], r["start"][i], r["length"][i], tv, dev, sigma)
        if ne >= 2 and z > -BETWEEN_Z and zr >= REST_Z:
            idx = r["order"][i][(r["start"][i] + np.arange(r["length"][i])) % len(tv)]
            out.update(s_rep=float(r["dim"][i]), rep_period=float(1 / freqs[i]), rep_visits=int(r["length"][i]),
                       rep_events=ne, rep_between_z=z, rep_rest_z=zr, rep_depth=float(-dev[idx].mean()))
            break
    return out


def star_features(epochs, col="frac_corr"):
    """One row per star with the period-search scores (G visits, as `detect.star_features`)."""
    e = epochs[epochs["band"] == 0].sort_values(["source_id", "t"], kind="stable")
    sid, t, y = e["source_id"].to_numpy(), e["t"].to_numpy(), e[col].to_numpy()
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    ends = np.r_[starts[1:], len(sid)]
    rows = []
    for a, b in zip(starts, ends):
        if b - a < detect.MIN_VISITS:
            continue
        tv, yv = detect.visits(t[a:b], y[a:b])
        if len(tv) < detect.MIN_VISITS:
            continue
        rows.append({"source_id": int(sid[a]), **star_scores(tv, yv, detect.visit_sigma(yv))})
    return pd.DataFrame(rows)
