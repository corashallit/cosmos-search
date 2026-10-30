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

import math

import numpy as np
import pandas as pd

from cosmos_search.slow import detect

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba ships with lsdb/hats; the numpy scan below still works without it
    njit = None

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


def scan(tv, yv, sigma, freqs, chunk=1000, compiled=True):
    """Best dimming and brightening window per trial frequency.

    Returns a dict of arrays over freqs: dim, bright (scores), and for the dimming window its start (index into the
    folded order), length and phase order, so the window's visits can be recovered.

    compiled=True runs the window search in one compiled loop (`_scan_compiled`), which gives the same numbers as
    the numpy version (`_scan`) without its (window length x frequency x visit) arrays: at DR4's ~30-40 visits over
    5.5 years the numpy version takes ~0.2-0.3 s a star, too slow for millions of selected stars."""
    n = len(yv)
    kmax = int(np.floor(detect.MAX_EPISODE_FRACTION * n))
    if kmax < detect.MIN_EPISODE_VISITS or not len(freqs):
        return None
    if compiled and njit is not None:
        return _scan_compiled(tv, yv - np.median(yv), sigma, freqs, kmax)
    parts = [_scan(tv, yv - np.median(yv), sigma, freqs[i:i + chunk], kmax) for i in range(0, len(freqs), chunk)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _scan_compiled(tv, dev, sigma, freqs, kmax):
    ph = np.outer(freqs, tv) % 1.0
    order = np.argsort(ph, axis=1)
    v = np.ascontiguousarray(np.take(dev, order))
    cyc = np.ascontiguousarray(np.floor(np.take(tv, order) * freqs[:, None]))
    dim, best, bright, bbest = _windows(v, cyc, float(sigma), kmax, detect.MIN_EPISODE_VISITS)
    n = len(dev)
    return {"dim": dim, "length": best // n + 1, "start": best % n, "order": order,
            "bright": bright, "b_length": bbest // n + 1, "b_start": bbest % n}


def _windows_py(v, cyc, sigma, kmax, kmin):
    """Per row of folded values v (and their cycle numbers cyc): the best dimming and brightening window over
    lengths kmin..kmax and every start, wrapping past phase 1, as `_scan` scores them. The index of a window is
    (length - 1) * n + start and ties go to the lowest index, as numpy's argmax over `_scan`'s flattened array."""
    m, n = v.shape
    dim = np.full(m, -np.inf)
    bright = np.full(m, -np.inf)
    best = np.zeros(m, np.int64)
    bbest = np.zeros(m, np.int64)
    norms = np.empty(kmax + 1)
    for k in range(1, kmax + 1):
        norms[k] = math.sqrt(max(k - 1, 1)) * sigma
    for f in range(m):
        bd, bdi, bb, bbi = -np.inf, 0, -np.inf, 0
        for s in range(n):
            acc, lo, hi, clo, chi = 0.0, 0.0, 0.0, 0.0, 0.0
            for k in range(kmax):
                j = s + k
                if j < n:
                    x, c = v[f, j], cyc[f, j]
                else:  # after the wrap the visit belongs to the cycle before its own number
                    x, c = v[f, j - n], cyc[f, j - n] - 1.0
                if k == 0:
                    acc, lo, hi, clo, chi = x, x, x, c, c
                else:
                    acc = acc + x
                    lo, hi = min(lo, x), max(hi, x)
                    clo, chi = min(clo, c), max(chi, c)
                if k + 1 < kmin or not chi > clo:
                    continue
                idx = k * n + s
                d = -(acc - lo) / norms[k + 1]
                if d > bd or (d == bd and idx < bdi):
                    bd, bdi = d, idx
                b = (acc - hi) / norms[k + 1]
                if b > bb or (b == bb and idx < bbi):
                    bb, bbi = b, idx
        dim[f], best[f], bright[f], bbest[f] = bd, bdi, bb, bbi
    return dim, best, bright, bbest


_windows = njit(cache=True)(_windows_py) if njit is not None else _windows_py


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
    bflat = bright.reshape(kmax, len(freqs), n).transpose(1, 0, 2).reshape(len(freqs), -1)
    bbest = np.argmax(bflat, axis=1)
    rows = np.arange(len(freqs))
    return {"dim": flat[rows, best], "length": best // n + 1, "start": best % n, "order": order,
            "bright": bflat[rows, bbest], "b_length": bbest // n + 1, "b_start": bbest % n}


def events(order_row, start, length, tv, dev, sigma, compiled=True):
    """For one window, in time order: the number of events (groups of window visits separated by visits outside
    the window); z_between, how faint the outside visits between the first and last event are, (mean deviation) *
    sqrt(n) / sigma; and z_rest, the deficit of the window's visits once the strongest event is taken out, -(mean
    deviation) * sqrt(n) / sigma. A repeating dimming returns to the baseline between events (z_between near 0 or
    above) and still shows with its strongest event removed (z_rest well above 0). One long episode fails one or
    the other: folded at a shorter period, the visits between are inside the episode; folded at a longer one, the
    window's other "events" are baseline visits that happen to share its phase.

    compiled=True gives the same numbers from one compiled function (`_events_compiled`, numpy's summation order
    included); the period search calls this up to TOP_PERIODS times a star."""
    if compiled and njit is not None:
        return _events_compiled(order_row, int(start), int(length), tv, dev, float(sigma))
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


def _pairwise_sum(a):
    """Sum as numpy's add.reduce computes it for a contiguous float64 array (pairwise, blocks of 8), so compiled
    code reproduces numpy's results to the last bit."""
    n = len(a)
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[i]
        return res
    if n <= 128:
        r0, r1, r2, r3, r4, r5, r6, r7 = a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
        i = 8
        while i < n - n % 8:
            r0 += a[i]
            r1 += a[i + 1]
            r2 += a[i + 2]
            r3 += a[i + 3]
            r4 += a[i + 4]
            r5 += a[i + 5]
            r6 += a[i + 6]
            r7 += a[i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _pairwise_sum(a[:n2]) + _pairwise_sum(a[n2:])


def _events_py(order_row, start, length, tv, dev, sigma):
    """`events` written for compilation: the same steps, in the same order of floating-point operations."""
    n = len(tv)
    inside = np.zeros(n, np.bool_)
    for k in range(length):
        inside[order_row[(start + k) % n]] = True
    by_time = np.argsort(tv)
    ins = inside[by_time]
    grp = np.zeros(n, np.int64)
    g = 0
    first, last = -1, -1
    for k in range(n):
        if ins[k]:
            if k == 0 or not ins[k - 1]:
                g += 1
            if first < 0:
                first = k
            last = k
        grp[k] = g
    n_ev = g
    between = np.empty(n, np.float64)
    nb = 0
    for k in range(first, last + 1):
        if not ins[k]:
            between[nb] = dev[by_time[k]]
            nb += 1
    z_between = _pairwise_sum(between[:nb]) / nb * math.sqrt(nb) / sigma if nb else np.nan
    sums = np.zeros(n_ev + 1)
    for k in range(n):
        if ins[k]:
            sums[grp[k]] += dev[by_time[k]]
    strongest = int(np.argmin(sums[1:])) + 1
    rest = np.empty(n, np.float64)
    nr = 0
    for k in range(n):
        if ins[k] and grp[k] != strongest:
            rest[nr] = dev[by_time[k]]
            nr += 1
    z_rest = -_pairwise_sum(rest[:nr]) / (sigma * math.sqrt(nr)) if nr else np.nan
    return n_ev, z_between, z_rest


if njit is not None:
    _pairwise_sum = njit(cache=True)(_pairwise_sum)
    _events_compiled = njit(cache=True)(_events_py)


def _window(order_row, start, length):
    return order_row[(start + np.arange(length)) % len(order_row)]


def band_z(t_window, bands, sign=1.0):
    """Deficit (sign=1) or excess (sign=-1) of the BP and RP visits at the window's G visit times, against each
    band's median, in units of its error: (z_bp + z_rp) / sqrt(2), or the one band available. The window is fixed
    by G, so this is a second, independent measurement of the same event with no search of its own."""
    zs = []
    for tb, yb, sb in bands:
        j = np.clip(np.searchsorted(tb, t_window), 1, len(tb) - 1)
        j = np.where(np.abs(tb[j - 1] - t_window) < np.abs(tb[j] - t_window), j - 1, j)
        j = np.unique(j[np.abs(tb[j] - t_window) < detect.VISIT_GAP_DAYS])
        if len(j):
            zs.append(sign * (np.median(yb) - yb[j].mean()) * np.sqrt(len(j)) / sb)
    return float(np.sum(zs) / np.sqrt(len(zs))) if zs else np.nan


def _detrend(tv, yv):
    x = tv - tv.mean()
    sxx = np.sum(x * x)
    return yv - yv.mean() - (np.sum(x * (yv - yv.mean())) / sxx * x if sxx > 0 else 0.0)


def star_scores(tv, yv, sigma, top=TOP_PERIODS, bands=()):
    """s_rep (best window among the top periods that passes the repeat conditions of `events`), its period, depth,
    length and events, and s_rep_bright (best repeated brightening, no event condition: the false-alarm estimate is
    deliberately conservative). rep_band_z and rep_bright_band_z: the BP and RP visits at the times of the chosen
    windows (`band_z`), reported for checking a candidate; no rule uses them (see the doc, "Colour agreement")."""
    out = {"s_rep": np.nan, "rep_period": np.nan, "rep_depth": np.nan, "rep_visits": 0, "rep_events": 0,
           "rep_between_z": np.nan, "rep_rest_z": np.nan,
           "s_rep_any": np.nan, "s_rep_bright": np.nan, "rep_band_z": np.nan, "rep_bright_band_z": np.nan}
    span = tv[-1] - tv[0]
    freqs = freq_grid(span)
    y = _detrend(tv, yv)
    r = scan(tv, y, sigma, freqs)
    if r is None:
        return out
    out["s_rep_any"], out["s_rep_bright"] = float(r["dim"].max()), float(r["bright"].max())
    dev = y - np.median(y)
    bands = [(tb, _detrend(tb, yb), sb) for tb, yb, sb in bands]
    if bands:
        i = int(np.argmax(r["bright"]))
        out["rep_bright_band_z"] = band_z(tv[_window(r["order"][i], r["b_start"][i], r["b_length"][i])], bands, -1.0)
    for i in np.argsort(r["dim"])[::-1][:top]:
        if not np.isfinite(r["dim"][i]):
            break
        ne, z, zr = events(r["order"][i], r["start"][i], r["length"][i], tv, dev, sigma)
        if ne >= 2 and z > -BETWEEN_Z and zr >= REST_Z:
            idx = _window(r["order"][i], r["start"][i], r["length"][i])
            out.update(s_rep=float(r["dim"][i]), rep_period=float(1 / freqs[i]), rep_visits=int(r["length"][i]),
                       rep_events=ne, rep_between_z=z, rep_rest_z=zr, rep_depth=float(-dev[idx].mean()))
            if bands:
                out["rep_band_z"] = band_z(tv[idx], bands)
            break
    return out


def star_features(epochs, col="frac_corr"):
    """One row per star with the period-search scores (G visits, as `detect.star_features`)."""
    e = epochs.sort_values(["source_id", "band", "t"], kind="stable")
    sid, band, t, y = e["source_id"].to_numpy(), e["band"].to_numpy(), e["t"].to_numpy(), e[col].to_numpy()
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    ends = np.r_[starts[1:], len(sid)]
    rows = []
    for a, b in zip(starts, ends):
        per = {}
        for code in (0, 1, 2):
            m = np.flatnonzero(band[a:b] == code) + a
            if len(m) < detect.MIN_VISITS:
                continue
            tv, yv = detect.visits(t[m], y[m])
            if len(tv) >= detect.MIN_VISITS:
                per[code] = (tv, yv, detect.visit_sigma(yv))
        if 0 not in per:
            continue
        tv, yv, sg = per[0]
        rows.append({"source_id": int(sid[a]),
                     **star_scores(tv, yv, sg, bands=[per[c] for c in (1, 2) if c in per])})
    return pd.DataFrame(rows)
