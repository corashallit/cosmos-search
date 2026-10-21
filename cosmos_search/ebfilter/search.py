"""Is a star's dip pattern a strict-period eclipse? A compression test over every epoch, in all surveys.

The question an eclipsing-binary filter has to answer is whether one strict period explains *which* epochs are in
dip. That is a question about information, so it is posed as one (minimum description length). Label every epoch
dip, flat or neither, and describe the sequence of dip and flat labels two ways:

- **H0, dips placed anywhere.** Send the number of dip epochs K and of runs E, then which of the C(N-K+1, E) C(K-1,
  E-1) sequences with those counts it is. This is the uniform code over every arrangement of E dip runs among N
  epochs: it makes no use of time or phase.
- **H_EB, eclipses.** Send a period, a phase window (start and width on a grid) and optionally a second window
  (the secondary eclipse), then which bands see the eclipse. Every epoch of a participating band inside a window
  is predicted to be in dip, every other epoch is coded as under H0. The cost is the period grid, the window grids,
  the in-window flats (exceptions) and the out-of-window dips.

The test statistic is the saving, delta = L_H0 - L_EB, in bits. Because the H_EB code is a valid prefix code over
all label sequences (its parameters come from fixed grids and are paid for), the no-hypercompression inequality
gives P(delta >= k | H0) <= 2^-k for any data whose dip runs are placed uniformly at random among the epochs. That
bound uses the star's own epochs, so the observing cadence, its aliases (ZTF's 1-day, Gaia's scanning law) and
the number of trial periods are all inside it: a period found at a 1-day alias of nightly sampling buys nothing
unless it also keeps the flat epochs out of its window. A search that misses the best period only lowers delta,
so the bound stays valid.

What the statistic does not use: dip depths (aperiodic occulters can have constant depths too), shapes, or
out-of-eclipse variation. Those are for the decision layer (`ebfilter.decide`) and for vetting.

Labels, per band, on the epochs given (BP and RP are skipped: they share G's transits): z = (1 - f/median) /
robust sigma. Dip: z >= max(DIP_SIGMA, the band's second-largest brightening), a mirror test so that heavy-tailed
noise does not label dips. Flat: z < FLAT_SIGMA. Epochs in between (ingress, egress, shallow eclipses) are left out
of both codes.
"""

import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba ships with lsdb/hats, but keep the module importable without it
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]

DIP_SIGMA = 4.0
FLAT_SIGMA = 2.0
SKIP_BANDS = ("BP", "RP")
MIN_EPOCHS_BAND = 10
P_MIN = 0.2                     # days
WIDTHS = (0.04, 0.1, 0.25, 0.6, 1.5, 4.0)   # trial eclipse durations, days
MAX_PHASE_WIDTH = 0.25          # an eclipse window is at most a quarter of the orbit
PHASE_STEPS = 4                 # window starts on a grid of q / PHASE_STEPS; the period grid drifts by at most that
SWEEP_EVENTS = 64               # deepest events used in the coarse sweep
SWEEP_GAP = 1.0                 # days; dip epochs closer than this (and than the width) are one event in the sweep
TOP_M = 300                     # candidates per width from the sweep that get an exact code length
TOP_SECONDARY = 25              # best primaries that are tried with a second window
LN2 = math.log(2.0)


# -- labels ------------------------------------------------------------------------------------------------------

def _robust_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def labels(lcs, skip=SKIP_BANDS):
    """Time-sorted (t, lab, band_index, z, band_names, frac) over usable bands; lab is 1 dip, 0 flat, -1 neither,
    frac = 1 - flux / median. lcs: objects with band, time, flux (vetting.candidate.LightCurve)."""
    ts, labs, bis, zs, names, fr = [], [], [], [], [], []
    for lc in lcs:
        if lc.band in skip:
            continue
        t, f = np.asarray(lc.time, float), np.asarray(lc.flux, float)
        ok = np.isfinite(t) & np.isfinite(f)
        t, f = t[ok], f[ok]
        if len(t) < MIN_EPOCHS_BAND:
            continue
        med = np.median(f)
        s = _robust_sigma(f / med)
        if not s > 0:
            continue
        z = (1 - f / med) / s
        bright = np.sort(-z)
        k = max(DIP_SIGMA, float(bright[-2]) if len(bright) > 1 else DIP_SIGMA)
        lab = np.full(len(t), -1, np.int8)
        lab[z >= k] = 1
        lab[z < FLAT_SIGMA] = 0
        if lc.band in names:
            bi = names.index(lc.band)
        else:
            names.append(lc.band)
            bi = len(names) - 1
        ts.append(t)
        labs.append(lab)
        bis.append(np.full(len(t), bi, np.int8))
        zs.append(z)
        fr.append(1 - f / med)
    if not ts:
        e = np.array([])
        return e, e.astype(np.int8), e.astype(np.int8), e, names, e
    t, lab, bi, z, frac = (np.concatenate(x) for x in (ts, labs, bis, zs, fr))
    o = np.argsort(t, kind="stable")
    return t[o], lab[o], bi[o], z[o], names, frac[o]


# -- code lengths (bits) -----------------------------------------------------------------------------------------

@njit(cache=True)
def _lchoose(n, k):
    if k < 0 or k > n:
        return 1e9
    return (math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)) / LN2


@njit(cache=True)
def _gamma_bits(n):
    """Elias gamma code length for an integer n >= 0 (coded as n + 1): cheap for the small counts exceptions have."""
    return 2.0 * math.floor(math.log2(n + 1.0)) + 1.0


@njit(cache=True)
def _runs_code(n, k, e):
    """Uniform code for a binary sequence of length n (known) with k ones in e runs: counts, then which one."""
    if k == 0:
        return _gamma_bits(0)
    return _gamma_bits(k) + math.log2(k) + _lchoose(n - k + 1, e) + _lchoose(k - 1, e - 1)


def null_bits(lab):
    """L_H0 for a label sequence (lab 1/0 only, time-ordered)."""
    lab = np.asarray(lab)
    k = int(lab.sum())
    e = int(np.sum(np.diff(np.r_[0, lab]) == 1))
    return float(_runs_code(len(lab), k, e)) if len(lab) else 0.0


# -- the period grid ---------------------------------------------------------------------------------------------

def freq_grid(span, w, p_min=P_MIN, max_phase=MAX_PHASE_WIDTH, steps=PHASE_STEPS):
    """Log-spaced frequencies (1/d) from 1/span to min(1/p_min, max_phase/w). Adjacent frequencies drift apart by
    w * f / steps in phase over the span, the same step as the window-start grid."""
    f_min, f_max = 1.0 / span, min(1.0 / p_min, max_phase / w)
    if f_max <= f_min:
        return np.empty(0)
    n = int(np.ceil(np.log(f_max / f_min) * steps * span / w)) + 1
    return f_min * np.exp(np.arange(n) * (w / (steps * span)))


def n_freq(span, w, **kw):
    f_min, f_max = 1.0 / span, min(1.0 / kw.get("p_min", P_MIN), kw.get("max_phase", MAX_PHASE_WIDTH) / w)
    return 0 if f_max <= f_min else int(np.ceil(np.log(f_max / f_min) * PHASE_STEPS * span / w)) + 1


# -- coarse sweep ------------------------------------------------------------------------------------------------

@njit(cache=True)
def _sweep(te, freqs, w, steps, out_key, out_start):
    """Per frequency: the window of width q = w f, start on a q/steps grid, holding the most events; key ~ bits a
    window with c events in phase width q would save over chance (c log2(1/q)), and that window's start."""
    n = len(te)
    ph = np.empty(n)
    ext = np.empty(2 * n)
    for i in range(len(freqs)):
        f = freqs[i]
        q = w * f
        d = q / steps
        for j in range(n):
            x = te[j] * f
            ph[j] = x - math.floor(x)
        ph.sort()
        for j in range(n):
            ext[j] = ph[j]
            ext[n + j] = ph[j] + 1.0
        best, bs = 0, 0.0
        hi = 0
        for j in range(n):
            s = math.floor(ph[j] / d) * d
            lo = j
            while lo > 0 and ext[lo - 1] >= s:
                lo -= 1
            if hi < j:
                hi = j
            while hi < 2 * n and ext[hi] <= s + q:
                hi += 1
            c = hi - lo
            if c > n:
                c = n
            if c > best:
                best, bs = c, s
        out_key[i] = best * math.log2(1.0 / q) if best > 1 else 0.0
        out_start[i] = bs


def _sweep_events(t, lab, w):
    """Representative times (deepest is not known here, so the middle epoch) of runs of dip epochs closer than
    min(SWEEP_GAP, w), capped at SWEEP_EVENTS by keeping the runs with the most dip epochs."""
    td = t[lab == 1]
    if len(td) == 0:
        return td
    gap = min(SWEEP_GAP, w)
    br = np.flatnonzero(np.diff(td) > gap) + 1
    starts, ends = np.r_[0, br], np.r_[br, len(td)]
    rep = np.array([td[(a + b - 1) // 2] for a, b in zip(starts, ends)])
    if len(rep) > SWEEP_EVENTS:
        size = ends - starts
        keep = np.sort(np.argsort(-size, kind="stable")[:SWEEP_EVENTS])
        rep = rep[keep]
    return rep


# -- exact code length of one model ------------------------------------------------------------------------------

@njit(cache=True)
def _model_bits(t, lab, bi, n_bands, f, s1, q1, s2, q2):
    """Code length (bits) of the labels given windows [s1, s1+q1] and optionally [s2, s2+q2] (q2 <= 0: none),
    excluding the parameter cost. Returns (bits, n_in, n_flat_in, k_out, e_out, n_dip_in)."""
    n = len(t)
    inwin = np.zeros(n, np.bool_)
    part = np.zeros(n_bands, np.bool_)
    for i in range(n):
        x = t[i] * f
        p = x - math.floor(x)
        y = p - s1
        if y < 0:
            y += 1.0
        a = y <= q1
        if not a and q2 > 0:
            y = p - s2
            if y < 0:
                y += 1.0
            a = y <= q2
        inwin[i] = a
        if a and lab[i] == 1:
            part[bi[i]] = True
    n_in, n_flat_in, n_dip_in = 0, 0, 0
    n_out, k_out, e_out = 0, 0, 0
    prev = 0
    for i in range(n):
        if inwin[i] and part[bi[i]]:
            n_in += 1
            if lab[i] == 0:
                n_flat_in += 1
            else:
                n_dip_in += 1
        else:
            n_out += 1
            if lab[i] == 1:
                k_out += 1
                if prev == 0:
                    e_out += 1
            prev = lab[i]
    bits = _gamma_bits(n_flat_in) + _lchoose(n_in, n_flat_in) + _runs_code(n_out, k_out, e_out)
    return bits, n_in, n_flat_in, k_out, e_out, n_dip_in


@njit(cache=True)
def _best_secondary(t, lab, bi, n_bands, f, s1, q1, widths, steps):
    """Best second window at frequency f (start on its own q/steps grid), by exact code length including the cost
    of its start. Each grid cell holding an out-of-window dip is tried once."""
    n = len(t)
    best_bits, best_s, best_q = 1e18, 0.0, 0.0
    for wi in range(len(widths)):
        q = widths[wi] * f
        if q > 0.25:
            continue
        d = q / steps
        cells = np.empty(n, np.int64)
        nc = 0
        for i in range(n):
            if lab[i] != 1:
                continue
            x = t[i] * f
            p = x - math.floor(x)
            y = p - s1
            if y < 0:
                y += 1.0
            if y <= q1:
                continue
            cells[nc] = int(math.floor(p / d))
            nc += 1
        cs = np.unique(cells[:nc])
        for c in cs:
            s = c * d
            b = _model_bits(t, lab, bi, n_bands, f, s1, q1, s, q)[0] + math.log2(math.ceil(steps / q))
            if b < best_bits:
                best_bits, best_s, best_q = b, s, q
    return best_bits, best_s, best_q


# -- the search --------------------------------------------------------------------------------------------------

def search(t, lab, bi, names, widths=WIDTHS, p_min=P_MIN, top_m=TOP_M, top_secondary=TOP_SECONDARY, n_keep=10):
    """Compression test on labels from `labels`. Returns a dict with delta_bits (L_H0 - L_EB, best model), the
    model (period, windows, bands), the counts it explains, and `solutions`: the n_keep best distinct periods."""
    keep = lab >= 0
    t, lab, bi = t[keep], lab[keep].astype(np.int8), bi[keep].astype(np.int8)
    n, k = len(t), int(lab.sum())
    out = {"n_epochs": int(n), "n_dip_epochs": k, "n_bands": len(names), "null_bits": null_bits(lab),
           "delta_bits": 0.0, "period": np.nan, "solutions": []}
    if k < 2 or n < 10:
        return out
    span = float(t[-1] - t[0])
    n_bands = len(names)
    l0 = out["null_bits"]
    w_arr = np.asarray(widths, float)
    param_fixed = math.log2(len(widths)) + 1.0 + n_bands      # width index, secondary flag, band participation
    cands = []
    for w in widths:
        freqs = freq_grid(span, w, p_min)
        if not len(freqs):
            continue
        te = _sweep_events(t, lab, w)
        if len(te) < 2:
            continue
        key, start = np.zeros(len(freqs)), np.zeros(len(freqs))
        _sweep(te, freqs, w, PHASE_STEPS, key, start)
        m = min(top_m, len(freqs))
        idx = np.argpartition(-key, m - 1)[:m]
        lf = math.log2(len(freqs))
        for i in idx:
            if key[i] <= 0:
                continue
            f, s, q = freqs[i], start[i], w * freqs[i]
            bits = _model_bits(t, lab, bi, n_bands, f, s, q, 0.0, -1.0)
            cost = param_fixed + lf + math.log2(math.ceil(PHASE_STEPS / q)) + bits[0]
            cands.append((l0 - cost, f, s, q, w, lf))
    if not cands:
        return out
    cands.sort(key=lambda c: -c[0])
    # Second window for the best primaries (a secondary eclipse at another phase, or the other eclipse of an
    # eccentric orbit). It pays for its own width index and phase grid.
    sols = []
    for d1, f, s, q, w, lf in cands[:top_secondary]:
        b2, s2, q2 = _best_secondary(t, lab, bi, n_bands, f, s, q, w_arr, PHASE_STEPS)
        d2 = l0 - (param_fixed + lf + math.log2(math.ceil(PHASE_STEPS / q)) + math.log2(len(widths)) + b2)
        sols.append((d2, f, s, q, s2, q2, w) if d2 > d1 else (d1, f, s, q, 0.0, -1.0, w))
    sols += [(d1, f, s, q, 0.0, -1.0, w) for d1, f, s, q, w, lf in cands[top_secondary:]]
    sols.sort(key=lambda c: -c[0])
    best = sols[0]
    out.update(_describe(t, lab, bi, n_bands, names, best))
    # Distinct periods among the best solutions (aliases and harmonics of one another are kept: the decision layer
    # checks each against the physics).
    seen = []
    for sol in sols:
        p = 1.0 / sol[1]
        if any(abs(p - x) / x < 1e-3 for x in seen):
            continue
        seen.append(p)
        out["solutions"].append({"delta_bits": round(float(sol[0]), 3), "period": float(p), "s1": float(sol[2]),
                                 "q1": float(sol[3]), "s2": float(sol[4]), "q2": float(sol[5])})
        if len(seen) >= n_keep:
            break
    return out


def _describe(t, lab, bi, n_bands, names, sol):
    d, f, s1, q1, s2, q2, w = sol
    bits, n_in, n_flat_in, k_out, e_out, n_dip_in = _model_bits(t, lab, bi, n_bands, f, s1, q1, s2, q2)
    ph = (t * f) % 1.0
    in1 = ((ph - s1) % 1.0) <= q1
    dips1 = in1 & (lab == 1)
    # Phase extent of the dip epochs in the primary window (times the period: a lower bound on eclipse duration).
    x = ((ph[dips1] - s1) % 1.0)
    dur_lo = float((x.max() - x.min()) / f) if dips1.sum() > 1 else 0.0
    cyc = np.unique(np.floor(t[lab == 1] * f - s1)[dips1[lab == 1]])
    return {"delta_bits": round(float(d), 3), "period": float(1 / f), "width_d": float(w),
            "s1": float(s1), "q1": float(q1), "s2": float(s2), "q2": float(q2), "secondary": bool(q2 > 0),
            "n_in_window": int(n_in), "n_flat_in_window": int(n_flat_in), "n_dip_in_window": int(n_dip_in),
            "n_dip_outside": int(k_out), "n_runs_outside": int(e_out), "n_cycles": int(len(cyc)),
            "duration_lo_d": dur_lo}


def covered(t, period, s1, q1, s2=0.0, q2=-1.0):
    """True where times t fall in a solution's windows."""
    ph = (np.asarray(t, float) / period) % 1.0
    out = ((ph - s1) % 1.0) <= q1
    if q2 > 0:
        out |= ((ph - s2) % 1.0) <= q2
    return out
