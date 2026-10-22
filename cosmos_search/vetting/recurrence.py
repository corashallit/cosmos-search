"""Does a star's dipping recur at one strict period? A test on every epoch the packet shows, in all surveys.

An eclipse comes back at a fixed period: fold the epochs at that period and the in-dip epochs gather in one phase
window that holds nothing else. Dips from an occulter that is not on a closed orbit (dust clumps, debris, a swarm
of bodies on different orbits, or something stranger) do not: at whatever period lines a few of them up, epochs
at normal brightness fall in the same window. A blind vetter reading tables of epochs cannot run that search
(docs/dayone.md, batch 3), so the pipeline runs it and puts the result in the packet.

Steps, on the visible epochs only (the packet's holdout is applied first):

1. In-dip epochs: per band, flux below the median by DIP_SIGMA robust sigma and by more than the band's largest
   brightening (a mirror test: noise outliers come in both signs, occultations in one). Gaia's BP and RP are left out
   because they share G's epochs; other surveys count as separate bands.
2. Events: in-dip epochs closer than EVENT_GAP days are one event (Gaia's two fields of view, a night of ZTF, a
   run of TESS bins). Each event is represented by its deepest epoch.
3. Search: over a log-spaced frequency grid from 2/span to 1/P_MIN and a few window widths, find the most events
   that fall in one phase window. Then, for the best few alignments, count the epochs at normal brightness in
   the phase range the aligned events span (purity = in-dip epochs / (in-dip + normal-brightness epochs) in it;
   epochs fainter than FLAT_SIGMA but short of the dip threshold, as in ingress, count for neither). An alignment
   counts only if purity >= MIN_PURITY.
4. At that period, a second pure window may hold more events (a secondary eclipse at another phase). Report the
   number of events, how many one strict period explains with up to two pure windows (n_aligned = n_primary +
   n_secondary), that period, the primary window's purity and the aligned fraction. An EB gives aligned_fraction
   near 1; an aperiodic train with many events gives a small one. With two or three events
   any train can be aligned by chance, so `testable` needs MIN_EVENTS.
"""

import numpy as np

DIP_SIGMA = 4.0
FLAT_SIGMA = 2.0               # an epoch less than this many robust sigma low is at normal brightness
SKIP_BANDS = ("BP", "RP")
EVENT_GAP = 1.0                 # days
P_MIN = 0.2                     # days; contact binaries go lower but do not dip in isolated events
WIDTHS = (0.05, 0.15, 0.5, 1.5, 5.0)   # trial eclipse durations, days
MIN_PURITY = 0.8
MIN_EVENTS = 4
APERIODIC_UNALIGNED = 2         # rule v3's flag: at least this many events no single period explains,
APERIODIC_FRACTION = 0.6        # and at most this share explained (calibrated in docs/dayone.md)
MIN_SECONDARY = 3              # two events line up at some phase of almost any period
TOP_K = 200                     # alignments per count level checked against every epoch
LEVEL_CAP = 2000
CHUNK = 4096


def _robust_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def epochs(lcs, with_sigma=False):
    """(t, in_dip, depth_sigma) over all usable bands, sorted by time. lcs: objects with band, time, flux.
    with_sigma: also each epoch's band robust sigma (relative flux), for `search`'s sigma."""
    ts, dips, zs, ss = [], [], [], []
    for lc in lcs:
        if lc.band in SKIP_BANDS:
            continue
        ok = np.isfinite(lc.time) & np.isfinite(lc.flux)
        t, f = lc.time[ok], lc.flux[ok]
        if len(t) < 10:
            continue
        med = np.median(f)
        s = _robust_sigma(f / med)
        if not s > 0:
            continue
        z = (1 - f / med) / s
        # Photometric outliers come in both signs; dips from an occulter only in one. A dip epoch must be deeper
        # than the band's largest brightening, so heavy-tailed noise (ZTF's) does not add one-epoch events.
        k = max(DIP_SIGMA, float(np.max(-z)))
        ts.append(t)
        dips.append(z > k)
        zs.append(z)
        ss.append(np.full(len(t), s))
    if not ts:
        empty = (np.array([]), np.array([], bool), np.array([]))
        return empty + (np.array([]),) if with_sigma else empty
    t, d, z, sg = np.concatenate(ts), np.concatenate(dips), np.concatenate(zs), np.concatenate(ss)
    o = np.argsort(t)
    return (t[o], d[o], z[o], sg[o]) if with_sigma else (t[o], d[o], z[o])


def events(t, in_dip, z):
    """Deepest epoch's time and each event's (start, end, depth in sigma), for runs of in-dip epochs."""
    td, zd = t[in_dip], z[in_dip]
    if len(td) == 0:
        return np.empty((0, 4))
    breaks = np.flatnonzero(np.diff(td) > EVENT_GAP) + 1
    out = []
    for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(td)]):
        k = a + int(np.argmax(zd[a:b]))
        out.append((td[k], td[a], td[b - 1], zd[k]))
    return np.array(out)


def _max_in_window(phases, q):
    """Per row: most points in any circular window of width q, and the phase where that window starts.
    phases is (n_freq, n) sorted along axis 1."""
    n = phases.shape[1]
    ext = np.concatenate([phases, phases + 1.0], axis=1)
    best = np.ones(len(phases), int)
    start = phases[:, 0].copy()
    for k in range(1, n):
        span = ext[:, k:k + n] - ext[:, :n]
        fits = span <= np.reshape(q, (-1, 1))
        anyfit = fits.any(axis=1)
        if not anyfit.any():
            break
        best[anyfit] = k + 1
        start[anyfit] = phases[anyfit, np.argmax(fits[anyfit], axis=1)]
    return best, start


def _purity(t, in_dip, flat, f, s, end):
    """In-dip share of the epochs whose phase (from start s) is within [0, end], counting only epochs that are
    either in dip or clearly at normal brightness: an epoch caught in ingress or egress is fainter than normal but
    short of the dip threshold, and is evidence for neither reading."""
    inside = ((t * f - s) % 1.0) <= end + 1e-9
    n_dip, n_flat = int(in_dip[inside].sum()), int(flat[inside].sum())
    return n_dip / (n_dip + n_flat) if n_dip + n_flat else 0.0


def _secondary(te, t, in_dip, flat, f, q, taken):
    """Most of the remaining events that one more pure window at the same frequency holds (a second eclipse)."""
    rest = te[~taken]
    if len(rest) < 1:
        return 0, np.nan
    ph = np.sort((rest * f) % 1.0)[None, :]
    c, s2 = _max_in_window(ph, np.array([q]))
    c, s2 = int(c[0]), float(s2[0])
    x = (rest * f - s2) % 1.0
    end = x[x <= q + 1e-12].max()
    if c < MIN_SECONDARY or _purity(t, in_dip, flat, f, s2, end) < MIN_PURITY:
        return 0, np.nan
    return c, s2


def search(t, in_dip, z, widths=WIDTHS, p_min=P_MIN, top_k=TOP_K, sigma=None):
    """Recurrence statistics (dict) for epochs from `epochs`.

    sigma (each epoch's band robust sigma, `epochs(..., with_sigma=True)`): an epoch then counts as normal
    brightness in a purity window only if its band could have shown the dips, i.e. the median in-dip depth is at
    least DIP_SIGMA of that band's sigma. Without it, a band too noisy to see an eclipse (TESS on a faint or
    blended star) reads as normal brightness inside the eclipse window and breaks a real period (review R18)."""
    ev = events(t, in_dip, z)
    n_ev = len(ev)
    out = {"n_epochs": int(len(t)), "n_in_dip": int(in_dip.sum()), "n_events": int(n_ev),
           "testable": bool(n_ev >= MIN_EVENTS)}
    if n_ev < 2 or len(t) < 2:
        return out
    span = t[-1] - t[0]
    f_min, f_max = 2.0 / span, 1.0 / p_min
    te = ev[:, 0]
    flat = z < FLAT_SIGMA
    if sigma is not None:
        flat &= DIP_SIGMA * sigma <= np.median(z[in_dip] * sigma[in_dip])
    best = {"n_aligned": 1, "n_primary": 1, "n_secondary": 0, "period": np.nan, "purity": np.nan,
            "width": np.nan, "n_cycles": 0}
    for w in widths:
        # A window drifts by span * df in phase; keep that under a quarter of the window, w * f.
        n_f = min(int(np.ceil(np.log(f_max / f_min) * span / (0.25 * w))), 5_000_000)
        freqs = np.geomspace(f_min, f_max, n_f)
        freqs = freqs[w * freqs <= 0.25]              # an eclipse lasts well under a quarter of the orbit
        if not len(freqs):
            continue
        levels = {}                                   # count -> [(freq, phase start)], first LEVEL_CAP kept
        for i in range(0, len(freqs), CHUNK):
            f = freqs[i:i + CHUNK]
            ph = np.sort((te[None, :] * f[:, None]) % 1.0, axis=1)
            cnt, start = _max_in_window(ph, w * f)
            for c in np.unique(cnt[2 * cnt > max(4, best["n_aligned"])]):
                lst = levels.setdefault(int(c), [])
                if len(lst) < LEVEL_CAP:
                    m = np.flatnonzero(cnt == c)[: LEVEL_CAP - len(lst)]
                    lst += list(zip(f[m], start[m]))
        # Highest count first. Purity is measured over the phase range the aligned events span, not the trial
        # width, so a wide trial window is not penalised for flat epochs beside a narrow eclipse. Chance alignments
        # of aperiodic dips fail on purity, so the search walks down the levels. A second pure window at the same
        # period (a secondary eclipse) adds its events; it holds no more than the first, so a level can no longer
        # win once twice its count is below the best total.
        for c in sorted(levels, reverse=True):
            if 2 * c <= best["n_aligned"]:
                break
            for f, s in levels[c][:top_k]:
                q = w * f
                x = (te * f - s) % 1.0
                taken = x <= q + 1e-12
                purity = _purity(t, in_dip, flat, f, s, x[taken].max())
                if purity < MIN_PURITY:
                    continue
                c2, _ = _secondary(te, t, in_dip, flat, f, q, taken)
                total = c + c2
                if total > best["n_aligned"] or (total == best["n_aligned"] and purity > np.nan_to_num(best["purity"])):
                    best = {"n_aligned": int(total), "n_primary": int(c), "n_secondary": int(c2),
                            "period": float(1 / f), "purity": float(purity), "width": float(w),
                            "n_cycles": int(len(np.unique(np.floor(te[taken] * f - s))))}
    out.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in best.items()})
    out["aligned_fraction"] = round(best["n_aligned"] / n_ev, 4)
    out["aperiodic"] = bool(out["testable"] and n_ev - best["n_aligned"] >= APERIODIC_UNALIGNED
                            and out["aligned_fraction"] <= APERIODIC_FRACTION)
    depths = ev[:, 3]
    out["event_depth_cv"] = round(float(np.std(depths) / np.mean(depths)), 4)
    return out


def test(lcs):
    """Packet test dict for a list of visible light curves (vetting.candidate.LightCurve)."""
    t, d, z, s = epochs(lcs, with_sigma=True)
    r = search(t, d, z, sigma=s)
    # What the test computes and what else produces the same result, but no calibration figures: those would
    # make the vetter's score repeat the flag rather than weigh the light curve (docs/dayone.md, review fixes).
    r["description"] = (
        "Recurrence across all surveys: n_events distinct dips (epochs >= 4 robust sigma low and deeper than the largest brightening in their band, merged within 1 d); "
        "n_aligned of them fall in one or two pure phase windows (n_primary, n_secondary: two eclipses) at one "
        "strict period (period, d); purity = share of epochs in the primary window that are in dip (counting normal-brightness epochs only from bands precise enough to have shown the dips); "
        "aligned_fraction = n_aligned / n_events. Eclipses give aligned_fraction near 1, dips that do not recur "
        "at one period give a small one. aperiodic is true when at least 2 events are left unexplained and "
        "aligned_fraction <= 0.6. It says only that no single strict period explains the dips: spot evolution, "
        "irregular pulsation, flaring and other variability of the star give dips that do not recur too. "
        "testable is false with fewer than 4 events, where any dips can be aligned by chance.")
    r["visible_only"] = True
    return r
