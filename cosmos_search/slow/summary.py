"""A summary of every star in a light-curve table, in one compiled pass, and trends measured against neighbours.

    summ = star_summary(lc, table)            # one row per star of a light-curve table (epoch.SCHEMA)
    ref = reference_table(summ)               # median trend of nearby stars of similar brightness
    summ = apply_reference(summ, ref)         # slope_z now against that median

Why. The release-day run used to score only the stars an archive-side cut picked (0.1% per G bin by `select.slow_z`).
Measured on DR3's Andromeda field on 2026-11-10, that cut picked none of the steady fades up to 3% a year injected
into quiet FGK dwarfs: the cut sits where stars with very large scatter are, whatever their smoothness. The bulk
light-curve files are read in full anyway, so every star in them is summarised here instead and nothing is chosen
before a light curve is read.

What a row holds. For G: the number of visits, their span, the visit scatter (`detect.visit_sigma`), the Abbe value,
the trend and its error, and the time-ordered episode scores `s_fade` and `s_rise` with their kinds and windows.
These are `detect.star_features`' numbers, computed by the same steps in one compiled loop over stars
(tests/test_slow_summary.py holds them equal), so the trend and episode parts of the rule need no further scoring:
every star is tested at every magnitude. For G, BP and RP on the transits all three bands have: the trend and its
error (as `detect`) and the Abbe value. From the uncorrected G epochs: the quiet-star test of `release.quiet_ids`,
and the unit-weight variance and Abbe value as DR4 publishes them in `all_source_photometry`, so a run can be
compared with the archive's own columns. G and BP-RP come from the light curve's median fluxes (`ZERO_POINTS`), so
no catalogue query is needed to place a star in its magnitude and colour cell.

The period search (`period`) is the one costly part left (4 ms a star on DR3's ~20 visits, about 8 times that on
DR4's). `rep_bound` and `rep_bound_bright` are upper bounds on its scores `s_rep` and `s_rep_bright`: a window of k
visits scores the sum of k - 1 of the star's detrended visit deviations over sqrt(k - 1) sigma, and no k - 1 visits
can sum lower than the k - 1 lowest. A star whose bound is below the repeat threshold cannot pass it, so the period
search is run only where the bound allows a pass, and loses nothing by it.

Trends against neighbours. The drift correction (`drift`) is fitted on quiet stars, which are 1% of stars at G < 14,
and on DR3 it left the median trend of Andromeda-field stars at +0.85 errors at G 12-13 (+0.074% a year). DR4's trend
errors are about 2.7 times smaller, so the same offset would be 2.3 errors. `reference_table` takes the median
trend of all stars (a median is not moved by the few that really change) in each sky pixel and 0.5 mag range of G,
at the finest HEALPix level where the cell holds at least REF_MIN_STARS stars, and `apply_reference` measures each
star's trend against it, with the median's error added to the star's own.
"""


import numpy as np
import pandas as pd

from cosmos_search.slow import detect, drift, period

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba ships with lsdb/hats; without it the loop runs uncompiled (slow)
    def njit(*args, **kwargs):
        return (lambda f: f) if not (args and callable(args[0])) else args[0]

# Vega zero points of DR3's G, BP and RP (Riello et al. 2021): mag = ZP - 2.5 log10(flux in e-/s). Used only to
# place a star in its magnitude and colour cell, where a few hundredths of a magnitude do not matter.
ZERO_POINTS = {"G": 25.6874, "BP": 25.3385, "RP": 24.7479}

QUIET_RATIO, QUIET_EXTREME, QUIET_MIN_N = 1.5, 8.0, 20  # release.quiet_ids' defaults

FADE_KINDS = np.array(["", "dip", "fade", "drop"], dtype=object)
RISE_KINDS = np.array(["", "bump", "rise", "jump"], dtype=object)

_COLS = ("n_g", "n_bp", "n_rp", "n_visits", "t_first", "t_last", "sigma_visit", "abbe", "slope", "slope_err",
         "s_fade", "fade_code", "dim_t0", "dim_t1", "dim_depth", "s_rise", "rise_code", "rise_t0", "rise_t1",
         "slope_g", "slope_g_err", "slope_bp", "slope_bp_err", "slope_rp", "slope_rp_err", "abbe_bp", "abbe_rp",
         "sigma_rob_g", "err_med_g", "quiet", "uwv_g", "abbe_g", "rep_bound", "rep_bound_bright")
_I = {c: i for i, c in enumerate(_COLS)}
INT_COLUMNS = ("n_g", "n_bp", "n_rp", "n_visits")
# Columns of a summary, in order. `slope_z` is slope / slope_err until `apply_reference` replaces it.
COLUMNS = ("source_id", "g_mag", "bp_rp", "gbin", "cbin", "n_g", "n_bp", "n_rp", "n_visits", "t_first", "t_last",
           "sigma_visit", "abbe", "slope", "slope_err", "slope_z", "s_fade", "fade_kind", "dim_t0", "dim_t1",
           "dim_depth", "s_rise", "rise_kind", "rise_t0", "rise_t1", "slope_g", "slope_g_err", "slope_bp",
           "slope_bp_err", "slope_rp", "slope_rp_err", "abbe_bp", "abbe_rp", "sigma_rob_g", "err_med_g", "quiet",
           "uwv_g", "abbe_g", "rep_bound", "rep_bound_bright")


@njit(cache=True)
def _visits(t, y, tv, yv, gap):
    """Visit mean times and values into tv, yv (as detect.visits); returns the number of visits."""
    n = len(t)
    if n == 0:
        return 0
    k, st, sy, m = 0, t[0], y[0], 1
    for i in range(1, n):
        if t[i] - t[i - 1] > gap:
            tv[k], yv[k] = st / m, sy / m
            k += 1
            st, sy, m = t[i], y[i], 1
        else:
            st += t[i]
            sy += y[i]
            m += 1
    tv[k], yv[k] = st / m, sy / m
    return k + 1


@njit(cache=True)
def _visit_sigma(yv, floor):
    d = np.diff(yv)
    return max(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0), floor)


@njit(cache=True)
def _abbe(yv):
    n = len(yv)
    m = yv.mean()
    v = np.sum((yv - m) ** 2) / (n - 1)
    return np.sum(np.diff(yv) ** 2) / (n - 1) / (2 * v) if v > 0 else np.nan


@njit(cache=True)
def _trend(tv, yv, sig):
    """Slope per year of visit means and its error (as detect._trend and detect.star_features)."""
    n = len(tv)
    x = (tv - tv.mean()) / 365.25
    sxx = np.sum(x * x)
    if not sxx > 0 or n < 3:
        return np.nan, np.nan
    ym = yv.mean()
    slope = np.sum(x * (yv - ym)) / sxx
    resid = yv - ym - slope * x
    return slope, max(sig, np.sqrt(np.sum(resid ** 2) / (n - 2))) / np.sqrt(sxx)


@njit(cache=True)
def _ordered(tv, yv, sigma, min_visits, max_fraction, min_days, out):
    """detect.ordered_episodes: the strongest getting-fainter and getting-brighter runs in time order, into
    out[0:5] = (s_fade, kind, t0, t1, depth) and out[5:9] = (s_rise, kind, t0, t1). Runs are visited in the order
    of numpy's triu_indices and the first of equal scores is kept, as numpy's argmax does."""
    n = len(yv)
    dev = yv - np.median(yv)
    c = np.zeros(n + 1)
    for k in range(n):
        c[k + 1] = c[k] + dev[k]
    found = False
    bf, bfi, bfj, bf_dim, bf_bright, bf_open, bf_end = -np.inf, 0, 0, 0.0, 0.0, False, False
    br, bri, brj, br_dim, br_bright, br_open, br_end = -np.inf, 0, 0, 0.0, 0.0, False, False
    for i in range(n + 1 - min_visits):
        lo, hi = np.inf, -np.inf
        for k in range(i, i + min_visits - 1):
            lo, hi = min(lo, dev[k]), max(hi, dev[k])
        for j in range(i + min_visits, n + 1):
            lo, hi = min(lo, dev[j - 1]), max(hi, dev[j - 1])
            length = j - i
            if length > max_fraction * n:
                break
            if tv[j - 1] - tv[i] < min_days:
                continue
            run = c[j] - c[i]
            norm = np.sqrt(length - 1.0) * sigma
            dim = -(run - lo) / norm
            bright = (run - hi) / norm
            half = run / length / 2
            if i == 0:
                open_lo, open_hi = True, True
            else:
                lead = c[i] / i
                open_lo, open_hi = lead < half, lead > half
            to_end = j == n
            fade = max(-np.inf if open_lo else dim, bright if (open_hi and not to_end) else -np.inf)
            rise = max(-np.inf if open_hi else bright, dim if (open_lo and not to_end) else -np.inf)
            if not found:
                found = True
                bfi, bfj, bf_dim, bf_bright, bf_open, bf_end = i, j, dim, bright, open_hi, to_end
                bri, brj, br_dim, br_bright, br_open, br_end = i, j, dim, bright, open_lo, to_end
            if fade > bf:
                bf, bfi, bfj, bf_dim, bf_bright, bf_open, bf_end = fade, i, j, dim, bright, open_hi, to_end
            if rise > br:
                br, bri, brj, br_dim, br_bright, br_open, br_end = rise, i, j, dim, bright, open_lo, to_end
    for k in range(9):
        out[k] = np.nan
    out[1], out[6] = 0.0, 0.0
    if not found:
        return
    out[0] = bf
    if bf_open and not bf_end and bf_bright >= bf_dim:
        out[1], out[2], out[3] = 3.0, tv[bfj], tv[n - 1]
    else:
        out[1], out[2], out[3] = (1.0 if bfj < n else 2.0), tv[bfi], tv[bfj - 1]
    # Depth of the window against the visits outside it (detect.star_features' dim_depth).
    n_in, s_in, n_out = 0, 0.0, 0
    outside = np.empty(n)
    for k in range(n):
        if tv[k] >= out[2] and tv[k] <= out[3]:
            n_in += 1
            s_in += yv[k]
        else:
            outside[n_out] = yv[k]
            n_out += 1
    if n_out > 0 and n_in > 0:
        out[4] = np.median(outside[:n_out]) - s_in / n_in
    out[5] = br
    if br_open and not br_end and br_dim >= br_bright:
        out[6], out[7], out[8] = 3.0, tv[brj], tv[n - 1]
    else:
        out[6], out[7], out[8] = (1.0 if brj < n else 2.0), tv[bri], tv[brj - 1]


@njit(cache=True)
def _rep_bounds(tv, yv, sigma, clip, max_fraction, min_visits, p_min, min_cycles):
    """Upper bounds on the period search's s_rep and s_rep_bright (module docstring), NaN where it gives none."""
    n = len(yv)
    kmax = int(np.floor(max_fraction * n))
    span = tv[n - 1] - tv[0]
    if kmax < min_visits or not span > 0 or min_cycles / span >= 1.0 / p_min:
        return np.nan, np.nan
    # period._detrend: a straight line fitted to the visits within clip sigma of the median.
    med = np.median(yv)
    keep = np.abs(yv - med) <= clip * sigma
    nk = keep.sum()
    if nk == n or nk < 3:
        tk, yk = tv, yv
    else:
        tk, yk = tv[keep], yv[keep]
    tm, ym = tk.mean(), yk.mean()
    x = tk - tm
    sxx = np.sum(x * x)
    slope = np.sum(x * (yk - ym)) / sxx if sxx > 0 else 0.0
    y = yv - ym - slope * (tv - tm)
    dev = np.sort(y - np.median(y))
    lo, hi, b_dim, b_bright = 0.0, 0.0, -np.inf, -np.inf
    for m in range(1, kmax):  # m = k - 1 visits left once the window's most extreme one is taken out
        lo += dev[m - 1]
        hi += dev[n - m]
        if m + 1 >= min_visits:
            norm = np.sqrt(float(m)) * sigma
            b_dim, b_bright = max(b_dim, -lo / norm), max(b_bright, hi / norm)
    return b_dim * (1 + 1e-9) + 1e-12, b_bright * (1 + 1e-9) + 1e-12


@njit(cache=True)
def _in_sorted(values, pool):
    """Whether each of `values` is in the sorted array `pool`."""
    out = np.zeros(len(values), np.bool_)
    if len(pool) == 0:
        return out
    pos = np.searchsorted(pool, values)
    for k in range(len(values)):
        out[k] = pos[k] < len(pool) and pool[pos[k]] == values[k]
    return out


@njit(cache=True)
def _kernel(starts, ends, band, t, y, raw, err, tid, gap, min_visits, min_ep_visits, max_fraction, min_days,
            floor, clip, p_min, min_cycles, q_ratio, q_extreme, q_min_n, ncol):
    """One row of statistics per star. Rows of a star are ordered by band (G, BP, RP) and time."""
    ns = len(starts)
    out = np.full((ns, ncol), np.nan)
    nmax = 0
    for s in range(ns):
        nmax = max(nmax, ends[s] - starts[s])
    tv, yv = np.empty(nmax), np.empty(nmax)
    ep = np.empty(9)
    for s in range(ns):
        a, b = starts[s], ends[s]
        ng, nbp = 0, 0
        for k in range(a, b):
            if band[k] == 0:
                ng += 1
            elif band[k] == 1:
                nbp += 1
        nrp = b - a - ng - nbp
        out[s, 0], out[s, 1], out[s, 2], out[s, 3] = ng, nbp, nrp, 0
        out[s, 11], out[s, 16], out[s, 29] = 0.0, 0.0, 0.0
        if ng == 0:
            continue
        g0, g1 = a, a + ng
        # Uncorrected G epochs: the quiet-star test and the archive-style summary.
        r, e = raw[g0:g1], err[g0:g1]
        dev = r - np.median(r)
        sig_rob = 1.4826 * np.median(np.abs(dev))
        err_med = np.median(e)
        out[s, 27], out[s, 28] = sig_rob, err_med
        out[s, 29] = 1.0 if (ng >= q_min_n and sig_rob < q_ratio * err_med and dev.min() > -q_extreme * sig_rob
                             and dev.max() < q_extreme * sig_rob) else 0.0
        if ng > 1:
            w = 1.0 / e ** 2
            mw = np.sum(w * r) / np.sum(w)
            res2 = np.sum((r - r.mean()) ** 2)
            out[s, 30] = np.sum(w * (r - mw) ** 2) / (ng - 1)
            out[s, 31] = ng / (2.0 * (ng - 1)) * np.sum(np.diff(r) ** 2) / res2 if res2 > 0 else np.nan
        nv = _visits(t[g0:g1], y[g0:g1], tv, yv, gap)
        out[s, 3] = nv
        if ng < min_visits or nv < min_visits:
            continue
        tg, yg = tv[:nv].copy(), yv[:nv].copy()
        sig = _visit_sigma(yg, floor)
        out[s, 4], out[s, 5], out[s, 6], out[s, 7] = tg[0], tg[nv - 1], sig, _abbe(yg)
        out[s, 8], out[s, 9] = _trend(tg, yg, sig)
        _ordered(tg, yg, sig, min_ep_visits, max_fraction, min_days, ep)
        out[s, 10], out[s, 11], out[s, 12], out[s, 13], out[s, 14] = ep[0], ep[1], ep[2], ep[3], ep[4]
        out[s, 15], out[s, 16], out[s, 17], out[s, 18] = ep[5], ep[6], ep[7], ep[8]
        out[s, 32], out[s, 33] = _rep_bounds(tg, yg, sig, clip, max_fraction, min_ep_visits, p_min, min_cycles)
        # Each band on the transits all three bands have (detect.same_transits).
        b0, r0 = g1, g1 + nbp
        sg, sb, sr = np.sort(tid[g0:g1]), np.sort(tid[b0:r0]), np.sort(tid[r0:b])
        for code in range(3):
            lo, hi = (g0, g1) if code == 0 else ((b0, r0) if code == 1 else (r0, b))
            ids = tid[lo:hi]
            if code == 0:
                m = _in_sorted(ids, sb) & _in_sorted(ids, sr)
            elif code == 1:
                m = _in_sorted(ids, sg) & _in_sorted(ids, sr)
            else:
                m = _in_sorted(ids, sg) & _in_sorted(ids, sb)
            if m.sum() < min_visits:
                continue
            nb = _visits(t[lo:hi][m], y[lo:hi][m], tv, yv, gap)
            if nb < min_visits:
                continue
            tb, yb = tv[:nb].copy(), yv[:nb].copy()
            out[s, 19 + 2 * code], out[s, 20 + 2 * code] = _trend(tb, yb, _visit_sigma(yb, floor))
            if code > 0:
                out[s, 24 + code] = _abbe(yb)
    return out


assert _COLS[10] == "s_fade" and _COLS[19] == "slope_g" and _COLS[25] == "abbe_bp" and _COLS[27] == "sigma_rob_g" \
    and _COLS[32] == "rep_bound" and len(_COLS) == 34


def magnitudes(med_flux):
    """G and BP-RP from median fluxes: a frame indexed by source_id with columns 0, 1, 2 (G, BP, RP)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = {b: ZERO_POINTS[n] - 2.5 * np.log10(med_flux[b].to_numpy(float)) if b in med_flux
               else np.full(len(med_flux), np.nan) for b, n in drift.BANDS.items()}
    return pd.DataFrame({"g_mag": mag[0], "bp_rp": mag[1] - mag[2]}, index=med_flux.index)


def corrected_epochs(lc, table=None, cells=None, marks=True, own=True):
    """Good epochs of a light-curve table with `frac_corr` (drift-corrected with `table`; uncorrected without one),
    ordered by star, band and time, and one row per star with g_mag, bp_rp, gbin and cbin.

    cells: G and colour bins from a catalogue (`drift.star_cells`), used for the stars it has; the rest, and every
    star when it is None, are placed by the magnitudes of their own median fluxes. marks, own: as
    `drift.normalise` (marks=False reads DR3 without its rejection marks, as DR4's bulk files arrive)."""
    ep = drift.normalise(lc, with_median=True, marks=marks, own=own)
    med = ep.groupby(["source_id", "band"], sort=True)["med_flux"].first().unstack("band")
    stars = magnitudes(med)
    own = drift.star_cells(stars.reset_index().rename(columns={"g_mag": "phot_g_mean_mag"}))
    if cells is not None:
        given = cells.reindex(own.index)
        has = given["gbin"].notna().to_numpy()
        for c in ("gbin", "cbin"):
            own[c] = np.where(has, given[c].to_numpy(), own[c].to_numpy()).astype(np.int8)
    stars[["gbin", "cbin"]] = own.to_numpy()
    order = np.lexsort((ep["t"].to_numpy(), ep["band"].to_numpy(), ep["source_id"].to_numpy()))
    if not np.all(order[1:] > order[:-1]):
        ep = ep.iloc[order].reset_index(drop=True)
    d = drift.drift_at(ep, own, table) if table is not None and len(ep) else 0.0
    return ep.assign(frac_corr=ep["frac"].to_numpy() - d), stars


def star_summary(lc, table=None, cells=None, epochs=None, marks=True, own=True):
    """One row per star of a light-curve table (epoch.SCHEMA) with any good epoch: COLUMNS (module docstring).

    table: the rule's drift table. Stars with fewer than detect.MIN_VISITS G visits keep their counts and
    magnitudes and have no statistics. epochs: `corrected_epochs`' output for `lc`, when the caller needs it too."""
    ep, stars = corrected_epochs(lc, table, cells, marks, own) if epochs is None else epochs
    if not len(ep):
        return pd.DataFrame(columns=list(COLUMNS))
    sid = ep["source_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    ends = np.r_[starts[1:], len(sid)]
    out = _kernel(starts, ends, ep["band"].to_numpy(np.int64), np.ascontiguousarray(ep["t"].to_numpy(float)),
                  np.ascontiguousarray(ep["frac_corr"].to_numpy(float)), np.ascontiguousarray(ep["frac"].to_numpy(float)),
                  np.ascontiguousarray(ep["err_frac"].to_numpy(float)), ep["transit_id"].to_numpy(np.int64),
                  detect.VISIT_GAP_DAYS, detect.MIN_VISITS, detect.MIN_EPISODE_VISITS, detect.MAX_EPISODE_FRACTION,
                  detect.MIN_EPISODE_DAYS, detect.SIGMA_FLOOR, period.DETREND_CLIP, period.P_MIN_DAYS,
                  period.MIN_CYCLES, QUIET_RATIO, QUIET_EXTREME, QUIET_MIN_N, len(_COLS))
    s = pd.DataFrame(out, columns=list(_COLS))
    s.insert(0, "source_id", sid[starts])
    s = s.join(stars, on="source_id")
    for c in INT_COLUMNS:
        s[c] = s[c].astype(np.int32)
    s["quiet"] = s["quiet"] > 0
    s["fade_kind"] = FADE_KINDS[s.pop("fade_code").to_numpy(int)]
    s["rise_kind"] = RISE_KINDS[s.pop("rise_code").to_numpy(int)]
    with np.errstate(invalid="ignore", divide="ignore"):
        s["slope_z"] = s["slope"] / s["slope_err"]
    return s[list(COLUMNS)]


def _hash(source_id):
    """splitmix64 of source_id (the mixing steps of features.control_mask), for a repeatable pseudo-random order."""
    with np.errstate(over="ignore"):
        z = np.asarray(source_id, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


# -- storage -------------------------------------------------------------------------------------------------------

# What a run stores per star: about 65 bytes as Parquet (float32, byte-stream split, zstd). Left out: gbin, cbin and
# slope_z (recomputed by `from_store`), and the uncorrected-epoch columns, which the full scores carry for the stars
# that have them. Episode windows are stored only for scores of STORE_MIN_SCORE or more; the shared-date test reads
# them from 4 up (rule_v3.SHARED_S_MIN).
STORED = ("source_id", "g_mag", "bp_rp", "n_g", "n_bp", "n_rp", "n_visits", "t_first", "t_last", "sigma_visit",
          "abbe", "slope", "slope_err", "s_fade", "fade_kind", "dim_t0", "dim_t1", "dim_depth", "s_rise", "rise_kind",
          "rise_t0", "rise_t1", "slope_g", "slope_g_err", "slope_bp", "slope_bp_err", "slope_rp", "slope_rp_err",
          "abbe_bp", "abbe_rp", "quiet", "rep_bound")
STORE_MIN_SCORE = 3.0


def to_store(summ, extra=()):
    """The stored form of a summary: STORED (plus `extra` columns), floats as float32, counts as int16, episode
    windows blanked below STORE_MIN_SCORE, and rep_bound the larger of the dimming and brightening bounds."""
    t = summ[[c for c in STORED if c in summ and c != "rep_bound"] + list(extra)].copy()
    t["rep_bound"] = np.fmax(summ["rep_bound"].to_numpy(float), summ["rep_bound_bright"].to_numpy(float)) \
        if "rep_bound_bright" in summ else summ["rep_bound"].to_numpy(float)
    for score, cols, kind in (("s_fade", ("dim_t0", "dim_t1", "dim_depth"), "fade_kind"),
                              ("s_rise", ("rise_t0", "rise_t1"), "rise_kind")):
        low = ~(t[score].to_numpy(float) >= STORE_MIN_SCORE)
        for c in cols:
            t[c] = np.where(low, np.nan, t[c].to_numpy(float))
        t[kind] = pd.Categorical(np.where(low, "", t[kind].to_numpy()),
                                 categories=list(FADE_KINDS if kind == "fade_kind" else RISE_KINDS))
    for c in t.columns:
        if t[c].dtype == np.float64:
            t[c] = t[c].astype(np.float32)
        elif c in INT_COLUMNS:
            t[c] = t[c].astype(np.int16)
    return t


def write(summ, path, extra=()):
    """Write `to_store(summ)` as Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pa.Table.from_pandas(to_store(summ, extra), preserve_index=False)
    floats = [f.name for f in t.schema if pa.types.is_float32(f.type)]
    tmp = str(path) + ".tmp"
    pq.write_table(t, tmp, compression="zstd", use_dictionary=[f.name for f in t.schema if f.name not in floats],
                   use_byte_stream_split=floats)
    __import__("os").replace(tmp, path)


def from_store(t):
    """A stored summary back in working form: float64, with gbin, cbin and slope_z."""
    out = t.copy()
    for c in out.columns:
        if out[c].dtype == np.float32:
            out[c] = out[c].astype(np.float64)
    for c in ("fade_kind", "rise_kind"):
        if c in out:
            out[c] = out[c].astype(object)
    cells = drift.star_cells(out.rename(columns={"g_mag": "phot_g_mean_mag"}))
    out["gbin"], out["cbin"] = cells["gbin"].to_numpy(), cells["cbin"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["slope_z"] = out["slope"] / out["slope_err"]
    return out


def random_share(source_id, share):
    """A repeatable pseudo-random `share` of stars, by a hash of source_id."""
    h = _hash(source_id)
    if share >= 1:
        return np.ones(h.shape, bool)
    return h < np.uint64(int(max(share, 0.0) * 2.0 ** 64))


# -- trends against neighbours of similar brightness -------------------------------------------------------------

REF_G_STEP = 0.5                         # width of the G ranges
REF_G_RANGE = (5.0, 21.5)                # brighter and fainter stars join the end ranges
REF_LEVELS = (6, 5, 4, 3, 2, 1, 0, -1)   # HEALPix levels tried, finest first; -1 is every star given
REF_MIN_STARS = 50


def g_range(g_mag):
    """Index of each star's REF_G_STEP range of G."""
    g = np.clip(np.nan_to_num(np.asarray(g_mag, float), nan=REF_G_RANGE[1]), *REF_G_RANGE)
    return np.floor((g - REF_G_RANGE[0]) / REF_G_STEP).astype(np.int16)


def _pixel(source_id, level):
    sid = np.asarray(source_id, np.int64)
    return drift.sky_pixel(sid, level) if level >= 0 else np.zeros(len(sid), np.int64)


def reference_table(summ, col="slope", levels=REF_LEVELS, min_stars=REF_MIN_STARS):
    """Median of `col` over the stars of each sky pixel and G range, for every level in `levels`, where at least
    min_stars stars have a value: level, pix, grange, n, med, err. err is the median's standard error from the
    cell's robust scatter, 1.2533 x 1.4826 MAD / sqrt(n).

    Build it from real stars only. A star with a fake change injected is then measured with `apply_reference`
    against the table its host helped make, and never enters it."""
    v = summ[col].to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(summ["g_mag"].to_numpy(float))
    sid, v = summ["source_id"].to_numpy(np.int64)[ok], v[ok]
    gr = g_range(summ["g_mag"].to_numpy(float)[ok])
    parts = []
    for level in levels:
        d = pd.DataFrame({"pix": _pixel(sid, level), "grange": gr, "v": v})
        g = d.groupby(["pix", "grange"], sort=False)["v"]
        med = g.transform("median")
        t = pd.DataFrame({"n": g.size(), "med": g.median(),
                          "mad": (d["v"] - med).abs().groupby([d["pix"], d["grange"]], sort=False).median()})
        t = t[t["n"] >= min_stars].reset_index()
        t["err"] = 1.2533 * 1.4826 * t["mad"] / np.sqrt(t["n"])
        t.insert(0, "level", level)
        parts.append(t[["level", "pix", "grange", "n", "med", "err"]])
    return pd.concat(parts, ignore_index=True)


def lookup(ref, source_id, g_mag):
    """Each star's reference from `reference_table`: the finest level with a cell for it. Returns ref (median),
    ref_err, n_ref and ref_level (NaN, NaN, 0 and -9 where no level has one)."""
    sid = np.asarray(source_id, np.int64)
    gr = g_range(g_mag)
    n = len(sid)
    med, err = np.full(n, np.nan), np.full(n, np.nan)
    cnt, lev = np.zeros(n, np.int64), np.full(n, -9, np.int16)
    for level in sorted(ref["level"].unique(), reverse=True):
        todo = np.flatnonzero(cnt == 0)
        if not len(todo):
            break
        r = ref[ref["level"] == level].set_index(["pix", "grange"])
        hit = r.index.get_indexer(pd.MultiIndex.from_arrays([_pixel(sid[todo], level), gr[todo]]))
        ok = hit >= 0
        med[todo[ok]], err[todo[ok]] = r["med"].to_numpy()[hit[ok]], r["err"].to_numpy()[hit[ok]]
        cnt[todo[ok]], lev[todo[ok]] = r["n"].to_numpy()[hit[ok]], level
    return med, err, cnt, lev


def apply_reference(summ, ref, g_mag=None):
    """`summ` with trends measured against `ref`: slope_ref, slope_ref_err, n_ref, ref_level, the unreferenced
    significance as slope_z_raw, and slope_z = (slope - slope_ref) / sqrt(slope_err^2 + slope_ref_err^2). Stars
    without a reference keep slope_z_raw as slope_z.

    g_mag: magnitudes to place the stars by instead of their own (a fake's host's, so the fake is measured
    against exactly its host's cell)."""
    out = summ.copy()
    med, err, cnt, lev = lookup(ref, out["source_id"].to_numpy(), out["g_mag"].to_numpy() if g_mag is None else g_mag)
    raw = out["slope_z_raw"].to_numpy(float) if "slope_z_raw" in out else out["slope_z"].to_numpy(float)
    slope = out["slope"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope_err = out["slope_err"].to_numpy(float) if "slope_err" in out else np.abs(slope / raw)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (slope - med) / np.sqrt(slope_err ** 2 + err ** 2)
    out["slope_ref"], out["slope_ref_err"], out["n_ref"], out["ref_level"] = med, err, cnt, lev
    out["slope_z_raw"] = raw
    out["slope_z"] = np.where(cnt > 0, z, raw)
    return out


def cell_sample(summ, level, per_cell=200, col="slope"):
    """Up to per_cell stars with a value of `col` from each sky pixel (at `level`) and G range, in a repeatable
    pseudo-random order: source_id, g_mag, `col`. A whole-sky run reads its summaries one sky region at a time;
    the cells too sparse for a region's own stars (bright stars at high latitude) take their reference from
    coarser pixels, built from this sample of every region (`reference_table(sample, levels=coarser)`)."""
    v = summ[col].to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(summ["g_mag"].to_numpy(float))
    d = pd.DataFrame({"source_id": summ["source_id"].to_numpy(np.int64)[ok], "g_mag": summ["g_mag"].to_numpy(float)[ok],
                      col: v[ok]})
    d = d.iloc[np.argsort(_hash(d["source_id"].to_numpy()), kind="stable")]
    first = d.groupby([_pixel(d["source_id"].to_numpy(), level), g_range(d["g_mag"].to_numpy())]).cumcount() < per_cell
    return d[first.to_numpy()].reset_index(drop=True)


def offset_by_magnitude(summ, col="slope_z", edges=(0, 12, 13, 14, 15, 16, 17, 18, 19, 20, 99)):
    """Median of `col` per G range with the star count: the release-day check that trends sit at zero."""
    ok = np.isfinite(summ[col].to_numpy(float))
    d = summ[ok]
    g = pd.cut(d["g_mag"], list(edges), right=False)
    return d.groupby(g, observed=True)[col].agg(n="size", median="median").reset_index()
