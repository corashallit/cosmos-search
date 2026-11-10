"""Calibration drift shared by stars of similar magnitude and colour, measured from quiet stars and removed.

Gaia's photometric calibration leaves small time-dependent offsets that all stars in the same magnitude range
(window class, gating) and colour share. On DR3 the median trend of quiet stars is +0.02 to +0.1 % per year at
G < 17 and +0.3 % per year at G 20-21 (docs/arms/slow-changes.md), comparable to the trends we are after, so it
is measured and subtracted before any star's trend is fitted.

Model: for each band, the drift in cell (G bin, colour bin, time bin) is the median fractional flux of the
reference stars in it, each star normalised by its own median. Cells with fewer than MIN_STARS stars fall back
to the same G bin with all colours pooled, then to zero. Reference stars are quiet (scatter close to their
errors, no deep extremes); a star's own epochs never enter the drift applied to it when `fit` is given only
the other half of the sample (`half`).

    epochs = normalise(lc)                                  # one row per good epoch, per-star fractional flux
    cells = star_cells(aux)                                 # source_id -> G bin, colour bin
    table = fit(epochs[ref_a], cells)                       # drift per (band, G bin, colour bin, time bin)
    epochs["frac_corr"] = epochs["frac"] - drift_at(epochs, cells, table)
    trends = trend(epochs, "frac_corr")                     # slope and its error per star and band
"""

import numpy as np
import pandas as pd

from cosmos_search.gaia import features

try:
    from numba import njit
except ImportError:  # pragma: no cover - without numba the loop runs uncompiled (slow)
    def njit(*args, **kwargs):
        return (lambda f: f) if not (args and callable(args[0])) else args[0]

# Our own test for bad epochs (`own_good`): DR4's bulk light curves carry no rejection marks.
import os  # EXPERIMENT
OWN_ERR_RATIO = float(os.environ.get("X_ERR", 2.0))   # an epoch's error against the star's median error in that band
OWN_SPIKE = float(os.environ.get("X_SPIKE", 5.0))       # an isolated epoch's distance from both neighbours, in epoch-to-epoch scatters
TBIN_DAYS = 15.0
G_EDGES = (0.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 23.0)
COLOUR_EDGES = (-9.0, 0.8, 1.3, 9.0)
MIN_STARS = 30
POOLED = -1  # colour bin code for the all-colour fallback cells
BANDS = {0: "G", 1: "BP", 2: "RP"}


def _columns(lc):
    return lc.column_names if hasattr(lc, "column_names") else list(lc.columns)


@njit(cache=True)
def own_good(sid, band, t, flux, err, k_err=OWN_ERR_RATIO, k_spike=OWN_SPIKE):
    """Epochs to keep by our own test, for rows ordered by star, band and time.

    DR3 marked the epochs its variability processing rejected (4.6% of G epochs in the Andromeda field) and
    `normalise` drops them. DR4's bulk epoch photometry has no such column, so the same job is done from the light
    curve alone, per star and band:

    - an epoch whose error is more than k_err times the star's median error is dropped (42% of DR3's rejected
      epochs, 4% of the ones it kept);
    - an isolated spike is dropped: an epoch that differs from the epoch before and the epoch after, in the same
      direction, by more than k_spike times the star's epoch-to-epoch scatter (robust, from first differences, and
      no smaller than its median error). At either end of the light curve the two nearest epochs stand in.

    The spike test compares neighbours in time, never the star's median: a dimming that lasts two epochs or
    more moves its neighbours with it and is kept whole, however deep. A clip about the median would remove the
    deep episodes of quiet stars, which are what the search is for."""
    n = len(sid)
    keep = np.ones(n, np.bool_)
    a = 0
    while a < n:
        b = a + 1
        while b < n and sid[b] == sid[a] and band[b] == band[a]:
            b += 1
        if b - a >= 3:
            ok = err[a:b] <= k_err * np.median(err[a:b])
            keep[a:b] = ok
            idx = np.flatnonzero(ok) + a
            k = len(idx)
            if k >= 4:
                f = flux[idx]
                mf = np.median(f)
                y = f / mf
                d = np.diff(y)
                sig = max(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0), np.median(err[idx]) / mf)
                lim = k_spike * sig
                for i in range(k):
                    if i == 0:
                        p, q = y[0] - y[1], y[0] - y[2]
                    elif i == k - 1:
                        p, q = y[i] - y[i - 1], y[i] - y[i - 2]
                    else:
                        p, q = y[i] - y[i - 1], y[i] - y[i + 1]
                    if p * q > 0 and abs(p) > lim and abs(q) > lim:
                        keep[idx[i]] = False
        a = b
    return keep


def normalise(lc, with_median=False, marks=True, own=True):
    """Good epochs of a long light-curve table (epoch.SCHEMA) as fractional flux about each star's median.

    Returns source_id, band (0 G, 1 BP, 2 RP), t (days), frac, err_frac, fov and transit_id, ordered by star,
    band and time. Dropped: epochs with non-positive flux or without an error, epochs DPAC marked as rejected
    (marks; DR3 has the marks, DR4's bulk files do not) and epochs our own test rejects (own, `own_good`).
    marks=False on DR3 reads it as DR4 will arrive. with_median adds med_flux, the median flux of the star in
    that band (e-/s), which gives its magnitude without a catalogue (`summary.magnitudes`).
    """
    band = np.asarray(lc["band"].combine_chunks().indices) if hasattr(lc["band"], "combine_chunks") \
        else np.asarray(lc["band"])
    df = pd.DataFrame({"source_id": np.asarray(lc["source_id"]), "band": band.astype(np.int8),
                       "t": np.asarray(lc["t"]), "flux": np.asarray(lc["flux"]), "err": np.asarray(lc["err"]),
                       "flags": np.asarray(lc["flags"]), "fov": np.asarray(lc["fov"]),
                       "transit_id": np.asarray(lc["transit_id"]) if "transit_id" in _columns(lc)
                       else np.zeros(len(band), np.int64)})
    good = (df["flux"] > 0) & (df["err"] > 0) & np.isfinite(df["err"])
    if marks:
        good &= (df["flags"] & features.FLAG_VARI_REJECT) == 0
    df = df[good]
    if own and len(df):
        order = np.lexsort((df["t"].to_numpy(), df["band"].to_numpy(), df["source_id"].to_numpy()))
        if not np.all(order[1:] > order[:-1]):
            df = df.iloc[order]
        df = df[own_good(df["source_id"].to_numpy(np.int64), df["band"].to_numpy(np.int64),
                         np.ascontiguousarray(df["t"].to_numpy(float)), np.ascontiguousarray(df["flux"].to_numpy(float)),
                         np.ascontiguousarray(df["err"].to_numpy(float)), OWN_ERR_RATIO, OWN_SPIKE)]
    med = df.groupby(["source_id", "band"])["flux"].transform("median")
    out = pd.DataFrame({"source_id": df["source_id"].to_numpy(), "band": df["band"].to_numpy(),
                        "t": df["t"].to_numpy(), "frac": (df["flux"] / med - 1).to_numpy(),
                        "err_frac": (df["err"] / med).to_numpy(), "fov": df["fov"].to_numpy(),
                        "transit_id": df["transit_id"].to_numpy()})
    if with_median:
        out["med_flux"] = med.to_numpy()
    return out


def star_cells(aux):
    """G bin and colour bin per source from a frame with source_id, phot_g_mean_mag and bp_rp."""
    g = np.digitize(aux["phot_g_mean_mag"].to_numpy(), G_EDGES[1:-1])
    c = np.digitize(np.nan_to_num(aux["bp_rp"].to_numpy(), nan=1.0), COLOUR_EDGES[1:-1])
    return pd.DataFrame({"gbin": g.astype(np.int8), "cbin": c.astype(np.int8)},
                        index=pd.Index(aux["source_id"].to_numpy(), name="source_id"))


def reference_mask(feat, max_ratio=1.5, max_extreme=8.0, min_n=20):
    """Quiet stars for the drift: G scatter within max_ratio of the errors and no epoch beyond max_extreme
    robust sigma on either side (features-v1 columns)."""
    f = feat
    return ((f["n_g"] >= min_n) & (f["sigma_rob_g"] < max_ratio * f["err_med_g"])
            & (f["min_z_rob"] > -max_extreme) & (f["max_z_rob"] < max_extreme)).to_numpy()


def half(source_id):
    """Deterministic split of sources into halves 0 and 1 (splitmix64 hash, as features.control_mask)."""
    return features.control_mask(np.asarray(source_id), 0.5).astype(np.int8)


def _tbin(t):
    return np.floor(np.asarray(t) / TBIN_DAYS).astype(np.int32)


def fit(epochs, cells, min_stars=MIN_STARS):
    """Drift table: median frac per (band, gbin, cbin, tbin) over the given epochs, plus POOLED colour cells.

    Each star counts once per cell (its median within the cell), so densely sampled stars don't dominate.
    """
    e = epochs.join(cells, on="source_id", how="inner")
    e = e.assign(tbin=_tbin(e["t"]))
    per_star = e.groupby(["band", "gbin", "cbin", "tbin", "source_id"], observed=True)["frac"].median()
    out = []
    for keys in (["band", "gbin", "cbin", "tbin"], ["band", "gbin", "tbin"]):
        s = per_star.reset_index()
        g = s.groupby(keys, observed=True)["frac"].agg(drift="median", n_stars="size").reset_index()
        if "cbin" not in keys:
            g["cbin"] = POOLED
        out.append(g[g["n_stars"] >= min_stars])
    return pd.concat(out, ignore_index=True)[["band", "gbin", "cbin", "tbin", "drift", "n_stars"]]


def drift_at(epochs, cells, table):
    """Drift for each epoch: its own colour cell when measured, else the pooled cell, else 0."""
    e = epochs[["source_id", "band", "t"]].join(cells, on="source_id", how="left")
    e = e.assign(tbin=_tbin(e["t"]))
    key = ["band", "gbin", "cbin", "tbin"]
    own = table[table["cbin"] != POOLED].set_index(key)["drift"]
    pooled = table[table["cbin"] == POOLED].set_index(["band", "gbin", "tbin"])["drift"]
    d_own = own.reindex(pd.MultiIndex.from_frame(e[key])).to_numpy()
    d_pool = pooled.reindex(pd.MultiIndex.from_frame(e[["band", "gbin", "tbin"]])).to_numpy()
    return np.where(np.isfinite(d_own), d_own, np.where(np.isfinite(d_pool), d_pool, 0.0))


def trend(epochs, col="frac", core=5.0, min_n=5):
    """Per star and band: OLS slope of `col` against time (fraction per year), its error from the residual
    scatter, and the epoch count. Epochs beyond `core` robust sigma of the star's median are left out, as in
    features.source_features, so single deep dips don't drive the slope."""
    e = epochs[["source_id", "band", "t", col]].rename(columns={col: "y"})
    grp = e.groupby(["source_id", "band"])["y"]
    med = grp.transform("median")
    mad = (e["y"] - med).abs().groupby([e["source_id"], e["band"]]).transform("median") * 1.4826
    e = e[(e["y"] - med).abs() <= core * np.maximum(mad, 1e-12)]
    x = e["t"] / features.DAYS_PER_YEAR
    e = e.assign(x=x, xx=x * x, xy=x * e["y"], yy=e["y"] ** 2)
    s = e.groupby(["source_id", "band"])[["x", "y", "xx", "xy", "yy"]].sum()
    n = e.groupby(["source_id", "band"]).size().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dxx = n * s["xx"] - s["x"] ** 2
        slope = (n * s["xy"] - s["x"] * s["y"]) / dxx
        icpt = (s["y"] - slope * s["x"]) / n
        rss = (s["yy"] - 2 * icpt * s["y"] - 2 * slope * s["xy"] + n * icpt ** 2 + 2 * slope * icpt * s["x"]
               + slope ** 2 * s["xx"])
        err = np.sqrt(np.maximum(rss, 0) / (n - 2) * n / dxx)
    out = pd.DataFrame({"slope": slope, "slope_err": err, "n": n.astype(int)})
    return out[out["n"] >= min_n].reset_index()


def sky_pixel(source_id, level=5):
    """HEALPix (nested) pixel at `level` from a Gaia source_id (level 12 is source_id // 2^35)."""
    return (np.asarray(source_id, dtype=np.int64) // 2**35) // 4 ** (12 - level)



def sky_offsets(trends, level=5, min_stars=10):
    """Median trend significance (slope / slope_err) of reference stars per sky pixel and band.

    On DR3 the trends left after `fit` still share a sky term: pixel medians of the two halves of the reference
    stars agree (r = 0.56 over 147 level-5 pixels), and their spread is 1.9x what shuffling stars gives.
    Subtracting half A's pixel medians from half B removes only a little of it (ratio 1.87 to 1.79), because
    ~25 stars per pixel give noisy medians; docs/arms/slow-changes.md has the numbers. Pixels with fewer than
    min_stars reference stars are left out.
    """
    t = trends.assign(pix=sky_pixel(trends["source_id"], level), z=trends["slope"] / trends["slope_err"])
    g = t.groupby(["band", "pix"])["z"].agg(offset_z="median", n_stars="size").reset_index()
    return g[g["n_stars"] >= min_stars]
