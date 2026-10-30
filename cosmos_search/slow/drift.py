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

TBIN_DAYS = 15.0
G_EDGES = (0.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 23.0)
COLOUR_EDGES = (-9.0, 0.8, 1.3, 9.0)
MIN_STARS = 30
POOLED = -1  # colour bin code for the all-colour fallback cells
BANDS = {0: "G", 1: "BP", 2: "RP"}


def _columns(lc):
    return lc.column_names if hasattr(lc, "column_names") else list(lc.columns)


def normalise(lc):
    """Good epochs of a long light-curve table (epoch.SCHEMA) as fractional flux about each star's median.

    Returns source_id, band (0 G, 1 BP, 2 RP), t (days), frac, err_frac, fov and transit_id. Epochs DPAC
    rejected, with non-positive flux or without an error are dropped.
    """
    band = np.asarray(lc["band"].combine_chunks().indices) if hasattr(lc["band"], "combine_chunks") \
        else np.asarray(lc["band"])
    df = pd.DataFrame({"source_id": np.asarray(lc["source_id"]), "band": band.astype(np.int8),
                       "t": np.asarray(lc["t"]), "flux": np.asarray(lc["flux"]), "err": np.asarray(lc["err"]),
                       "flags": np.asarray(lc["flags"]), "fov": np.asarray(lc["fov"]),
                       "transit_id": np.asarray(lc["transit_id"]) if "transit_id" in _columns(lc)
                       else np.zeros(len(band), np.int64)})
    good = ((df["flags"] & features.FLAG_VARI_REJECT) == 0) & (df["flux"] > 0) & (df["err"] > 0) \
        & np.isfinite(df["err"])
    df = df[good]
    med = df.groupby(["source_id", "band"])["flux"].transform("median")
    return pd.DataFrame({"source_id": df["source_id"].to_numpy(), "band": df["band"].to_numpy(),
                         "t": df["t"].to_numpy(), "frac": (df["flux"] / med - 1).to_numpy(),
                         "err_frac": (df["err"] / med).to_numpy(), "fov": df["fov"].to_numpy(),
                         "transit_id": df["transit_id"].to_numpy()})


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
