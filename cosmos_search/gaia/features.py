"""Per-source summary features from Gaia epoch photometry, vectorised over whole bulk files.

These are ingredients for the dimmer detectors (docs/arms/gaia-dimmers.md), not a detector. Choices:

- Scale is robust (1.4826 × MAD of good G fluxes), because a real dip inflates the standard deviation and
  would hide itself; the published errors are reported alongside, since their ratio (`excess`) separates
  intrinsically variable stars from noisy measurements.
- Every faint-side statistic has a bright-side mirror (`min_*` / `max_*`, `n_low*` / `n_high*`).
  Instrumental noise is roughly symmetric, occultation is one-sided, so the mirror gives an empirical null
  per stratum without a noise model.
- Gaia's redundancy is used where DR3 allows: the deepest point's partner transit in the other field of
  view (~106.5 min apart), and BP and RP in the same transit. DR4 adds per-CCD G.
- DPAC's variability rejections are excluded from the scale but counted when they are faint outliers
  (`n_rej_low3`), because a rejection rule tuned for ordinary variables can throw away a real deep dip.
"""

import numpy as np
import pandas as pd
import pyarrow as pa

from .epoch import BANDS, FLAG_VARI_REJECT

PARTNER_MAX_DT = 0.1  # days; FoV pairs are 106.5 min = 0.074 d apart
EPISODE_GAP = 1.0  # days between low points that start a new episode
DAYS_PER_YEAR = 365.25

CONTROL_SALT = 0x9E3779B97F4A7C15 ^ 0x636F736D6F735F31  # public: controls are not truth
CONTROL_FRACTION = 0.01


def control_mask(source_id, fraction=CONTROL_FRACTION):
    """Deterministic pseudo-random subset of sources (splitmix64 of source_id), for random controls."""
    with np.errstate(over="ignore"):
        z = np.asarray(source_id, dtype=np.uint64) ^ np.uint64(CONTROL_SALT)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z < np.uint64(fraction * 2.0**64)


class _Groups:
    """Row -> group bookkeeping for keys that are already sorted."""

    def __init__(self, keys, uniq):
        self.inv = np.searchsorted(uniq, keys)
        self.n = len(uniq)

    def count(self, mask):
        return np.bincount(self.inv[mask], minlength=self.n)

    def sum(self, values, mask):
        return np.bincount(self.inv[mask], weights=values[mask], minlength=self.n)

    def quantile(self, values, mask, q):
        g, v = self.inv[mask], values[mask]
        order = np.lexsort((v, g))
        v = v[order]
        c = np.bincount(g, minlength=self.n)
        start = np.concatenate([[0], np.cumsum(c)[:-1]])
        out = np.full(self.n, np.nan)
        ok = c > 0
        pos = start[ok] + q * (c[ok] - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, start[ok] + c[ok] - 1)
        frac = pos - lo
        out[ok] = v[lo] * (1 - frac) + v[hi] * frac
        return out

    def arg(self, values, mask, largest=False):
        """Row index of the min (or max) value per group, -1 where the group has no rows in mask."""
        rows = np.flatnonzero(mask)
        v = values[rows]
        order = np.lexsort((-v if largest else v, self.inv[rows]))
        rows = rows[order]
        g = self.inv[rows]
        first = np.ones(len(rows), bool)
        first[1:] = g[1:] != g[:-1]
        out = np.full(self.n, -1, np.int64)
        out[g[first]] = rows[first]
        return out


def _take(values, idx, fill=np.nan):
    out = np.full(len(idx), fill, dtype=np.result_type(values.dtype, type(fill)))
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def _partners(sid, t, fov):
    """For rows sorted by (source, t): index of the nearest other-FoV row within PARTNER_MAX_DT, else -1."""
    n = len(t)
    best = np.full(n, -1, np.int64)
    best_dt = np.full(n, np.inf)
    for shift in (-1, 1):
        j = np.arange(n) + shift
        ok = (j >= 0) & (j < n)
        j = np.where(ok, j, 0)
        dt = np.abs(t[j] - t)
        cand = ok & (sid[j] == sid) & (fov[j] != fov) & (dt < PARTNER_MAX_DT) & (dt < best_dt)
        best[cand], best_dt[cand] = j[cand], dt[cand]
    return best


def _band_stats(lc, band_code, uniq):
    m = lc["band"] == band_code
    sid, t, flux, err = lc["source_id"][m], lc["t"][m], lc["flux"][m], lc["err"][m]
    flags, fov = lc["flags"][m], lc["fov"][m]
    G = _Groups(sid, uniq)
    good = ((flags & FLAG_VARI_REJECT) == 0) & (flux > 0) & np.isfinite(err) & (err > 0)
    med = G.quantile(flux, good, 0.5)
    mad = G.quantile(np.abs(flux - med[G.inv]), good, 0.5)
    sigma = 1.4826 * mad
    medg = med[G.inv]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (flux - medg) / sigma[G.inv]
        zerr = (flux - medg) / err
        frac = flux / medg - 1
    z = np.where(np.isfinite(z), z, np.nan)
    zf = np.where(good & np.isfinite(z), z, np.nan)
    return dict(mask=m, sid=sid, t=t, flux=flux, err=err, flags=flags, fov=fov, G=G, good=good,
                med=med, sigma=sigma, z=z, zf=zf, zerr=zerr, frac=frac,
                tid=lc["transit_id"][m], row=lc["ccd_row"][m], nobs=lc["n_obs"][m])


def source_features(table):
    """Compute one row of features per source from a light-curve table (epoch.SCHEMA). Returns pandas."""
    lc = {name: table[name].to_numpy() if name != "band" else
          np.asarray(table["band"].combine_chunks().indices) for name in table.column_names}
    uniq = np.unique(lc["source_id"])
    out = {"source_id": uniq}
    b = {code: _band_stats(lc, code, uniq) for code in range(len(BANDS))}
    g = b[0]
    G, good = g["G"], g["good"]
    finite = good & np.isfinite(g["z"])

    out["n_g"] = G.count(good)
    out["n_rej"] = G.count(~good)
    out["med_flux_g"] = g["med"]
    out["sigma_rob_g"] = g["sigma"] / g["med"]
    out["err_med_g"] = G.quantile(g["err"], good, 0.5) / g["med"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["excess_g"] = out["sigma_rob_g"] / out["err_med_g"]
    q05, q95 = G.quantile(g["flux"], good, 0.05), G.quantile(g["flux"], good, 0.95)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["tail_asym_g"] = (g["med"] - q05) / (q95 - g["med"])
        out["time_span_d"] = G.quantile(g["t"], good, 1.0) - G.quantile(g["t"], good, 0.0)

    for th in (3, 5):
        out[f"n_low{th}"] = G.count(finite & (g["z"] <= -th))
        out[f"n_high{th}"] = G.count(finite & (g["z"] >= th))
    out["n_rej_low3"] = G.count(~good & np.isfinite(g["z"]) & (g["z"] <= -3))

    partner = _partners(g["sid"], g["t"], g["fov"])
    pz = _take(g["z"], partner)
    for side, largest, cmp in (("min", False, -2.0), ("max", True, 2.0)):
        i = G.arg(np.where(finite, g["z"], np.inf if not largest else -np.inf), finite, largest=largest)
        out[f"{side}_z_rob"] = _take(g["z"], i)
        out[f"{side}_z_err"] = _take(g["zerr"], i)
        out[f"{side}_frac"] = _take(g["frac"], i)
        out[f"{side}_partner_z"] = _take(pz, i)
        out[f"t_{side}"] = _take(g["t"], i)
        out[f"fov_{side}"] = _take(g["fov"], i, fill=0)
        out[f"row_{side}"] = _take(g["row"], i, fill=0)
        out[f"nobs_{side}"] = _take(g["nobs"], i, fill=0)
        # Same-transit BP and RP deviations, for dip colour.
        key = pd.DataFrame({"tid": _take(g["tid"], i, fill=-1), "sid": uniq})
        for code, name in ((1, "bp"), (2, "rp")):
            o = b[code]
            df = pd.DataFrame({"sid": o["sid"], "tid": o["tid"], "f": np.where(o["good"], o["frac"], np.nan),
                               "zz": np.where(o["good"], o["z"], np.nan)})
            j = key.merge(df.drop_duplicates(["sid", "tid"]), on=["sid", "tid"], how="left")
            out[f"{side}_frac_{name}"] = j["f"].to_numpy()
            out[f"{side}_z_{name}"] = j["zz"].to_numpy()
        # Of the faint (bright) 3-sigma points with a partner, the fraction whose partner agrees.
        sel = finite & ((g["z"] <= -3) if not largest else (g["z"] >= 3)) & (partner >= 0)
        agree = sel & (((pz <= cmp) if not largest else (pz >= cmp)) & np.isfinite(pz))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"partner_agree_{'low' if not largest else 'high'}"] = G.count(agree) / G.count(sel)

    low = finite & (g["z"] <= -3)
    rows = np.flatnonzero(low)
    new = np.ones(len(rows), bool)
    new[1:] = (g["sid"][rows][1:] != g["sid"][rows][:-1]) | (np.diff(g["t"][rows]) > EPISODE_GAP)
    out["n_low_episodes"] = np.bincount(G.inv[rows[new]], minlength=G.n)
    out["n_low3_fov2"] = G.count(low & (g["fov"] == 2))

    # Secular trend of G (fraction per year), from points within 5 robust sigma so dips don't drive it.
    core = finite & (np.abs(g["z"]) < 5)
    x = g["t"] / DAYS_PER_YEAR
    y = g["frac"]
    n = G.count(core).astype(float)
    sx, sy = G.sum(x, core), G.sum(y, core)
    sxx, sxy, syy = G.sum(x * x, core), G.sum(x * y, core), G.sum(y * y, core)
    with np.errstate(divide="ignore", invalid="ignore"):
        dxx = n * sxx - sx * sx
        slope = (n * sxy - sx * sy) / dxx
        icpt = (sy - slope * sx) / n
        rss = syy - 2 * icpt * sy - 2 * slope * sxy + n * icpt**2 + 2 * slope * icpt * sx + slope**2 * sxx
        slope_err = np.sqrt(np.maximum(rss, 0) / (n - 2) * n / dxx)
    out["slope_g"] = np.where(n >= 5, slope, np.nan)
    out["slope_err_g"] = np.where(n >= 5, slope_err, np.nan)

    for code, name in ((1, "bp"), (2, "rp")):
        o = b[code]
        fin = o["good"] & np.isfinite(o["z"])
        out[f"n_{name}"] = o["G"].count(o["good"])
        out[f"sigma_rob_{name}"] = o["sigma"] / o["med"]
        out[f"min_z_{name}_own"] = _take(o["z"], o["G"].arg(np.where(fin, o["z"], np.inf), fin))
        out[f"n_low3_{name}"] = o["G"].count(fin & (o["z"] <= -3))
        out[f"n_high3_{name}"] = o["G"].count(fin & (o["z"] >= 3))

    df = pd.DataFrame(out)
    df["control"] = control_mask(df["source_id"].to_numpy())
    return df


def keep_light_curve(feat):
    """Loose, symmetric rule for which full light curves to keep from the DR3 pass.

    Faint-side and bright-side criteria are mirrored so the kept set carries its own null, and random
    controls are always kept. Criteria are in robust-sigma units only: fractional depth cuts select
    nearly every faint source, and OLS trend significance is inflated for the (variable) DR3 sample, so
    both are left to the ranking stage, which works from the features. Keeps ~13% of sources.
    """
    f = feat
    faint = (f["min_z_rob"] <= -5) | (f["n_low3"] >= 3) | (f["n_rej_low3"] >= 1)
    bright = (f["max_z_rob"] >= 5) | (f["n_high3"] >= 3)
    return (faint | bright | f["control"]).to_numpy()


def features_schema_version():
    return "dr3-features-v0"


def to_arrow(df):
    return pa.Table.from_pandas(df, preserve_index=False)
