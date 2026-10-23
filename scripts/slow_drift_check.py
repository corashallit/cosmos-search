"""Check the calibration-drift correction (cosmos_search.slow.drift) on DR3 light curves.

    python scripts/slow_drift_check.py --run /home/user/slow/dr3 \\
        --aux /mnt/project-files/data/gaia-dr3-epoch/aux/aux_gaia_source.parquet \\
        --known known.parquet --out slow_check/

--run is a dr3_pass output directory that kept every light curve (features/ and lightcurves/ per bulk file).
Quiet stars (drift.reference_mask) are split in two by source_id hash: the drift is fitted on half A only and
judged on half B, so no star's own epochs enter the correction applied to it. For half B the script reports,
per band and G bin, the median trend (the shared drift) and the spread of trends between stars, before and
after the correction, against the trend errors. A correction that works takes the median to ~0 and leaves the
spread no wider than the errors allow. Known answers with light curves in the run get their corrected trends.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cosmos_search.slow import drift


def mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def load(run, keep_ids=None):
    """Features for every source in the run, and normalised epochs of the reference stars (plus keep_ids)."""
    feats, eps = [], []
    for fp in sorted((Path(run) / "features").glob("*.parquet")):
        f = pq.read_table(fp, columns=["source_id", "n_g", "sigma_rob_g", "err_med_g", "min_z_rob",
                                       "max_z_rob"]).to_pandas()
        ref = f["source_id"].to_numpy()[drift.reference_mask(f)]
        want = np.union1d(ref, keep_ids) if keep_ids is not None else ref
        lc = pq.read_table(Path(run) / "lightcurves" / fp.name, filters=[("source_id", "in", list(map(int, want)))])
        f["reference"] = drift.reference_mask(f)
        feats.append(f)
        eps.append(drift.normalise(lc))
    return pd.concat(feats, ignore_index=True), pd.concat(eps, ignore_index=True)


def summarise(tr, cells, label):
    t = tr.join(cells, on="source_id")
    t["z"] = t[f"slope_{label}"] / t["slope_err"]
    rows = []
    for (band, g), d in t.groupby(["band", "gbin"]):
        if len(d) < 20:
            continue
        rows.append({"band": drift.BANDS[band], "g_bin": f"{drift.G_EDGES[g]:g}-{drift.G_EDGES[g + 1]:g}",
                     "n": len(d), f"median_{label}_pct_yr": 100 * float(d[f"slope_{label}"].median()),
                     f"spread_{label}_pct_yr": 100 * mad(d[f"slope_{label}"]),
                     "median_err_pct_yr": 100 * float(d["slope_err"].median()),
                     f"spread_z_{label}": mad(d["z"])})
    return pd.DataFrame(rows)


def sky_correlation(g, level=5, min_stars=10, n_perm=200, seed=0):
    """Is what's left of the G trends shared by stars close on the sky?

    Stars are grouped in HEALPix pixels (level 5, ~3.4 deg^2; the pixel is source_id // 2^35 // 4^(12-level)).
    The statistic is the variance of the pixel medians of slope z; the null shuffles z between stars, which
    keeps each pixel's star count. A ratio near 1 means no shared sky term left; well above 1 means neighbours
    share a trend the magnitude/colour drift did not remove.
    """
    d = g[np.isfinite(g["z"])].copy()
    d["pix"] = (d["source_id"].to_numpy() // 2**35) // 4 ** (12 - level)
    counts = d.groupby("pix")["z"].transform("size")
    d = d[counts >= min_stars]
    if d["pix"].nunique() < 5:
        return {"pixels": int(d["pix"].nunique())}
    obs = float(d.groupby("pix")["z"].median().var())
    rng = np.random.default_rng(seed)
    z = d["z"].to_numpy()
    null = [float(pd.Series(rng.permutation(z), index=d.index).groupby(d["pix"]).median().var())
            for _ in range(n_perm)]
    return {"pixels": int(d["pix"].nunique()), "stars": int(len(d)), "var_pixel_median_z": obs,
            "null_mean": float(np.mean(null)), "ratio": obs / float(np.mean(null)),
            "p_value": float((1 + np.sum(np.asarray(null) >= obs)) / (1 + n_perm))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--known")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    known = pd.read_parquet(a.known) if a.known else None
    kid = known["source_id"].dropna().astype("int64").to_numpy() if known is not None else None
    feat, ep = load(a.run, kid)
    aux = pd.read_parquet(a.aux, columns=["source_id", "phot_g_mean_mag", "bp_rp"])
    aux = aux[aux["source_id"].isin(feat["source_id"])]
    cells = drift.star_cells(aux)

    ref = feat.loc[feat["reference"], "source_id"].to_numpy()
    h = pd.Series(drift.half(ref), index=ref)
    a_ids, b_ids = h.index[h == 0], h.index[h == 1]
    table = drift.fit(ep[ep["source_id"].isin(a_ids)], cells)
    table.to_parquet(out / "drift_table.parquet", index=False)
    ep["frac_corr"] = ep["frac"] - drift.drift_at(ep, cells, table)

    eb = ep[ep["source_id"].isin(b_ids)]
    raw = drift.trend(eb, "frac").rename(columns={"slope": "slope_raw"})
    cor = drift.trend(eb, "frac_corr").rename(columns={"slope": "slope_corr", "slope_err": "slope_err_corr"})
    tr = raw.merge(cor[["source_id", "band", "slope_corr", "slope_err_corr"]], on=["source_id", "band"])
    s = summarise(tr, cells, "raw").merge(
        summarise(tr.assign(slope_err=tr["slope_err_corr"]), cells, "corr")
        .drop(columns=["n", "median_err_pct_yr"]), on=["band", "g_bin"])
    s.to_csv(out / "half_b_trends.csv", index=False)
    print(s.round(4).to_string(index=False))

    g = tr[tr["band"] == 0].copy()
    g["z"] = g["slope_corr"] / g["slope_err_corr"]
    sym = {f">{thr}%/yr": {"fades": int(((g["slope_corr"] < -thr / 100) & (g["z"] < -5)).sum()),
                           "brightenings": int(((g["slope_corr"] > thr / 100) & (g["z"] > 5)).sum())}
           for thr in (0.1, 0.3, 1.0)}
    print("half B, G band, |z| > 5:", sym)

    sky = sky_correlation(g)
    print("sky correlation of corrected G trends:", sky)
    # Is the shared sky term the same in the other half? Pixel medians of half A against half B.
    ta = drift.trend(ep[ep["source_id"].isin(a_ids) & (ep["band"] == 0)], "frac_corr")
    ta["z"] = ta["slope"] / ta["slope_err"]
    pix = lambda d: (d["source_id"].to_numpy() // 2**35) // 4 ** (12 - 5)
    ma = ta.assign(pix=pix(ta)).groupby("pix")["z"].agg(["median", "size"])
    mb = g.assign(pix=pix(g)).groupby("pix")["z"].agg(["median", "size"])
    both = ma.join(mb, lsuffix="_a", rsuffix="_b", how="inner")
    both = both[(both["size_a"] >= 10) & (both["size_b"] >= 10)]
    sky["halves_pixels"] = int(len(both))
    sky["halves_corr"] = float(np.corrcoef(both["median_a"], both["median_b"])[0, 1]) if len(both) > 4 else None
    print("pixel median z, half A vs half B:", sky["halves_pixels"], sky["halves_corr"])
    # Each pixel's median trend significance from half A, removed from half B.
    za = drift.sky_offsets(ta.drop(columns="z")).set_index("pix")["offset_z"]
    g3 = g.assign(z=g["z"].to_numpy() - za.reindex(drift.sky_pixel(g["source_id"])).fillna(0.0).to_numpy())
    sky["after_sky_offset_z"] = sky_correlation(g3)
    print("after z-unit sky offsets from half A:", sky["after_sky_offset_z"])

    res = {"n_sources": int(len(feat)), "n_reference": int(len(ref)), "n_half_a": int(len(a_ids)),
           "n_half_b": int(len(b_ids)), "drift_cells": int(len(table)), "fade_brighten_symmetry": sym,
           "sky_correlation": sky}
    if known is not None:
        kt = drift.trend(ep[ep["source_id"].isin(kid)], "frac_corr")
        kt = kt[kt["band"] == 0].merge(known[["name", "kind", "source_id"]].astype({"source_id": "int64"},
                                                                                    errors="ignore"),
                                       on="source_id")
        kt.to_csv(out / "known_trends.csv", index=False)
        res["known_with_lightcurves"] = int(kt["source_id"].nunique())
        print(kt.sort_values("slope").to_string(index=False))
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
