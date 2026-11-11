"""Which stars to fetch first when light curves only come one star at a time.

ESA's DR4 page names DataLink as the way to epoch photometry, and DataLink gives 0.3 to 1.5 light curves a second:
26,000 to 130,000 stars a day against 2 x 10^9 in the release. If the bulk files are late or absent, the search
cannot read every star (`summary`), so the order of fetching decides what it can find. Three lists, in order:

1. `watch`: stars DR3 already showed fading at its end (the DR3 trial's fading list: G < 17, a fade still under
   way, strength above 3, no eclipsing, pulsating or AGN class). Each carries a prediction written before DR4
   (`predict`), so DR4 confirms or refutes it star by star: the first hour's check of the whole chain.
2. `known`: the known answers (`known`: Boyajian's star, fading Gaia alerts, R CrB, UX Ori and DY Per stars).
3. `abbe`: on release day, stars brighter than a limit whose G Abbe value in `all_source_photometry` is in the
   lowest share of their magnitude bin (`abbe_adql`). The Abbe value alone is the right archive-side cut for
   smooth changes: on DR3's Andromeda field a 1% share picked most steady fades of 1% a year on quiet stars,
   where the earlier `select.slow_z` cut at 0.1% picked none up to 3% a year (docs/arms/slow-changes.md).

`scripts/slow_fallback.py` writes the lists.
"""

import numpy as np
import pandas as pd

from cosmos_search.slow import drift

DR4_END_BTJD = 1868.5          # 2020-01-20, the last DR4 observation, in the light curves' time (JD - 2457000)
YEAR = 365.25
ABBE_COLUMNS = "source_id, phot_g_mean_mag, bp_rp, phot_g_n_cal, phot_g_abbe_value"


def predict(watch, t_end=DR4_END_BTJD):
    """What DR4 should show for each DR3 watch-list star if its fade went on, from its DR3 trend (`slope`,
    fraction of flux a year; `slope_z`; `t_first`, `t_last`):

    - pred_slope: the same trend over DR4's span;
    - pred_change_since_dr3: the level at DR4's end against DR3's last visits, as a fraction (negative: fainter);
    - pred_slope_z: the trend's significance over DR4's span at the same scatter and cadence, which grows as the
      span to the power 1.5;
    - prediction: the same in words.

    `outcome` compares these with DR4's trend."""
    s, z = watch["slope"].to_numpy(float), watch["slope_z"].to_numpy(float)
    t0, t1 = watch["t_first"].to_numpy(float), watch["t_last"].to_numpy(float)
    years = (t_end - t1) / YEAR
    out = pd.DataFrame({"pred_slope": s, "pred_change_since_dr3": s * years,
                        "pred_slope_z": z * ((t_end - t0) / np.maximum(t1 - t0, 1.0)) ** 1.5}, index=watch.index)
    kind = watch["fade_kind"].to_numpy() if "fade_kind" in watch else np.full(len(watch), "")
    out["prediction"] = [
        (f"keeps fading at {-100 * a:.1f}% a year: {-100 * c:.1f}% fainter at DR4's end than at DR3's"
         if a < 0 else "DR3 trend not downward: the DR3 episode should still show in DR4's first three years")
        + ("; the drop DR3 saw is not recovered" if k == "drop" else "")
        for a, c, k in zip(s, out["pred_change_since_dr3"], kind)]
    return out


def outcome(pred_slope, slope, slope_err):
    """DR4's verdict on a watch-list prediction, from DR4's trend and its error: "continued" (at least half the
    predicted rate, and that far from zero by 3 errors), "slowed" (same sign, 3 errors from zero, under half the
    rate), "stopped" (within 3 errors of zero) or "reversed"."""
    p, s, e = (np.asarray(x, float) for x in (pred_slope, slope, slope_err))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = s / e
        same = np.sign(s) == np.sign(p)
        out = np.where(np.abs(z) < 3, "stopped", np.where(~same, "reversed",
                       np.where(np.abs(s) >= 0.5 * np.abs(p), "continued", "slowed")))
    return np.where(np.isfinite(z) & np.isfinite(p), out, "")


def abbe_sample_adql(n, table="gaiadr4.all_source_photometry", min_n=20, g_max=16.0):
    """A random sample of sources brighter than g_max (random_index_as is a random permutation), from which the
    Abbe cut per G bin is set on DR4's own stars."""
    return (f"SELECT {ABBE_COLUMNS}\nFROM {table}\n"
            f"WHERE random_index_as < {int(n)} AND phot_g_n_cal >= {min_n} AND phot_g_mean_mag < {g_max:g}")


def abbe_cuts(sample, share=0.01):
    """The G Abbe value below which `share` of the sample's stars lie, per G bin: gbin, abbe_max."""
    d = sample.assign(gbin=drift.star_cells(sample)["gbin"].to_numpy())
    return d.groupby("gbin")["phot_g_abbe_value"].quantile(share).rename("abbe_max").reset_index()


def abbe_adql(cuts, table="gaiadr4.all_source_photometry", min_n=20, source_id_range=None):
    """Stars whose G Abbe value is below their G bin's cut (`abbe_cuts`), optionally in a source_id range (a slice
    of the sky, `select.slice_ranges`). Column names follow the DR4 draft data model of 2026-06-26."""
    e = drift.G_EDGES
    bins = "\n     OR ".join(
        f"(phot_g_mean_mag >= {e[int(c.gbin)]:g} AND phot_g_mean_mag < {e[int(c.gbin) + 1]:g} "
        f"AND phot_g_abbe_value < {c.abbe_max:.5g})" for c in cuts.sort_values("gbin").itertuples())
    where = f"phot_g_n_cal >= {min_n}"
    if source_id_range is not None:
        where += f" AND source_id BETWEEN {int(source_id_range[0])} AND {int(source_id_range[1])}"
    return f"SELECT {ABBE_COLUMNS}\nFROM {table}\nWHERE {where}\n  AND ({bins})"


def ordered(watch=None, known=None, abbe=None):
    """One fetch order from the three lists: source_id, list, order, phot_g_mean_mag and each list's own columns.
    Watch-list stars brightest first, known answers as given, Abbe stars by how far below their bin's cut they
    are (smoothest first). A star keeps its first place."""
    parts = []
    if watch is not None and len(watch):
        w = pd.concat([watch, predict(watch)], axis=1).sort_values("phot_g_mean_mag")
        parts.append(w.assign(list="watch"))
    if known is not None and len(known):
        k = known[known["source_id"].notna()].copy()
        k["source_id"] = k["source_id"].astype(np.int64)
        if "phot_g_mean_mag" not in k and "g_mag" in k:
            k = k.rename(columns={"g_mag": "phot_g_mean_mag"})
        parts.append(k.assign(list="known"))
    if abbe is not None and len(abbe):
        a = abbe.copy()
        if "abbe_max" in a:
            a = a.assign(abbe_ratio=a["phot_g_abbe_value"] / a["abbe_max"]).sort_values("abbe_ratio")
        parts.append(a.assign(list="abbe"))
    if not parts:
        return pd.DataFrame(columns=["source_id", "list", "order"])
    out = pd.concat(parts, ignore_index=True).drop_duplicates("source_id").reset_index(drop=True)
    out.insert(1, "order", np.arange(len(out)))
    return out
