"""Whole-sky selection of slow-change candidates from DR4's per-source summary columns, before any light curve.

DR4's epoch photometry comes through DataLink only, so the light curves the detector (`detect`) needs can be
fetched for a selection of stars, not for all of them. `gaiadr4.all_source_photometry` publishes, for nearly
every source and band, the epoch count (`phot_g_n_cal`), the unit-weight variance of the fluxes
(`phot_g_flux_uwv`: excess scatter over the errors) and the Abbé value in flux space (`phot_g_abbe_value`).

For noise with variance sigma^2 plus a change slower than the gap between measurements with variance s^2,
consecutive differences hold only the noise, so abbe ~ sigma^2 / (sigma^2 + s^2) and uwv ~ (sigma^2 + s^2) /
err^2. Then

    slow_excess = uwv * (1 - abbe) ~ s^2 / err^2

is the variance of the slow part in units of the errors: near 0 for noise (whatever the error scale), large for a
slow change, and small for a variable that changes faster than Gaia's visits (high uwv, abbe ~ 1). Its noise
scatter falls as 1 / sqrt(n), so the score is `slow_z = sqrt(n) * slow_excess`. Thresholds come per G bin from
the stars themselves (a fixed share of each bin), because the error scale and the population change with G.
"""

import numpy as np
import pandas as pd

from cosmos_search.slow import drift


def epoch_summary(epochs, col="frac", band=0):
    """DR4-like summary per star from its epochs in one band: n, uwv (weighted variance over errors, n - 1
    degrees of freedom) and the Abbé value (von Neumann ratio / 2 of the time-ordered values)."""
    e = epochs[epochs["band"] == band].sort_values(["source_id", "t"], kind="stable")
    sid = e["source_id"].to_numpy()
    y = e[col].to_numpy()
    w = 1.0 / e["err_frac"].to_numpy() ** 2
    g = pd.Series(sid)
    n = g.groupby(sid).transform("size").to_numpy()
    sw = pd.Series(w).groupby(sid).transform("sum").to_numpy()
    mw = pd.Series(w * y).groupby(sid).transform("sum").to_numpy() / sw
    m = pd.Series(y).groupby(sid).transform("mean").to_numpy()
    same = np.r_[False, sid[1:] == sid[:-1]]
    d2 = np.where(same, np.r_[0.0, np.diff(y)] ** 2, 0.0)
    out = pd.DataFrame({"source_id": sid, "w_res2": w * (y - mw) ** 2, "res2": (y - m) ** 2, "d2": d2})
    s = out.groupby("source_id").sum()
    s["n"] = pd.Series(n, index=sid).groupby(level=0).first()
    with np.errstate(divide="ignore", invalid="ignore"):
        s["uwv"] = s["w_res2"] / (s["n"] - 1)
        s["abbe"] = s["n"] / (2 * (s["n"] - 1)) * s["d2"] / s["res2"]
    return s[["n", "uwv", "abbe"]].reset_index()


def slow_z(n, uwv, abbe):
    """Archive-side score for a slow change: sqrt(n) * uwv * (1 - abbe) (module docstring)."""
    return np.sqrt(np.asarray(n, float)) * np.asarray(uwv, float) * (1 - np.asarray(abbe, float))


def thresholds(df, share, by="gbin"):
    """slow_z threshold per G bin that `share` of the given stars exceed."""
    return df.groupby(by)["slow_z"].quantile(1 - share).rename("slow_z_min").reset_index()


COLUMNS = ("source_id, phot_g_mean_mag, bp_rp, phot_g_n_cal, phot_g_flux_uwv, phot_g_abbe_value, phot_bp_abbe_value, "
           "phot_rp_abbe_value, phot_g_harmonic_amplitude")
SLOW_Z = "SQRT(phot_g_n_cal) * phot_g_flux_uwv * (1 - phot_g_abbe_value)"


def sample_adql(n, table="gaiadr4.all_source_photometry", min_n=20):
    """A representative random sample of n sources (random_index_as is a random permutation of 0..N-1), from
    which the thresholds per G bin are set on DR4's own stars before the selection is run."""
    return (f"SELECT {COLUMNS}, {SLOW_Z} AS slow_z\nFROM {table}\n"
            f"WHERE random_index_as < {int(n)} AND phot_g_n_cal >= {min_n}")


def slice_ranges(level=3):
    """source_id ranges of every HEALPix pixel at `level` (nested), for running `adql` over the whole sky as many
    jobs. On DR3's gaia_source (1.8 x 10^9 rows) a query of this shape took 4-5 s per level-3 slice (768 slices,
    up to 5 x 10^6 rows each) and 2-4 s per level-5 slice, so job overhead dominates and level 3 suits."""
    width = 2 ** 35 * 4 ** (12 - level)
    return [(p * width, (p + 1) * width - 1) for p in range(12 * 4 ** level)]


def run_selection(tap, cuts, ranges, table="gaiadr4.all_source_photometry", min_n=20, workers=5, log=print,
                  query=None):
    """`adql` over each source_id range, `workers` jobs at a time (ESA allows 5 per user). Returns the selection
    (one row per star) and the ranges that failed after one retry, to run again. query(cuts, range) -> ADQL
    replaces `adql` (for a trial on a table with other columns)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    query = query or (lambda c, r: adql(c, table=table, min_n=min_n, source_id_range=r))

    def one(r):
        for attempt in range(2):
            try:
                t = tap.query(query(cuts, r), cache=False).to_pandas()
                t.columns = [c.lower() for c in t.columns]
                return t
            except Exception:  # noqa: BLE001
                if attempt:
                    raise

    parts, failed = [], []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one, r): r for r in ranges}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                parts.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                failed.append({"range": futures[fut], "error": repr(exc)[:300]})
            if k % 50 == 0 or k == len(ranges):
                log(f"[{k}/{len(ranges)}] {sum(len(p) for p in parts)} selected, {len(failed)} failed")
    sel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return sel, failed


def adql(cuts, table="gaiadr4.all_source_photometry", min_n=20, source_id_range=None):
    """DR4 query selecting stars whose slow_z exceeds their G bin's threshold (`cuts` from `thresholds`, one row
    per G bin; stars in a bin missing from `cuts` are not selected).

    source_id_range=(lo, hi) restricts to a slice of the sky (source_id // 2**35 is the level-12 HEALPix pixel),
    for running the whole sky as many smaller jobs. Column names follow the DR4 draft data model of 2026-06-26;
    check them against the released schema."""
    edges = drift.G_EDGES
    bins = "\n     OR ".join(
        f"(phot_g_mean_mag >= {edges[int(r.gbin)]:g} AND phot_g_mean_mag < {edges[int(r.gbin) + 1]:g} "
        f"AND {SLOW_Z} > {r.slow_z_min:.4g})" for r in cuts.sort_values("gbin").itertuples())
    rng = f"\n  AND source_id BETWEEN {source_id_range[0]} AND {source_id_range[1]}" if source_id_range else ""
    # The Gaia archive's ADQL has no CASE (tried on DR3, 2026-10-23), so each G bin is its own OR term.
    return f"""SELECT {COLUMNS}, {SLOW_Z} AS slow_z
FROM {table}
WHERE phot_g_n_cal >= {min_n}{rng}
  AND ({bins})"""
