"""Release-day slow-change search over light curves: score every star, then flag it with rule slow-v3 (or with the
thresholds `release` sets from DR4's own stars, written in the same format).

    stats = score(lc, cells, table)   # one light-curve table at a time (a bulk file, a mirror fetch), in parallel
    flags = flag(stats, rule_dir)     # once every file is scored: the shared-date test needs whole sky patches
    summary(flags)                    # list sizes per magnitude range, and the mirror (getting brighter) beside them

`score` gives each star the detector's scores (`detect`), the period search's (`period`) and the summary DR4
publishes for every source in `all_source_photometry` (n, unit-weight variance and Abbé value of the uncorrected
G fluxes, and `select.slow_z` from them), so a run over every star also shows what the whole-sky selection would
have kept. `flag` adds the rising mirror of each list: getting brighter by the same scores and thresholds, the
false-alarm estimate the thresholds were set from. `strength` (each score over its threshold, the largest)
orders the lists; it is not a ranking by value.

`scripts/slow_search.py` runs the steps on DR4 (bulk files, a mirror or light-curve files) and on DR3 as a trial.
"""

import numpy as np
import pandas as pd

from cosmos_search.slow import detect, drift, period, rule_v3, select
from cosmos_search.slow.rule import mag_group

KEEP_FRACTION = 0.8   # light curves are kept when a score reaches this fraction of its threshold, either direction


def score(lc, cells, table, period_search=True):
    """Scores of every star in a light-curve table (epoch.SCHEMA) that has a cell (G and colour bin) and enough
    visits: detector and period-search scores on epochs corrected with the drift `table`, and the archive-side
    summary on the uncorrected G fluxes, as DR4 computes it."""
    ep = drift.normalise(lc)
    # Only this table's stars: joining against a whole-sky cell table (10^7 rows) costs seconds per call.
    pos = cells.index.get_indexer(pd.unique(ep["source_id"]))
    cells = cells.iloc[pos[pos >= 0]]
    ep = ep[ep["source_id"].isin(cells.index)]
    if not len(ep):
        return pd.DataFrame()
    ep = ep.assign(frac_corr=ep["frac"].to_numpy() - drift.drift_at(ep, cells, table))
    out = detect.star_features(ep)
    if not len(out):
        return out
    if period_search:
        out = out.merge(period.star_features(ep), on="source_id", how="left")
    a = select.epoch_summary(ep, col="frac").rename(columns={"n": "n_epochs", "uwv": "uwv_g", "abbe": "abbe_g"})
    out = out.merge(a, on="source_id", how="left")
    out["slow_z"] = select.slow_z(out["n_epochs"], out["uwv_g"], out["abbe_g"])
    return out


def thresholds_for(gbin, thr, thr_p):
    """Episode, trend and repeat thresholds for each star's magnitude range (arrays)."""
    groups = mag_group(gbin)
    th = pd.DataFrame(thr).T.reindex(groups)
    rep = pd.Series(thr_p).reindex(groups).to_numpy() if thr_p else np.full(len(groups), np.inf)
    return th["s"].to_numpy(), th["z"].to_numpy(), rep


def keep_mask(stats, thr, thr_p, fraction=KEEP_FRACTION):
    """Stars whose light curves are worth keeping: any score within `fraction` of its threshold, fading or
    getting brighter, so both the lists and their mirrors can be looked at again."""
    s_th, z_th, rep_th = thresholds_for(stats["gbin"], thr, thr_p)
    f = fraction
    m = (stats["s_fade"].to_numpy() > f * s_th) | (stats["s_rise"].to_numpy() > f * s_th) \
        | (np.abs(stats["slope_z"].to_numpy()) > f * z_th)
    for c in ("s_rep", "s_rep_bright"):
        if c in stats:
            m |= stats[c].to_numpy() > f * rep_th
    return np.nan_to_num(m, nan=False).astype(bool)


def flag(stats, rule_dir=rule_v3.RULE_DIR, shared_flags=None):
    """rule_v3.apply on every scored star, plus the mirror: `rising` (getting brighter by the episode score, its
    change date not shared, or by the trend), `rising_rep` (repeated brightening, which has no repeat conditions
    and so over-counts), `strength` (largest of each fading score over its threshold) and `rise_strength` (the same
    for getting brighter, repeated brightening included).

    stats: `score` output for whole sky patches (every scored star of a level-6 HEALPix pixel at once), joined
    with the cells (gbin)."""
    thr, thr_p, _ = rule_v3.load(rule_dir)
    sh = rule_v3.shared(stats) if shared_flags is None else shared_flags
    out = rule_v3.apply(stats, thr, thr_p, sh)
    s_th, z_th, rep_th = thresholds_for(stats["gbin"], thr, thr_p)
    s_rise, slope_z = stats["s_rise"].to_numpy(), stats["slope_z"].to_numpy()
    out["shared_rise"] = sh["shared_rise"].to_numpy()
    out["rising"] = ((s_rise > s_th) & ~out["shared_rise"].to_numpy()) | (slope_z > z_th)
    out["rising_rep"] = stats["s_rep_bright"].to_numpy() > rep_th if "s_rep_bright" in stats else False
    s_rise_ns = np.where(out["shared_rise"].to_numpy(), np.nan, s_rise)
    rep_b = stats["s_rep_bright"].to_numpy() if "s_rep_bright" in stats else np.full(len(stats), np.nan)
    out["strength"] = _strength(stats["s_fade"].to_numpy() / s_th, -slope_z / z_th, stats["s_rep"].to_numpy() / rep_th)
    out["rise_strength"] = _strength(s_rise_ns / s_th, slope_z / z_th, rep_b / rep_th)
    for c in ("slow_z", "slope", "dim_depth", "dim_t0", "dim_t1", "dim_visits", "depth_bp", "depth_bp_err",
              "depth_rp", "depth_rp_err", "rep_period", "rep_depth", "rep_band_z", "t_first", "t_last"):
        if c in stats:
            out[c] = stats[c].to_numpy()
    return out


def _strength(*ratios):
    with np.errstate(invalid="ignore", divide="ignore"):
        parts = np.vstack(ratios)
    return np.max(np.where(np.isfinite(parts), parts, -np.inf), axis=0)


STRENGTH_STEPS = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)


def tail(flags, steps=STRENGTH_STEPS):
    """Stars getting fainter and, as the false-alarm estimate, getting brighter, above each multiple of the
    thresholds (strength), per magnitude range. The thresholds are 0.1% points of quiet stars, so over 10^9
    stars the lists at strength 1 hold millions of chance passes; where the fading count stands clear of the
    rising one is where a list is worth reading."""
    out = {}
    for grp, d in flags.groupby(mag_group(flags["gbin"])):
        f, r = d["strength"].to_numpy(), d["rise_strength"].to_numpy()
        out[grp] = {str(x): {"fading": int(np.sum(f > x)), "rising": int(np.sum(r > x))} for x in steps}
    return out


def summary(flags):
    """Per magnitude range: stars scored, the fading list by score, the `early` list, and the rising mirror."""
    out = {}
    tails = tail(flags) if "rise_strength" in flags else {}
    for grp, d in flags.groupby(mag_group(flags["gbin"])):
        by = d.loc[d["flagged"], "by"]
        out[grp] = {
            "scored": int(len(d)),
            "fading": int(d["flagged"].sum()),
            "fading_share": float(d["flagged"].mean()),
            "fading_by": {k: int(by.str.contains(k).sum()) for k in ("episode", "trend", "repeat")},
            "fade_kinds": d.loc[d["flagged"] & d["by"].str.contains("episode"), "fade_kind"].value_counts().to_dict(),
            "early": int(d["early"].sum()),
            "rising": int(d["rising"].sum()),
            "rising_rep": int(np.sum(d["rising_rep"])),
            "shared_fade_set_aside": int(d["shared_fade"].sum()),
            "shared_rise_set_aside": int(d["shared_rise"].sum()),
            "above_strength": tails.get(grp, {}),
        }
    return out
