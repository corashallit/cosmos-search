"""Release-day slow-change search over light curves: score every star, then flag it with rule slow-v3 (or with the
thresholds `release` sets from DR4's own stars, written in the same format).

    summ, stats, keep = score_all(lc, table, thr, thr_p)   # one light-curve table at a time (a chunk of a bulk file)
    flags = flag(merge(summ, stats), rule_dir)             # once every file is scored, one sky region at a time
    summary(flags)                    # list sizes per magnitude range, and the mirror (getting brighter) beside them

`score_all` summarises every star of the table (`summary.star_summary`: trend, episode scores and their windows,
the same numbers as the detector's), so the trend and episode parts of the rule are tested on every star at every
magnitude and no star is chosen before its light curve is read. The detector itself (`detect`, for the episode's
depth in each band) then runs only on stars within KEEP_FRACTION of a threshold, and the period search (`period`)
only on stars whose bound on its score (`summary`, rep_bound) allows a pass, down to PERIOD_G_MAX. Neither choice
can lose a star the rule would flag. `score` is the earlier form: detector and period search on every star given.

`flag` takes the trend measured against neighbours of similar brightness (`summary.apply_reference`) and adds the
rising mirror of each list: getting brighter by the same scores and thresholds, the false-alarm estimate the
thresholds were set from. `strength` (each score over its threshold, the largest) says how far past a threshold a
star is; reading order comes from `kinds.rank`, within kind of star.

`still` marks stars whose parallax and proper motion are both consistent with zero: on the DR3 trial run nearly
all were quasars (and DR3's AGN class nearly all still), which crowd the top of every list at G >= 19. They go to a
list of their own, read separately, since stars in other galaxies are still too.

`scripts/slow_search.py` runs the steps on DR4 (bulk files, a mirror or light-curve files) and on DR3 as a trial.
"""

import numpy as np
import pandas as pd

from cosmos_search.gaia import features
from cosmos_search.slow import detect, drift, period, rule_v3, select
from cosmos_search.slow import summary as per_star  # `summary` below is the lists' summary
from cosmos_search.slow.rule import mag_group

KEEP_FRACTION = 0.8   # light curves are kept when a score reaches this fraction of its threshold, either direction
PERIOD_G_MAX = 16.0   # the period search runs on stars brighter than this whose bound allows a pass
LC_FRACTION = 0.9     # light curves kept from this fraction of a threshold up, on the fading side
LC_G_MAX = 19.0       # and only for stars brighter than this
# G, BP and RP changes behind `checks.band_check` (dim_depth and the rest of the episode's are copied anyway).
BAND_COLUMNS = ("dim_depth_err", "rep_depth_err", "depth_g", "depth_g_err",
                *(f"{k}_{b}{x}" for k in ("slope", "rep_depth") for b in ("g", "bp", "rp") for x in ("", "_err")))
STILL_P = 1e-3        # parallax and proper motion consistent with zero at this level: `still`
MOTION_COLUMNS = ("parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error", "pmra_pmdec_corr",
                  "parallax_pmra_corr", "parallax_pmdec_corr")


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


def score_all(lc, table, thr, thr_p, cells=None, period_search=True, period_g_max=PERIOD_G_MAX, random_share=0.0,
              fraction=KEEP_FRACTION, keep_fraction=LC_FRACTION, keep_g_max=LC_G_MAX):
    """Every star of a light-curve table (epoch.SCHEMA, whole stars, ideally one compact patch of sky).

    Returns (summ, stats, keep):
    - summ: `summary.star_summary` for every star, with `slope_z_local`, the trend against the table's own stars
      of similar brightness (the whole run's reference is applied later, by `flag`), and `full`, whether the star
      is in stats;
    - stats: `score`'s columns for the stars that need more than the summary: those within `fraction` of the
      episode or trend threshold either way (by the unreferenced or the local trend), those brighter than
      period_g_max whose bound allows a repeat score past its threshold (the period search runs on these only),
      and a random share (`summary.random_share`). With them the archive-style summary of the uncorrected G
      epochs (n_g, uwv_g, abbe_g, slow_z) and the two bounds;
    - keep: source_ids whose light curves are worth keeping: stars brighter than keep_g_max within keep_fraction
      of a threshold on the fading side, and the random share."""
    ep, stars = per_star.corrected_epochs(lc, table, cells)
    summ = per_star.star_summary(lc, epochs=(ep, stars))
    if not len(summ):
        return summ.assign(slope_z_local=np.nan, full=False), pd.DataFrame(), np.array([], np.int64)
    summ["slope_z_local"] = per_star.apply_reference(summ, per_star.reference_table(summ))["slope_z"].to_numpy()
    s_th, z_th, rep_th = thresholds_for(summ["gbin"], thr, thr_p)
    z, zl, g = summ["slope_z"].to_numpy(), summ["slope_z_local"].to_numpy(), summ["g_mag"].to_numpy()
    with np.errstate(invalid="ignore"):
        near = (summ["s_fade"].to_numpy() > fraction * s_th) | (summ["s_rise"].to_numpy() > fraction * s_th) \
            | (np.abs(z) > fraction * z_th) | (np.abs(zl) > fraction * z_th)
        rep = (np.fmax(summ["rep_bound"].to_numpy(), summ["rep_bound_bright"].to_numpy()) > rep_th) \
            & (g < period_g_max) & bool(period_search)
    sid = summ["source_id"].to_numpy()
    scored = np.isfinite(summ["sigma_visit"].to_numpy())
    random = per_star.random_share(sid, random_share) & scored if random_share > 0 else np.zeros(len(sid), bool)
    need = (near | rep | random) & scored
    summ["full"] = need
    if not need.any():
        return summ, pd.DataFrame(), np.array([], np.int64)
    sub = ep[ep["source_id"].isin(sid[need])]
    stats = detect.star_features(sub)
    if rep.any():
        stats = stats.merge(period.star_features(sub[sub["source_id"].isin(sid[rep & scored])]), on="source_id",
                            how="left")
    extra = summ.loc[need, ["source_id", "gbin", "cbin", "g_mag", "bp_rp", "n_g", "uwv_g", "abbe_g",
                            "slope_z_local", "rep_bound", "rep_bound_bright"]]
    extra = extra.assign(slow_z=select.slow_z(extra["n_g"], extra["uwv_g"], extra["abbe_g"]))
    stats = stats.merge(extra, on="source_id", how="left")
    th = thresholds_for(stats["gbin"], thr, thr_p)
    with np.errstate(invalid="ignore"):
        fade = _strength(stats["s_fade"].to_numpy() / th[0], -stats["slope_z"].to_numpy() / th[1],
                         -stats["slope_z_local"].to_numpy() / th[1],
                         (stats["s_rep"].to_numpy() if "s_rep" in stats else np.full(len(stats), np.nan)) / th[2])
        keep = stats.loc[(fade > keep_fraction) & (stats["g_mag"].to_numpy() < keep_g_max), "source_id"].to_numpy()
    return summ, stats, np.union1d(keep, sid[random])


# Columns `flag` takes from the full scores when a star has them (the summary holds the rest for every star).
STATS_ONLY = ("s_rep", "s_rep_bright", "rep_period", "rep_depth", "rep_band_z", "dim_visits", "depth_bp",
              "depth_bp_err", "depth_rp", "depth_rp_err", "dim_depth_err", "rep_depth_err", "depth_g", "depth_g_err",
              *(f"rep_depth_{b}{x}" for b in ("g", "bp", "rp") for x in ("", "_err")))


def merge(summ, stats):
    """One row per star for `flag`: the summary of every star with the full scores' extra columns where a star has
    them (NaN elsewhere: a star without a period-search score cannot pass the repeat threshold)."""
    out = summ.copy()
    cols = [c for c in STATS_ONLY if stats is not None and c in stats]
    if cols:
        out = out.merge(stats[["source_id", *cols]], on="source_id", how="left")
    for c in ("s_rep", "s_rep_bright"):
        if c not in out:
            out[c] = np.nan
    return out


def flag(stats, rule_dir=rule_v3.RULE_DIR, shared_flags=None):
    """rule_v3.apply on every scored star, plus the mirror: `rising` (getting brighter by the episode score, its
    change date not shared, or by the trend), `rising_rep` (repeated brightening, which has no repeat conditions
    and so over-counts), `strength` (largest of each fading score over its threshold) and `rise_strength` (the same
    for getting brighter, repeated brightening included).

    stats: one row per star for whole sky patches (every star of a level-6 HEALPix pixel at once) with gbin:
    `merge(summ, stats)` with the neighbour reference applied (`summary.apply_reference`), or `score` output
    joined with the cells."""
    thr, thr_p, _ = rule_v3.load(rule_dir)
    sh = rule_v3.shared(stats) if shared_flags is None else shared_flags
    out = rule_v3.apply(stats, thr, thr_p, sh)
    s_th, z_th, rep_th = thresholds_for(stats["gbin"], thr, thr_p)
    s_rise, slope_z = stats["s_rise"].to_numpy(), stats["slope_z"].to_numpy()
    out["shared_rise"] = sh["shared_rise"].to_numpy()
    out["rising"] = ((s_rise > s_th) & ~out["shared_rise"].to_numpy()) | (slope_z > z_th)
    out["rising_rep"] = stats["s_rep_bright"].to_numpy() > rep_th if "s_rep_bright" in stats else False
    # Episodes past the threshold that the shared-date test set aside (shared_* alone also marks weak changes).
    out["fade_set_aside"] = (stats["s_fade"].to_numpy() > s_th) & out["shared_fade"].to_numpy()
    out["rise_set_aside"] = (s_rise > s_th) & out["shared_rise"].to_numpy()
    s_rise_ns = np.where(out["shared_rise"].to_numpy(), np.nan, s_rise)
    rep_b = stats["s_rep_bright"].to_numpy() if "s_rep_bright" in stats else np.full(len(stats), np.nan)
    fade_ratios = (stats["s_fade"].to_numpy() / s_th, -slope_z / z_th, stats["s_rep"].to_numpy() / rep_th)
    out["strength"] = _strength(*fade_ratios)
    out["strength_by"] = _strength_by(*fade_ratios)
    out["rise_strength"] = _strength(s_rise_ns / s_th, slope_z / z_th, rep_b / rep_th)
    for c in ("slow_z", "slope", "dim_depth", "dim_t0", "dim_t1", "dim_visits", "depth_bp", "depth_bp_err",
              "depth_rp", "depth_rp_err", "rep_period", "rep_depth", "rep_band_z", "t_first", "t_last", "rise_t0",
              "g_mag", "bp_rp", "slope_err", "slope_z_raw", "slope_ref", "slope_ref_err", "n_ref", "ref_level",
              "quiet", "abbe_bp", "abbe_rp") + BAND_COLUMNS:
        if c in stats:
            out[c] = stats[c].to_numpy()
    return out


def _strength_by(*ratios):
    """Which of episode, trend and repeat gives the strength (NaN ratios never win; "" when none is finite)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        parts = np.vstack(ratios)
    parts = np.where(np.isfinite(parts), parts, -np.inf)
    return np.where(np.isfinite(parts.max(axis=0)), np.array(["episode", "trend", "repeat"])[parts.argmax(axis=0)], "")


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
            "fade_set_aside_shared_date": int(d["fade_set_aside"].sum()),
            "rise_set_aside_shared_date": int(d["rise_set_aside"].sum()),
            "above_strength": tails.get(grp, {}),
        }
    return out


def motion(g):
    """chi2 of (parallax, pmra, pmdec) against zero with their correlations, and `still` (its 3-degree-of-freedom
    probability above STILL_P). g: gaia_source rows with MOTION_COLUMNS. Two-parameter solutions (no parallax or
    proper motion) get NaN and are not still.

    On the DR3 trial's 10,000 strongest fading stars at G >= 19, 9,665 of the 9,688 DPAC calls AGN were still and
    20 of the other 312; at G < 19, 2,367 of 2,376 and 15 of 7,624."""
    from scipy.stats import chi2

    cols = ["parallax", "pmra", "pmdec"]
    x = g[cols].to_numpy(float)
    e = g[[f"{c}_error" for c in cols]].to_numpy(float)
    r = {k: np.nan_to_num(g[k].to_numpy(float)) for k in ("parallax_pmra_corr", "parallax_pmdec_corr", "pmra_pmdec_corr")}
    cov = np.empty((len(g), 3, 3))
    for i in range(3):
        cov[:, i, i] = e[:, i] ** 2
    for (i, j), k in (((0, 1), "parallax_pmra_corr"), ((0, 2), "parallax_pmdec_corr"), ((1, 2), "pmra_pmdec_corr")):
        cov[:, i, j] = cov[:, j, i] = r[k] * e[:, i] * e[:, j]
    ok = np.isfinite(x).all(axis=1) & np.isfinite(e).all(axis=1)
    c2 = np.full(len(g), np.nan)
    if ok.any():
        c2[ok] = np.einsum("ni,ni->n", x[ok], np.linalg.solve(cov[ok], x[ok][..., None])[..., 0])
    with np.errstate(invalid="ignore"):
        still = ok & (chi2.sf(np.where(ok, c2, np.inf), 3) > STILL_P)
    return pd.DataFrame({"chi2_motion": c2, "still": still}, index=g.index)


def tail_moving(flags, still_ids, steps=STRENGTH_STEPS):
    """`tail` without the still stars."""
    return tail(flags[~flags["id"].isin(still_ids)], steps)
