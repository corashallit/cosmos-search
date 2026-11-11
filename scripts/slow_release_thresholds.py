"""Set the slow-change rule's thresholds from a sample of light curves (release day: DR4's own quiet stars).

    python scripts/slow_release_thresholds.py --lightcurves DIR --out prereg/rules/slow-v3-dr4 \\
        [--aux aux.parquet] [--sky 7] [--workers 4] [--check-against prereg/rules/slow-v3]

--lightcurves is a directory of light-curve Parquet files in epoch.SCHEMA: `slow_search.py pass --keep-lightcurves
all` on a share of the bulk files, or `epoch.mirror_epoch_photometry` / DataLink results written to disk. The
sample should be every star in a random set of sky patches, so that each star has the same chance of being in it
and its neighbours are there for the shared-date test and the trend reference. Each star's G and colour come from
its own light curve; --aux (source_id, phot_g_mean_mag, bp_rp) replaces them where a catalogue is at hand.

Steps (cosmos_search.slow.release, slow.summary):
1. quiet stars from the light curves, and their epochs for the drift fit (a --drift-share of them when the sample
   is larger than memory);
2. drift tables fitted on each half of the quiet stars, each correcting the other half's stars (and every other
   star by its own half); with --sky, sky offsets per HEALPix pixel at that level from the quiet stars of the
   other half (slow.sky);
3. the summary of every star (`summary.star_summary`: trend and episode scores, as the detector's), each trend
   measured against the median of nearby stars of similar brightness (`summary.apply_reference`), and the
   shared-date test;
4. the period search, only on the quiet stars whose bound on its score is among the largest (--period-share): the
   0.1% point needs only the top of the distribution, and the script checks that every star left out is bounded
   below the threshold found, widening the share until that holds;
5. the 0.1% points of the brightening mirrors per magnitude range, with the stars beyond them and bootstrap
   intervals; the rule files. --check-against prints the change from an existing rule's thresholds.
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cosmos_search.slow import drift, period, release, rule_v3, sky, summary
from cosmos_search.slow.rule import mag_group

_STATE = {}
EPOCH_COLUMNS = ["source_id", "band", "t", "frac", "err_frac"]


def _init(state):
    """Give a worker the state. A forked worker (Linux) has it already; one started as a new interpreter (macOS)
    imports this script again and would see it empty."""
    _STATE.update(state)


def _epochs(path):
    """Good epochs of one file (uncorrected) and its stars' magnitudes and cells."""
    return summary.corrected_epochs(pq.read_table(path), None, _STATE.get("cells"), marks=_STATE.get("marks", True),
                                    own=_STATE.get("own", True))


def _quiet(path):
    """Pass 1: quiet stars of one file, the epochs of a share of them for the drift fit, and every star's cell."""
    ep, stars = _epochs(path)
    q = release.quiet_ids(ep)
    use = q[summary.random_share(q, _STATE["drift_share"])]
    return ep.loc[ep["source_id"].isin(use), EPOCH_COLUMNS], q, stars[drift.CELL_COLUMNS]


def _corrected(path):
    """One file's epochs, drift-corrected by the other half's table (and sky-corrected)."""
    ep, stars = _epochs(path)
    cells, tables = stars[drift.CELL_COLUMNS], _STATE["tables"]
    half = drift.half(ep["source_id"].to_numpy())
    corr = np.zeros(len(ep))
    for h in (0, 1):
        m = half == h
        if m.any():
            corr[m] = drift.drift_at(ep[m], cells, tables[1 - h])
    ep = ep.assign(frac_corr=ep["frac"].to_numpy() - corr)
    if _STATE.get("sky") is not None:
        off, _ = sky.correct_by_halves(ep, ep["source_id"].unique(), _STATE["sky"])
        ep = ep.assign(frac_corr=ep["frac_corr"].to_numpy() - off)
    return ep, stars


def _score(path):
    """Pass 2: the summary of every star of one file."""
    return summary.star_summary(None, epochs=_corrected(path))


def _period(job):
    """Pass 3: period-search scores of the given stars of one file."""
    path, ids = job
    ep, _ = _corrected(path)
    ep = ep[ep["source_id"].isin(ids)]
    return period.star_features(ep) if len(ep) else pd.DataFrame()


def period_scores(pool, files, stats, quiet, share, start=0.08, log=print):
    """s_rep_bright (and the other period-search scores) for the quiet stars that decide its `share` point.

    Stars are taken in order of their bound (`summary`, rep_bound_bright), a `start` share of each magnitude range
    first; the share doubles until the threshold found is at or above the bound of every star left out, which
    makes it the threshold the period search on every quiet star would give."""
    q = stats[stats["source_id"].isin(quiet) & np.isfinite(stats["rep_bound_bright"].to_numpy())]
    grp = mag_group(q["gbin"])
    rank = q["rep_bound_bright"].groupby(grp).rank(ascending=False, pct=True).to_numpy()
    done = pd.DataFrame()
    frac = start
    while True:
        want = q.loc[rank <= frac, "source_id"].to_numpy()
        todo = np.setdiff1d(want, done["source_id"].to_numpy() if len(done) else [])
        parts = [x for x in pool.map(_period, [(f, todo) for f in files], chunksize=1) if len(x)] if len(todo) else []
        done = pd.concat([done, *parts], ignore_index=True)
        got = q[["source_id", "rep_bound_bright"]].merge(done[["source_id", "s_rep_bright"]], on="source_id", how="left")
        exact = True
        for g in np.unique(grp):
            m = grp == g
            x = np.nan_to_num(got.loc[m, "s_rep_bright"].to_numpy(float), nan=-np.inf)
            v = np.quantile(x, 1 - share)
            left = got.loc[m & got["s_rep_bright"].isna().to_numpy() & (rank > frac), "rep_bound_bright"]
            if len(left) and not v >= left.max():
                exact = False
        log(f"period search on {len(done)} of {len(q)} quiet stars (top {frac:.0%} by bound): "
            f"{'threshold exact' if exact else 'bound of the stars left out reaches the threshold, widening'}")
        if exact or frac >= 1:
            return done
        frac = min(1.0, 2 * frac)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lightcurves", required=True)
    ap.add_argument("--aux", help="optional catalogue magnitudes (source_id, phot_g_mean_mag, bp_rp)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sky", type=int, help="subtract sky offsets in HEALPix pixels at this level")
    ap.add_argument("--no-period", action="store_true", help="episode and trend thresholds only (faster)")
    ap.add_argument("--period-share", type=float, default=0.08,
                    help="share of quiet stars, largest bound first, the period search starts with")
    ap.add_argument("--drift-share", type=float, default=1.0,
                    help="random share of the quiet stars whose epochs fit the drift table")
    ap.add_argument("--no-marks", action="store_true",
                    help="ignore DR3's rejected-epoch marks, to read DR3 as DR4's bulk files arrive")
    ap.add_argument("--no-own-test", action="store_true",
                    help="skip our own bad-epoch test (drift.own_good); only to measure what it does")
    ap.add_argument("--share", type=float, default=release.SHARE)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check-against", help="an existing rule directory to compare with")
    ap.add_argument("--label", default="", help="what the sample is, for release.json")
    a = ap.parse_args(argv)

    files = [f for f in sorted(Path(a.lightcurves).glob("*.parquet")) if pq.read_metadata(f).num_rows][: a.limit or None]
    if a.aux:
        aux = pd.read_parquet(a.aux, columns=["source_id", "phot_g_mean_mag", "bp_rp"]).drop_duplicates("source_id")
        _STATE["cells"] = drift.star_cells(aux)
    _STATE.update(drift_share=a.drift_share, sky=a.sky, marks=not a.no_marks, own=not a.no_own_test)
    with Pool(a.workers, initializer=_init, initargs=(_STATE,)) as pool:
        loaded = pool.map(_quiet, files, chunksize=1)
    qep = pd.concat([x[0] for x in loaded], ignore_index=True)
    quiet = np.concatenate([x[1] for x in loaded])
    cells = pd.concat([x[2] for x in loaded])
    cells = cells[~cells.index.duplicated()]
    del loaded
    half = drift.half(qep["source_id"].to_numpy())
    _STATE["tables"] = {h: drift.fit(qep[half == h], cells) for h in (0, 1)}
    table = drift.fit(qep, cells)
    print(f"{len(files)} files, {len(cells)} stars, {len(quiet)} quiet, drift fitted on "
          f"{qep['source_id'].nunique()} of them", flush=True)
    del qep

    with Pool(a.workers, initializer=_init, initargs=(_STATE,)) as pool:  # started after the tables exist
        stats = pd.concat(pool.map(_score, files, chunksize=1), ignore_index=True)
        stats = stats[np.isfinite(stats["sigma_visit"].to_numpy())].reset_index(drop=True)
        ref = summary.reference_table(stats)
        stats = summary.apply_reference(stats, ref)
        sh = rule_v3.shared(stats)
        stats["s_rise_ns"] = np.where(sh["shared_rise"].to_numpy(), np.nan, stats["s_rise"].to_numpy())
        stats["shared_fade"], stats["shared_rise"] = sh["shared_fade"].to_numpy(), sh["shared_rise"].to_numpy()
        mirrors = dict(release.MIRRORS)
        if a.no_period:
            mirrors.pop("rep")
        else:
            rep = period_scores(pool, files, stats, quiet, a.share, a.period_share,
                                log=lambda s: print(s, flush=True))
            stats = stats.merge(rep, on="source_id", how="left") if len(rep) else stats.assign(s_rep_bright=np.nan)
    qs = stats[stats["source_id"].isin(quiet)]
    thr = release.thresholds(qs, mirrors, share=a.share)
    # The trend threshold the unreferenced significance would have given, for the record.
    raw = release.thresholds(qs, {"z_unreferenced": "slope_z_raw"}, share=a.share, n_boot=0)
    meta = {"rule": rule_v3.NAME, "label": a.label, "lightcurves": str(a.lightcurves), "files": len(files),
            "stars": len(stats), "quiet_stars": len(qs), "sky_level": a.sky, "mirrors": mirrors,
            "rejected_epoch_marks_used": not a.no_marks, "own_bad_epoch_test": not a.no_own_test,
            "trend": "against the median of nearby stars of similar brightness (summary.apply_reference)",
            "trend_unreferenced": {g: v["z_unreferenced"]["value"] for g, v in raw.items() if v},
            "trend_offset_by_G": summary.offset_by_magnitude(stats, "slope_z_raw").assign(
                g_mag=lambda d: d["g_mag"].astype(str)).to_dict("records"),
            "period_search_stars": int(stats["s_rep_bright"].notna().sum()) if "s_rep_bright" in stats else 0,
            "stars_needed_for_20pct": release.stars_needed(a.share, 0.2)}
    out = release.write_rule(a.out, thr, table, meta, share=a.share)
    stats.to_parquet(out / "stats.parquet", index=False)
    ref.to_parquet(out / "reference.parquet", index=False)
    print(json.dumps(thr, indent=1))
    if a.check_against:
        old = json.loads((Path(a.check_against) / "thresholds.json").read_text())
        old_p = Path(a.check_against) / "period_thresholds.json"
        old_p = json.loads(old_p.read_text()) if old_p.exists() else {}
        key = str(a.share)
        for g, v in thr.items():
            if g in old:
                was = {**old[g][key], "rep": old_p.get(g, {}).get(key, float("nan"))}
                print(g, {k: (round(was[k], 2), round(v[k]["value"], 2)) for k in ("s", "z", "rep") if k in v})


if __name__ == "__main__":
    main()
