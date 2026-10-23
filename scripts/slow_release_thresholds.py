"""Set the slow-change rule's thresholds from a sample of light curves (release day: DR4's own quiet stars).

    python scripts/slow_release_thresholds.py --lightcurves DIR --aux aux.parquet --out prereg/rules/slow-v3-dr4 \\
        [--sky 7] [--workers 4] [--check-against prereg/rules/slow-v3]

--lightcurves is a directory of light-curve Parquet files in epoch.SCHEMA (a dr3_pass run's lightcurves/, the DR4
bulk-file reader's output, or `epoch.mirror_epoch_photometry` / DataLink results written to disk). --aux has
source_id, phot_g_mean_mag and bp_rp for those stars. The sample should be every star in a random set of small
sky patches (`release.patch_ranges`), so that each star has the same chance of being in it and, with --sky, its
neighbours are there to correct it.

Steps (cosmos_search.slow.release): quiet stars from the light curves; drift tables fitted on each half of them,
each correcting the other half's stars (and every other star by its own half); with --sky, sky offsets per HEALPix pixel at that level from the quiet stars of
the other half (slow.sky); the detector scores of every star (for the shared-date test, rule_v3.shared) and the period-search scores of quiet
stars; the 0.1% points of their
brightening mirrors per magnitude range, with the stars beyond them and bootstrap intervals; the rule files.
--check-against prints the change from an existing rule's thresholds.
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cosmos_search.slow import detect, drift, period, release, rule_v3, sky

_STATE = {}


def _quiet(path):
    """Pass 1: quiet stars' epochs from one file."""
    ep = drift.normalise(pq.read_table(path))
    q = release.quiet_ids(ep)
    return ep[ep["source_id"].isin(q)], q


def _score(path):
    """Pass 2: every star of one file, drift-corrected by the other half's table (and sky-corrected), scored by
    the detector; quiet stars also by the period search."""
    cells, tables, quiet = _STATE["cells"], _STATE["tables"], _STATE["quiet"]
    ep = drift.normalise(pq.read_table(path))
    ep = ep[ep["source_id"].isin(cells.index)]
    half = drift.half(ep["source_id"].to_numpy())
    corr = np.zeros(len(ep))
    for h in (0, 1):
        m = half == h
        corr[m] = drift.drift_at(ep[m], cells, tables[1 - h])
    ep = ep.assign(frac_corr=ep["frac"].to_numpy() - corr)
    if _STATE.get("sky") is not None:
        off, _ = sky.correct_by_halves(ep, ep["source_id"].unique(), _STATE["sky"])
        ep = ep.assign(frac_corr=ep["frac_corr"].to_numpy() - off)
    out = detect.star_features(ep)
    if _STATE.get("period"):
        qe = ep[ep["source_id"].isin(quiet)]
        if len(qe):
            out = out.merge(period.star_features(qe), on="source_id", how="left")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lightcurves", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sky", type=int, help="subtract sky offsets in HEALPix pixels at this level")
    ap.add_argument("--no-period", action="store_true", help="detector thresholds only (faster)")
    ap.add_argument("--share", type=float, default=release.SHARE)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check-against", help="an existing rule directory to compare with")
    ap.add_argument("--label", default="", help="what the sample is, for release.json")
    a = ap.parse_args(argv)

    files = sorted(Path(a.lightcurves).glob("*.parquet"))[: a.limit or None]
    with Pool(a.workers) as pool:
        loaded = pool.map(_quiet, files, chunksize=1)
    qep = pd.concat([x[0] for x in loaded], ignore_index=True)
    quiet = np.concatenate([x[1] for x in loaded])
    aux = pd.read_parquet(a.aux, columns=["source_id", "phot_g_mean_mag", "bp_rp"])
    ids = set()
    for f in files:
        ids.update(pq.read_table(f, columns=["source_id"])["source_id"].to_numpy().tolist())
    cells = drift.star_cells(aux[aux["source_id"].isin(ids)])
    quiet = np.intersect1d(quiet, cells.index.to_numpy())
    qep = qep[qep["source_id"].isin(quiet)]
    half = drift.half(qep["source_id"].to_numpy())
    tables = {h: drift.fit(qep[half == h], cells) for h in (0, 1)}
    table = drift.fit(qep, cells)
    print(f"{len(files)} files, {len(cells)} stars, {len(quiet)} quiet")

    _STATE.update(cells=cells, tables=tables, quiet=set(map(int, quiet)), sky=a.sky, period=not a.no_period)
    with Pool(a.workers) as pool:
        stats = pd.concat(pool.map(_score, files, chunksize=1), ignore_index=True)
    stats = stats.join(cells, on="source_id")
    sh = rule_v3.shared(stats)
    stats["s_rise_ns"] = np.where(sh["shared_rise"].to_numpy(), np.nan, stats["s_rise"].to_numpy())
    stats["shared_fade"], stats["shared_rise"] = sh["shared_fade"].to_numpy(), sh["shared_rise"].to_numpy()
    qs = stats[stats["source_id"].isin(quiet)]
    mirrors = dict(release.MIRRORS) if not a.no_period else {k: v for k, v in release.MIRRORS.items() if k != "rep"}
    thr = release.thresholds(qs, mirrors, share=a.share)
    meta = {"rule": rule_v3.NAME, "label": a.label, "lightcurves": str(a.lightcurves), "files": len(files),
            "stars": int(len(stats)), "quiet_stars": int(len(qs)), "sky_level": a.sky, "mirrors": mirrors,
            "stars_needed_for_20pct": release.stars_needed(a.share, 0.2)}
    out = release.write_rule(a.out, thr, table, meta, share=a.share)
    stats.to_parquet(out / "stats.parquet", index=False)
    print(json.dumps(thr, indent=1))
    if a.check_against:
        old = json.loads((Path(a.check_against) / "thresholds.json").read_text())
        key = str(a.share)
        for g, v in thr.items():
            if g in old:
                print(g, {k: (round(old[g][key][k], 2), round(v[k]["value"], 2)) for k in ("s", "z") if k in v})


if __name__ == "__main__":
    main()
