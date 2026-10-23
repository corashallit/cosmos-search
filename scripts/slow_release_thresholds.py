"""Set the slow-change rule's thresholds from a sample of light curves (release day: DR4's own quiet stars).

    python scripts/slow_release_thresholds.py --lightcurves DIR --aux aux.parquet --out prereg/rules/slow-v3-dr4 \\
        [--sky 7] [--workers 4] [--check-against prereg/rules/slow-v3]

--lightcurves is a directory of light-curve Parquet files in epoch.SCHEMA (a dr3_pass run's lightcurves/, the DR4
bulk-file reader's output, or `epoch.mirror_epoch_photometry` / DataLink results written to disk). --aux has
source_id, phot_g_mean_mag and bp_rp for those stars. The sample should be every star in a random set of small
sky patches (`release.patch_ranges`), so that each star has the same chance of being in it and, with --sky, its
neighbours are there to correct it.

Steps (cosmos_search.slow.release): quiet stars from the light curves; drift tables fitted on each half of them,
each correcting the other half; with --sky, sky offsets per HEALPix pixel at that level from the quiet stars of
the other half (slow.sky); the detector and period-search scores of every quiet star; the 0.1% points of their
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

from cosmos_search.slow import detect, drift, period, release, sky

_STATE = {}


def _load(path):
    ep = drift.normalise(pq.read_table(path))
    q = release.quiet_ids(ep)
    return ep[ep["source_id"].isin(q)], q


def _score(job):
    ep = job
    out = detect.star_features(ep)
    if _STATE.get("period"):
        out = out.merge(period.star_features(ep), on="source_id", how="left")
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
    _STATE["period"] = not a.no_period

    files = sorted(Path(a.lightcurves).glob("*.parquet"))[: a.limit or None]
    with Pool(a.workers) as pool:
        loaded = pool.map(_load, files, chunksize=1)
    ep = pd.concat([x[0] for x in loaded], ignore_index=True)
    quiet = np.concatenate([x[1] for x in loaded])
    aux = pd.read_parquet(a.aux, columns=["source_id", "phot_g_mean_mag", "bp_rp"])
    aux = aux[aux["source_id"].isin(quiet)]
    cells = drift.star_cells(aux)
    ep = ep[ep["source_id"].isin(cells.index)]
    quiet = np.intersect1d(quiet, cells.index.to_numpy())
    print(f"{len(files)} files, {len(quiet)} quiet stars, {len(ep)} epochs")

    ep, table = release.cross_corrected(ep, cells, quiet)
    if a.sky is not None:
        off, _ = sky.correct_by_halves(ep, quiet, a.sky)
        ep = ep.assign(frac_corr=ep["frac_corr"].to_numpy() - off)

    # Score in chunks of whole stars.
    ep = ep.sort_values(["source_id", "band", "t"], kind="stable")
    ids = np.array_split(quiet, max(1, a.workers * 8))
    sid = ep["source_id"].to_numpy()
    chunks = [ep.iloc[np.searchsorted(sid, c[0], "left"):np.searchsorted(sid, c[-1], "right")] for c in ids if len(c)]
    with Pool(a.workers) as pool:
        stats = pd.concat(pool.map(_score, chunks, chunksize=1), ignore_index=True)
    stats = stats.join(cells, on="source_id")
    mirrors = dict(release.MIRRORS) if _STATE["period"] else {k: v for k, v in release.MIRRORS.items() if k != "rep"}
    thr = release.thresholds(stats, mirrors, share=a.share)
    meta = {"label": a.label, "lightcurves": str(a.lightcurves), "files": len(files), "quiet_stars": int(len(quiet)),
            "sky_level": a.sky, "mirrors": mirrors, "stars_needed_for_20pct": release.stars_needed(a.share, 0.2)}
    out = release.write_rule(a.out, thr, table, meta, share=a.share)
    stats.to_parquet(out / "quiet_stats.parquet", index=False)
    print(json.dumps(thr, indent=1))
    if a.check_against:
        old = json.loads((Path(a.check_against) / "thresholds.json").read_text())
        key = str(a.share)
        for g, v in thr.items():
            if g in old:
                print(g, {k: (round(old[g][key][k], 2), round(v[k]["value"], 2)) for k in ("s", "z") if k in v})


if __name__ == "__main__":
    main()
