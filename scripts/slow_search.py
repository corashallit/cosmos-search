"""Release-day slow-change search, step by step (cosmos_search.slow.search; docs/arms/slow-changes.md, "Release day").

    python scripts/slow_search.py sample --out S --patches 300 --level 9 --schema gaiadr4
    python scripts/slow_release_thresholds.py --lightcurves S/lightcurves --aux S/aux.parquet --out RULE \\
        --check-against prereg/rules/slow-v3
    python scripts/slow_search.py score --lightcurves DIR --aux AUX --rule RULE --out R
    python scripts/slow_search.py flag --run R --aux AUX --rule RULE

`sample`: light curves of every star in random small sky patches (`release.patch_ranges`) from a mirror's
epoch-photometry table, with each star's G and BP-RP, for the release-day thresholds. `score`: every star of a
directory of light-curve files (a `dr4_bulk pull`, mirror fetches or DataLink results, in epoch.SCHEMA), or with
`--dr3-bulk` every star of DR3's bulk files streamed from ESA's CDN (the trial run); writes scores per file and keeps
the light curves of stars near a threshold, with done-markers so a second run resumes. `flag`: the shared-date test
over the whole run, then the fading list, the `early` list and their rising mirrors (`search.flag`), each ordered
by strength, and summary.json.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cosmos_search.gaia import dr3_pass, epoch
from cosmos_search.slow import drift, release, rule_v3, search

_STATE = {}
# What `flag` reads of the scores (the rest stays in the per-file stats).
STATS_COLUMNS = {"source_id", "gbin", "n_visits", "sigma_visit", "abbe", "s_fade", "s_rise", "fade_kind", "rise_kind",
                 "slope_z", "slope", "s_rep", "s_rep_bright", "rep_period", "rep_depth", "rep_band_z", "dim_t0",
                 "dim_t1", "dim_visits", "dim_depth", "depth_bp", "depth_bp_err", "depth_rp", "depth_rp_err",
                 "rise_t0", "slow_z", "t_first", "t_last"}
LIST_COLUMNS = ["id", "phot_g_mean_mag", "bp_rp", "gbin", "by", "strength", "n_visits", "sigma_visit", "abbe",
                "s_fade", "fade_kind", "slope_z", "slope", "s_rep", "rep_period", "s_rise", "rise_kind", "slow_z",
                "dim_depth", "dim_t0", "dim_t1", "dim_visits", "depth_bp", "depth_bp_err", "depth_rp",
                "depth_rp_err", "rep_depth", "rep_band_z", "t_first", "t_last"]


# -- sample -----------------------------------------------------------------------------------------------------

def _split_fetch(fetch, lo, hi, depth=3):
    """fetch(lo, hi), and if the service refuses (dense fields can exceed a synchronous query's limits: AIP answered
    400 on 8 of 1,000 DR3 level-5 patches), the same range as four quarters, down to `depth` levels."""
    try:
        return [fetch(lo, hi)]
    except Exception:  # noqa: BLE001
        if depth == 0:
            raise
    q = (hi - lo + 1) // 4
    return [x for k in range(4) for x in _split_fetch(fetch, lo + k * q, hi if k == 3 else lo + (k + 1) * q - 1,
                                                      depth - 1)]


def _fetch_patch(job):
    i, (lo, hi), a = job
    from cosmos_search.net.tap import Tap

    t0 = time.time()
    lcs = _split_fetch(lambda x, y: epoch.mirror_epoch_photometry(
        source_id_range=(x, y), mirror=a["mirror"], table=f"{a['schema']}.epoch_photometry"), lo, hi)
    lc = pa.concat_tables(lcs)
    ids = np.unique(np.asarray(lc["source_id"]))
    aux = pd.DataFrame(columns=["source_id", "phot_g_mean_mag", "bp_rp"])
    if len(ids):
        tap = Tap(epoch.MIRRORS.get(a["aux_mirror"], a["aux_mirror"]))
        aux = pd.concat(_split_fetch(lambda x, y: tap.query(
            f"SELECT source_id, phot_g_mean_mag, bp_rp FROM {a['schema']}.gaia_source WHERE source_id BETWEEN {x} AND "
            f"{y} AND has_epoch_photometry = 'True'", sync=True, maxrec=10**7).to_pandas(), lo, hi), ignore_index=True)
    return i, lc, aux, {"patch": i, "lo": lo, "hi": hi, "stars": int(len(ids)), "pieces": len(lcs),
                        "seconds": round(time.time() - t0, 1)}


def sample(a):
    out = Path(a.out)
    (out / "lightcurves").mkdir(parents=True, exist_ok=True)
    ranges = release.patch_ranges(a.patches, level=a.level, seed=a.seed)
    (out / "patches.json").write_text(json.dumps({"level": a.level, "seed": a.seed, "ranges": ranges}))
    done = {int(p.stem.split("_")[1]) for p in (out / "lightcurves").glob("patch_*.parquet")}
    jobs = [(i, r, vars(a)) for i, r in enumerate(ranges) if i not in done]
    print(f"{len(ranges)} patches at level {a.level}, {len(jobs)} to fetch", flush=True)
    t0, n_stars = time.time(), 0
    with ProcessPoolExecutor(a.workers) as pool, (out / "manifest.jsonl").open("a") as fh:
        futures = {pool.submit(_fetch_patch, j): j[0] for j in jobs}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                i, lc, aux, rec = fut.result()
            except Exception as exc:  # noqa: BLE001 (retried on the next run)
                fh.write(json.dumps({"patch": futures[fut], "error": repr(exc)[:300]}) + "\n")
                continue
            aux.to_parquet(out / "lightcurves" / f"aux_{i:05d}.parquet.part", index=False)
            pq.write_table(lc, out / "lightcurves" / f"patch_{i:05d}.parquet", compression="zstd")
            n_stars += rec["stars"]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 50 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] {n_stars} stars, {n_stars / (time.time() - t0):.0f} light curves/s", flush=True)
    parts = sorted((out / "lightcurves").glob("aux_*.parquet.part"))
    aux = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).drop_duplicates("source_id")
    aux.to_parquet(out / "aux.parquet", index=False)
    print(f"aux: {len(aux)} stars")


# -- select -----------------------------------------------------------------------------------------------------

# DR3 has no all_source_photometry: a trial runs the same query shape on gaia_source with a stand-in score, to time
# the archive (the stars it picks mean nothing).
DR3_STANDIN = "SQRT(phot_g_n_obs) * phot_g_mean_flux_error / phot_g_mean_flux"


def select_cmd(a):
    from cosmos_search.gaia.archive import tap
    from cosmos_search.slow import select

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = tap()
    t0 = time.time()
    if a.dr3_standin:
        smp = t.query(f"SELECT source_id, phot_g_mean_mag, bp_rp, {DR3_STANDIN} AS slow_z FROM gaiadr3.gaia_source "
                      f"WHERE random_index < {a.sample} AND phot_g_n_obs >= 20", cache=False).to_pandas()
    else:
        smp = t.query(select.sample_adql(a.sample, table=a.table), cache=False).to_pandas()
    smp.columns = [c.lower() for c in smp.columns]
    smp["gbin"] = drift.star_cells(smp)["gbin"].to_numpy()
    cuts = select.thresholds(smp, a.share)
    cuts.to_csv(out / "cuts.csv", index=False)
    print(f"sample {len(smp)} stars in {time.time() - t0:.0f} s; cuts per G bin:\n{cuts}", flush=True)

    query = None
    if a.dr3_standin:
        def query(c, r):
            e = drift.G_EDGES
            bins = "\n     OR ".join(
                f"(phot_g_mean_mag >= {e[int(x.gbin)]:g} AND phot_g_mean_mag < {e[int(x.gbin) + 1]:g} "
                f"AND {DR3_STANDIN} > {x.slow_z_min:.4g})" for x in c.sort_values("gbin").itertuples())
            return (f"SELECT source_id, phot_g_mean_mag, bp_rp, {DR3_STANDIN} AS slow_z FROM gaiadr3.gaia_source\n"
                    f"WHERE phot_g_n_obs >= 20 AND source_id BETWEEN {r[0]} AND {r[1]}\n  AND ({bins})")
    ranges = select.slice_ranges(a.level)[: a.limit or None]
    t0 = time.time()
    sel, failed = select.run_selection(t, cuts, ranges, table=a.table, workers=a.workers, query=query,
                                       log=lambda s: print(s, flush=True))
    el = time.time() - t0
    sel.to_parquet(out / "selection.parquet", index=False)
    rec = {"slices": len(ranges), "level": a.level, "selected": int(len(sel)), "failed": failed,
           "seconds": round(el, 1), "share": a.share, "sample": int(len(smp))}
    (out / "selection.json").write_text(json.dumps(rec, indent=1))
    print(f"{len(sel)} stars selected from {len(ranges)} slices in {el / 60:.1f} min, {len(failed)} failed")


# -- score ------------------------------------------------------------------------------------------------------

def _cells(aux_path):
    if aux_path not in _STATE:
        aux = pd.read_parquet(aux_path, columns=["source_id", "phot_g_mean_mag", "bp_rp"])
        _STATE[aux_path] = drift.star_cells(aux)
    return _STATE[aux_path]


def _chunks(item):
    """Light-curve tables of one input: a DR3 bulk file streamed from the CDN, or a light-curve Parquet file."""
    if "url" in item:
        from cosmos_search.net import get_client

        with get_client().streamed(item["url"], md5=item.get("md5")) as path:
            yield from epoch.iter_epoch_csv(path, chunk_rows=5000)
    else:
        f = pq.ParquetFile(item["path"])
        yield f.read()


def score_one(item, a):
    """Score every star of one input; write its scores, the light curves near a threshold and a done-marker."""
    t0 = time.time()
    out = Path(a["out"])
    cells = _cells(a["aux"])
    thr, thr_p, table = rule_v3.load(a["rule"])
    stats, kept, n_rows, t_read = [], [], 0, 0.0
    t_read0 = time.time()
    for lc in _chunks(item):
        t_read += time.time() - t_read0
        n_rows += lc.num_rows
        s = search.score(lc, cells, table, period_search=not a["no_period"])
        if len(s):
            s = s.join(cells, on="source_id")
            keep = s.loc[search.keep_mask(s, thr, thr_p), "source_id"].to_numpy()
            kept.append(lc.filter(pc.is_in(lc["source_id"], value_set=pa.array(keep, pa.int64()))))
            stats.append(s)
        t_read0 = time.time()
    name = item["name"]
    st = pd.concat(stats, ignore_index=True) if stats else pd.DataFrame()
    for kind, obj in (("stats", st), ("lightcurves", kept)):
        (out / kind).mkdir(parents=True, exist_ok=True)
    tmp = out / "stats" / f"{name}.parquet.tmp"
    st.to_parquet(tmp, index=False)
    tmp.replace(out / "stats" / f"{name}.parquet")
    lcs = pa.concat_tables(kept) if kept else epoch.SCHEMA.empty_table()
    tmp = out / "lightcurves" / f"{name}.parquet.tmp"
    pq.write_table(lcs, tmp, compression="zstd")
    tmp.replace(out / "lightcurves" / f"{name}.parquet")
    rec = {"name": name, "stars": int(len(st)), "kept": int(len(np.unique(np.asarray(lcs["source_id"])))),
           "rows": n_rows, "read_s": round(t_read, 1), "seconds": round(time.time() - t0, 1)}
    (out / "done").mkdir(exist_ok=True)
    (out / "done" / f"{name}.json").write_text(json.dumps(rec) + "\n")
    return rec


def score(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.dr3_bulk:
        items = [{"name": dr3_pass.stem(f["name"]), "url": f["url"], "md5": f.get("md5"), "size": f.get("size")}
                 for f in epoch.bulk_files() if dr3_pass.in_shard(f["name"], a.shard)]
    else:
        items = [{"name": p.stem, "path": str(p)} for p in sorted(Path(a.lightcurves).glob("*.parquet"))
                 if dr3_pass.in_shard(p.name, a.shard)]
    todo = [it for it in items if not (out / "done" / f"{it['name']}.json").exists()][: a.limit or None]
    # Largest first, so the long files do not finish last on one worker.
    todo.sort(key=lambda it: -(it.get("size") or 0))
    print(f"{len(items)} inputs, {len(todo)} to score", flush=True)
    opts = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "func"}
    t0, n, failures = time.time(), 0, 0
    with ProcessPoolExecutor(a.workers) as pool, (out / "manifest.jsonl").open("a") as fh:
        futures = {pool.submit(score_one, it, opts): it for it in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
                n += rec["stars"]
            except Exception as exc:  # noqa: BLE001 (the input is retried on the next run)
                failures += 1
                rec = {"name": futures[fut]["name"], "error": repr(exc)[:500]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 25 == 0 or "error" in rec or k == len(todo):
                el = time.time() - t0
                print(f"[{k}/{len(todo)}] {n} stars in {el / 60:.1f} min ({n / el:.0f}/s) {rec}", flush=True)
    return failures


# -- flag -------------------------------------------------------------------------------------------------------

def flag(a):
    run = Path(a.run)
    out = Path(a.out or run)
    parts = []
    for p in sorted((run / "stats").glob("*.parquet")):
        names = pq.read_schema(p).names
        d = pd.read_parquet(p, columns=[c for c in names if c in STATS_COLUMNS])
        if len(d):
            parts.append(d.astype({c: "category" for c in ("fade_kind", "rise_kind") if c in d}))
    stats = pd.concat(parts, ignore_index=True)
    for c in ("fade_kind", "rise_kind"):  # categories differ between files; strings cost ~1 GB at 10^7 stars
        stats[c] = stats[c].astype(str).astype("category")
    aux = pd.read_parquet(a.aux, columns=["source_id", "phot_g_mean_mag", "bp_rp"])
    print(f"{len(stats)} stars scored in {len(parts)} files", flush=True)
    t0 = time.time()
    flags = search.flag(stats, a.rule)
    print(f"flagged in {time.time() - t0:.0f} s", flush=True)
    flags = flags.merge(aux.rename(columns={"source_id": "id"}), on="id", how="left")
    cols = [c for c in LIST_COLUMNS if c in flags]
    lists = {"fading": flags["flagged"], "early": flags["early"], "rising": flags["rising"]}
    for name, m in lists.items():
        flags.loc[m, cols].sort_values("strength" if name == "fading" else "id", ascending=name != "fading") \
            .to_parquet(out / f"{name}.parquet", index=False)
    flags[["id", "gbin", "flagged", "early", "rising", "rising_rep", "shared_fade", "shared_rise", "by",
           "strength"]].to_parquet(out / "flags.parquet", index=False)
    summ = {"rule": str(a.rule), "stars": int(len(flags)), "by_magnitude": search.summary(flags)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def _shard(text):
    i, n = (int(x) for x in text.split("/"))
    return i, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="light curves and G, BP-RP of every star in random sky patches, from a mirror")
    s.add_argument("--out", required=True)
    s.add_argument("--patches", type=int, default=300)
    s.add_argument("--level", type=int, default=9, help="HEALPix level of the patches (9: ~0.013 deg^2)")
    s.add_argument("--seed", type=int)
    s.add_argument("--schema", default="gaiadr4")
    s.add_argument("--mirror", default="aip", help="light-curve TAP service (a name in epoch.MIRRORS, or a URL)")
    s.add_argument("--aux-mirror", default="aip", help="TAP service for gaia_source")
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(func=sample)
    e = sub.add_parser("select", help="whole-sky selection on the archive's per-source summary (slow_z per G bin)")
    e.add_argument("--out", required=True)
    e.add_argument("--share", type=float, default=0.001, help="share of stars selected in each G bin")
    e.add_argument("--sample", type=int, default=2_000_000, help="random stars that set the cuts")
    e.add_argument("--level", type=int, default=3, help="HEALPix level of the source_id slices")
    e.add_argument("--workers", type=int, default=5, help="jobs at once (ESA allows 5 per user)")
    e.add_argument("--table", default="gaiadr4.all_source_photometry")
    e.add_argument("--limit", type=int, default=0, help="first N slices only")
    e.add_argument("--dr3-standin", action="store_true", help="trial on DR3's gaia_source with a stand-in score")
    e.set_defaults(func=select_cmd)
    c = sub.add_parser("score", help="score every star of light-curve files (or DR3's bulk files)")
    c.add_argument("--lightcurves", help="directory of light-curve Parquet files (epoch.SCHEMA)")
    c.add_argument("--dr3-bulk", action="store_true", help="stream DR3's bulk files from ESA's CDN instead")
    c.add_argument("--aux", required=True, help="Parquet with source_id, phot_g_mean_mag, bp_rp")
    c.add_argument("--rule", default=str(rule_v3.RULE_DIR), help="rule directory (drift table; thresholds for "
                                                                   "which light curves to keep)")
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=os.cpu_count())
    c.add_argument("--shard", type=_shard, default=(0, 1))
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--no-period", action="store_true")
    c.set_defaults(func=score)
    f = sub.add_parser("flag", help="shared-date test, lists and summary over a scored run")
    f.add_argument("--run", required=True)
    f.add_argument("--aux", required=True)
    f.add_argument("--rule", default=str(rule_v3.RULE_DIR))
    f.add_argument("--out")
    f.set_defaults(func=flag)
    a = ap.parse_args(argv)
    if a.cmd == "score" and not (a.lightcurves or a.dr3_bulk):
        ap.error("score needs --lightcurves or --dr3-bulk")
    r = a.func(a)
    return 1 if isinstance(r, int) and r else 0


if __name__ == "__main__":
    sys.exit(main())
