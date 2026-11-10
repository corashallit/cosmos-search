"""Release-day slow-change search, step by step (cosmos_search.slow.search; docs/arms/slow-changes.md, "Release day").

    python scripts/slow_search.py pass --out S --dr4-bulk --files-share 0.01 --keep-lightcurves all
    python scripts/slow_release_thresholds.py --lightcurves S/lightcurves --out RULE \\
        --check-against prereg/rules/slow-v3
    python scripts/slow_search.py pass --out R --rule RULE --dr4-bulk [--shard i/n] [--durable DIR]
    python scripts/slow_search.py flag --run R --rule RULE --out L
    python scripts/slow_search.py cuts --run L
    python scripts/slow_checks.py --fading L/fading.parquet --out L/checks --skip-still --skip-band --skip-class ECL

`pass`: one pass over light curves, scoring every star in them. Inputs: DR4's bulk files on ESA's CDN (--dr4-bulk;
each is downloaded, read and deleted), DR3's (--dr3-bulk, the trial run), or a directory of light-curve Parquet
files in epoch.SCHEMA (--lightcurves: mirror fetches, DataLink results). Per input file it writes summary/ (one
row for every star: `summary.star_summary`, which holds the trend and episode scores), stats/ (the detector's and
period search's full scores for the stars that need them, `search.score_all`), lightcurves/ (stars near a
threshold and a random share) and a done-marker, so a second run resumes and machines split the files with
--shard. Without --rule it is the sample for the thresholds: with --files-share and --keep-lightcurves all it
keeps every light curve of a random share of the files, which are whole patches of sky.
`flag`: over the whole run, one sky region at a time: each star's trend against the median of nearby stars of
similar brightness (`summary.reference_table`), the shared-date test, then the fading list, the `early` list and
their rising mirrors (`search.flag`), summary.json, and reference.parquet (the median trend per sky pixel and
magnitude range, which should sit near zero).
`cuts`: parallax, proper motion and colour of the listed stars from the archive, setting aside those consistent
with zero motion (quasars, galaxies, stars in other galaxies) as `still`; whether each fade's G change agrees
with its BP and RP changes (`band_verdict`); the kind of each star and the reading order within kind
(`kinds.rank`). This is the run's only archive query, and it comes after the lists exist.
`sample` fetches the thresholds' sample from a mirror's epoch-photometry table instead of bulk files. `select`
is the earlier archive-side choice of stars, kept for a release without bulk files (scripts/slow_fallback.py).
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
from cosmos_search.slow import drift, kinds, release, rule_v3, search, summary

_STATE = {}
# What `flag` reads of each star's summary and, where a star has them, of its full scores.
SUMMARY_COLUMNS = ("source_id", "g_mag", "bp_rp", "n_visits", "t_first", "t_last", "sigma_visit", "abbe",
                   "slope", "slope_err", "s_fade", "fade_kind", "dim_t0", "dim_t1", "dim_depth", "s_rise",
                   "rise_kind", "rise_t0", "slope_g", "slope_g_err", "slope_bp", "slope_bp_err", "slope_rp",
                   "slope_rp_err", "abbe_bp", "abbe_rp", "quiet", "ref", "full")
LIST_COLUMNS = ["id", "phot_g_mean_mag", "bp_rp", "gbin", "by", "strength", "rise_strength", "n_visits", "sigma_visit", "abbe",
                "s_fade", "fade_kind", "slope_z", "slope", "slope_err", "slope_z_raw", "slope_ref", "slope_ref_err",
                "n_ref", "s_rep", "rep_period", "s_rise", "rise_kind", "slow_z",
                "dim_depth", "dim_t0", "dim_t1", "dim_visits", "depth_bp", "depth_bp_err", "depth_rp",
                "depth_rp_err", "rep_depth", "rep_band_z", "t_first", "t_last", "quiet", "strength_by",
                *search.BAND_COLUMNS]


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


# -- pass -------------------------------------------------------------------------------------------------------

def _cells(aux_path):
    """G and colour bins from a catalogue file (source_id, phot_g_mean_mag, bp_rp), when one is given."""
    if aux_path and aux_path not in _STATE:
        aux = pd.read_parquet(aux_path, columns=["source_id", "phot_g_mean_mag", "bp_rp"]).drop_duplicates("source_id")
        _STATE[aux_path] = drift.star_cells(aux)
    return _STATE.get(aux_path)


def _split(lc, stars):
    """A light-curve table as tables of about `stars` whole stars each, in source_id order."""
    sid = np.asarray(lc["source_id"])
    if len(sid) and not np.all(sid[1:] >= sid[:-1]):
        lc = lc.take(pa.array(np.argsort(sid, kind="stable")))
        sid = np.asarray(lc["source_id"])
    first = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    for k in range(0, len(first), stars):
        end = first[k + stars] if k + stars < len(first) else len(sid)
        yield lc.slice(first[k], end - first[k])


def _chunks(item, a):
    """Light-curve tables of whole stars from one input: a DR3 bulk file streamed from the CDN, a DR4 bulk file
    (downloaded to out/tmp and deleted after reading, or a local copy), or a light-curve Parquet file."""
    if item["kind"] == "dr3":
        from cosmos_search.net import get_client

        with get_client().streamed(item["url"], md5=item.get("md5")) as path:
            yield from epoch.iter_epoch_csv(path, chunk_rows=a["chunk_stars"])
    elif item["kind"] == "dr4":
        from cosmos_search.gaia import dr4, dr4_bulk

        def read(path):
            if hasattr(dr4_bulk, "iter_lightcurves"):  # bounded memory: whole stars, source_id order
                yield from dr4_bulk.iter_lightcurves(path, keep_all=True)
            else:
                rows = dr4_bulk.read_kept(path, keep_all=True)
                if rows.num_rows:
                    yield from _split(dr4.epoch_photometry_to_lc(rows), a["chunk_stars"])

        if item.get("local"):
            yield from read(item["local"])
        else:
            tmp = Path(a["out"]) / "tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            dr4_bulk.check_space(tmp, item.get("size"))
            with dr4_bulk.fetched(item["url"], tmp / f"{item['name']}.parquet", item.get("size"), item.get("md5"),
                                  a["streams"]) as path:
                yield from read(path)
    else:
        yield from _split(pq.read_table(item["path"]), a["chunk_stars"])


def pass_one(item, a):
    """Every star of one input: its summary, the full scores of the stars that need them, the light curves worth
    keeping, and a done-marker.

    Summary rows stored: every star brighter than --summary-g-max, and of the fainter ones those with full scores
    and a random --faint-share. `ref` marks the rows that are an unbiased sample of their magnitude (all of the
    bright ones, the random share of the faint ones): the neighbour reference is built from those only. The
    done-marker counts every star by G bin, stored or not."""
    t0 = time.time()
    out = Path(a["out"])
    cells = _cells(a.get("aux"))
    thr = thr_p = table = None
    if a["rule"]:
        thr, thr_p, table = rule_v3.load(a["rule"])
    summs, stats, kept, n_rows, t_read, n_period = [], [], [], 0, 0.0, 0
    n_stars, by_gbin = 0, np.zeros(len(drift.G_EDGES) - 1, np.int64)
    t_read0 = time.time()
    for lc in _chunks(item, a):
        t_read += time.time() - t_read0
        n_rows += lc.num_rows
        if a["rule"]:
            sm, st, keep = search.score_all(lc, table, thr, thr_p, cells=cells, period_search=not a["no_period"],
                                            period_g_max=a["period_g_max"], random_share=a["random_share"],
                                            keep_fraction=a["keep_fraction"], keep_g_max=a["keep_g_max"])
        else:
            sm, st, keep = summary.star_summary(lc, cells=cells).assign(full=False), pd.DataFrame(), np.array([], np.int64)
        if len(sm):
            n_stars += len(sm)
            scored = np.isfinite(sm["sigma_visit"].to_numpy())
            by_gbin += np.bincount(sm["gbin"].to_numpy()[scored], minlength=len(by_gbin))[:len(by_gbin)]
            g = sm["g_mag"].to_numpy()
            sm["ref"] = (g < a["summary_g_max"]) | (summary.random_share(sm["source_id"].to_numpy(), a["faint_share"])
                                                    & np.isfinite(g))
            summs.append(sm[sm["ref"] | sm["full"]])
        if len(st):
            stats.append(st)
            n_period += int(st["s_rep_any"].notna().sum()) if "s_rep_any" in st else 0
        if a["keep_lightcurves"] == "all":
            kept.append(lc)
        elif a["keep_lightcurves"] == "near" and len(keep):
            kept.append(lc.filter(pc.is_in(lc["source_id"], value_set=pa.array(keep, pa.int64()))))
        t_read0 = time.time()
    name = item["name"]
    paths = {k: out / k / f"{name}.parquet" for k in ("summary", "stats", "lightcurves")}
    for q in paths.values():
        q.parent.mkdir(parents=True, exist_ok=True)
    sm = pd.concat(summs, ignore_index=True) if summs else pd.DataFrame(columns=[*summary.COLUMNS, "full", "ref"])
    summary.write(sm, paths["summary"], extra=("full", "ref"))
    st = pd.concat(stats, ignore_index=True) if stats else pd.DataFrame()
    tmp = paths["stats"].with_suffix(".tmp")
    st.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(paths["stats"])
    lcs = pa.concat_tables(kept) if kept else epoch.SCHEMA.empty_table()
    tmp = paths["lightcurves"].with_suffix(".tmp")
    pq.write_table(lcs, tmp, compression="zstd")
    tmp.replace(paths["lightcurves"])
    rec = {"name": name, "stars": int(n_stars), "with_scores": int(by_gbin.sum()),
           "with_scores_by_gbin": [int(x) for x in by_gbin], "summary_rows": int(len(sm)),
           "full_scores": int(len(st)), "period_search": n_period,
           "kept": int(len(pc.unique(lcs["source_id"]))), "rows": n_rows, "read_s": round(t_read, 1),
           "seconds": round(time.time() - t0, 1)}
    if a.get("durable"):
        dr3_pass._publish(paths, Path(a["durable"]), name, rec)
    else:
        (out / "done").mkdir(exist_ok=True)
        (out / "done" / f"{name}.json").write_text(json.dumps(rec) + "\n")
    return rec


def in_share(name, share, seed=0):
    """A repeatable pseudo-random `share` of the input files, by a hash of the file name."""
    import zlib

    return share >= 1 or zlib.crc32(f"{seed}:{name}".encode()) / 2 ** 32 < share


def pass_inputs(a):
    if a.dr3_bulk:
        items = [{"kind": "dr3", "name": dr3_pass.stem(f["name"]), "url": f["url"], "md5": f.get("md5"),
                  "size": f.get("size")} for f in epoch.bulk_files()]
    elif a.dr4_bulk or a.dr4_files:
        from cosmos_search.gaia import dr4_bulk

        if a.dr4_files:
            items = [{"kind": "dr4", "name": q.stem, "local": str(q), "size": q.stat().st_size}
                     for q in sorted(Path(a.dr4_files).rglob("*.parquet"))]
        else:
            items = [{"kind": "dr4", **f} for f in dr4_bulk.find_files(a.prefix)]
    else:
        items = [{"kind": "lc", "name": q.stem, "path": str(q), "size": q.stat().st_size}
                 for q in sorted(Path(a.lightcurves).glob("*.parquet"))]
    return [it for it in items if dr3_pass.in_shard(it["name"], a.shard) and in_share(it["name"], a.files_share, a.seed)]


def pass_cmd(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.rule and a.keep_lightcurves != "all":
        print("without --rule the pass is the thresholds' sample: add --keep-lightcurves all", file=sys.stderr)
        return 2
    items = pass_inputs(a)
    done = Path(a.durable or out) / "done"
    todo = [it for it in items if not (done / f"{it['name']}.json").exists()][: a.limit or None]
    # Largest first, so the long files do not finish last on one worker.
    todo.sort(key=lambda it: -(it.get("size") or 0))
    print(f"{len(items)} inputs, {len(todo)} to do ({sum(it.get('size') or 0 for it in todo) / 1e9:.1f} GB)", flush=True)
    opts = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "func"}
    (out / "pass.json").write_text(json.dumps({k: v for k, v in opts.items() if k != "shard"}, indent=1))
    t0, n, failures = time.time(), 0, 0
    with ProcessPoolExecutor(a.workers) as pool, (out / "manifest.jsonl").open("a") as fh:
        futures = {pool.submit(pass_one, it, opts): it for it in todo}
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

def _id_ranges(files):
    """Smallest and largest source_id of each per-file table, from the Parquet footers."""
    out = []
    for f in files:
        md = pq.read_metadata(f)
        col = md.schema.to_arrow_schema().get_field_index("source_id")
        st = [md.row_group(i).column(col).statistics for i in range(md.num_row_groups)]
        out.append((min(s.min for s in st), max(s.max for s in st)))
    return out


def _read(files, ranges, lo, hi, columns):
    """Rows of the stars with lo <= source_id < hi, read from the files whose source_id range overlaps it."""
    import pyarrow.dataset as ds

    use = [f for f, (a, b) in zip(files, ranges) if a < hi and b >= lo]
    if not use:
        return pd.DataFrame()
    d = ds.dataset(use, format="parquet")
    t = d.to_table(columns=[c for c in columns if c in d.schema.names],
                   filter=(ds.field("source_id") >= lo) & (ds.field("source_id") < hi)).to_pandas()
    for c in t.columns:
        if t[c].dtype == np.float32:
            t[c] = t[c].astype(np.float64)
    return t


def _files(directory):
    return [str(q) for q in sorted(Path(directory).glob("*.parquet")) if pq.read_metadata(q).num_rows]


FLAG_COLUMNS = ["id", "gbin", "flagged", "early", "rising", "rising_rep", "shared_fade", "shared_rise",
                "fade_set_aside", "rise_set_aside", "by", "fade_kind", "strength", "rise_strength"]
FLAGS_MIN = 0.8        # flags.parquet keeps stars at this strength or more, either way
Z_EDGES = np.linspace(-12, 12, 961)  # histogram of trend significance per G bin, for its median over the run


def _add(total, part):
    """Add the numbers of a nested dict of counts into `total`."""
    for k, v in part.items():
        if isinstance(v, dict):
            _add(total.setdefault(k, {}), v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            total[k] = total.get(k, 0) + v
    return total


def _hist_median(h):
    c = np.cumsum(h)
    return float(np.interp(c[-1] / 2, c, Z_EDGES[1:])) if c[-1] else float("nan")


def coarse_reference(sfiles, level, per_cell=200):
    """The reference for cells too sparse at `level` and finer: medians in coarser pixels, from a bounded sample
    of every (pixel at `level`, G range) cell of the run's reference rows."""
    parts = []
    for f in sfiles:
        t = pd.read_parquet(f, columns=["source_id", "g_mag", "slope", "ref"])
        parts.append(summary.cell_sample(t[t["ref"]], level, per_cell))
    sample = summary.cell_sample(pd.concat(parts, ignore_index=True), level, per_cell)
    return summary.reference_table(sample, levels=tuple(x for x in summary.REF_LEVELS if x < level))


def scored_counts(run):
    """Stars with scores per magnitude range over the whole run, from the done-markers (the summaries hold every
    star only down to the pass's --summary-g-max)."""
    by_gbin = np.zeros(len(drift.G_EDGES) - 1, np.int64)
    stars = 0
    for f in Path(run, "done").glob("*.json"):
        rec = json.loads(f.read_text())
        by_gbin += np.array(rec["with_scores_by_gbin"], np.int64)
        stars += rec["stars"]
    grp = rule_v3.mag_group(np.arange(len(by_gbin)))
    return stars, {g: int(by_gbin[grp == g].sum()) for g in np.unique(grp)}, by_gbin


def flag(a):
    """One sky region (HEALPix pixel at --chunk-level) at a time: the shared-date test only compares stars within a
    level-6 pixel, which never crosses a region's edge, and the neighbour reference at levels 6 down to the
    region's own comes from the region's stars, so the lists are the same as from one pass over the whole sky
    and memory follows the largest region."""
    run = Path(a.run)
    out = Path(a.out or run)
    out.mkdir(parents=True, exist_ok=True)
    sfiles, tfiles = _files(run / "summary"), _files(run / "stats")
    sranges, tranges = _id_ranges(sfiles), _id_ranges(tfiles)
    t0 = time.time()
    fine_levels = tuple(x for x in summary.REF_LEVELS if x >= a.chunk_level)
    coarse = coarse_reference(sfiles, a.chunk_level)
    print(f"coarse reference from {len(sfiles)} files in {time.time() - t0:.0f} s: {len(coarse)} cells", flush=True)
    aux = None
    if a.aux:
        aux = pd.read_parquet(a.aux, columns=["source_id", "phot_g_mean_mag", "bp_rp"]).set_index("source_id")
    width = 2 ** 35 * 4 ** (12 - a.chunk_level)
    small, refs, lists = [], [coarse], {"fading": [], "early": [], "rising": []}
    summ_total, n = {}, 0
    hist = {k: np.zeros((len(drift.G_EDGES) - 1, len(Z_EDGES) - 1), np.int64) for k in ("raw", "referenced")}
    for k in range(12 * 4 ** a.chunk_level):
        sm = _read(sfiles, sranges, k * width, (k + 1) * width, SUMMARY_COLUMNS)
        if not len(sm):
            continue
        sm = summary.from_store(sm[np.isfinite(sm["sigma_visit"].to_numpy())].reset_index(drop=True))
        if not len(sm):
            continue
        n += len(sm)
        st = _read(tfiles, tranges, k * width, (k + 1) * width, ("source_id",) + search.STATS_ONLY)
        fine = summary.reference_table(sm[sm["ref"]], levels=fine_levels)
        refs.append(fine)
        m = summary.apply_reference(search.merge(sm, st if len(st) else None), pd.concat([fine, coarse]))
        for name, col in (("raw", "slope_z_raw"), ("referenced", "slope_z")):
            z, gb = m.loc[m["ref"], col].to_numpy(), m.loc[m["ref"], "gbin"].to_numpy()
            for b in np.unique(gb):
                hist[name][b] += np.histogram(np.clip(z[(gb == b) & np.isfinite(z)], -11.99, 11.99), Z_EDGES)[0]
        f = search.flag(m, a.rule)
        f["phot_g_mean_mag"], f["bp_rp"] = m["g_mag"].to_numpy(), m["bp_rp"].to_numpy()
        if aux is not None:  # catalogue magnitudes where the catalogue has the star
            cat = aux.reindex(f["id"].to_numpy())
            for c in ("phot_g_mean_mag", "bp_rp"):
                f[c] = np.where(cat[c].notna().to_numpy(), cat[c].to_numpy(), f[c].to_numpy())
        cols = [c for c in LIST_COLUMNS if c in f]
        lists["fading"].append(f.loc[f["flagged"], cols])
        lists["early"].append(f.loc[f["early"], cols])
        lists["rising"].append(f.loc[f["rising"], cols])
        _add(summ_total, search.summary(f))
        small.append(f.loc[(f["strength"] >= FLAGS_MIN) | (f["rise_strength"] >= FLAGS_MIN), FLAG_COLUMNS])
    stars, scored, by_gbin = scored_counts(a.passed or run)
    print(f"{stars} stars, {sum(scored.values())} with scores, {n} summary rows read from {len(sfiles)} files, "
          f"flagged in {time.time() - t0:.0f} s", flush=True)
    for name, parts in lists.items():
        d = pd.concat(parts, ignore_index=True)
        key = {"fading": "strength", "rising": "rise_strength"}.get(name)
        (d.sort_values(key, ascending=False) if key else d.sort_values("id")).to_parquet(out / f"{name}.parquet",
                                                                                          index=False)
    flags = pd.concat(small, ignore_index=True)
    flags.drop(columns="fade_kind").to_parquet(out / "flags.parquet", index=False)
    pd.concat(refs, ignore_index=True).to_parquet(out / "reference.parquet", index=False)
    for grp, g in summ_total.items():
        g["scored"] = scored.get(grp, g["scored"])
        g["fading_share"] = g["fading"] / max(g["scored"], 1)
    edges = drift.G_EDGES
    offsets = {f"{edges[b]:g}-{edges[b + 1]:g}": {"stars": int(by_gbin[b]),
                                                  "median_trend_z": round(_hist_median(hist["raw"][b]), 3),
                                                  "median_trend_z_against_neighbours":
                                                      round(_hist_median(hist["referenced"][b]), 3)}
               for b in range(len(edges) - 1) if hist["raw"][b].sum()}
    summ = {"rule": str(a.rule), "stars": int(stars), "with_scores": int(sum(scored.values())),
            "by_magnitude": summ_total, "trend_offset_by_G": offsets}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


# -- cuts -------------------------------------------------------------------------------------------------------

def cuts_cmd(a):
    """Automated cuts before anyone reads the lists, added to them as columns:

    - `still`: parallax and proper motion consistent with zero (`search.motion`), for every star above strength 1
      either way: quasars, galaxies and stars in other galaxies, read as a list of their own;
    - `band_verdict` on the fading list (`checks.band_check`): whether G's change agrees with BP's and RP's, with
      each star's BP/RP blended-transit fraction (`blend`) from the same gaia_source rows.

    cuts_summary.json gives the fading and rising counts above each strength without the still stars, and the
    fading counts by band verdict."""
    from concurrent.futures import ThreadPoolExecutor

    from astropy.table import Table

    from cosmos_search.gaia.archive import tap
    from cosmos_search.slow import checks
    from cosmos_search.vetting.stage0 import BLEND_COLS

    run = Path(a.run)
    flags = pd.read_parquet(run / "flags.parquet", columns=["id", "gbin", "strength", "rise_strength"])
    ids = flags.loc[(flags["strength"] > 1) | (flags["rise_strength"] > 1), "id"].to_numpy()
    cols = ", ".join(f"g.{c}" for c in ("source_id", "astrometric_params_solved") + search.MOTION_COLUMNS + BLEND_COLS)
    t, t0 = tap(), time.time()

    def fetch(chunk):
        q = (f"SELECT {cols} FROM TAP_UPLOAD.ids AS u JOIN {a.schema}.gaia_source AS g "
             f"ON g.source_id = u.source_id")
        for attempt in (0, 1):
            try:
                d = t.query(q, uploads={"ids": Table({"source_id": chunk})}).to_pandas()
                d.columns = [c.lower() for c in d.columns]
                return d
            except Exception:  # noqa: BLE001 (one retry, then the step fails loudly)
                if attempt:
                    raise

    chunks = [ids[i:i + a.chunk] for i in range(0, len(ids), a.chunk)]
    with ThreadPoolExecutor(a.workers) as ex:
        g = pd.concat(list(ex.map(fetch, chunks)), ignore_index=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        blend = np.fmax(g["phot_bp_n_blended_transits"] / g["phot_bp_n_obs"],
                        g["phot_rp_n_blended_transits"] / g["phot_rp_n_obs"])
    m = pd.concat([g[["source_id"]], search.motion(g)], axis=1).rename(columns={"source_id": "id"})
    m["blend"] = blend.to_numpy(float)
    m.to_parquet(run / "cuts.parquet", index=False)
    print(f"{len(g)} of {len(ids)} stars' gaia_source rows in {time.time() - t0:.0f} s ({len(chunks)} queries); "
          f"still {int(m['still'].sum())}", flush=True)
    summ = {"still": int(m["still"].sum()), "looked_up": int(len(ids)), "found": int(len(g))}
    for name in ("fading", "early", "rising"):
        d = pd.read_parquet(run / f"{name}.parquet").drop(
            columns=["chi2_motion", "still", "blend", "band_z", "band_verdict"], errors="ignore")
        d = d.merge(m, on="id", how="left")
        d["still"] = d["still"].fillna(False).astype(bool)
        if name == "fading" and "strength_by" in d:
            d = pd.concat([d, checks.band_check(d, d["blend"])], axis=1)
            moving = d[~d["still"]]
            summ["fading_band_verdicts_moving"] = {
                grp: x["band_verdict"].value_counts().to_dict()
                for grp, x in moving.groupby(np.where(moving["gbin"] <= 7, "G<19", "G>=19"))}
        d.to_parquet(run / f"{name}.parquet", index=False)
    summ["above_strength_moving"] = search.tail_moving(flags, m.loc[m["still"], "id"])
    (run / "cuts_summary.json").write_text(json.dumps(summ, indent=1))
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
    c = sub.add_parser("pass", help="one pass over light curves, scoring every star")
    c.add_argument("--out", required=True)
    c.add_argument("--dr4-bulk", action="store_true", help="DR4's bulk files on ESA's CDN (download, read, delete)")
    c.add_argument("--dr4-files", help="directory of DR4-shaped bulk Parquet files already on disk")
    c.add_argument("--dr3-bulk", action="store_true", help="stream DR3's bulk files from ESA's CDN (the trial run)")
    c.add_argument("--lightcurves", help="directory of light-curve Parquet files (epoch.SCHEMA)")
    c.add_argument("--prefix", default="Gaia/gdr4/", help="where --dr4-bulk looks on the CDN")
    c.add_argument("--rule", help="rule directory (drift table and thresholds); without it the pass only "
                                  "summarises, as the sample for the thresholds")
    c.add_argument("--aux", help="optional Parquet with source_id, phot_g_mean_mag, bp_rp (else each star's G and "
                                 "colour come from its light curve)")
    c.add_argument("--durable", help="durable output directory (kore, project files); done-markers live there")
    c.add_argument("--workers", type=int, default=os.cpu_count())
    c.add_argument("--streams", type=int, default=4, help="connections per DR4 file")
    c.add_argument("--shard", type=_shard, default=(0, 1))
    c.add_argument("--files-share", type=float, default=1.0, help="a repeatable random share of the input files")
    c.add_argument("--seed", type=int, default=0, help="seed of --files-share")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--chunk-stars", type=int, default=20_000, help="stars scored at a time")
    c.add_argument("--keep-lightcurves", choices=("near", "all", "none"), default="near",
                   help="light curves kept: stars near a threshold and the random share, every star, or none")
    c.add_argument("--random-share", type=float, default=0.001, help="random share of stars always fully scored "
                                                                      "and kept")
    c.add_argument("--period-g-max", type=float, default=search.PERIOD_G_MAX,
                   help="period search for stars brighter than this")
    c.add_argument("--keep-fraction", type=float, default=search.LC_FRACTION,
                   help="light curves kept from this fraction of a threshold up, on the fading side")
    c.add_argument("--keep-g-max", type=float, default=search.LC_G_MAX, help="and only brighter than this")
    c.add_argument("--summary-g-max", type=float, default=19.0,
                   help="summary rows stored for every star brighter than this")
    c.add_argument("--faint-share", type=float, default=0.1,
                   help="random share of fainter stars whose summary rows are stored (those near a threshold "
                        "always are)")
    c.add_argument("--no-period", action="store_true")
    c.set_defaults(func=pass_cmd)
    f = sub.add_parser("flag", help="trends against neighbours, shared-date test, lists and summary over a run")
    f.add_argument("--run", required=True, help="a pass's output (or its --durable directory)")
    f.add_argument("--passed", help="where the pass's done-markers are, if not under --run")
    f.add_argument("--aux", help="optional catalogue magnitudes for the lists (source_id, phot_g_mean_mag, bp_rp)")
    f.add_argument("--rule", default=str(rule_v3.RULE_DIR))
    f.add_argument("--out")
    f.add_argument("--chunk-level", type=int, default=3, help="HEALPix level of the sky regions flagged one at a time")
    f.set_defaults(func=flag)
    m = sub.add_parser("cuts", help="still stars (no measurable motion) and G against BP and RP, on the lists")
    m.add_argument("--run", required=True, help="the flag step's output directory")
    m.add_argument("--schema", default="gaiadr4")
    m.add_argument("--chunk", type=int, default=50_000, help="source_ids per uploaded query")
    m.add_argument("--workers", type=int, default=4)
    m.set_defaults(func=cuts_cmd)
    a = ap.parse_args(argv)
    if a.cmd == "pass" and sum(bool(x) for x in (a.lightcurves, a.dr3_bulk, a.dr4_bulk, a.dr4_files)) != 1:
        ap.error("pass needs one of --dr4-bulk, --dr4-files, --dr3-bulk or --lightcurves")
    r = a.func(a)
    return 1 if isinstance(r, int) and r else 0


if __name__ == "__main__":
    sys.exit(main())
