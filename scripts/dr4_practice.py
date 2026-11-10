"""Practice run of release day's file route, on fake DR4 files built from DR3.

    python -m cosmos_search.gaia.dr4_fake build --out FAKE --m31 --random 60 --keep-dr3
    python scripts/dr4_practice.py --root FAKE --out OUT [--flaky 0.02] [--workers 4] [--datalink 200]

    # all of DR3, on a machine that holds its files (kore)
    python -m cosmos_search.gaia.dr4_fake build --out FAKE --dr3 DIR --workers 6 [--resume]
    python scripts/dr4_practice.py --root FAKE --out OUT --dr3 DIR --workers 6

FAKE is served on localhost as ESA's file listing, file server and DataLink endpoint, and the release-day code is
run against it unchanged:

1. `dr4_bulk.summary` and `find_files`: the listing, checksums, what is not a light-curve file.
2. `dr4_bulk.pull(keep_all=True)`: every file downloaded (ranges for the large ones, `--flaky` share of requests
   answered 503) and read into light curves. Wall time, download and read time, the largest worker's memory.
3. `dr4_check.file_report` on the smallest and the largest file: the first-hour check of a file.
4. Every star and band compared with DR3 read directly by `epoch.iter_epoch_csv` (the DR3 files: `--keep-dr3` at
   build, or `--dr3 DIR`): number of epochs, sums of time, flux, error, field of view, CCD row, CCDs used, transit
   ids, and the count of epochs rejected by photometry processing. Sums over a star's epochs, written per file to
   OUT/digest/ and compared a 48th of the sky at a time, so all of DR3 compares in little memory.
5. `dr4_datalink.pull` for `--datalink` stars, compared with the same stars read from the files.

Writes OUT/practice.json; exits 1 if anything differs or failed.
"""

import argparse
import json
import resource
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cosmos_search.gaia import dr4_fake, epoch

SUMS = ("t", "flux", "err", "phot_reject", "vari_reject", "noisy", "fov", "ccd_row", "n_obs", "tid", "flux_nan",
        "err_nan")
KEY = ["source_id", "band"]


def digest(lc):
    """Per star and band: number of epochs and sums over them (see module docstring), as a pandas frame."""
    flux, err, flags = lc["flux"], lc["err"], lc["flags"]
    zero = pa.scalar(0.0)
    t = pa.table({
        "source_id": lc["source_id"], "band": pc.cast(lc["band"], pa.string()), "t": lc["t"],
        "flux": pc.if_else(pc.is_nan(flux), zero, flux), "err": pc.if_else(pc.is_nan(err), zero, err),
        "flux_nan": pc.cast(pc.is_nan(flux), pa.int64()), "err_nan": pc.cast(pc.is_nan(err), pa.int64()),
        "phot_reject": pc.cast(pc.not_equal(pc.bit_wise_and(flags, epoch.FLAG_PHOT_REJECT), 0), pa.int64()),
        "vari_reject": pc.cast(pc.not_equal(pc.bit_wise_and(flags, epoch.FLAG_VARI_REJECT), 0), pa.int64()),
        "noisy": pc.cast(pc.not_equal(pc.bit_wise_and(flags, epoch.FLAG_NOISY), 0), pa.int64()),
        "fov": pc.cast(lc["fov"], pa.int64()), "ccd_row": pc.cast(lc["ccd_row"], pa.int64()),
        "n_obs": pc.cast(lc["n_obs"], pa.int64()),
        "tid": pc.cast(pc.bit_wise_and(lc["transit_id"], (1 << 40) - 1), pa.int64()),
    })
    g = t.group_by(KEY).aggregate([("t", "count"), *[(c, "sum") for c in SUMS]]).to_pandas()
    return g.rename(columns={"t_count": "n", **{f"{c}_sum": c for c in SUMS}})


def _merge(parts):
    if not parts:
        return pd.DataFrame(columns=[*KEY, "n", *SUMS])
    return pd.concat(parts, ignore_index=True).groupby(KEY, sort=False, as_index=False).sum()


def digest_dr3(path):
    """Digest of one DR3 bulk file, read by the DR3 reader."""
    return _merge([digest(lc) for lc in epoch.iter_epoch_csv(path, chunk_rows=5000)])


def digest_lightcurves(path):
    """Digest of one light-curve Parquet file (epoch.SCHEMA), a row group at a time."""
    f = pq.ParquetFile(path)
    return _merge([digest(f.read_row_group(i)) for i in range(f.num_row_groups)])


COUNTS = ("n", "phot_reject", "fov", "ccd_row", "n_obs", "tid", "flux_nan", "err_nan")
PART_SHIFT = 57      # source_id >> 57 is the HEALPix level-1 pixel: the sky in 48 parts


def _tally(a, b, flux_rtol=2e-7):
    """Counts and largest differences between two digests, in a form that adds up over parts of the sky."""
    m = a.merge(b, on=KEY, how="outer", suffixes=("_a", "_b"), indicator=True)
    both = m[m["_merge"] == "both"]
    out = {"star_bands": len(m), "only_dr3": int((m["_merge"] == "left_only").sum()),
           "only_dr4": int((m["_merge"] == "right_only").sum()),
           "epochs_dr3": int(a["n"].sum()), "epochs_dr4": int(b["n"].sum())}
    for c in COUNTS:
        out[f"{c}_differs"] = int((both[f"{c}_a"] != both[f"{c}_b"]).sum())
    out["t_max_abs_diff_days"] = float(np.abs(both["t_a"] - both["t_b"]).max()) if len(both) else 0.0
    for c in ("flux", "err"):
        scale = np.maximum(np.abs(both[f"{c}_a"]), 1e-300)
        rel = np.abs(both[f"{c}_a"] - both[f"{c}_b"]) / scale
        out[f"{c}_max_rel_diff"] = float(rel.max()) if len(both) else 0.0
        out[f"{c}_differs"] = int((rel > flux_rtol).sum())
    # What DR3 marks and the draft DR4 table has no column for.
    for side, d in (("dr3", a), ("dr4", b)):
        out[f"{side}_vari_rejected"], out[f"{side}_noisy"] = int(d["vari_reject"].sum()), int(d["noisy"].sum())
    return out


def _add(total, part):
    """Sum of two tallies (largest differences: the larger)."""
    if total is None:
        return dict(part)
    return {k: (max(v, part[k]) if "_max_" in k else v + part[k]) for k, v in total.items()}


def _result(tally):
    out = dict(tally)
    for side in ("dr3", "dr4"):
        n = max(out[f"epochs_{side}"], 1)
        out[f"{side}_share_vari_rejected"] = out.pop(f"{side}_vari_rejected") / n
        out[f"{side}_share_noisy"] = out.pop(f"{side}_noisy") / n
    out["equal"] = not any(out[k] for k in out if k.endswith("_differs") or k.startswith("only_")) and \
        out["t_max_abs_diff_days"] < 1e-9
    return out


def compare(a, b, flux_rtol=2e-7):
    """Differences between two digests (a: DR3 read directly, b: the DR4 route). float32 fluxes in the fake files
    round each flux by up to 6e-8 of itself."""
    return _result(_tally(a, b, flux_rtol))


def _digest_file(job):
    """Digest of one file written next to the others; returns (path, lowest and highest source id, rows). Each
    worker returns three numbers, so the digests of every star of a release never sit in one process."""
    kind, src, dest = job
    d = digest_dr3(src) if kind == "dr3" else digest_lightcurves(src)
    d.to_parquet(dest, index=False)
    return (str(dest), int(d["source_id"].min()), int(d["source_id"].max()), len(d)) if len(d) else None


def digest_files(pool, kind, files, out, log=print):
    """Digests of many files under out/, in the pool. Returns [(path, lowest id, highest id, rows)]."""
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(kind, str(f), str(out / (Path(f).name.split(".")[0] + ".parquet"))) for f in files]
    index = []
    for k, r in enumerate(pool.map(_digest_file, jobs), 1):
        if r:
            index.append(r)
        if k % 200 == 0 or k == len(jobs):
            log(f"digests ({kind}): {k}/{len(jobs)}")
    return index


def _read_part(index, part):
    lo, hi = part << PART_SHIFT, ((part + 1) << PART_SHIFT) - 1
    got = [pd.read_parquet(p) for p, a, b, _ in index if a <= hi and b >= lo]
    got = [d[(d["source_id"] >= lo) & (d["source_id"] <= hi)] for d in got]
    return _merge([d for d in got if len(d)])


def compare_by_part(index_a, index_b):
    """`compare` over two sets of digest files, one part of the sky at a time (memory: one part's stars)."""
    parts = sorted({x for idx in (index_a, index_b) for _, a, b, _ in idx
                    for x in range(a >> PART_SHIFT, (b >> PART_SHIFT) + 1)})
    total = None
    for part in parts:
        total = _add(total, _tally(_read_part(index_a, part), _read_part(index_b, part)))
    return _result(total) if total else compare(_merge([]), _merge([]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="fake DR4 files (dr4_fake build)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flaky", type=float, default=0.02, help="share of the stand-in server's answers that are 503")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--datalink", type=int, default=200, help="stars to pull through the stand-in DataLink")
    ap.add_argument("--no-compare", action="store_true", help="skip the comparison with DR3 read directly")
    ap.add_argument("--dr3", help="directory of the DR3 files the fake set was built from (default: ROOT/_dr3)")
    a = ap.parse_args(argv)
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    build = json.loads((root / "build.json").read_text())
    rec = {"root": str(root), "build": {k: build[k] for k in ("files", "bytes", "transits", "order", "rejected")},
           "flaky": a.flaky, "workers": a.workers}

    srv = dr4_fake.make_server(root, 0, flaky=a.flaky)
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    dr4_fake.point_archive_at(url)
    from cosmos_search.gaia import dr4_bulk, dr4_check, dr4_datalink

    # 1. is it out, and what is there
    rec["listing"] = dr4_bulk.summary()
    files = dr4_bulk.find_files()
    print("listing:", rec["listing"], flush=True)

    # 2. download and read everything (before anything large is read here: the workers are forked from this process)
    t0 = time.time()
    failures = dr4_bulk.pull(out, files, keep_all=True, workers=a.workers, log=lambda m: print(m, flush=True))
    wall = time.time() - t0
    done = [json.loads(p.read_text()) for p in sorted((out / "done").glob("*.json"))]
    rec["pull"] = {
        "failures": failures, "files": len(done), "wall_s": round(wall, 1),
        "download_s": round(sum(d["download_s"] for d in done), 1),
        "read_s": round(sum(d["process_s"] for d in done), 1),
        "transits": int(sum(d["transits_kept"] for d in done)), "stars": int(sum(d["stars_kept"] for d in done)),
        "lightcurve_rows": int(sum(d["rows"] for d in done)),
        "bytes": int(sum(d["bytes"] or 0 for d in done)),
        "largest_worker_rss_gb": dr4_bulk.peak_rss_gb(resource.RUSAGE_CHILDREN),
        "largest_file_bytes": int(max(d["bytes"] or 0 for d in done)),
        "server_requests": dict(srv.RequestHandlerClass.counts),
    }
    rec["pull"]["transits_per_core_s"] = round(rec["pull"]["transits"] / max(rec["pull"]["read_s"], 1e-9))
    print("pull:", rec["pull"], flush=True)

    # 3. first-hour check of a file, on the smallest and the largest
    by_size = sorted(files, key=lambda f: f["size"])
    rec["file_reports"] = {}
    for label, f in (("smallest", by_size[0]), ("largest", by_size[-1])):
        with dr4_bulk.fetched(f["url"], out / "tmp-check.parquet", f["size"], f.get("md5")) as path:
            rep = dr4_check.file_report(path)
        rec["file_reports"][label] = {"key": f["key"], **rep}
        print(f"{label} file {f['key']}: {rep['rows']} rows, {rep['stars']} stars, problems: {rep['problems']}, "
              f"notes: {rep['notes']}", flush=True)

    # 4. compare with DR3 read directly: a digest file per input file, then one part of the sky at a time
    lc_files = sorted((out / "lightcurves").glob("*.parquet"))
    log = lambda m: print(m, flush=True)  # noqa: E731
    with ProcessPoolExecutor(a.workers) as pool:
        idx4 = digest_files(pool, "dr4", lc_files, out / "digest" / "dr4", log)
        if not a.no_compare:
            dr3_dir = Path(a.dr3) if a.dr3 else root / "_dr3"
            dr3_files = [dr3_dir / n for n in build.get("dr3_files", []) if (dr3_dir / n).exists()] or \
                sorted(dr3_dir.glob("EpochPhotometry_*.csv.gz"))
            if not dr3_files:
                raise SystemExit(f"{dr3_dir} holds no DR3 files; build with --keep-dr3, pass --dr3 DIR or --no-compare")
            t0 = time.time()
            idx3 = digest_files(pool, "dr3", sorted(dr3_files, key=lambda p: -p.stat().st_size),
                                out / "digest" / "dr3", log)
            rec["compare"] = {**compare_by_part(idx3, idx4), "dr3_files": len(dr3_files),
                              "dr3_read_s": round(time.time() - t0, 1)}
            print("compare with DR3 read directly:", rec["compare"], flush=True)

    # 5. DataLink, for stars drawn from a few of the files
    if a.datalink and idx4:
        rng = np.random.default_rng(1)
        some = [idx4[i] for i in rng.choice(len(idx4), min(20, len(idx4)), replace=False)]
        ref = _merge([pd.read_parquet(p) for p, *_ in some])
        ids = ref["source_id"].drop_duplicates().sample(min(a.datalink, ref["source_id"].nunique()), random_state=1)
        t0 = time.time()
        fails = dr4_datalink.pull(ids.tolist(), out / "datalink", workers=a.workers, chunk=5,
                                  log=lambda m: print(m, flush=True))
        d = _merge([digest_lightcurves(p) for p in sorted((out / "datalink" / "lightcurves").glob("*.parquet"))])
        ref = ref[ref["source_id"].isin(set(ids))]
        rec["datalink"] = {"stars": len(ids), "failed_calls": fails, "seconds": round(time.time() - t0, 1),
                           **{k: v for k, v in compare(ref, d).items() if not k.startswith("dr")}}
        print("DataLink:", rec["datalink"], flush=True)

    srv.shutdown()
    ok = not failures and rec.get("compare", {"equal": True})["equal"] and \
        rec.get("datalink", {"equal": True, "failed_calls": 0})["equal"] and \
        not rec.get("datalink", {"failed_calls": 0})["failed_calls"] and \
        not any(r["problems"] for r in rec["file_reports"].values())
    rec["ok"] = bool(ok)
    (out / "practice.json").write_text(json.dumps(rec, indent=1, default=str))
    print("practice", "passed" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
