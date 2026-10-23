"""Gaia DR4 epoch photometry from ESA's bulk download files, streamed into a local light-curve cache.

    python -m cosmos_search.gaia.dr4_bulk list
    python -m cosmos_search.gaia.dr4_bulk pull --out DIR [--select sel.parquet] [--random-share 0.001] \\
        [--durable DIR] [--workers 8] [--shard i/n] [--limit N]

ESA's approved plan for the DR4 bulk download (technical note GAIA-CZ-TN-ESA-JDB-128, 2024-10-21) puts the
transit-level `epoch_photometry` (~9 TB) on the CDN as Parquet in a HATS-like HEALPix layout, and leaves the
per-CCD G (`epoch_photometry_ccd`), epoch astrometry, images and epoch spectra on DataLink only. Nothing about
DR4 is on the CDN before release, so the file layout here is a guess from that note and the draft data model
(2026-06-26): any Parquet file under `Gaia/gdr4/` whose path names epoch photometry, one row per transit, with
the columns `gaia.dr4.COLUMNS` names. `list` shows what is there; check it on release day before a long pull.

`pull` downloads files in parallel (each worker one file at a time, md5-checked when ESA publishes sums), keeps
the transits of the chosen stars, converts them to the shared light-curve schema (`epoch.SCHEMA`) and writes one
Parquet per bulk file plus a done-marker, so a stopped or reclaimed run resumes where it stopped and sessions
split the files with --shard without talking. Stars kept: those in --select (a Parquet with source_id) and a
deterministic random share (`features.control_mask`, the same hash as the DR3 controls).

Nothing stages the release: each raw file is deleted as soon as it is read, so peak disk per machine is about
--workers raw files (DR3's epoch files reached 2.2 GB; ESA aims for evenly sized DR4 files) plus the kept light
curves (~3.9 kB a star). The whole release is 7.2 TB, which no machine of ours holds. `keep_all` (API only, no CLI
flag) keeps every star of a file and is meant for a detector that consumes each file's output and discards it.

Measured 2026-10-23: one cloud container reads ESA's CDN at ~32 MB/s per stream and ~120 MB/s with 8 streams;
kore pulled DR3's 73 GB at ~45 MB/s. At 120 MB/s DR4's light curves take ~17 h to download in one container.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cosmos_search.net import get_client, s3_list

from . import archive, dr3_pass, dr4, epoch, features

PREFIX = "Gaia/gdr4/"
# A path names transit-level epoch photometry if a directory or file is called epoch_photometry (any case, with
# or without the underscore), but not the per-CCD table or the crowded-field product.
_PRODUCT = re.compile(r"(^|/)epoch_?photometry(/|_|\.|$)", re.IGNORECASE)
_EXCLUDE = re.compile(r"ccd|crowded", re.IGNORECASE)
_MD5 = re.compile(r"md5", re.IGNORECASE)

# Columns read from each file: what dr4.epoch_photometry_to_lc uses. Missing optional ones are skipped.
READ_COLUMNS = ("source_id", "transit_id", "g_n_obs", "transit_proc_flags", "phot_ccd_proc_flags",
                *(c for cols in dr4.COLUMNS.values() for c in cols))
REQUIRED = ("source_id", "transit_id", *dr4.COLUMNS["G"])


def is_epoch_photometry(key):
    return key.endswith(".parquet") and bool(_PRODUCT.search(key)) and not _EXCLUDE.search(key)


def stem(key, prefix=PREFIX):
    """Output name for a bulk file: its path below the prefix with / as __ (HATS names like Npix=12.parquet
    repeat across directories)."""
    return key.removeprefix(prefix).removesuffix(".parquet").replace("/", "__")


def find_files(prefix=PREFIX, client=None):
    """Epoch photometry bulk files under prefix on ESA's CDN, as dicts with name, key, url, size, md5.

    MD5 sums come from any file under the same tree whose name mentions md5, in `md5sum` format; paths in it may
    be bare file names or relative paths."""
    client = client or get_client()
    entries = list(s3_list(archive.CDN_LIST_URL, prefix, client=client))
    sums = {}
    for e in entries:
        if _MD5.search(e.key.rsplit("/", 1)[-1]) and not e.key.endswith(".parquet"):
            text = client.get_bytes(f"{archive.CDN_URL}/{e.key}").decode(errors="replace")
            for line in text.splitlines():
                parts = line.split()
                if len(parts) == 2:
                    sums[parts[1].lstrip("*./")] = parts[0]
    files = []
    for e in entries:
        if not is_epoch_photometry(e.key):
            continue
        md5 = sums.get(e.key.rsplit("/", 1)[-1]) or next(
            (v for k, v in sums.items() if e.key.endswith("/" + k)), None)
        files.append({"name": stem(e.key, prefix), "key": e.key, "url": f"{archive.CDN_URL}/{e.key}",
                      "size": e.size, "md5": md5})
    return sorted(files, key=lambda f: f["key"])


def summary(prefix=PREFIX, client=None):
    """What DR4 has on the CDN: top-level folders and the epoch photometry files found (count, bytes)."""
    client = client or get_client()
    top = [e.key for e in s3_list(archive.CDN_LIST_URL, prefix, delimiter="/", client=client)]
    files = find_files(prefix, client)
    return {"prefix": prefix, "top_level": top, "epoch_photometry_files": len(files),
            "epoch_photometry_bytes": int(sum(f["size"] for f in files)),
            "with_md5": int(sum(f["md5"] is not None for f in files)),
            "examples": [f["key"] for f in files[:3]]}


_SELECT = {}


def _load_select(path):
    """Unique source_ids of a selection Parquet, loaded once per worker process."""
    if path is None:
        return None
    if path not in _SELECT:
        ids = pq.read_table(path, columns=["source_id"])["source_id"].combine_chunks()
        _SELECT[path] = pc.unique(ids.cast(pa.int64()))
    return _SELECT[path]


def keep_mask(source_id, select=None, random_share=0.0, keep_all=False):
    """Boolean mask over transit rows: the row's star is selected, in the random share, or everything."""
    sid = source_id.cast(pa.int64())
    if keep_all:
        return np.ones(len(sid), bool)
    mask = np.zeros(len(sid), bool)
    if select is not None:
        mask |= np.asarray(pc.is_in(sid, value_set=select).fill_null(False))
    if random_share > 0:
        mask |= features.control_mask(np.asarray(sid), random_share)
    return mask


def read_kept(path, select=None, random_share=0.0, keep_all=False, batch_rows=1 << 20):
    """Transit rows of the kept stars from one bulk Parquet file, as a pyarrow Table (READ_COLUMNS present)."""
    f = pq.ParquetFile(path)
    names = set(f.schema_arrow.names)
    missing = [c for c in REQUIRED if c not in names]
    if missing:
        raise KeyError(f"{path}: no column(s) {missing}; columns are {sorted(names)} (update gaia.dr4.COLUMNS)")
    cols = [c for c in READ_COLUMNS if c in names]
    parts = []
    for batch in f.iter_batches(batch_size=batch_rows, columns=cols):
        m = keep_mask(batch.column("source_id"), select, random_share, keep_all)
        if m.any():
            parts.append(pa.Table.from_batches([batch]).filter(pa.array(m)))
    if not parts:
        return f.schema_arrow.empty_table().select(cols)
    return pa.concat_tables(parts)


# Free space a download needs beyond the file itself, so a full disk fails the file cleanly instead of mid-write.
DISK_MARGIN = 2e9


def check_space(directory, size, margin=DISK_MARGIN):
    free = shutil.disk_usage(directory).free
    if free < (size or 0) + margin:
        raise OSError(f"{free / 1e9:.1f} GB free in {directory}, need {((size or 0) + margin) / 1e9:.1f} GB for the "
                      "next file; free space or run fewer --workers")


# Ranged downloads: one connection is capped well below a machine's link (kore: ~12 MB/s per stream from the Gaia
# CDN, ~2 MB/s from S3; a cloud container ~21-32 MB/s), so a large file is fetched as RANGE_BYTES pieces over
# `streams` connections and written at their offsets. Each piece is held in memory once, so memory per worker is
# about streams x RANGE_BYTES.
RANGE_BYTES = 32 << 20
RANGE_MIN = 2 * RANGE_BYTES
STREAMS = 4


def _fetch_range(client, url, start, end, attempts=4):
    want = end - start + 1
    for i in range(attempts):
        resp = client.get(url, headers={"Range": f"bytes={start}-{end}"})
        if resp.status_code == 206 and len(resp.content) == want:
            return resp.content
        if resp.status_code == 200:  # server ignored the range: give up on splitting
            raise _NoRanges
        time.sleep(2 ** i)
    raise OSError(f"range {start}-{end} of {url}: {resp.status_code}, {len(resp.content)} of {want} bytes")


class _NoRanges(Exception):
    pass


@contextmanager
def fetched(url, dest, size=None, md5=None, streams=STREAMS, client=None):
    """Download url to dest, yield its Path, delete it afterwards (also on errors). Files of RANGE_MIN or more are
    fetched as byte ranges over `streams` connections; smaller ones, or servers that ignore ranges, in one stream."""
    client = client or get_client()
    dest = Path(dest)
    try:
        if streams > 1 and size and size >= RANGE_MIN:
            try:
                with open(dest, "wb") as fh:
                    fh.truncate(size)
                spans = [(a, min(a + RANGE_BYTES, size) - 1) for a in range(0, size, RANGE_BYTES)]

                def one(span):
                    body = _fetch_range(client, url, *span)
                    with open(dest, "r+b") as fh:
                        fh.seek(span[0])
                        fh.write(body)

                with ThreadPoolExecutor(streams) as pool:
                    list(pool.map(one, spans))
                if md5:
                    h = hashlib.md5()
                    with open(dest, "rb") as fh:
                        for block in iter(lambda: fh.read(1 << 24), b""):
                            h.update(block)
                    if h.hexdigest() != md5.lower():
                        raise OSError(f"md5 mismatch for {url}")
                yield dest
                return
            except _NoRanges:
                dest.unlink(missing_ok=True)
        with client.streamed(url, dest=dest, md5=md5) as path:
            yield path
    finally:
        dest.unlink(missing_ok=True)


def process_file(info, out, select_path=None, random_share=0.0, keep_all=False, durable=None, local=None,
                 streams=STREAMS):
    """Download one bulk file (or read `local`), keep the chosen stars, write their light curves (epoch.SCHEMA).

    The raw file lives in out/tmp only while it is read and is deleted afterwards, also on errors, so a machine
    holds at most one raw file per worker."""
    out = Path(out)
    t0 = time.time()
    dest = out / "lightcurves" / f"{info['name']}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    select = _load_select(select_path)
    if local:
        rows, t_dl = read_kept(local, select, random_share, keep_all), 0.0
    else:
        tmp_dir = out / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        check_space(tmp_dir, info.get("size"))
        with fetched(info["url"], tmp_dir / f"{info['name']}.parquet", info.get("size"), info.get("md5"), streams) as path:
            t_dl = time.time() - t0
            rows = read_kept(path, select, random_share, keep_all)
    t1 = time.time()
    lc = dr4.epoch_photometry_to_lc(rows) if rows.num_rows else epoch.SCHEMA.empty_table()
    meta = {b"source_file": info.get("key", info["name"]).encode(), b"md5": (info.get("md5") or "").encode(),
            b"select": (str(select_path) if select_path else "").encode(), b"random_share": str(random_share).encode(),
            b"keep_all": str(bool(keep_all)).encode()}
    tmp = dest.with_suffix(".tmp")
    pq.write_table(lc.replace_schema_metadata(meta), tmp, compression="zstd")
    tmp.replace(dest)
    rec = {"name": info["name"], "key": info.get("key"), "bytes": info.get("size"), "md5": info.get("md5"),
           "transits_kept": rows.num_rows, "stars_kept": len(pc.unique(rows["source_id"])) if rows.num_rows else 0,
           "rows": lc.num_rows, "download_s": round(t_dl, 1), "process_s": round(time.time() - t1, 1),
           "seconds": round(time.time() - t0, 1)}
    if durable:
        dr3_pass._publish({"lightcurves": dest}, Path(durable), info["name"], rec)
    else:
        done = out / "done" / f"{info['name']}.json"
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps(rec) + "\n")
    return rec


def is_done(name, out, durable=None):
    return (Path(durable or out) / "done" / f"{name}.json").exists()


def pull(out, files=None, select=None, random_share=0.0, keep_all=False, durable=None, workers=8, shard=(0, 1),
         limit=None, log=print, streams=STREAMS):
    """Process every bulk file in this shard that is not done yet. Returns the number of failures."""
    if not (keep_all or select or random_share > 0):
        raise ValueError("choose stars: select, random_share or keep_all")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = files if files is not None else find_files()
    mine = [f for f in files if dr3_pass.in_shard(f["name"], shard)]
    todo = [f for f in mine if not is_done(f["name"], out, durable)][:limit]
    total = sum(f.get("size") or 0 for f in todo)
    log(f"{len(files)} files, {len(mine)} in shard {shard[0]}/{shard[1]}, {len(todo)} to do ({total / 1e9:.1f} GB)")
    failures, done_bytes, t0 = 0, 0, time.time()
    with ProcessPoolExecutor(workers) as pool, (out / "manifest.jsonl").open("a") as fh:
        futures = {pool.submit(process_file, f, out, select, random_share, keep_all, durable, f.get("local"), streams): f
                   for f in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            f = futures[fut]
            try:
                rec = fut.result()
                done_bytes += f.get("size") or 0
            except Exception as exc:  # keep going; the file is retried on the next run
                failures += 1
                rec = {"name": f["name"], "error": repr(exc)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 20 == 0 or "error" in rec or i == len(todo):
                rate = done_bytes / 1e6 / max(time.time() - t0, 1e-9)
                log(f"[{i}/{len(todo)}] {rate:.1f} MB/s {rec}")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.gaia.dr4_bulk", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="what DR4 epoch photometry the CDN holds")
    ls.add_argument("--prefix", default=PREFIX)
    pl = sub.add_parser("pull", help="stream bulk files into a light-curve cache")
    pl.add_argument("--out", required=True, help="local working directory")
    pl.add_argument("--durable", help="durable output directory (kore, project files); done-markers live here")
    pl.add_argument("--prefix", default=PREFIX)
    pl.add_argument("--select", help="Parquet with a source_id column: stars to keep")
    pl.add_argument("--random-share", type=float, default=0.0, help="also keep this share of stars at random")
    pl.add_argument("--workers", type=int, default=8, help="files at a time (each holds one raw file on disk)")
    pl.add_argument("--streams", type=int, default=STREAMS, help="connections per file (byte ranges)")
    pl.add_argument("--shard", type=dr3_pass._shard, default=(0, 1))
    pl.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        print(json.dumps(summary(a.prefix), indent=1))
        return 0
    files = find_files(a.prefix)
    if not files:
        print(f"no epoch photometry Parquet files under {a.prefix}; run `list` to see what is there", file=sys.stderr)
        return 2
    return 1 if pull(a.out, files, a.select, a.random_share, False, a.durable, a.workers, a.shard, a.limit,
                     log=lambda m: print(m, flush=True), streams=a.streams) else 0


if __name__ == "__main__":
    sys.exit(main())
