"""Mirror the columns cosmos-search reads from a remote HATS catalogue into a local HATS catalogue.

    cd ~/cosmos-data/cosmos-search
    .venv/bin/python -I scripts/mirror_hats.py gaia_dr3 [--partitions 8 --streams 4]
    .venv/bin/python -I scripts/mirror_hats.py --list
    .venv/bin/python -I scripts/mirror_hats.py --verify gaia_dr3 [--cones 8]

Each partition is fetched as byte ranges: the Parquet footer, then only the column chunks we keep, written at
their own offsets into a sparse temp file, so pyarrow reads it as if it were the whole file. The kept columns
are rewritten with zstd and byte-stream-split floats (lossless; ~35% smaller than the source's zstd), and the
partition is renamed into place only after its row count matches the footer. A rerun skips finished
partitions, so an interrupted mirror resumes. The catalogue-level files (properties, partition_info, skymaps)
are copied, and _common_metadata / _metadata are rebuilt for the kept columns. mirror.json records the source,
the columns and the row counts; cosmos_search.xmatch.mirror reads it to decide whether a survey can be served
locally.

Requests go through the repo's archive client (per-host limits, retries, polite User-Agent).
"""

import argparse
import json
import os
import struct
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cosmos_search.net import get_client
from cosmos_search.net.local import data_dir
from cosmos_search.xmatch.surveys import GAIA_DR3, SURVEYS

ROOT = (data_dir() or Path.home() / "cosmos-data") / "mirror" / "hats"
LSDB = "https://data.lsdb.io/hats"
MAST_S3 = "https://stpubdata.s3.amazonaws.com"

# Extra gaia_source columns beyond the matcher's, so dayone.rank.context can be answered locally for DR3
# (l, b and ecl_lat are left out: 46 GB, and they follow from ra/dec).
GAIA_EXTRA = ("astrometric_params_solved", "ruwe", "ipd_frac_multi_peak", "phot_bp_rp_excess_factor",
              "phot_bp_n_obs", "phot_bp_n_blended_transits", "phot_rp_n_obs", "phot_rp_n_blended_transits")

# name -> (catalogue URL to copy from, URL the code knows it by, columns). Columns default to the survey's.
CATALOGS = {
    "gaia_dr3": (f"{MAST_S3}/gaia/gaia_dr3/public/hats/gaia", GAIA_DR3.hats, (*GAIA_DR3.columns, *GAIA_EXTRA)),
    "2mass": (f"{LSDB}/two_mass/two_mass", SURVEYS["2mass"].hats, None),
    "allwise": (f"{LSDB}/wise/allwise/allwise", SURVEYS["allwise"].hats, None),
    "tic": (f"{MAST_S3}/tess/public/hats/tic/tic", SURVEYS["tic"].hats, None),
    "erosita_dr2": (f"{LSDB}/erosita_dr2_erass3/erass3_main", SURVEYS["erosita_dr2"].hats, "all"),
    "vsx": (f"{LSDB}/vsx/2026-06", SURVEYS["vsx"].hats, "all"),
    "ztf_dr22": (f"{LSDB}/ztf_dr22/ztf_lc", SURVEYS["ztf_dr22"].hats, ("objra", "objdec", "objectid", "filterid",
                                                                       "nepochs", "fieldid", "rcid")),
}
CATALOG_FILES = ("hats.properties", "properties", "partition_info.csv", "point_map.fits", "skymap.fits",
                 "skymap.2.fits", "skymap.4.fits", "skymap.6.fits")
TAIL = 1 << 16
GAP = 256 << 10          # merge column-chunk ranges closer than this
MAX_RANGE = 16 << 20     # and split ranges longer than this
ROW_GROUP = 200_000


def fetch_range(client, url, start, end, total=None):
    """Bytes [start, end) of url (end=None with start<0: the last -start bytes). Returns (bytes, file size)."""
    rng = f"bytes={start}" if end is None else f"bytes={start}-{end - 1}"  # start < 0: "bytes=-N", the tail
    for attempt in range(6):
        r = client.get(url, headers={"Range": rng})
        if r.status_code == 206:
            size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
            want = (-start if end is None else end - start)
            if len(r.content) == min(want, size):
                return r.content, size
        elif r.status_code == 200 and end is None:  # small file served whole
            return r.content[start:], len(r.content)
        else:
            r.raise_for_status()
        time.sleep(min(60, 2 ** attempt))
    raise IOError(f"short or failed range {rng} for {url}")


def needed_ranges(md, keep):
    ranges = []
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        for j in range(rg.num_columns):
            c = rg.column(j)
            if c.path_in_schema.split(".")[0] not in keep:
                continue
            start = c.dictionary_page_offset if c.has_dictionary_page and c.dictionary_page_offset else c.data_page_offset
            start = min(start, c.data_page_offset)
            ranges.append((start, start + c.total_compressed_size))
    ranges.sort()
    merged = []
    for a, b in ranges:
        if merged and a - merged[-1][1] <= GAP:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        while b - a > MAX_RANGE:
            out.append((a, a + MAX_RANGE))
            a += MAX_RANGE
        out.append((a, b))
    return out


def mirror_partition(client, src, dst, keep, tmpdir, streams):
    """Copy one partition's kept columns. Returns (rows, bytes fetched)."""
    tail, size = fetch_range(client, src, -TAIL, None)
    footer_len = struct.unpack("<I", tail[-8:-4])[0]
    if tail[-4:] != b"PAR1":
        raise IOError(f"{src}: not a Parquet file")
    if footer_len + 8 > len(tail):
        tail, _ = fetch_range(client, src, -(footer_len + 8), None)
    fd, tmp = tempfile.mkstemp(dir=tmpdir, suffix=".parquet")
    try:
        os.ftruncate(fd, size)
        os.pwrite(fd, tail, size - len(tail))
        md = pq.read_metadata(tmp)
        names = set(md.schema.to_arrow_schema().names)
        cols = [c for c in names if c in keep] if keep != "all" else sorted(names)
        keepset = set(cols)
        ranges = needed_ranges(md, keepset)
        got = 0

        def one(r):
            body, _ = fetch_range(client, src, r[0], r[1])
            os.pwrite(fd, body, r[0])
            return len(body)

        with ThreadPoolExecutor(streams) as ex:
            for n in ex.map(one, ranges):
                got += n
        t = pq.read_table(tmp, columns=[c for c in md.schema.to_arrow_schema().names if c in keepset])
        if t.num_rows != md.num_rows:
            raise IOError(f"{src}: read {t.num_rows} rows, footer says {md.num_rows}")
    finally:
        os.close(fd)
        os.unlink(tmp)
    write_table(t, dst)
    return t.num_rows, got + len(tail)


def write_table(t, dst):
    floats = [f.name for f in t.schema if pa.types.is_floating(f.type)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(f".{dst.name}.part")
    pq.write_table(t, part, compression="zstd", compression_level=3, row_group_size=ROW_GROUP,
                   use_byte_stream_split=floats or False,
                   use_dictionary=[c for c in t.column_names if c not in floats])
    os.replace(part, dst)


def partitions(client, base):
    text = client.get_bytes(f"{base}/partition_info.csv").decode()
    rows = [line.split(",") for line in text.strip().splitlines()[1:]]
    return [(int(o), int(p)) for o, p, *_ in rows]


def rel_path(order, pix):
    return f"dataset/Norder={order}/Dir={pix // 10000 * 10000}/Npix={pix}.parquet"


def finish(out, base, client, name, alias, keep, stats):
    """Rebuild _common_metadata and _metadata for the kept columns, and write mirror.json."""
    files = sorted((out / "dataset").glob("Norder=*/Dir=*/Npix=*.parquet"))
    metas, rows, nbytes = [], 0, 0
    schema = None
    for f in files:
        md = pq.read_metadata(f)
        md.set_file_path(str(f.relative_to(out / "dataset")))
        metas.append(md)
        rows += md.num_rows
        nbytes += f.stat().st_size
        if schema is None:
            schema = pq.read_schema(f)
    pq.write_metadata(schema, out / "dataset" / "_common_metadata")
    pq.write_metadata(schema, out / "dataset" / "_metadata", metadata_collector=metas)
    manifest = {"name": name, "source": base, "url": alias, "columns": schema.names,
                "rows": rows, "partitions": len(files), "bytes": nbytes,
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **stats}
    (out / "mirror.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def run(name, partitions_at_once, streams, limit=None):
    base, alias, keep = CATALOGS[name]
    keep = SURVEYS[name].columns if keep is None else keep
    if keep != "all":
        keep = {*keep, "_healpix_29", "Norder", "Dir", "Npix"}
    client = get_client()
    out = ROOT / name
    tmpdir = ROOT / ".tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for f in CATALOG_FILES:
        body = client.get_bytes(f"{base}/{f}", missing_ok=True)
        if body is not None:
            (out / f).write_bytes(body)
    parts = partitions(client, base)
    todo = [(o, p) for o, p in parts if not (out / rel_path(o, p)).exists()]
    if limit:
        todo = todo[:limit]
    print(f"{name}: {len(parts)} partitions, {len(parts) - len(todo)} done, {len(todo)} to fetch", flush=True)
    t0, done, fetched, rows = time.time(), 0, 0, 0
    lock = threading.Lock()
    failed = []

    def work(op):
        o, p = op
        return mirror_partition(client, f"{base}/{rel_path(o, p)}", out / rel_path(o, p), keep, tmpdir, streams)

    with ThreadPoolExecutor(partitions_at_once) as ex:
        futs = {ex.submit(work, op): op for op in todo}
        for fut in as_completed(futs):
            try:
                n, b = fut.result()
            except Exception as e:  # report and carry on; a rerun picks it up
                failed.append(futs[fut])
                print(f"  FAILED {futs[fut]}: {type(e).__name__}: {e}", flush=True)
                continue
            with lock:
                done += 1
                fetched += b
                rows += n
            if done % 20 == 0 or done == len(todo):
                dt = time.time() - t0
                print(f"  {done}/{len(todo)} partitions, {fetched / 1e9:.1f} GB in {dt / 60:.1f} min "
                      f"({fetched / dt / 1e6:.1f} MB/s)", flush=True)
    if failed:
        print(f"{name}: {len(failed)} partitions failed; rerun to retry", flush=True)
        return 1
    if limit:
        return 0
    m = finish(out, base, client, name, alias, keep, {})
    print(f"{name}: complete, {m['rows']:,} rows, {m['bytes'] / 1e9:.1f} GB on disk", flush=True)
    return 0


def verify(name, n_cones=8, radius_deg=0.05, seed=None):
    """Row count against the source's properties, then random cones read locally and remotely, row for row."""
    import random

    import pandas as pd

    from cosmos_search.xmatch import mirror
    from cosmos_search.xmatch.sources import HatsSource
    from cosmos_search.xmatch.surveys import SURVEYS

    base, alias, _ = CATALOGS[name]
    m = json.loads((ROOT / name / "mirror.json").read_text())
    props = dict(line.split("=", 1) for line in (ROOT / name / "hats.properties").read_text().splitlines()
                 if "=" in line and not line.startswith("#"))
    want_rows = int(props.get("hats_nrows", props.get("total_rows", -1)))
    ok = m["rows"] == want_rows
    print(f"{name}: {m['rows']:,} rows mirrored, source properties say {want_rows:,}: {'ok' if ok else 'MISMATCH'}")
    survey = GAIA_DR3 if name == "gaia_dr3" else SURVEYS[name]
    local, remote = HatsSource(), HatsSource(urls={name: base})
    assert local.location(survey) == str(ROOT / name), mirror.manifests().keys()
    rng = random.Random(seed)
    parts = partitions(get_client(), base)
    for _ in range(n_cones):
        o, p = rng.choice(parts)
        import cdshealpix

        lon, lat = cdshealpix.nested.healpix_to_lonlat(np.array([p], dtype=np.uint64), o)
        ra, dec = lon.deg[0], lat.deg[0]
        a = local.fetch(survey, float(ra), float(dec), radius_deg)
        b = remote.fetch(survey, float(ra), float(dec), radius_deg)
        cols = [c for c in survey.columns if c in a.columns]
        a = a[cols].sort_values(survey.id).reset_index(drop=True)
        b = b[cols].sort_values(survey.id).reset_index(drop=True)
        same = len(a) == len(b) and all(a[c].equals(b[c]) or pd.Series(a[c]).fillna(-9e99).equals(
            pd.Series(b[c]).fillna(-9e99)) for c in cols)
        ok &= same
        print(f"  cone ({float(ra):.3f}, {float(dec):.3f}) r={radius_deg} deg: {len(a)} local, {len(b)} remote, "
              f"{'identical' if same else 'DIFFERENT'}", flush=True)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--partitions", type=int, default=6, help="partitions fetched at once")
    ap.add_argument("--streams", type=int, default=4, help="range requests at once per partition")
    ap.add_argument("--limit", type=int, help="fetch only this many partitions (testing)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify", action="store_true", help="compare the named mirrors with their sources")
    ap.add_argument("--cones", type=int, default=8)
    a = ap.parse_args(argv)
    if a.verify:
        return max((verify(n, a.cones) for n in a.names), default=0)
    if a.list or not a.names:
        for n, (src, alias, cols) in CATALOGS.items():
            m = ROOT / n / "mirror.json"
            state = json.loads(m.read_text()) if m.exists() else None
            print(f"{n:12s} {'complete' if state else 'missing':9s} {src}")
        return 0
    rc = 0
    for n in a.names:
        rc |= run(n, a.partitions, a.streams, a.limit)
    return rc


if __name__ == "__main__":
    sys.exit(main())
