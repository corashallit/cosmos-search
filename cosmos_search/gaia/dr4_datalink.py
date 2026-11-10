"""DR4 light curves through ESA's DataLink, for lists of up to ~10^4 stars.

    python -m cosmos_search.gaia.dr4_datalink --ids list.parquet --out DIR [--workers 4] [--chunk 5] [--limit N]

DataLink is the one route to DR4's epoch photometry that ESA has confirmed (the DR4 content page, 2026-09-14), so it
is how the first stars are read on release day: Boyajian's star, the DR3 comparison stars, the DR3 watch-list.
It is slow (0.1-1.5 light curves a second on DR3, at most 50 ids a call; docs/infra/dr4-downloads.md), so the
whole-sky pass uses the bulk files if they exist.

DR3's DataLink answered `DATA_STRUCTURE=DATAMODEL_GAIA, FORMAT=csv` with one row per source holding arrays. DR4's
draft data model describes epoch_photometry as one row per transit, and what the service will send for several ids
(one table, or an archive of one file per source) is not known before release. `read_response` therefore looks at
the bytes it gets: a zip or tar of files, gzip, VOTable, FITS, Parquet, ECSV or csv; one row per transit or DR3's
arrays. Whatever the layout, the result is a light-curve table in `epoch.SCHEMA`.

`pull` asks for `chunk` ids per call, `workers` calls at a time (small calls in parallel were 2-3 times faster
than large ones on DR3), writes one Parquet per call and resumes from the files already written.
"""

import argparse
import gzip
import io
import json
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cosmos_search.net import get_client

from . import archive, dr4, epoch

CHUNK = 5
WORKERS = 4
RELEASE = "Gaia DR4"


def _text_table(raw):
    """csv or ECSV bytes -> pandas frame (ECSV's header lines start with '#')."""
    import pandas as pd

    return pd.read_csv(io.BytesIO(raw), comment="#", skipinitialspace=True)


def read_response(content):
    """A DataLink EPOCH_PHOTOMETRY response body -> light-curve table (epoch.SCHEMA), whatever its packaging."""
    raw = bytes(content)
    head = raw[:512].lstrip()
    if raw[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            parts = [read_response(z.read(n)) for n in z.namelist() if not n.endswith("/")]
        return pa.concat_tables(parts) if parts else epoch.SCHEMA.empty_table()
    if raw[:2] == b"\x1f\x8b":
        return read_response(gzip.decompress(raw))
    if len(raw) > 262 and raw[257:262] == b"ustar":
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            parts = [read_response(tar.extractfile(m).read()) for m in tar.getmembers() if m.isfile()]
        return pa.concat_tables(parts) if parts else epoch.SCHEMA.empty_table()
    if raw[:4] == b"PAR1":
        return dr4.epoch_photometry_to_lc(pq.read_table(io.BytesIO(raw)))
    if head.startswith(b"<?xml") or b"<VOTABLE" in head:
        from astropy.io.votable import parse_single_table

        return _rows_to_lc(parse_single_table(io.BytesIO(raw)).to_table())
    if raw[:6] == b"SIMPLE":
        from astropy.table import Table

        return _rows_to_lc(Table.read(io.BytesIO(raw), format="fits"))
    if not raw.strip():
        return epoch.SCHEMA.empty_table()
    df = _text_table(raw)
    names = {str(c).lower() for c in df.columns}
    if "n_transits" in names and "g_transit_time" in names:  # DR3's layout: arrays per source
        return epoch.read_epoch_csv(raw)
    return _rows_to_lc(df)


def _rows_to_lc(table):
    if not len(table):
        return epoch.SCHEMA.empty_table()
    missing = dr4.missing_required(dr4._names(table))
    if missing:
        raise KeyError(f"DataLink response has no column(s) {missing}; columns are {sorted(map(str, dr4._names(table)))}")
    return dr4.epoch_photometry_to_lc(table)


def fetch(source_ids, release=RELEASE, fmt="csv", client=None, timeout=600):
    """Light curves of up to epoch.DATALINK_MAX_IDS sources in one DataLink call, as a light-curve table."""
    client = client or get_client()
    archive.login(client)
    ids = [int(s) for s in source_ids]
    if len(ids) > epoch.DATALINK_MAX_IDS:
        raise ValueError(f"{len(ids)} ids in one call; the archive takes at most {epoch.DATALINK_MAX_IDS}")
    resp = client.post(archive.DATALINK_URL, data={
        "RETRIEVAL_TYPE": "EPOCH_PHOTOMETRY", "ID": ",".join(map(str, ids)), "RELEASE": release,
        "DATA_STRUCTURE": "DATAMODEL_GAIA", "FORMAT": fmt, "VALID_DATA": "false"}, timeout=timeout)
    resp.raise_for_status()
    return read_response(resp.content)


def _one(part, name, out, release, fmt):
    t0 = time.time()
    lc = fetch(part, release, fmt)
    tmp = out / "lightcurves" / (name + ".tmp")
    pq.write_table(lc, tmp, compression="zstd")
    tmp.replace(out / "lightcurves" / name)
    got = len(np.unique(np.asarray(lc["source_id"])))
    return {"chunk": name, "requested": len(part), "returned": int(got), "rows": lc.num_rows,
            "seconds": round(time.time() - t0, 1)}


def pull(ids, out, release=RELEASE, fmt="csv", workers=WORKERS, chunk=CHUNK, limit=None, log=print):
    """Light curves of `ids`, in the order given, `chunk` ids per call and `workers` calls at a time, one Parquet
    per call under out/lightcurves. Files are named by the offset of their first id and their size, so resume with
    the same order and --chunk. Returns the number of calls that failed (retried on the next run)."""
    out = Path(out)
    (out / "lightcurves").mkdir(parents=True, exist_ok=True)
    ids = [int(i) for i in (ids[:limit] if limit else ids)]
    chunk = min(int(chunk), epoch.DATALINK_MAX_IDS)
    todo = []
    for k in range(0, len(ids), chunk):
        part = ids[k:k + chunk]
        name = f"chunk-{k:08d}-{len(part)}.parquet"
        if not (out / "lightcurves" / name).exists():
            todo.append((part, name))
    failures, stars, t0 = 0, 0, time.time()
    with (out / "timing.jsonl").open("a") as fh, ThreadPoolExecutor(max(1, workers)) as pool:
        futures = {pool.submit(_one, part, name, out, release, fmt): name for part, name in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
                stars += rec["returned"]
            except Exception as exc:  # noqa: BLE001
                failures += 1
                rec = {"chunk": futures[fut], "error": repr(exc)[:300]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 20 == 0 or "error" in rec or i == len(todo):
                log(f"[{i}/{len(todo)}] {stars / max(time.time() - t0, 1e-9):.2f} light curves/s {rec}")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.gaia.dr4_datalink", description=__doc__.splitlines()[0])
    ap.add_argument("--ids", required=True, help="Parquet or csv with a source_id column, in the order to pull")
    ap.add_argument("--out", required=True)
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--format", default="csv")
    ap.add_argument("--workers", type=int, default=WORKERS, help="DataLink calls at a time")
    ap.add_argument("--chunk", type=int, default=CHUNK, help=f"ids per call (at most {epoch.DATALINK_MAX_IDS})")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    import pandas as pd

    table = pd.read_csv(a.ids) if a.ids.endswith(".csv") else pd.read_parquet(a.ids)
    col = "source_id" if "source_id" in table else "id"
    failed = pull(table[col].tolist(), a.out, a.release, a.format, a.workers, a.chunk, a.limit,
                  log=lambda m: print(m, flush=True))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
