"""Fake Gaia DR4 light-curve files built from DR3's, and a local stand-in for ESA's file server and DataLink.

    python -m cosmos_search.gaia.dr4_fake build --out ROOT (--dr3 DIR | --m31 | --random N | --match REGEX) \\
        [--order 5] [--rejected keep|blank] [--workers 4]
    python -m cosmos_search.gaia.dr4_fake serve --root ROOT [--port 8765] [--flaky 0.02]

For practice runs of the release-day code before DR4 exists (docs/release-day.md). `build` rewrites DR3's bulk
epoch-photometry files (csv.gz, one row per source with arrays) in the layout the release-day reader expects of DR4:
Parquet with one row per transit and the draft data model's columns (2026-06-26), under
`Gaia/gdr4/epoch_photometry/Norder=K/Dir=D/Npix=P.parquet` with an md5 file, plus two kinds of file a listing must
ignore (the per-CCD table and catalogue metadata). `serve` answers the three things the code asks of ESA: the
bucket listing (S3 ListObjectsV2), file downloads with byte ranges, and DataLink EPOCH_PHOTOMETRY for a list of
source ids. It prints the environment lines that point `gaia.archive` at it.

What DR4 will really look like is not known, so the fake varies what it can: half the files write missing fluxes
as nulls and half as NaN; half hold the 12 CCD flags as a fixed-size list and half as a variable list; fluxes are
float32 as the draft model says; row groups are small so stars straddle them. `--rejected` chooses what happens to
the transits DR3 marked as rejected by variability processing, a mark the draft DR4 table does not have: `keep`
leaves them in unmarked (the default, what the draft implies), `blank` removes their fluxes.

The DR3 numbers behind each file are unchanged, so every star's fake DR4 light curve can be compared with DR3's.
"""

import argparse
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from xml.sax.saxutils import escape

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import epoch

PRODUCT = "Gaia/gdr4/epoch_photometry"
# DR3 Vega zero points (Riello et al. 2021), for the g_mag, bp_mag, rp_mag columns the draft model carries.
ZERO_POINT = {"g": 25.6874, "bp": 25.3385, "rp": 24.7479}
SOLUTION_ID = 4545469030156206080  # any constant: the release-day code never reads it
# The six level-5 HEALPix pixels of DR3's M31 field, the one part of DR3 with a light curve for every star.
M31_HPX5 = (659, 662, 668, 676, 678, 691)

SCHEMA_FIELDS = [
    ("solution_id", pa.int64()), ("source_id", pa.int64()), ("transit_id", pa.int64()), ("g_n_obs", pa.int32()),
    ("g_obs_time", pa.float64()), ("g_flux", pa.float32()), ("g_flux_error", pa.float32()),
    ("g_flux_over_error", pa.float32()), ("g_mag", pa.float32()),
    ("bp_obs_time", pa.float64()), ("bp_flux", pa.float32()), ("bp_flux_error", pa.float32()),
    ("bp_flux_over_error", pa.float32()), ("bp_mag", pa.float32()),
    ("rp_obs_time", pa.float64()), ("rp_flux", pa.float32()), ("rp_flux_error", pa.float32()),
    ("rp_flux_over_error", pa.float32()), ("rp_mag", pa.float32()),
    ("transit_acq_flags", pa.int16()), ("transit_proc_flags", pa.int16()),
]


def schema(fixed_flags=True):
    flags = pa.list_(pa.int16(), 12) if fixed_flags else pa.list_(pa.int16())
    return pa.schema(SCHEMA_FIELDS + [("phot_ccd_proc_flags", flags), ("xp_ccd_info_flags", pa.list_(pa.int8(), 2))])


def pixel_of(source_id, order):
    """Nested HEALPix pixel at `order` from Gaia source ids (source_id // 2**35 is the level-12 pixel)."""
    return np.asarray(source_id, np.int64) >> (35 + 2 * (12 - order))


def key_of(pixel, order):
    return f"{PRODUCT}/Norder={order}/Dir={int(pixel) // 10000 * 10000}/Npix={int(pixel)}.parquet"


def to_dr4_rows(sid, c, rejected="keep", nulls=True, fixed_flags=True):
    """DR3's per-transit columns (`epoch.iter_epoch_csv_columns`) as a DR4-shaped table, one row per transit."""
    n = len(sid)
    cols = {"solution_id": np.full(n, SOLUTION_ID, np.int64), "source_id": sid, "transit_id": c["transit_id"],
            "g_n_obs": np.nan_to_num(c["g_transit_n_obs"], nan=0).astype(np.int32)}
    for band, tcol, fcol, ecol in (("g", "g_transit_time", "g_transit_flux", "g_transit_flux_error"),
                                   ("bp", "bp_obs_time", "bp_flux", "bp_flux_error"),
                                   ("rp", "rp_obs_time", "rp_flux", "rp_flux_error")):
        f, e = c[fcol].astype(np.float64), c[ecol].astype(np.float64)
        if rejected == "blank":
            bad = c[f"variability_flag_{band}_reject"]
            f, e = np.where(bad, np.nan, f), np.where(bad, np.nan, e)
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = np.where(f > 0, ZERO_POINT[band] - 2.5 * np.log10(f), np.nan)
            cols.update({f"{band}_obs_time": c[tcol], f"{band}_flux": f, f"{band}_flux_error": e,
                         f"{band}_flux_over_error": f / e, f"{band}_mag": mag})
    cols["transit_acq_flags"] = np.full(n, 0x2000, np.int16)  # bit 13 "always set" in the draft model
    cols["transit_proc_flags"] = np.zeros(n, np.int16)
    phot = np.zeros((n, 12), np.int16)
    phot[c["photometry_flag_bp_reject"], 10] = 0x2000  # the reader's "BP/RP unrecoverable" bit
    phot[c["photometry_flag_rp_reject"], 11] = 0x2000
    sch = schema(fixed_flags)
    arrays = []
    for field in sch:
        if field.name == "phot_ccd_proc_flags":
            flat = pa.array(phot.ravel(), pa.int16())
            arrays.append(pa.FixedSizeListArray.from_arrays(flat, 12) if fixed_flags else
                          pa.ListArray.from_arrays(pa.array(np.arange(0, 12 * n + 1, 12, dtype=np.int32)), flat))
        elif field.name == "xp_ccd_info_flags":
            arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(np.zeros(2 * n, np.int8)), 2))
        else:
            v = cols[field.name]
            is_flux = pa.types.is_float32(field.type)
            # Fluxes absent in DR3 become nulls or NaN (the draft says NaN; Parquet writers often use nulls).
            arrays.append(pa.array(v, field.type, mask=~np.isfinite(v)) if is_flux and nulls
                          else pa.array(v, field.type))
    return pa.Table.from_arrays(arrays, schema=sch)


def _build_one(path, root, order, rejected, row_group):
    """One DR3 file -> one part per pixel under root/_parts/<pixel>/. Returns {pixel: rows}."""
    root, name = Path(root), Path(path).name.split(".")[0]
    writers, rows = {}, {}
    try:
        for sid, c in epoch.iter_epoch_csv_columns(path, 5000):
            pix = pixel_of(sid, order)
            for p in np.unique(pix):
                p = int(p)
                m = pix == p
                t = to_dr4_rows(sid[m], {k: v[m] for k, v in c.items()}, rejected, nulls=p % 2 == 0,
                                fixed_flags=p % 4 < 2)
                if p not in writers:
                    d = root / "_parts" / str(p)
                    d.mkdir(parents=True, exist_ok=True)
                    writers[p] = pq.ParquetWriter(d / f"{name}.parquet", t.schema, compression="zstd")
                writers[p].write_table(t, row_group_size=row_group)
                rows[p] = rows.get(p, 0) + t.num_rows
    finally:
        for w in writers.values():
            w.close()
    done = root / "_parts_done" / f"{name}.json"
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(rows))
    return rows


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def finish(root, order, row_group=200_000):
    """Parts -> one file per pixel (parts of a pixel that spans two DR3 files are merged, sorted by source id), the
    md5 file, and the files a listing must ignore."""
    root = Path(root)
    made = []
    for d in sorted((root / "_parts").iterdir(), key=lambda x: int(x.name)):
        dest = root / key_of(int(d.name), order)
        dest.parent.mkdir(parents=True, exist_ok=True)
        parts = sorted(d.glob("*.parquet"))
        if not parts and dest.exists():      # moved into place by a build that stopped here (--resume)
            made.append(dest)
            continue
        if len(parts) == 1:
            parts[0].replace(dest)
        else:
            t = pa.concat_tables([pq.read_table(p) for p in parts])
            tmp = dest.with_name(dest.name + ".tmp")
            pq.write_table(t.take(pc.sort_indices(t, sort_keys=[("source_id", "ascending")])), tmp,
                           compression="zstd", row_group_size=row_group)
            tmp.replace(dest)
        made.append(dest)
    shutil.rmtree(root / "_parts")
    shutil.rmtree(root / "_parts_done", ignore_errors=True)
    base = root / PRODUCT
    (base / "_MD5SUM.txt").write_text("".join(f"{_md5(p)}  {p.relative_to(base).as_posix()}\n" for p in made))
    (base / "properties").write_text("obs_collection=epoch_photometry\nhats_order=%d\n" % order)
    (base / "partition_info.csv").write_text("Norder,Npix\n" + "".join(
        f"{order},{p.stem.split('=')[1]}\n" for p in made))
    ccd = root / f"{PRODUCT}_ccd" / f"Norder={order}" / "Dir=0" / "Npix=0.parquet"
    ccd.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"source_id": pa.array([1], pa.int64()), "transit_id": pa.array([1], pa.int64())}), ccd)
    return made


def dr3_inputs(a, work):
    """Local DR3 bulk files for the build: a directory of them, or files downloaded from ESA's CDN into `work`."""
    if a.dr3:
        files = sorted(Path(a.dr3).glob("EpochPhotometry_*.csv.gz"))
        if a.match:
            files = [f for f in files if re.search(a.match, f.name)]
        return files[: a.limit or None]
    from cosmos_search.net import get_client

    listed = epoch.bulk_files()
    pick = []
    if a.m31:
        lo = [p * 64 for p in M31_HPX5]
        pick += [f for f in listed if any(f["hpx8"][0] <= x + 63 and f["hpx8"][1] >= x for x in lo)]
    if a.match:
        pick += [f for f in listed if re.search(a.match, f["name"]) and f not in pick]
    if a.random:
        rest = [f for f in listed if f not in pick and f["size"] < a.max_bytes]
        pick += random.Random(a.seed).sample(rest, min(a.random, len(rest)))
    work.mkdir(parents=True, exist_ok=True)
    client, out = get_client(), []
    for f in pick[: a.limit or None]:
        dest = work / f["name"]
        if not dest.exists():
            client.download(f["url"], dest, md5=f.get("md5"))
        out.append(dest)
    return out


def build(a):
    root = Path(a.out)
    if (root / "Gaia").exists() and not (a.add or a.resume):
        raise SystemExit(f"{root} already holds fake files; remove it, or pass --add or --resume")
    t0 = time.time()
    inputs = dr3_inputs(a, root / "_dr3")
    print(f"{len(inputs)} DR3 files, {sum(p.stat().st_size for p in inputs) / 1e9:.2f} GB", flush=True)
    rows = {}
    # A build that stopped part-way (--resume): DR3 files whose parts are complete are not read again.
    done = {p.stem: json.loads(p.read_text()) for p in (root / "_parts_done").glob("*.json")} if a.resume else {}
    if not a.resume and not a.add:
        shutil.rmtree(root / "_parts", ignore_errors=True)
        shutil.rmtree(root / "_parts_done", ignore_errors=True)
    todo = [p for p in inputs if p.name.split(".")[0] not in done]
    for r in done.values():
        for pix, n in r.items():
            rows[int(pix)] = rows.get(int(pix), 0) + n
    if done:
        print(f"{len(done)} DR3 files already split, {len(todo)} to do", flush=True)
    with ProcessPoolExecutor(a.workers) as pool:
        futures = {pool.submit(_build_one, str(p), str(root), a.order, a.rejected, a.row_group): p for p in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            for pix, n in fut.result().items():
                rows[pix] = rows.get(pix, 0) + n
            if k % 50 == 0 or k == len(todo) or len(todo) < 300:
                print(f"[{k}/{len(todo)}] {futures[fut].name} ({time.time() - t0:.0f} s)", flush=True)
    made = finish(root, a.order, a.row_group)
    rec = {"files": len(made), "bytes": int(sum(p.stat().st_size for p in made)), "transits": int(sum(rows.values())),
           "order": a.order, "rejected": a.rejected, "dr3_files": [p.name for p in inputs],
           "seconds": round(time.time() - t0, 1)}
    (root / "build.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "dr3_files"}, indent=1))
    if not a.keep_dr3 and not a.dr3:
        shutil.rmtree(root / "_dr3", ignore_errors=True)
    return 0


# -- the stand-in server ----------------------------------------------------------------------------------------

def list_keys(root):
    """Every servable file under root as (key, size), sorted by key. Working folders (_dr3, _parts) are left out."""
    root = Path(root)
    out = []
    for p in root.rglob("*"):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and not rel.startswith(("_dr3/", "_parts/", "_parts_done/")) and rel != "build.json":
            out.append((rel, p.stat().st_size))
    return sorted(out)


def list_xml(keys, prefix="", delimiter=None, max_keys=1000, token=None):
    """S3 ListObjectsV2 response for (key, size) pairs: contents, common prefixes and a continuation token."""
    items, seen = [], set()
    for key, size in keys:
        if not key.startswith(prefix):
            continue
        if delimiter and delimiter in key[len(prefix):]:
            cp = key[: len(prefix) + key[len(prefix):].index(delimiter) + 1]
            if cp not in seen:
                seen.add(cp)
                items.append((cp, None))
        else:
            items.append((key, size))
    if token:
        items = [it for it in items if it[0] > token]
    page, more = items[:max_keys], len(items) > max_keys
    body = ['<?xml version="1.0" encoding="UTF-8"?>',
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">',
            f"<Name>gaia</Name><Prefix>{escape(prefix)}</Prefix><KeyCount>{len(page)}</KeyCount>",
            f"<MaxKeys>{max_keys}</MaxKeys><IsTruncated>{'true' if more else 'false'}</IsTruncated>"]
    for key, size in page:
        body.append(f"<CommonPrefixes><Prefix>{escape(key)}</Prefix></CommonPrefixes>" if size is None else
                    f"<Contents><Key>{escape(key)}</Key><Size>{size}</Size></Contents>")
    if more:
        body.append(f"<NextContinuationToken>{escape(page[-1][0])}</NextContinuationToken>")
    body.append("</ListBucketResult>")
    return "".join(body).encode()


def datalink_csv(root, ids, order=None):
    """DataLink EPOCH_PHOTOMETRY as csv in the draft model's layout: one row per transit of the requested sources,
    in the order asked, array columns as "[a, b, ...]" and missing values empty."""
    root = Path(root)
    order = order if order is not None else json.loads((root / "build.json").read_text())["order"]
    ids = [int(i) for i in ids]
    by_file = {}
    for sid in ids:
        by_file.setdefault(root / key_of(pixel_of(sid, order), order), []).append(sid)
    parts = []
    for path, want in by_file.items():
        if path.exists():
            t = pq.read_table(path, filters=[("source_id", "in", want)])
            if t.num_rows:
                parts.append(t.cast(schema(True)) if t.schema.field("phot_ccd_proc_flags").type != pa.list_(pa.int16(), 12)
                             else t)
    if not parts:
        return (",".join(f.name for f in schema()) + "\n").encode()
    df = pa.concat_tables(parts).to_pandas()
    rank = {s: i for i, s in enumerate(ids)}
    df = df.sort_values("g_obs_time", kind="stable")
    df = df.iloc[np.argsort(df["source_id"].map(rank).to_numpy(), kind="stable")]
    for c in ("phot_ccd_proc_flags", "xp_ccd_info_flags"):
        df[c] = ["[" + ", ".join(str(int(x)) for x in v) + "]" for v in df[c]]
    buf = io.StringIO()
    df.to_csv(buf, index=False, na_rep="")
    return buf.getvalue().encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    root, flaky, quiet = Path("."), 0.0, True
    counts = {"list": 0, "get": 0, "range": 0, "datalink": 0, "refused": 0}
    lock = threading.Lock()

    def log_message(self, fmt, *args):
        if not self.quiet:
            super().log_message(fmt, *args)

    def _count(self, what):
        with self.lock:
            self.counts[what] += 1

    def _send(self, status, body=b"", ctype="application/octet-stream", extra=None, head=False):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def do_HEAD(self):
        self._file(urlsplit(self.path).path, head=True)

    def do_GET(self):
        parts = urlsplit(self.path)
        q = parse_qs(parts.query)
        if parts.path in ("", "/") and ("list-type" in q or "prefix" in q):
            self._count("list")
            body = list_xml(list_keys(self.root), q.get("prefix", [""])[0], q.get("delimiter", [None])[0],
                            int(q.get("max-keys", ["1000"])[0]),
                            q.get("continuation-token", q.get("start-after", [None]))[0])
            return self._send(200, body, "application/xml")
        if parts.path == "/counts":
            return self._send(200, json.dumps(self.counts).encode(), "application/json")
        self._file(parts.path, head=False)

    def _file(self, path, head):
        target = (self.root / path.lstrip("/")).resolve()
        if not target.is_file() or self.root.resolve() not in target.parents:
            return self._send(404, b"not found", "text/plain", head=head)
        if self.flaky and random.random() < self.flaky:
            self._count("refused")
            return self._send(503, b"busy", "text/plain", head=head)
        size = target.stat().st_size
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            a, b = int(a or 0), min(int(b) if b else size - 1, size - 1)
            self._count("range")
            with open(target, "rb") as fh:
                fh.seek(a)
                body = fh.read(b - a + 1)
            return self._send(206, body, extra={"Content-Range": f"bytes {a}-{b}/{size}"}, head=head)
        self._count("get")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if not head:
            with open(target, "rb") as fh:
                shutil.copyfileobj(fh, self.wfile, 1 << 20)

    def do_POST(self):
        path = urlsplit(self.path).path
        form = parse_qs(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
        if path.endswith("/login"):
            return self._send(200, b"OK", "text/plain", extra={"Set-Cookie": "JSESSIONID=practice; Path=/"})
        if path.endswith("/data"):
            self._count("datalink")
            ids = [x for x in form.get("ID", [""])[0].split(",") if x.strip()]
            if len(ids) > epoch.DATALINK_MAX_IDS:  # the archive's own limit and answer
                return self._send(500, b"Maximum number of Source Ids reached (max: 50)", "text/plain")
            if form.get("RETRIEVAL_TYPE", [""])[0] != "EPOCH_PHOTOMETRY":
                return self._send(400, b"unknown RETRIEVAL_TYPE", "text/plain")
            return self._send(200, datalink_csv(self.root, ids), "text/csv")
        self._send(404, b"not found", "text/plain")


def env_lines(url):
    return [f"export COSMOS_GAIA_CDN_URL={url}", f"export COSMOS_GAIA_CDN_LIST_URL={url}",
            f"export COSMOS_GAIA_DATALINK_URL={url}/data-server/data", f"export COSMOS_GAIA_LOGIN_URL={url}/tap-server/login"]


def point_archive_at(url):
    """Point this process (and the worker processes it starts) at a stand-in server: the same as exporting
    `env_lines(url)` before starting Python."""
    from . import archive

    for line in env_lines(url):
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ[key] = value
    archive.CDN_URL = archive.CDN_LIST_URL = url
    archive.DATALINK_URL, archive.LOGIN_URL = f"{url}/data-server/data", f"{url}/tap-server/login"


def make_server(root, port=0, flaky=0.0, quiet=True):
    """A server over `root` on 127.0.0.1 (port 0 = any free port). Call serve_forever() or use it in a thread."""
    handler = type("PracticeHandler", (Handler,), {
        "root": Path(root), "flaky": flaky, "quiet": quiet,
        "counts": {k: 0 for k in Handler.counts}, "lock": threading.Lock()})
    return _Server(("127.0.0.1", port), handler)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        """A client that hangs up mid-response (it retries) is not worth a traceback."""
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


def serve(a):
    srv = make_server(a.root, a.port, a.flaky, quiet=not a.verbose)
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    print("\n".join(env_lines(url)), flush=True)
    if a.env_file:
        Path(a.env_file).write_text("\n".join(env_lines(url)) + "\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.gaia.dr4_fake", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="DR3 bulk files -> fake DR4 files under ROOT")
    b.add_argument("--out", required=True)
    b.add_argument("--dr3", help="directory of DR3 EpochPhotometry_*.csv.gz files (kore's cache); else ESA's CDN")
    b.add_argument("--m31", action="store_true", help="the three DR3 files of the M31 field (4.6 GB)")
    b.add_argument("--random", type=int, default=0, help="this many other DR3 files, drawn at random")
    b.add_argument("--max-bytes", type=float, default=60e6, help="largest file --random may draw")
    b.add_argument("--seed", type=int, default=20261202)
    b.add_argument("--match", help="DR3 file names matching this regular expression")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--order", type=int, default=5, help="HEALPix order of the fake files (5: 12,288 over the sky)")
    b.add_argument("--rejected", choices=("keep", "blank"), default="keep",
                   help="transits DR3 marked rejected by variability processing: left in unmarked, or blanked")
    b.add_argument("--row-group", type=int, default=200_000)
    b.add_argument("--workers", type=int, default=os.cpu_count())
    b.add_argument("--add", action="store_true", help="add to an existing ROOT")
    b.add_argument("--resume", action="store_true", help="go on with a build that stopped before it finished")
    b.add_argument("--keep-dr3", action="store_true", help="keep the downloaded DR3 files under ROOT/_dr3")
    b.set_defaults(func=build)
    s = sub.add_parser("serve", help="stand in for ESA's file server and DataLink over ROOT")
    s.add_argument("--root", required=True)
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--flaky", type=float, default=0.0, help="share of file requests answered 503")
    s.add_argument("--env-file", help="also write the export lines here")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=serve)
    a = ap.parse_args(argv)
    if a.cmd == "build" and not (a.dr3 or a.m31 or a.random or a.match):
        ap.error("build needs --dr3 DIR, --m31, --random N or --match REGEX")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
