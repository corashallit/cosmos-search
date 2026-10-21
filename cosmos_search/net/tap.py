"""TAP queries through the shared client, with a job-slot limit and a result cache.

Results are cached as the raw VOTable bytes the service returned, keyed by (service, ADQL with
whitespace normalised, maxrec, upload contents), so a rerun of a frozen query costs nothing.
"""

import hashlib
import io
import re
import threading

import numpy as np
import pyvo
from astropy.io import votable
from astropy.table import Table

from .blind import check_blind
from .cache import ResponseCache
from .client import get_client

JOB_TIMEOUT = 3 * 3600.0


class Tap:
    def __init__(self, url, *, client=None, job_slots=2):
        self.url = url.rstrip("/")
        self.client = client or get_client()
        self._slots = threading.BoundedSemaphore(job_slots)

    def query(self, adql, *, uploads=None, maxrec=None, sync=False, cache=True):
        """Run adql and return an astropy Table.

        uploads maps table names to astropy Tables (referenced as TAP_UPLOAD.<name>).
        sync=True suits small, fast queries; everything else runs as an async job, which is deleted
        from the server once its result is fetched.
        """
        check_blind()
        key = self.cache_key(adql, uploads=uploads, maxrec=maxrec) if cache else None
        if key:
            hit = self.client.cache.get(key)
            if hit is not None:
                return _parse(hit[0])
        with self._slots:
            raw = self._sync(adql, uploads, maxrec) if sync else self._async(adql, uploads, maxrec)
        table = _parse(raw)
        if key:
            self.client.cache.put(key, raw, {"service": self.url, "rows": len(table)})
        return table

    def cache_key(self, adql, *, uploads=None, maxrec=None):
        ups = {name: table_digest(t) for name, t in (uploads or {}).items()}
        return ResponseCache.key("tap", self.url, normalise_adql(adql), maxrec, ups)

    def _sync(self, adql, uploads, maxrec):
        data = {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": adql, "FORMAT": "votable"}
        if maxrec is not None:
            data["MAXREC"] = str(maxrec)
        files = None
        if uploads:
            data["UPLOAD"] = ";".join(f"{n},param:{n}" for n in uploads)
            files = {n: (f"{n}.xml", _votable_bytes(t), "application/x-votable+xml") for n, t in uploads.items()}
        resp = self.client.post(f"{self.url}/sync", data=data, files=files)
        resp.raise_for_status()
        _raise_votable_error(resp.content)
        return resp.content

    def _async(self, adql, uploads, maxrec):
        ups = {n: io.BytesIO(_votable_bytes(t)) for n, t in (uploads or {}).items()} or None
        job = pyvo.dal.AsyncTAPJob.create(self.url, adql, maxrec=maxrec, uploads=ups, session=self.client)
        try:
            job.run()
            job.wait(timeout=JOB_TIMEOUT)
            job.raise_if_error()
            resp = self.client.get(job.result_uri)
            resp.raise_for_status()
            return resp.content
        finally:
            try:
                job.delete()
            except Exception:
                pass


def normalise_adql(adql):
    return re.sub(r"\s+", " ", adql).strip()


def table_digest(table):
    """Stable digest of a table's column names, dtypes and values."""
    h = hashlib.sha256()
    for name in table.colnames:
        col = np.asarray(table[name])
        h.update(name.encode())
        h.update(str(col.dtype).encode())
        h.update(np.ascontiguousarray(col).tobytes())
    return h.hexdigest()


def _votable_bytes(table):
    buf = io.BytesIO()
    Table(table).write(buf, format="votable")
    return buf.getvalue()


def _parse(raw):
    """Parse a VOTable. Column names are lower-cased (ADQL identifiers are case-insensitive, and Gaia
    returns SOURCE_ID from async jobs but source_id from sync ones). FIELD names win over IDs, since IRSA
    sets ID="col_N" alongside the real name."""
    table = votable.parse_single_table(io.BytesIO(raw)).to_table(use_names_over_ids=True)
    table.rename_columns(table.colnames, [c.lower() for c in table.colnames])
    return table


def _raise_votable_error(raw):
    head = raw[:4096]
    if b'name="QUERY_STATUS"' in head and b'value="ERROR"' in head:
        msg = re.search(rb'value="ERROR"[^>]*>(.*?)</INFO>', raw[:65536], re.S)
        raise pyvo.DALQueryError(msg.group(1).decode(errors="replace").strip() if msg else "TAP query error")
