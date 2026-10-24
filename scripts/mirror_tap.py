"""Mirror whole TAP tables the cross-match reads by cone (ASAS-SN variables on VizieR, SIMBAD basic) as local HATS.

    cd ~/cosmos-data/cosmos-search
    .venv/bin/python -I scripts/mirror_tap.py asassn_vars simbad

The survey's columns (cosmos_search.xmatch.surveys) are fetched in chunks through the repo's TAP client (CDS
limits: 2 at once, 0.5 s apart), renamed back to the survey's spelling, and written with lsdb as a HATS catalogue
under ~/cosmos-data/mirror/hats/<name>, with the same mirror.json as mirror_hats.py writes. A rerun reuses chunks
already fetched (they sit in the persistent TAP cache) and rebuilds the catalogue.
"""

import argparse
import io
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from cosmos_search.net import Tap
from cosmos_search.net.local import data_dir
from cosmos_search.xmatch.surveys import SURVEYS

ROOT = (data_dir() or Path.home() / "cosmos-data") / "mirror" / "hats"
# name -> (chunk column, chunk edges or None for one query)
CHUNKS = {
    "asassn_vars": (None, None),
    "simbad": ("oid", 1_000_000),
}


def quote(c):
    return f'"{c}"' if not c.isidentifier() or c != c.lower() else c


def fetch(name):
    s = SURVEYS[name]
    url, table = s.tap
    tap = Tap(url)
    cols = ", ".join(quote(c) for c in s.columns)
    key, step = CHUNKS[name]
    if key is None:
        t = tap.query(f"SELECT {cols} FROM {table}", sync=True, maxrec=2_000_000)
        if len(t) >= 2_000_000:
            raise RuntimeError(f"{name}: hit maxrec")
        parts = [t.to_pandas()]
    else:
        lo_hi = tap.query(f"SELECT MIN({key}) AS lo, MAX({key}) AS hi, COUNT(*) AS n FROM {table}", sync=True)
        lo, hi, n = (int(lo_hi[c][0]) for c in ("lo", "hi", "n"))
        print(f"{name}: {n:,} rows, {key} {lo}-{hi}", flush=True)
        parts, a = [], lo
        while a <= hi:
            b = a + step
            # CSV over sync: SIMBAD's binary VOTable carries a few non-ASCII bytes in char columns that astropy
            # refuses to parse.
            r = tap.client.post(f"{url}/sync", data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
                                                     "MAXREC": "5000000", "QUERY": f"SELECT {cols} FROM {table} "
                                                     f"WHERE {key} >= {a} AND {key} < {b}"},
                                cache=True, timeout=3600)
            r.raise_for_status()
            t = pd.read_csv(io.BytesIO(r.content), encoding="utf-8", encoding_errors="replace", low_memory=False)
            if len(t) >= 5_000_000:
                raise RuntimeError(f"{name}: chunk {a}-{b} hit maxrec; use a smaller step")
            parts.append(t)
            print(f"  {key} < {b}: {sum(map(len, parts)):,} rows", flush=True)
            a = b
    df = pd.concat(parts, ignore_index=True)
    lower = {c.lower(): c for c in s.columns}
    df = df.rename(columns=lambda c: lower.get(c, c))
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode() if isinstance(v, bytes) else v).astype("string")
    return s, df


def build(name):
    import lsdb
    import pyarrow.parquet as pq

    s, df = fetch(name)
    df = df[df[s.ra].notna() & df[s.dec].notna()].reset_index(drop=True)
    out = ROOT / name
    tmp = ROOT / f".{name}.building"
    shutil.rmtree(tmp, ignore_errors=True)
    cat = lsdb.from_dataframe(df, ra_column=s.ra, dec_column=s.dec, catalog_name=name)
    cat.write_catalog(tmp)
    files = sorted((tmp / name / "dataset").glob("Norder=*/Dir=*/Npix=*.parquet"))  # a collection: catalogue + margin
    schema = pq.read_schema(files[0])
    manifest = {"name": name, "source": f"{s.tap[0]} {s.tap[1]}", "url": None, "columns": schema.names,
                "rows": len(df), "partitions": len(files), "bytes": sum(f.stat().st_size for f in files),
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (tmp / "mirror.json").write_text(json.dumps(manifest, indent=1))
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(f"{name}: complete, {len(df):,} rows, {manifest['bytes'] / 1e9:.2f} GB on disk", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=sorted(CHUNKS))
    a = ap.parse_args(argv)
    for n in a.names:
        build(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
