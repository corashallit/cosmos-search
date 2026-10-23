"""Local HATS mirrors of the surveys, read before the remote copies.

scripts/mirror_hats.py (for the data.lsdb.io and MAST catalogues) and scripts/mirror_tap.py (for tables
only on TAP) write <data dir>/mirror/hats/<survey name>/, a HATS catalogue of the columns the code reads, with a
mirror.json naming its source, columns and row count. A survey is served locally when that mirror is complete and
has every column the read asks for; otherwise the read goes to the remote copy as before.

COSMOS_MIRROR=off disables the mirrors (to compare against the archives, say).
"""

import json
import os
from functools import lru_cache
from pathlib import Path

from cosmos_search.net.local import data_dir


def root():
    d = data_dir()
    return d / "mirror" / "hats" if d else None


@lru_cache(maxsize=None)
def _manifest(path):
    try:
        return json.loads((Path(path) / "mirror.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def manifests():
    """{name: manifest} for every complete mirror on this machine."""
    r = root()
    if r is None or not r.is_dir() or os.environ.get("COSMOS_MIRROR", "").lower() == "off":
        return {}
    out = {}
    for d in sorted(p for p in r.iterdir() if p.is_dir()):
        m = _manifest(str(d))
        if m:
            out[m["name"]] = dict(m, path=str(d))
    return out


def local(name=None, url=None, columns=()):
    """Path of a local mirror for survey `name` (or for remote `url`) holding all `columns`, else None."""
    for m in manifests().values():
        if (name is not None and m["name"] == name) or (url is not None and _same(url, m.get("url"))):
            if set(columns) <= set(m["columns"]):
                return m["path"]
    return None


def resolve(url, columns=()):
    """The local mirror's path for a remote HATS url when one holds the columns, else the url unchanged."""
    return local(url=url, columns=columns) or url


def _same(a, b):
    return bool(a and b) and a.rstrip("/") == b.rstrip("/")
