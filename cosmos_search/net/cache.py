"""Content-addressed response cache on local disk.

Keys are SHA-256 over the request's identity (service, method, normalised URL, body). Entries are written
atomically, so threads and processes can share one cache directory. The cache is pruned oldest-first
(by last access) when it grows past max_bytes, because local disk is small and ephemeral.
"""

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path

from .blind import check_blind

DEFAULT_MAX_GB = 5.0


def default_root():
    """COSMOS_CACHE_DIR, else the local data store's cache/http (net/local.py), else ~/.cache."""
    from .local import data_dir

    root = os.environ.get("COSMOS_CACHE_DIR")
    if root:
        return Path(root)
    store = data_dir()
    return store / "cache" / "http" if store else Path.home() / ".cache" / "cosmos-search" / "http"


def default_max_gb():
    from .local import PERSISTENT_CACHE_GB, data_dir

    return PERSISTENT_CACHE_GB if data_dir() else DEFAULT_MAX_GB


class ResponseCache:
    def __init__(self, root=None, max_bytes=None):
        self.root = Path(root) if root else default_root()
        if max_bytes is None:
            max_bytes = float(os.environ.get("COSMOS_CACHE_MAX_GB", default_max_gb())) * 1e9
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._written = 0

    @staticmethod
    def key(*parts):
        blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=_jsonable)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _paths(self, key):
        d = self.root / key[:2]
        return d / f"{key}.bin", d / f"{key}.json"

    def get(self, key):
        """Return (body, meta) or None."""
        check_blind()
        body_path, meta_path = self._paths(key)
        try:
            meta = json.loads(meta_path.read_text())
            body = body_path.read_bytes()
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        if len(body) != meta.get("size", len(body)):
            return None
        try:
            os.utime(body_path)  # last access drives pruning
        except OSError:
            pass
        return body, meta

    def put(self, key, body, meta=None):
        check_blind()
        body_path, meta_path = self._paths(key)
        body_path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(meta or {}, size=len(body))
        _atomic_write(body_path, body)
        _atomic_write(meta_path, json.dumps(meta, sort_keys=True).encode())
        with self._lock:
            self._written += len(body)
            due = self._written > self.max_bytes // 20
            if due:
                self._written = 0
        if due:
            self.prune()

    def size(self):
        return sum(p.stat().st_size for p in self.root.glob("*/*.bin"))

    def prune(self, target=None):
        """Delete least-recently used entries until the cache is under target bytes (default 90%)."""
        target = int(self.max_bytes * 0.9) if target is None else target
        entries = []
        for p in self.root.glob("*/*.bin"):
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            entries.append((st.st_mtime, st.st_size, p))
        total = sum(e[1] for e in entries)
        for _, size, p in sorted(entries):
            if total <= target:
                break
            for q in (p, p.with_suffix(".json")):
                try:
                    q.unlink()
                except FileNotFoundError:
                    pass
            total -= size
        return total


def _atomic_write(path, data):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _jsonable(obj):
    if isinstance(obj, bytes):
        return {"sha256": hashlib.sha256(obj).hexdigest()}
    return str(obj)
