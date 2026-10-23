"""Where survey rows come from. Every source answers one question: the rows within a cone.

- HatsSource: HATS Parquet catalogues through lsdb (data.lsdb.io, IRSA/MAST S3). Preferred for anything
  sky-wide: static files, partition pruning, nothing to download.
- TapSource: an ADQL cone on a TAP service through the shared client (IRSA for CatWISE2020 and unWISE).
- MemorySource: a DataFrame already in memory (tests, or a catalogue fetched once for a small field).
- AutoSource: a local mirror when this machine has one (mirror.py), else HATS when the survey has a HATS
  copy, TAP otherwise.

lsdb reads over fsspec/aiohttp, so those reads bypass cosmos_search.net. They still refuse to run under
COSMOS_BLIND, send the project User-Agent, and stay within the per-host limits in docs/infra/net.md
(lsdb runs at most `workers` partition reads at a time; 4 here against a limit of 8 for data.lsdb.io).
"""

import random
import threading
import time

import numpy as np
import pandas as pd

from . import mirror
from .geometry import MAS_PER_DEG, separation_mas

HATS_WORKERS = 4
HATS_RETRIES = 5


def _transient(exc):
    """Whether an exception from a HATS read looks like a busy or flaky server rather than a real error."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in ("clientresponseerror", "429", "500", "502", "503", "504", "timeout",
                                   "timed out", "connection", "reset", "serverdisconnected", "payload"))


def hats_storage_options():
    """fsspec options for HATS reads: blind check first, then the project User-Agent."""
    from cosmos_search import net

    net.check_blind()
    if hasattr(net, "storage_options"):
        return net.storage_options()
    return {"headers": {"User-Agent": net.USER_AGENT}}


class MemorySource:
    def __init__(self, frames):
        """frames: a DataFrame (used for every survey) or {survey name: DataFrame}."""
        self.frames = frames

    def fetch(self, survey, ra, dec, radius_deg):
        df = self.frames if isinstance(self.frames, pd.DataFrame) else self.frames[survey.name]
        sep = separation_mas(ra, dec, np.asarray(df[survey.ra], float), np.asarray(df[survey.dec], float))
        return df[sep <= radius_deg * MAS_PER_DEG]


class HatsSource:
    def __init__(self, urls=None, workers=HATS_WORKERS):
        self.urls = urls or {}
        self.workers = workers
        self._cats = {}
        self._lock = threading.Lock()

    def location(self, survey):
        """Where survey is read from: an explicit url, else a local mirror holding its columns, else its HATS url."""
        if survey.name in self.urls:
            return self.urls[survey.name]
        return mirror.local(name=survey.name, columns=survey.columns) or survey.hats

    def catalog(self, survey):
        import lsdb

        with self._lock:
            if survey.name not in self._cats:
                url = self.location(survey)
                if not url:
                    raise ValueError(f"{survey.name} has no HATS location")
                opts = hats_storage_options()
                if "://" not in str(url):  # local mirror: no HTTP options (the blind check above still ran)
                    opts = None
                self._cats[survey.name] = lsdb.open_catalog(url, columns=survey.columns, storage_options=opts)
            return self._cats[survey.name]

    def fetch(self, survey, ra, dec, radius_deg):
        import dask

        hats_storage_options()  # blind check on every read, not only on open
        cat = self.catalog(survey).cone_search(ra=float(ra), dec=float(dec), radius_arcsec=float(radius_deg) * 3600)
        # fsspec does not retry HTTP errors; a busy static host (429/5xx, dropped connections) gets the same
        # full-jitter backoff the shared client uses.
        for attempt in range(HATS_RETRIES + 1):
            try:
                with dask.config.set(scheduler="threads", num_workers=self.workers):
                    df = cat.compute()
                return df.reset_index(drop=True)
            except Exception as e:  # aiohttp errors arrive wrapped in various types; _transient decides
                if attempt == HATS_RETRIES or not _transient(e):
                    raise
                time.sleep(random.uniform(0, min(60.0, 2.0 * 2**attempt)))


class TapSource:
    def __init__(self):
        self._taps = {}
        self._lock = threading.Lock()

    def tap(self, url):
        from cosmos_search.net import Tap

        with self._lock:
            if url not in self._taps:
                self._taps[url] = Tap(url)
            return self._taps[url]

    def fetch(self, survey, ra, dec, radius_deg):
        url, table = survey.tap
        # Cone on the service's indexed position; the matcher then uses survey.ra/dec.
        cols = ", ".join(f'"{c}"' if not c.isidentifier() or c != c.lower() else c for c in survey.columns)
        pra, pdec = (f'"{c}"' if c != c.lower() else c for c in survey.tap_pos)
        adql = (f"SELECT {cols} FROM {table} WHERE CONTAINS(POINT('ICRS', {pra}, {pdec}), "
                f"CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {radius_deg:.8f})) = 1")
        t = self.tap(url).query(adql, sync=True)
        df = t.to_pandas()
        # TAP lower-cases column names; map back to the survey's spelling.
        lower = {c.lower(): c for c in survey.columns}
        return df.rename(columns=lambda c: lower.get(c, c))


class AutoSource:
    def __init__(self, hats=None, tap=None):
        self.hats = hats or HatsSource()
        self.tapsrc = tap or TapSource()

    def fetch(self, survey, ra, dec, radius_deg):
        if survey.hats or mirror.local(name=survey.name, columns=survey.columns):
            return self.hats.fetch(survey, ra, dec, radius_deg)
        if survey.tap:
            return self.tapsrc.fetch(survey, ra, dec, radius_deg)
        raise ValueError(f"{survey.name}: no HATS or TAP location")
