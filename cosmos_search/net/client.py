"""The shared HTTP client.

`Client` is a requests.Session, so pyvo and astroquery can take it as `session=`. On top of requests it adds:
- a User-Agent with the project contact;
- per-host concurrency and minimum-interval limits (policy.py), shared by every thread in the process;
- retries with full-jitter exponential backoff on 429/5xx and connection errors, honouring Retry-After;
- a circuit breaker that stops hitting a host after repeated blacklist signals or exhausted retries;
- an opt-in response cache (cache=True per call);
- the COSMOS_BLIND refusal.

Use `get_client()` rather than constructing one, so throttle state is shared across the process.
"""

import hashlib
import os
import random
import tempfile
import threading
import time
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from importlib import metadata
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from requests.structures import CaseInsensitiveDict

from . import policy as _policy
from .blind import check_blind
from .cache import ResponseCache

CONTACT = "cora.shallit@gmail.com"

try:
    _VERSION = metadata.version("cosmos-search")
except metadata.PackageNotFoundError:
    _VERSION = "0"

USER_AGENT = f"cosmos-search/{_VERSION} (Gaia DR4 anomaly search; mailto:{CONTACT})"

RETRY_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

# Longest Retry-After we will sleep through inside one call.
MAX_RETRY_AFTER = 3600.0

_CACHED_HEADERS = ("Content-Type", "Content-Encoding", "Last-Modified", "ETag")


class HostBlocked(RuntimeError):
    """The circuit breaker for this host is open; retry after `until` (time.monotonic seconds)."""

    def __init__(self, key, until, reason):
        super().__init__(f"{key}: {reason}; host paused for {max(0.0, until - time.monotonic()):.0f} s")
        self.key, self.until, self.reason = key, until, reason


class BudgetExhausted(HostBlocked):
    """A daily request budget (e.g. ADS) is spent."""


class IntegrityError(requests.exceptions.ConnectionError):
    """A download's checksum did not match; retried like a dropped connection."""


class _Limiter:
    def __init__(self, key, policy, clock, sleep):
        self.key, self.policy = key, policy
        self._clock, self._sleep = clock, sleep
        self._sem = threading.BoundedSemaphore(policy.max_concurrent)
        self._lock = threading.Lock()
        self._next_start = 0.0
        self._strikes = 0
        self._open_until = 0.0
        self._open_reason = ""
        self._open_exc = HostBlocked

    @contextmanager
    def slot(self):
        with self._sem:
            if self.policy.min_interval:
                with self._lock:
                    now = self._clock()
                    start = max(now, self._next_start)
                    self._next_start = start + self.policy.min_interval
                if start > now:
                    self._sleep(start - now)
            yield

    def check_open(self):
        with self._lock:
            if self._clock() < self._open_until:
                raise self._open_exc(self.key, self._open_until, self._open_reason)

    def ok(self):
        with self._lock:
            self._strikes = 0

    def strike(self, reason):
        with self._lock:
            self._strikes += 1
            if self._strikes >= self.policy.breaker_threshold:
                self._open(self.policy.breaker_cooldown, reason, HostBlocked)
                self._strikes = 0

    def pause(self, seconds, reason, exc=HostBlocked):
        with self._lock:
            self._open(seconds, reason, exc)

    def _open(self, seconds, reason, exc):
        self._open_until = self._clock() + seconds
        self._open_reason, self._open_exc = reason, exc


class Client(requests.Session):
    def __init__(self, *, cache=None, policies=None, user_agent=USER_AGENT,
                 clock=time.monotonic, sleep=time.sleep, rng=None):
        super().__init__()
        self.headers["User-Agent"] = user_agent
        self.cache = cache if cache is not None else ResponseCache()
        self.policies = _policy.POLICIES if policies is None else policies
        self._clock, self._sleep = clock, sleep
        self._rng = rng or random.Random()
        self._limiters = {}
        self._limiters_lock = threading.Lock()

    # -- public helpers ------------------------------------------------------------------------------

    def get_bytes(self, url, *, cache=False, missing_ok=False, **kwargs):
        """GET url and return the body. With missing_ok, a 404 returns None."""
        resp = self.get(url, cache=cache, **kwargs)
        if missing_ok and resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content

    def download(self, url, dest=None, *, md5=None, chunk_size=1 << 20, **kwargs):
        """Stream url to dest (default: a temp file) and return its Path.

        The host's concurrency slot is held for the whole transfer, and a dropped connection or md5
        mismatch is retried with backoff like any other transient failure.
        """
        check_blind()
        dest = Path(dest) if dest else None
        kwargs.setdefault("timeout", self._policy_for(url).timeout)

        def send():
            resp = super(Client, self).request("GET", url, stream=True, **kwargs)
            if resp.status_code != 200:
                resp.close()
                return resp
            fd, tmp = tempfile.mkstemp(dir=dest.parent if dest else None, prefix=".dl-")
            h = hashlib.md5() if md5 else None
            try:
                with os.fdopen(fd, "wb") as f:
                    for block in resp.iter_content(chunk_size):
                        f.write(block)
                        if h:
                            h.update(block)
                if h and h.hexdigest() != md5.lower():
                    raise IntegrityError(f"md5 mismatch for {url}")
            except BaseException:
                os.unlink(tmp)
                raise
            finally:
                resp.close()
            resp.path = Path(tmp)
            return resp

        resp = self._send_with_retries(url, send)
        resp.raise_for_status()
        if dest is None:
            return resp.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(resp.path, dest)
        return dest

    @contextmanager
    def streamed(self, url, **kwargs):
        """Download url to a temp file, yield its Path, and delete it afterwards."""
        path = self.download(url, **kwargs)
        try:
            yield path
        finally:
            path.unlink(missing_ok=True)

    # -- requests.Session override -------------------------------------------------------------------

    def request(self, method, url, *args, cache=False, cache_tag=None, **kwargs):
        check_blind()
        if cache and kwargs.get("stream"):
            raise ValueError("cache=True cannot be combined with stream=True; use download()")
        kwargs.setdefault("timeout", self._policy_for(url).timeout)
        key = None
        if cache:
            key = self._cache_key(method, url, cache_tag, kwargs)
            hit = self.cache.get(key)
            if hit is not None:
                return _response_from_cache(url, *hit)
        resp = self._send_with_retries(url, lambda: super(Client, self).request(method, url, *args, **kwargs))
        if key is not None and resp.status_code == 200:
            meta = {"status": 200, "url": resp.url,
                    "headers": {h: resp.headers[h] for h in _CACHED_HEADERS if h in resp.headers}}
            self.cache.put(key, resp.content, meta)
        return resp

    # -- internals -----------------------------------------------------------------------------------

    def _policy_for(self, url):
        return _policy.match(url, self.policies)[1]

    def _limiter(self, url):
        key, pol = _policy.limiter_key(url, self.policies)
        with self._limiters_lock:
            lim = self._limiters.get(key)
            if lim is None:
                lim = self._limiters[key] = _Limiter(key, pol, self._clock, self._sleep)
        return lim

    def _send_with_retries(self, url, send):
        lim = self._limiter(url)
        pol = lim.policy
        resp, exc = None, None
        for attempt in range(pol.max_retries + 1):
            lim.check_open()
            resp, exc = None, None
            with lim.slot():
                try:
                    resp = send()
                except RETRY_EXCEPTIONS as e:
                    exc = e
            if resp is not None:
                status = resp.status_code
                if status in pol.block_statuses:
                    lim.strike(f"HTTP {status} (blacklist signal)")
                elif status not in pol.retry_statuses:
                    lim.ok()
                    self._check_budget(lim, resp)
                    return resp
            if attempt == pol.max_retries:
                break
            self._sleep(self._delay(pol, attempt, resp))
        lim.strike("retries exhausted")
        if resp is not None:
            return resp
        raise exc

    def _delay(self, pol, attempt, resp):
        if resp is not None:
            ra = _retry_after(resp.headers.get("Retry-After"))
            if ra is not None:
                return min(ra, MAX_RETRY_AFTER)
        return self._rng.uniform(0, min(pol.backoff_cap, pol.backoff_base * 2 ** attempt))

    def _check_budget(self, lim, resp):
        reserve = lim.policy.budget_reserve
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if reserve is None or remaining is None:
            return
        try:
            remaining = int(remaining)
        except ValueError:
            return
        if remaining <= reserve:
            try:
                wait = float(resp.headers.get("X-RateLimit-Reset", 0)) - time.time()
            except ValueError:
                wait = 0
            lim.pause(max(wait, 60.0), f"daily budget down to {remaining}", BudgetExhausted)

    @staticmethod
    def _cache_key(method, url, tag, kwargs):
        prepared = requests.Request(method.upper(), url, params=kwargs.get("params"),
                                    data=kwargs.get("data"), json=kwargs.get("json")).prepare()
        parts = urlsplit(prepared.url)
        query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
        body = prepared.body.encode() if isinstance(prepared.body, str) else prepared.body
        return ResponseCache.key("http", tag, method.upper(), urlunsplit(parts._replace(query=query)), body)


def _retry_after(value):
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def _response_from_cache(url, body, meta):
    resp = requests.Response()
    resp.status_code = meta.get("status", 200)
    resp._content = body
    resp.headers = CaseInsensitiveDict(meta.get("headers", {}))
    resp.url = meta.get("url", url)
    resp.reason = "OK (cached)"
    resp.from_cache = True
    return resp


_default = None
_default_lock = threading.Lock()


def get_client():
    """The process-wide client. All modules should use this one so limits are shared."""
    global _default
    with _default_lock:
        if _default is None:
            _default = Client()
        return _default


def storage_options():
    """fsspec storage_options for reads that bypass the client (lsdb/HATS over aiohttp).

    Carries the User-Agent and enforces the blind-run refusal; callers cap their own concurrency to the
    policy table in docs/infra/net.md.
    """
    check_blind()
    return {"headers": {"User-Agent": USER_AGENT}}
