import hashlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from astropy.table import Table

from cosmos_search.net import (
    USER_AGENT, BlindModeError, BudgetExhausted, Client, HostBlocked, HostPolicy, IntegrityError,
    ResponseCache, Tap, s3_list,
)
from cosmos_search.net import policy as policy_mod


class FakeArchive:
    """A local HTTP server with scripted responses: path -> list of (status, headers, body)."""

    def __init__(self):
        self.script, self.hits, self.agents, self.bodies = {}, {}, [], []
        self.active = self.max_active = 0
        self.delay = 0.0
        lock = threading.Lock()
        archive = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _serve(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                with lock:
                    archive.active += 1
                    archive.max_active = max(archive.max_active, archive.active)
                    archive.agents.append(self.headers.get("User-Agent"))
                    archive.bodies.append(body)
                    path = self.path.split("?")[0]
                    k = archive.hits.get(path, 0)
                    archive.hits[path] = k + 1
                    steps = archive.script.get(path, [(404, {}, b"missing")])
                    status, headers, payload = steps[min(k, len(steps) - 1)]
                    if callable(payload):
                        payload = payload(self.path)
                time.sleep(archive.delay)
                self.send_response(status)
                for h, v in headers.items():
                    self.send_header(h, v)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                with lock:
                    archive.active -= 1

            do_GET = do_POST = do_DELETE = _serve

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def archive():
    a = FakeArchive()
    yield a
    a.close()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("COSMOS_BLIND", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


def make_client(tmp_path, **policy):
    sleeps = []
    pol = HostPolicy(**{"backoff_base": 0.01, "backoff_cap": 0.05, **policy})
    c = Client(cache=ResponseCache(tmp_path / "cache"), policies={"127.0.0.1": pol},
               sleep=lambda s: sleeps.append(s))
    c.sleeps = sleeps
    return c


def test_user_agent_carries_contact(archive, tmp_path):
    archive.script["/x"] = [(200, {}, b"ok")]
    make_client(tmp_path).get(f"{archive.url}/x")
    assert archive.agents == [USER_AGENT]
    assert "mailto:cora.shallit@gmail.com" in USER_AGENT


def test_retries_transient_errors_then_succeeds(archive, tmp_path):
    archive.script["/x"] = [(503, {}, b""), (502, {}, b""), (200, {}, b"ok")]
    c = make_client(tmp_path)
    assert c.get_bytes(f"{archive.url}/x") == b"ok"
    assert archive.hits["/x"] == 3
    assert len(c.sleeps) == 2 and all(0 <= s <= 0.05 for s in c.sleeps)


def test_honours_retry_after(archive, tmp_path):
    archive.script["/x"] = [(429, {"Retry-After": "7"}, b""), (200, {}, b"ok")]
    c = make_client(tmp_path)
    c.get(f"{archive.url}/x")
    assert c.sleeps == [7.0]


def test_gives_up_and_returns_last_response(archive, tmp_path):
    archive.script["/x"] = [(503, {}, b"")]
    c = make_client(tmp_path, max_retries=2, breaker_threshold=99)
    assert c.get(f"{archive.url}/x").status_code == 503
    assert archive.hits["/x"] == 3


def test_connection_errors_are_retried_then_raised(tmp_path):
    c = make_client(tmp_path, max_retries=2)
    with pytest.raises(Exception):
        c.get("http://127.0.0.1:9/unreachable", timeout=1)
    assert len(c.sleeps) == 2


def test_breaker_opens_on_blacklist_signals(archive, tmp_path):
    archive.script["/x"] = [(403, {}, b"blacklisted")]
    c = make_client(tmp_path, block_statuses=frozenset({403}), breaker_threshold=2, max_retries=5)
    with pytest.raises(HostBlocked):
        c.get(f"{archive.url}/x")
    assert archive.hits["/x"] == 2
    with pytest.raises(HostBlocked):
        c.get(f"{archive.url}/other")
    assert "/other" not in archive.hits


def test_daily_budget_pauses_host(archive, tmp_path):
    archive.script["/ads"] = [(200, {"X-RateLimit-Remaining": "10"}, b"{}")]
    c = make_client(tmp_path, budget_reserve=50)
    c.get(f"{archive.url}/ads")
    with pytest.raises(BudgetExhausted):
        c.get(f"{archive.url}/ads")


def test_cache_is_opt_in_and_keyed_on_body(archive, tmp_path):
    archive.script["/q"] = [(200, {"Content-Type": "text/plain"}, lambda p: f"{time.time()}".encode())]
    c = make_client(tmp_path)
    a = c.post(f"{archive.url}/q", data={"q": "1"}, cache=True).content
    b = c.post(f"{archive.url}/q", data={"q": "1"}, cache=True)
    assert b.content == a and b.from_cache and b.headers["Content-Type"] == "text/plain"
    c.post(f"{archive.url}/q", data={"q": "2"}, cache=True)
    c.post(f"{archive.url}/q", data={"q": "1"})
    assert archive.hits["/q"] == 3


def test_cache_key_ignores_param_order(archive, tmp_path):
    archive.script["/q"] = [(200, {}, b"x")]
    c = make_client(tmp_path)
    c.get(f"{archive.url}/q", params={"a": 1, "b": 2}, cache=True)
    c.get(f"{archive.url}/q", params={"b": 2, "a": 1}, cache=True)
    assert archive.hits["/q"] == 1


def test_errors_are_not_cached(archive, tmp_path):
    archive.script["/q"] = [(404, {}, b"no")]
    c = make_client(tmp_path)
    assert c.get_bytes(f"{archive.url}/q", cache=True, missing_ok=True) is None
    assert c.get_bytes(f"{archive.url}/q", cache=True, missing_ok=True) is None
    assert archive.hits["/q"] == 2


def test_blind_mode_refuses_network_and_cache(archive, tmp_path, monkeypatch):
    archive.script["/q"] = [(200, {}, b"real data")]
    c = make_client(tmp_path)
    c.get(f"{archive.url}/q", cache=True)
    monkeypatch.setenv("COSMOS_BLIND", "1")
    with pytest.raises(BlindModeError):
        c.get(f"{archive.url}/q", cache=True)
    with pytest.raises(BlindModeError):
        c.cache.get("anything")
    assert archive.hits["/q"] == 1


def test_concurrency_limit_holds_across_threads(archive, tmp_path):
    archive.script["/slow"] = [(200, {}, b"ok")]
    archive.delay = 0.05
    c = Client(cache=ResponseCache(tmp_path / "cache"), policies={"127.0.0.1": HostPolicy(max_concurrent=3)})
    threads = [threading.Thread(target=c.get, args=(f"{archive.url}/slow",)) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert archive.hits["/slow"] == 12
    assert archive.max_active <= 3


def test_min_interval_spaces_request_starts(tmp_path):
    clock = [0.0]

    def sleep(s):
        clock[0] += s

    c = Client(cache=ResponseCache(tmp_path), policies={"127.0.0.1": HostPolicy(min_interval=0.5)},
               clock=lambda: clock[0], sleep=sleep)
    lim = c._limiter("http://127.0.0.1/x")
    starts = []
    for _ in range(4):
        with lim.slot():
            starts.append(clock[0])
    assert np.allclose(np.diff(starts), 0.5)


def test_download_checks_md5_and_retries(archive, tmp_path):
    good = b"x" * 5000
    archive.script["/f"] = [(200, {}, b"corrupt"), (200, {}, good)]
    c = make_client(tmp_path)
    path = c.download(f"{archive.url}/f", tmp_path / "f.bin", md5=hashlib.md5(good).hexdigest())
    assert path.read_bytes() == good and archive.hits["/f"] == 2
    archive.script["/g"] = [(200, {}, b"always wrong")]
    with pytest.raises(IntegrityError):
        make_client(tmp_path, max_retries=1).download(f"{archive.url}/g", md5="0" * 32)
    assert not list(tmp_path.glob(".dl-*"))


def test_streamed_deletes_temp_file(archive, tmp_path):
    archive.script["/f"] = [(200, {}, b"data")]
    with make_client(tmp_path).streamed(f"{archive.url}/f") as p:
        assert p.read_bytes() == b"data"
    assert not p.exists()


def test_s3_list_follows_continuation_tokens(archive, tmp_path):
    def page(path):
        token = "continuation-token=t2" in path
        keys = ["b/2.fits", "b/3.fits"] if token else ["a/0.fits", "a/1.fits"]
        contents = "".join(f"<Contents><Key>{k}</Key><Size>10</Size></Contents>" for k in keys)
        more = "" if token else "<NextContinuationToken>t2</NextContinuationToken>"
        return (f'<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                f"<IsTruncated>{'false' if token else 'true'}</IsTruncated>{more}{contents}"
                "</ListBucketResult>").encode()

    archive.script["/"] = [(200, {}, page)]
    entries = list(s3_list(archive.url, "a/", client=make_client(tmp_path)))
    assert [e.key for e in entries] == ["a/0.fits", "a/1.fits", "b/2.fits", "b/3.fits"]
    assert all(e.size == 10 and not e.is_prefix for e in entries)


def test_policy_matching():
    _, tesscut = policy_mod.match("https://mast.stsci.edu/tesscut/api/v0.1/astrocut")
    _, mast = policy_mod.match("https://mast.stsci.edu/api/v0/invoke")
    _, s3 = policy_mod.match("https://stpubdata.s3.amazonaws.com/tess/")
    assert (tesscut.max_concurrent, mast.max_concurrent, s3.max_concurrent) == (8, 4, 32)
    k1, cds = policy_mod.limiter_key("https://simbad.cds.unistra.fr/simbad/sim-tap")
    k2, _ = policy_mod.limiter_key("https://tapvizier.cds.unistra.fr/TAPVizieR/tap")
    assert k1 == k2 and cds.min_interval >= 1 / 6
    assert policy_mod.match("https://example.org/")[1] == policy_mod.DEFAULT


VOTABLE = b"""<?xml version="1.0"?>
<VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3"><RESOURCE type="results">
<INFO name="QUERY_STATUS" value="OK"/><TABLE><FIELD name="source_id" datatype="long"/>
<FIELD name="g" datatype="double"/><DATA><TABLEDATA>
<TR><TD>1</TD><TD>12.5</TD></TR><TR><TD>2</TD><TD>13.0</TD></TR>
</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>"""


def test_tap_sync_query_is_cached(archive, tmp_path):
    archive.script["/tap/sync"] = [(200, {"Content-Type": "application/x-votable+xml"}, VOTABLE)]
    tap = Tap(f"{archive.url}/tap", client=make_client(tmp_path))
    t1 = tap.query("SELECT  source_id, g\nFROM x", sync=True)
    t2 = tap.query("SELECT source_id, g FROM x", sync=True)
    assert list(t1["source_id"]) == [1, 2] and list(t2["g"]) == [12.5, 13.0]
    assert archive.hits["/tap/sync"] == 1


def test_tap_upload_changes_cache_key(tmp_path):
    tap = Tap("http://127.0.0.1/tap", client=make_client(tmp_path))
    a = Table({"source_id": np.array([1, 2], dtype=np.int64)})
    b = Table({"source_id": np.array([1, 3], dtype=np.int64)})
    q = "SELECT * FROM TAP_UPLOAD.ids"
    assert tap.cache_key(q, uploads={"ids": a}) == tap.cache_key(q, uploads={"ids": a.copy()})
    assert tap.cache_key(q, uploads={"ids": a}) != tap.cache_key(q, uploads={"ids": b})


def test_cache_prunes_least_recently_used(tmp_path):
    cache = ResponseCache(tmp_path, max_bytes=10_000)
    for i in range(30):
        cache.put(cache.key(i), b"x" * 1000)
    assert cache.size() <= 10_000
    assert cache.get(cache.key(29)) is not None


def test_storage_options_carry_user_agent_and_respect_blind(monkeypatch):
    from cosmos_search.net import storage_options
    assert storage_options()["headers"]["User-Agent"] == USER_AGENT
    monkeypatch.setenv("COSMOS_BLIND", "1")
    with pytest.raises(BlindModeError):
        storage_options()
