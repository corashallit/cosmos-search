"""ASAS-SN Sky Patrol v2 (Hart et al. 2023, arXiv:2304.03791): aperture photometry of ~10^8 catalogued stars on
every ASAS-SN image, V band from 2012 to 2018 and g band from 2017 on, the whole sky every night or two.

It is the one survey here that watched bright stars (about 10.5 to 16.5 mag) all over the sky through Gaia's whole
span and ever since, so for a bright candidate it answers both "same years" and "after".

Where it answers from. The service is plain http on port 9006 of asassn-lb01.ifa.hawaii.edu, the address the
project's own client (`skypatrol` on PyPI) uses; nothing is served on ports 80 or 443. Cloud containers reach
only those two ports, so from one this reader raises `Unreachable` after a single connection attempt. It works
from a machine with ordinary network access (kore).

The calls are those of the `skypatrol` client, written out here so that they go through the shared client
(User-Agent, limits, cache) and need no extra dependency:

1. POST /lookup_cone/radius<deg>_ra<deg>_dec<deg> with JSON {catalog, cols, format: "arrow", download: true}
   answers a Parquet table of the catalogued targets in the cone.
2. GET /get_block_servers answers the list of hosts that serve light curves.
3. GET http://<host>:9006/get_block/query_hash-<h>-block_idx-0-catalog-<catalog> answers a Parquet table of
   epochs for every target of that cone, where <h> is the base64 of the query's description (`query_hash`).

Epoch columns: asas_sn_id, jd, flux (mJy), flux_err, mag, mag_err, limit, fwhm, image_id, camera, quality.
Fluxes are measured at the catalogue position on every image, detected or not, so a fading star keeps its epochs;
this reader uses flux, never mag. Kept epochs: quality "G" (good image). Cameras ba to bh observed in V, the rest
in g (`skypatrol`'s own table); each camera is its own group, since their zero points differ by up to a few
percent and a camera change in 2017-2018 would otherwise read as a step.

ASAS-SN's pixels are 8 arcsec and its point-spread function about 16 arcsec, so neighbours matter more here than
anywhere else: `slow.independent` divides the expected change by 1 + (neighbours' flux inside that beam over the
star's).
"""

import base64
import io
import socket

import numpy as np
import pandas as pd

from cosmos_search.net import ResponseCache, check_blind, get_client

from . import Unreachable, empty, table

HOST = "asassn-lb01.ifa.hawaii.edu"
PORT = 9006
CATALOG = "master_list"
COLS = ["asas_sn_id", "ra_deg", "dec_deg", "catalog_sources"]
SEARCH_ARCSEC = 8.0     # one ASAS-SN pixel
MATCH_ARCSEC = 3.0      # catalogue positions are Gaia's or better than an arcsec; Gaia's is propagated to ~2019
V_CAMERAS = frozenset({"ba", "bb", "bc", "bd", "be", "bf", "bg", "bh"})
PROBE_SECONDS = 8.0

_reachable = {}


def base_url(host=HOST, port=PORT):
    return f"http://{host}:{port}"


def reachable(host=HOST, port=PORT, timeout=PROBE_SECONDS):
    """Whether a TCP connection to the service opens, tried once per process and host."""
    check_blind()
    key = (host, port)
    if key not in _reachable:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                _reachable[key] = True
        except OSError:
            _reachable[key] = False
    return _reachable[key]


def cone(ra, dec, radius_arcsec=SEARCH_ARCSEC):
    """The cone as the three strings the service takes in the URL and in the query's description."""
    return repr(round(radius_arcsec / 3600, 8)), repr(round(float(ra), 6)), repr(round(float(dec), 6))


def query_hash(radius, ra, dec, catalog=CATALOG, cols=COLS):
    """The name under which the service keeps a cone query's light curves: base64 (with the line breaks Python's
    `encodebytes` adds) of the query's description, exactly as the `skypatrol` client builds it."""
    text = f"conectr-{radius}_conera-{ra}_conedec-{dec}|catalog-{catalog}|cols-" + "/".join(cols)
    return base64.encodebytes(text.encode()).decode()


def pick_target(index, ra, dec, max_arcsec=MATCH_ARCSEC):
    """asas_sn_id of the catalogued target nearest (ra, dec) within max_arcsec, or None."""
    if not len(index):
        return None
    sep = 3600 * np.hypot((index["ra_deg"].to_numpy(float) - ra) * np.cos(np.radians(dec)),
                          index["dec_deg"].to_numpy(float) - dec)
    return index["asas_sn_id"].to_numpy()[np.argmin(sep)] if sep.min() <= max_arcsec else None


def request(ra, dec, client=None, host=HOST, port=PORT):
    """(index of targets in the cone, epochs of all of them) as DataFrames; the epochs are empty when the cone
    holds no catalogued target."""
    client = client or get_client()
    if not reachable(host, port):
        raise Unreachable(
            f"ASAS-SN Sky Patrol answers only on {host}:{port} (plain http), and this machine cannot open that "
            "port. Cloud containers reach ports 80 and 443 only; run this from a machine with ordinary network "
            "access (kore).")
    radius, ra_s, dec_s = cone(ra, dec)
    api = base_url(host, port)
    resp = client.post(f"{api}/lookup_cone/radius{radius}_ra{ra_s}_dec{dec_s}",
                       json={"catalog": CATALOG, "cols": COLS, "format": "arrow", "download": True})
    resp.raise_for_status()
    index = pd.read_parquet(io.BytesIO(resp.content))
    if not len(index):
        return index, pd.DataFrame()
    servers = client.get(f"{api}/get_block_servers").json() or [host]
    qh = query_hash(radius, ra_s, dec_s)
    last = None
    for server in servers:
        try:
            resp = client.get(f"http://{server}:{port}/get_block/query_hash-{qh}-block_idx-0-catalog-{CATALOG}")
            resp.raise_for_status()
            return index, pd.read_parquet(io.BytesIO(resp.content))
        except Exception as e:  # noqa: BLE001 (the block servers are interchangeable: try the next)
            last = e
    raise last


def clean(rows, source_id):
    """Good-image epochs as a COLUMNS table: band asassn_V or asassn_g by camera, one group per camera."""
    if not len(rows):
        return empty()
    r = rows[rows["quality"].astype(str) == "G"] if "quality" in rows else rows
    cam = r["camera"].astype(str)
    band = np.where(cam.isin(V_CAMERAS), "asassn_V", "asassn_g")
    mag = pd.to_numeric(r["mag"], errors="coerce").to_numpy(float) if "mag" in r else None
    if mag is not None:
        mag = np.where(mag < 90, mag, np.nan)    # 99.99 marks a non-detection
    return table(source_id, "asassn", band, cam.to_numpy(), r["jd"].to_numpy(float) - 2457000.0,
                 r["flux"].to_numpy(float), r["flux_err"].to_numpy(float), mag)


def fetch(source_id, ra, dec, cache=None, client=None, host=HOST, port=PORT):
    """ASAS-SN epochs of the catalogued target at (ra, dec), the Gaia position at about 2019. The target's
    epochs are kept in the response cache on machines with a local data store (or with cache=True)."""
    from cosmos_search.net.local import persistent

    client = client or get_client()
    cache = persistent() if cache is None else cache
    key = ResponseCache.key("asassn", *cone(ra, dec))
    hit = client.cache.get(key) if cache else None
    if hit is not None:
        rows = pd.read_parquet(io.BytesIO(hit[0]))
    else:
        index, epochs = request(ra, dec, client=client, host=host, port=port)
        target = pick_target(index, ra, dec)
        rows = epochs[epochs["asas_sn_id"] == target] if target is not None and len(epochs) else pd.DataFrame()
        if cache:
            buf = io.BytesIO()
            rows.to_parquet(buf, index=False)
            client.cache.put(key, buf.getvalue(), {"service": "asassn", "rows": len(rows)})
    return clean(rows, source_id)
