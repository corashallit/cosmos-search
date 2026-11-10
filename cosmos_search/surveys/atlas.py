"""ATLAS forced photometry (fallingstar-data.com/forcedphot): cyan (c, 420-650 nm) and orange (o, 560-820 nm)
fluxes at a given position on every ATLAS image, 2015-07 on from Hawaii (two telescopes from 2017) and from 2022
also from Chile and South Africa, which opened the sky south of dec -50.

Of the surveys read here it is the one that covers both Gaia's last years and the years since for most of the
sky, at four 30 s exposures a night every one to two nights, from about 12.5 mag (saturation) to 19.

Access needs a free account. The token on the site's "API token" page goes in the header
`Authorization: Token <token>`. This reader takes it from the environment variable ATLAS_TOKEN; without that it
sends no Authorization header, so a network secret for fallingstar-data.com (header `Authorization`, value
`Token <token>`) also works, as the ADS key does. A 401 or 403 raises `MissingCredential`.

One request is a job: POST /queue/ answers 201 with the task's URL (429 with Retry-After when throttled, which
the shared client waits out); the task URL is polled until `finishtimestamp` is set; `result_url` is a text table;
the task is then deleted from the server's queue.

Reduced images, not difference images, by default. Difference photometry measures the change against a reference
image, and ATLAS replaced its reference images near MJD 58417 (2018-10-26) and 58882 (2020-02-03): a constant
star's difference flux steps on those dates, which is exactly a slow change in Gaia's last years (ATLAS FAQ). On
reduced images the flux is the star's own. Its photometry fits one point source at a time, so in crowded fields
the sky can be overestimated (ATLAS FAQ); `slow.independent` treats neighbours inside the point-spread function as
dilution. Difference mode stays available with `use_reduced=False` and needs the star's flux in the reference
image (`ref_ujy`) to turn differences into fractions.

Kept measurements are the ATLAS FAQ's own cuts (`keep`): a finite flux error under 10 mJy, error flag 0, away
from the detector's edges, a point-spread function between 1.6 and 5 pixels, a sensible aperture correction, and
a clear, dark enough sky (5-sigma limit and sky brightness both fainter than 17 mag). One group per site and
filter: the observation name starts with the telescope (01a, 02a Hawaii; 03a South Africa; 04a Chile).
"""

import contextlib
import io
import os
import time

import numpy as np
import pandas as pd

from cosmos_search.net import ResponseCache, get_client

from . import MJD_TO_BTJD, MissingCredential, empty, table

BASE_URL = "https://fallingstar-data.com/forcedphot"
TOKEN_ENV = "ATLAS_TOKEN"
MJD_MIN = 57200.0            # 2015-06-27, before ATLAS's first survey images
POLL_SECONDS = 10.0
TIMEOUT_SECONDS = 3600.0
BANDS = {"o": "atlas_o", "c": "atlas_c"}
COLUMNS = ["MJD", "m", "dm", "uJy", "duJy", "F", "err", "chi/N", "RA", "Dec", "x", "y", "maj", "min", "phi",
           "apfit", "mag5sig", "Sky", "Obs"]


def headers():
    token = os.environ.get(TOKEN_ENV, "").strip()
    h = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"Token {token}"
    return h


def _check(resp, what):
    if resp.status_code in (401, 403):
        raise MissingCredential(
            f"ATLAS refused the {what} (HTTP {resp.status_code}). Set {TOKEN_ENV} to the token from "
            f"{BASE_URL}/apitoken/, or add a network secret for fallingstar-data.com with header Authorization "
            "and value 'Token <token>'; new values reach only sessions started afterwards.")
    resp.raise_for_status()


def check_token(client=None, base_url=BASE_URL):
    """Raise `MissingCredential` unless ATLAS accepts this session's token (one GET of the account's task list)."""
    _check((client or get_client()).get(f"{base_url}/queue/", headers=headers()), "token check")


def request(ra, dec, mjd_min=MJD_MIN, mjd_max=None, use_reduced=True, client=None, base_url=BASE_URL,
            poll=POLL_SECONDS, timeout=TIMEOUT_SECONDS, sleep=time.sleep):
    """Run one forced-photometry job and return the result text."""
    client = client or get_client()
    data = {"ra": f"{ra:.7f}", "dec": f"{dec:.7f}", "mjd_min": f"{mjd_min:.3f}", "use_reduced": bool(use_reduced),
            "send_email": False, "comment": "cosmos-search"}
    if mjd_max is not None:
        data["mjd_max"] = f"{mjd_max:.3f}"
    resp = client.post(f"{base_url}/queue/", headers=headers(), data=data)
    _check(resp, "request")
    task_url = resp.json()["url"]
    waited = 0.0
    try:
        while True:
            resp = client.get(task_url, headers=headers())
            _check(resp, "status check")
            task = resp.json()
            if task.get("finishtimestamp"):
                break
            if waited >= timeout:
                raise TimeoutError(f"ATLAS task {task_url} not finished after {timeout:.0f} s")
            sleep(poll)
            waited += poll
        if task.get("error_msg"):
            # "No data returned" is ATLAS's answer for a position it has no images of.
            if "no data" in str(task["error_msg"]).lower():
                return ""
            raise RuntimeError(f"ATLAS task failed: {task['error_msg']}")
        if not task.get("result_url"):
            return ""
        resp = client.get(task["result_url"], headers=headers())
        _check(resp, "result")
        return resp.text
    finally:
        # Leaving a finished task in the queue listing is harmless, so a failed delete is not an error.
        with contextlib.suppress(Exception):
            client.delete(task_url, headers=headers())


def parse(text):
    """The result text as a DataFrame with COLUMNS (the header line starts with ###)."""
    text = text.replace("###", "").strip()
    if not text:
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(io.StringIO(text), sep=r"\s+")


def keep(r):
    """The ATLAS FAQ's cuts for good measurements, as a boolean mask over the result rows."""
    c = {k: pd.to_numeric(r[k], errors="coerce").to_numpy(float) for k in
         ("duJy", "err", "x", "y", "maj", "min", "apfit", "mag5sig", "Sky")}
    return ((c["duJy"] < 10000) & (c["err"] == 0) & (c["x"] > 100) & (c["x"] < 10460) & (c["y"] > 100)
            & (c["y"] < 10460) & (c["maj"] > 1.6) & (c["maj"] < 5) & (c["min"] > 1.6) & (c["min"] < 5)
            & (c["apfit"] > -1) & (c["apfit"] < -0.1) & (c["mag5sig"] > 17) & (c["Sky"] > 17))


def clean(rows, source_id, ref_ujy=None):
    """Kept measurements (module notes) as a COLUMNS table. ref_ujy: {filter: flux in microjansky of the star in the
    reference image}, for difference-image results, whose fluxes are changes and are turned into ref + change."""
    if not len(rows):
        return empty()
    r = rows[keep(rows)]
    r = r[r["F"].astype(str).isin(list(BANDS)) & (r["duJy"] > 0)]
    if not len(r):
        return empty()
    filt = r["F"].astype(str)
    flux = r["uJy"].to_numpy(float)
    if ref_ujy is not None:
        flux = flux + filt.map(ref_ujy).to_numpy(float)
    band = filt.map(BANDS).to_numpy()
    group = (r["Obs"].astype(str).str[:3] + "_" + filt).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = np.where(flux > 0, 23.9 - 2.5 * np.log10(np.where(flux > 0, flux, 1.0)), np.nan)
    return table(source_id, "atlas", band, group, r["MJD"].to_numpy(float) + MJD_TO_BTJD, flux,
                 r["duJy"].to_numpy(float), mag)


def fetch(source_id, ra, dec, mjd_min=MJD_MIN, mjd_max=None, use_reduced=True, ref_ujy=None, cache=None,
          client=None, **job):
    """ATLAS epochs at (ra, dec), the Gaia position at about 2020.5. The result text is kept in the response
    cache on machines with a local data store (or with cache=True), since a job takes minutes when the queue is
    long."""
    from cosmos_search.net.local import persistent

    if not use_reduced and ref_ujy is None:
        raise ValueError("difference-image photometry needs ref_ujy, the star's flux in the reference image")
    client = client or get_client()
    cache = persistent() if cache is None else cache
    key = ResponseCache.key("atlas", f"{ra:.6f}", f"{dec:.6f}", mjd_min, mjd_max, bool(use_reduced))
    hit = client.cache.get(key) if cache else None
    if hit is not None:
        text = hit[0].decode()
    else:
        text = request(ra, dec, mjd_min, mjd_max, use_reduced, client=client, **job)
        if cache:
            client.cache.put(key, text.encode(), {"service": "atlas"})
    return clean(parse(text), source_id, ref_ujy)
