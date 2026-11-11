"""ATLAS forced photometry (fallingstar-data.com/forcedphot): cyan (c, 420-650 nm) and orange (o, 560-820 nm)
fluxes at a given position on every ATLAS image, 2015-07 on from Hawaii (two telescopes from 2017) and from 2022
also from Chile and South Africa, which opened the sky south of dec -50.

Of the surveys read here it is the one that covers both Gaia's last years and the years since for most of the
sky, at four 30 s exposures a night every one to two nights, from about 12.5 mag (saturation) to 19.

Access needs a free account. The token on the site's "API token" page goes in the header
`Authorization: Token <token>`. This reader takes it from the environment variable ATLAS_TOKEN; without that it
sends no Authorization header, so a network secret for fallingstar-data.com (header `Authorization`, value
`Token <token>`) also works, as the ADS key does. A 401 or 403 raises `MissingCredential`.

Forced photometry measures the flux at a fixed position. With an ATLAS image FWHM near 3.7 arcsec, a star
0.5 arcsec from that position loses about 2.5% of its measured flux, and since the offset grows with time from the
position's epoch, a fixed position turns a proper motion into a slow change: about 2% at the ends of 2015-2026 for
80 mas/yr, 10-15% for 200 mas/yr, which is common among the nearby dwarfs the slow-change search looks at. So the
request carries Gaia's proper motion and the epoch of the position, and the server moves the position to each
image's date (its `propermotion_ra`, `propermotion_dec` and `radec_epoch_year` fields). Checked on live results
2026-11-11 from the RA and Dec it reports per image: a star at dec -48 with Gaia's pmra 10.5 and pmdec 5.6 mas/yr
moved by 10.3 and 6.8 mas/yr, so the server takes pmra with the cos dec in it, as Gaia gives it.

One request is a job: POST /queue/ answers 201 with the task's URL (429 with Retry-After when throttled, which
the shared client waits out); the task URL is polled until `finishtimestamp` is set; `result_url` is a text table;
the task is then deleted from the server's queue.

The queue is shared by every user of the service (5,069 unfinished tasks at 2026-11-11 01:25 UTC). A new task
is placed by how many of the user's tasks are waiting (`userqueuedtasks_on_submit`) and, it seems, by their
recent use: sent with none waiting, a task went in at position 7 the first time and at 74 two hours later, after
four of ours had run; with one, two and three waiting, at about 50, 88 and 120. The head of the queue moved 30 to
60 places an hour, and a task for 2015-2026 ran 18.5 minutes (3,900 images). So a user gets about one task an
hour when the queue is busy, and sending many at once only puts them further back. This module keeps at most
MAX_WAITING tasks of its process waiting to start (two, so that one is near the head while the other runs) and
sends the next when one has started. A rerun first looks in the account's task list for a task asking for the
same thing (a run stopped by a restart leaves its tasks queued, running or finished but not collected) and takes
that one up instead of queueing the job again.

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
import threading
import time

import numpy as np
import pandas as pd

from cosmos_search.net import ResponseCache, get_client

from . import MJD_TO_BTJD, MissingCredential, empty, table

BASE_URL = "https://fallingstar-data.com/forcedphot"
TOKEN_ENV = "ATLAS_TOKEN"
MJD_MIN = 57200.0            # 2015-06-27, before ATLAS's first survey images
POLL_SECONDS = 30.0         # the API guide polls every 10 s; a job takes about 20 minutes to run
TIMEOUT_SECONDS = 6 * 3600.0
MAX_WAITING = 2             # this process's tasks waiting to start, at most (module notes)
_gate = threading.Condition()
_waiting = 0
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


def _hold(block=True):
    """Count one more task waiting to start, first waiting (block) until fewer than MAX_WAITING are."""
    global _waiting
    with _gate:
        if block:
            _gate.wait_for(lambda: _waiting < MAX_WAITING)
        _waiting += 1


def _release():
    global _waiting
    with _gate:
        _waiting -= 1
        _gate.notify_all()


def _same(task, data):
    """Whether an existing task asked for what `data` asks for."""
    def num(v):
        return np.nan if v in (None, "") else float(v)

    pairs = [("ra", 1e-6), ("dec", 1e-6), ("mjd_min", 1e-3), ("mjd_max", 1e-3), ("propermotion_ra", 1e-3),
             ("propermotion_dec", 1e-3), ("radec_epoch_year", 0.01)]
    for k, tol in pairs:
        a, b = num(task.get(k)), num(data.get(k))
        if np.isnan(a) != np.isnan(b) or (not np.isnan(a) and abs(a - b) > tol):
            return False
    return bool(task.get("use_reduced")) == bool(data["use_reduced"])


def _existing(client, base_url, data):
    """The URL of a task still in this account's list that asked for the same thing (one left queued, running or
    finished but not collected by a run that was stopped), so that a rerun takes it up instead of queueing the job
    again; None if there is none."""
    resp = client.get(f"{base_url}/queue/", headers=headers())
    _check(resp, "task list")
    for task in resp.json().get("results", []):
        if _same(task, data):
            return task["url"]
    return None


def _moves(pm):
    """(pmra, pmdec) in mas/yr as floats when the star has a proper motion, else None."""
    if pm is None:
        return None
    pmra, pmdec = (0.0 if v is None or not np.isfinite(v) else float(v) for v in pm)
    return (pmra, pmdec) if pmra or pmdec else None


def request(ra, dec, mjd_min=MJD_MIN, mjd_max=None, use_reduced=True, pm=None, epoch=None, client=None,
            base_url=BASE_URL, poll=POLL_SECONDS, timeout=TIMEOUT_SECONDS, sleep=time.sleep):
    """Run one forced-photometry job and return the result text. pm: (pmra, pmdec) in mas/yr as Gaia gives them
    (pmra includes cos dec), with (ra, dec) the position at `epoch` (a year); the server then moves the position
    to each image's date (module notes)."""
    client = client or get_client()
    data = {"ra": f"{ra:.7f}", "dec": f"{dec:.7f}", "mjd_min": f"{mjd_min:.3f}", "use_reduced": bool(use_reduced),
            "send_email": False, "comment": "cosmos-search"}
    if mjd_max is not None:
        data["mjd_max"] = f"{mjd_max:.3f}"
    moves = _moves(pm)
    if moves:
        if epoch is None:
            raise ValueError("a proper motion needs the epoch of the position")
        data.update(propermotion_ra=f"{moves[0]:.4f}", propermotion_dec=f"{moves[1]:.4f}",
                    radec_epoch_year=f"{float(epoch):.1f}")
    task_url = _existing(client, base_url, data)
    _hold(block=task_url is None)
    started = False
    try:
        if task_url is None:
            resp = client.post(f"{base_url}/queue/", headers=headers(), data=data)
            _check(resp, "request")
            task_url = resp.json()["url"]
        waited = 0.0
        while True:
            resp = client.get(task_url, headers=headers())
            _check(resp, "status check")
            task = resp.json()
            if not started and (task.get("starttimestamp") or task.get("finishtimestamp")):
                started = True
                _release()
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
        if not started:
            _release()
        # Deleting takes a finished task off the account's list and an unstarted one off the queue. A failed
        # delete is not an error: a finished task left in the list costs the server nothing.
        if task_url:
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


def fetch(source_id, ra, dec, mjd_min=MJD_MIN, mjd_max=None, use_reduced=True, ref_ujy=None, pm=None, epoch=None,
          cache=None, client=None, **job):
    """ATLAS epochs at (ra, dec), the Gaia position at `epoch` (`surveys.MEAN_YEAR`, about 2020.5), moved to each
    image's date by pm (mas/yr) when given. The result text is kept in the response cache on machines with a
    local data store (or with cache=True), since a job takes 15-20 minutes of the server's time."""
    from cosmos_search.net.local import persistent

    if not use_reduced and ref_ujy is None:
        raise ValueError("difference-image photometry needs ref_ujy, the star's flux in the reference image")
    client = client or get_client()
    cache = persistent() if cache is None else cache
    moves = _moves(pm)
    key = ResponseCache.key("atlas", f"{ra:.6f}", f"{dec:.6f}", mjd_min, mjd_max, bool(use_reduced),
                            *((f"{moves[0]:.4f}", f"{moves[1]:.4f}", f"{float(epoch):.1f}") if moves else ()))
    hit = client.cache.get(key) if cache else None
    if hit is not None:
        text = hit[0].decode()
    else:
        text = request(ra, dec, mjd_min, mjd_max, use_reduced, pm=moves, epoch=epoch, client=client, **job)
        if cache:
            client.cache.put(key, text.encode(), {"service": "atlas"})
    return clean(parse(text), source_id, ref_ujy)
