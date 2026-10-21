"""Predicted Gaia transits for any position and period, from ESA's Gaia Observation Forecast Tool (GOST).

DR4's epoch photometry covers 2014-07-25 to 2020-01-20, but the published commanded scanning law
(Gaia/gedr3/auxiliary/commanded_scan_law) stops at DR3's end (2017-05-28). GOST predicts every field-of-view
transit up to 2025-01-15 from the scanning law Gaia flew, so it gives DR4's cadence for any star before the release:
which visits a dip could land on, and which external epochs fall near them.

A prediction is a transit the scanning law put the star through. Gaia's own data have fewer: dead time, on-board
deletions in dense fields, and transits DPAC rejected. `thin` removes transits to match a retention fraction
measured on DR3 (`retention` compares GOST with DR3's epochs for the same stars).

    from cosmos_search.gaia import gost
    tr = gost.transits(targets)        # targets: DataFrame with source_id, ra, dec (<= 10,000 per call)

The service is a web form: a session cookie, a multipart POST of a CSV (name, ra, dec in degrees, no header), then
the result page polled until the export link appears. A 300-target query over DR4's span took 43 s
(2026-10-21).
"""

import io
import re
import time

import numpy as np
import pandas as pd

BASE = "https://gaia.esac.esa.int/gost/"
MAX_TARGETS = 10000
# GOST rejects a start before its queryable interval (2014-07-25T10:31:26 TCB), so start a few seconds later.
DR4_FROM, DR4_TO = "2014-07-25T10:31:30", "2020-01-20T22:00:00"
DR3_FROM, DR3_TO = "2014-07-25T10:31:30", "2017-05-28T08:46:00"
BTJD0 = 2457000.0
COLUMNS = {"Target": "name", "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]": "bjd",
           "Fov[FovP=preceding/FovF=following]": "fov", "CcdRow[1-7]": "ccd_row",
           "scanAngle[rad]": "scan_angle"}


def _session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "cosmos-search (https://github.com/corashallit/cosmos-search)"
    s.get(BASE, timeout=60).raise_for_status()
    return s


def query(targets, t_from=DR4_FROM, t_to=DR4_TO, poll=5.0, max_wait=3600.0, log=print):
    """Raw GOST rows for up to MAX_TARGETS (source_id, ra, dec) between two UTC ISO times."""
    if len(targets) > MAX_TARGETS:
        raise ValueError(f"GOST takes at most {MAX_TARGETS} targets per query")
    s = _session()
    body = "".join(f"{int(sid)},{ra:.8f},{dec:.8f}\n" for sid, ra, dec in
                   zip(targets["source_id"], targets["ra"], targets["dec"]))
    # The form validates the single-position fields even in file mode, so they need parseable values.
    form = {"inputmode": (None, "file"), "serviceCode": (None, "1"), "srcname": (None, ""),
            "srcra": (None, "0.0"), "srcdec": (None, "0.0"), "from": (None, t_from), "to": (None, t_to),
            "csvfilename": ("targets.csv", body.encode(), "text/csv")}
    r = s.post(BASE + "GostServlet", files=form, timeout=600)
    r.raise_for_status()
    m = re.search(r"Submitted with id (\d+)", r.text)
    if not m:
        raise RuntimeError(f"GOST refused the query: {re.sub('<[^>]+>', ' ', r.text).strip()[:300]}")
    job = m.group(1)
    t0 = time.time()
    while True:
        page = s.get(BASE + "result.jsp", timeout=120).text
        ids = re.findall(r'export\.jsp\?id=([^&"]+/' + job + r')&format=csv"', page)
        if ids:
            break
        if time.time() - t0 > max_wait:
            raise TimeoutError(f"GOST job {job} not done after {max_wait:.0f} s")
        time.sleep(poll)
    csv = s.get(BASE + "export.jsp", params={"id": ids[0], "format": "csv"}, timeout=1200)
    csv.raise_for_status()
    log(f"GOST job {job}: {len(targets)} targets in {time.time() - t0:.0f} s")
    return parse(csv.content)


def parse(content):
    """GOST CSV export -> (source_id, t [BTJD], fov, ccd_row, scan_angle), one row per FoV transit."""
    df = pd.read_csv(io.BytesIO(content), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=COLUMNS)[list(COLUMNS.values())]
    out = pd.DataFrame({"source_id": df["name"].astype("int64"), "t": df["bjd"].astype(float) - BTJD0,
                        "fov": df["fov"].str.strip(), "ccd_row": df["ccd_row"].astype("int8"),
                        "scan_angle": df["scan_angle"].astype(float)})
    return out.sort_values(["source_id", "t"], ignore_index=True)


def transits(targets, t_from=DR4_FROM, t_to=DR4_TO, log=print):
    """Predicted transits for any number of targets, in chunks of MAX_TARGETS."""
    parts = [query(targets.iloc[k:k + MAX_TARGETS], t_from, t_to, log=log)
             for k in range(0, len(targets), MAX_TARGETS)]
    if not parts:
        return pd.DataFrame({"source_id": pd.Series(dtype="int64"), "t": pd.Series(dtype=float),
                             "fov": pd.Series(dtype=object), "ccd_row": pd.Series(dtype="int8"),
                             "scan_angle": pd.Series(dtype=float)})
    return pd.concat(parts, ignore_index=True)


def thin(tr, keep, rng):
    """Drop predicted transits at random so that a fraction `keep` survives (Gaia's losses on DR3's span)."""
    return tr[rng.random(len(tr)) < keep].reset_index(drop=True)


def retention(predicted, observed, tol_days=0.002):
    """Per star: predicted transits, observed G epochs, and the share of predictions with an observed epoch within
    tol_days (3 min). `observed` is a long table (source_id, t) of real epochs over the same span."""
    rows = []
    obs = {sid: np.sort(g["t"].to_numpy()) for sid, g in observed.groupby("source_id")}
    for sid, g in predicted.groupby("source_id"):
        tp = g["t"].to_numpy()
        to = obs.get(sid, np.array([]))
        if len(to):
            j = np.clip(np.searchsorted(to, tp), 1, len(to) - 1) if len(to) > 1 else np.zeros(len(tp), int)
            near = np.minimum(np.abs(to[j] - tp), np.abs(to[np.maximum(j - 1, 0)] - tp))
            hit = near <= tol_days
        else:
            hit = np.zeros(len(tp), bool)
        rows.append({"source_id": sid, "n_pred": len(tp), "n_obs": len(to), "kept": float(hit.mean())})
    return pd.DataFrame(rows)
