"""Known answers for the slow-change search, matched to Gaia DR3 source ids.

    python -m cosmos_search.slow known --out known.parquet

Three sources, all with changes inside DR4's span (2014-07-25 to 2020-01-20) or known to recur:

- Boyajian's star (KIC 8462852): slow fading in Kepler and after, dips in 2017-2019. Not flagged variable in DR3,
  so DR3 has no light curve for it; DR4 will.
- Gaia Science Alerts whose alert magnitude is fainter than the historic one, raised within DR4's span. Alerts
  need a change of about a magnitude, so these are the large-amplitude end only.
- VSX stars typed R CrB, UX Ori or DY Per: deep, irregular, dust-driven fades.

Each row keeps the nearest Gaia DR3 source within RADIUS_ARCSEC of the catalogue position. VSX positions are
J2000 and alert positions come from Gaia itself, so a few arcsec covers both; `sep_arcsec` and the G magnitude
let a reader reject a wrong neighbour.
"""

import io

import numpy as np
import pandas as pd

from cosmos_search.net import get_client

ALERTS_URL = "http://gsaweb.ast.cam.ac.uk/alerts/alerts.csv"
VSX_HATS = "https://data.lsdb.io/hats/vsx/2026-06"
DR4_START, DR4_END = "2014-07-25", "2020-01-20"
RADIUS_ARCSEC = 2.0
VSX_KINDS = {"RCB": "rcb", "UXOR": "uxor", "DYPER": "dyper"}

BOYAJIAN = {"name": "KIC 8462852", "kind": "boyajian", "ra": 301.564350, "dec": 44.456810,
            "note": "Boyajian's star: slow fading plus dips of up to 22% (Kepler), 1-3% dips in 2017-2019"}


def fading_alerts(text, start=DR4_START, end=DR4_END):
    """Gaia Science Alerts that faded (alert magnitude fainter than historic), raised between start and end."""
    a = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    a.columns = [c.strip("#").strip() for c in a.columns]
    a["Date"] = pd.to_datetime(a["Date"], errors="coerce")
    sel = (a["Date"] >= start) & (a["Date"] <= end) & (a["AlertMag"] > a["HistoricMag"])
    f = a[sel]
    return pd.DataFrame({"name": f["Name"], "kind": "alert_fade", "ra": f["RaDeg"], "dec": f["DecDeg"],
                         "note": (f["Date"].dt.strftime("%Y-%m-%d") + ": " + f["Comment"].fillna("")
                                  + " [" + f["Class"].fillna("") + "; dG "
                                  + (f["AlertMag"] - f["HistoricMag"]).round(2).astype(str) + "]")}
                        ).reset_index(drop=True)


def vsx_kind(vsx_type):
    """'rcb', 'uxor' or 'dyper' if any component of a VSX type string is one of those classes, else None."""
    parts = [p.rstrip(":").upper() for p in pd.Series([vsx_type]).str.split(r"[|+/ ]").iloc[0]]
    for p in parts:
        if p in VSX_KINDS:
            return VSX_KINDS[p]
    return None


def vsx_slow(hats=VSX_HATS):
    """VSX stars typed R CrB, UX Ori or DY Per (uncertain types included)."""
    import lsdb

    df = lsdb.read_hats(hats, columns=["OID", "Name", "Type", "RAdeg", "DEdeg", "MagMax", "MagMin",
                                       "MaxPassband"]).compute()
    df = df[df["Type"].astype(str).str.contains("RCB|UXOR|DYP", case=False)]
    kind = df["Type"].astype(str).map(vsx_kind)
    df = df[kind.notna()]
    return pd.DataFrame({"name": df["Name"].astype(str), "kind": kind[kind.notna()], "ra": df["RAdeg"],
                         "dec": df["DEdeg"],
                         "note": df["Type"].astype(str) + " " + df["MagMax"].astype(str) + "-"
                                 + df["MagMin"].astype(str) + " " + df["MaxPassband"].astype(str)}
                        ).reset_index(drop=True)


def match_gaia(known, tap, radius_arcsec=RADIUS_ARCSEC):
    """Nearest Gaia DR3 source within radius_arcsec of each row (source_id <NA> when none).

    One cone query per row: the archive answers a single cone in ~0.3 s, while a positional join against an
    uploaded table timed out (HTTP 408) even for 50 rows on 2026-10-23.
    """
    rows = []
    for i, r in enumerate(known.itertuples(index=False)):
        res = tap.query(
            "SELECT source_id, phot_g_mean_mag AS g_mag, bp_rp, phot_variable_flag, has_epoch_photometry, "
            f"DISTANCE(POINT({r.ra}, {r.dec}), POINT(ra, dec)) * 3600 AS sep_arcsec FROM gaiadr3.gaia_source "
            f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {r.ra}, {r.dec}, {radius_arcsec / 3600}))",
            sync=True).to_pandas()
        if len(res):
            rows.append(res.sort_values("sep_arcsec").iloc[0].to_dict() | {"row": i})
    m = pd.DataFrame(rows)
    m["phot_variable_flag"] = m["phot_variable_flag"].astype(str)
    res = known.reset_index(drop=True).join(m.set_index("row"))
    res["source_id"] = res["source_id"].astype("Int64")
    return res


def build(tap, alerts_text=None, hats=VSX_HATS):
    """All known answers as one DataFrame, matched to Gaia DR3."""
    if alerts_text is None:
        alerts_text = get_client().get(ALERTS_URL).text
    known = pd.concat([pd.DataFrame([BOYAJIAN]), fading_alerts(alerts_text), vsx_slow(hats)], ignore_index=True)
    return match_gaia(known, tap)
