"""Gaia DR3 epoch photometry: bulk CDN files and DataLink, parsed to one long-format light-curve table.

Light-curve schema (shared with the TESS arm; one row per observation):

| column | type | meaning |
|---|---|---|
| source_id | int64 | Gaia DR3 source_id |
| band | string | G, BP or RP |
| t | float64 | BTJD = BJD_TDB − 2457000 (Gaia's BJD_TCB − 2455197.5 converted) |
| flux | float64 | e⁻/s as published |
| err | float64 | e⁻/s |
| flags | int32 | 0 = good; bits below |
| transit_id | int64 | Gaia transit; G, BP and RP of one transit share it |
| fov | int8 | 1 = preceding, 2 = following field of view (decoded from transit_id) |
| ccd_row | int8 | 1–7 (decoded from transit_id) |
| n_obs | int8 | CCDs contributing to the transit-averaged G flux (0 for BP/RP) |

Observations with no flux (NaN, i.e. unavailable or rejected by photometry before publication) are dropped.
"""

import csv
import gzip
import io
import itertools
import re

import numpy as np
import pyarrow as pa

from cosmos_search.net import get_client, s3_list

from . import archive

FLAG_VARI_REJECT = 1  # rejected by DPAC variability processing (or negative flux)
FLAG_PHOT_REJECT = 2  # BP/RP rejected by photometry processing
FLAG_NOISY = 4  # G scatter across CCDs larger than expected

BULK_PREFIX = "Gaia/gdr3/Photometry/epoch_photometry/"
BANDS = ("G", "BP", "RP")

SCHEMA = pa.schema([
    ("source_id", pa.int64()), ("band", pa.dictionary(pa.int8(), pa.string())), ("t", pa.float64()),
    ("flux", pa.float64()), ("err", pa.float64()), ("flags", pa.int32()), ("transit_id", pa.int64()),
    ("fov", pa.int8()), ("ccd_row", pa.int8()), ("n_obs", pa.int8()),
])

# TCB -> TDB (IAU 2006 Resolution B3): TDB = TCB - L_B (JD_TCB - T0) 86400 s + TDB0.
_L_B = 1.550519768e-8
_T0 = 2443144.5003725
_TDB0_DAYS = -6.55e-5 / 86400
_GAIA_REF = 2455197.5
_BTJD_REF = 2457000.0


def gaia_time_to_btjd(t):
    jd_tcb = np.asarray(t, dtype=np.float64) + _GAIA_REF
    return jd_tcb - _L_B * (jd_tcb - _T0) + _TDB0_DAYS - _BTJD_REF


def decode_transit_id(transit_id):
    """Return (obmt_ns, fov, ccd_row, af1_pixel) from Gaia transit IDs (DR3 data model §epoch_photometry)."""
    tid = np.asarray(transit_id, dtype=np.int64)
    obmt = 204800 * ((tid >> 17) & 0x000003FFFFFFFFFF)
    fov = 1 + ((tid >> 15) & 0x03)
    row = (tid >> 12) & 0x07
    return obmt, fov.astype(np.int8), row.astype(np.int8), (tid & 0x0FFF).astype(np.int16)


# -- bulk files --------------------------------------------------------------------------------------

_RANGE = re.compile(r"EpochPhotometry_(\d+)-(\d+)\.csv\.gz$")


def bulk_files(client=None):
    """List DR3 bulk files as dicts with name, url, size, md5 and HEALPix-8 range."""
    client = client or get_client()
    md5 = {}
    text = client.get_bytes(f"{archive.CDN_URL}/{BULK_PREFIX}_MD5SUM.txt", cache=True).decode()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            md5[parts[1]] = parts[0]
    files = []
    for e in s3_list(archive.CDN_LIST_URL, BULK_PREFIX, client=client):
        m = _RANGE.search(e.key)
        if not m:
            continue
        name = e.key.rsplit("/", 1)[-1]
        files.append({"name": name, "url": f"{archive.CDN_URL}/{e.key}", "size": e.size,
                      "md5": md5.get(name), "hpx8": (int(m.group(1)), int(m.group(2)))})
    return files


# -- parsing ----------------------------------------------------------------------------------------

_NUMERIC = {
    "transit_id": np.int64, "g_transit_time": np.float64, "g_transit_flux": np.float64,
    "g_transit_flux_error": np.float64, "g_transit_n_obs": np.float64, "bp_obs_time": np.float64,
    "bp_flux": np.float64, "bp_flux_error": np.float64, "rp_obs_time": np.float64,
    "rp_flux": np.float64, "rp_flux_error": np.float64,
}
_BOOL = (
    "photometry_flag_noisy_data", "photometry_flag_bp_reject", "photometry_flag_rp_reject",
    "variability_flag_g_reject", "variability_flag_bp_reject", "variability_flag_rp_reject",
)


def read_epoch_csv(source):
    """Parse a whole DR3 epoch-photometry CSV/ECSV into one light-curve table (see iter_epoch_csv)."""
    tables = list(iter_epoch_csv(source))
    return pa.concat_tables(tables) if tables else SCHEMA.empty_table()


def iter_epoch_csv(source, chunk_rows=2000):
    """Parse a DR3 epoch-photometry CSV/ECSV (bulk file, gz or not, or DataLink DATAMODEL_GAIA csv/ecsv).

    source is a path or bytes. Yields light-curve tables of up to chunk_rows sources each, streaming, so
    memory stays bounded: the largest bulk files (Galactic bulge) are 2.2 GB compressed.
    """
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        text = io.StringIO(raw.decode())
    else:
        with open(source, "rb") as f:
            gz = f.read(2) == b"\x1f\x8b"
        text = gzip.open(source, "rt") if gz else open(source, "rt")
    with text:
        reader = csv.reader(line for line in text if not line.startswith("#"))
        header = next(reader, None)
        if header is None:
            return
        idx = {name: i for i, name in enumerate(header)}
        while True:
            rows = list(itertools.islice(reader, chunk_rows))
            if not rows:
                return
            yield _rows_to_table(rows, idx)


def _rows_to_table(rows, idx):
    source_id = np.array([int(r[idx["source_id"]]) for r in rows], dtype=np.int64)
    n = np.array([int(r[idx["n_transits"]]) for r in rows], dtype=np.int64)
    cols = {name: _parse_arrays([r[idx[name]] for r in rows], dtype, n) for name, dtype in _NUMERIC.items()}
    for name in _BOOL:
        cols[name] = _parse_bools([r[idx[name]] for r in rows], n)
    return _to_long(np.repeat(source_id, n), cols)


def _parse_arrays(cells, dtype, n):
    body = ",".join(c.strip()[1:-1] if c.startswith(("[", "(")) else c for c in cells)
    body = body.replace('"', "").replace("null", "NaN").replace(",,", ",NaN,")
    if dtype is np.int64:
        out = np.array(body.replace("NaN", "-1").split(","), dtype=np.int64) if body else np.empty(0, np.int64)
    else:
        out = np.array(body.split(","), dtype=np.float64) if body else np.empty(0)
    if out.size != n.sum():
        raise ValueError(f"array length mismatch: {out.size} values for {n.sum()} transits")
    return out


def _parse_bools(cells, n):
    body = ",".join(c.strip()[1:-1] for c in cells).replace('"', "")
    # Bulk files write True/False in brackets; DataLink (DATAMODEL_GAIA csv) writes "(true, false, ...)". An exact
    # "True" match read every DataLink flag as False, so DataLink light curves carried no DPAC rejections.
    out = np.array([v.strip().lower() == "true" for v in body.split(",")], dtype=bool) if body else np.empty(0, bool)
    if out.size != n.sum():
        raise ValueError("flag array length mismatch")
    return out


def _to_long(sid, c):
    _, fov, row, _ = decode_transit_id(c["transit_id"])
    n_obs = np.nan_to_num(c["g_transit_n_obs"], nan=0).astype(np.int8)
    parts = []
    for band, tcol, fcol, ecol, vari, phot in (
        ("G", "g_transit_time", "g_transit_flux", "g_transit_flux_error", "variability_flag_g_reject", None),
        ("BP", "bp_obs_time", "bp_flux", "bp_flux_error", "variability_flag_bp_reject", "photometry_flag_bp_reject"),
        ("RP", "rp_obs_time", "rp_flux", "rp_flux_error", "variability_flag_rp_reject", "photometry_flag_rp_reject"),
    ):
        keep = np.isfinite(c[fcol]) & np.isfinite(c[tcol])
        flags = np.where(c[vari], FLAG_VARI_REJECT, 0)
        if phot:
            flags |= np.where(c[phot], FLAG_PHOT_REJECT, 0)
        else:
            flags |= np.where(c["photometry_flag_noisy_data"], FLAG_NOISY, 0)
        k = int(keep.sum())
        parts.append({
            "source_id": sid[keep], "band": np.full(k, BANDS.index(band), np.int8),
            "t": gaia_time_to_btjd(c[tcol][keep]), "flux": c[fcol][keep], "err": c[ecol][keep],
            "flags": flags[keep].astype(np.int32), "transit_id": c["transit_id"][keep],
            "fov": fov[keep], "ccd_row": row[keep], "n_obs": n_obs[keep] if band == "G" else np.zeros(k, np.int8),
        })
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.lexsort((merged["t"], merged["band"], merged["source_id"]))
    arrays = []
    for field in SCHEMA:
        v = merged[field.name][order]
        if field.name == "band":
            arrays.append(pa.DictionaryArray.from_arrays(pa.array(v, pa.int8()), pa.array(BANDS)))
        else:
            arrays.append(pa.array(v, field.type))
    return pa.Table.from_arrays(arrays, schema=SCHEMA)


# -- DataLink -----------------------------------------------------------------------------------------

DATALINK_MAX_IDS = 50  # archive limit since the 2026-10-14 release ("Maximum number of Source Ids reached (max: 50)")


def datalink_epoch_photometry(source_ids, client=None, cache=True):
    """Fetch DR3 epoch photometry for up to DATALINK_MAX_IDS sources per call, as a light-curve table."""
    client = client or get_client()
    archive.login(client)
    ids = [int(s) for s in source_ids]
    tables = []
    for i in range(0, len(ids), DATALINK_MAX_IDS):
        chunk = ids[i:i + DATALINK_MAX_IDS]
        resp = client.post(archive.DATALINK_URL, data={
            "RETRIEVAL_TYPE": "EPOCH_PHOTOMETRY", "ID": ",".join(map(str, chunk)), "RELEASE": "Gaia DR3",
            "DATA_STRUCTURE": "DATAMODEL_GAIA", "FORMAT": "csv", "VALID_DATA": "false",
        }, cache=cache)
        resp.raise_for_status()
        tables.append(read_epoch_csv(resp.content))
    return pa.concat_tables(tables) if tables else SCHEMA.empty_table()


# -- partner data centres ---------------------------------------------------------------------------

# Gaia@AIP serves DR3 epoch photometry as a TAP table with the bulk files' columns, one row per source. Measured
# 2026-10-23: 20,000 light curves by source_id range in 13 s (~1,500 per second; 4 queries in parallel ~3,000),
# against 0.3-1.5 per second from ESA's DataLink. An IN list of 500 ids took 21 s, so query by source_id range
# (a range is a patch of sky: source_id // 2**35 is the level-12 HEALPix pixel). Whether partner centres
# (CDS, ASDC, ARI, AIP, Flatiron) carry DR4's epoch photometry is not known before release.
MIRRORS = {"aip": "https://gaia.aip.de/tap"}
MIRROR_TABLE = "gaiadr3.epoch_photometry"


def mirror_adql(table=MIRROR_TABLE, source_id_range=None, source_ids=None, top=None):
    """ADQL for light curves from a partner centre's epoch-photometry table (DR3 bulk-file columns)."""
    cols = ", ".join(("source_id", "n_transits") + tuple(_NUMERIC) + _BOOL)
    where = []
    if source_id_range is not None:
        where.append(f"source_id BETWEEN {int(source_id_range[0])} AND {int(source_id_range[1])}")
    if source_ids is not None:
        where.append(f"source_id IN ({', '.join(str(int(s)) for s in source_ids)})")
    top = f"TOP {int(top)} " if top else ""
    return f"SELECT {top}{cols} FROM {table}" + (f" WHERE {' AND '.join(where)}" if where else "")


def mirror_rows_to_table(rows):
    """Light-curve table (SCHEMA) from an astropy table of mirror rows (array-valued columns)."""
    if not len(rows):
        return SCHEMA.empty_table()

    def arrays(name, dtype, fill):
        parts = [np.ma.filled(np.ma.asarray(v, dtype=dtype), fill) for v in rows[name]]
        return np.concatenate(parts) if parts else np.empty(0, dtype)

    n = np.array([len(np.atleast_1d(v)) for v in rows["transit_id"]], dtype=np.int64)
    cols = {name: arrays(name, dtype, -1 if dtype is np.int64 else np.nan) for name, dtype in _NUMERIC.items()}
    for name in _BOOL:
        cols[name] = arrays(name, bool, False)
    if any(len(v) != n.sum() for v in cols.values()):
        raise ValueError("array length mismatch in mirror rows")
    return _to_long(np.repeat(np.asarray(rows["source_id"], np.int64), n), cols)


def mirror_epoch_photometry(source_id_range=None, source_ids=None, mirror="aip", table=MIRROR_TABLE,
                            client=None, cache=False):
    """Light curves from a partner data centre's TAP service, as a light-curve table (SCHEMA)."""
    from cosmos_search.net.tap import Tap

    tap = Tap(MIRRORS.get(mirror, mirror), client=client, job_slots=4)
    adql = mirror_adql(table, source_id_range=source_id_range, source_ids=source_ids)
    return mirror_rows_to_table(tap.query(adql, sync=True, maxrec=10**7, cache=cache))
