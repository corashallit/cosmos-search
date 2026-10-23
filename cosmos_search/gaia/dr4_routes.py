"""Release-day check of the routes to DR4 light curves, fastest first, with a recommendation.

    python -m cosmos_search.gaia.dr4_routes [--out routes.json]
    python -m cosmos_search.gaia.dr4_routes --release "Gaia DR3" --schema gaiadr3 --cdn-prefix Gaia/gdr3/   # dry run

Three routes, in the order to prefer them (measured on DR3, 2026-10-23):

| Route | DR3 rate | What it suits |
|---|---|---|
| ESA bulk files on the CDN (`gaia.dr4_bulk`) | ~120 MB/s per container, so all 7.2 TB pass in ~17 h; scales with containers | any selection size, read as the files stream past |
| A mirror's database table (AIP: `epoch.mirror_epoch_photometry`) | ~1,500 light curves/s, ~3,000/s with 4 queries | a selection of millions |
| ESA DataLink | 0.2-1.5 light curves/s, ≤ 50 ids per call | ≤ ~10^4 stars, and per-CCD G (DataLink only) |

ESA's approved bulk-download plan (2024) puts epoch photometry on the CDN; AIP served DR2's and DR3's light curves
as a table. Neither is announced for DR4, so this checks each one when the archive reopens.
"""

import argparse
import json
import sys
import time

from cosmos_search.net import get_client

from . import archive, dr4_bulk

MIRRORS = {
    "esa": archive.TAP_URL,
    "aip": "https://gaia.aip.de/tap",
    "ari": "https://gaia.ari.uni-heidelberg.de/tap",
}


def _sync_first_column(url, adql, client, timeout=60):
    """First column of a small sync TAP query, as strings (VOTable, which every service returns)."""
    import io

    from astropy.io.votable import parse_single_table

    resp = client.post(f"{url}/sync", data={"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": adql}, timeout=timeout)
    resp.raise_for_status()
    table = parse_single_table(io.BytesIO(resp.content)).to_table()
    return [str(v) for v in table.columns[0]] if len(table.columns) else []


def check_tables(schema="gaiadr4", client=None):
    """Per service: its tables in `schema` whose name mentions epoch_photometry (TAP_SCHEMA)."""
    client = client or get_client()
    out = {}
    for name, url in MIRRORS.items():
        adql = (f"SELECT table_name FROM tap_schema.tables WHERE table_name LIKE '{schema}.%epoch_photometry%'"
                f" OR table_name LIKE '{schema}.all_source_photometry'")
        try:
            out[name] = {"tables": sorted(_sync_first_column(url, adql, client))}
        except Exception as exc:  # report and go on; one service down must not hide the others
            out[name] = {"error": repr(exc)[:300]}
    return out


def check_cdn(prefix=dr4_bulk.PREFIX, client=None):
    try:
        return dr4_bulk.summary(prefix, client)
    except Exception as exc:
        return {"prefix": prefix, "error": repr(exc)[:300]}


def check_datalink(release="Gaia DR4", schema="gaiadr4", n=5, client=None):
    """Time one DataLink EPOCH_PHOTOMETRY call for n sources taken from the release's gaia_source (those flagged
    has_epoch_photometry where the column exists, as in DR3; DR4 has light curves for nearly every source)."""
    client = client or get_client()
    base = f"SELECT TOP {int(n)} source_id FROM {schema}.gaia_source WHERE phot_g_mean_mag BETWEEN 15 AND 17"
    try:
        archive.login(client)
        try:
            ids = _sync_first_column(archive.TAP_URL, base + " AND has_epoch_photometry = 'True'", client)
        except Exception:
            ids = _sync_first_column(archive.TAP_URL, base, client)
        t0 = time.time()
        resp = client.post(archive.DATALINK_URL, data={
            "RETRIEVAL_TYPE": "EPOCH_PHOTOMETRY", "ID": ",".join(ids), "RELEASE": release,
            "DATA_STRUCTURE": "DATAMODEL_GAIA", "FORMAT": "csv", "VALID_DATA": "false"}, timeout=600)
        dt = time.time() - t0
        return {"release": release, "n": len(ids), "status": resp.status_code, "bytes": len(resp.content),
                "content_type": resp.headers.get("content-type"), "seconds": round(dt, 1),
                "per_second": round(len(ids) / dt, 2) if dt > 0 and resp.content else None}
    except Exception as exc:
        return {"release": release, "error": repr(exc)[:300]}


def recommend(r):
    cdn, tables, dl = r["cdn"], r["tables"], r["datalink"]
    if cdn.get("epoch_photometry_files"):
        return (f"Bulk files are up ({cdn['epoch_photometry_files']} files, {cdn['epoch_photometry_bytes'] / 1e12:.2f} TB): "
                "check one with `dr4_bulk list`, then `dr4_bulk pull --select ... --random-share ...`, split by --shard "
                "across kore and cloud containers. Each raw file is deleted once read.")
    mirrors = [m for m, v in tables.items() if m != "esa" and any("epoch_photometry" in t for t in v.get("tables", []))]
    if mirrors:
        return (f"No bulk files yet; {', '.join(mirrors)} has an epoch_photometry table: run the whole-sky selection "
                "on ESA TAP and pull its light curves from the mirror in source_id ranges. Re-check the CDN daily.")
    rate = dl.get("per_second")
    return ("No bulk files and no mirror table yet: pull only the top of the selection through DataLink"
            + (f" (~{rate * 86400:,.0f} light curves a day at the measured {rate}/s)" if rate else "")
            + ", and re-check the CDN and mirrors daily.")


def check(release="Gaia DR4", schema="gaiadr4", cdn_prefix=dr4_bulk.PREFIX, datalink_n=5):
    client = get_client()
    r = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "cdn": check_cdn(cdn_prefix, client), "tables": check_tables(schema, client),
         "datalink": check_datalink(release, schema, datalink_n, client) if datalink_n else {}}
    r["recommendation"] = recommend(r)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.gaia.dr4_routes", description=__doc__.splitlines()[0])
    ap.add_argument("--release", default="Gaia DR4")
    ap.add_argument("--schema", default="gaiadr4")
    ap.add_argument("--cdn-prefix", default=dr4_bulk.PREFIX)
    ap.add_argument("--datalink-n", type=int, default=5, help="sources in the timed DataLink call (0 skips it)")
    ap.add_argument("--out", help="also write the result as JSON here")
    a = ap.parse_args(argv)
    r = check(a.release, a.schema, a.cdn_prefix, a.datalink_n)
    text = json.dumps(r, indent=1)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
