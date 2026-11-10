"""First-hour checks of DR4's epoch photometry: is a file what the reader expects, does DR4 agree with DR3.

    python -m cosmos_search.gaia.dr4_check file PATH_OR_URL [--json F]  # one bulk file, before the whole-sky pass
    python -m cosmos_search.gaia.dr4_check compare --dr4 LC --dr3 LC    # the same stars in both releases

(Which routes answer at all is `gaia.dr4_routes`.) Each check returns a dict with `problems` (stop and fix before
the whole-sky pass) and `notes` (expected to differ from the draft data model somewhere; read them, carry on).
They assume nothing beyond what the reader needs, so they also say plainly when a file is not in the layout the
reader was written for.

`file`: the Parquet schema against the reader's columns and the draft data model; whether each star's rows are
contiguous (if not, the reader sorts the file in memory); observation times against DR4's span in the archive's
time system (days since 2010-01-01 TCB), which catches a different zero point; transits per star; missing values;
the magnitude zero point implied by flux and magnitude; and the reader run on the first rows.

`compare`: DR4 against DR3 light curves of the same stars, matched by transit_id (identical between releases for
the same field-of-view crossing) and otherwise by time. Reports the share of DR3 epochs found in DR4, time
offsets, the flux scale DR4 / DR3 by band and magnitude, and the scatter of the differences in units of the
errors. A flux scale far from 1 or a time offset says the reader's assumptions (units, zero points, time system)
do not hold for DR4.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import dr4, dr4_bulk, epoch

# DR4 covers 2014-07-25 to 2020-01-20; in days since 2010-01-01 (JD 2455197.5), with a margin for the ecliptic
# pole scanning that preceded the nominal scanning law.
T_MIN, T_MAX = 1640.0, 3700.0
MAG_COLUMNS = {"G": "g_mag", "BP": "bp_mag", "RP": "rp_mag"}


def _np(table, name, dtype=np.float64):
    return dr4._col(table, name, dtype)


def _quantiles(x, qs=(0.05, 0.5, 0.95)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.quantile(x, qs)] if len(x) else [None] * len(qs)


def file_report(path, sample_rows=3_000_000):
    """Checks of one epoch-photometry Parquet file (see module docstring). Content checks use the first row groups
    up to `sample_rows` rows; contiguity is checked over the whole file."""
    from cosmos_search.targets import releases

    path = Path(path)
    f = pq.ParquetFile(path)
    md, schema = f.metadata, f.schema_arrow
    names = schema.names
    rep = {"file": path.name, "bytes": path.stat().st_size, "rows": md.num_rows, "row_groups": md.num_row_groups,
           "columns": {n: str(schema.field(n).type) for n in names}, "problems": [], "notes": []}
    problems, notes = rep["problems"], rep["notes"]

    missing = dr4.missing_required(names)
    if missing:
        problems.append(f"required column(s) absent: {missing}; add their names to dr4.ALIASES or dr4.COLUMNS")
        return rep
    have = dr4.resolve(names)
    absent = [c for c in dr4.READ_COLUMNS if c not in have]
    if absent:
        notes.append(f"columns the reader uses if present are absent: {absent}")
    renamed = {c: have[c] for c in have if have[c] != c}
    if renamed:
        notes.append(f"columns found under another spelling: {renamed}")
    draft = releases.draft_columns().get("epoch_photometry", [])
    lower = {n.lower() for n in names}
    rep["draft_columns_absent"] = [c for c in draft if c not in lower and c not in have]
    rep["columns_not_in_draft"] = sorted(lower - set(draft))
    if rep["draft_columns_absent"] or rep["columns_not_in_draft"]:
        notes.append(f"differs from the draft data model: absent {rep['draft_columns_absent']}, "
                     f"new {rep['columns_not_in_draft']}")

    tid_type = schema.field(have["transit_id"]).type
    nested = pa.types.is_list(tid_type) or pa.types.is_large_list(tid_type) or pa.types.is_fixed_size_list(tid_type)
    rep["one_row_per"] = "source (arrays)" if nested else "transit"
    if nested:
        notes.append("one row per source holding arrays (DR3's layout); the reader unpacks it")
    rep["stars_contiguous"] = True if nested else bool(dr4_bulk.stars_contiguous(path, have["source_id"]))
    if not rep["stars_contiguous"]:
        notes.append("a star's rows are not contiguous: the reader sorts each file in memory, so allow memory of "
                     "several times the file size per worker and lower --workers to fit")

    groups, n = [], 0
    for i in range(md.num_row_groups):
        groups.append(i)
        n += md.row_group(i).num_rows
        if n >= sample_rows:
            break
    t = f.read_row_groups(groups, columns=[c for c in names if c in set(have.values())
                                           or c.lower() in MAG_COLUMNS.values()])
    if nested:
        t = dr4.explode(t)
    rep["rows_checked"] = t.num_rows
    if not t.num_rows:
        problems.append("the file has no rows")
        return rep

    sid = _np(t, have["source_id"], np.int64)
    tid = _np(t, have["transit_id"], np.int64)
    new_star = np.r_[True, sid[1:] != sid[:-1]]
    counts = np.diff(np.r_[np.flatnonzero(new_star), len(sid)])
    if rep["stars_contiguous"] and len(counts) > 2:
        counts = counts[:-1]  # the last star of the sample may be cut by the row-group boundary
    rep["stars_checked"] = int(new_star.sum())
    rep["stars"] = int(rep["stars_checked"] * md.num_rows / max(t.num_rows, 1)) if t.num_rows < md.num_rows \
        else rep["stars_checked"]
    rep["transits_per_star"] = dict(zip(("p05", "median", "p95"), _quantiles(counts)))
    rep["source_id_increasing"] = bool(np.all(np.diff(sid[new_star]) > 0))
    if not rep["source_id_increasing"]:
        notes.append("stars are not in source_id order within the file")
    order = np.lexsort((tid, sid))
    dup = (np.diff(sid[order]) == 0) & (np.diff(tid[order]) == 0)
    rep["duplicate_transits"] = int(dup.sum())
    if rep["duplicate_transits"]:
        problems.append(f"{rep['duplicate_transits']} repeated (source_id, transit_id) pairs in the rows checked")

    rep["bands"] = {}
    for band, (tc, fc, ec) in dr4.COLUMNS.items():
        if tc not in have or fc not in have:
            continue
        time, flux = _np(t, have[tc]), _np(t, have[fc])
        b = {"missing_time_share": float(np.mean(~np.isfinite(time))),
             "missing_flux_share": float(np.mean(~np.isfinite(flux))),
             "negative_flux_share": float(np.mean(flux < 0)),
             "time_range": [float(np.nanmin(time)), float(np.nanmax(time))] if np.isfinite(time).any() else None}
        if ec in have:
            err = _np(t, have[ec])
            b["missing_error_share"] = float(np.mean(~np.isfinite(err)))
            b["flux_over_error"] = dict(zip(("p05", "median", "p95"), _quantiles(flux / err)))
        mag_col = next((n for n in names if n.lower() == MAG_COLUMNS[band]), None)
        if mag_col:
            mag = _np(t, mag_col)
            ok = np.isfinite(mag) & (flux > 0)
            if ok.any():
                zp = mag[ok] + 2.5 * np.log10(flux[ok])
                b["zero_point"] = {"median": float(np.median(zp)), "spread": float(np.ptp(np.quantile(zp, (.01, .99))))}
                if b["zero_point"]["spread"] > 0.01:
                    notes.append(f"{band}: magnitude is not a fixed zero point minus 2.5 log10(flux) "
                                 f"(spread {b['zero_point']['spread']:.3f} mag)")
        rep["bands"][band] = b
        if b["time_range"] and not (T_MIN <= b["time_range"][0] and b["time_range"][1] <= T_MAX):
            problems.append(f"{band} times span {b['time_range']}, outside DR4's {T_MIN}-{T_MAX} days since "
                            "2010-01-01 TCB: another time zero point or unit; fix epoch.gaia_time_to_btjd's input")
    g = rep["bands"]["G"]
    if g["missing_flux_share"] > 0.5:
        problems.append(f"{g['missing_flux_share']:.0%} of G fluxes are missing")
    gt = _np(t, have[dr4.COLUMNS["G"][0]])
    same = sid[1:] == sid[:-1]
    both = same & np.isfinite(gt[1:]) & np.isfinite(gt[:-1])
    rep["time_ordered_within_star"] = float(np.mean(gt[1:][both] >= gt[:-1][both])) if both.any() else None
    span = np.maximum.reduceat(np.where(np.isfinite(gt), gt, -np.inf), np.flatnonzero(new_star)) - \
        np.minimum.reduceat(np.where(np.isfinite(gt), gt, np.inf), np.flatnonzero(new_star))
    rep["time_span_per_star_days"] = dict(zip(("p05", "median", "p95"), _quantiles(span)))

    try:
        lc = dr4.epoch_photometry_to_lc(t)
    except Exception as exc:  # noqa: BLE001 (reported as the check's result)
        problems.append(f"the reader failed on the rows checked: {exc!r}")
        return rep
    flags = np.asarray(lc["flags"])
    band = np.asarray(pc.cast(lc["band"], pa.string()))
    rep["reader"] = {
        "lightcurve_rows": {b: int((band == b).sum()) for b in epoch.BANDS},
        "t_btjd_range": [float(pc.min(lc["t"]).as_py()), float(pc.max(lc["t"]).as_py())],
        "flag_shares": {name: float(np.mean((flags & bit) != 0)) for name, bit in (
            ("variability_rejected", epoch.FLAG_VARI_REJECT), ("photometry_rejected", epoch.FLAG_PHOT_REJECT),
            ("saturation", dr4.FLAG_SATURATION), ("satellite_event", dr4.FLAG_SATELLITE_EVENT),
            ("cosmetic", dr4.FLAG_COSMETIC), ("multi_source", dr4.FLAG_MULTI_SOURCE),
            ("attitude", dr4.FLAG_ATTITUDE))},
    }
    if rep["reader"]["flag_shares"]["variability_rejected"] == 0:
        notes.append("no epoch is marked as rejected by variability processing (the draft table has no such "
                     "column); DR3 marked 2-5% per band")
    if not all(rep["reader"]["lightcurve_rows"].values()):
        notes.append(f"a band has no epochs: {rep['reader']['lightcurve_rows']}")
    return rep


def compare_releases(dr4_lc, dr3_lc, id_map=None, time_tolerance=0.02):
    """DR4 against DR3 light curves (both epoch.SCHEMA, pyarrow or pandas) of the same stars.

    `id_map`: {DR4 source_id: DR3 source_id} from dr3_neighbourhood where ids changed; ids are otherwise taken as
    equal. Epochs are matched on (source, band, transit_id), and DR3 epochs left over are matched to the nearest
    DR4 epoch of the same star and band within `time_tolerance` days."""
    import pandas as pd

    a = dr3_lc.to_pandas() if isinstance(dr3_lc, pa.Table) else dr3_lc.copy()
    b = dr4_lc.to_pandas() if isinstance(dr4_lc, pa.Table) else dr4_lc.copy()
    for d in (a, b):
        d["band"] = d["band"].astype(str)
    if id_map:
        b["source_id"] = b["source_id"].map(lambda s: id_map.get(s, s))
    stars = sorted(set(a["source_id"]) & set(b["source_id"]))
    out = {"stars_dr3": int(a["source_id"].nunique()), "stars_dr4": int(b["source_id"].nunique()),
           "stars_in_both": len(stars), "problems": [], "notes": [], "bands": {}}
    if not stars:
        out["problems"].append("no star is in both sets; map DR4 ids to DR3's through dr3_neighbourhood")
        return out
    a, b = a[a["source_id"].isin(stars)], b[b["source_id"].isin(stars)]
    out["dr3_t_range"] = [float(a["t"].min()), float(a["t"].max())]
    out["dr4_t_range"] = [float(b["t"].min()), float(b["t"].max())]
    key = ["source_id", "band", "transit_id"]
    m = a.merge(b, on=key, how="left", suffixes=("_3", "_4"), indicator=True)
    by_id = m["_merge"] == "both"
    out["dr3_epochs"] = len(a)
    out["matched_by_transit_id"] = float(by_id.mean())
    pairs = m.loc[by_id, ["source_id", "band", "t_3", "t_4", "flux_3", "flux_4", "err_3", "err_4"]]
    rest = m.loc[~by_id, ["source_id", "band", "t_3", "flux_3", "err_3"]]
    if len(rest):
        left = rest.rename(columns={"t_3": "t"}).sort_values("t")
        right = b[["source_id", "band", "t", "flux", "err"]].rename(
            columns={"flux": "flux_4", "err": "err_4"}).assign(t_4=lambda d: d["t"]).sort_values("t")
        near = pd.merge_asof(left, right, on="t", by=["source_id", "band"], direction="nearest",
                             tolerance=time_tolerance).dropna(subset=["t_4"]).rename(columns={"t": "t_3"})
        out["matched_by_time"] = float(len(near) / len(a))
        pairs = pd.concat([pairs, near[pairs.columns]], ignore_index=True)
    else:
        out["matched_by_time"] = 0.0
    out["dr3_epochs_found_in_dr4"] = float(len(pairs) / len(a))
    if out["dr3_epochs_found_in_dr4"] < 0.8:
        out["problems"].append(f"only {out['dr3_epochs_found_in_dr4']:.0%} of DR3 epochs have a DR4 epoch of the "
                               "same transit or time: ids, times or bands do not line up")
    elif out["matched_by_transit_id"] < 0.5:
        out["notes"].append("transit ids differ between the releases; epochs were matched by time")
    if not len(pairs):
        return out
    dt = (pairs["t_4"] - pairs["t_3"]).to_numpy()
    out["time_offset_days"] = dict(zip(("p05", "median", "p95"), _quantiles(dt)))
    if abs(out["time_offset_days"]["median"]) > 1e-3:
        out["problems"].append(f"DR4 times are offset from DR3's by {out['time_offset_days']['median']:.5f} days")
    star_flux = a[a["band"] == "G"].groupby("source_id")["flux"].median()
    pairs = pairs.assign(g=epoch_mag(pairs["source_id"].map(star_flux).to_numpy()))
    for band, p in pairs.groupby("band"):
        ok = (p["flux_3"] > 0) & (p["flux_4"] > 0) & np.isfinite(p["err_3"]) & np.isfinite(p["err_4"])
        p = p[ok]
        if not len(p):
            continue
        ratio = (p["flux_4"] / p["flux_3"]).to_numpy()
        z = ((p["flux_4"] - p["flux_3"]) / np.hypot(p["err_3"], p["err_4"])).to_numpy()
        d = {"epochs": len(p), "flux_scale": dict(zip(("p05", "median", "p95"), _quantiles(ratio))),
             "difference_over_error": dict(zip(("p05", "median", "p95"), _quantiles(z))),
             "error_ratio_median": float(np.median(p["err_4"] / p["err_3"])), "by_g": {}}
        for lo, hi in ((0, 13), (13, 16), (16, 18), (18, 25)):
            sel = (p["g"] >= lo) & (p["g"] < hi)
            if sel.sum() >= 20:
                d["by_g"][f"{lo}-{hi}"] = {"epochs": int(sel.sum()), "flux_scale": float(np.median(ratio[sel])),
                                           "error_ratio": float(np.median((p["err_4"] / p["err_3"])[sel]))}
        out["bands"][band] = d
        if not 0.9 < d["flux_scale"]["median"] < 1.1:
            out["problems"].append(f"{band}: DR4 fluxes are {d['flux_scale']['median']:.3f} times DR3's for the "
                                   "same epochs: another unit or zero point")
        elif abs(d["flux_scale"]["median"] - 1) > 0.01:
            out["notes"].append(f"{band}: DR4 flux scale is {d['flux_scale']['median']:.4f} of DR3's")
    return out


def epoch_mag(flux, zero_point=25.6874):
    """G magnitude from a G flux in e-/s (DR3's Vega zero point; within a few mmag of any later one)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return zero_point - 2.5 * np.log10(np.asarray(flux, dtype=float))


def _read_lightcurves(path):
    p = Path(path)
    files = sorted(p.glob("*.parquet")) if p.is_dir() else [p]
    return pa.concat_tables([pq.read_table(f) for f in files])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.gaia.dr4_check", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("file", help="check one bulk file (a local path, or a key or URL on the CDN)")
    f.add_argument("path")
    f.add_argument("--rows", type=int, default=3_000_000)
    c = sub.add_parser("compare", help="DR4 against DR3 light curves of the same stars")
    c.add_argument("--dr4", required=True, help="light-curve Parquet file or directory (epoch.SCHEMA)")
    c.add_argument("--dr3", required=True)
    c.add_argument("--id-map", help="Parquet with source_id (DR4) and dr3_source_id, from dr3_neighbourhood")
    for p in (f, c):
        p.add_argument("--json", help="also write the result to this file")
    a = ap.parse_args(argv)

    if a.cmd == "file":
        if Path(a.path).exists():
            rep = file_report(a.path, a.rows)
        else:
            from . import archive

            url = a.path if "://" in a.path else f"{archive.CDN_URL}/{a.path.lstrip('/')}"
            tmp = Path(a.json).with_suffix(".tmp.parquet") if a.json else Path("dr4-check.tmp.parquet")
            with dr4_bulk.fetched(url, tmp) as path:
                rep = file_report(path, a.rows)
        code = 1 if rep["problems"] else 0
    else:
        id_map = None
        if a.id_map:
            m = pq.read_table(a.id_map, columns=["source_id", "dr3_source_id"]).to_pandas()
            id_map = dict(zip(m["source_id"], m["dr3_source_id"]))
        rep = compare_releases(_read_lightcurves(a.dr4), _read_lightcurves(a.dr3), id_map)
        code = 1 if rep["problems"] else 0
    text = json.dumps(rep, indent=1, default=str)
    print(text)
    if a.json:
        Path(a.json).write_text(text + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
