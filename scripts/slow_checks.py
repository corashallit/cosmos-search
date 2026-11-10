"""Run the per-candidate checks (cosmos_search.slow.checks) on the strongest stars of a search's fading list.

    python scripts/slow_checks.py --fading R/fading.parquet --out R/checks [--n 100] [--schema gaiadr4] [--max-g 17]
        [--skip-class ECL] [--skip-still] [--skip-band] [--workers 6] [--lightcurves R/lightcurves]
        [--surveys ztf,atlas,skymapper,noirlab[,asassn] | none] [--no-tess]

--fading is the list `scripts/slow_search.py flag` writes (ordered by strength). For each of the first --n stars
(optionally only those brighter than --max-g): gaia_source astrometry and quality columns and DPAC's variability
class, the colour of the change, NEOWISE W1/W2 against what each explanation predicts (inside the faint window
against outside for an episode, the slope over the light curve's years for a trend; a repeating dimming has no
single window and gets neither), and ZTF levels before and after 2020-01-20. Writes checks.csv (one row per star)
and the raw NEOWISE and ZTF epochs. With --lightcurves, also whether a strict-period eclipsing binary explains the
dips on G and ZTF together (`checks.eclipse`; `eb_explained` stars leave the review). A failed fetch is recorded and
the next star goes on. Stars are checked
--workers at a time: one star's NEOWISE and ZTF fetches take one to two minutes, so 10^3 candidates one after
another would take over a day.

Other surveys (--surveys, on by default; docs/arms/slow-changes.md "Other surveys"): each star is read from every
listed survey that this machine can reach (`cosmos_search.surveys`; ATLAS needs a token and ASAS-SN, which is read
only when named, port 9006; a survey that cannot be read is named once at the start and left out), and compared
with Gaia's change (`slow.independent`): `same_k`, `same_k_err` and `same_years` say whether the surveys saw
Gaia's change over Gaia's own dates (1 the same change, 0 none; seen, not_seen, opposite, undecided, no_data),
`after` and its columns what they saw since Gaia's last visit, each also per survey (`<survey>_k`, ...), with
`<survey>_dilution` the share of neighbours' light in that survey's beam. For stars brighter than TESS_MAX_G,
`tess_amp` is the variability within a month from TESS (`surveys.tess_activity`), `tess_quiet` whether that is no
more than constant stars show, and `activity_ratio` Gaia's change as a multiple of it (a lower limit for a quiet
star). Writes surveys.parquet (all epochs read) and tess.parquet (one row per star and sector).
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search import surveys
from cosmos_search.dayone import external
from cosmos_search.energy import budget
from cosmos_search.gaia.archive import tap
from cosmos_search.net.tap import Tap
from cosmos_search.slow import checks
from cosmos_search.slow import independent as ind
from cosmos_search.surveys import tess_activity

TESS_MAX_G = 16.0      # TGLC has no light curve for stars fainter than about Tmag 16
MIRROR_TAP = "https://gaia.aip.de/tap"
_esa_failed = False
_mirror = None

GAIA_COLS = ("source_id, ra, dec, pmra, pmdec, parallax, parallax_error, ref_epoch, phot_g_mean_mag, bp_rp, ruwe, "
             "ipd_frac_multi_peak, phot_bp_rp_excess_factor")


def candidates(fading, n, max_g=None):
    f = fading if max_g is None else fading[fading["phot_g_mean_mag"] < max_g]
    return f.sort_values("strength", ascending=False).head(n).rename(columns={"id": "source_id"})


def gaia_query(adql):
    """A synchronous Gaia archive query as a DataFrame with lower-case columns: ESA's archive, logged in, and the
    AIP mirror when ESA fails. Under load ESA rejects synchronous queries with 503 after a minute of retries (seen
    2026-11-10, logged in), so once it has failed the mirror is asked first for the rest of the run."""
    global _esa_failed, _mirror
    order = ("mirror", "esa") if _esa_failed else ("esa", "mirror")
    last = None
    for which in order:
        try:
            if which == "esa":
                d = tap().query(adql, sync=True).to_pandas()
            else:
                if _mirror is None:
                    _mirror = Tap(MIRROR_TAP, job_slots=2)
                d = _mirror.query(adql, sync=True).to_pandas()
            d.columns = [x.lower() for x in d.columns]
            return d
        except Exception as e:  # noqa: BLE001 (either service may be down or lack the table; try the other)
            last = e
            if which == "esa" and not _esa_failed:
                _esa_failed = True
                print(f"ESA Gaia archive failed ({e!r:.100}); asking {MIRROR_TAP} first from here on", flush=True)
    raise last


def gaia_rows(ids, schema):
    """gaia_source columns and DPAC's best variability class for the given source ids."""
    g, c = [], []
    for i in range(0, len(ids), 200):
        chunk = ",".join(str(int(x)) for x in ids[i:i + 200])
        g.append(gaia_query(f"SELECT {GAIA_COLS} FROM {schema}.gaia_source WHERE source_id IN ({chunk})"))
        c.append(gaia_query(f"SELECT source_id, best_class_name, best_class_score FROM "
                            f"{schema}.vari_classifier_result WHERE source_id IN ({chunk})"))
    return pd.concat(g, ignore_index=True).merge(pd.concat(c, ignore_index=True), on="source_id", how="left")


def at_2019(r):
    """Position propagated from ref_epoch to 2019.0 (NEOWISE-R's middle), small-angle."""
    dt = 2019.0 - r.ref_epoch
    pmra, pmdec = (0.0 if pd.isna(r.pmra) else r.pmra), (0.0 if pd.isna(r.pmdec) else r.pmdec)
    return (r.ra + pmra * dt / 3.6e6 / np.cos(np.radians(r.dec)), r.dec + pmdec * dt / 3.6e6)


def g_epochs(lightcurves, ids):
    """Good G epochs (source_id, t, flux relative to the median) of the given stars from a directory of light-curve
    files (epoch.SCHEMA, e.g. the kept light curves of `slow_search.py score`), found by each file's source_id
    range."""
    import pyarrow.parquet as pq

    from cosmos_search.slow import drift

    files = sorted(Path(lightcurves).glob("*.parquet"))
    ids = np.sort(np.asarray(ids, np.int64))
    out = []
    for f in files:
        md = pq.read_metadata(f)
        col = md.schema.to_arrow_schema().get_field_index("source_id")
        st = [md.row_group(i).column(col).statistics for i in range(md.num_row_groups)]
        mine = ids[(ids >= min(x.min for x in st)) & (ids <= max(x.max for x in st))]
        if len(mine):
            ep = drift.normalise(pq.read_table(f, filters=[("source_id", "in", [int(x) for x in mine])]))
            ep = ep[ep["band"] == 0]
            out.append(pd.DataFrame({"source_id": ep["source_id"], "t": ep["t"], "flux": 1 + ep["frac"]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["source_id", "t", "flux"])


def other_surveys(r, z, names, schema, with_tess):
    """The comparison with other surveys for one candidate (module notes): (columns, epochs read, TESS rows per
    sector or None, error text). z: the star's ZTF epochs when already read (with their `group` column)."""
    row, err = {}, ""
    lc, failed = surveys.fetch_all(r, [s for s in names if s != "ztf"])
    for s, e in failed.items():
        err += f"{s}: {e:.120}; "
    if "ztf" in names and z is not None and len(z) and surveys.covers("ztf", r.dec, r.phot_g_mean_mag):
        lc = pd.concat([lc, z.assign(survey="ztf", mag=np.nan)[surveys.COLUMNS]], ignore_index=True)
    change = ind.gaia_change(r)
    c = {}
    if len(lc):
        try:
            nb = gaia_query(ind.neighbours_adql(r.ra, r.dec, schema))
            c = ind.dilutions(nb, r.source_id, r.ra, r.dec, r.phot_g_mean_mag, pd.unique(lc["survey"]))
        except Exception as e:  # noqa: BLE001 (without neighbours the comparison runs undiluted, and says so)
            err += f"neighbours: {e!r:.120}; "
    row.update({f"{s}_dilution": v for s, v in c.items()})
    row.update(ind.columns(ind.same_years(lc, change, c), ind.after(lc, change, c)))
    sectors = None
    if with_tess and r.phot_g_mean_mag < TESS_MAX_G:
        try:
            summary, sectors = tess_activity.activity(r.source_id, r.ra, r.dec)
            row.update(summary)
            if summary.get("tess_sectors"):
                floor = tess_activity.quiet_floor(r.phot_g_mean_mag)
                row["tess_quiet"] = bool(summary["tess_amp"] <= floor)
                row["activity_ratio"] = ind.activity(change, summary["tess_amp"], floor)
        except Exception as e:  # noqa: BLE001
            err += f"TESS: {e!r:.120}; "
    return row, lc, sectors, err


def check_one(r, irsa, g=None, names=(), schema="gaiadr4", with_tess=False):
    """The checks for one candidate: (row, NEOWISE epochs, ZTF epochs or None, other surveys' epochs or None, TESS
    rows per sector or None) with fetch times. g: its G epochs, for the eclipsing-binary check (`checks.eclipse`,
    on G and ZTF together). names: the other surveys to compare with (`other_surveys`)."""
    episode = "episode" in r.by
    row = {"source_id": r.source_id, "by": r.by, "strength": r.strength, "g": r.phot_g_mean_mag,
           "bp_rp": r.bp_rp, "class": r.best_class_name, "ruwe": r.ruwe, "ipd_multi": r.ipd_frac_multi_peak,
           "s_fade": r.s_fade, "fade_kind": r.fade_kind, "slope_pct_yr": 100 * r.slope, "slope_z": r.slope_z,
           "s_rep": r.s_rep, "rep_period": r.rep_period, "abbe": r.abbe, "dim_depth": r.dim_depth,
           "dim_t0_mjd": r.dim_t0 + checks.BTJD_T0_MJD, "dim_t1_mjd": r.dim_t1 + checks.BTJD_T0_MJD,
           "errors": ""}
    row.update({f"colour_{k}": v for k, v in
                checks.colour(r.depth_bp, r.depth_bp_err, r.depth_rp, r.depth_rp_err).items()})
    teff = float(budget.teff_from_bp_rp(r.bp_rp if pd.notna(r.bp_rp) else 0.82))
    ra, dec = at_2019(r)
    t0 = time.time()
    try:
        w = checks.neowise_clean(irsa.query(checks.neowise_adql(ra, dec), sync=True).to_pandas())
    except Exception as e:  # noqa: BLE001 (one failed fetch should not stop the rest)
        row["errors"] += f"NEOWISE: {e!r:.120}; "
        w = checks.neowise_clean([])
    row["neowise_s"] = time.time() - t0
    if episode:
        wc = checks.wise_change(w, row["dim_t0_mjd"], row["dim_t1_mjd"])
        for b in ("W1", "W2"):
            row[f"{b}_change"], row[f"{b}_err"], row[f"{b}_visits_in"] = wc[b]
            for k, v in checks.wise_predictions(max(r.dim_depth, 0.0) if pd.notna(r.dim_depth) else 0.0,
                                                teff)[b].items():
                row[f"{b}_pred_{k}"] = v
    elif "trend" in r.by:
        # A steady fade: W1/W2 slope over the light curve's years against Gaia's (fraction per year).
        ws = checks.wise_slope(w, r.t_first + checks.BTJD_T0_MJD, r.t_last + checks.BTJD_T0_MJD)
        for b in ("W1", "W2"):
            row[f"{b}_slope"], row[f"{b}_slope_err"], row[f"{b}_visits_in"] = ws[b]
            row[f"{b}_slope_pred_dust_in_front"] = r.slope * checks.WISE_OVER_G_DUST[b]
    z = None
    if r.dec > -30:
        t0 = time.time()
        try:
            z = external.ztf(r.source_id, r.ra, r.dec, groups=True)
        except Exception as e:  # noqa: BLE001
            row["errors"] += f"ZTF: {e!r:.120}; "
            z = external._empty()
        row["ztf_s"] = time.time() - t0
        for b, v in checks.after(z).items():
            row[f"{b}_change_after_2020"] = v["change"]
            row[f"{b}_slope_after_2020"] = v["slope_post_per_yr"]
    if g is not None and len(g):
        src = {k: getattr(r, k) for k in ("phot_g_mean_mag", "bp_rp", "parallax", "parallax_error")}
        try:
            row.update(checks.eclipse(g, z if z is not None else external._empty(), src))
        except Exception as e:  # noqa: BLE001
            row["errors"] += f"eclipse: {e!r:.120}; "
    lc = sectors = None
    if names or with_tess:
        t0 = time.time()
        cols, lc, sectors, err = other_surveys(r, z, names, schema, with_tess)
        row.update(cols)
        row["errors"] += err
        row["surveys_s"] = time.time() - t0
    return row, w.assign(source_id=r.source_id), z, lc, sectors


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fading", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-g", type=float)
    ap.add_argument("--schema", default="gaiadr4")
    ap.add_argument("--skip-class", action="append", default=[],
                    help="leave out stars DPAC gives this best class (repeatable), before taking the first --n")
    ap.add_argument("--skip-still", action="store_true",
                    help="leave out stars `slow_search.py cuts` marked still (no measurable parallax or motion)")
    ap.add_argument("--skip-band", action="store_true",
                    help="leave out fades whose G change disagrees with BP and RP (`slow_search.py cuts`)")
    ap.add_argument("--workers", type=int, default=6, help="stars checked at once (IRSA and ZTF fetches)")
    ap.add_argument("--lightcurves", help="the run's kept light curves (`slow_search.py score` R/lightcurves): adds "
                                          "the eclipsing-binary check on G and ZTF")
    ap.add_argument("--surveys", default=",".join(surveys.DEFAULT),
                    help="other surveys to compare Gaia's change with, comma-separated, or none; asassn is read "
                         "only when named here")
    ap.add_argument("--no-tess", action="store_true", help="leave out the TESS variability of bright stars")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names, why = surveys.usable([] if a.surveys == "none" else a.surveys.split(","))
    for s, reason in why.items():
        print(f"{s} left out: {reason}", flush=True)
    fading = pd.read_parquet(a.fading)
    if a.skip_still:
        fading = fading[~fading["still"]]
    if a.skip_band:
        fading = fading[~fading["band_verdict"].isin(["g_only", "bp_rp_only"])]
    pool = a.n * (5 if a.skip_class else 1)
    while True:
        cand = candidates(fading, pool, a.max_g)
        cand = cand.drop(columns=["phot_g_mean_mag", "bp_rp"], errors="ignore").merge(
            gaia_rows(cand["source_id"].to_numpy(), a.schema), on="source_id", how="left")
        keep = cand[~cand["best_class_name"].isin(a.skip_class)]
        if len(keep) >= a.n or len(cand) < pool:
            cand = keep.sort_values("strength", ascending=False).head(a.n)
            break
        pool *= 4
    irsa = Tap(checks.IRSA_TAP, job_slots=a.workers)
    gep = g_epochs(a.lightcurves, cand["source_id"]) if a.lightcurves else None
    g_of = dict(tuple(gep.groupby("source_id"))) if gep is not None else {}

    rows, wise_all, ztf_all, lc_all, tess_all = [], [], [], [], []
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(check_one, r, irsa, g_of.get(r.source_id), names, a.schema, not a.no_tess): r
                for r in cand.itertuples(index=False)}
        for fut in as_completed(futs):
            r = futs[fut]
            row, w, z, lc, sectors = fut.result()
            rows.append(row)
            wise_all.append(w)
            if z is not None:
                ztf_all.append(z)
            if lc is not None and len(lc):
                lc_all.append(lc)
            if sectors is not None and len(sectors):
                tess_all.append(sectors)
            print(r.source_id, r.by, f"G {r.phot_g_mean_mag:.1f}", r.best_class_name, row["errors"] or "", flush=True)
    res = pd.DataFrame(rows).sort_values("strength", ascending=False)
    res.to_csv(out / "checks.csv", index=False)
    pd.concat(wise_all, ignore_index=True).to_parquet(out / "neowise.parquet", index=False)
    if ztf_all:
        pd.concat(ztf_all, ignore_index=True).to_parquet(out / "ztf.parquet", index=False)
    if lc_all:
        pd.concat(lc_all, ignore_index=True).to_parquet(out / "surveys.parquet", index=False)
    if tess_all:
        pd.concat(tess_all, ignore_index=True).to_parquet(out / "tess.parquet", index=False)
    el = time.time() - t0
    print(f"{len(res)} stars in {el:.0f} s ({el / max(len(res), 1):.1f} s per star with {a.workers} at once; "
          f"per fetch: NEOWISE median {res['neowise_s'].median():.0f} s, ZTF median "
          f"{res['ztf_s'].median() if 'ztf_s' in res else float('nan'):.0f} s); "
          f"fetch errors on {int((res['errors'] != '').sum())}")
    print(res.groupby(["by", "class"], dropna=False).size())
    for col in ("same_years", "after"):
        if col in res:
            print(f"{col}: {res[col].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
