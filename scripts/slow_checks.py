"""Run the per-candidate checks (cosmos_search.slow.checks) on the strongest stars of a search's fading list.

    python scripts/slow_checks.py --fading R/fading.parquet --out R/checks [--n 100] [--schema gaiadr4] [--max-g 17]
        [--skip-class ECL] [--skip-still] [--skip-band] [--workers 6] [--lightcurves R/lightcurves [--lightcurves D]]

--fading is the list `scripts/slow_search.py flag` writes (ordered by strength). For each of the first --n stars
(optionally only those brighter than --max-g): gaia_source astrometry and quality columns and DPAC's variability
class, the colour of the change, NEOWISE W1/W2 against what each explanation predicts (inside the faint window
against outside for an episode, the slope over the light curve's years for a trend; a repeating dimming has no
single window and gets neither), and ZTF levels before and after 2020-01-20. Writes checks.csv (one row per star)
and the raw NEOWISE and ZTF epochs. With --lightcurves, also whether a strict-period eclipsing binary explains the
dips on G and ZTF together (`checks.eclipse`; `eb_explained` stars leave the review). The scoring pass keeps light
curves only for stars brighter than G 19 (`search.LC_G_MAX`), and the fading list also holds fainter stars: a star
with no G epochs in any --lightcurves directory gets no eclipsing-binary test, `g_lightcurve` False in checks.csv
and a line in no_lightcurve.csv. Fetch those stars through DataLink and give that directory as a second
--lightcurves, or keep the checks to brighter stars with --max-g. A failed fetch is recorded and the next star
goes on. Stars are checked
--workers at a time: one star's NEOWISE and ZTF fetches take one to two minutes, so 10^3 candidates one after
another would take over a day.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.dayone import external
from cosmos_search.energy import budget
from cosmos_search.gaia.archive import tap
from cosmos_search.net.tap import Tap
from cosmos_search.slow import checks
from cosmos_search.slow.search import LC_G_MAX

GAIA_COLS = ("ra", "dec", "pmra", "pmdec", "parallax", "parallax_error", "ref_epoch", "phot_g_mean_mag", "bp_rp", "ruwe",
             "ipd_frac_multi_peak", "phot_bp_rp_excess_factor", "best_class_name", "best_class_score")
RELEASE_OF = {"gaiadr3": "dr3", "gaiadr4": "dr4"}


def candidates(fading, n, max_g=None):
    f = fading if max_g is None else fading[fading["phot_g_mean_mag"] < max_g]
    return f.sort_values("strength", ascending=False).head(n).rename(columns={"id": "source_id"})


def gaia_rows(ids, schema):
    """Astrometry, mean photometry and DPAC's best variability class for the given source ids, under DR3's column
    names whatever the release calls them and whichever tables hold them (targets/releases.py)."""
    from cosmos_search.targets import releases

    return releases.fetch_star_rows(tap(), ids, RELEASE_OF.get(schema, schema), GAIA_COLS)


def at_2019(r):
    """Position propagated from ref_epoch to 2019.0 (NEOWISE-R's middle), small-angle."""
    dt = 2019.0 - r.ref_epoch
    pmra, pmdec = (0.0 if pd.isna(r.pmra) else r.pmra), (0.0 if pd.isna(r.pmdec) else r.pmdec)
    return (r.ra + pmra * dt / 3.6e6 / np.cos(np.radians(r.dec)), r.dec + pmdec * dt / 3.6e6)


def g_epochs(lightcurves, ids):
    """Good G epochs (source_id, t, flux relative to the median) of the given stars from one or more directories
    of light-curve files (epoch.SCHEMA, e.g. the kept light curves of `slow_search.py pass`), found by each
    file's source_id range. A star in more than one directory is read from the first that has it."""
    import pyarrow.parquet as pq

    from cosmos_search.slow import drift

    if not isinstance(lightcurves, (str, Path)):
        out, left = [], np.asarray(ids, np.int64)
        for d in lightcurves:
            got = g_epochs(d, left)
            out.append(got)
            left = np.setdiff1d(left, got["source_id"].to_numpy(np.int64))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["source_id", "t", "flux"])
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


def without_lightcurve(cand, gep):
    """The candidates with no G epoch in the light curves read (`g_epochs`): source_id, phot_g_mean_mag, strength
    and by, strongest first. No eclipsing-binary test can be run on them."""
    have = np.unique(gep["source_id"].to_numpy(np.int64)) if len(gep) else np.array([], np.int64)
    cols = [c for c in ("source_id", "phot_g_mean_mag", "strength", "by") if c in cand]
    return cand.loc[~cand["source_id"].isin(have), cols].sort_values("strength", ascending=False)


def check_one(r, irsa, g=None):
    """The checks for one candidate: (row, NEOWISE epochs, ZTF epochs or None) with fetch times. g: its G epochs,
    for the eclipsing-binary check (`checks.eclipse`, on G and ZTF together)."""
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
            z = external.ztf(r.source_id, r.ra, r.dec)
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
    return row, w.assign(source_id=r.source_id), z


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
    ap.add_argument("--lightcurves", action="append",
                    help="the run's kept light curves (`slow_search.py pass` R/lightcurves): adds the eclipsing-"
                         "binary check on G and ZTF. Repeatable: a second directory (light curves fetched through "
                         "DataLink) covers listed stars the pass kept no light curve for (fainter than G 19)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
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
        skip = {c.upper() for c in a.skip_class}  # DR4's class names may differ in case from DR3's
        keep = cand[~cand["best_class_name"].astype(str).str.upper().isin(skip)]
        if len(keep) >= a.n or len(cand) < pool:
            cand = keep.sort_values("strength", ascending=False).head(a.n)
            break
        pool *= 4
    irsa = Tap(checks.IRSA_TAP, job_slots=a.workers)
    gep = g_epochs(a.lightcurves, cand["source_id"]) if a.lightcurves else None
    g_of = dict(tuple(gep.groupby("source_id"))) if gep is not None else {}
    if gep is not None:
        none = without_lightcurve(cand, gep)
        none.to_csv(out / "no_lightcurve.csv", index=False)
        if len(none):
            faint = int((none["phot_g_mean_mag"] >= LC_G_MAX).sum())
            print(f"WARNING: {len(none)} of {len(cand)} stars have no G light curve in {', '.join(a.lightcurves)} "
                  f"({faint} of them fainter than G {LC_G_MAX:g}, where the scoring pass keeps none): no eclipsing-"
                  f"binary test for these. They are listed in {out / 'no_lightcurve.csv'}; fetch their light curves "
                  "through DataLink and add that directory as a second --lightcurves, or use --max-g.", flush=True)

    rows, wise_all, ztf_all = [], [], []
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(check_one, r, irsa, g_of.get(r.source_id)): r for r in cand.itertuples(index=False)}
        for fut in as_completed(futs):
            r = futs[fut]
            row, w, z = fut.result()
            if gep is not None:
                row["g_lightcurve"] = r.source_id in g_of
            rows.append(row)
            wise_all.append(w)
            if z is not None:
                ztf_all.append(z)
            print(r.source_id, r.by, f"G {r.phot_g_mean_mag:.1f}", r.best_class_name, row["errors"] or "", flush=True)
    res = pd.DataFrame(rows).sort_values("strength", ascending=False)
    res.to_csv(out / "checks.csv", index=False)
    pd.concat(wise_all, ignore_index=True).to_parquet(out / "neowise.parquet", index=False)
    if ztf_all:
        pd.concat(ztf_all, ignore_index=True).to_parquet(out / "ztf.parquet", index=False)
    el = time.time() - t0
    print(f"{len(res)} stars in {el:.0f} s ({el / max(len(res), 1):.1f} s per star with {a.workers} at once; "
          f"per fetch: NEOWISE median {res['neowise_s'].median():.0f} s, ZTF median "
          f"{res['ztf_s'].median() if 'ztf_s' in res else float('nan'):.0f} s); "
          f"fetch errors on {int((res['errors'] != '').sum())}")
    print(res.groupby(["by", "class"], dropna=False).size())


if __name__ == "__main__":
    main()
