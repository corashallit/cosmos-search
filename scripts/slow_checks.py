"""Run the per-candidate checks (cosmos_search.slow.checks) on the strongest stars of a search's fading list.

    python scripts/slow_checks.py --fading R/fading.parquet --out R/checks [--n 100] [--schema gaiadr4] [--max-g 17]
        [--skip-class ECL] [--skip-still] [--skip-band] [--workers 6] [--lightcurves R/lightcurves [--lightcurves D]]
        [--surveys ztf,skymapper,noirlab[,asassn] | none] [--atlas-n 24] [--no-tess]
        [--ids D/reviewed.csv [--ids-where shortlisted]]
    python scripts/slow_checks.py --fading R/fading.parquet --out R/checks --atlas-only --atlas-n 100
        [--ids D/reviewed.csv --ids-where shortlisted]

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

Other surveys (--surveys, on by default; docs/arms/slow-changes.md "Other surveys"): each star is read from every
listed survey that this machine can reach (`cosmos_search.surveys`; ASAS-SN, which is read only when named, needs
port 9006; a survey that cannot be read is named once at the start and left out), and compared
with Gaia's change (`slow.independent`): `same_k`, `same_k_err` and `same_years` say whether the surveys saw
Gaia's change over Gaia's own dates (1 the same change, 0 none; seen, not_seen, opposite, undecided, no_data),
`after` and its columns what they saw since Gaia's last visit, each also per survey (`<survey>_k`, ...), with
`<survey>_dilution` the share of neighbours' light in that survey's beam. For stars brighter than TESS_MAX_G,
`tess_amp` is the variability within a month from TESS (`surveys.tess_activity`), `tess_quiet` whether that is no
more than constant stars show, and `activity_ratio` Gaia's change as a multiple of it (a lower limit for a quiet
star). Writes surveys.parquet (all epochs read) and tess.parquet (one row per star and sector).

ATLAS (--atlas-n N, off by default; needs a token). One star's forced photometry takes about 18.5 minutes of
ATLAS's server time in a queue shared by all its users, so an account gets about one star an hour (measured
2026-11-11, `surveys/atlas.py`), and the checks of 10^3 stars cannot wait for it. So ATLAS is asked last and only
about the N strongest stars it covers: every file above is written first, without ATLAS, and the steps that read
them can start as soon as this script prints "written". The script then stays running, sends those N stars to
ATLAS strongest first, ATLAS_AT_ONCE at a time, and after each answer adds the star's ATLAS epochs to
surveys.parquet, works out its comparison columns again with ATLAS among the surveys, and rewrites checks.csv
(each file is replaced in one step, so a reader never sees half of one). Stopping the script (Ctrl-C or a
TERM signal) loses nothing that was written and takes its requests that have not been answered off ATLAS's queue,
so that they do not hold back the account's next ones. ATLAS's answers are also kept in this machine's response
cache. A request that failed is in the star's `atlas_error`, apart from `errors`.

--atlas-only carries on from the checks already in --out and fetches nothing else: it sends ATLAS the N strongest
of their stars that ATLAS covers and has not answered yet (a failed request counts as not answered), and rewrites
the same two files after each answer. That is the way to go on after a stop, and to send a review's shortlist.

--ids FILE keeps a run to the stars in a CSV's source_id column (all of them unless --n is given), and
--ids-where COLUMN to its rows where that column is true: `--atlas-only --ids D/reviewed.csv --ids-where
shortlisted` sends ATLAS the stars the review put on the shortlist (`slow_review.py table`), strongest first.
"""

import argparse
import os
import queue
import signal
import threading
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
from cosmos_search.slow.search import LC_G_MAX
from cosmos_search.surveys import tess_activity

TESS_MAX_G = 16.0      # TGLC has no light curve for stars fainter than about Tmag 16
# ATLAS places a new task by how many of the account's tasks are already waiting: with two sent at a time one is
# near the head of the queue while the other runs, and more only wait further back (`surveys/atlas.py`).
ATLAS_AT_ONCE = 2
STOP_SECONDS = 60.0    # on a stop, how long to wait for the requests still waiting to be taken off ATLAS's queue
MIRROR_TAP = "https://gaia.aip.de/tap"
_esa_failed = False
_mirror = None

GAIA_COLS = ("ra", "dec", "pmra", "pmdec", "parallax", "parallax_error", "ref_epoch", "phot_g_mean_mag", "bp_rp", "ruwe",
             "ipd_frac_multi_peak", "phot_bp_rp_excess_factor", "best_class_name", "best_class_score")
RELEASE_OF = {"gaiadr3": "dr3", "gaiadr4": "dr4"}


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
    """Astrometry, mean photometry and DPAC's best variability class for the given source ids, under DR3's column
    names whatever the release calls them and whichever tables hold them (targets/releases.py). From ESA's
    archive, or from the AIP mirror once ESA has failed (`gaia_query`)."""
    global _esa_failed, _mirror
    from cosmos_search.targets import releases

    release = RELEASE_OF.get(schema, schema)
    if not _esa_failed:
        try:
            return releases.fetch_star_rows(tap(), ids, release, GAIA_COLS)
        except Exception as e:  # noqa: BLE001 (ESA down or overloaded; the mirror holds the same tables)
            _esa_failed = True
            print(f"ESA Gaia archive failed ({e!r:.100}); asking {MIRROR_TAP} first from here on", flush=True)
    if _mirror is None:
        _mirror = Tap(MIRROR_TAP, job_slots=2)
    return releases.fetch_star_rows(_mirror, ids, release, GAIA_COLS)


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


def other_surveys(r, z, names, schema, with_tess, have=None, **options):
    """The comparison with other surveys for one candidate (module notes): (columns, epochs read, TESS rows per
    sector or None, error text). z: the star's ZTF epochs when already read (with their `group` column). have:
    the star's epochs from surveys read earlier, which those of `names` are added to before the comparison (ATLAS
    answers long after the rest, `atlas_step`). options: {survey: {keyword: value}} for that survey's reader."""
    row, err = {}, ""
    lc, failed = surveys.fetch_all(r, [s for s in names if s != "ztf"], **options)
    for s, e in failed.items():
        err += f"{s}: {e:.120}; "
    if "ztf" in names and z is not None and len(z) and surveys.covers("ztf", r.dec, r.phot_g_mean_mag):
        lc = pd.concat([lc, z.assign(survey="ztf", mag=np.nan)[surveys.COLUMNS]], ignore_index=True)
    if have is not None and len(have):
        lc = pd.concat([have, lc], ignore_index=True) if len(lc) else have
    change = ind.gaia_change(r)
    c = {}
    if len(lc):
        try:
            nb = gaia_query(ind.neighbours_adql(r.ra, r.dec, schema))
            c = ind.dilutions(nb, r.source_id, r.ra, r.dec, r.phot_g_mean_mag, pd.unique(lc["survey"]))
        except Exception as e:  # noqa: BLE001 (without neighbours the comparison runs undiluted, and says so)
            err += f"neighbours: {e!r:.120}; "
    if names:
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


def write(df, path):
    """Write a table (csv or parquet by the name) so that a reader never sees half of it: the ATLAS step rewrites
    checks.csv and surveys.parquet while later steps may be reading them."""
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".csv":
        df.to_csv(tmp, index=False)
    else:
        df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def listed(path, where=None):
    """The source_ids of a CSV (--ids), only its rows where the column `where` is true when that is given."""
    t = pd.read_csv(path)
    if where:
        t = t[t[where].astype(str).str.lower().isin(["true", "1", "1.0"])]
    return t["source_id"].astype("int64").to_numpy()


def with_gaia(cand, schema):
    """Candidates with their gaia_source columns (`gaia_rows`) in place of the fading list's magnitude and colour."""
    return cand.drop(columns=["phot_g_mean_mag", "bp_rp"], errors="ignore").merge(
        gaia_rows(cand["source_id"].to_numpy(), schema), on="source_id", how="left")


def written(out, fading, schema):
    """The checks already in `out` (--atlas-only): (the candidates among the stars of `fading`, every row of
    checks.csv as a dict, {source_id: the star's epochs in surveys.parquet})."""
    old = pd.read_csv(out / "checks.csv")
    for col in ("errors", "atlas_error"):
        if col in old:
            old[col] = old[col].fillna("").astype(str)
    cand = with_gaia(candidates(fading[fading["id"].isin(old["source_id"])], len(fading)), schema)
    sp = out / "surveys.parquet"
    lc_of = {int(k): g.reset_index(drop=True) for k, g in pd.read_parquet(sp).groupby("source_id")} \
        if sp.exists() else {}
    return cand, old.to_dict("records"), lc_of


def unanswered(row):
    """Whether the ATLAS step still has to run for a star: it never did, or its last run recorded an error (a
    failed request, or the neighbours query failing after an answer, which then comes from the cache)."""
    seconds, error = row.get("atlas_s"), row.get("atlas_error")
    return seconds is None or pd.isna(seconds) or (isinstance(error, str) and error != "")


def atlas_first(cand, row_of, n):
    """The n strongest candidates that ATLAS covers and has not answered yet, strongest first."""
    rows = [r for r in cand.sort_values("strength", ascending=False).itertuples(index=False)
            if surveys.covers("atlas", r.dec, r.phot_g_mean_mag) and unanswered(row_of[r.source_id])]
    return rows[:n]


class Stopped(Exception):
    """The run was stopped while a star waited for ATLAS's answer."""


def atlas_step(stars, row_of, lc_of, schema, save, at_once=ATLAS_AT_ONCE, log=print):
    """ATLAS for `stars` (candidate rows, strongest first), `at_once` at a time and in that order (module notes).
    row_of: {source_id: the star's row of the checks}, lc_of: {source_id: its epochs from the other surveys}; both
    are updated with each answer, and save() is then called. A star ATLAS fails on keeps its row, with the error
    in `atlas_error` (empty after an answer), so that a later run asks again.

    On a stop (KeyboardInterrupt) the requests still waiting are ended at their next look at the queue, which
    makes the reader take them off ATLAS's queue (`atlas.request` deletes its task on the way out); their stars'
    rows stay as they were. The workers are daemon threads, so a stop never waits longer than STOP_SECONDS."""
    todo, done, stop = queue.SimpleQueue(), queue.SimpleQueue(), threading.Event()
    for r in stars:
        todo.put(r)

    def pause(seconds):
        """The reader's wait between two looks at the queue; ends the request when the run is stopped."""
        if stop.wait(seconds):
            raise Stopped("the run was stopped before ATLAS answered")

    def work():
        while not stop.is_set():
            try:
                r = todo.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            try:
                cols, lc, _, err = other_surveys(r, None, ["atlas"], schema, False, have=lc_of.get(r.source_id),
                                                 atlas={"cache": True, "sleep": pause})
            except Exception as e:  # noqa: BLE001 (one star's failure is recorded and the next goes on)
                cols, lc, err = {}, None, f"atlas: {e!r:.120}; "
            done.put((r, cols, lc, err, time.time() - t0))

    workers = [threading.Thread(target=work, daemon=True) for _ in range(min(at_once, len(stars)))]
    for w in workers:
        w.start()
    try:
        for k in range(len(stars)):
            while True:      # waits a second at a time, so that a stop is seen whichever thread the signal reached
                try:
                    r, cols, lc, err, seconds = done.get(timeout=1.0)
                    break
                except queue.Empty:
                    continue
            row = row_of[r.source_id]
            row.update(cols)
            row["atlas_error"] = err
            row["atlas_s"] = seconds
            if lc is not None and len(lc):
                lc_of[r.source_id] = lc
            save()
            log(f"ATLAS {k + 1} of {len(stars)}: {r.source_id} G {r.phot_g_mean_mag:.1f} in {seconds / 60:.0f} "
                f"min: over Gaia's years {row.get('atlas_same', 'no_data')}, since "
                f"{row.get('atlas_after', 'no_data')}{'; ' + err if err else ''}", flush=True)
    except KeyboardInterrupt:
        stop.set()
        log("Stopped: taking the requests still waiting off ATLAS's queue. What was written stays.", flush=True)
        for w in workers:
            w.join(STOP_SECONDS)
        raise


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fading", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, help="the strongest N stars (default 100, or every star of --ids)")
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
    ap.add_argument("--surveys", default=",".join(s for s in surveys.DEFAULT if s != "atlas"),
                    help="other surveys to compare Gaia's change with, comma-separated, or none; asassn is read "
                         "only when named here, and ATLAS only through --atlas-n")
    ap.add_argument("--atlas-n", type=int, default=0,
                    help="after everything else is written, ask ATLAS about the N strongest stars it covers, "
                         "strongest first (about one star an hour; default 0, ATLAS is not asked)")
    ap.add_argument("--atlas-only", action="store_true",
                    help="fetch nothing else: carry on from the checks already in --out and ask ATLAS about the "
                         "--atlas-n strongest of their stars it has not answered yet")
    ap.add_argument("--ids", help="a CSV with a source_id column: check only these stars of the fading list")
    ap.add_argument("--ids-where", help="with --ids: only its rows where this column is true (shortlisted, for "
                                        "the D/reviewed.csv of slow_review.py table)")
    ap.add_argument("--no-tess", action="store_true", help="leave out the TESS variability of bright stars")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    asked = [] if a.surveys == "none" else [s for s in a.surveys.split(",") if s]
    if "atlas" in asked:
        ap.error("ATLAS answers about one star an hour and is asked with --atlas-n N (the N strongest stars, after "
                 "everything else is written), not through --surveys")
    if a.ids_where and not a.ids:
        ap.error("--ids-where needs --ids")
    if a.atlas_only and a.atlas_n <= 0:
        ap.error("--atlas-only needs --atlas-n N, the number of stars to send")
    names, why = surveys.usable(([] if a.atlas_only else asked) + (["atlas"] if a.atlas_n > 0 else []))
    for s, reason in why.items():
        print(f"{s} left out: {reason}", flush=True)
    with_atlas = "atlas" in names
    names = [s for s in names if s != "atlas"]
    fading = pd.read_parquet(a.fading)
    if a.ids:
        ids = listed(a.ids, a.ids_where)
        fading = fading[fading["id"].isin(ids)]
        print(f"{len(fading)} of the {len(ids)} stars of {a.ids} are in {a.fading}", flush=True)
        if not len(fading):
            return 1
    rows, lc_of = [], {}

    def save():
        """checks.csv and surveys.parquet as they stand; returns the checks table."""
        table = pd.DataFrame(rows).sort_values("strength", ascending=False)
        write(table, out / "checks.csv")
        if lc_of:
            write(pd.concat(list(lc_of.values()), ignore_index=True), out / "surveys.parquet")
        return table

    if a.atlas_only:
        if not with_atlas:
            return 1
        cand, old, lc_of_old = written(out, fading, a.schema)
        rows.extend(old)
        lc_of.update(lc_of_old)
        print(f"{len(rows)} stars in {out / 'checks.csv'}, {len(cand)} of them asked for here", flush=True)
    else:
        cand = first_checks(a, out, fading, names, rows, lc_of, save)
    if with_atlas:
        first = atlas_first(cand, {row["source_id"]: row for row in rows}, a.atlas_n)
        print(f"ATLAS: {len(first)} stars, strongest first, {ATLAS_AT_ONCE} at a time, about one an hour. "
              f"{out / 'checks.csv'} and surveys.parquet are rewritten after each answer; stopping this script "
              "loses nothing that is written.", flush=True)
        # A TERM signal (a stopped background job) ends the step the way Ctrl-C does.
        main_thread = threading.current_thread() is threading.main_thread()
        before = signal.signal(signal.SIGTERM, signal.default_int_handler) if main_thread else None
        try:
            atlas_step(first, {row["source_id"]: row for row in rows}, lc_of, a.schema, save)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, before)
        res = save()
        for col in ("same_years", "after", "atlas_same", "atlas_after"):
            if col in res:
                print(f"{col}: {res[col].value_counts().to_dict()}")
    return 0


def first_checks(a, out, fading, names, rows, lc_of, save):
    """Everything but ATLAS for the strongest stars: fills rows and lc_of, writes the files, returns the candidates."""
    if a.n is None:
        a.n = max(len(fading), 1) if a.ids else 100
    if a.skip_still:
        fading = fading[~fading["still"]]
    if a.skip_band:
        fading = fading[~fading["band_verdict"].isin(["g_only", "bp_rp_only"])]
    pool = a.n * (5 if a.skip_class else 1)
    while True:
        cand = with_gaia(candidates(fading, pool, a.max_g), a.schema)
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

    wise_all, ztf_all, tess_all = [], [], []
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(check_one, r, irsa, g_of.get(r.source_id), names, a.schema, not a.no_tess): r
                for r in cand.itertuples(index=False)}
        for fut in as_completed(futs):
            r = futs[fut]
            row, w, z, lc, sectors = fut.result()
            if gep is not None:
                row["g_lightcurve"] = r.source_id in g_of
            rows.append(row)
            wise_all.append(w)
            if z is not None:
                ztf_all.append(z)
            if lc is not None and len(lc):
                lc_of[r.source_id] = lc
            if sectors is not None and len(sectors):
                tess_all.append(sectors)
            print(r.source_id, r.by, f"G {r.phot_g_mean_mag:.1f}", r.best_class_name, row["errors"] or "", flush=True)
    res = save()
    write(pd.concat(wise_all, ignore_index=True), out / "neowise.parquet")
    if ztf_all:
        write(pd.concat(ztf_all, ignore_index=True), out / "ztf.parquet")
    if tess_all:
        write(pd.concat(tess_all, ignore_index=True), out / "tess.parquet")
    el = time.time() - t0
    print(f"{len(res)} stars in {el:.0f} s ({el / max(len(res), 1):.1f} s per star with {a.workers} at once; "
          f"per fetch: NEOWISE median {res['neowise_s'].median():.0f} s, ZTF median "
          f"{res['ztf_s'].median() if 'ztf_s' in res else float('nan'):.0f} s); "
          f"fetch errors on {int((res['errors'] != '').sum())}")
    print(res.groupby(["by", "class"], dropna=False).size())
    for col in ("same_years", "after"):
        if col in res:
            print(f"{col}: {res[col].value_counts().to_dict()}")
    print(f"{out / 'checks.csv'} and the epochs read are written: the steps that read them can start.", flush=True)
    return cand


if __name__ == "__main__":
    raise SystemExit(main())
