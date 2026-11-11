"""Blind review of slow-change candidates on release day: from the checks to review files, and from the archived
verdicts to the reviewed list (docs/arms/slow-changes.md, "Release day, end to end", step 10).

    # 1. one review file per star the checks covered, with planted fades of known answer among them
    python scripts/slow_review.py files --checks L/checks --fading L/fading.parquet --lightcurves R/lightcurves \\
        --rule RULE --hosts S/lightcurves --out D [--run R] [--lists L] [--fakes N] [--schema gaiadr4] [--n 300]
        [--no-context] [--gaia FILE] [--with-eclipse]
    # 2. the reviews: the Workflow tool with scripts/stage_a_reviews.js and each list of arguments this prints
    #    (never the Agent tool: its subagents are given the thread's recent messages)
    python scripts/slow_review.py launch --dir D [--chunks 9]
    # 3. every transcript checked against the adopted reviewer, then the verdicts archived and pushed
    python scripts/stage_a_audit.py --transcripts <session>/subagents --prompts D/prompts --replies D/replies \\
        --out D/audit.json --reviewer prereg/reviewers/slow-cal-1.json
    python -m cosmos_search.vetting commit --packets D/products.json --replies D/replies --vetter slow-cal-1 \\
        --batch D/batch --archive prereg/verdicts/<name> --audit D/audit.json --plan D/plan.json.age
    git push
    # 4. the list to read, and how the reviewer did on the planted fades
    python scripts/slow_review.py table --dir D --checks L/checks

`files` builds each star's file through `slow.vet.review_file`, the path the calibration's files took
(docs/vetting.md §14), from what the run already holds: the star's row of the fading list (`slow_search.py flag`
and `cuts`), its epochs in the run's kept light curves with the rule's drift removed, and the NEOWISE and ZTF
epochs the checks fetched (`slow_checks.py`). Two things are fetched here and kept under D/work: the star's
gaia_source values (ESA TAP, by the release's own table names) and its AllWISE and 2MASS magnitudes (the
cross-match of `cosmos_search.xmatch`); `--gaia FILE` and `--no-context` replace or skip them (a practice run
without network). A star whose NEOWISE or ZTF fetch failed in the checks is left out and listed in
D/left_out.csv, so that the checks are run again for it rather than the reviewer reading a file with a hole.
So is a star without a kept light curve: the scoring pass keeps them for stars brighter than G 19 only
(`search.LC_G_MAX`). Either check only those (`slow_checks.py --max-g 19`), or pull the listed stars through
DataLink (`python -m cosmos_search.gaia.dr4_datalink --ids D/left_out.csv --out DL`) and run `files` again with
`--lightcurves R/lightcurves DL/lightcurves --overwrite` before any review is launched.

The files hold what the calibration's files held: the four tests of `vet.CALIBRATED_TESTS` and nothing from the
other surveys the checks read (SkyMapper, NOIRLab, ATLAS, TESS, ASAS-SN). The eclipsing-binary check of
`slow_checks.py --lightcurves` (`eb_explained`, its period) stays out of them unless `--with-eclipse`: no
calibration batch has measured the reviewer on files that carry it, and on fakes it also explained repeating
long dips (docs/arms/slow-changes.md, "Eclipsing binaries among the candidates", decision of 2026-11-11). It is
in the key and in the table of step 4 beside each verdict. files.json says whether the files are of the
calibrated kind (`calibrated_format`), and `table` says so when they are not.

Planted fades (docs/vetting.md §16). Among the stars' files go fakes with known answers: `--fakes N` of them
(default 30, or 5% of the stars' files where that is more): grey, dust and re-emitting-dust changes in the
calibration's proportions, shapes and signal-to-noise (`slow_cal.make_signal`). Each is put into the Gaia epochs
of a quiet star that is on no list, drawn in a secret order from `--hosts`: light-curve files that hold every
star of their sky, such as the thresholds' sample S/lightcurves. A fake is kept when the run would have flagged
it: it is scored as the pass scores a star, its trend measured against the run's reference of real stars and its
change date compared with the real stars of its sky pixel (`--run`, the pass output with summary/, by default the
directory above the first --lightcurves; `--lists`, the flag output with flags.parquet and reference.parquet, by
default the directory of --fading), as `scripts/slow_inject_test.py` does for the recall table. The same change is
put into the host's NEOWISE and ZTF epochs, fetched here the way the checks fetch a candidate's, and the file is
built by the same `vet.review_file` from the same kinds of input, so a fake's file differs from a star's only in
holding a change that was put there. Which files are fakes, their hosts and their answers are written only
encrypted to the project's two public keys (D/plan.json.age and D/key.json.age; D/plan.json holds the counts and
the plan's hash, which `vetting commit --plan` writes into the verdict archive). `table` opens the key with COSMOS_REVEAL_KEY once the
verdicts are committed and pushed, writes the stars' list without the fakes (D/reviewed.csv) and the fakes'
result against the calibration (D/fakes.json, D/fakes_reviewed.csv). `--fakes 0` builds a batch without them:
practice only, because nothing then says whether the reviewer of that batch still finds what the calibration's
reviewer found. Its key holds no answers and is not encrypted.
"""

import argparse
import json
import math
import random
import secrets
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cosmos_search.inject import lightcurve as inj
from cosmos_search.inject.sealed import write_sealed
from cosmos_search.slow import checks, detect, drift, rule_v3, search, summary, vet
from cosmos_search.vetting import packet as packet_mod
from cosmos_search.vetting import reviewer
from cosmos_search.vetting.__main__ import pending, prompt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slow_cal  # the calibration's fakes: make_signal, SHAPES, expected_checks, the public keys
import slow_search  # reading a run's summaries by source_id range; the fading list's columns

ROOT = Path(__file__).resolve().parents[1]
RELEASE_OF = {"gaiadr3": "dr3", "gaiadr4": "dr4"}
# Planted fades (docs/vetting.md §16): the calibration's fakes worth following up, in its proportions (25 grey, 15
# dust and 10 re-emitting dust of 50; §14) and at its signal-to-noise.
FAKE_MIX = (("grey", 0.5), ("dust", 0.3), ("reemit", 0.2))
FAKE_SNR_MIN = 20.0
FAKE_FLOOR, FAKE_SHARE = 30, 0.05   # per batch: at least 30, or this share of the stars' files where that is more
FAKE_SPARE = 0.5                    # spares drawn per flavour, for hosts that turn out unusable
HOSTS_PER_FILE = 40                 # hosts tried from one light-curve file before the next is opened
RECALL_FLOOR = 0.78                 # one-sided 95% lower bound of the calibration's recall on these fakes (44 of 50)
CALIBRATION_RECALL = (44, 50)
RECIPIENTS = slow_cal.RECIPIENTS
HOST_COLUMNS = ("source_id", "g_mag", "bp_rp", "n_visits", "sigma_visit", "quiet")
NEAR_COLUMNS = ("source_id", "s_fade", "s_rise", "dim_t0", "rise_t0", "sigma_visit")
# gaia_source values the review file shows (vet.SOURCE_KEYS) and the astrometry the cross-match propagates
# (xmatch.surveys.GAIA_ASTROMETRY), under DR3's names (targets/releases.py maps them for each release).
GAIA_COLS = ("ra", "dec", "ref_epoch", "parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error",
             "ra_error", "dec_error", "ra_dec_corr", "ra_parallax_corr", "ra_pmra_corr", "ra_pmdec_corr",
             "dec_parallax_corr", "dec_pmra_corr", "dec_pmdec_corr", "parallax_pmra_corr", "parallax_pmdec_corr",
             "pmra_pmdec_corr", "phot_g_mean_mag", "bp_rp", "ruwe", "ipd_frac_multi_peak", "phot_bp_rp_excess_factor")
ECLIPSE_COLS = ("eb_explained", "eb_periodic_unphysical", "eb_delta_bits", "eb_period", "eb_duration_ok",
                "eb_target_covered", "eb_n_dip_epochs", "eb_n_epochs", "eb_bands")
ECLIPSE_TEXT = (
    "Whether one strict period explains which G and ZTF epochs are in dip (the deepest G epoch is the selected "
    "dip): bits_saved is how much shorter the list of dip epochs is to write down given the period, and explained "
    "(1 or 0) means it saves enough and the dips are short enough for two stars at that period. The duration is measured "
    "from unbroken runs of dip epochs, which on ZTF's cadence cannot measure dips of many days, so a strictly "
    "periodic dimming that lasts days to weeks can also come out explained: compare how long the dips last with "
    "the period in the light curves. period_unphysical: the dips are too long for any pair of stars at that period.")


def fading_rows(path, ids):
    """The given stars' rows of a fading list (a Parquet file whose id column is `id` or `source_id`)."""
    col = "id" if "id" in pq.read_schema(path).names else "source_id"
    f = pd.read_parquet(path, filters=[(col, "in", [int(x) for x in ids])])
    return f.rename(columns={"id": "source_id"})


def checked_stars(checks_dir, fading, n=None, keep_fetch_errors=False):
    """(stars to review, stars left out with the reason): the rows of checks.csv joined to their rows of the
    fading list (a path, or a frame), strongest first."""
    c = pd.read_csv(Path(checks_dir) / "checks.csv")
    c["errors"] = c["errors"].fillna("").astype(str)
    f = fading.rename(columns={"id": "source_id"}) if isinstance(fading, pd.DataFrame) else \
        fading_rows(fading, c["source_id"])
    stars = c[["source_id", "errors", *[k for k in ECLIPSE_COLS if k in c]]].merge(f, on="source_id", how="left")
    stars = stars.sort_values("strength", ascending=False, na_position="last")
    left = [(int(s), "not on the fading list") for s in stars.loc[stars["by"].isna(), "source_id"]]
    stars = stars[stars["by"].notna()]
    if not keep_fetch_errors:
        bad = stars["errors"].str.contains("NEOWISE|ZTF")
        left += [(int(r.source_id), f"fetch failed in the checks: {r.errors}") for r in stars[bad].itertuples()]
        stars = stars[~bad]
    if n:
        stars = stars.head(n)
    return stars.reset_index(drop=True), left


def corrected_epochs(lightcurves, stars, table):
    """Drift-corrected epochs of the given stars from one or more directories of light-curve files (epoch.SCHEMA:
    the run's kept light curves, a DataLink pull), found by each file's source_id range; a star in two
    directories is read from the first. stars: source_id, phot_g_mean_mag, bp_rp."""
    cells = drift.star_cells(stars)
    ids = np.sort(stars["source_id"].to_numpy(np.int64))
    dirs = [lightcurves] if isinstance(lightcurves, (str, Path)) else list(lightcurves)
    out = []
    for f in [f for d in dirs for f in sorted(Path(d).glob("*.parquet"))]:
        md = pq.read_metadata(f)
        col = md.schema.to_arrow_schema().get_field_index("source_id")
        st = [md.row_group(i).column(col).statistics for i in range(md.num_row_groups)]
        st = [x for x in st if x is not None and x.has_min_max]
        if not st:
            continue
        mine = ids[(ids >= min(x.min for x in st)) & (ids <= max(x.max for x in st))]
        if len(mine):
            t = pq.read_table(f, filters=[("source_id", "in", [int(x) for x in mine])])
            if t.num_rows:
                out.append(vet.corrected(t, cells, table))
                ids = np.setdiff1d(ids, out[-1]["source_id"].unique())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["source_id", "band", "t", "frac", "err_frac", "fov", "transit_id", "frac_corr"])


def gaia_rows(ids, schema, cache):
    """GAIA_COLS for the given stars, fetched once and kept at `cache`."""
    cache = Path(cache)
    if cache.exists():
        have = pd.read_parquet(cache)
        if set(ids) <= set(have["source_id"]):
            return have
    from cosmos_search.gaia.archive import tap
    from cosmos_search.targets import releases

    rows = releases.fetch_star_rows(tap(), ids, RELEASE_OF.get(schema, schema), GAIA_COLS)
    cache.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(cache, index=False)
    return rows


def infrared_context(gaia_path, out):
    """AllWISE and 2MASS magnitudes of the stars in gaia_path (the cross-match's context table), kept under out."""
    out = Path(out).resolve()                # the cross-match runs from the repository, whatever --out is relative to
    if not (out / "context.parquet").exists():
        subprocess.run([sys.executable, "-m", "cosmos_search.xmatch", str(Path(gaia_path).resolve()), "--surveys",
                        "allwise", "2mass", "--context", "--out", str(out)], check=True, cwd=ROOT)
    return pd.read_parquet(out / "context.parquet").set_index("source_id")


def context_of(xm, sid):
    from cosmos_search.dayone.vet import CONTEXT_KEYS, wise_limits

    if xm is None or sid not in xm.index:
        return {}
    row = xm.loc[sid]
    ctx = {name: (float(row[col]) if col in row.index and pd.notna(row[col]) else float("nan"))
           for col, name in CONTEXT_KEYS.items()}
    return wise_limits(ctx, row)


def eclipse_of(r):
    """The eclipsing-binary check's columns of one checks.csv row, as plain values (None where absent)."""
    out = {}
    for k in ECLIPSE_COLS:
        v = r.get(k)
        out[k] = None if v is None or (not isinstance(v, str) and pd.isna(v)) else vet._plain(v)
    return out


def eclipse_test(ecl):
    if ecl.get("eb_explained") is None:
        return {}
    # The file builder keeps a test's numbers and its description, so yes/no values go in as 1 or 0.
    flag = lambda v: None if v is None else int(bool(v))
    return {"eclipsing_binary": {
        "explained": flag(ecl["eb_explained"]), "period_d": ecl.get("eb_period"),
        "bits_saved": ecl.get("eb_delta_bits"), "period_unphysical": flag(ecl.get("eb_periodic_unphysical")),
        "dip_epochs": ecl.get("eb_n_dip_epochs"), "epochs": ecl.get("eb_n_epochs"),
        "description": ECLIPSE_TEXT}}


def build(stars, epochs, gaia, xm, neowise, ztf, release="dr4", with_eclipse=False):
    """(packets, keys, refused, left out) for the stars: one review file each, through `vet.review_file`."""
    gaia = gaia.set_index("source_id", drop=False)
    ep_of = dict(tuple(epochs.groupby("source_id")))
    nw_of = dict(tuple(neowise.groupby("source_id"))) if len(neowise) else {}
    zt_of = dict(tuple(ztf.groupby("source_id"))) if len(ztf) else {}
    packets, keys, refused, left = [], [], [], []
    for r in stars.to_dict("records"):
        sid = int(r["source_id"])
        ep = ep_of.get(sid)
        if ep is None or (ep["band"] == 0).sum() < detect.MIN_VISITS:
            left.append((sid, "no light curve among the run's kept light curves"))
            continue
        if sid not in gaia.index:
            left.append((sid, "no gaia_source row"))
            continue
        g = gaia.loc[sid]
        nw = nw_of.get(sid)
        nw = nw.drop(columns="source_id") if nw is not None else checks.neowise_clean([])
        zt = zt_of.get(sid)
        zt = zt.drop(columns="source_id") if zt is not None else pd.DataFrame(columns=["band", "t", "flux", "err"])
        ecl = eclipse_of(r)
        try:
            pk, key, tests = vet.review_file(f"gaia_{release}:{sid}", r, ep, g, context_of(xm, sid), nw, zt,
                                             extra_tests=eclipse_test(ecl) if with_eclipse else None)
        except packet_mod.LeakError as e:
            refused.append({"source_id": sid, "error": str(e)})
            continue
        key.update(kind="candidate", label=str(r["by"]), source_id=sid, flagged_by=str(r["by"]), flagged=True,
                   g=float(g["phot_g_mean_mag"]), strength=float(r["strength"]), measured=vet.measured(tests),
                   eclipse=ecl, truth={})
        packets.append(pk)
        keys.append(key)
    return packets, keys, refused, left


# --- planted fades (docs/vetting.md §16) --------------------------------------------------------------------------


def n_fakes(n_stars):
    """Planted fades for a batch of n_stars files."""
    return max(FAKE_FLOOR, math.ceil(FAKE_SHARE * n_stars))


def fake_slots(n):
    """How many of n planted fades are of each flavour: FAKE_MIX, the remainder going to the largest fractions."""
    exact = [(f, n * w) for f, w in FAKE_MIX]
    out = {f: math.floor(x + 1e-9) for f, x in exact}
    for f, _ in sorted(exact, key=lambda e: e[1] - out[e[0]], reverse=True)[: n - sum(out.values())]:
        out[f] += 1
    return out


def pass_options(run):
    """(marks, own) of the pass that made `run` (its pass.json), so that a fake's epochs are chosen as the run's
    were (`drift.normalise`)."""
    p = Path(run) / "pass.json"
    o = json.loads(p.read_text()) if p.exists() else {}
    return not o.get("no_marks", False), not o.get("no_own_test", False)


def scoring_context(run, lists, rule_dir, marks=True, own=True):
    """What scoring a fake as the run scored its stars needs: the rule, the run's reference of real stars
    (lists/reference.parquet) and, for the shared-date test, the real stars of a level-6 sky pixel, read from the
    run's summaries when first asked for."""
    thr, thr_p, table = rule_v3.load(rule_dir)
    sfiles = slow_search._files(Path(run) / "summary")
    if not sfiles:
        raise SystemExit(f"no summaries under {Path(run) / 'summary'}: planted fades are scored against the run's "
                         "stars (--run)")
    sranges = slow_search._id_ranges(sfiles)
    width = 2 ** 35 * 4 ** (12 - rule_v3.SHARED_LEVEL)
    cache = {}

    def near(pix):
        if pix not in cache:
            sm = slow_search._read(sfiles, sranges, pix * width, (pix + 1) * width, NEAR_COLUMNS)
            if len(sm):
                sm = sm[np.isfinite(sm["sigma_visit"].to_numpy(float))].reset_index(drop=True)
            else:
                sm = pd.DataFrame({c: pd.Series(dtype="int64" if c == "source_id" else "float64")
                                   for c in NEAR_COLUMNS})
            cache[pix] = sm
        return cache[pix]

    return {"thr": thr, "thr_p": thr_p, "table": table, "rule": rule_dir, "marks": marks, "own": own,
            "ref": pd.read_parquet(Path(lists) / "reference.parquet"), "near": near, "sfiles": sfiles,
            "sranges": sranges}


def _bands(lc):
    """(band name of each row, as an array of codes, and the names by code) of a light-curve table."""
    band = lc["band"].combine_chunks()
    return np.asarray(band.indices), [str(x) for x in band.dictionary.to_pylist()]


def injected(lc, sig):
    """One star's light-curve rows (epoch.SCHEMA) with the signal multiplied into flux and error, band by band
    (`inject.lightcurve.inject`)."""
    codes, names = _bands(lc)
    t = np.asarray(lc["t"], float)
    k = np.ones(len(t))
    for code, name in enumerate(names):
        m = codes == code
        if m.any():
            k[m] = sig.factor(t[m], name)
    k = np.maximum(k, 1e-3)
    for c in ("flux", "err"):
        lc = lc.set_column(lc.schema.get_field_index(c), c, pa.array(np.asarray(lc[c], float) * k))
    return lc


def score_fake(lc, g_mag, ctx):
    """The fading-list row of one star's light curve with a fake in it, scored as the run scores a star (the
    steps of `scripts/slow_inject_test.py`): `search.score_all`, the trend measured against the run's reference of
    real stars at the host's magnitude (the fake never enters the reference), the shared-date test against the
    real stars of its sky pixel, `search.flag`, and the band check of `slow_search.py cuts` (without a blend
    count). Columns as the fading list's (`slow_search.LIST_COLUMNS`) with `flagged` and `band_verdict`; None for
    a light curve too short to score."""
    summ, stats, _ = search.score_all(lc, ctx["table"], ctx["thr"], ctx["thr_p"], marks=ctx["marks"], own=ctx["own"])
    m = search.merge(summ, stats if len(stats) else None)
    m = m[np.isfinite(m["sigma_visit"].to_numpy(float))].reset_index(drop=True)
    if len(m) != 1:
        return None
    m = summary.apply_reference(m, ctx["ref"], g_mag=np.array([g_mag], float))
    pix = drift.sky_pixel(m["source_id"].to_numpy(), rule_v3.SHARED_LEVEL)
    sh = rule_v3.shared(ctx["near"](int(pix[0])), targets=m, target_patch=pix)
    fl = search.flag(m, ctx["rule"], shared_flags=sh)
    fl["phot_g_mean_mag"], fl["bp_rp"] = m["g_mag"].to_numpy(), m["bp_rp"].to_numpy()   # as `slow_search.py flag`
    fl = pd.concat([fl, checks.band_check(fl, np.full(len(fl), np.nan))], axis=1)
    return {c: (fl[c].iloc[0] if c in fl else np.nan) for c in (*slow_search.LIST_COLUMNS, "flagged", "band_verdict")}


def draw_fakes(n, hosts_dir, ctx, exclude, rnd):
    """Planted fades the run would have flagged: ({flavour: [fake, ...]} with spares, {flavour: number wanted},
    counts).

    Hosts are quiet stars brighter than G 19 with scores that are in a light-curve file of hosts_dir and in the
    run's summaries, and not in `exclude` (every star of flags.parquet and of the checks). Files are opened in the
    order `rnd` gives; up to HOSTS_PER_FILE hosts of a file are tried, in `rnd`'s order, and each host carries one
    draw: a flavour still wanted, one of the calibration's shapes (`slow_cal.SHAPES`) and a seed
    (`slow_cal.make_signal`). The draw is kept when its signal-to-noise is at least FAKE_SNR_MIN (`vet.snr`, as
    the calibration's), the rule flags it, and its G change agrees with BP and RP (the checks' --skip-band).

    A fake is its plan entry (host, flavour, shape, seed, snr, by, strength, the host's values used) and, under
    names that start with "_", the injected light curve, the signal and its fading-list row. Nothing printed
    names a host."""
    want = fake_slots(n)
    need = {f: k + math.ceil(FAKE_SPARE * k) for f, k in want.items() if k}
    got = {f: [] for f in need}
    files = slow_search._files(hosts_dir)
    rnd.shuffle(files)
    tried = opened = 0
    for path in files:
        if all(len(got[f]) >= need[f] for f in need):
            break
        (lo, hi), = slow_search._id_ranges([path])
        sm = slow_search._read(ctx["sfiles"], ctx["sranges"], lo, hi + 1, HOST_COLUMNS)
        if not len(sm):
            continue
        ok = (np.isfinite(sm["sigma_visit"].to_numpy(float)) & sm["quiet"].to_numpy(bool)
              & (sm["g_mag"].to_numpy(float) < search.LC_G_MAX) & (sm["n_visits"].to_numpy() >= detect.MIN_VISITS)
              & ~sm["source_id"].isin(exclude).to_numpy())
        pool = sm[ok]
        if not len(pool):
            continue
        pool = pool.iloc[rnd.sample(range(len(pool)), min(len(pool), HOSTS_PER_FILE))]
        lc = pq.read_table(path, filters=[("source_id", "in", [int(x) for x in pool["source_id"]])])
        sid = np.asarray(lc["source_id"])
        opened += 1
        for h in pool.itertuples(index=False):
            open_ = [f for f in need if len(got[f]) < need[f]]
            if not open_:
                break
            star = lc.filter(pa.array(sid == h.source_id))
            if not star.num_rows:
                continue
            codes, names = _bands(star)
            tg = np.sort(np.asarray(star["t"], float)[codes == names.index("G")]) if "G" in names else np.array([])
            if len(tg) < detect.MIN_VISITS:
                continue
            tried += 1
            flavour = rnd.choices(open_, weights=[need[f] - len(got[f]) for f in open_])[0]
            spec = {"host": int(h.source_id), "flavour": flavour, "shape": rnd.choice(slow_cal.SHAPES),
                    "seed": rnd.getrandbits(63), "bp_rp": float(h.bp_rp), "g_mag": float(h.g_mag),
                    "sigma_visit": float(h.sigma_visit), "t_min": float(tg[0]), "t_max": float(tg[-1])}
            sig, extra = slow_cal.make_signal(spec, spec["t_min"], spec["t_max"], spec["bp_rp"])
            tv, _ = detect.visits(tg, tg)
            snr = vet.snr(sig, tv, spec["sigma_visit"])
            if not snr >= FAKE_SNR_MIN:
                continue
            fake = injected(star, sig)
            row = score_fake(fake, spec["g_mag"], ctx)
            if row is None or not bool(row["flagged"]) or row["band_verdict"] in ("g_only", "bp_rp_only"):
                continue
            spec.update(snr=float(snr), by=str(row["by"]), strength=float(row["strength"]))
            got[flavour].append({**spec, "_lc": fake, "_sig": sig, "_extra": extra, "_row": row})
    return got, want, {"hosts_tried": tried, "host_files_opened": opened, "host_files": len(files)}


_irsa = None


def host_neowise(g):
    """Clean NEOWISE single exposures at a host, by the query the checks run for a candidate
    (`slow_checks.check_one`): the position at 2019, `checks.neowise_adql`, `checks.neowise_clean`."""
    global _irsa
    if _irsa is None:
        from cosmos_search.net.tap import Tap

        _irsa = Tap(checks.IRSA_TAP, job_slots=4)
    ra, dec = slow_cal.at_2019(g)
    return checks.neowise_clean(_irsa.query(checks.neowise_adql(ra, dec), sync=True).to_pandas())


def host_ztf(g):
    """ZTF epochs at a host with each ZTF object's id (`vet.ztf_raw`: the service and the cuts of the checks'
    `external.ztf`); None south of declination -30, where the checks do not ask."""
    return vet.ztf_raw(float(g["ra"]), float(g["dec"])) if float(g["dec"]) > -30 else None


def fetch_host(g, tries=3, wait=5.0):
    """(NEOWISE, ZTF) of one host, or the last error after `tries` attempts."""
    err = None
    for k in range(tries):
        try:
            return host_neowise(g), host_ztf(g)
        except Exception as e:  # noqa: BLE001 (a fake whose fetch fails is dropped; a spare takes its place)
            err = e
            if k + 1 < tries:
                time.sleep(wait * 2 ** k)
    return err


def neowise_with(clean, sig):
    """A host's clean NEOWISE exposures with the signal in, each band relative to its median again (as
    `slow_cal.py build` made the calibration's)."""
    if not len(clean):
        return clean
    k = np.ones(len(clean))
    for b in ("W1", "W2"):
        m = (clean["band"] == b).to_numpy()
        k[m] = sig.factor(clean["mjd"].to_numpy(float)[m] - checks.BTJD_T0_MJD, b)
    out = clean.assign(flux=clean["flux"].to_numpy(float) * k, err=clean["err"].to_numpy(float) * k)
    for b in ("W1", "W2"):
        m = (out["band"] == b).to_numpy()
        if m.any():
            med = float(np.median(out.loc[m, "flux"]))
            out.loc[m, ["flux", "err"]] = out.loc[m, ["flux", "err"]] / med
    return out


def ztf_with(raw, sig):
    """A host's ZTF epochs (`vet.ztf_raw`) with the signal in, each ZTF object relative to its median magnitude
    and with its `group`, as `external.ztf` gives a candidate's to the checks."""
    out = []
    for oid, r in raw.groupby("oid") if raw is not None and len(raw) else ():
        k = np.maximum(sig.factor(r["t"].to_numpy(float), str(r["band"].iloc[0])), 1e-3)
        mag = r["mag"].to_numpy(float) - 2.5 * np.log10(k)
        flux = 10 ** (-0.4 * (mag - np.median(mag)))
        out.append(pd.DataFrame({"band": r["band"].to_numpy(), "t": r["t"].to_numpy(float), "flux": flux,
                                 "err": flux * r["magerr"].to_numpy(float) * np.log(10) / 2.5, "group": str(oid)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["band", "t", "flux", "err", "group"])


def fake_eclipse(lc, zt, g):
    """The checks' eclipsing-binary test (`checks.eclipse`, as `slow_checks.py --lightcurves` runs it) on a fake's
    G and ZTF epochs; empty values when it fails."""
    from cosmos_search.dayone import external

    ep = drift.normalise(lc)
    ep = ep[ep["band"] == 0]
    src = {k: g.get(k, np.nan) for k in ("phot_g_mean_mag", "bp_rp", "parallax", "parallax_error")}
    try:
        return eclipse_of(checks.eclipse(pd.DataFrame({"t": ep["t"].to_numpy(float), "flux": 1 + ep["frac"].to_numpy(float)}),
                                         zt if len(zt) else external._empty(), src))
    except Exception:  # noqa: BLE001 (as the checks: recorded as no result)
        return eclipse_of({})


def fake_file(fake, g, context, nw_clean, zt_raw, table, release="dr4", with_eclipse=False, eclipse_check=True):
    """(packet, key) of one planted fade, through the steps of a star's file (`build`): its fading-list row, its
    injected epochs with the rule's drift removed, the host's catalogue values and infrared magnitudes, and the
    host's NEOWISE and ZTF epochs with the same change in. The key holds the answer (`truth`)."""
    sid, sig = fake["host"], fake["_sig"]
    row = {**fake["_row"], "source_id": sid}
    cells = drift.star_cells(pd.DataFrame({"source_id": [sid], "phot_g_mean_mag": [row["phot_g_mean_mag"]],
                                           "bp_rp": [row["bp_rp"]]}))
    ep = vet.corrected(fake["_lc"], cells, table)
    nw, zt = neowise_with(nw_clean, sig), ztf_with(zt_raw, sig)
    ecl = fake_eclipse(fake["_lc"], zt, g) if eclipse_check else eclipse_of({})
    pk, key, tests = vet.review_file(f"gaia_{release}:{sid}", row, ep, g, context, nw, zt,
                                     extra_tests=eclipse_test(ecl) if with_eclipse else None)
    truth = slow_cal.expected_checks(sig, ep, row, nw)
    truth.update(flavour=fake["flavour"], shape=fake["shape"], snr=fake["snr"], **fake["_extra"],
                 signal=inj.truth(sig))
    key.update(kind="fake", label=fake["flavour"], source_id=sid, flagged_by=str(row["by"]), flagged=True,
               g=float(g["phot_g_mean_mag"]), strength=float(row["strength"]), measured=vet.measured(tests),
               eclipse=ecl, truth=truth)
    return pk, key


def build_fakes(drawn, want, gaia, xm, table, release="dr4", with_eclipse=False, eclipse_check=True, workers=4):
    """(packets, keys, the fakes built, counts): for each flavour the first `want` drawn fakes whose host has a
    catalogue row and moves on the sky (the checks' --skip-still), whose NEOWISE and ZTF fetches succeed and whose
    file passes the leak scan. A spare takes the place of one that does not."""
    gaia = gaia.drop_duplicates("source_id").set_index("source_id", drop=False)
    still = search.motion(gaia)["still"] if set(search.MOTION_COLUMNS) <= set(gaia.columns) else \
        pd.Series(False, index=gaia.index)
    queue = {f: [x for x in fakes if x["host"] in gaia.index and not bool(still.loc[x["host"]])]
             for f, fakes in drawn.items()}
    dropped = {"no catalogue row, or no measurable parallax and motion":
               sum(len(v) for v in drawn.values()) - sum(len(v) for v in queue.values()),
               "fetch failed": 0, "refused by the leak scan": 0}
    packets, keys, used, built = [], [], [], {f: 0 for f in drawn}
    while True:
        batch = []
        for f in queue:
            k = want[f] - built[f]
            batch, queue[f] = batch + queue[f][:k], queue[f][k:]
        if not batch:
            break
        with ThreadPoolExecutor(workers) as ex:
            fetched = list(ex.map(lambda x: fetch_host(gaia.loc[x["host"]]), batch))
        for x, res in zip(batch, fetched):
            if isinstance(res, Exception):
                dropped["fetch failed"] += 1
                continue
            try:
                pk, key = fake_file(x, gaia.loc[x["host"]], context_of(xm, x["host"]), res[0], res[1], table,
                                    release, with_eclipse, eclipse_check)
            except packet_mod.LeakError:
                dropped["refused by the leak scan"] += 1
                continue
            packets.append(pk)
            keys.append(key)
            used.append(x)
            built[x["flavour"]] += 1
    return packets, keys, used, {"built": built, "dropped": dropped}


def planted(a, d, n, table, listed, release, eclipse_check, rnd):
    """The batch's planted fades: (packets, keys, the public record written to D/plan.json). The sealed plan
    (D/plan.json.age) holds each fake's host, flavour, shape and seed, from which its signal is made again."""
    if not a.hosts:
        raise SystemExit("planted fades need --hosts: light-curve files that hold every star of their sky, such as "
                         "the thresholds' sample S/lightcurves. --fakes 0 builds a practice batch without them.")
    run = Path(a.run) if a.run else Path(a.lightcurves[0]).parent
    lists = Path(a.lists) if a.lists else Path(a.fading).parent
    marks, own = pass_options(run)
    ctx = scoring_context(run, lists, a.rule, marks and not a.no_marks, own and not a.no_own_test)
    exclude = {int(x) for x in pd.read_parquet(lists / "flags.parquet", columns=["id"])["id"]} | set(listed)
    drawn, want, counts = draw_fakes(n, a.hosts, ctx, exclude, rnd)
    print(f"planted fades: {sum(len(v) for v in drawn.values())} drawn with spares from {counts['hosts_tried']} "
          f"hosts tried in {counts['host_files_opened']} of {counts['host_files']} files", flush=True)
    hosts = [x["host"] for v in drawn.values() for x in v]
    work = d / "work" / "hosts"
    try:
        if a.gaia:
            g = pd.read_parquet(a.gaia)
            g = g[g["source_id"].isin(hosts)]
            work.mkdir(parents=True, exist_ok=True)
            g.to_parquet(work / "gaia.parquet", index=False)
        else:
            g = gaia_rows(hosts, a.schema, work / "gaia.parquet") if hosts else pd.DataFrame({"source_id": []})
        xm = None if a.no_context or not len(g) else infrared_context(work / "gaia.parquet", work / "xm")
        packets, keys, used, res = build_fakes(drawn, want, g, xm, table, release, a.with_eclipse, eclipse_check)
    finally:
        shutil.rmtree(work, ignore_errors=True)   # the hosts are named on disk only inside the sealed files
    if len(used) < n:
        raise SystemExit(f"only {len(used)} of {n} planted fades could be built ({counts['hosts_tried']} hosts "
                         f"tried in {counts['host_files_opened']} of {counts['host_files']} files; dropped: "
                         f"{res['dropped']}). No review file was written. Give host files that hold every star of "
                         "their sky (--hosts), run again if fetches failed, or pass --fakes with a smaller number.")
    sha = write_sealed({"fakes": json.loads(json.dumps([{k: v for k, v in x.items() if not k.startswith("_")}
                                                        for x in used], default=_json))},
                       d / "plan.json.age", RECIPIENTS)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT,
                            check=False).stdout.strip()
    rec = {"planted": len(used), "wanted": n, "by_flavour": res["built"], "wanted_by_flavour": want,
           "snr_min": FAKE_SNR_MIN, "plan_sha256": sha, **counts, "dropped": res["dropped"],
           "code_commit": commit or None}
    (d / "plan.json").write_text(json.dumps(rec, indent=1))
    return packets, keys, rec


def set_aside(d):
    """What belongs to review files that are being replaced (prompts, replies, the audit, committed verdicts and
    the tables made from them), moved to D/replaced-<k>/: their reviews are void, and the launch and table steps
    must not find them."""
    old = [p for p in (d / "prompts", d / "replies") if p.exists() and any(p.iterdir())]
    old += [p for p in (d / n for n in ("batch", "audit.json", "reviewed.csv", "fakes.json", "fakes_reviewed.csv"))
            if p.exists()]
    if old:
        dest = d / f"replaced-{1 + len(list(d.glob('replaced-*')))}"
        dest.mkdir()
        for p in old:
            p.rename(dest / p.name)


def write(d, packets, keys, rng=None, sealed=False):
    """products.json, prompts/, packet_ids.txt, the key and an empty replies/, in an order unrelated to the list's
    and to which files are fakes. sealed: the key holds answers and is written encrypted (key.json.age) only."""
    d = Path(d)
    order = list(range(len(packets)))
    (rng or secrets.SystemRandom()).shuffle(order)
    packets, keys = [packets[i] for i in order], [keys[i] for i in order]
    (d / "prompts").mkdir(parents=True, exist_ok=True)
    (d / "replies").mkdir(exist_ok=True)
    (d / "products.json").write_text(json.dumps(packets, default=_json))
    for p in packets:
        system, user = prompt(p)
        (d / "prompts" / f"{p['packet_id']}.md").write_text(f"{system}\n\n---\n\n{user}\n")
    (d / "packet_ids.txt").write_text("\n".join(p["packet_id"] for p in packets) + "\n")
    (d / ("key.json" if sealed else "key.json.age")).unlink(missing_ok=True)
    if sealed:
        write_sealed(json.loads(json.dumps(keys, default=_json)), d / "key.json.age", RECIPIENTS)
    else:
        (d / "key.json").write_text(json.dumps(keys, default=_json))
    return packets, keys


def _json(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


def cmd_files(a):
    d = Path(a.out)
    if (d / "products.json").exists():
        if not a.overwrite:
            raise SystemExit(f"{d} already holds review files; reviews may be under way. Pass --overwrite to replace "
                             "them (their reviews are then void).")
        set_aside(d)
    d.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.practice_seed) if a.practice_seed is not None else secrets.SystemRandom()
    stars, left = checked_stars(a.checks, a.fading, a.n, a.keep_fetch_errors)
    listed = {int(x) for x in stars["source_id"]} | {sid for sid, _ in left}
    print(f"{len(stars)} stars from {a.checks}, {len(left)} left out so far", flush=True)
    _, _, table = rule_v3.load(a.rule)
    epochs = corrected_epochs(a.lightcurves, stars[["source_id", "phot_g_mean_mag", "bp_rp"]], table)
    ids = stars["source_id"].to_numpy(np.int64)
    gaia = pd.read_parquet(a.gaia) if a.gaia else gaia_rows(ids, a.schema, d / "work" / "gaia.parquet")
    if a.gaia:
        (d / "work").mkdir(exist_ok=True)
        gaia[gaia["source_id"].isin(ids)].to_parquet(d / "work" / "gaia.parquet", index=False)
    xm = None if a.no_context else infrared_context(d / "work" / "gaia.parquet", d / "work" / "xm")
    neowise = pd.read_parquet(Path(a.checks) / "neowise.parquet")
    zp = Path(a.checks) / "ztf.parquet"
    ztf = pd.read_parquet(zp) if zp.exists() else pd.DataFrame(columns=["source_id", "band", "t", "flux", "err"])
    release = RELEASE_OF.get(a.schema, a.schema)
    packets, keys, refused, more = build(stars, epochs, gaia, xm, neowise, ztf, release, a.with_eclipse)
    left += more
    n_stars = len(packets)
    n_fake = n_fakes(n_stars) if a.fakes is None else a.fakes
    plan = {}
    if n_fake:
        # The fakes get the eclipsing-binary test when the checks ran it on the stars (slow_checks.py --lightcurves).
        fp, fk, plan = planted(a, d, n_fake, table, listed, release, "eb_explained" in stars, rnd)
        packets, keys = packets + fp, keys + fk
    else:
        for name in ("plan.json", "plan.json.age"):
            (d / name).unlink(missing_ok=True)
    write(d, packets, keys, rng=rnd, sealed=bool(n_fake))
    pd.DataFrame(left, columns=["source_id", "reason"]).to_csv(d / "left_out.csv", index=False)
    tests = sorted(set().union(*[p["tests"] for p in packets])) if packets else []
    rec = {"files": len(packets), "stars": n_stars, "planted_fades": plan.get("planted", 0), "left_out": len(left),
           "refused_by_the_leak_scan": refused,
           "checks": str(a.checks), "fading": str(a.fading), "lightcurves": [str(x) for x in a.lightcurves],
           "rule": str(a.rule),
           "schema": a.schema, "infrared_context": not a.no_context, "with_eclipse": bool(a.with_eclipse),
           "reviewer": reviewer.ADOPTED, "tests_in_the_files": tests,
           "calibrated_format": set(tests) <= set(vet.CALIBRATED_TESTS)}
    (d / "files.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    if not n_fake:
        print("No planted fades in this batch (--fakes 0): practice only. Nothing in it says whether its reviewer "
              "finds what the calibration's reviewer found (docs/vetting.md §16).")
    if not rec["calibrated_format"]:
        print("These files carry tests the calibration's files did not: the calibration's recall and "
              "false-shortlist rate (docs/vetting.md §14) do not describe their reviews.")
    return 0


def cmd_launch(a):
    """The `args` of scripts/stage_a_reviews.js for the files still without a reply, in --chunks lists (a workflow
    runs a few reviewers at a time; launch the lists together)."""
    d = Path(a.dir)
    ids = pending(d / "prompts", d / "replies")
    n = max(1, min(a.chunks, len(ids)))
    out = [reviewer.launch_args(d, ids[i::n]) for i in range(n)] if ids else []
    print(json.dumps({"script": "scripts/stage_a_reviews.js", "pending": len(ids), "args": out}, indent=1))
    return 0


def reviewed(verdicts, keys, checks_csv=None):
    """One row per reviewed star: the reviewer's score (the summed probability of the classes worth following up,
    `dayone.vet.INTERESTING`), whether it reaches the shortlist threshold, and the eclipsing-binary check beside
    it. A star whose review failed stays on the list to read, marked."""
    from cosmos_search.dayone.vet import INTERESTING, S_SHORTLIST

    rows = []
    for v in verdicts:
        k = keys[v.packet_id]
        s = sum(v.class_probs.get(c, 0.0) for c in INTERESTING) if v.ok else float("nan")
        ecl = k.get("eclipse") or {}
        rows.append({
            "source_id": int(k["source_id"]), "strength": k.get("strength"), "by": k.get("flagged_by"),
            "g": k.get("g"), "reviewed": bool(v.ok), "score": s,
            "shortlisted": bool(s >= S_SHORTLIST) if v.ok else True,
            "top_class": v.top() if v.ok else None,
            "top_class_p": max(v.class_probs.values()) if v.ok and v.class_probs else float("nan"),
            "p_unexplained": v.p_unexplained if v.ok else float("nan"), "recognised_as": v.recognised_as,
            "eb_explained": ecl.get("eb_explained"), "eb_period": ecl.get("eb_period"),
            "eb_period_unphysical": ecl.get("eb_periodic_unphysical"),
            "review_error": v.error if not v.ok else "", "packet_id": v.packet_id})
    df = pd.DataFrame(rows)
    if checks_csv is not None and len(df):
        c = pd.read_csv(checks_csv)
        cols = [x for x in ("class", "ruwe", "fade_kind", "dim_depth", "colour_ratio", "colour_z_grey") if x in c]
        df = df.merge(c[["source_id", *cols]], on="source_id", how="left")
    return df.sort_values(["shortlisted", "score", "strength"], ascending=False, na_position="first").reset_index(
        drop=True) if len(df) else df


def fakes_reviewed(verdicts, keys):
    """One row per planted fade: its answer (flavour, shape, signal-to-noise), the reviewer's score and whether it
    reached the shortlist. A review that failed counts as not shortlisted, as in the calibration (docs/vetting.md
    §14, Procedure)."""
    from cosmos_search.dayone.vet import INTERESTING, S_SHORTLIST

    rows = []
    for v in verdicts:
        k = keys[v.packet_id]
        t = k.get("truth") or {}
        s = sum(v.class_probs.get(c, 0.0) for c in INTERESTING) if v.ok else float("nan")
        rows.append({
            "source_id": int(k["source_id"]), "flavour": t.get("flavour"), "shape": t.get("shape"),
            "snr": t.get("snr"), "strength": k.get("strength"), "by": k.get("flagged_by"), "g": k.get("g"),
            "reviewed": bool(v.ok), "score": s, "shortlisted": bool(v.ok and s >= S_SHORTLIST),
            "top_class": v.top() if v.ok else None, "p_unexplained": v.p_unexplained if v.ok else float("nan"),
            "recognised_as": v.recognised_as, "eb_explained": (k.get("eclipse") or {}).get("eb_explained"),
            "review_error": v.error if not v.ok else "", "packet_id": v.packet_id})
    return pd.DataFrame(rows)


def fakes_result(ft):
    """The batch's planted fades against the calibration (docs/vetting.md §16): the batch is consistent with it
    when the one-sided 95% upper bound of the share shortlisted reaches RECALL_FLOOR, the lower bound of the
    calibration's 44 of 50."""
    from cosmos_search.vetting import power

    n, k = len(ft), int(ft["shortlisted"].sum())
    f = lambda x: float(np.asarray(x).ravel()[0])
    upper = f(power.cp_upper(k, n))
    flag = ft["by"].fillna("none").astype(str) if "by" in ft else pd.Series("none", index=ft.index)
    return {"planted": n, "shortlisted": k, "recall": k / n, "lower_95_one_sided": f(power.cp_lower(k, n)),
            "upper_95_one_sided": upper,
            "calibration": {"shortlisted": CALIBRATION_RECALL[0], "planted": CALIBRATION_RECALL[1],
                            "lower_95_one_sided": RECALL_FLOOR},
            "consistent_with_calibration": bool(upper >= RECALL_FLOOR),
            "reviews_failed": int((~ft["reviewed"]).sum()),
            "explained_by_the_eclipsing_binary_check": int(ft["eb_explained"].eq(True).sum()),
            "by_flavour": {fl: {"planted": len(g), "shortlisted": int(g["shortlisted"].sum())}
                           for fl, g in ft.groupby("flavour")},
            # by what flagged the fake (episode, trend, repeat): reported, no part of the check
            "by_flag": {b: {"planted": len(g), "shortlisted": int(g["shortlisted"].sum())}
                        for b, g in ft.groupby(flag)}}


def cmd_table(a):
    from cosmos_search.vetting.commit import load_verdicts, reveal

    d = Path(a.dir)
    sealed = d / "key.json.age"
    keys = reveal(d / "batch", sealed if sealed.exists() else d / "key.json",
                  remote=None if a.unpublished else "origin")
    verdicts = load_verdicts(d / "batch")
    fake = [v for v in verdicts if keys[v.packet_id].get("kind") == "fake"]
    df = reviewed([v for v in verdicts if keys[v.packet_id].get("kind") != "fake"], keys,
                  Path(a.checks) / "checks.csv" if a.checks else None)
    df.to_csv(d / "reviewed.csv", index=False)
    if len(df):
        listed = df["shortlisted"]
        n_failed, n_eb = int((~df["reviewed"]).sum()), int((listed & df["eb_explained"].eq(True)).sum())
        print(f"{len(df)} stars, {len(df) - n_failed} with a review that counts: {int(listed.sum())} on the "
              f"shortlist ({n_failed} of them because the review failed or does not count), {n_eb} of the shortlist "
              f"explained by an eclipsing binary's period; table at {d / 'reviewed.csv'}")
    rec = json.loads((d / "files.json").read_text()) if (d / "files.json").exists() else {}
    if not rec.get("calibrated_format", True):
        print("These files carried tests the calibration's files did not "
              f"({', '.join(sorted(set(rec.get('tests_in_the_files', [])) - set(vet.CALIBRATED_TESTS)))}): the "
              "calibration's recall and false-shortlist rate (docs/vetting.md §14) do not describe these reviews.")
    if fake:
        ft = fakes_reviewed(fake, keys)
        ft.to_csv(d / "fakes_reviewed.csv", index=False)
        res = fakes_result(ft)
        (d / "fakes.json").write_text(json.dumps(res, indent=1))
        print(f"Planted fades: {res['shortlisted']} of {res['planted']} shortlisted (one-sided 95% bounds "
              f"{res['lower_95_one_sided']:.0%} and {res['upper_95_one_sided']:.0%}; the calibration's 44 of 50 has "
              f"a lower bound of {RECALL_FLOOR:.0%}).")
        if res["consistent_with_calibration"]:
            print("Consistent with the calibration: its rates may be quoted for this batch (docs/vetting.md §16).")
        else:
            print("NOT consistent with the calibration: do not set stars aside on this batch's verdicts or quote "
                  "the calibration's rates for it; read the list in the checks' order with the verdicts as a "
                  "column, and calibrate on this release's stars (docs/vetting.md §16).")
    else:
        print("This batch held no planted fades: nothing in it says whether its reviewer finds what the "
              "calibration's reviewer found (docs/vetting.md §16).")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("files")
    s.add_argument("--checks", required=True, help="output directory of slow_checks.py")
    s.add_argument("--fading", required=True, help="the fading list of slow_search.py flag (after cuts)")
    s.add_argument("--lightcurves", required=True, nargs="+",
                   help="the run's kept light curves; more directories (a DataLink pull) for stars not kept")
    s.add_argument("--rule", default=str(rule_v3.RULE_DIR), help="rule directory (its drift table)")
    s.add_argument("--out", required=True)
    s.add_argument("--schema", default="gaiadr4")
    s.add_argument("--n", type=int, help="only the strongest N checked stars")
    s.add_argument("--gaia", help="gaia_source values of the stars as a Parquet file, instead of the archive")
    s.add_argument("--no-context", action="store_true", help="no AllWISE and 2MASS magnitudes (practice, offline)")
    s.add_argument("--with-eclipse", action="store_true",
                   help="put the eclipsing-binary check into the files (not what the calibration's files held)")
    s.add_argument("--fakes", type=int,
                   help=f"planted fades with known answers (default {FAKE_FLOOR}, or {FAKE_SHARE:.0%}% of the stars' "
                        "files where that is more); 0 for a practice batch without them")
    s.add_argument("--hosts", help="light-curve files that hold every star of their sky (the thresholds' sample "
                                   "S/lightcurves): the quiet stars the planted fades are put into")
    s.add_argument("--run", help="the pass output whose summary/ holds the real stars a fake is scored against "
                                 "(default: the directory above the first --lightcurves)")
    s.add_argument("--lists", help="the flag output with flags.parquet and reference.parquet (default: the "
                                   "directory of --fading)")
    s.add_argument("--no-marks", action="store_true", help="as the pass's --no-marks (read from the run's pass.json)")
    s.add_argument("--no-own-test", action="store_true",
                   help="as the pass's --no-own-test (read from the run's pass.json)")
    s.add_argument("--practice-seed", type=int,
                   help="practice and tests only: a repeatable draw of hosts and order of files (a seed on a "
                        "command line is not secret)")
    s.add_argument("--keep-fetch-errors", action="store_true",
                   help="also stars whose NEOWISE or ZTF fetch failed in the checks")
    s.add_argument("--overwrite", action="store_true",
                   help="replace the files in --out; their prompts and replies move to replaced-<k>/")
    s.set_defaults(fn=cmd_files)
    s = sub.add_parser("launch")
    s.add_argument("--dir", required=True)
    s.add_argument("--chunks", type=int, default=9)
    s.set_defaults(fn=cmd_launch)
    s = sub.add_parser("table")
    s.add_argument("--dir", required=True)
    s.add_argument("--checks", help="output directory of slow_checks.py: adds its class and colour columns")
    s.add_argument("--unpublished", action="store_true", help="practice only: skip the check that verdicts are on origin")
    s.set_defaults(fn=cmd_table)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
