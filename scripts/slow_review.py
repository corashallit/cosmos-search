"""Blind review of slow-change candidates on release day: from the checks to review files, and from the archived
verdicts to the reviewed list (docs/arms/slow-changes.md, "Release day, end to end", step 10).

    # 1. one review file per star the checks covered
    python scripts/slow_review.py files --checks L/checks --fading L/fading.parquet --lightcurves R/lightcurves \\
        --rule RULE --out D [--schema gaiadr4] [--n 300] [--no-context] [--gaia FILE] [--with-eclipse]
    # 2. the reviews: the Workflow tool with scripts/stage_a_reviews.js and each list of arguments this prints
    #    (never the Agent tool: its subagents are given the thread's recent messages)
    python scripts/slow_review.py launch --dir D [--chunks 9]
    # 3. every transcript checked against the adopted reviewer, then the verdicts archived and pushed
    python scripts/stage_a_audit.py --transcripts <session>/subagents --prompts D/prompts --replies D/replies \\
        --out D/audit.json --reviewer prereg/reviewers/slow-cal-1.json
    python -m cosmos_search.vetting commit --packets D/products.json --replies D/replies --vetter slow-cal-1 \\
        --batch D/batch --archive prereg/verdicts/<name> --audit D/audit.json
    git push
    # 4. the list to read
    python scripts/slow_review.py table --dir D --checks L/checks

`files` builds each star's file through `slow.vet.review_file`, the path the calibration's files took
(docs/vetting.md §14), from what the run already holds: the star's row of the fading list (`slow_search.py flag`
and `cuts`), its epochs in the run's kept light curves with the rule's drift removed, and the NEOWISE and ZTF
epochs the checks fetched (`slow_checks.py`). Two things are fetched here and kept under D/work: the star's
gaia_source values (ESA TAP, by the release's own table names) and its AllWISE and 2MASS magnitudes (the
cross-match of `cosmos_search.xmatch`); `--gaia FILE` and `--no-context` replace or skip them (a practice run
without network). A star whose NEOWISE or ZTF fetch failed in the checks is left out and listed in
D/left_out.csv, so that the checks are run again for it rather than the reviewer reading a file with a hole.

The files hold what the calibration's files held. The eclipsing-binary check of `slow_checks.py --lightcurves`
(`eb_explained`, its period) stays out of them unless `--with-eclipse`: no calibration batch has measured the
reviewer on files that carry it, and on fakes it also explained repeating long dips (docs/arms/slow-changes.md,
"Eclipsing binaries among the candidates"). It is in the key and in the table of step 4 beside each verdict.

D/key.json maps each file's random id to its star. Nothing here is a fake with a known answer, so the key holds
no answers and is not encrypted; `table` still opens it only once the verdicts are committed and, unless
`--unpublished` (practice), on origin.
"""

import argparse
import json
import secrets
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cosmos_search.slow import checks, detect, drift, rule_v3, vet
from cosmos_search.vetting import packet as packet_mod
from cosmos_search.vetting import reviewer
from cosmos_search.vetting.__main__ import pending, prompt

ROOT = Path(__file__).resolve().parents[1]
RELEASE_OF = {"gaiadr3": "dr3", "gaiadr4": "dr4"}
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
    """Drift-corrected epochs of the given stars from a directory of light-curve files (epoch.SCHEMA, the run's
    kept light curves), found by each file's source_id range. stars: source_id, phot_g_mean_mag, bp_rp."""
    cells = drift.star_cells(stars)
    ids = np.sort(stars["source_id"].to_numpy(np.int64))
    out = []
    for f in sorted(Path(lightcurves).glob("*.parquet")):
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
    out = Path(out)
    if not (out / "context.parquet").exists():
        subprocess.run([sys.executable, "-m", "cosmos_search.xmatch", str(gaia_path), "--surveys", "allwise", "2mass",
                        "--context", "--out", str(out)], check=True, cwd=ROOT)
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


def write(d, packets, keys, rng=None):
    """products.json, prompts/, packet_ids.txt, key.json and an empty replies/, in an order unrelated to the
    list's."""
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
    if (d / "products.json").exists() and not a.overwrite:
        raise SystemExit(f"{d} already holds review files; reviews may be under way. Pass --overwrite to replace "
                         "them (their reviews are then void).")
    d.mkdir(parents=True, exist_ok=True)
    stars, left = checked_stars(a.checks, a.fading, a.n, a.keep_fetch_errors)
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
    packets, keys, refused, more = build(stars, epochs, gaia, xm, neowise, ztf, RELEASE_OF.get(a.schema, a.schema),
                                         a.with_eclipse)
    left += more
    write(d, packets, keys)
    pd.DataFrame(left, columns=["source_id", "reason"]).to_csv(d / "left_out.csv", index=False)
    rec = {"files": len(packets), "left_out": len(left), "refused_by_the_leak_scan": refused,
           "checks": str(a.checks), "fading": str(a.fading), "lightcurves": str(a.lightcurves), "rule": str(a.rule),
           "schema": a.schema, "infrared_context": not a.no_context, "with_eclipse": bool(a.with_eclipse),
           "reviewer": reviewer.ADOPTED,
           "tests_in_each_file": sorted(packets[0]["tests"]) if packets else []}
    (d / "files.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
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


def cmd_table(a):
    from cosmos_search.vetting.commit import load_verdicts, reveal

    d = Path(a.dir)
    keys = reveal(d / "batch", d / "key.json", remote=None if a.unpublished else "origin")
    df = reviewed(load_verdicts(d / "batch"), keys, Path(a.checks) / "checks.csv" if a.checks else None)
    df.to_csv(d / "reviewed.csv", index=False)
    listed = df["shortlisted"]
    n_failed, n_eb = int((~df["reviewed"]).sum()), int((listed & df["eb_explained"].eq(True)).sum())
    print(f"{len(df)} stars reviewed: {int(listed.sum())} on the shortlist ({n_failed} of them because the review "
          f"failed), {n_eb} of the shortlist explained by an eclipsing binary's period; table at "
          f"{d / 'reviewed.csv'}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("files")
    s.add_argument("--checks", required=True, help="output directory of slow_checks.py")
    s.add_argument("--fading", required=True, help="the fading list of slow_search.py flag (after cuts)")
    s.add_argument("--lightcurves", required=True, help="the run's kept light curves")
    s.add_argument("--rule", default=str(rule_v3.RULE_DIR), help="rule directory (its drift table)")
    s.add_argument("--out", required=True)
    s.add_argument("--schema", default="gaiadr4")
    s.add_argument("--n", type=int, help="only the strongest N checked stars")
    s.add_argument("--gaia", help="gaia_source values of the stars as a Parquet file, instead of the archive")
    s.add_argument("--no-context", action="store_true", help="no AllWISE and 2MASS magnitudes (practice, offline)")
    s.add_argument("--with-eclipse", action="store_true",
                   help="put the eclipsing-binary check into the files (not what the calibration's files held)")
    s.add_argument("--keep-fetch-errors", action="store_true",
                   help="also stars whose NEOWISE or ZTF fetch failed in the checks")
    s.add_argument("--overwrite", action="store_true")
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
