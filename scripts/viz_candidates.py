"""Release-day figures for candidates, and the DR3 gallery that shows what each figure looks like.

    # once per run: the comparison sample with DPAC classes, and a reference sample for the HR diagram (TAP)
    python scripts/viz_candidates.py reference --run RUN --out REF [--release dr4]

    # figures and a page for chosen stars of a run (light curves from --lightcurves, else the AIP mirror for DR3)
    python scripts/viz_candidates.py figures --run RUN --ref REF --ids 123 456 --out OUT

    # the gallery: real DR3 stars of known kinds, injected fakes, a short dip and the astrometry figures
    python scripts/viz_candidates.py gallery --run RUN --ref REF --out OUT

RUN is a slow-change run directory laid out like /mnt/project-files/slow-changes/trial-dr3/ (lists/, checks/,
rule-*/ with release.json, drift_table.parquet and stats.parquet). Figures land in OUT as <id>-<figure>-<theme>.png and
.svg, and OUT/index.html shows them all in cora.earth's style. See docs/viz.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cosmos_search.slow import detect, drift
from cosmos_search.viz import data, html, sheet, style

TABLES = {"dr3": "gaiadr3", "dr4": "gaiadr4"}
RELEASE_END = {"dr3": ("Gaia DR3", 902.0), "dr4": ("Gaia DR4", 1870.0)}
TITLE = "Gaia release-day figures"
POP_COLUMNS = ["source_id", "n_visits", "sigma_visit", "abbe", "slope", "slope_z", "s_fade", "s_rise", "s_rise_ns",
               "s_rep", "s_rep_bright", "fade_kind", "dim_depth", "dim_t0", "dim_t1", "depth_bp", "depth_bp_err",
               "depth_rp", "depth_rp_err", "gbin"]


def rule_dir(run):
    dirs = sorted(Path(run).glob("rule-*"))
    if not dirs:
        raise SystemExit(f"no rule-* directory in {run}")
    return dirs[0]


def tap_upload(tap, ids, adql, chunk=50000):
    """Run an upload query over ids in chunks (the ESA archive refuses very large uploads)."""
    from astropy.table import Table
    parts = []
    ids = np.asarray(ids, np.int64)
    for i in range(0, len(ids), chunk):
        up = {"ids": Table({"source_id": ids[i:i + chunk]})}
        parts.append(tap.query(adql, uploads=up).to_pandas())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def cmd_reference(a):
    from cosmos_search.gaia import archive
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    schema = TABLES[a.release]
    rd = rule_dir(a.run)
    stats = pd.read_parquet(rd / "stats.parquet")
    stats = stats[[c for c in POP_COLUMNS if c in stats]]
    stats = stats.sample(min(a.sample, len(stats)), random_state=a.seed).reset_index(drop=True)
    tap = archive.tap()
    src = tap_upload(tap, stats["source_id"], "SELECT u.source_id, s.phot_g_mean_mag, s.bp_rp FROM TAP_UPLOAD.ids AS u "
                     f"JOIN {schema}.gaia_source AS s ON s.source_id = u.source_id")
    cls = tap_upload(tap, stats["source_id"], "SELECT u.source_id, c.best_class_name FROM TAP_UPLOAD.ids AS u "
                     f"JOIN {schema}.vari_classifier_result AS c ON c.source_id = u.source_id")
    cls["best_class_name"] = cls["best_class_name"].astype(str)
    pop = stats.merge(src, on="source_id", how="left").merge(cls, on="source_id", how="left")
    pop = pd.concat([pop, data.strengths(pop, data.thresholds(rd / "release.json"))], axis=1)
    pop.to_parquet(out / "pop.parquet")
    print(f"comparison sample: {len(pop):,} stars, {pop['best_class_name'].notna().sum():,} with a DPAC class")
    hr = tap.query("SELECT TOP 400000 source_id, phot_g_mean_mag, bp_rp, parallax, parallax_error "
                   f"FROM {schema}.gaia_source WHERE random_index < 40000000 AND parallax_over_error > 10 "
                   "AND bp_rp IS NOT NULL").to_pandas()
    hr.to_parquet(out / "hr.parquet")
    print(f"HR reference: {len(hr):,} stars")


def load_context(run, ref, release):
    pop = pd.read_parquet(Path(ref) / "pop.parquet")
    hr_path = Path(ref) / "hr.parquet"
    hr = pd.read_parquet(hr_path) if hr_path.exists() else None
    lists = {}
    run = Path(run)
    summary = json.loads((run / "lists" / "summary.json").read_text())
    fad = pd.read_parquet(run / "lists" / "fading.parquet", columns=["strength", "phot_g_mean_mag"])
    ris = pd.read_parquet(run / "lists" / "rising.parquet", columns=["rise_strength", "phot_g_mean_mag"])
    for grp, d in summary["by_magnitude"].items():
        f = fad.loc[data.group(fad["phot_g_mean_mag"]) == grp, "strength"].to_numpy()
        r = ris.loc[data.group(ris["phot_g_mean_mag"]) == grp, "rise_strength"].to_numpy()
        lists[grp] = (f, r, int(d["scored"]))
    return sheet.Context(pop=pop, lists=lists, hr=hr, release=RELEASE_END[release], label=release.upper())


def list_rows(run, ids):
    """Rows of the fading list (else the rising list) for ids, keyed by source id."""
    out = {}
    for name in ("fading", "rising", "early"):
        p = Path(run) / "lists" / f"{name}.parquet"
        if not p.exists():
            continue
        t = pd.read_parquet(p, filters=[("id", "in", [int(i) for i in ids])])
        for _, r in t.iterrows():
            out.setdefault(int(r["id"]), r)
    return out


def lightcurves(ids, path=None):
    if path:
        p = Path(path)
        files = sorted(p.glob("*.parquet")) if p.is_dir() else [p]
        tabs = [pq.read_table(f, filters=[("source_id", "in", [int(i) for i in ids])]) for f in files]
        return pa.concat_tables(tabs)
    from cosmos_search.gaia import epoch
    return epoch.mirror_epoch_photometry(source_ids=[int(i) for i in ids])


def star_inputs(run, rows, lc):
    """Per star: drift-corrected epochs, ZTF + NEOWISE, the checks row, from the run's files."""
    rd = rule_dir(run)
    table = pd.read_parquet(rd / "drift_table.parquet")
    aux = pd.DataFrame([{"source_id": sid, "phot_g_mean_mag": r["phot_g_mean_mag"], "bp_rp": r["bp_rp"]}
                        for sid, r in rows.items()])
    ep = data.epochs(lc, drift.star_cells(aux), table) if len(aux) else pd.DataFrame()
    chk_dir = Path(run) / "checks"
    checks = pd.read_csv(chk_dir / "checks.csv") if (chk_dir / "checks.csv").exists() else pd.DataFrame()
    ztf = pd.read_parquet(chk_dir / "ztf.parquet") if (chk_dir / "ztf.parquet").exists() else pd.DataFrame()
    neo = pd.read_parquet(chk_dir / "neowise.parquet") if (chk_dir / "neowise.parquet").exists() else pd.DataFrame()
    from cosmos_search.slow import vet
    out = {}
    for sid in rows:
        c = checks[checks["source_id"] == sid] if len(checks) else checks
        z = ztf[ztf["source_id"] == sid] if len(ztf) else ztf
        n = neo[neo["source_id"] == sid] if len(neo) else neo
        nv = vet.neowise_visits(n) if len(n) else None
        out[sid] = {"epochs": ep[ep["source_id"] == sid] if len(ep) else ep,
                    "others": data.others(z, nv), "checks": c.iloc[0].to_dict() if len(c) else None}
    return out


def context_rows(ids, release):
    """gaia_source parallax and DPAC class for a few stars (TAP)."""
    from cosmos_search.gaia import archive
    schema = TABLES[release]
    tap = archive.tap()
    src = tap_upload(tap, ids, "SELECT u.source_id, s.parallax, s.parallax_error, s.ruwe FROM TAP_UPLOAD.ids AS u "
                     f"JOIN {schema}.gaia_source AS s ON s.source_id = u.source_id")
    cls = tap_upload(tap, ids, "SELECT u.source_id, c.best_class_name FROM TAP_UPLOAD.ids AS u "
                     f"JOIN {schema}.vari_classifier_result AS c ON c.source_id = u.source_id")
    cls["best_class_name"] = cls["best_class_name"].astype(str)
    return src.merge(cls, on="source_id", how="left").set_index("source_id")


def star_section(ctx, sid, row, inp, ctxrow, out, heading=None, text="", name=None, label="this star"):
    cls = ctxrow.get("best_class_name") if ctxrow is not None else None
    cls = cls if isinstance(cls, str) and cls not in ("nan", "None") else None
    figs = sheet.slow_candidate(ctx, row, inp["epochs"], out, others=inp["others"], checks=inp["checks"],
                                astro=ctxrow, cls=cls, name=name, label=label, sid=sid)
    return {"heading": heading or (name or f"Gaia {ctx.label} {sid}"), "anchor": f"s{sid}", "text": text,
            "figures": figs}


def cmd_figures(a):
    ctx = load_context(a.run, a.ref, a.release)
    ids = [int(i) for i in a.ids]
    rows = list_rows(a.run, ids)
    missing = sorted(set(ids) - set(rows))
    if missing:
        print("not in the lists, skipped:", missing)
    lc = lightcurves(list(rows), a.lightcurves)
    inputs = star_inputs(a.run, rows, lc)
    crow = context_rows(list(rows), a.release)
    sections = [star_section(ctx, sid, rows[sid], inputs[sid], crow.loc[sid].to_dict() if sid in crow.index else None,
                             a.out) for sid in rows]
    for name, fragment in (("index.html", False), ("artifact.html", True)) if a.artifact else (("index.html", False),):
        p = html.page(a.title, sections, Path(a.out) / name, meta=f"Gaia {a.release.upper()} slow-change search",
                      png_links=not fragment, fragment=fragment)
        print(p)


# --- gallery ---------------------------------------------------------------------------------------------------

GALLERY = [  # (source id, heading, text): real DR3 stars of known kinds from the DR3 trial's fading list
    (5847323477883945472, "A spotted star (RS CVn type)",
     "DPAC calls this a spotted star: starspots that grow over months make it fade. It is near the top of the DR3 "
     "list at G < 17 among stars that are not eclipsing binaries. The change is slow and smooth, and the BP and "
     "RP panels show it in both of Gaia's colours."),
    (5885363968852633856, "An eclipsing binary",
     "The commonest star at the top of the fading list: at G < 19, 82% of the strongest fades in DR3 are "
     "eclipsing binaries. Gaia catches a few visits inside eclipses, which read as a dimming episode; the "
     "visits jump about (a high Abbe value) where a real slow fade would be smooth."),
    (1340986169606559488, "A quasar",
     "A quasar's brightness wanders by tens of percent over years. At G ≥ 19, 93% of the top of the DR3 list have "
     "no measurable parallax or proper motion, nearly all quasars, so the search sets them aside to read "
     "separately."),
    (511239410478024192, "A Be star",
     "A hot star that throws off a disc of gas. The disc adds light while it lasts, so its fading away reads as "
     "a slow fade."),
    (424592431008600704, "A long-period variable",
     "A cool giant that pulsates over hundreds of days and changes by magnitudes, sampled by Gaia at a few "
     "dozen points."),
    (1807455283392608768, "An R CrB star",
     "R Coronae Borealis stars puff out clouds of carbon dust that hide them for months; the infrared shows "
     "the warm dust."),
]

FAKES = [  # (key, heading, text, signal factory(t_mid, t_lo, t_hi), host G range)
    ("fake-grey-fade", "Injected: a steady grey fade of 1% a year",
     "Not a real star. A quiet G ≈ 15 star with a fade of 1% a year added to every band alike, the way an opaque "
     "body slowly covering more of it would look. This is the kind of change the search is built to find; it "
     "shows what such a find would look like in each figure.",
     lambda tm, lo, hi: __import__("cosmos_search.inject.lightcurve", fromlist=["x"]).SecularTrend(-0.01, tm)),
    ("fake-grey-dip", "Injected: a grey dimming episode of 2% for 300 days",
     "Not a real star. A quiet star with a long grey dip added: 2% fainter for about 300 days in every band, "
     "with 60-day ingress and egress.",
     lambda tm, lo, hi: __import__("cosmos_search.inject.lightcurve", fromlist=["x"]).Dip(
         t0=lo + 0.35 * (hi - lo), covering=0.02, ingress=60.0, egress=60.0, flat=180.0)),
]


def fake_row(sid, epochs, g, bp_rp, thr):
    """The detector's row for an injected light curve, with strength and 'by' as the lists give them."""
    e = epochs.rename(columns={"frac": "frac_corr"})
    row = detect.star_features(e).iloc[0]
    st = data.strengths(pd.DataFrame([{**row.to_dict(), "phot_g_mean_mag": g}]), thr).iloc[0]
    grp = str(data.group([g])[0])
    by = "episode" if row["s_fade"] / thr[grp]["s"] >= -row["slope_z"] / thr[grp]["z"] else "trend"
    return pd.Series({**row.to_dict(), "id": sid, "phot_g_mean_mag": g, "bp_rp": bp_rp, "by": by,
                      "strength": float(st["fade_strength"])})


def quiet_host(pop, g=(14.8, 15.3), seed=3):
    """A star to inject fakes into: no DPAC class, well below the thresholds, scatter no more than typical."""
    q = pop[(pop["fade_strength"] < 0.3) & (pop["rise_strength"] < 0.3) & (pop["n_visits"] >= 26)
            & pop["phot_g_mean_mag"].between(*g) & pop["bp_rp"].between(0.8, 1.2) & pop["best_class_name"].isna()]
    q = q[q["sigma_visit"] <= q["sigma_visit"].median()]
    return q.sample(1, random_state=seed).iloc[0]


def cmd_gallery(a):
    from cosmos_search.inject import lightcurve as inj
    out = Path(a.out)
    ctx = load_context(a.run, a.ref, "dr3")
    thr = data.thresholds(rule_dir(a.run) / "release.json")
    ids = [g[0] for g in GALLERY]
    rows = list_rows(a.run, ids)
    host = quiet_host(ctx.pop)
    host_id = int(host["source_id"])
    lc = lightcurves(ids + [host_id], a.lightcurves)
    inputs = star_inputs(a.run, rows, lc)
    crow = context_rows(ids + [host_id], "dr3")
    sections = []
    for sid, heading, text in GALLERY:
        if sid not in rows:
            print("missing from the lists:", sid)
            continue
        cr = crow.loc[sid].to_dict() if sid in crow.index else None
        sections.append(star_section(ctx, sid, rows[sid], inputs[sid], cr, out, heading=heading,
                                     text=f"<p>{text}</p>"))
        print("done", sid, flush=True)

    # injected fakes in a quiet host
    rd = rule_dir(a.run)
    table = pd.read_parquet(rd / "drift_table.parquet")
    hlc = lc.filter(pc.equal(lc["source_id"], host_id)).to_pandas()
    for k, (key, heading, text, make) in enumerate(FAKES):
        g = hlc["band"].astype(str) == "G"
        t = hlc["t"].to_numpy(float)
        sig = make(float(np.median(t[g])), float(t[g].min()), float(t[g].max()))
        f = hlc.copy()
        for b in ("G", "BP", "RP"):
            m = (f["band"].astype(str) == b).to_numpy()
            f.loc[m, "flux"], f.loc[m, "err"] = inj.inject(t[m], f.loc[m, "flux"], f.loc[m, "err"], b, sig)
        fake_id = 9000 + k
        f["source_id"] = fake_id
        aux = pd.DataFrame([{"source_id": fake_id, "phot_g_mean_mag": host["phot_g_mean_mag"], "bp_rp": host["bp_rp"]}])
        ep = data.epochs(pa.Table.from_pandas(f, preserve_index=False), drift.star_cells(aux), table)
        row = fake_row(fake_id, ep, float(host["phot_g_mean_mag"]), float(host["bp_rp"]), thr)
        cr = crow.loc[host_id].to_dict() if host_id in crow.index else None
        if cr:
            cr = {**cr, "best_class_name": None}
        figs = sheet.slow_candidate(ctx, row, ep, out, astro=cr, name=f"injected into Gaia DR3 {host_id}",
                                    label="the fake")
        sections.append({"heading": heading, "anchor": key, "text": f"<p>{text}</p>", "figures": figs})
        print("done", key, flush=True)

    sections += gallery_extras(a, ctx, out)
    intro = ("<p>Figures for release day, drawn for real Gaia DR3 stars of known kinds so that we know what each "
             "figure normally looks like before DR4 comes out on 2 December 2026. Every figure compares one star "
             "(orange) with ordinary stars (blue) and with the classes Gaia's variability team (DPAC) assigns. "
             "The comparison stars are a random 300,000 from the DR3 trial run's 1,000 random patches of sky. "
             "DR3 published light curves only for stars DPAC flagged as variable, so here the ordinary stars are "
             "those; in DR4 every star has a light curve.</p>"
             "<p>Two injected fakes show what the search is built to find. Hover over points for their values. "
             "Code: <code>cosmos_search/viz</code>; this page: <code>scripts/viz_candidates.py gallery</code>; "
             "how to use it on the day: <code>docs/viz.md</code>.</p>")
    for name, fragment in (("index.html", False), ("artifact.html", True)) if a.artifact else (("index.html", False),):
        p = html.page(TITLE, sections, out / name, intro=intro, meta="Gaia DR3 examples · 2026-10-30",
                      png_links=a.png_links and not fragment, fragment=fragment)
        print(p)


def gallery_extras(a, ctx, out):
    """A short dip on a real eclipsing binary, and the astrometry figures on DR4 pre-release data."""
    from cosmos_search.astrometry import fit, physics
    from cosmos_search.viz import astrometry as vast
    sections = []
    if a.features:
        feats = pd.concat([pd.read_parquet(f, columns=["source_id", "min_frac", "t_min"])
                           for f in sorted(Path(a.features).glob("part-*.parquet"))], ignore_index=True)
        pop = ctx.pop.merge(feats, on="source_id", how="inner")
        dctx = sheet.Context(pop=pop, label="DR3", release=ctx.release)
        sid = a.dip_id
        f = pop[pop["source_id"] == sid]
        if not len(f):
            f = feats[feats["source_id"] == sid].assign(phot_g_mean_mag=np.nan)
        lc = lightcurves([sid], a.lightcurves)
        if len(f) and lc.num_rows:
            r = f.iloc[0]
            if not np.isfinite(r.get("phot_g_mean_mag", np.nan)):
                aux = pd.read_parquet(a.aux, filters=[("source_id", "=", sid)])
                r = pd.Series({**r.to_dict(), **aux.iloc[0].to_dict()})
            ep = data.epochs(lc)
            figs = sheet.short_dip_candidate(dctx, r, ep, float(r["t_min"]), out, sid=sid)
            sections.append({"heading": "A short dip: one visit inside an eclipse", "anchor": "dip",
                             "text": "<p>The short-dip search looks for dips caught in one or two of Gaia's visits. "
                                     "This one is an eclipsing binary seen in eclipse; most such dips in DR3 "
                                     "are.</p>", "figures": figs})
    if a.prerelease:
        pre = pd.read_csv(a.prerelease)
        g = pre[pre["source_id"] == 4318465066420528000]
        k = max(1.0, np.sqrt(g.chi2_intra.sum() / g.dof_intra.sum()))
        src = fit.Source(g.t, g.psi, g.f_al, g.w, np.sqrt((g.sigma_w * k) ** 2 + 0.02 ** 2))
        k1 = src.kepler()
        fa = physics.mass_function(k1.params["a0_mas"], k1.x[2], k1.params["period_yr"])
        figs = []
        paths = style.render(vast.along_scan, out / "bh3-alongscan", src, k1, name="Gaia BH3 (DR4 pre-release)")
        figs.append({"key": "alongscan", "title": "Positions against a single star's path", "paths": paths,
                     "caption": "Gaia BH3's measured positions along each scan after removing its parallax and "
                                "proper motion, from the 12 DR4 epoch-astrometry sources ESA released early. A single "
                                "star would scatter about zero; a star orbiting something unseen swings by "
                                "the orbit (orange, the fitted orbit at each transit)."})
        if a.orbits:
            orb = pd.read_parquet(a.orbits, columns=["source_id", "period", "f_A", "m1"])
            paths = style.render(vast.kepler_bound, out / "bh3-kepler", orb,
                                 {"period": k1.params["period_yr"] * 365.25, "f_A": fa, "label": "Gaia BH3"})
            figs.append({"key": "kepler", "title": "Does the orbit need more mass than we can see?", "paths": paths,
                         "caption": "Every Gaia DR3 orbit by period and mass function, the least mass the unseen "
                                    "companion can have. Dots need a companion heavier than the star itself that "
                                    "gives no light; Gaia BH1 and BH2 are black holes found this way. The dots above "
                                    "100 solar masses are faulty fits, most at periods under 100 days near harmonics "
                                    "of the 63-day cycle of Gaia's scanning. Orange: BH3 from the DR4 pre-release "
                                    "data, fitted here."})
        sections.append({"heading": "Astrometry: Gaia BH3", "anchor": "astrometry",
                         "text": "<p>The astrometric search looks for stars whose positions no ordinary model "
                                 "fits. BH3, a 33-solar-mass black hole, is the clearest known example of a "
                                 "position track that differs from a single star's.</p>", "figures": figs})
    return sections


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--run", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--release", default="dr3", choices=sorted(TABLES))
    r.add_argument("--sample", type=int, default=300000)
    r.add_argument("--seed", type=int, default=1)
    f = sub.add_parser("figures")
    f.add_argument("--run", required=True)
    f.add_argument("--ref", required=True)
    f.add_argument("--ids", nargs="+", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--release", default="dr3", choices=sorted(TABLES))
    f.add_argument("--lightcurves", help="parquet file or directory of light-curve tables (gaia.epoch.SCHEMA)")
    f.add_argument("--title", default="Candidates")
    f.add_argument("--artifact", action="store_true", help="also write artifact.html, for publishing on claude.ai")
    g = sub.add_parser("gallery")
    g.add_argument("--run", required=True)
    g.add_argument("--ref", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--lightcurves")
    g.add_argument("--features", help="DR3 features directory, for the short-dip example")
    g.add_argument("--aux", help="aux_gaia_source.parquet, for the short-dip example's magnitude")
    g.add_argument("--dip-id", type=int, default=5885363968852633856)
    g.add_argument("--prerelease", default=str(Path(__file__).parent.parent / "tests" / "fixtures"
                                               / "dr4_prerelease_transits.csv"))
    g.add_argument("--orbits", help="orbits_kb.parquet from scripts/astrometry_dr3_nss.py")
    g.add_argument("--png-links", action="store_true")
    g.add_argument("--artifact", action="store_true", help="also write artifact.html, for publishing on claude.ai")
    a = p.parse_args(argv)
    {"reference": cmd_reference, "figures": cmd_figures, "gallery": cmd_gallery}[a.cmd](a)


if __name__ == "__main__":
    main()
