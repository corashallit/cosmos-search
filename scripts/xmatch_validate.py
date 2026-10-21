"""Live validation of cosmos_search.xmatch on three fields, against Gaia DPAC's own best-neighbour tables.

    python scripts/xmatch_validate.py --out report.json [--surveys 2mass allwise ...] [--fields highlat ...]

For each field: Gaia DR3 sources from HATS, matched to each survey with calibration on; the report holds the
fitted error model, prior by magnitude, false-match accounting and the chi^2 check. For 2MASS and AllWISE it
also compares our best matches with gaiadr3.tmass_psc_xsc_best_neighbour / allwise_best_neighbour.
"""

import argparse
import json
import time

import numpy as np

FIELDS = {
    # name: (ra, dec, radius_deg); Galactic (l, b) in the comment
    "highlat": (180.0, 30.0, 0.30),       # l=196, b=+78: sparse, in eROSITA-DE
    "kepler": (301.5644, 44.4568, 0.12),  # l=80, b=+6.6: around KIC 8462852, moderately crowded
    "plane": (114.066, -24.876, 0.05),    # l=240, b=-2: crowded, in eROSITA-DE and ZTF
}

GAIA_COLS = ["source_id", "ra", "dec", "ref_epoch", "parallax", "pmra", "pmdec", "ra_error", "dec_error",
             "parallax_error", "pmra_error", "pmdec_error", "ra_dec_corr", "ra_parallax_corr", "ra_pmra_corr",
             "ra_pmdec_corr", "dec_parallax_corr", "dec_pmra_corr", "dec_pmdec_corr", "parallax_pmra_corr",
             "parallax_pmdec_corr", "pmra_pmdec_corr", "phot_g_mean_mag", "phot_rp_mean_mag", "bp_rp", "ruwe"]

DPAC = {
    "2mass": ("gaiadr3.tmass_psc_xsc_best_neighbour", "gaiadr3.tmass_psc_xsc_join", "original_psc_source_id"),
    "allwise": ("gaiadr3.allwise_best_neighbour", None, "original_ext_source_id"),
}


def gaia_field(ra, dec, radius):
    import lsdb

    from cosmos_search.xmatch.sources import hats_storage_options

    cat = lsdb.open_catalog("https://data.lsdb.io/hats/gaia_dr3/gaia", columns=GAIA_COLS,
                            storage_options=hats_storage_options())
    return cat.cone_search(ra=ra, dec=dec, radius_arcsec=radius * 3600).compute().reset_index(drop=True)


def dpac_compare(survey, gaia, best):
    """Agreement between our best match (P >= 0.5) and DPAC's best neighbour, per Gaia source."""
    from astropy.table import Table

    from cosmos_search.gaia.archive import tap

    table, _, col = DPAC[survey]
    ids = Table({"source_id": np.asarray(gaia.source_id, dtype=np.int64)})
    if survey == "2mass":
        adql = (f"SELECT b.source_id, b.angular_distance, j.original_psc_source_id AS ext_id "
                f"FROM TAP_UPLOAD.ids AS u JOIN {table} AS b ON b.source_id = u.source_id "
                f"JOIN gaiadr3.tmass_psc_xsc_join AS j ON j.clean_tmass_psc_xsc_oid = b.clean_tmass_psc_xsc_oid")
    else:
        adql = (f"SELECT b.source_id, b.angular_distance, b.original_ext_source_id AS ext_id "
                f"FROM TAP_UPLOAD.ids AS u JOIN {table} AS b ON b.source_id = u.source_id")
    d = tap().query(adql, uploads={"ids": ids}).to_pandas()
    d["ext_id"] = d["ext_id"].astype(str).str.strip()
    ours = best[best.group == 0].set_index("source_id")
    ours_id = ours[f"{survey}_id"].astype(str).str.strip().str.lstrip("J")
    ours_ok = ours.p_best >= 0.5
    dp = d.set_index("source_id")["ext_id"].str.lstrip("J")
    both = ours_ok[ours_ok].index.intersection(dp.index)
    agree = int((ours_id[both] == dp[both]).sum())
    only_ours = ours_ok[ours_ok].index.difference(dp.index)
    only_dpac = dp.index.difference(ours_ok[ours_ok].index)
    return {
        "dpac_matches": int(len(dp)), "ours_p50": int(ours_ok.sum()), "both": int(len(both)),
        "agree": agree, "disagree": int(len(both) - agree), "only_ours": int(len(only_ours)),
        "only_dpac": int(len(only_dpac)),
        "only_dpac_our_p": [float(x) for x in ours.p_best.reindex(only_dpac).fillna(0).to_numpy()[:50]],
        "disagree_ids": [int(i) for i in both[(ours_id[both] != dp[both]).to_numpy()][:20]],
    }


def main():
    from cosmos_search.xmatch import crossmatch

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fields", nargs="*", default=list(FIELDS))
    ap.add_argument("--surveys", nargs="*", default=["2mass", "allwise", "catwise2020", "tic", "erosita_dr2",
                                                      "ztf_dr22", "vsx", "unwise", "neowise"])
    ap.add_argument("--scale-radius", type=float, default=1.0)
    args = ap.parse_args()
    report = {}
    for f in args.fields:
        ra, dec, rad = FIELDS[f]
        t = time.time()
        gaia = gaia_field(ra, dec, rad * args.scale_radius)
        report[f] = {"n_gaia": len(gaia), "gaia_fetch_s": round(time.time() - t, 1), "surveys": {}}
        print(f, len(gaia), "Gaia sources", flush=True)
        for s in args.surveys:
            t = time.time()
            try:
                res = crossmatch(gaia, s)
            except Exception as e:  # keep going: one survey's outage shouldn't stop the others
                report[f]["surveys"][s] = {"error": repr(e)[:500]}
                print(f, s, "ERROR", repr(e)[:300], flush=True)
                continue
            r = dict(res.report)
            r["elapsed_s"] = round(time.time() - t, 1)
            r["n_best_p90"] = int((res.best.p_best >= 0.9).sum())
            r["multiplicity_gt_1.5"] = int((res.matches.survey_multiplicity > 1.5).sum())
            if s in DPAC:
                try:
                    r["dpac"] = dpac_compare(s, gaia, res.best)
                except Exception as e:
                    r["dpac"] = {"error": repr(e)[:500]}
            report[f]["surveys"][s] = r
            th = {x["threshold"]: x for x in r["thresholds"]}[0.9]
            print(f, s, f"{r['elapsed_s']}s scale={res.model.scale:.2f} floor={res.model.floor_mas:.0f}mas "
                  f"acc90={th['accepted']} fmr={th['fmr_shift']:.4f} conf={th['confusion']:.4f} "
                  f"false_shift={th['false_shift']:.1f} false_model={th['false_model']:.1f} "
                  f"chi2={r['chi2_check']['mean']}", r.get("dpac", ""), flush=True)
            with open(args.out, "w") as fh:
                json.dump(report, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
