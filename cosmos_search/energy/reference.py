"""Reference photometry for the photosphere model and the calibration: Gaia DR3 joined to 2MASS and AllWISE.

Everything comes from the Gaia archive in one ADQL join (DPAC's best-neighbour tables and the copies of the 2MASS
PSC and AllWISE catalogues it hosts), so no HATS copy is involved: the data.lsdb.io AllWISE copy is shifted by one
column from `use_src` on (docs/infra/xmatch.md), which is harmless for the w?mpro columns but not worth the risk.

    python -m cosmos_search.energy reference --out ref/      # general.parquet, bright.parquet, yso.parquet, ecl.parquet

Samples:
- general: a random_index slice of stars with a 2MASS and an AllWISE best neighbour and parallax_over_error > 5.
  Photosphere loci for K-W1 and W1-W2, injection hosts, and the false-excess rate.
- mid: G 9.5-12. W1 is unsaturated there (W1 > 8.2) and a photosphere is still well detected in W3, so the
  W1-W3 locus comes from here.
- bright: G < 9.5. Only these have a photosphere detected in W4 (with W3 unsaturated, W3 > 3.8), so the W3-W4
  locus comes from here.
- yso, ecl: DPAC's YSO and eclipsing-binary classes (vari_classifier_result). Real dust dimmers should show an
  excess; eclipsing binaries are dimmers without dust and should not.
"""

from pathlib import Path

import numpy as np

import pandas as pd

COLUMNS = """g.source_id, g.ra, g.dec, g.l, g.b, g.phot_g_mean_mag, g.bp_rp, g.parallax, g.parallax_error, g.ruwe,
 g.phot_g_mean_flux_over_error, g.phot_g_n_obs,
 a.w1mpro, a.w1mpro_error, a.w2mpro, a.w2mpro_error, a.w3mpro, a.w3mpro_error, a.w4mpro, a.w4mpro_error,
 a.ph_qual AS w_ph_qual, a.cc_flags AS w_cc_flags, a.ext_flag AS w_ext_flag, a.var_flag AS w_var_flag,
 ab.angular_distance AS w_sep, ab.number_of_mates AS w_mates,
 t.j_m, t.j_msigcom, t.h_m, t.h_msigcom, t.ks_m, t.ks_msigcom, t.ph_qual AS t_ph_qual,
 tb.number_of_mates AS t_mates"""

JOINS = """FROM gaiadr3.gaia_source AS g
JOIN gaiadr3.allwise_best_neighbour AS ab ON ab.source_id = g.source_id
JOIN gaiadr1.allwise_original_valid AS a ON a.allwise_oid = ab.allwise_oid
JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS tb ON tb.source_id = g.source_id
JOIN gaiadr3.tmass_psc_xsc_join AS tj ON tj.clean_tmass_psc_xsc_oid = tb.clean_tmass_psc_xsc_oid
JOIN gaiadr1.tmass_original_valid AS t ON t.designation = tj.original_psc_source_id"""

SAMPLES = {
    "general": f"SELECT {COLUMNS} {JOINS} WHERE g.random_index < 1500000 AND g.parallax_over_error > 5",
    "mid": f"SELECT {COLUMNS} {JOINS} WHERE g.random_index < 60000000 AND g.phot_g_mean_mag BETWEEN 9.5 AND 12 "
           "AND g.parallax_over_error > 5",
    # Fresh stars for the half-C check (docs/arms/energy-budget.md §8): disjoint random_index slices.
    "general_c": f"SELECT {COLUMNS} {JOINS} WHERE g.random_index >= 1500000 AND g.random_index < 2500000 "
                 "AND g.parallax_over_error > 5",
    "mid_c": f"SELECT {COLUMNS} {JOINS} WHERE g.random_index >= 60000000 AND g.random_index < 80000000 "
             "AND g.phot_g_mean_mag BETWEEN 9.5 AND 12 AND g.parallax_over_error > 5",
    "bright": f"SELECT {COLUMNS} {JOINS} WHERE g.random_index < 600000000 AND g.phot_g_mean_mag < 9.5 "
              "AND g.parallax_over_error > 5",
    "yso": f"SELECT {COLUMNS} {JOINS} JOIN gaiadr3.vari_classifier_result AS v ON v.source_id = g.source_id "
           "WHERE v.best_class_name = 'YSO' AND g.random_index < 900000000",
    "ecl": f"SELECT {COLUMNS} {JOINS} JOIN gaiadr3.vari_classifier_result AS v ON v.source_id = g.source_id "
           "WHERE v.best_class_name = 'ECL' AND g.random_index < 30000000",
}


def fetch(out, samples=SAMPLES, tap=None, log=print):
    """Run each sample's query and write <name>.parquet under out (cached by the shared TAP client)."""
    from cosmos_search.gaia.archive import tap as gaia_tap

    tap = tap or gaia_tap()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, adql in samples.items():
        df = tap.query(adql).to_pandas()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: v.decode() if isinstance(v, bytes) else v)
        paths[name] = out / f"{name}.parquet"
        df.to_parquet(paths[name])
        log(f"{name}: {len(df)} rows -> {paths[name]}")
    return paths


def fetch_ids(source_ids, tap=None):
    """The same photometry for a list of Gaia DR3 source_ids (TAP upload). Stars without a 2MASS or AllWISE best
    neighbour come back with those columns empty (LEFT JOINs), so a missing match is visible as such."""
    from astropy.table import Table

    from cosmos_search.gaia.archive import tap as gaia_tap

    tap = tap or gaia_tap()
    ids = Table({"source_id": np.asarray(source_ids, np.int64)})
    left = JOINS.replace("JOIN gaiadr3.allwise", "LEFT JOIN gaiadr3.allwise").replace(
        "JOIN gaiadr1.allwise", "LEFT JOIN gaiadr1.allwise").replace("JOIN gaiadr3.tmass", "LEFT JOIN gaiadr3.tmass").replace(
        "JOIN gaiadr1.tmass", "LEFT JOIN gaiadr1.tmass")
    adql = (f"SELECT {COLUMNS}, g.matched_transits FROM TAP_UPLOAD.ids AS u JOIN gaiadr3.gaia_source AS g "
            f"ON g.source_id = u.source_id {left.split(chr(10), 1)[1]}")
    df = tap.query(adql, uploads={"ids": ids}).to_pandas()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode() if isinstance(v, bytes) else v)
    return df
