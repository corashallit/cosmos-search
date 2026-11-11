"""Table and column names per Gaia release. The one place to edit on DR4 release day if the archive's names differ.

DR4 names follow the draft data model of 2026-06-26 (gaia-dr4-prerelease-draft-data-model.pdf; its tables and
columns are in configs/dr4/draft-data-model-2026-06-26.json, and tests/test_releases.py holds every release-day
query to them). The schema name "gaiadr4" is an assumption until release; `check_release` confirms tables and
columns against the archive's TAP_SCHEMA.

What changed from DR3 that matters here:

- `gaia_source` holds only sources with good astrometry and photometry. Every source is in `all_source_astrometry`,
  `all_source_photometry` and `all_source_flags`, so per-star lookups go there.
- BP/RP blend counts are in `all_source_photometry` only; `phot_bp_rp_excess_factor` is not published (it is
  (BP + RP flux) / G flux, computed here); `astrometric_params_solved` is `astrometric_params`.
- `has_epoch_photometry` and the merged variability class (`phot_variable_best_class`) are in `all_source_flags`.
  `vari_classifier_result` has one row per classifier (`classifier_name`, `classifier_best_class_name`,
  `classifier_best_class_prob`).
- `vari_summary` keeps every DR3 G-FoV column used by the short-dip prefilter and adds flux-space moments, per-CCD
  G statistics and G/BP/RP correlations.
- `dr3_neighbourhood` links DR4 source ids to DR3's (`dr3_source_id`, `angular_distance`, `probable_match`).

Per-star lookups name their columns by the DR3 names (`STAR_COLUMNS`), whatever the release calls them:

    rows = fetch_star_rows(tap, ids, "dr4", ("parallax", "parallax_error", "best_class_name"))
"""

import json
import re
from pathlib import Path

DRAFT_DATA_MODEL = Path(__file__).resolve().parents[2] / "configs" / "dr4" / "draft-data-model-2026-06-26.json"

_ASTROMETRY = ("ra", "dec", "ref_epoch", "parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error",
               "parallax_pmra_corr", "parallax_pmdec_corr", "pmra_pmdec_corr", "ruwe", "ipd_frac_multi_peak")
# Position errors and the correlations with position: the cross-match to other surveys propagates them.
_POSITION_ERRORS = ("ra_error", "dec_error", "ra_dec_corr", "ra_parallax_corr", "ra_pmra_corr", "ra_pmdec_corr",
                    "dec_parallax_corr", "dec_pmra_corr", "dec_pmdec_corr")
_PHOTOMETRY = ("phot_g_mean_mag", "bp_rp", "phot_bp_n_obs", "phot_rp_n_obs", "phot_bp_n_blended_transits",
               "phot_rp_n_blended_transits")

# Names a caller can ask for (DR3's gaia_source names, plus best_class_name / best_class_score).
STAR_COLUMNS = (*_ASTROMETRY, *_POSITION_ERRORS, "astrometric_params_solved", *_PHOTOMETRY, "phot_bp_rp_excess_factor",
                "has_epoch_photometry", "best_class_name", "best_class_score")

RELEASES = {
    "dr3": {
        "schema": "gaiadr3",
        "vari_summary": "gaiadr3.vari_summary",
        "gaia_source": "gaiadr3.gaia_source",
        "n_vari_summary": 11_754_237,
        # Per table: the expression for each name, with {t} for the table alias.
        "star_tables": {
            "gaiadr3.gaia_source": {
                **{c: "{t}." + c for c in (*_ASTROMETRY, *_POSITION_ERRORS, "astrometric_params_solved", *_PHOTOMETRY,
                                           "phot_bp_rp_excess_factor", "has_epoch_photometry")}},
            "gaiadr3.vari_classifier_result": {"best_class_name": "{t}.best_class_name",
                                               "best_class_score": "{t}.best_class_score"},
        },
    },
    "dr4": {
        "schema": "gaiadr4",
        "vari_summary": "gaiadr4.vari_summary",
        "gaia_source": "gaiadr4.gaia_source",
        "all_source_photometry": "gaiadr4.all_source_photometry",
        "dr3_neighbourhood": "gaiadr4.dr3_neighbourhood",
        "n_vari_summary": 403_061_676,  # draft data model row count
        "star_tables": {
            "gaiadr4.all_source_astrometry": {
                **{c: "{t}." + c for c in (*_ASTROMETRY, *_POSITION_ERRORS)},
                "astrometric_params_solved": "{t}.astrometric_params"},
            "gaiadr4.all_source_photometry": {
                **{c: "{t}." + c for c in _PHOTOMETRY},
                "phot_bp_rp_excess_factor": "({t}.phot_bp_mean_flux + {t}.phot_rp_mean_flux) / {t}.phot_g_mean_flux"},
            "gaiadr4.all_source_flags": {"has_epoch_photometry": "{t}.has_epoch_photometry",
                                         "best_class_name": "{t}.phot_variable_best_class"},
            # best_class_score: no single score in DR4 (one row per classifier); left empty.
        },
    },
}

# Columns the whole-sky queries need, beyond the per-star lookups: the short-dip prefilter (targets/), the
# archive-side slow-change score kept for a release without bulk files (slow/select.py), and the DR3 link.
REQUIRED_COLUMNS = {
    "dr3": {
        "gaiadr3.vari_summary": ("source_id", "num_selected_g_fov", "min_mag_g_fov", "max_mag_g_fov",
                                 "median_mag_g_fov", "mad_mag_g_fov", "skewness_mag_g_fov", "outlier_median_g_fov"),
        "gaiadr3.gaia_source": ("source_id", "random_index", "phot_g_mean_mag", "bp_rp", "parallax_over_error"),
    },
    "dr4": {
        "gaiadr4.vari_summary": ("source_id", "num_selected_g_fov", "min_mag_g_fov", "max_mag_g_fov",
                                 "median_mag_g_fov", "mad_mag_g_fov", "skewness_mag_g_fov", "outlier_median_g_fov"),
        "gaiadr4.gaia_source": ("source_id", "random_index", "phot_g_mean_mag", "bp_rp", "parallax",
                                "parallax_error", "parallax_over_error"),
        "gaiadr4.all_source_photometry": ("source_id", "random_index_as", "phot_g_mean_mag", "bp_rp", "phot_g_n_cal",
                                          "phot_g_flux_uwv", "phot_g_abbe_value", "phot_bp_abbe_value",
                                          "phot_rp_abbe_value", "phot_g_harmonic_amplitude"),
        "gaiadr4.dr3_neighbourhood": ("source_id", "dr3_source_id", "angular_distance", "probable_match"),
    },
}

_COLUMN = re.compile(r"\{t\}\.([a-z0-9_]+)")


def tables(release):
    try:
        return RELEASES[release]
    except KeyError:
        raise ValueError(f"unknown release {release!r}; known: {sorted(RELEASES)}") from None


def required_columns(release):
    """{table: sorted archive columns} every release-day query needs: REQUIRED_COLUMNS plus the per-star lookups."""
    need = {t: set(cols) for t, cols in REQUIRED_COLUMNS[release].items()}
    for table, exprs in tables(release)["star_tables"].items():
        cols = need.setdefault(table, set())
        cols.add("source_id")
        for e in exprs.values():
            cols.update(_COLUMN.findall(e))
    return {t: sorted(c) for t, c in need.items()}


def missing_columns(release, have):
    """Required columns absent from `have` ({table: column names}; table names with or without the schema, any
    case). A table that is absent altogether is reported as 'schema.table (table)'."""
    have = {k.lower().split(".")[-1]: {c.lower() for c in v} for k, v in have.items()}
    out = []
    for table, cols in required_columns(release).items():
        got = have.get(table.split(".")[-1])
        if got is None:
            out.append(f"{table} (table)")
            continue
        out += [f"{table}.{c}" for c in cols if c not in got]
    return out


def draft_columns():
    """{table: columns} of the DR4 draft data model (also: which tables TAP serves and which only DataLink)."""
    return {k: v["columns"] for k, v in json.loads(DRAFT_DATA_MODEL.read_text()).items()}


def live_columns(release, tap):
    """{table: columns} from the archive's TAP_SCHEMA for the tables this release's queries use."""
    out = {}
    for table in required_columns(release):
        rows = tap.query(f"SELECT column_name FROM tap_schema.columns WHERE table_name = '{table}'",
                         sync=True, cache=False)
        cols = {str(c).lower() for c in rows["column_name"]}
        if cols:
            out[table] = cols
    return out


def check_release(release, tap):
    """Return the required tables and columns missing from the archive for this release (empty = ready)."""
    return missing_columns(release, live_columns(release, tap))


def star_queries(release, columns, upload="TAP_UPLOAD.ids"):
    """ADQL for `columns` (names from STAR_COLUMNS) of the uploaded source_ids: one query per table that holds any
    of them, as (adql, names). Each is an inner join on the upload, so a star missing from a table has no row in
    that query; merge the results on source_id. Names the release does not publish come back in no query."""
    unknown = [c for c in columns if c not in STAR_COLUMNS]
    if unknown:
        raise ValueError(f"not star columns: {unknown}; known: {STAR_COLUMNS}")
    out = []
    for table, exprs in tables(release)["star_tables"].items():
        names = [c for c in columns if c in exprs]
        if not names:
            continue
        sel = ", ".join(f"{exprs[c].format(t='s')} AS {c}" for c in names)
        out.append((f"SELECT u.source_id, {sel} FROM {upload} AS u JOIN {table} AS s ON s.source_id = u.source_id",
                    names))
    return out


def fetch_star_rows(tap, ids, release, columns, chunk=50_000, workers=4):
    """`columns` (STAR_COLUMNS names) for each source id, as a pandas frame with one row per id: uploaded in
    chunks, one query per table, merged on source_id. A column the release lacks, or a star absent from the table
    that holds it, is NaN / None. Several classifier rows per star (DR3's vari_classifier_result has one) keep the
    highest best_class_score."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import pandas as pd
    from astropy.table import Table

    ids = np.unique(np.asarray(ids, np.int64))
    queries = star_queries(release, columns)
    jobs = [(q, names, ids[i:i + chunk]) for q, names in queries for i in range(0, len(ids), chunk)]

    def one(job):
        q, names, part = job
        for attempt in (0, 1):
            try:
                d = tap.query(q, uploads={"ids": Table({"source_id": part})}, cache=False).to_pandas()
                d.columns = [c.lower() for c in d.columns]
                return names, d
            except Exception:
                if attempt:
                    raise

    with ThreadPoolExecutor(max(1, workers)) as pool:
        results = list(pool.map(one, jobs))
    out = pd.DataFrame({"source_id": ids})
    for _, names in queries:
        parts = [d for n, d in results if n is names and len(d)]
        if not parts:
            continue
        d = pd.concat(parts, ignore_index=True)
        if "best_class_score" in d:
            d = d.sort_values("best_class_score", ascending=False)
        out = out.merge(d.drop_duplicates("source_id"), on="source_id", how="left")
    for c in columns:
        if c not in out:
            out[c] = None if c == "best_class_name" else np.nan
    return out[["source_id", *columns]]
