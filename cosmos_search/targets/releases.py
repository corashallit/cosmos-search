"""Table names per Gaia release. The one place to edit on DR4 day one if the archive's names differ.

DR4 names follow the draft data model of 2026-06-26 (gaia-dr4-prerelease-draft-data-model.pdf):
vari_summary keeps every DR3 G-FoV column used here (min/max/median_mag_g_fov, mad_mag_g_fov, scaled by
1.4826 in both releases, num_selected_g_fov, skewness_mag_g_fov, outlier_median_g_fov) and adds flux-space
moments, per-CCD G statistics and G/BP/RP correlations. The schema name "gaiadr4" is an assumption until
release; `check_release` confirms it against TAP_SCHEMA.
"""

RELEASES = {
    "dr3": {
        "schema": "gaiadr3",
        "vari_summary": "gaiadr3.vari_summary",
        "gaia_source": "gaiadr3.gaia_source",
        "n_vari_summary": 11_754_237,
    },
    "dr4": {
        "schema": "gaiadr4",
        "vari_summary": "gaiadr4.vari_summary",
        "gaia_source": "gaiadr4.gaia_source",
        "n_vari_summary": 403_061_676,  # draft data model row count
    },
}

REQUIRED_COLUMNS = {
    "vari_summary": ("source_id", "num_selected_g_fov", "min_mag_g_fov", "max_mag_g_fov",
                     "median_mag_g_fov", "mad_mag_g_fov", "skewness_mag_g_fov", "outlier_median_g_fov"),
}


def tables(release):
    try:
        return RELEASES[release]
    except KeyError:
        raise ValueError(f"unknown release {release!r}; known: {sorted(RELEASES)}") from None


def check_release(release, tap):
    """Return the required columns missing from the archive for this release (empty = ready)."""
    t = tables(release)
    missing = []
    for key, cols in REQUIRED_COLUMNS.items():
        rows = tap.query(f"SELECT column_name FROM tap_schema.columns WHERE table_name = '{t[key]}'",
                         sync=True, cache=False)
        have = {c.lower() for c in rows["column_name"]}
        missing += [f"{t[key]}.{c}" for c in cols if c not in have]
    return missing
