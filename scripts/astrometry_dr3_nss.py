"""Fetch the Gaia DR3 astrometric non-single-star solutions used by the DR3 rehearsal of the Kepler-bound filter.

    python scripts/astrometry_dr3_nss.py --out /mnt/project-files/astrometry/dr3

Writes orbits.parquet (nss_two_body_orbit, astrometric solution types) and accel.parquet (nss_acceleration_astro),
each joined with the gaia_source and astrophysical_parameters columns the filter needs."""

import argparse
from pathlib import Path

from cosmos_search.gaia import archive

SRC = ("g.phot_g_mean_mag, g.bp_rp, g.ruwe, g.ipd_frac_multi_peak, g.ipd_gof_harmonic_amplitude, "
       "g.ipd_frac_odd_win, g.phot_variable_flag, g.astrometric_excess_noise, g.ra, g.dec, g.l, g.b, "
       "g.duplicated_source, ap.radius_gspphot, ap.mass_flame, ap.radius_flame, ap.lum_flame, ap.teff_gspphot")
ORBITS = f"""SELECT n.source_id, n.nss_solution_type, n.period, n.period_error, n.eccentricity, n.eccentricity_error,
 n.a_thiele_innes, n.b_thiele_innes, n.f_thiele_innes, n.g_thiele_innes,
 n.a_thiele_innes_error, n.b_thiele_innes_error, n.f_thiele_innes_error, n.g_thiele_innes_error,
 n.parallax, n.parallax_error, n.goodness_of_fit, n.significance, n.flags, n.obj_func, {SRC}
FROM gaiadr3.nss_two_body_orbit AS n JOIN gaiadr3.gaia_source AS g USING (source_id)
LEFT JOIN gaiadr3.astrophysical_parameters AS ap USING (source_id)
WHERE n.a_thiele_innes IS NOT NULL"""
ACCEL = f"""SELECT n.source_id, n.nss_solution_type, n.accel_ra, n.accel_dec, n.accel_ra_error, n.accel_dec_error,
 n.deriv_accel_ra, n.deriv_accel_dec, n.deriv_accel_ra_error, n.deriv_accel_dec_error,
 n.parallax, n.parallax_error, n.goodness_of_fit, n.significance, n.flags, {SRC}
FROM gaiadr3.nss_acceleration_astro AS n JOIN gaiadr3.gaia_source AS g USING (source_id)
LEFT JOIN gaiadr3.astrophysical_parameters AS ap USING (source_id)"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, q in (("orbits", ORBITS), ("accel", ACCEL)):
        t = archive.tap().query(q)
        df = t.to_pandas()
        df.to_parquet(out / f"{name}.parquet")
        print(name, len(df), flush=True)


if __name__ == "__main__":
    main()
