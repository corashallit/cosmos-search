"""Run the per-candidate checks (cosmos_search.slow.checks) on the strongest stars of a search's fading list.

    python scripts/slow_checks.py --fading R/fading.parquet --out R/checks [--n 100] [--schema gaiadr4] [--max-g 17]

--fading is the list `scripts/slow_search.py flag` writes (ordered by strength). For each of the first --n stars
(optionally only those brighter than --max-g): gaia_source astrometry and quality columns and DPAC's variability
class, the colour of the change, NEOWISE W1/W2 against what each explanation predicts (inside the faint window
against outside for an episode, the slope over the light curve's years for a trend; a repeating dimming has no
single window and gets neither), and ZTF levels before and after 2020-01-20. Writes checks.csv (one row per star)
and the raw NEOWISE and ZTF epochs. A failed fetch is recorded and the next star goes on.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.dayone import external
from cosmos_search.energy import budget
from cosmos_search.gaia.archive import tap
from cosmos_search.net.tap import Tap
from cosmos_search.slow import checks

GAIA_COLS = ("source_id, ra, dec, pmra, pmdec, parallax, ref_epoch, phot_g_mean_mag, bp_rp, ruwe, "
             "ipd_frac_multi_peak, phot_bp_rp_excess_factor")


def candidates(fading, n, max_g=None):
    f = fading if max_g is None else fading[fading["phot_g_mean_mag"] < max_g]
    return f.sort_values("strength", ascending=False).head(n).rename(columns={"id": "source_id"})


def gaia_rows(ids, schema):
    """gaia_source columns and DPAC's best variability class for the given source ids."""
    t = tap()
    g, c = [], []
    for i in range(0, len(ids), 200):
        chunk = ",".join(str(int(x)) for x in ids[i:i + 200])
        g.append(t.query(f"SELECT {GAIA_COLS} FROM {schema}.gaia_source WHERE source_id IN ({chunk})",
                         sync=True).to_pandas())
        c.append(t.query(f"SELECT source_id, best_class_name, best_class_score FROM {schema}.vari_classifier_result "
                         f"WHERE source_id IN ({chunk})", sync=True).to_pandas())
    for d in g + c:
        d.columns = [x.lower() for x in d.columns]
    return pd.concat(g, ignore_index=True).merge(pd.concat(c, ignore_index=True), on="source_id", how="left")


def at_2019(r):
    """Position propagated from ref_epoch to 2019.0 (NEOWISE-R's middle), small-angle."""
    dt = 2019.0 - r.ref_epoch
    pmra, pmdec = (0.0 if pd.isna(r.pmra) else r.pmra), (0.0 if pd.isna(r.pmdec) else r.pmdec)
    return (r.ra + pmra * dt / 3.6e6 / np.cos(np.radians(r.dec)), r.dec + pmdec * dt / 3.6e6)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fading", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-g", type=float)
    ap.add_argument("--schema", default="gaiadr4")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cand = candidates(pd.read_parquet(a.fading), a.n, a.max_g)
    cand = cand.drop(columns=["phot_g_mean_mag", "bp_rp"], errors="ignore").merge(
        gaia_rows(cand["source_id"].to_numpy(), a.schema), on="source_id", how="left")
    irsa = Tap(checks.IRSA_TAP)

    rows, wise_all, ztf_all = [], [], []
    t0 = time.time()
    for r in cand.itertuples(index=False):
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
        try:
            w = checks.neowise_clean(irsa.query(checks.neowise_adql(ra, dec), sync=True).to_pandas())
        except Exception as e:  # noqa: BLE001 (one failed fetch should not stop the rest)
            row["errors"] += f"NEOWISE: {e!r:.120}; "
            w = checks.neowise_clean([])
        wise_all.append(w.assign(source_id=r.source_id))
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
        if r.dec > -30:
            try:
                z = external.ztf(r.source_id, r.ra, r.dec)
            except Exception as e:  # noqa: BLE001
                row["errors"] += f"ZTF: {e!r:.120}; "
                z = external._empty()
            ztf_all.append(z)
            for b, v in checks.after(z).items():
                row[f"{b}_change_after_2020"] = v["change"]
                row[f"{b}_slope_after_2020"] = v["slope_post_per_yr"]
        rows.append(row)
        print(r.source_id, r.by, f"G {r.phot_g_mean_mag:.1f}", r.best_class_name, row["errors"] or "", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "checks.csv", index=False)
    pd.concat(wise_all, ignore_index=True).to_parquet(out / "neowise.parquet", index=False)
    if ztf_all:
        pd.concat(ztf_all, ignore_index=True).to_parquet(out / "ztf.parquet", index=False)
    print(f"{len(res)} stars in {time.time() - t0:.0f} s; fetch errors on {int((res['errors'] != '').sum())}")
    print(res.groupby(["by", "class"], dropna=False).size())


if __name__ == "__main__":
    main()
