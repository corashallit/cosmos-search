"""Run the per-candidate checks (cosmos_search.slow.checks) on the strongest slow-change candidates of a DR3 run.

    python scripts/slow_checks.py --detect slow_detect/ --out slow_checks/ [--n-episode 30 --n-trend 10]

Candidates: stars at G < 17 that pass the detector's 0.1% thresholds, the strongest dimming episodes (smooth
light curves, Abbé < 0.5) and the strongest fading trends. For each: Gaia DR3 astrometry and quality columns and
DPAC's variability class, the colour of the change, NEOWISE W1/W2 inside the episode's dates against outside with
what each explanation predicts, and ZTF levels before and after 2020-01-20. Writes checks.csv (one row per star)
and the raw NEOWISE and ZTF epochs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.dayone import external
from cosmos_search.energy import budget
from cosmos_search.gaia.archive import tap
from cosmos_search.net.tap import Tap
from cosmos_search.slow import checks

DR3_SPAN = (-136.0, 901.5)  # BTJD: DR3 epoch photometry, 2014-07-25 to 2017-05-28
GAIA_COLS = ("source_id, ra, dec, pmra, pmdec, parallax, ref_epoch, phot_g_mean_mag, bp_rp, ruwe, "
             "ipd_frac_multi_peak, phot_bp_rp_excess_factor")


def candidates(stats, thr, n_episode, n_trend):
    s = stats[(stats["gbin"] <= 5) & stats["s_dim"].notna()]
    th = thr["G<19"]["0.001"]
    ep = s[(s["s_dim"] > th["s"]) & (s["abbe"] < 0.5) & (s["dim_visits"] >= 3)].nlargest(n_episode, "s_dim")
    tr = s[(s["slope_z"] < -th["z"]) & ~s["source_id"].isin(ep["source_id"])].nsmallest(n_trend, "slope_z")
    return pd.concat([ep.assign(why="episode"), tr.assign(why="trend")], ignore_index=True)


def gaia_rows(ids):
    """gaia_source columns and DPAC's best variability class for the given DR3 source ids."""
    t = tap()
    g, c = [], []
    for i in range(0, len(ids), 200):
        chunk = ",".join(str(int(x)) for x in ids[i:i + 200])
        g.append(t.query(f"SELECT {GAIA_COLS} FROM gaiadr3.gaia_source WHERE source_id IN ({chunk})",
                         sync=True).to_pandas())
        c.append(t.query("SELECT source_id, best_class_name, best_class_score FROM gaiadr3.vari_classifier_result "
                         f"WHERE source_id IN ({chunk})", sync=True).to_pandas())
    return pd.concat(g, ignore_index=True).merge(pd.concat(c, ignore_index=True), on="source_id", how="left")


def at_2019(r):
    """Position propagated from ref_epoch to 2019.0 (NEOWISE-R's middle), small-angle."""
    dt = 2019.0 - r.ref_epoch
    pmra, pmdec = (0.0 if pd.isna(r.pmra) else r.pmra), (0.0 if pd.isna(r.pmdec) else r.pmdec)
    return (r.ra + pmra * dt / 3.6e6 / np.cos(np.radians(r.dec)), r.dec + pmdec * dt / 3.6e6)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-episode", type=int, default=30)
    ap.add_argument("--n-trend", type=int, default=10)
    a = ap.parse_args(argv)
    d, out = Path(a.detect), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = pd.read_parquet(d / "stats.parquet")
    thr = json.loads((d / "summary.json").read_text())["thresholds"]
    cand = candidates(stats, thr, a.n_episode, a.n_trend)
    gaia = gaia_rows(cand["source_id"].to_numpy())
    cand = cand.merge(gaia, on="source_id", how="left")
    irsa = Tap(checks.IRSA_TAP)

    rows, wise_all, ztf_all = [], [], []
    for r in cand.itertuples(index=False):
        row = {"source_id": r.source_id, "why": r.why, "g": r.phot_g_mean_mag, "bp_rp": r.bp_rp,
               "class": r.best_class_name, "ruwe": r.ruwe, "ipd_multi": r.ipd_frac_multi_peak,
               "s_dim": r.s_dim, "slope_pct_yr": 100 * r.slope, "slope_z": r.slope_z, "abbe": r.abbe,
               "dim_depth": r.dim_depth, "dim_t0_mjd": r.dim_t0 + checks.BTJD_T0_MJD,
               "dim_t1_mjd": r.dim_t1 + checks.BTJD_T0_MJD}
        row.update({f"colour_{k}": v for k, v in
                    checks.colour(r.depth_bp, r.depth_bp_err, r.depth_rp, r.depth_rp_err).items()})
        teff = float(budget.teff_from_bp_rp(r.bp_rp if pd.notna(r.bp_rp) else 0.82))
        ra, dec = at_2019(r)
        try:
            w = checks.neowise_clean(irsa.query(checks.neowise_adql(ra, dec), sync=True).to_pandas())
        except Exception as e:  # noqa: BLE001 (one failed fetch should not stop the rest)
            print(r.source_id, "NEOWISE:", e)
            w = checks.neowise_clean([])
        wise_all.append(w.assign(source_id=r.source_id))
        if r.why == "episode":
            wc = checks.wise_change(w, row["dim_t0_mjd"], row["dim_t1_mjd"])
            for b in ("W1", "W2"):
                row[f"{b}_change"], row[f"{b}_err"], row[f"{b}_visits_in"] = wc[b]
                for k, v in checks.wise_predictions(max(r.dim_depth, 0.0), teff)[b].items():
                    row[f"{b}_pred_{k}"] = v
        else:
            # A steady fade: W1/W2 slope over the light curve's years against Gaia's (fraction per year).
            t0, t1 = getattr(r, "t_first", DR3_SPAN[0]), getattr(r, "t_last", DR3_SPAN[1])
            ws = checks.wise_slope(w, t0 + checks.BTJD_T0_MJD, t1 + checks.BTJD_T0_MJD)
            for b in ("W1", "W2"):
                row[f"{b}_slope"], row[f"{b}_slope_err"], row[f"{b}_visits_in"] = ws[b]
                row[f"{b}_slope_pred_dust_in_front"] = r.slope * checks.WISE_OVER_G_DUST[b]
        if r.dec > -30:
            try:
                z = external.ztf(r.source_id, r.ra, r.dec)
            except Exception as e:  # noqa: BLE001
                print(r.source_id, "ZTF:", e)
                z = external._empty()
            ztf_all.append(z)
            for b, v in checks.after(z).items():
                row[f"{b}_change_after_2020"] = v["change"]
                row[f"{b}_slope_after_2020"] = v["slope_post_per_yr"]
        rows.append(row)
        print(r.source_id, r.why, f"G {r.phot_g_mean_mag:.1f}", r.best_class_name,
              {k: round(row.get(f"{k}_change", row.get(f"{k}_slope", np.nan)), 4) for k in ("W1", "W2")})
    res = pd.DataFrame(rows)
    res.to_csv(out / "checks.csv", index=False)
    pd.concat(wise_all, ignore_index=True).to_parquet(out / "neowise.parquet", index=False)
    if ztf_all:
        pd.concat(ztf_all, ignore_index=True).to_parquet(out / "ztf.parquet", index=False)
    print(res.groupby(["why", "class"], dropna=False).size())


if __name__ == "__main__":
    main()
