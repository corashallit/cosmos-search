"""Count Gaia DR3 sources near the 100 pc boundary (scoping probe, 2026-10-21).

Run: python scripts/probe_gaia_counts.py   (needs pyvo, requests)
"""
import pyvo, requests, time
UA = "cosmos-search/0.0 (100 pc census scoping; mailto:cora.shallit@gmail.com)"
s = requests.Session(); s.headers["User-Agent"] = UA
tap = pyvo.dal.TAPService("https://gea.esac.esa.int/tap-server/tap", session=s)
qs = {
 "plx>10 (all)": "SELECT COUNT(*) FROM gaiadr3.gaia_source WHERE parallax > 10",
 "plx>10, plx/err>5": "SELECT COUNT(*) FROM gaiadr3.gaia_source WHERE parallax > 10 AND parallax_over_error > 5",
 "plx>10, plx/err>10": "SELECT COUNT(*) FROM gaiadr3.gaia_source WHERE parallax > 10 AND parallax_over_error > 10",
 "plx>8 (all)": "SELECT COUNT(*) FROM gaiadr3.gaia_source WHERE parallax > 8",
 "plx>10 has_xp_continuous": "SELECT COUNT(*) FROM gaiadr3.gaia_source WHERE parallax > 10 AND has_xp_continuous='True'",
 "plx>10 has_rvs": "SELECT COUNT(*) FROM gaiadr3.gaia_source WHERE parallax > 10 AND has_rvs='True'",
 "plx>10 has_epoch_photometry": "SELECT COUNT(*) FROM gaiadr3.gaia_source WHERE parallax > 10 AND has_epoch_photometry='True'",
 "plx>10 nss": "SELECT COUNT(*) FROM gaiadr3.gaia_source WHERE parallax > 10 AND non_single_star > 0",
 "GCNS main": "SELECT COUNT(*) FROM external.gaiaedr3_gcns_main_1",
}
for k,q in qs.items():
    t=time.time()
    try:
        r = tap.run_async(q, maxrec=10)
        print(f"{k}: {r.to_table()[0][0]}  ({time.time()-t:.0f}s)", flush=True)
    except Exception as e:
        print(f"{k}: ERROR {e}", flush=True)
