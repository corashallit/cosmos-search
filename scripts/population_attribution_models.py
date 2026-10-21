"""Attribution refits for docs/population.md §12: GLM with DPAC class and all-Gaia source density (attribution only,
never the null), a half-A rescan under them, and window 1 against its mirror positions about the bulge."""
import json, numpy as np, pandas as pd
from cosmos_search.population import analysis, scan, run, glm
out = "/home/user/pop/full"
stars = analysis.load_stars(out, analysis.STAR_COLS + ["best_class_name"])
dens = pd.read_parquet("/home/user/pop/aux/density_hpx6.parquet").set_index("hpx").n
area = 41252.96 / 49152
stars["log_dens"] = np.log10(stars.pixel.map(dens).to_numpy() / area)
win = scan.Windows()
rep = json.load(open("population/replication_B.json"))
evA = run.events_A(out); mA = np.load(f"{out}/mask_A.npy")
evB = pd.DataFrame(run.load_sealed(f"{out}/events_B.json.age")); _, mB = run.masks(out)
y = {"A": analysis.outcome(stars, evA, 0, mA), "B": analysis.outcome(stars, evB, 1, mB)}
E = {"A": stars.E_gen_A.to_numpy(), "B": stars.E_gen_B.to_numpy()}
codes, _ = glm.covariate_codes(stars)
codes["dpac"] = pd.factorize(stars.best_class_name.replace("", "none"))[0].astype(np.int16)
codes["dens"] = np.digitize(stars.log_dens, np.quantile(stars.log_dens, np.linspace(0, 1, 11)[1:-1])).astype(np.int16)
terms = [t for t in json.load(open("population/candidates_A.json"))["model"]["terms"] if t != "cmd"]
variants = {"null": terms, "class": terms + ["dpac"], "density": terms + ["dens"], "class+density": terms + ["dpac", "dens"]}
lam = {}
for h in "AB":
    for k, tt in variants.items():
        m = glm.fit(codes, y[h], E[h], tt)
        lam[(h, k)] = m["predict"](codes, E[h])
        if h == "A" and k == "density":
            print("density coefs", {l: round(b, 2) for l, b in zip(m["labels"], m["beta"]) if l.startswith("dens")})
px, sh = stars.pixel.to_numpy(), stars.shell.to_numpy()
rows = []
for c in rep["results"]:
    mem = analysis.window_members(stars, win, c)
    row = {"ra": round(c["ra"], 1), "dec": round(c["dec"], 1), "median_log_dens": round(float(np.median(stars.log_dens[mem])), 2)}
    for h in "AB":
        row[h] = {"O": float(y[h][mem].sum()), **{k: round(float(lam[(h, k)][mem].sum()), 1) for k in variants}}
    rows.append(row); print(json.dumps(row))
print("sky median log density", round(float(np.median(stars.log_dens)), 2))
# re-scan half A under the class+density model: does anything remain?
for k in ("density", "class+density"):
    obs, lc = scan.cell_sums(px, sh, y["A"]), scan.cell_sums(px, sh, lam[("A", k)])
    phi, _ = scan.dispersion(win, obs, lc)
    null = scan.null_max(win, lc, n=199, rng=np.random.default_rng(3))
    cands, top = scan.candidates(win, obs, lc, null, phi=phi)
    print(k, "rescan A:", [(round(x["ra"],1), round(x["dec"],1), x["radius_deg"], x["distance"], x["O"], round(x["E"],1), x["p_fw"]) for x in cands])
json.dump(rows, open("/home/user/pop/attribution_models.json", "w"), indent=1)

from astropy.coordinates import SkyCoord
import astropy.units as u, healpy as hp
from scipy import stats
lc_cd = {h: lam[(h, "class+density")] for h in "AB"}
print("window 1 mirrors (class+density model), 8 deg, all distances and 2000-8000 pc:")
for (l, b) in [(348.4, -12.4), (11.6, -12.4), (348.4, 12.4), (11.6, 12.4), (0.0, -12.4), (0.0, 12.4), (330, 12), (30, -12)]:
    c = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic").icrs
    pix = hp.query_disc(64, hp.ang2vec(c.ra.deg, c.dec.deg, lonlat=True), np.radians(8), nest=True)
    for lo, hi in ((0, 8), (4, 6)):
        m = np.isin(px, pix) & (sh >= lo) & (sh < hi)
        o = {h: y[h][m].sum() for h in "AB"}; e = {h: lc_cd[h][m].sum() for h in "AB"}
        print(f"  l={l:6.1f} b={b:6.1f} shells {lo}-{hi}: A {o['A']:.0f}/{e['A']:.0f}  B {o['B']:.0f}/{e['B']:.0f}")
