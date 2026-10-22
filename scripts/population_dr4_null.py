"""DR3 check of the DR4 null (docs/population.md §9): LRTs for the new terms, overdispersion before and after,
the frozen windows under the new null, and a half-A rescan with phi = max(phi_SOV, phi_SOV1) (review R21).
Exploratory: both DR3 halves are already open, so nothing here is confirmatory."""
import argparse, json
import numpy as np, pandas as pd
from cosmos_search.population import analysis, glm, run, scan

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/home/user/pop/full")
ap.add_argument("--density", default="/home/user/pop/aux/density_hpx6.parquet")
ap.add_argument("--write", default="/home/user/pop/dr4null/check.json")
a = ap.parse_args()

stars = analysis.load_stars(a.out, density=a.density)
evA = run.events_A(a.out); mA = np.load(f"{a.out}/mask_A.npy")
evB = pd.DataFrame(run.load_sealed(f"{a.out}/events_B.json.age")); _, mB = run.masks(a.out)
y = {"A": analysis.outcome(stars, evA, 0, mA), "B": analysis.outcome(stars, evB, 1, mB)}
y1 = {"A": analysis.outcome(stars, evA, 0, mA, cls="SOV1"), "B": analysis.outcome(stars, evB, 1, mB, cls="SOV1")}
E = {"A": stars.E_gen_A.to_numpy(), "B": stars.E_gen_B.to_numpy()}
codes, _ = glm.covariate_codes(stars)
frozen = [t for t in json.load(open("population/candidates_A.json"))["model"]["terms"] if t != "cmd"]
dr4 = list(glm.DR4)
res = {"n_stars": len(stars), "frozen_terms": frozen, "dr4_terms": dr4, "lrt_A": {}}

base = list(frozen)
for t in ("dens", "vt", "gal"):
    stat, df, p = glm.lrt(codes, y["A"], E["A"], base, t)
    res["lrt_A"][t] = {"stat": stat, "df": df, "p": p}
    print("LRT", t, round(stat, 1), df, p, flush=True)
    base.append(t)

win = scan.Windows()
px, sh = stars.pixel.to_numpy(), stars.shell.to_numpy()
lam, phi = {}, {}
for name, terms in (("frozen", frozen), ("dr4", dr4)):
    for h in "AB":
        m = glm.fit(codes, y[h], E[h], terms)
        lam[(name, h)] = m["predict"](codes, E[h])
        phi[(name, h, "SOV")] = scan.dispersion(win, scan.cell_sums(px, sh, y[h]), scan.cell_sums(px, sh, lam[(name, h)]))[1]
        phi[(name, h, "SOV1")] = analysis.monitor_dispersion(stars, y1[h], E[h], terms, win)
        if name == "dr4" and h == "A":
            res["dr4_coef_A"] = {l: round(float(b), 3) for l, b in zip(m["labels"], m["beta"])
                                 if l.split("=")[0] in ("dens", "vt", "gal")}
        print(name, h, "phi SOV max/median", phi[(name, h, "SOV")].max().round(2), np.median(phi[(name, h, "SOV")]).round(2),
              "SOV1", phi[(name, h, "SOV1")].max().round(2), flush=True)
res["phi"] = {f"{n}/{h}/{c}": v.round(3).tolist() for (n, h, c), v in phi.items()}

from scipy import stats
rep = json.load(open("population/replication_B.json"))
rows = []
for c in rep["results"]:
    mem = analysis.window_members(stars, win, c)
    ri, j = int(win.disc[c["disc"], 1]), c["shell_window"]
    row = {"ra": round(c["ra"], 1), "dec": round(c["dec"], 1), "radius": c["radius_deg"], "distance": c["distance"]}
    for name in ("frozen", "dr4"):
        for h in "AB":
            o, e = float(y[h][mem].sum()), float(lam[(name, h)][mem].sum())
            f = float(max(phi[(name, h, "SOV")][ri, j], phi[(name, h, "SOV1")][ri, j]))
            p = float(stats.nbinom.sf(o - 1, e / (f - 1), 1 / f)) if f > 1 else float(stats.poisson.sf(o - 1, e))
            row[f"{name}_{h}"] = {"O": o, "E": round(e, 1), "phi_max": round(f, 2), "p_nb": p}
    rows.append(row); print(json.dumps(row), flush=True)
res["frozen_windows"] = rows

cands, info, _, _ = analysis.search(stars, y["A"], E["A"], win=win, terms=dr4, y_monitor=y1["A"],
                                    rng=np.random.default_rng(7), k_max=10)
info.pop("_lam")
for c in cands:
    mem = analysis.window_members(stars, win, c)
    o, e = float(y["B"][mem].sum()), float(lam[("dr4", "B")][mem].sum())
    ri, j = int(win.disc[c["disc"], 1]), c["shell_window"]
    f = float(max(phi[("dr4", "B", "SOV")][ri, j], phi[("dr4", "B", "SOV1")][ri, j]))
    c["B"] = {"O": o, "E": e, "phi_max": f,
              "p_nb": float(stats.nbinom.sf(o - 1, e / (f - 1), 1 / f)) if f > 1 else float(stats.poisson.sf(o - 1, e))}
    print("rescan", round(c["ra"], 1), round(c["dec"], 1), c["radius_deg"], c["distance"], c["O"], round(c["E"], 1),
          c["p_fw"], "B", o, round(e, 1), c["B"]["p_nb"], flush=True)
res["rescan_A"] = {"candidates": cands, "max_llr": info["max_llr"], "null_quantiles": info["null_quantiles"],
                   "phi_sov": info["phi_sov_per_scale"], "phi_monitor": info["phi_monitor_per_scale"]}
json.dump(res, open(a.write, "w"), indent=1, default=float)
print("done", flush=True)
