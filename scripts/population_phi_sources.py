"""Where does the window overdispersion come from? (docs/population.md §14, exploratory on DR3.)

1. Static or transient. Halves A and B interleave by day, so structure fixed on the sky (unmodelled populations,
   extinction, static calibration) shows up in both halves' residuals at the same windows, while structure tied to
   particular scans (per-transit systematics) does not. Per scale, with z = (O - E) / sqrt(E):
   phi_h - 1 = static + transient_h, static = robust cov(z_A, z_B) = (s(z_A + z_B)^2 - s(z_A - z_B)^2) / 4,
   s = 1.4826 MAD.
2. Scan coincidence. Events whose deepest epochs fall within 0.1 d and 2 deg of each other are one scan; counting
   each group once (declustered) removes per-scan correlation. phi is recomputed on declustered counts.
3. Subsets: phi of each event subset under the null refitted to that subset alone.
4. Variants of the null: DPAC class (diagnostic only: it reads the light curve), the dip-blind features of
   `features.outside_deepest` (run.py features), and both."""
import argparse, json
import numpy as np, pandas as pd
from cosmos_search.population import analysis, glm, run, scan

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/home/user/pop/full")
ap.add_argument("--density", default="/home/user/pop/aux/density_hpx6.parquet")
ap.add_argument("--write", default="/home/user/pop/phi/phi_sources.json")
a = ap.parse_args()

stars = analysis.load_stars(a.out, analysis.STAR_COLS + ["best_class_name"], density=a.density)
codes, _ = glm.covariate_codes(stars)
terms = list(glm.DR4)
win = scan.Windows()
px, sh = stars.pixel.to_numpy(), stars.shell.to_numpy()
E = {"A": stars.E_gen_A.to_numpy(), "B": stars.E_gen_B.to_numpy()}
evA = run.events_A(a.out); mA = np.load(f"{a.out}/mask_A.npy")
evB = pd.DataFrame(run.load_sealed(f"{a.out}/events_B.json.age")); _, mB = run.masks(a.out)
pos = stars.set_index("source_id")[["ra", "dec"]]


def kept(ev, half, mask, cls):
    e = ev[(ev.cls == cls) & (ev.half == half) & ev.source_id.isin(stars.source_id)]
    return e[~mask[run.hour_bin(e.t_deep.to_numpy()), run.mask_pixel(e.source_id.to_numpy())]].copy()


def decluster(e):
    """Keep one event per scan group (`analysis.decluster`)."""
    e = e.reset_index(drop=True)
    p = pos.loc[e.source_id]
    keep, size = analysis.decluster(e.t_deep.to_numpy(), p.ra.to_numpy(), p.dec.to_numpy())
    e["group_size"] = size
    return e.iloc[keep], e


def counts(e):
    return stars.source_id.map(e.groupby("source_id").size()).fillna(0).to_numpy(float)


def split(yA, yB, label):
    """Per scale: phi_A, phi_B, static, transient (`analysis.static_transient`), from DR4-null fits per half."""
    cells = {}
    for h, y in (("A", yA), ("B", yB)):
        lam = glm.fit(codes, y, E[h], terms)["predict"](codes, E[h])
        cells[h] = (scan.cell_sums(px, sh, y), scan.cell_sums(px, sh, lam))
    st = analysis.static_transient(win, *cells["A"], *cells["B"])
    EA = win.totals(cells["A"][1])
    out = {}
    for (ri, j), v in st.items():
        sw = win.shells[j]
        out[f"{scan.RADII_DEG[ri]:g}deg/{sw[0]}-{sw[1]}"] = {
            "n_windows": v["n_windows"], "median_E": float(np.median(EA[win.disc[:, 1] == ri, j])), "phi_A": round(v["phi_1"], 3),
            "phi_B": round(v["phi_2"], 3), "static": round(v["static"], 3), "transient_A": round(v["transient_1"], 3),
            "transient_B": round(v["transient_2"], 3), "corr": round(v["corr"], 3)}
    print(label, {k: (v["phi_A"], v["static"], v["corr"]) for k, v in out.items() if k.startswith(("8deg", "4deg"))},
          flush=True)
    return out


res = {}
for cls in ("SOV", "SOV1"):
    eA, eB = kept(evA, 0, mA, cls), kept(evB, 1, mB, cls)
    res[cls] = split(counts(eA), counts(eB), cls)
    dA, fullA = decluster(eA); dB, fullB = decluster(eB)
    res[f"{cls}_coincidence"] = {"A_events": len(eA), "A_declustered": len(dA), "B_events": len(eB),
                                 "B_declustered": len(dB),
                                 "A_group_sizes": fullA.group_size.value_counts().sort_index().to_dict(),
                                 "B_group_sizes": fullB.group_size.value_counts().sort_index().to_dict()}
    print(cls, res[f"{cls}_coincidence"], flush=True)
    res[f"{cls}_declustered"] = split(counts(dA), counts(dB), f"{cls} declustered")
    json.dump(res, open(a.write, "w"), indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))

# SOV subsets: DPAC class, magnitude, depth
eA, eB = kept(evA, 0, mA, "SOV"), kept(evB, 1, mB, "SOV")
cls_of = stars.set_index("source_id").best_class_name
g_of = pd.read_parquet(f"{a.out}/stars.parquet", columns=["source_id", "phot_g_mean_mag"]).set_index("source_id").phot_g_mean_mag
subsets = {
    "solar_like": lambda e: cls_of.reindex(e.source_id).to_numpy() == "SOLAR_LIKE",
    "not_solar_like": lambda e: cls_of.reindex(e.source_id).to_numpy() != "SOLAR_LIKE",
    "g_lt_16": lambda e: g_of.reindex(e.source_id).to_numpy() < 16,
    "g_ge_16": lambda e: g_of.reindex(e.source_id).to_numpy() >= 16,
    "depth_lt_5pct": lambda e: e.depth.to_numpy() < 0.05,
    "depth_ge_5pct": lambda e: e.depth.to_numpy() >= 0.05,
}
for k, f in subsets.items():
    res[f"SOV_{k}"] = split(counts(eA[f(eA)]), counts(eB[f(eB)]), k)
    res[f"SOV_{k}"]["n_events_A"] = int(f(eA).sum())
    json.dump(res, open(a.write, "w"), indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
# 4. null variants
import glob
f = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{a.out}/features/*.parquet")])
fs = stars[["source_id"]].merge(f, on="source_id", how="left")
for key, col in (("flow", "f_low"), ("fhigh", "f_high"), ("bow", "bowley")):
    v = fs[col].to_numpy()
    edges = np.unique(np.nanquantile(v, [.2, .4, .6, .8, .9, .97]))
    codes[key] = np.where(np.isfinite(v), np.digitize(v, edges), len(edges) + 1).astype(np.int16)
codes["dpac"] = pd.factorize(stars.best_class_name.fillna("").replace("", "none"))[0].astype(np.int16)
yA, yB = counts(eA), counts(eB)
for label, extra in (("dpac", ["dpac"]), ("features", ["flow", "fhigh", "bow"]),
                     ("features+dpac", ["flow", "fhigh", "bow", "dpac"])):
    terms[:] = list(glm.DR4) + extra
    res[f"SOV_null+{label}"] = split(yA, yB, label)
    json.dump(res, open(a.write, "w"), indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
# 5. class mix of hosts
cls = stars.set_index("source_id").best_class_name.fillna("none").replace("", "none")
e = pd.concat([eA, eB])
res["class_mix"] = pd.DataFrame({"stars": cls.value_counts(), "sov": cls.reindex(e.source_id).value_counts()}) \
    .fillna(0).sort_values("stars", ascending=False).head(15).to_dict()
json.dump(res, open(a.write, "w"), indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
print("done", flush=True)
