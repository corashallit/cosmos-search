"""Attribution summary for docs/population.md §12: region overlaps, host class mix, mirror and SOV-1 counts, crowding."""
import json, numpy as np, pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from cosmos_search.population import analysis, regions, scan, run, glm
out = "/home/user/pop/full"
cols = analysis.STAR_COLS + ["best_class_name", "l", "b", "r_med_photogeo", "excess", "sigma", "n_good"]
stars = analysis.load_stars(out, cols)
win = scan.Windows()
rep = json.load(open("population/replication_B.json"))
evA = run.events_A(out); mA = np.load(f"{out}/mask_A.npy")
yA = analysis.outcome(stars, evA, 0, mA)
mirA = analysis.outcome(stars, evA, 0, mA, cls="MIR")
sov1A = analysis.outcome(stars, evA, 0, mA, cls="SOV1")
evB = pd.DataFrame(run.load_sealed(f"{out}/events_B.json.age")); _, mB = run.masks(out)
yB = analysis.outcome(stars, evB, 1, mB)
cls_all = stars.best_class_name.replace("", "none").value_counts(normalize=True)
hosts_all = stars.best_class_name[yA + yB > 0].replace("", "none").value_counts(normalize=True)
res = []
for c in rep["results"]:
    m = analysis.window_members(stars, win, c)
    g = SkyCoord(c["ra"] * u.deg, c["dec"] * u.deg).galactic
    lo, hi = regions.window_distance_range(c)
    host = m & (yA + yB > 0)
    mix = stars.best_class_name[host].replace("", "none").value_counts(normalize=True)
    lift = (mix / hosts_all.reindex(mix.index)).round(2)
    # per-class excess: where do the extra events sit?
    t = pd.DataFrame({"cls": stars.best_class_name[m].replace("", "none"), "y": (yA + yB)[m]})
    by = t.groupby("cls").y.sum().sort_values(ascending=False)
    ev = pd.concat([evA[evA.source_id.isin(stars.source_id[m]) & (evA.cls == "SOV")], evB[evB.source_id.isin(stars.source_id[m]) & (evB.cls == "SOV")]])
    days = np.floor(ev.t_deep).value_counts()
    res.append({"ra": c["ra"], "dec": round(c["dec"], 2), "l": round(g.l.deg, 1), "b": round(g.b.deg, 1),
                "radius": c["radius_deg"], "dist": c["distance"], "replicated": c["replicated"],
                "regions": regions.overlapping(c["ra"], c["dec"], c["radius_deg"], lo, hi),
                "n_stars": int(m.sum()), "events_AB": int((yA + yB)[m].sum()),
                "top_classes_events": by.head(4).to_dict(), "class_lift_vs_all_hosts": lift.head(5).to_dict(),
                "mirror_A_O": float(mirA[m].sum()), "sov1_A_O": float(sov1A[m].sum()),
                "median_ipd": float(stars.ipd_frac_multi_peak[m].median()), "median_ruwe": float(stars.ruwe[m].median()),
                "max_events_one_day": int(days.max()) if len(days) else 0, "n_days": int(len(days))})
# whole-sky reference for mirror & sov1 per star
print("sky mirror/SOV ratio", mirA.sum() / yA.sum(), "sov1/SOV", sov1A.sum() / yA.sum())
for r in res:
    print(json.dumps(r))
json.dump(res, open("/home/user/pop/attribution.json", "w"), indent=1, default=float)


# --- model refits with DPAC class and source density (attribution only) ---

