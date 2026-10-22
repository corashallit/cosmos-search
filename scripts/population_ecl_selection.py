"""DR3 selection check for docs/population.md §14: do faint ECL hosts carry other faint points, and does the SOV rate
per pixel track the faint-ECL share? Needs run.py features output."""
import glob, numpy as np, pandas as pd
from cosmos_search.population import run
out = "/home/user/pop/full"
st = pd.read_parquet(f"{out}/stars.parquet", columns=["source_id", "best_class_name", "phot_g_mean_mag", "n_good", "excluded", "pixel"])
st = st[~st.excluded]
f = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{out}/features/*.parquet")])
st = st.merge(f, on="source_id")
ev = run.events_A(out); ev = ev[(ev.cls == "SOV") & (ev.half == 0)]
st["host"] = st.source_id.isin(ev.source_id)
st["cls"] = st.best_class_name.fillna("none").replace("", "none")
st["faint"] = st.phot_g_mean_mag >= 16
for c in ("ECL", "LPV", "RS", "SOLAR_LIKE"):
    for faint in (False, True):
        s = st[(st.cls == c) & (st.faint == faint)]
        h, n = s[s.host], s[~s.host]
        print(f"{c:10s} faint={faint!s:5s} hosts {len(h):6d} f_low=0 {np.mean(h.f_low == 0):.2f} | others {len(n):8d} f_low=0 {np.mean(n.f_low == 0):.2f} | host rate {len(h) / len(s) * 1e4:.1f}/1e4")
# spatial: does the share of faint ECL among all stars per pixel track the SOV rate per pixel?
g = st.groupby("pixel").agg(n=("source_id", "size"), ecl_faint=("cls", lambda x: 0), hosts=("host", "sum"))
g["ecl_faint"] = st[(st.cls == "ECL") & st.faint].groupby("pixel").size().reindex(g.index).fillna(0)
g = g[g.n >= 200]
g["ecl_share"] = g.ecl_faint / g.n; g["host_rate"] = g.hosts / g.n
print("pixels", len(g), "spearman(ecl_share, host_rate) =", round(g[["ecl_share", "host_rate"]].corr(method="spearman").iloc[0, 1], 3))
q = pd.qcut(g.ecl_share, 5)
print(g.groupby(q, observed=True).agg(ecl_share=("ecl_share", "median"), host_per_1e4=("host_rate", lambda x: 1e4 * x.median())).round(4))
