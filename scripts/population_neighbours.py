"""Do deep dips on faint stars come from neighbours? (docs/population.md §14.)

For SOV hosts (both DR3 halves) and controls matched by nside-64 pixel and G (+-0.25 mag) with no event of any
class, count all-Gaia neighbours within 1, 2, 4 and 8 arcsec, brighter than the target + 1 mag and brighter than
the target, through one Gaia TAP upload join per 5,000 rows. A static neighbour field (scan-angle contamination
of the window) predicts hosts with more close, bright neighbours than controls, most for deep dips on faint
stars."""
import argparse, io, os
import numpy as np, pandas as pd
from cosmos_search.population import analysis, run

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/home/user/pop/full")
ap.add_argument("--work", default="/home/user/pop/phi")
ap.add_argument("--max-hosts", type=int, default=8000)
a = ap.parse_args()
rng = np.random.default_rng(20261022)

stars = analysis.load_stars(a.out, ["source_id", "ra", "dec", "pixel", "excluded", "phot_g_mean_mag"])
evA = run.events_A(a.out); mA = np.load(f"{a.out}/mask_A.npy")
evB = pd.DataFrame(run.load_sealed(f"{a.out}/events_B.json.age")); _, mB = run.masks(a.out)
any_event = set(evA.source_id) | set(evB.source_id)
ev = pd.concat([evA[(evA.cls == "SOV") & (evA.half == 0)], evB[(evB.cls == "SOV") & (evB.half == 1)]])
hosts = stars[stars.source_id.isin(ev.source_id)].merge(ev.groupby("source_id").depth.max().rename("depth"), on="source_id")
hosts = hosts.sample(min(a.max_hosts, len(hosts)), random_state=1)
pool = stars[~stars.source_id.isin(any_event)]
pool = pool.assign(gb=(pool.phot_g_mean_mag * 4).round().astype(int))
groups = {k: g.source_id.to_numpy() for k, g in pool.groupby(["pixel", "gb"])}
ctrl = []
for h in hosts.itertuples():
    c = groups.get((h.pixel, int(round(h.phot_g_mean_mag * 4))))
    ctrl.append(rng.choice(c) if c is not None and len(c) else -1)
hosts["control_id"] = ctrl
hosts = hosts[hosts.control_id >= 0]
targets = pd.concat([hosts[["source_id", "ra", "dec", "phot_g_mean_mag"]].assign(role="host"),
                     stars.set_index("source_id").loc[hosts.control_id, ["ra", "dec", "phot_g_mean_mag"]]
                     .reset_index().assign(role="control")], ignore_index=True)
print("targets", targets.role.value_counts().to_dict(), flush=True)

from astroquery.gaia import Gaia
from astropy.table import Table
Gaia.login(user=os.environ["GAIA_USER"], password=os.environ["GAIA_PASSWORD"])
rows = []
for k in range(0, len(targets), 5000):
    part = targets.iloc[k:k + 5000]
    path = f"{a.work}/nb_part_{k // 5000}.parquet"
    if os.path.exists(path):
        rows.append(pd.read_parquet(path)); continue
    up = Table.from_pandas(part.rename(columns={"phot_g_mean_mag": "g"})[["source_id", "ra", "dec", "g"]])
    q = """SELECT u.source_id AS target, g.phot_g_mean_mag AS g_n,
             DISTANCE(POINT(u.ra, u.dec), POINT(g.ra, g.dec)) * 3600 AS sep
           FROM tap_upload.t AS u JOIN gaiadr3.gaia_source AS g
             ON 1 = CONTAINS(POINT(g.ra, g.dec), CIRCLE(u.ra, u.dec, 8.0 / 3600))
           WHERE g.source_id <> u.source_id"""
    job = Gaia.launch_job_async(q, upload_resource=up, upload_table_name="t", verbose=False)
    df = job.get_results().to_pandas()
    df.to_parquet(path, index=False)
    rows.append(df)
    print("chunk", k // 5000, len(df), flush=True)
nb = pd.concat(rows, ignore_index=True).merge(targets, left_on="target", right_on="source_id")
out = []
for (role, faint), grp in targets.assign(faint=targets.phot_g_mean_mag >= 16).groupby(["role", "faint"]):
    n = len(grp)
    sub = nb[nb.target.isin(grp.source_id)]
    row = {"role": role, "faint": bool(faint), "n": n}
    for r in (1, 2, 4, 8):
        for lab, cut in (("any", 99), ("brighter+1", 1), ("brighter", 0)):
            m = (sub.sep < r) & (sub.g_n.fillna(99) < sub.phot_g_mean_mag + cut) if cut < 99 else (sub.sep < r)
            row[f"<{r}\"_{lab}"] = float(sub[m].target.nunique() / n)
    out.append(row)
res = pd.DataFrame(out)
print(res.round(3).to_string(), flush=True)
res.to_csv(f"{a.work}/neighbours.csv", index=False)
# deep vs shallow hosts
h = hosts.set_index("source_id")
for lab, sel in (("deep", h.depth >= 0.05), ("shallow", h.depth < 0.05)):
    ids = h.index[sel]; cids = h.control_id[sel]
    for who, idx in (("host", ids), ("control", cids)):
        sub = nb[nb.target.isin(idx)]
        f = lambda r, cut: sub[(sub.sep < r) & (sub.g_n.fillna(99) < sub.phot_g_mean_mag + cut)].target.nunique() / len(idx)
        print(lab, who, len(idx), "<2\" brighter+1:", round(f(2, 1), 4), "<4\":", round(f(4, 1), 4), flush=True)
print("done", flush=True)
