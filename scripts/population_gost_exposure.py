"""DR4 cadence from GOST for the population test (docs/population.md §9).

1. Sample DR3 epoch-photometry stars across the sky; query GOST for their predicted transits over DR4's span.
2. Validate: E_gen from GOST transits on DR3's span (thinned to the measured retention) against E_gen from the
   stars' real DR3 epochs, both unmasked, on the same sigma and threshold.
3. Forecast: E_gen on DR4's span from GOST, per half, against DR3's.
Writes a per-star table and a summary json."""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from cosmos_search.gaia import gost
from cosmos_search.population import analysis, events, exposure, inject

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/home/user/pop/full")
ap.add_argument("--lc", default="/home/user/pop/lc")
ap.add_argument("--work", default="/home/user/pop/dr4null")
ap.add_argument("--n-files", type=int, default=100)
ap.add_argument("--per-file", type=int, default=50)
ap.add_argument("--n-draws", type=int, default=256)
a = ap.parse_args()
work = Path(a.work)
rng = np.random.default_rng(20261022)

# 1. sample: random compact files (HEALPix-8 ranges, so spread over the sky), random stars in each
idx = inject.file_index(a.lc)
stars_all = analysis.load_stars(a.out, ["source_id", "ra", "dec", "excluded", "phot_g_mean_mag", "sigma", "K",
                                        "E_gen_A", "E_gen_B"])
samp_path = work / "gost_sample.parquet"
if samp_path.exists():
    samp = pd.read_parquet(samp_path)
else:
    pick = rng.choice(len(idx), a.n_files, replace=False)
    parts = []
    pix8 = stars_all.source_id.to_numpy() >> 43
    for k in pick:
        lo, hi, _ = idx[k]
        s = stars_all[(pix8 >= lo) & (pix8 <= hi)]
        parts.append(s.sample(min(a.per_file, len(s)), random_state=int(k)))
    samp = pd.concat(parts, ignore_index=True)
    samp.to_parquet(samp_path, index=False)
print("sample", len(samp), flush=True)

tr_path = work / "gost_dr4_transits.parquet"
if tr_path.exists():
    tr = pd.read_parquet(tr_path)
else:
    # 1,000 targets per job: a 4,950-target export (~470k rows) was cut off by the proxy
    parts = []
    for k in range(0, len(samp), 1000):
        part = work / f"gost_part_{k // 1000}.parquet"
        for attempt in range(4):
            if part.exists():
                break
            try:
                gost.query(samp.iloc[k:k + 1000][["source_id", "ra", "dec"]]).to_parquet(part, index=False)
            except Exception as e:
                print(f"GOST chunk {k // 1000} attempt {attempt}: {e!r:.200}", flush=True)
                time.sleep(2 ** (attempt + 2))
        parts.append(pd.read_parquet(part))
    tr = pd.concat(parts, ignore_index=True)
    tr.to_parquet(tr_path, index=False)
print("transits", len(tr), tr.source_id.nunique(), flush=True)

lc = inject.load_light_curves(a.lc, samp.source_id)
rows_real, st_real = events.prepare(lc)
DR3_END = pd.Timestamp(gost.DR3_TO).to_julian_date() - gost.BTJD0   # UTC vs TCB: < 2 min, below the 1-d visit gap


def synth(times, ref):
    """Rows for predicted epochs, carrying each star's real sigma and threshold."""
    tab = pd.DataFrame({"source_id": times.source_id.to_numpy(), "t": times.t.to_numpy(), "flux": 1.0,
                        "err_rel": 0.0, "flags": np.int8(0)})
    rows, st = events.prepare(tab, min_epochs=1)
    r = ref.set_index("source_id").loc[st.source_id]
    st["sigma"], st["K"] = r.sigma.to_numpy(), r.K.to_numpy()
    return rows, st


def e_gen(rows, st, seed):
    eg = exposure.generic(rows, st, n_draws=a.n_draws, rng=np.random.default_rng(seed))
    return pd.DataFrame({"source_id": st.source_id, "A": eg[:, 0], "B": eg[:, 1], "n_ep": st.n_good,
                         "n_vis": st.n_visits})


ref = st_real[["source_id", "sigma", "K"]]
real = e_gen(rows_real, st_real, 1)
obs = pd.DataFrame({"source_id": rows_real["source_id"], "t": rows_real["t"]})
ret = gost.retention(tr[tr.t <= DR3_END], obs)
keep = float(np.median(ret.kept))
have = tr.source_id.isin(st_real.source_id)
g3 = tr[have & (tr.t <= DR3_END)]
g3_thin = gost.thin(g3, keep, np.random.default_rng(2))
g4_thin = gost.thin(tr[have], keep, np.random.default_rng(3))
gd3 = e_gen(*synth(g3_thin, ref), 4)
gd3_full = e_gen(*synth(g3, ref), 5)
gd4 = e_gen(*synth(g4_thin, ref), 6)
df = real.merge(gd3, on="source_id", suffixes=("", "_g3")).merge(gd3_full, on="source_id", suffixes=("", "_g3full")) \
    .merge(gd4, on="source_id", suffixes=("", "_g4")).merge(ret, on="source_id", how="left") \
    .merge(samp[["source_id", "ra", "dec", "phot_g_mean_mag", "E_gen_A", "E_gen_B"]], on="source_id")
df.to_parquet(work / "gost_exposure.parquet", index=False)


def ratio(num, den):
    m = den > 0
    r = num[m] / den[m]
    return {"median": float(np.median(r)), "q16": float(np.quantile(r, 0.16)), "q84": float(np.quantile(r, 0.84)),
            "sum_ratio": float(num.sum() / den.sum()), "n": int(m.sum())}


tot = lambda d, s="": d[f"A{s}"] + d[f"B{s}"]
summary = {
    "n_stars": int(len(df)), "retention_median": keep,
    "retention_by_g": df.groupby(pd.cut(df.phot_g_mean_mag, [0, 13, 15, 17, 19, 21]), observed=True).kept.median().to_dict(),
    "epochs_median": {"dr3_real": float(df.n_ep.median()), "dr3_gost_thin": float(df.n_ep_g3.median()),
                      "dr4_gost_thin": float(df.n_ep_g4.median())},
    "visits_median": {"dr3_real": float(df.n_vis.median()), "dr3_gost_thin": float(df.n_vis_g3.median()),
                      "dr4_gost_thin": float(df.n_vis_g4.median())},
    "validate_gost_dr3_over_real": ratio(tot(df, "_g3"), tot(df)),
    "validate_unthinned_over_real": ratio(tot(df, "_g3full"), tot(df)),
    "check_rerun_over_stored": ratio(tot(df), df.E_gen_A + df.E_gen_B),
    "forecast_dr4_over_dr3_gost": ratio(tot(df, "_g4"), tot(df, "_g3")),
    "forecast_dr4_over_dr3_real": ratio(tot(df, "_g4"), tot(df)),
    "half_balance_dr4": float(df.A_g4.sum() / tot(df, "_g4").sum()),
}
summary["retention_by_g"] = {str(k): float(v) for k, v in summary["retention_by_g"].items()}
json.dump(summary, open(work / "gost_exposure.json", "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
