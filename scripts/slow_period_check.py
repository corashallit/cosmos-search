"""Calibrate the slow period search (cosmos_search.slow.period) on DR3 light curves.

    python scripts/slow_period_check.py --run /home/user/slow/dr3 --detect slow_detect/ --out slow_period/

--detect is the output of scripts/slow_detect_check.py on the same run (its stats.parquet gives each star's G and
colour cell and which stars are quiet, half B). For quiet stars (both halves) and a random SAMPLE share of all
stars the script computes the period-search scores on drift-corrected epochs (the drift table frozen with the
detector settings, prereg/rules/slow-v1/). Into each quiet half-B star it injects one repeating long dip, one
slow cycle (period.FAMILIES) and one single long dip, and scores them with both the period search and the
detector.

Reported: s_rep against its mirror s_rep_bright for quiet stars; thresholds at 1% and 0.1% of quiet half-B
stars' repeated brightenings, per magnitude range; recall of the period search alone, of the detector alone and
of both together, by signal-to-noise and by cycles in the span; how often a single long dip is called repeating;
how many of all stars pass.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cosmos_search.inject import lightcurve as inj
from cosmos_search.slow import detect, drift, period, rule

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slow_detect_check import BAND_NAMES, corrected, depth_g, scored

SEED = 20261024
SAMPLE = 0.05
FALSE_ALARM = ("0.01", "0.001")
INJECT = period.FAMILIES + ("long_dip",)

_STATE = {}


def injected(df, hosts, file_index):
    out, truth = [], []
    base = -(file_index + 1) * 10**7
    for sid, d in df[df["source_id"].isin(hosts)].groupby("source_id", sort=False):
        tg = np.sort(d.loc[d["band"] == "G", "t"].to_numpy())
        if len(tg) < detect.MIN_VISITS:
            continue
        tv, _ = detect.visits(tg, tg)
        for k, fam in enumerate(INJECT):
            rng = np.random.default_rng([SEED, int(sid) & 0xFFFFFFFF, int(sid) >> 32, k])
            sig = inj.sample(rng, fam, float(tg.min()), float(tg.max()))
            flux, err = d["flux"].to_numpy().copy(), d["err"].to_numpy().copy()
            for b in BAND_NAMES:
                m = (d["band"] == b).to_numpy()
                flux[m], err[m] = inj.inject(d["t"].to_numpy()[m], flux[m], err[m], b, sig)
            fid = base - len(truth)
            out.append(d.assign(source_id=fid, flux=flux, err=err))
            dep, nin = depth_g(sig, tv)
            p = sig.period if fam == "slow_cycle" else (np.diff([x.t0 for x in sig.parts]).mean()
                                                        if fam == "repeat_long_dip" else np.nan)
            truth.append({"source_id": fid, "host": int(sid), "family": fam, "depth_g": dep, "visits_in": nin,
                          "true_period": float(p), "span": float(tg.max() - tg.min())})
    if not out:
        return None, pd.DataFrame()
    fake = pd.concat(out, ignore_index=True)
    fake["band"] = fake["band"].map({b: i for i, b in enumerate(BAND_NAMES)}).astype(np.int8)
    return fake, pd.DataFrame(truth)


def one_file(job):
    file_index, name = job
    parts = _STATE["parts"]
    done = (parts / f"stars_{name}", parts / f"fakes_{name}")
    if all(p.exists() for p in done):
        return pd.read_parquet(done[0]), pd.read_parquet(done[1])
    cells, table, quiet, hosts = _STATE["cells"], _STATE["table"], _STATE["quiet"], _STATE["hosts"]
    lc = pq.read_table(Path(_STATE["run"]) / "lightcurves" / name)
    sid = np.asarray(lc["source_id"])
    pick = np.isin(sid, list(quiet)) | (drift.features.control_mask(sid, SAMPLE).astype(bool))
    lc = lc.filter(pick)
    stars = period.star_features(corrected(lc, cells, table))
    df = lc.to_pandas()
    df["band"] = df["band"].astype(str)
    fake, truth = injected(df, hosts, file_index)
    if fake is not None:
        fcells = cells.reindex(truth["host"].to_numpy())
        fcells.index = pd.Index(truth["source_id"].to_numpy(), name="source_id")
        ep = corrected(fake, pd.concat([cells, fcells]), table)
        truth = truth.merge(period.star_features(ep), on="source_id", how="left").merge(
            scored(ep)[["source_id", "s_dim", "s_bright", "slope_z", "abbe"]], on="source_id", how="left")
    stars.to_parquet(done[0], index=False)
    truth.to_parquet(done[1], index=False)
    return stars, truth


def detector_flag(d, thr):
    th = pd.DataFrame(thr).T.reindex(rule.mag_group(d["gbin"]))
    return (d["s_dim"].to_numpy() > th["s"].to_numpy()) | (d["slope_z"].to_numpy() < -th["z"].to_numpy())


def thresholds(quiet):
    return {g: {fa: float(np.nanquantile(d["s_rep_bright"], 1 - float(fa))) for fa in FALSE_ALARM}
            for g, d in quiet.groupby(rule.mag_group(quiet["gbin"]))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--detect", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    st = pd.read_parquet(Path(a.detect) / "stats.parquet")
    names = sorted(p.name for p in (Path(a.run) / "lightcurves").glob("*.parquet"))[: a.limit or None]
    feat = pd.concat([pq.read_table(Path(a.run) / "features" / n,
                                    columns=["source_id", "n_g", "sigma_rob_g", "err_med_g", "min_z_rob",
                                             "max_z_rob"]).to_pandas() for n in names], ignore_index=True)
    quiet = feat["source_id"].to_numpy()[drift.reference_mask(feat)]
    _, table = rule.load()
    _STATE.update(run=a.run, parts=out / "parts", cells=st.set_index("source_id")[["gbin", "cbin"]], table=table,
                  quiet=set(map(int, quiet)), hosts=set(map(int, quiet[drift.half(quiet) == 1])))
    with Pool(a.workers) as pool:
        res = pool.map(one_file, list(enumerate(names)), chunksize=1)
    stars = pd.concat([r[0] for r in res], ignore_index=True).merge(
        st[["source_id", "gbin", "s_dim", "s_bright", "slope_z", "abbe", "sigma_visit"]], on="source_id", how="left")
    stars["quiet"] = stars["source_id"].isin(_STATE["quiet"])
    stars["half_b"] = drift.half(stars["source_id"].to_numpy()) == 1
    fakes = pd.concat([r[1] for r in res if len(r[1])], ignore_index=True)
    stars.to_parquet(out / "stars.parquet", index=False)
    fakes.to_parquet(out / "fakes.parquet", index=False)

    qb = stars[stars["quiet"] & stars["half_b"]]
    qa = stars[stars["quiet"] & ~stars["half_b"]]
    thr = thresholds(qb)
    v1, _ = rule.load()
    summary = {"files": len(names), "quiet_b": len(qb), "quiet_a": len(qa), "sampled": int((~stars["quiet"]).sum()),
               "thresholds": thr}
    for label, d in (("quiet_a", qa), ("sample_all", stars[~stars["quiet"]])):
        th = pd.DataFrame(thr).T.reindex(rule.mag_group(d["gbin"]))
        for fa in FALSE_ALARM:
            summary[f"{label}_pass_{fa}"] = {"n": len(d), "dim": int((d["s_rep"] > th[fa].to_numpy()).sum()),
                                             "bright": int((d["s_rep_bright"] > th[fa].to_numpy()).sum())}
    if len(stars[~stars["quiet"]]):
        d = stars[~stars["quiet"] & (stars["gbin"] <= rule.FAINT_GBIN - 1)]
        th = pd.DataFrame(thr).T.reindex(rule.mag_group(d["gbin"]))
        old = detector_flag(d, v1)
        new = d["s_rep"].to_numpy() > th["0.001"].to_numpy()
        summary["sample_g_lt_19"] = {"n": len(d), "detector": int(old.sum()), "period": int(new.sum()),
                                     "period_only": int((new & ~old).sum())}

    hq = qb.set_index("source_id")[["gbin", "sigma_visit"]].rename(columns={"sigma_visit": "host_sigma_visit"})
    f = fakes.join(hq, on="host")
    f = f[f["gbin"] <= rule.FAINT_GBIN - 1].copy()
    f["snr"] = f["depth_g"] * np.sqrt(np.maximum(f["visits_in"], 1)) / f["host_sigma_visit"]
    f["cycles"] = f["span"] / f["true_period"]
    f["detector"] = detector_flag(f, v1)
    rows = []
    for fa in FALSE_ALARM:
        f["period"] = f["s_rep"] > thr["G<19"][fa]
        f["either"] = f["period"] | f["detector"]
        for by, bins in (("snr", [0, 5, 10, 20, 40, np.inf]), ("cycles", [0, 1.5, 2, 3, 5, 100])):
            for (fam, b), d in f.groupby(["family", pd.cut(f[by], bins)], observed=True):
                rows.append({"false_alarm": fa, "family": fam, "by": by, "bin": str(b), "n": len(d),
                             **{k: round(float(d[k].mean()), 3) for k in ("period", "detector", "either")}})
    rc = pd.DataFrame(rows)
    rc.to_csv(out / "recall.csv", index=False)
    ld = f[f["family"] == "long_dip"]
    summary["long_dip_called_repeating_0.001"] = float((ld["s_rep"] > thr["G<19"]["0.001"]).mean())
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(rc[(rc["false_alarm"] == "0.001")].to_string(index=False))


if __name__ == "__main__":
    main()
