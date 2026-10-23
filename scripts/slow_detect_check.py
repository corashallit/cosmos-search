"""Check the slow-change detector (cosmos_search.slow.detect) on DR3 light curves.

    python scripts/slow_detect_check.py --run /home/user/slow/dr3 \\
        --aux /mnt/project-files/data/gaia-dr3-epoch/aux/aux_gaia_source.parquet \\
        --drift slow_check/drift_table.parquet --known known.parquet --alert-lcs alert_lcs.parquet \\
        --out slow_detect/

--run is a dr3_pass output directory that kept every light curve; --drift is the table that
scripts/slow_drift_check.py fitted on half A of the quiet stars. For every star in the run the script computes
the detector's statistics on drift-corrected epochs. Into the quiet stars of half B (never used to fit the drift)
it injects fake long dips, late-starting fades and steady fades (inject.lightcurve families long_dip, ramp_fade,
secular_fade), one of each per star, and computes the same statistics.

Reported:
- fades against brightenings beyond each threshold, per G bin, for all stars and for half-B quiet stars: the
  brightenings are the false-alarm estimate for the fades (docs/statistics.md §2);
- thresholds on s_dim and slope_z set so that a given fraction of uninjected half-B quiet stars brightens past
  them, and the share of fakes found at those thresholds, by depth and duration;
- the known answers' statistics and how many pass. DR3 ends (2017-05-28) before most fading alerts, so the
  alerts are also run on their own Gaia light curves over DR4's span (--alert-lcs), where they count as passing
  at the 0.1% thresholds of their magnitude range.
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cosmos_search.inject import lightcurve as inj
from cosmos_search.slow import detect, drift
from cosmos_search.slow import known as known_mod

FAMILIES = ("long_dip", "ramp_fade", "secular_fade")
BAND_NAMES = ("G", "BP", "RP")
FALSE_ALARM = (0.01, 0.001)
SEED = 20261023
FAINT_GBIN = 8  # drift.G_EDGES[8] = 19

_STATE = {}


def _init(cells, table, hosts):
    _STATE.update(cells=cells, table=table, hosts=hosts)


def corrected(lc_df, cells, table):
    ep = drift.normalise(lc_df)
    ep["frac_corr"] = ep["frac"] - drift.drift_at(ep, cells, table)
    return ep


def depth_g(sig, tv):
    """Largest fractional G deficit the signal reaches, and how many G visits sit below half of it."""
    k = sig.factor(tv, "G")
    d = float(1 - k.min())
    return d, int(np.sum(1 - k > d / 2)) if d > 0 else 0


def injected(lc, hosts, file_index):
    """One fake per family per host: light-curve rows with fresh negative ids, and their truth rows."""
    df = lc.to_pandas()
    df["band"] = df["band"].astype(str)
    df = df[df["source_id"].isin(hosts)]
    out, truth = [], []
    base = -(file_index + 1) * 10**7
    for sid, d in df.groupby("source_id", sort=False):
        tg = np.sort(d.loc[d["band"] == "G", "t"].to_numpy())
        if len(tg) < detect.MIN_VISITS:
            continue
        tv, _ = detect.visits(tg, tg)
        for k, fam in enumerate(FAMILIES):
            rng = np.random.default_rng([SEED, int(sid) & 0xFFFFFFFF, int(sid) >> 32, k])
            sig = inj.sample(rng, fam, float(tg.min()), float(tg.max()))
            flux, err = d["flux"].to_numpy().copy(), d["err"].to_numpy().copy()
            for b in BAND_NAMES:
                m = (d["band"] == b).to_numpy()
                flux[m], err[m] = inj.inject(d["t"].to_numpy()[m], flux[m], err[m], b, sig)
            fid = base - len(truth)
            out.append(d.assign(source_id=fid, flux=flux, err=err))
            dep, nin = depth_g(sig, tv)
            row = {"source_id": fid, "host": int(sid), "family": fam, "depth_g": dep, "visits_in": nin}
            if fam == "long_dip":
                row.update(duration=sig.ingress + sig.flat + sig.egress, dust=bool(np.isfinite(sig.tau_ref)))
            elif fam == "ramp_fade":
                row.update(duration=sig.duration, rate_per_year=sig.fraction / sig.duration * 365.25)
            else:
                row.update(duration=float(tg.max() - tg.min()), rate_per_year=sig.rate_per_year)
            truth.append(row)
    if not out:
        return None, pd.DataFrame()
    fake = pd.concat(out, ignore_index=True)
    fake["band"] = fake["band"].map({b: i for i, b in enumerate(BAND_NAMES)}).astype(np.int8)
    return fake, pd.DataFrame(truth)


def one_file(job):
    file_index, name = job
    run, parts = _STATE["run"], _STATE["parts"]
    done = (parts / f"stats_{name}", parts / f"fakes_{name}")
    if all(p.exists() for p in done):
        return pd.read_parquet(done[0]), pd.read_parquet(done[1])
    cells, table, hosts = _STATE["cells"], _STATE["table"], _STATE["hosts"]
    lc = pq.read_table(Path(run) / "lightcurves" / name)
    feat = detect.star_features(corrected(lc, cells, table))
    fake, truth = injected(lc, hosts, file_index)
    if fake is not None:
        fcells = cells.reindex(truth["host"].to_numpy())
        fcells.index = pd.Index(truth["source_id"].to_numpy(), name="source_id")
        ff = detect.star_features(corrected(fake, pd.concat([cells, fcells]), table))
        truth = truth.merge(ff, on="source_id", how="left")
    feat.to_parquet(done[0], index=False)
    truth.to_parquet(done[1], index=False)
    return feat, truth


def mag_group(gbin):
    """G < 19 and G >= 19: on DR3 the faint quiet stars have a far heavier tail in both directions (blue, slowly
    wandering: mostly quasars), so each range gets its own thresholds."""
    return np.where(np.asarray(gbin) <= FAINT_GBIN - 1, "G<19", "G>=19")


def thresholds(quiet):
    """Per magnitude range: s_dim and slope_z thresholds that the given fractions of quiet stars exceed when
    brightening."""
    out = {}
    for grp, d in quiet.groupby(mag_group(quiet["gbin"])):
        out[grp] = {fa: {"s": float(np.nanquantile(d["s_bright"], 1 - fa)),
                         "z": float(np.nanquantile(d["slope_z"], 1 - fa))} for fa in FALSE_ALARM}
    return out


def passes(df, thr, fa, mirror=False):
    th = pd.DataFrame({g: v[fa] for g, v in thr.items()}).T.reindex(mag_group(df["gbin"]))
    if mirror:
        return ((df["s_bright"].to_numpy() > th["s"].to_numpy()) | (df["slope_z"].to_numpy() > th["z"].to_numpy()))
    return (df["s_dim"].to_numpy() > th["s"].to_numpy()) | (df["slope_z"].to_numpy() < -th["z"].to_numpy())


def mirror_table(f, label):
    rows = []
    for g, d in f.groupby("gbin"):
        for thr in (5, 8, 12):
            rows.append({"sample": label, "g_bin": f"{drift.G_EDGES[g]:g}-{drift.G_EDGES[g + 1]:g}", "n": len(d),
                         "thr": thr, "dim": int((d["s_dim"] > thr).sum()), "bright": int((d["s_bright"] > thr).sum()),
                         "fade_z": int((d["slope_z"] < -thr).sum()), "brighten_z": int((d["slope_z"] > thr).sum())})
    return pd.DataFrame(rows)


def recall(truth, thr):
    """Share of fakes found, by family, depth and duration, and by expected signal-to-noise: depth times the
    square root of the visits inside the event, over the host's own sigma_visit."""
    t = truth[np.isfinite(truth["s_dim"])].copy()
    t["mag"] = mag_group(t["gbin"])
    t["snr"] = t["depth_g"] * np.sqrt(np.maximum(t["visits_in"], 1)) / t["host_sigma_visit"]
    t["depth_bin"] = pd.cut(t["depth_g"], [0, 0.003, 0.01, 0.03, 0.1, 1.0])
    t["dur_bin"] = pd.cut(t["duration"], [0, 60, 200, 600, 5000])
    t["snr_bin"] = pd.cut(t["snr"], [0, 5, 10, 20, 40, np.inf])
    rows = []
    for fa in FALSE_ALARM:
        th = pd.DataFrame({g: v[fa] for g, v in thr.items()}).T.reindex(t["mag"])
        t["found_episode"] = t["s_dim"].to_numpy() > th["s"].to_numpy()
        t["found_trend"] = t["slope_z"].to_numpy() < -th["z"].to_numpy()
        t["found"] = t["found_episode"] | t["found_trend"]
        for by in (["depth_bin", "dur_bin"], ["snr_bin"]):
            for key, d in t.groupby(["mag", "family"] + by, observed=True):
                if len(d) < 10:
                    continue
                row = {"false_alarm": fa, "mag": key[0], "family": key[1], "n": len(d),
                       "episode": round(d["found_episode"].mean(), 3), "trend": round(d["found_trend"].mean(), 3),
                       "either": round(d["found"].mean(), 3)}
                row.update({b: str(v) for b, v in zip(by, key[2:])})
                rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--drift", required=True)
    ap.add_argument("--known")
    ap.add_argument("--alert-lcs", help="fading alerts' Gaia light curves (python -m cosmos_search.slow alert-lcs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="first N bulk files only (smoke test)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in (Path(a.run) / "features").glob("*.parquet"))
    if a.limit:
        names = names[:a.limit]
    feat = pd.concat([pq.read_table(Path(a.run) / "features" / n,
                                    columns=["source_id", "n_g", "sigma_rob_g", "err_med_g", "min_z_rob",
                                             "max_z_rob"]).to_pandas() for n in names], ignore_index=True)
    ref = feat["source_id"].to_numpy()[drift.reference_mask(feat)]
    hosts = ref[drift.half(ref) == 1]
    aux = pd.read_parquet(a.aux, columns=["source_id", "phot_g_mean_mag", "bp_rp"])
    aux = aux[aux["source_id"].isin(feat["source_id"])]
    cells = drift.star_cells(aux)
    table = pd.read_parquet(a.drift)

    _STATE["run"], _STATE["parts"] = a.run, out / "parts"
    _STATE["parts"].mkdir(exist_ok=True)
    with Pool(a.workers, initializer=_init, initargs=(cells, table, set(map(int, hosts)))) as pool:
        res = pool.map(one_file, list(enumerate(names)), chunksize=1)
    stats = pd.concat([r[0] for r in res], ignore_index=True).join(cells, on="source_id")
    truth = pd.concat([r[1] for r in res if len(r[1])], ignore_index=True)
    stats["quiet_b"] = stats["source_id"].isin(hosts)
    stats.to_parquet(out / "stats.parquet", index=False)
    truth.to_parquet(out / "fakes.parquet", index=False)

    quiet = stats[stats["quiet_b"]]
    mt = pd.concat([mirror_table(stats, "all"), mirror_table(quiet, "quiet_half_b")], ignore_index=True)
    mt.to_csv(out / "mirror.csv", index=False)
    print(mt.groupby(["sample", "thr"])[["n", "dim", "bright", "fade_z", "brighten_z"]].sum())

    thr = thresholds(quiet)
    hq = quiet.set_index("source_id")[["gbin", "sigma_visit"]].rename(columns={"sigma_visit": "host_sigma_visit"})
    rc = recall(truth.join(hq, on="host"), thr)
    rc.to_csv(out / "recall.csv", index=False)
    print(json.dumps({g: {str(fa): v for fa, v in d.items()} for g, d in thr.items()}, indent=1))
    print(rc.to_string(index=False))

    summary = {"files": len(names), "stars": len(stats), "quiet_half_b": len(quiet),
               "fakes": len(truth),
               "thresholds": {g: {str(fa): v for fa, v in d.items()} for g, d in thr.items()},
               "abbe_median_quiet": float(quiet["abbe"].median()), "abbe_median_all": float(stats["abbe"].median())}
    for fa in FALSE_ALARM:
        for grp, d in stats.groupby(mag_group(stats["gbin"])):
            summary[f"all_stars_pass_{fa}_{grp}"] = {"stars": len(d),
                                                     "fade_or_episode": int(passes(d, thr, fa).sum()),
                                                     "mirror": int(passes(d, thr, fa, mirror=True).sum())}
    if a.known:
        known = pd.read_parquet(a.known).dropna(subset=["source_id"]).astype({"source_id": "int64"})
        k = known[["name", "kind", "source_id"]].merge(stats, on="source_id")
        k["passes"] = passes(k, thr, FALSE_ALARM[-1])
        k.to_csv(out / "known_stats.csv", index=False)
        summary["known"] = {kind: {"n": len(d), "passes": int(d["passes"].sum()),
                                   "median_s_dim": float(d["s_dim"].median()),
                                   "median_abbe": float(d["abbe"].median())}
                            for kind, d in k.groupby("kind")}
    if a.alert_lcs:
        lcs = pd.read_parquet(a.alert_lcs)
        ep = known_mod.alert_epochs(lcs)
        al = detect.star_features(ep)
        al["name"] = al["source_id"].map(ep.drop_duplicates("source_id").set_index("source_id")["name"])
        gmed = al["name"].map(lcs.groupby("name")["mag"].median())
        al["gbin"] = np.digitize(gmed, drift.G_EDGES[1:-1])
        al["passes"] = passes(al, thr, FALSE_ALARM[-1])
        al["slow"] = al["abbe"] < 0.5
        al.to_csv(out / "alert_stats.csv", index=False)
        summary["alerts"] = {"n": len(al), "passes": int(al["passes"].sum()),
                             "abbe_below_0.5": int(al["slow"].sum()),
                             "passes_abbe_below_0.5": int((al["passes"] & al["slow"]).sum()),
                             "passes_abbe_above_0.5": int((al["passes"] & ~al["slow"]).sum())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
