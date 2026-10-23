"""Compare ways of scoring slow fades on the output of scripts/slow_detect_check.py, at the same false-alarm rate.

    python scripts/slow_fade_compare.py --run /home/user/slow/dr3 --detect slow_detect/ [--detect-sky slow_detect_sky/] \\
        --out compare.json

Variants: the frozen median-baseline episode score (s_dim, mirror s_bright) and the time-ordered one (s_fade,
mirror s_rise), each with the fading trend (slope_z); with --detect-sky, the same on sky-corrected light curves.
For each, thresholds are the 0.1% points of the mirror among all quiet stars of the run (both halves: half B
alone leaves ~6 stars beyond the point), per magnitude range. Reported: those thresholds; the share of quiet
stars of each half, and of all stars, flagged as fading and as brightening; recall of the fakes by family and
signal-to-noise; how often a late-starting brightening is flagged as a fade; and, with --detect-sky, how much of
the sky pattern in quiet stars' trends is left (variance of pixel medians of slope_z over the variance that
shuffling stars between pixels gives).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cosmos_search.slow import drift, sky
from cosmos_search.slow.rule import mag_group

SHARE = 0.001
SHARED_LEVEL = 6
# name: (episode score, its mirror). "_shared" variants first drop scores whose change date other stars in the same
# level-6 pixel share (sky.shared_changes), on both sides.
VARIANTS = {"median": ("s_dim", "s_bright"), "time_order": ("s_fade", "s_rise"),
            "time_order_shared": ("s_fade_ns", "s_rise_ns")}
SNR_BINS = [0, 5, 10, 20, 40, np.inf]


def quiet_ids(run):
    names = sorted(p.name for p in (Path(run) / "features").glob("*.parquet"))
    feat = pd.concat([pq.read_table(Path(run) / "features" / n,
                                    columns=["source_id", "n_g", "sigma_rob_g", "err_med_g", "min_z_rob",
                                             "max_z_rob"]).to_pandas() for n in names], ignore_index=True)
    return feat["source_id"].to_numpy()[drift.reference_mask(feat)]


def flags(d, thr, ep, ep_mirror, mirror=False):
    th = pd.DataFrame(thr).T.reindex(mag_group(d["gbin"]))
    if mirror:
        return (d[ep_mirror].to_numpy() > th["s"].to_numpy()) | (d["slope_z"].to_numpy() > th["z"].to_numpy())
    return (d[ep].to_numpy() > th["s"].to_numpy()) | (d["slope_z"].to_numpy() < -th["z"].to_numpy())


def sky_ratio(q, level=5, min_stars=10, n_shuffle=200, seed=0):
    """Variance of per-pixel median slope_z over its mean under shuffling stars between pixels."""
    q = q[np.isfinite(q["slope_z"])].assign(pix=drift.sky_pixel(q["source_id"].to_numpy(), level))
    n = q.groupby("pix")["slope_z"].transform("size")
    q = q[n >= min_stars]
    real = q.groupby("pix")["slope_z"].median().var()
    rng = np.random.default_rng(seed)
    z = q["slope_z"].to_numpy()
    sh = [pd.Series(rng.permutation(z)).groupby(q["pix"].to_numpy()).median().var() for _ in range(n_shuffle)]
    return {"pixels": int(q["pix"].nunique()), "stars": int(len(q)), "ratio": float(real / np.mean(sh)),
            "p": float(np.mean(np.array(sh) >= real))}


def add_shared(stats, fakes):
    """s_fade_ns / s_rise_ns: s_fade / s_rise, NaN where the change date is shared with other stars nearby. Fakes
    are tested against the real stars of their host's pixel."""
    pix = drift.sky_pixel(stats["source_id"].to_numpy(), SHARED_LEVEL)
    sh = sky.shared_changes(stats, pix)
    fsh = sky.shared_changes(stats, pix, fakes, drift.sky_pixel(fakes["host"].to_numpy(), SHARED_LEVEL))
    for d, x in ((stats, sh), (fakes, fsh)):
        d["s_fade_ns"] = np.where(x["shared_fade"].to_numpy(), np.nan, d["s_fade"].to_numpy())
        d["s_rise_ns"] = np.where(x["shared_rise"].to_numpy(), np.nan, d["s_rise"].to_numpy())
        d["shared_fade"], d["shared_rise"] = x["shared_fade"].to_numpy(), x["shared_rise"].to_numpy()
    return stats, fakes


def evaluate(stats, fakes, quiet, ep, ep_mirror):
    stats = stats.copy()
    q = stats[stats["source_id"].isin(quiet)]
    thr = {}
    for g, d in q.groupby(mag_group(q["gbin"])):
        # NaN (a vetoed score) counts as not passing: quantile over all quiet stars, vetoed ones at -inf.
        thr[g] = {"s": float(np.quantile(np.nan_to_num(d[ep_mirror].to_numpy(), nan=-np.inf), 1 - SHARE)),
                  "z": float(np.nanquantile(d["slope_z"], 1 - SHARE)), "quiet": int(len(d))}
    out = {"thresholds": thr, "flagged": {}}
    half = drift.half(q["source_id"].to_numpy())
    for label, d in (("quiet_half_a", q[half == 0]), ("quiet_half_b", q[half == 1]), ("all", stats)):
        for g, dd in d.groupby(mag_group(d["gbin"])):
            out["flagged"][f"{label} {g}"] = {"n": int(len(dd)), "fading": int(flags(dd, thr, ep, ep_mirror).sum()),
                                              "brightening": int(flags(dd, thr, ep, ep_mirror, True).sum())}
    f = fakes[np.isfinite(fakes[ep])].copy()
    f = f[mag_group(f["gbin"]) == "G<19"]
    f["found"] = flags(f, thr, ep, ep_mirror)
    f["by_episode"] = f[ep].to_numpy() > thr["G<19"]["s"]
    f["snr_bin"] = pd.cut(f["snr"], SNR_BINS).astype(str)
    out["recall"] = {fam: {b: {"n": int(len(dd)), "found": round(float(dd["found"].mean()), 3)}
                           for b, dd in d.groupby("snr_bin")} | {"snr>=20": round(float(d.loc[d["snr"] >= 20, "found"].mean()), 3)}
                     for fam, d in f.groupby("family")}
    out["found_ids"] = f.loc[f["found"], "source_id"].astype(int).tolist()
    if "rise_kind" in f and ep_mirror.startswith("s_rise"):
        # Second list: faint at the first visit, normal later ("jump"): a brightening or a dip already under way.
        def jump(d):
            th = pd.DataFrame(thr).T.reindex(mag_group(d["gbin"]))["s"].to_numpy()
            return (d[ep_mirror].to_numpy() > th) & (d["rise_kind"].to_numpy() == "jump")

        def drop(d):
            th = pd.DataFrame(thr).T.reindex(mag_group(d["gbin"]))["s"].to_numpy()
            return (d[ep].to_numpy() > th) & (d["fade_kind"].to_numpy() == "drop")
        out["jump_list"] = {"quiet": int(jump(q).sum()), "quiet_mirror_drops": int(drop(q).sum()),
                            "all": int(jump(stats).sum()), "all_mirror_drops": int(drop(stats).sum())}
        f["either"] = f["found"] | jump(f)
        out["recall_with_jump_list"] = {fam: round(float(d.loc[d["snr"] >= 20, "either"].mean()), 3)
                                        for fam, d in f.groupby("family")}
    return out


def load(detect, quiet):
    stats = pd.read_parquet(Path(detect) / "stats.parquet")
    fakes = pd.read_parquet(Path(detect) / "fakes.parquet")
    hq = stats.set_index("source_id")[["gbin", "sigma_visit"]].rename(columns={"sigma_visit": "host_sigma_visit"})
    fakes = fakes.drop(columns=[c for c in ("gbin",) if c in fakes]).join(hq, on="host")
    fakes["snr"] = fakes["depth_g"] * np.sqrt(np.maximum(fakes["visits_in"], 1)) / fakes["host_sigma_visit"]
    return stats, fakes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--detect", required=True)
    ap.add_argument("--detect-sky")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    quiet = quiet_ids(a.run)
    res = {}
    for tag, path in (("", a.detect), ("sky ", a.detect_sky)):
        if not path:
            continue
        stats, fakes = load(path, quiet)
        if "rise_t0" in stats:
            stats, fakes = add_shared(stats, fakes)
        res[f"{tag}sky_pattern"] = sky_ratio(stats[stats["source_id"].isin(quiet)])
        for name, (ep, mir) in VARIANTS.items():
            if ep in stats:
                res[f"{tag}{name}"] = evaluate(stats, fakes, quiet, ep, mir)
    found = {k: set(v.pop("found_ids")) for k, v in res.items() if isinstance(v, dict) and "found_ids" in v}
    keys = list(found)
    res["found_overlap"] = {f"{x} vs {y}": {"only_first": len(found[x] - found[y]), "only_second": len(found[y] - found[x])}
                            for i, x in enumerate(keys) for y in keys[i + 1:]}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
