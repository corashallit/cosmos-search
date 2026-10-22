"""Activity score from the Gaia G light curve alone (review R17, docs/arms/residual.md).

The recurrence flag fires on about two thirds of testable spotted stars, because a spotted star's low states are
aperiodic dips. Dip timing cannot tell those apart from occulters (the residual search's rotation prototype tried).
What can is the rest of the light curve. An occulter dims a star that is otherwise flat: outside its dips the star
sits on a ceiling set by the noise. A spotted star has no ceiling: it is as often above its median as below it, by
more than the noise. So compare the spread of the bright half of the light curve with the noise:

    activity = log10(s_up / s_pair)

- s_up: the bright half's spread, median(x | x > 0) / 0.6745 with x = f / median(f) - 1. Dips only touch it by
  shifting the median slightly.
- s_pair: the noise, measured on the star's own close pairs of epochs (consecutive G epochs less than PAIR_DT apart:
  Gaia's two fields of view see a star 106.5 min apart). Spots do not change in two hours, so the pair difference is
  noise only. That makes the score self-calibrating: no reliance on the formal errors or on a magnitude model.
  Epochs below -DIP_K s_up are left out of the pairs, so a dip on one epoch of a pair does not count as noise.

No periodicity and no colour: with DR3's few G epochs the BP/RP correlation of out-of-dip variability separated
active stars less well (half A), and a rotation period from ZTF picked up the nightly alias in 60-70% of injected
trains. The score is Gaia-only so ranking can use it before any external fetch.

    python -m cosmos_search.residual.activity run --lc joint.parquet --ctx ctx.parquet --classes v3b_flag_by_class.csv \\
        --half A --out act_A.parquet
    python -m cosmos_search.residual.activity choose act_A.parquet
    python -m cosmos_search.residual.activity report act_B.parquet --threshold 0.4
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

PAIR_DT = 0.1                   # days; consecutive G epochs closer than this are a noise pair
DIP_K = 3.0                     # epochs below -DIP_K * s_up are left out of the pairs
MIN_PAIRS = 5
THRESHOLD = 0.35                # chosen on half A by the pre-registered rule (docs/arms/residual.md)
THRESHOLDS = tuple(np.round(np.arange(0.2, 0.61, 0.05), 2))
SEEDS = {"A": 12169237894723442052, "B": 17678711268131745800}   # the residual search's half seeds
FLIP_MAX = 0.05                 # pre-registered: injection-induced labels at most this share of injections
CAPTURE_MIN = 0.5               # pre-registered: share of real flagged stars labelled active, at least


def score(t, flux):
    """(activity, s_up, s_pair, n_pair) for one band's epochs; activity is NaN with fewer than MIN_PAIRS pairs."""
    t, f = np.asarray(t, float), np.asarray(flux, float)
    ok = np.isfinite(t) & np.isfinite(f) & (f > 0)
    t, f = t[ok], f[ok]
    if len(t) < 2 * MIN_PAIRS:
        return np.nan, np.nan, np.nan, 0
    o = np.argsort(t)
    t, f = t[o], f[o]
    x = f / np.median(f) - 1
    up = x[x > 0]
    s_up = float(np.median(up) / 0.6745) if len(up) else np.nan
    keep = x > -DIP_K * s_up
    close = (np.diff(t) < PAIR_DT) & keep[1:] & keep[:-1]
    d = np.diff(x)[close]
    if len(d) < MIN_PAIRS:
        return np.nan, s_up, np.nan, len(d)
    s_pair = float(1.4826 * np.median(np.abs(d)) / np.sqrt(2))
    a = float(np.log10(s_up / s_pair)) if s_pair > 0 and s_up > 0 else np.nan
    return a, s_up, s_pair, len(d)


def from_lightcurves(lcs, band="G"):
    lc = next((lc for lc in lcs if lc.band == band), None)
    return score(lc.time, lc.flux)[0] if lc is not None else np.nan


# -- calibration ---------------------------------------------------------------------------------------------------

def _job(job):
    from cosmos_search.ebfilter import cal as EC
    from cosmos_search.vetting import recurrence

    kind, family, sid, src, df, seed, _ = job
    g0 = df[df.band == "G"]
    a0 = score(g0.t, g0.flux)[0]
    if kind == "real_class":
        return {"kind": kind, "family": family, "source_id": sid, "activity": a0, "g_mag": src.get("phot_g_mean_mag")}
    rng = np.random.default_rng(seed)
    sig, _ = EC._draw(rng, "family", family, src, df, float(df.t.min()), float(df.t.max()))
    if sig is None:
        return None
    inj = EC._apply(df, sig)
    g = inj[inj.band == "G"]
    t_, d_, z_, s_ = recurrence.epochs(EC._lcs(inj, EC.GAIA_BANDS + EC.ZTF_BANDS), with_sigma=True)
    rec = recurrence.search(t_, d_, z_, sigma=s_)
    return {"kind": kind, "family": family, "source_id": sid, "activity_host": a0, "activity": score(g.t, g.flux)[0],
            "testable": bool(rec["testable"]), "aperiodic": bool(rec.get("aperiodic", False)),
            "g_mag": src.get("phot_g_mean_mag")}


def run(lc, ctx, classes, which, n_family=60, workers=4):
    """Half `which`: every testable real star of the review table (its v3b flag, no injection) and the residual
    search's injections of the target families (`ebfilter.cal.jobs`, same seed and sizes), scored before and after."""
    from cosmos_search.ebfilter import cal as EC

    from . import cal as RC

    js, _ = RC.jobs(lc, ctx, which, 300, n_family, 120, np.random.default_rng(SEEDS[which]), classes,
                    ["anomaly", "real_class"])
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(_job, js, chunksize=4) if r]
    out = pd.DataFrame(rows)
    flag = classes.drop_duplicates("source_id").set_index("source_id")
    real = out.kind.eq("real_class")
    out.loc[real, "testable"] = True
    out.loc[real, "aperiodic"] = flag.reindex(out.loc[real, "source_id"]).aperiodic.astype(bool).to_numpy()
    out["half"] = which
    out["is_target_family"] = out.family.isin(EC.TARGET_FAMILIES)
    return out


def rates(df, th):
    """At threshold th: capture (real flagged stars labelled active), cost (injected flagged stars labelled), its
    host-alone part, J = capture - cost with a 95% Wald interval, flips (injections that cross th upwards) and the
    label rate on all testable real stars."""
    real = df[df.kind.eq("real_class") & df.activity.notna()]
    inj = df[df.kind.eq("anomaly") & df.activity.notna() & df.activity_host.notna()]
    rf, jf = real[real.aperiodic.astype(bool)], inj[inj.aperiodic.astype(bool)]
    cap, cost = float((rf.activity >= th).mean()), float((jf.activity >= th).mean())
    se = np.sqrt(cap * (1 - cap) / len(rf) + cost * (1 - cost) / len(jf))
    flips = float(((inj.activity >= th) & (inj.activity_host < th)).mean())
    return {"threshold": th, "n_real_flagged": len(rf), "capture": cap, "n_injected_flagged": len(jf), "cost": cost,
            "cost_host": float((jf.activity_host >= th).mean()), "J": cap - cost, "J_lo": cap - cost - 1.96 * se,
            "J_hi": cap - cost + 1.96 * se, "n_injected": len(inj), "flips": flips,
            "real_label_rate": float((real.activity >= th).mean())}


def choose(df):
    """Pre-registered (half A): the threshold in THRESHOLDS with the largest J; ties go to the higher threshold."""
    t = pd.DataFrame([rates(df, th) for th in THRESHOLDS])
    best = t.J.max()
    return float(t[np.isclose(t.J, best)].threshold.max()), t


def decision(df, th):
    """Pre-registered (half B): gate the aperiodic flag by activity if J's lower bound > 0, capture >= CAPTURE_MIN
    and flips <= FLIP_MAX."""
    r = rates(df, th)
    r["adopt"] = bool(r["J_lo"] > 0 and r["capture"] >= CAPTURE_MIN and r["flips"] <= FLIP_MAX)
    return r


def by_class(df, th):
    real = df[df.kind.eq("real_class") & df.activity.notna()]
    return real.groupby("family").agg(n=("activity", "size"), labelled=("activity", lambda a: float((a >= th).mean())),
                                      flagged=("aperiodic", "sum"), median=("activity", "median"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.residual.activity")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--lc", required=True)
    r.add_argument("--ctx", required=True)
    r.add_argument("--classes", required=True)
    r.add_argument("--half", required=True, choices=["A", "B"])
    r.add_argument("--n-family", type=int, default=60)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--out", required=True)
    c = sub.add_parser("choose")
    c.add_argument("table")
    p = sub.add_parser("report")
    p.add_argument("table")
    p.add_argument("--threshold", type=float, required=True)
    a = ap.parse_args(argv)
    pd.set_option("display.width", 200)
    if a.cmd == "run":
        lc = pd.read_parquet(a.lc)
        lc["band"] = lc["band"].astype(str)
        out = run(lc, pd.read_parquet(a.ctx), pd.read_csv(a.classes), a.half, a.n_family, a.workers)
        out.to_parquet(a.out, index=False)
        print(f"{len(out)} rows -> {a.out}")
    elif a.cmd == "choose":
        th, t = choose(pd.read_parquet(a.table))
        print(t.round(3).to_string(index=False))
        print(f"chosen threshold: {th}")
    else:
        df = pd.read_parquet(a.table)
        print(pd.Series(decision(df, a.threshold)).to_string())
        print(by_class(df, a.threshold).round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
