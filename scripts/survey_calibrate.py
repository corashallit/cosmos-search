"""Measure what the other surveys can and cannot say about a slow change (cosmos_search.surveys, slow.independent).

    python scripts/survey_calibrate.py sample --n 80 --out C/sample.parquet [--dec-max 10] [--dec-min -90]
    python scripts/survey_calibrate.py fetch  --stars C/sample.parquet --surveys skymapper,noirlab --out C/lc
    python scripts/survey_calibrate.py tess   --stars C/sample.parquet --out C/tess.parquet
    python scripts/survey_calibrate.py floors --stars C/sample.parquet --lc C/lc --out C/floors.json
    python scripts/survey_calibrate.py null   --stars C/sample.parquet --lc C/lc --out C/null
    python scripts/survey_calibrate.py known  --stars W/watch.parquet --lc W/lc --out W/known [--tess W/tess.parquet]

`sample` draws constant stars: random Gaia DR3 sources in four G ranges (11-13, 13-15, 15-17, 17-18.5) that DPAC
did not flag as variable, with RUWE < 1.4, and whose G scatter over DR3 (sqrt(n_obs) / flux_over_error) is below
the median of their half-magnitude. Gaia calls them quiet over 2014-2017; a few percent will still vary, which is
why every summary below uses medians and percentiles.

`fetch` reads each survey for each star (a Parquet or CSV table with source_id, ra, dec and, where known, pmra,
pmdec, ref_epoch) into <out>/<survey>.parquet, a few stars at once; rerunning skips stars already read. `tess` does the
same for `surveys.tess_activity`.

`floors`: the scatter of 180-day bin means on the constant stars beyond their quoted errors, per survey: the
value f for which (bin - group median) / sqrt(err^2 + f^2) has unit robust scatter. These are
`slow.independent.SEASON_FLOOR`.

`null`: on the constant stars, with the floors in place, the comparison with a made-up Gaia trend over DR4's dates
(and separately the trend after them). Reports per survey and G range: the share of stars the survey can test at
all, the median error of the survey's own slope (so 3 times it is the smallest trend it can confirm), the scatter
of k / k_err (1 if the errors are right), and how often a constant star reads "seen" or "opposite" (false
confirmations). Then the same with the made-up trend injected into the epochs, where k should come back as 1.

`known`: on stars with a measured Gaia change (the DR3 fading list or the watch-list, joined with positions),
the comparison itself: k, its error and the verdicts, per star and summarised.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search import surveys
from cosmos_search.slow import independent as ind

G_RANGES = ((11.0, 13.0), (13.0, 15.0), (15.0, 17.0), (17.0, 18.5))
DR4_SPAN = (-136.06, 1868.5)     # BTJD, DR4's first and last epoch
RANDOM_INDEX_MAX = 1_811_709_770


def robust_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def sample(a):
    from cosmos_search.gaia.archive import tap

    rng = np.random.default_rng(a.seed)
    # Share of Gaia sources in each G range, roughly: sets how wide a slice of random_index each query needs.
    share = {G_RANGES[0]: 0.004, G_RANGES[1]: 0.015, G_RANGES[2]: 0.07, G_RANGES[3]: 0.12}
    sky = (np.sin(np.radians(a.dec_max)) - np.sin(np.radians(a.dec_min))) / 2
    parts = []
    for lo, hi in G_RANGES:
        width = int(min(20 * a.n / (share[(lo, hi)] * sky), RANDOM_INDEX_MAX / 4))
        start = int(rng.integers(0, RANDOM_INDEX_MAX - width))
        q = ("SELECT source_id, ra, dec, pmra, pmdec, ref_epoch, phot_g_mean_mag, bp_rp, phot_variable_flag, "
             "phot_g_mean_flux_over_error, phot_g_n_obs, ruwe, b FROM gaiadr3.gaia_source "
             f"WHERE random_index BETWEEN {start} AND {start + width} AND phot_g_mean_mag BETWEEN {lo} AND {hi} "
             f"AND dec BETWEEN {a.dec_min} AND {a.dec_max} AND ruwe < 1.4")
        d = tap().query(q).to_pandas()
        d["scatter"] = np.sqrt(d["phot_g_n_obs"]) / d["phot_g_mean_flux_over_error"]
        half = np.floor(d["phot_g_mean_mag"] * 2) / 2
        quiet = (d["scatter"] < d.groupby(half)["scatter"].transform("median")) & \
            (d["phot_variable_flag"].astype(str) != "VARIABLE") & (d["b"].abs() > a.min_b)
        d = d[quiet]
        parts.append(d.sample(min(a.n, len(d)), random_state=a.seed))
        print(f"G {lo}-{hi}: {len(parts[-1])} of {len(d)} quiet stars drawn", flush=True)
    out = pd.concat(parts, ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"{len(out)} stars -> {a.out}")


def read_stars(path):
    """A star table from Parquet or CSV (the lists under configs/surveys/ are CSV)."""
    path = Path(path)
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def _fetch_one(survey, star):
    if not surveys.covers(survey, star["dec"]):
        return surveys.empty(), "outside"
    ra, dec = surveys.at_year(star["ra"], star["dec"], star.get("pmra"), star.get("pmdec"),
                              star.get("ref_epoch") or 2016.0, surveys.MEAN_YEAR[survey])
    try:
        return surveys.fetch(survey, int(star["source_id"]), ra, dec), ""
    except (surveys.MissingCredential, surveys.Unreachable):
        raise
    except Exception as e:  # noqa: BLE001
        return surveys.empty(), f"{type(e).__name__}: {e}"[:160]


def fetch(a):
    stars = read_stars(a.stars)
    if a.max_g is not None:
        stars = stars[stars["phot_g_mean_mag"] < a.max_g]
    if a.min_g is not None:
        stars = stars[stars["phot_g_mean_mag"] >= a.min_g]
    if a.limit:
        stars = stars.head(a.limit)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for survey in a.surveys.split(","):
        path, log = out / f"{survey}.parquet", out / f"{survey}_log.csv"
        done = pd.read_csv(log) if log.exists() else pd.DataFrame(columns=["source_id", "n", "error"])
        have = [pd.read_parquet(path)] if path.exists() else []
        todo = stars[~stars["source_id"].isin(done["source_id"])]
        print(f"{survey}: {len(todo)} stars to read ({len(done)} done before)", flush=True)
        rows = done.to_dict("records")
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(_fetch_one, survey, s): s for s in todo.to_dict("records")}
            for i, fut in enumerate(as_completed(futs), 1):
                s = futs[fut]
                lc, err = fut.result()
                rows.append({"source_id": s["source_id"], "n": len(lc), "error": err})
                if len(lc):
                    have.append(lc)
                if i % a.save_every == 0 or i == len(futs):
                    pd.DataFrame(rows).to_csv(log, index=False)
                    (pd.concat(have, ignore_index=True) if have else surveys.empty()).to_parquet(path, index=False)
                    print(f"  {survey}: {i}/{len(futs)}", flush=True)
        r = pd.DataFrame(rows)
        print(f"{survey}: {int((r['n'] > 0).sum())} of {len(r)} stars with epochs; errors on "
              f"{int((r['error'].fillna('') != '').sum() - (r['error'] == 'outside').sum())}", flush=True)


def tess(a):
    from cosmos_search.surveys import tess_activity

    stars = read_stars(a.stars)
    if a.max_g is not None:
        stars = stars[stars["phot_g_mean_mag"] < a.max_g]
    if a.limit:
        stars = stars.head(a.limit)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["source_id"])
    todo = stars[~stars["source_id"].isin(done["source_id"])]
    rows = done.to_dict("records")

    def one(s):
        try:
            summ, _ = tess_activity.activity(int(s["source_id"]), s["ra"], s["dec"], a.max_sectors)
            return {"source_id": s["source_id"], **summ, "error": ""}
        except Exception as e:  # noqa: BLE001
            return {"source_id": s["source_id"], "tess_sectors": 0, "error": f"{type(e).__name__}: {e}"[:160]}

    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, todo.to_dict("records")), 1):
            rows.append(r)
            if i % 50 == 0 or i == len(todo):
                pd.DataFrame(rows).to_parquet(out, index=False)
                print(f"  tess: {i}/{len(todo)}", flush=True)
    r = pd.DataFrame(rows)
    print(f"tess: {int((r['tess_sectors'] > 0).sum())} of {len(r)} stars with a TGLC sector")


def load(lc_dir, names=None):
    parts = [pd.read_parquet(p) for p in sorted(Path(lc_dir).glob("*.parquet"))
             if names is None or p.stem in names]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else surveys.empty()


def season_floor(b):
    """f such that (bin - its group's median) / sqrt(err^2 + f^2) has unit robust scatter, over bins of groups
    with at least 3 bins. The group median is itself noisy, which shrinks the residuals by sqrt(1 - 1/n) on
    average; that factor is put back."""
    key = b[["source_id", "survey", "band", "group"]].astype(str).agg("|".join, axis=1)
    n = key.map(key.value_counts())
    med = b.groupby(key)["flux"].transform("median")
    use = (n >= 3).to_numpy()
    resid = (b["flux"] - med).to_numpy()[use] / np.sqrt(1 - 1 / n.to_numpy()[use])
    err = b["err"].to_numpy()[use]
    if use.sum() < 30:
        return np.nan, int(use.sum())
    lo, hi = 0.0, 0.5
    for _ in range(40):
        mid = (lo + hi) / 2
        if robust_sd(resid / np.hypot(err, mid)) > 1:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2), int(use.sum())


def _bins_by_star(lc, floors):
    out = []
    for sid, d in lc.groupby("source_id"):
        b = ind.bins(ind.visits(d), floors=floors)
        b["source_id"] = sid
        out.append(b)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def floors(a):
    stars = read_stars(a.stars)[["source_id", "phot_g_mean_mag"]]
    lc = load(a.lc)
    zero = {s: 0.0 for s in pd.unique(lc["survey"])}
    b = _bins_by_star(lc, zero).merge(stars, on="source_id")
    res = {}
    for s, d in b.groupby("survey"):
        res[s] = {"all": dict(zip(("floor", "bins"), season_floor(d))), "stars": int(d["source_id"].nunique())}
        for lo, hi in G_RANGES:
            m = d["phot_g_mean_mag"].between(lo, hi)
            f, n = season_floor(d[m])
            res[s][f"G{lo:g}-{hi:g}"] = {"floor": f, "bins": n, "median_err": float(d.loc[m, "err"].median())
                                         if m.any() else np.nan}
        print(s, json.dumps(res[s]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


def _floors_arg(a):
    """The season floors to use: `slow.independent.SEASON_FLOOR`, with those of a --floors file (the output of
    `floors`, or a plain {survey: value}) in place of its entries."""
    out = dict(ind.SEASON_FLOOR)
    if getattr(a, "floors", None):
        for s, v in json.loads(Path(a.floors).read_text()).items():
            v = v["all"]["floor"] if isinstance(v, dict) else v
            if v is not None and np.isfinite(v):
                out[s] = float(v)
    return out


def _fake_change(slope, span=DR4_SPAN):
    return {"kind": "trend", "t_first": span[0], "t_last": span[1], "slope": slope, "d_bp": np.nan, "e_bp": np.nan,
            "d_rp": np.nan, "e_rp": np.nan}


def null(a):
    stars = read_stars(a.stars)[["source_id", "phot_g_mean_mag"]]
    lc = load(a.lc)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    change = _fake_change(a.slope)
    fl = _floors_arg(a)
    rows = []
    for sid, d in lc.groupby("source_id"):
        for label, dd in (("null", d), ("injected", d.assign(
                flux=d["flux"] * (1 + a.slope * (d["t"] - DR4_SPAN[0]) / 365.25)))):
            same, aft = ind.same_years(dd, change, floors=fl), ind.after(dd, change, floors=fl)
            for s in (k for k in same if k != "all"):
                rows.append({"source_id": sid, "survey": s, "case": label, "which": "same", **same[s]})
            for s in (k for k in aft if k != "all"):
                rows.append({"source_id": sid, "survey": s, "case": label, "which": "after", "k": aft[s]["ratio"],
                             "k_err": aft[s]["ratio_err"], "n_bins": aft[s]["n_bins"],
                             "verdict": aft[s]["verdict"]})
    r = pd.DataFrame(rows).merge(stars, on="source_id")
    r["slope_err"] = r["k_err"] * abs(a.slope)
    r.to_parquet(out / "null.parquet", index=False)
    n_stars = lc.groupby("survey")["source_id"].nunique()
    summ = []
    for (s, which, case), d in r.groupby(["survey", "which", "case"]):
        for lo, hi in (*G_RANGES, (0, 99)):
            m = d[d["phot_g_mean_mag"].between(lo, hi) & np.isfinite(d["k"])]
            if len(m) < 5:
                continue
            z = (m["k"] - (1 if case == "injected" else 0)) / m["k_err"]
            summ.append({"survey": s, "which": which, "case": case, "G": "all" if hi == 99 else f"{lo:g}-{hi:g}",
                         "stars_tested": len(m), "median_slope_err_pct_yr": 100 * m["slope_err"].median(),
                         "p10_slope_err_pct_yr": 100 * m["slope_err"].quantile(0.1),
                         "median_k": m["k"].median(), "z_robust_sd": robust_sd(z), "z_median": float(z.median()),
                         "beyond_3": float((z.abs() > 3).mean()),
                         **{v: float((m["verdict"] == v).mean()) for v in pd.unique(m["verdict"])}})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    print(f"stars with epochs: {n_stars.to_dict()}")
    print(summ.round(3).to_string(index=False))


def known(a):
    stars = read_stars(a.stars)
    lc = load(a.lc)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tess_rows = pd.read_parquet(a.tess).set_index("source_id") if a.tess else None
    rows = []
    fl = _floors_arg(a)
    by_star = dict(tuple(lc.groupby("source_id")))
    for s in stars.to_dict("records"):
        d = by_star.get(s["source_id"])
        if d is None:
            continue
        change = ind.gaia_change(s)
        row = {"source_id": s["source_id"], "g": s["phot_g_mean_mag"], "kind": change["kind"],
               "total_change": ind.total_change(change), "slope": s.get("slope"), "class": s.get("dpac_class"),
               **ind.columns(ind.same_years(d, change, floors=fl), ind.after(d, change, floors=fl))}
        if tess_rows is not None and s["source_id"] in tess_rows.index:
            t = tess_rows.loc[s["source_id"]]
            row.update(tess_amp=t.get("tess_amp"), tess_sectors=t.get("tess_sectors"),
                       activity_ratio=ind.activity(change, t.get("tess_amp")))
        rows.append(row)
    r = pd.DataFrame(rows)
    r.to_parquet(out / "known.parquet", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"{len(r)} stars with epochs in at least one survey")
    for col in ("same_years", "after"):
        if col in r:
            print(col, r.groupby("kind")[col].value_counts().to_dict())
    for s in [c[:-5] for c in r.columns if c.endswith("_same")]:
        t = r[np.isfinite(r[f"{s}_k"])]
        strong = t[t[f"{s}_k_err"] < 1 / 3]
        print(f"{s}: tested {len(t)}; could see Gaia's change at 3 errors: {len(strong)}; of those seen "
              f"{int((strong[f'{s}_same'] == 'seen').sum())}, not seen {int((strong[f'{s}_same'] == 'not_seen').sum())}, "
              f"opposite {int((strong[f'{s}_same'] == 'opposite').sum())}; median k {strong[f'{s}_k'].median():.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--n", type=int, default=80, help="stars per G range")
    p.add_argument("--out", required=True)
    p.add_argument("--dec-min", type=float, default=-90.0)
    p.add_argument("--dec-max", type=float, default=90.0)
    p.add_argument("--min-b", type=float, default=10.0, help="smallest |galactic latitude|, deg (crowding)")
    p.add_argument("--seed", type=int, default=20261110)
    p.set_defaults(fn=sample)
    p = sub.add_parser("fetch")
    p.add_argument("--stars", required=True)
    p.add_argument("--surveys", required=True, help="comma-separated: " + ",".join(surveys.ALL))
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--min-g", type=float)
    p.add_argument("--max-g", type=float)
    p.add_argument("--save-every", type=int, default=25)
    p.set_defaults(fn=fetch)
    p = sub.add_parser("tess")
    p.add_argument("--stars", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-g", type=float)
    p.add_argument("--max-sectors", type=int, default=6)
    p.set_defaults(fn=tess)
    p = sub.add_parser("floors")
    p.add_argument("--stars", required=True)
    p.add_argument("--lc", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=floors)
    p = sub.add_parser("null")
    p.add_argument("--stars", required=True)
    p.add_argument("--lc", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--slope", type=float, default=-0.02, help="made-up Gaia trend, fraction per year")
    p.add_argument("--floors", help="season floors to use in place of the module's (output of `floors`)")
    p.set_defaults(fn=null)
    p = sub.add_parser("known")
    p.add_argument("--stars", required=True)
    p.add_argument("--lc", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tess")
    p.add_argument("--floors", help="season floors to use in place of the module's (output of `floors`)")
    p.set_defaults(fn=known)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
