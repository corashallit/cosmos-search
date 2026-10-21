"""How much concurrent photometry from other surveys raises the share of Gaia DR4 one-look dips that can be tested.

    python -m cosmos_search.dayone.concurrent sample --per-bin 200 --out cz/sample.parquet
    python -m cosmos_search.dayone.concurrent fetch --sample cz/sample.parquet --out cz/
    python -m cosmos_search.dayone.concurrent run --dir cz/ --out cz/grid.parquet

Why: Gaia sees a star in short visits (one transit pair, 106.5 min apart, sometimes a second scan 6 h later)
weeks apart, so most dips it catches are on one visit, and one visit bounds a dip's duration only at 106.5 min,
which an eclipse of almost any period meets (docs/dayone.md, single dips, step back). DR4 doubles the baseline but
keeps the visit pattern. What DR4 adds is overlap with other surveys: ZTF from 2018-03-17 and TESS sectors 1-20
fall inside its span (2014-07-25 to 2020-01-20).

What this measures, per star of a random Gaia sample in bins of G, on DR4's real cadence:

- **Gaia DR4 cadence** from ESA's forecast tool (`gaia.gost`), thinned to the share of predicted transits DR3
  kept. The scanning law repeats nothing on the scale of a release, so DR3's cadence projected forward would be a
  guess; the forecast is the schedule Gaia flew.
- **External epochs** as they are: ZTF from IRSA (catflags 0) and TGLC for the TESS sectors inside DR4's span,
  real fluxes and errors, so their outliers and gaps are the real ones.
- **Injected dips** (trapezoids, ingress a fifth of the duration, grey) over a grid of duration and depth, at
  random times over DR4's span, multiplied into the real external fluxes and into Gaia epochs drawn with DR3's
  median per-transit error at the star's G.

Per injected dip: `caught` (a G epoch >= 5 sigma low, as decoys require), `one_look` (every such epoch within
one transit pair, 0.08 d), and for one-look dips whether the single-event test's gate passes with each external
set (`vetting.single_event`: one event across surveys, >= 2 in-dip epochs incl. G, beyond one transit,
bracketed). The gate is the test's own code on the injected light curves, so a ZTF epoch that the test sets aside
(a lone low epoch) or that splits the event does not count.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .external import GAIA_DR4_SPAN

G_BINS = ((10.0, 13.0), (13.0, 15.0), (15.0, 17.0), (17.0, 19.0), (19.0, 20.5))
ZTF_START = 1194.5          # BTJD of 2018-03-17, the start of ZTF's public survey
SAMPLE_COLS = ("source_id", "ra", "dec", "phot_g_mean_mag", "bp_rp", "parallax", "parallax_error",
               "matched_transits", "phot_g_n_obs", "l", "b", "ecl_lat")


def sample(tap, per_bin=200, bins=G_BINS, seed_index=0):
    """per_bin random gaia_source rows per G bin (random_index order, so the draw is reproducible)."""
    parts = []
    for lo, hi in bins:
        q = (f"SELECT TOP {int(per_bin)} " + ", ".join(SAMPLE_COLS) + " FROM gaiadr3.gaia_source "
             f"WHERE random_index >= {int(seed_index)} AND phot_g_mean_mag >= {lo} AND phot_g_mean_mag < {hi} "
             "ORDER BY random_index")
        df = tap.query(q).to_pandas()
        df["g_bin"] = f"{lo:g}-{hi:g}"
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["source_id"] = out["source_id"].astype("int64")
    return out


def _chunks(done_dir, name):
    done = set()
    for p in sorted(Path(done_dir).glob(f"{name}-*.parquet")):
        done |= set(pd.read_parquet(p, columns=["source_id"])["source_id"].astype("int64"))
    return done


def _fetch_survey(sample_df, out, name, one, workers, chunk=50, log=print):
    """Run one(sid, ra, dec) for every star not already in out/name-*.parquet, in parallel, saving per chunk. A star
    with no data gets a marker row (band 'none') so a rerun does not retry it."""
    from concurrent.futures import ThreadPoolExecutor

    done = _chunks(out, name)
    todo = sample_df[~sample_df["source_id"].isin(done)]
    rows = list(zip(todo["source_id"].astype("int64"), todo["ra"].astype(float), todo["dec"].astype(float)))

    def safe(r):
        try:
            df = one(int(r[0]), r[1], r[2])
        except Exception as exc:
            log(f"{r[0]}: {name} failed ({type(exc).__name__}: {exc})")
            return None
        if df.empty:
            df = pd.DataFrame({"source_id": [int(r[0])], "band": ["none"], "t": [np.nan], "flux": [np.nan],
                               "err": [np.nan]})
        return df

    with ThreadPoolExecutor(workers) as ex:
        for k in range(0, len(rows), chunk):
            got = [d for d in ex.map(safe, rows[k:k + chunk]) if d is not None]
            if got:
                pd.concat(got, ignore_index=True).astype({"source_id": "int64"}).to_parquet(
                    Path(out) / f"{name}-{len(done) + k:06d}.parquet")
            log(f"{name} {min(k + chunk, len(rows))}/{len(rows)}")


def tess_dr4(sid, ra, dec, last_sector=20):
    """TGLC epochs from the TESS sectors inside DR4's span only (sectors 1-20)."""
    from . import external

    secs = [s for s in external.tess_sectors(ra, dec) if s[0] <= last_sector]
    if not secs:
        return external._empty()
    return external.tess(sid, ra, dec, near=np.mean(GAIA_DR4_SPAN), max_sectors=len(secs)) \
        .query(f"t <= {GAIA_DR4_SPAN[1]}")


def fetch(sample_df, out, surveys=("gost", "ztf", "tess"), log=print):
    """GOST DR4 transits, ZTF (IRSA) and DR4-window TESS (TGLC) for the sample."""
    from cosmos_search.gaia import gost

    from . import external

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if "gost" in surveys and not (out / "gost_dr4.parquet").exists():
        gost.transits(sample_df[["source_id", "ra", "dec"]], log=log).to_parquet(out / "gost_dr4.parquet")
    if "ztf" in surveys:
        _fetch_survey(sample_df, out, "ztf", external.ztf, workers=4, log=log)
    if "tess" in surveys:
        _fetch_survey(sample_df, out, "tess", tess_dr4, workers=8, chunk=40, log=log)


DURATIONS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)     # days, full width
DEPTHS = (0.01, 0.03, 0.1, 0.3)
INGRESS_FRACTION = 0.2
CATCH_SIGMA = 5.0
G_ERR_FLOOR = 0.001
EXT_SETS = {"ztf": ("ztf_g", "ztf_r", "ztf_i"), "tess": ("TESS",), "both": ("ztf_g", "ztf_r", "ztf_i", "TESS")}
ZTF_DEC_MIN = -31.0         # ZTF's survey reaches about -30 deg; no star south of this was fetched


def gaia_error(g, table):
    """DR3's median per-transit G flux error (relative) at magnitude g, with a floor."""
    return max(float(np.interp(g, table["gb"] + 0.25, table["err"])), G_ERR_FLOOR)


def keep_fraction(sample_df, gost_dr4, dr3_end=901.8656):
    """Per G bin: DR3 matched transits over GOST's predicted transits on DR3's span (median), capped at 1."""
    n3 = gost_dr4[gost_dr4["t"] <= dr3_end].groupby("source_id").size().rename("n_pred")
    d = sample_df.set_index("source_id").join(n3)
    return (d["matched_transits"] / d["n_pred"]).groupby(d["g_bin"]).median().clip(upper=1.0).to_dict()


def trapezoid(t, t0, duration, depth, ingress=INGRESS_FRACTION):
    """Flux factor of a grey trapezoid dip."""
    tau = ingress * duration
    x = duration / 2 - np.abs(np.asarray(t, float) - t0)
    return 1 - depth * np.clip(x / tau, 0, 1)


def caught_windows(tg, sigma, duration, depth, ingress=INGRESS_FRACTION):
    """Half-width h such that a dip centred within h of a Gaia epoch moves it by >= CATCH_SIGMA sigma (None if the
    dip is too shallow ever to be caught)."""
    need = CATCH_SIGMA * sigma / depth
    if need > 1:
        return None
    return duration / 2 - ingress * duration * need


def draw_caught(tg, h, n, rng, span=GAIA_DR4_SPAN):
    """n dip centres uniform over the times at which a dip is caught (the union of [t_i - h, t_i + h]), and the
    probability that a dip centred uniformly over the span is caught."""
    tg = np.sort(tg)
    lo, hi = tg - h, tg + h
    # Union length of the windows.
    starts, ends = [lo[0]], [hi[0]]
    for a, b in zip(lo[1:], hi[1:]):
        if a <= ends[-1]:
            ends[-1] = max(ends[-1], b)
        else:
            starts.append(a)
            ends.append(b)
    p_caught = float(np.sum(np.array(ends) - np.array(starts)) / (span[1] - span[0]))
    out = []
    while len(out) < n:
        i = rng.integers(len(tg), size=4 * n)
        t0 = tg[i] + rng.uniform(-h, h, size=4 * n)
        cover = np.searchsorted(tg, t0 + h, side="right") - np.searchsorted(tg, t0 - h, side="left")
        out.extend(t0[rng.random(4 * n) < 1 / np.maximum(cover, 1)].tolist())
    return np.array(out[:n]), p_caught


def gate(lcs):
    """The single-event test's gate on these light curves, without its Monte Carlo: (status, details).

    status: 'testable' (one event, >= 2 in-dip epochs incl. G, beyond one transit, bracketed), 'one_look' (one event
    not beyond one transit), 'unbracketed', 'no_g' (the event has no G epoch), 'multi' (more than one event),
    'none' (no in-dip epoch)."""
    from cosmos_search.vetting import single_event as se

    ep = se._band_epochs(lcs, {"TESS": 0.0})
    if ep is None:
        return "none", {}
    event, n_ev = se._event(ep)
    if event is None:
        return ("multi" if n_ev > 1 else "none"), {"n_events": n_ev}
    idx = event["idx"]
    bands = set(ep["band"][idx].tolist())
    n_in = len(idx)
    beyond = bool(event["t_b"] - event["t_a"] > se.ONE_LOOK_DAYS or n_in >= 3)
    bracketed = bool(np.isfinite(event["t_before"]) and np.isfinite(event["t_after"]))
    det = {"n_in": n_in, "bands": ",".join(sorted(bands)), "span": event["t_b"] - event["t_a"],
           "dur_hi": event["t_after"] - event["t_before"]}
    if "G" not in bands:
        return "no_g", det
    if not beyond or n_in < se.MIN_IN_DIP:
        return "one_look", det
    if not bracketed:
        return "unbracketed", det
    return "testable", det


def _star_lcs(ext):
    """Real external light curves of one star: {band: (t, flux, err)}."""
    out = {}
    for band, g in ext.groupby("band"):
        if band == "none":
            continue
        g = g.sort_values("t")
        out[band] = (g["t"].to_numpy(float), g["flux"].to_numpy(float), g["err"].to_numpy(float))
    return out


def run_star(star, tg, ext, err_table, n_draws, rng, durations=DURATIONS, depths=DEPTHS):
    """Rows (one per injected, caught dip) for one star."""
    from cosmos_search.vetting.candidate import LightCurve

    sigma = gaia_error(star["phot_g_mean_mag"], err_table)
    tg = np.sort(np.asarray(tg, float))
    real = _star_lcs(ext)
    rows = []
    for dur in durations:
        for dep in depths:
            h = caught_windows(tg, sigma, dur, dep)
            if h is None or h <= 0 or not len(tg):
                rows.append({"duration": dur, "depth": dep, "p_caught": 0.0, "draw": -1})
                continue
            t0s, p_caught = draw_caught(tg, h, n_draws, rng)
            for k, t0 in enumerate(t0s):
                fg = trapezoid(tg, t0, dur, dep) + sigma * rng.standard_normal(len(tg))
                g_lc = LightCurve("G", tg, fg, np.full(len(tg), sigma))
                status_g, det_g = gate([g_lc])
                row = {"duration": dur, "depth": dep, "p_caught": p_caught, "draw": k, "t0": t0,
                       "gaia": status_g, "n_in_g": det_g.get("n_in", 0)}
                for name, bands in EXT_SETS.items():
                    lcs = [g_lc]
                    n_cov = 0
                    for b in bands:
                        if b not in real:
                            continue
                        t, f, e = real[b]
                        k_ext = trapezoid(t, t0, dur, dep)
                        lcs.append(LightCurve(b, t, f * k_ext, e * np.sqrt(k_ext)))
                        n_cov += int(np.sum(k_ext < 1 - 4 * np.nanmedian(e)))
                    row[f"has_{name}"] = len(lcs) > 1
                    row[f"n_cov_{name}"] = n_cov
                    if len(lcs) == 1:
                        row[name] = status_g
                        continue
                    st, det = gate(lcs)
                    row[name] = st
                    row[f"bands_{name}"] = det.get("bands", "")
                    row[f"dur_hi_{name}"] = det.get("dur_hi", np.nan)
                rows.append(row)
    for r in rows:
        r["source_id"] = int(star["source_id"])
        r["g_bin"] = star["g_bin"]
    return rows


def _load_dir(d):
    d = Path(d)
    sample_df = pd.read_parquet(d / "sample.parquet")
    gost_dr4 = pd.read_parquet(d / "gost_dr4.parquet")
    parts = [pd.read_parquet(p) for p in sorted(d.glob("ztf-*.parquet")) + sorted(d.glob("tess-*.parquet"))]
    ext = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["source_id", "band", "t", "flux",
                                                                                  "err"])
    return sample_df, gost_dr4, ext


def _worker(args):
    star, tg, ext, err_table, n_draws, seed = args
    return run_star(star, tg, ext, err_table, n_draws, np.random.default_rng(seed))


def run(d, err_table, n_draws=30, seed=0, workers=4, log=print):
    """The injection grid over every star in directory d (sample, GOST, ZTF, TESS from `fetch`)."""
    from concurrent.futures import ProcessPoolExecutor

    from cosmos_search.gaia import gost

    sample_df, gost_dr4, ext = _load_dir(d)
    keep = keep_fraction(sample_df, gost_dr4)
    log(f"kept share of predicted transits per G bin: {keep}")
    rng = np.random.default_rng(seed)
    tr = {sid: g for sid, g in gost_dr4.groupby("source_id")}
    ext_by = {sid: g for sid, g in ext.groupby("source_id")}
    empty = ext.iloc[:0]
    jobs = []
    for star in sample_df.to_dict("records"):
        sid = int(star["source_id"])
        g = tr.get(sid)
        if g is None:
            continue
        g = gost.thin(g, keep.get(star["g_bin"], 1.0), rng)
        jobs.append((star, g["t"].to_numpy(), ext_by.get(sid, empty), err_table, n_draws, int(rng.integers(2**31))))
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_worker, jobs, chunksize=4)):
            rows.extend(r)
            if (i + 1) % 50 == 0:
                log(f"run {i + 1}/{len(jobs)}")
    return pd.DataFrame(rows)


def summarise(grid, sample_df=None):
    """Per G bin, duration and depth: P(caught), shares of caught dips that are one-look and testable on Gaia alone,
    and the share of one-look dips each external set makes testable (over all stars, and over stars with data)."""
    caught = grid[grid["draw"] >= 0]
    keys = ["g_bin", "duration", "depth"]
    pc = grid.groupby(keys + ["source_id"])["p_caught"].first().groupby(keys).mean().rename("p_caught")
    s = caught.groupby(keys).agg(n=("draw", "size"),
                                 one_look=("gaia", lambda x: np.mean(x == "one_look")),
                                 testable_gaia=("gaia", lambda x: np.mean(x == "testable")))
    ol = caught[caught["gaia"] == "one_look"]
    for name in EXT_SETS:
        s[f"ol_testable_{name}"] = ol.groupby(keys)[name].apply(lambda x: np.mean(x == "testable"))
        s[f"ol_with_{name}"] = ol.groupby(keys)[f"has_{name}"].mean()
    return pc.to_frame().join(s).reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.dayone.concurrent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--per-bin", type=int, default=200)
    s.add_argument("--out", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--sample", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--surveys", default="gost,ztf,tess")
    r = sub.add_parser("run")
    r.add_argument("--dir", required=True)
    r.add_argument("--gaia-err", required=True, help="csv of gb (G bin start, 0.5 mag), err (median relative error)")
    r.add_argument("--draws", type=int, default=30)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        grid = run(a.dir, pd.read_csv(a.gaia_err), a.draws, a.seed, a.workers, log=lambda m: print(m, flush=True))
        grid.to_parquet(a.out)
        print(summarise(grid).to_string())
    if a.cmd == "fetch":
        fetch(pd.read_parquet(a.sample), a.out, tuple(a.surveys.split(",")), log=lambda m: print(m, flush=True))
    if a.cmd == "sample":
        from cosmos_search.gaia.archive import tap
        df = sample(tap(), a.per_bin)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(a.out)
        print(df.groupby("g_bin").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
