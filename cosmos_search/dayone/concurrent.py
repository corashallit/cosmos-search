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
    a = ap.parse_args(argv)
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
