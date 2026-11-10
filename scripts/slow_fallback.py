"""The DataLink fallback list: which stars to fetch first if DR4's bulk light-curve files are late or absent
(cosmos_search.slow.fallback).

    python scripts/slow_fallback.py list --watch watchlist_dr3.parquet [--known known.parquet] --out fallback.parquet
    python scripts/slow_fallback.py abbe --out DIR [--share 0.01] [--g-max 16]             # release day, DR4 archive
    python scripts/slow_fallback.py list --watch ... --known ... --abbe DIR/abbe.parquet --out fallback.parquet
    python scripts/slow_fallback.py check --list fallback.parquet --summary R/summary --out watch_outcome.parquet

`list` (no archive needed): the DR3 watch-list with a prediction per star, then the known answers
(`python -m cosmos_search.slow known`), then the Abbe selection if it exists, as one fetch order. source_id is
DR3's for the first two lists; release day maps them through DR4's dr3_neighbourhood.
`abbe` (release day): sets the Abbe cut per G bin from a random sample of DR4's `all_source_photometry`, then
selects the stars below it over the whole sky in source_id slices (`select.run_selection`).
`check`: once the watch-list's light curves are scored (`slow_search.py pass --lightcurves`), compares each
star's DR4 trend with its prediction (`fallback.outcome`).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.slow import drift, fallback, select, summary


def list_cmd(a):
    watch = pd.read_parquet(a.watch) if a.watch else None
    known = pd.read_parquet(a.known) if a.known else None
    abbe = pd.read_parquet(a.abbe) if a.abbe else None
    out = fallback.ordered(watch, known, abbe)
    out.to_parquet(a.out, index=False)
    counts = out["list"].value_counts().to_dict()
    print(json.dumps({"stars": int(len(out)), "by_list": counts,
                      "days_at_1_per_second": round(len(out) / 86400, 2)}, indent=1))
    if watch is not None:
        w = out[out["list"] == "watch"]
        print(w[["source_id", "phot_g_mean_mag", "slope", "pred_change_since_dr3", "pred_slope_z", "prediction"]]
              .head(5).to_string())


def abbe_cmd(a):
    from cosmos_search.gaia.archive import tap

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, t0 = tap(), time.time()
    smp = t.query(fallback.abbe_sample_adql(a.sample, a.table, g_max=a.g_max), cache=False).to_pandas()
    smp.columns = [c.lower() for c in smp.columns]
    cuts = fallback.abbe_cuts(smp, a.share)
    cuts.to_csv(out / "abbe_cuts.csv", index=False)
    print(f"sample {len(smp)} stars in {time.time() - t0:.0f} s; Abbe cut per G bin:\n{cuts}", flush=True)
    ranges = select.slice_ranges(a.level)[: a.limit or None]
    sel, failed = select.run_selection(t, cuts, ranges, table=a.table, workers=a.workers,
                                       query=lambda c, r: fallback.abbe_adql(c, a.table, source_id_range=r),
                                       log=lambda s: print(s, flush=True))
    sel.columns = [c.lower() for c in sel.columns]
    sel = sel.assign(gbin=drift.star_cells(sel)["gbin"].to_numpy())
    sel = sel.merge(cuts, on="gbin", how="left")
    sel.to_parquet(out / "abbe.parquet", index=False)
    (out / "abbe.json").write_text(json.dumps({"selected": int(len(sel)), "failed": failed, "share": a.share,
                                               "g_max": a.g_max, "seconds": round(time.time() - t0, 1)}, indent=1))
    print(f"{len(sel)} stars selected, {len(failed)} slices failed")


def check_cmd(a):
    lst = pd.read_parquet(a.list)
    w = lst[lst["list"] == "watch"]
    cols = ["source_id", "g_mag", "n_visits", "slope", "slope_err", "s_fade", "fade_kind"]
    s = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(Path(a.summary).glob("*.parquet"))],
                  ignore_index=True)
    s = summary.from_store(s).rename(columns={"slope": "dr4_slope", "slope_err": "dr4_slope_err",
                                              "s_fade": "dr4_s_fade", "fade_kind": "dr4_fade_kind"})
    ids = w["dr4_source_id"] if "dr4_source_id" in w else w["source_id"]
    m = w.assign(source_id_dr4=ids.to_numpy()).merge(
        s[["source_id", "n_visits", "dr4_slope", "dr4_slope_err", "dr4_s_fade", "dr4_fade_kind"]],
        left_on="source_id_dr4", right_on="source_id", how="left", suffixes=("", "_y")).drop(columns="source_id_y")
    m["outcome"] = fallback.outcome(m["pred_slope"], m["dr4_slope"], m["dr4_slope_err"])
    m.to_parquet(a.out, index=False)
    got = m[m["outcome"] != ""]
    print(json.dumps({"watch_list": int(len(w)), "with_dr4_trend": int(len(got)),
                      "outcomes": got["outcome"].value_counts().to_dict()}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list", help="one fetch order from the watch-list, the known answers and the Abbe selection")
    s.add_argument("--watch", help="DR3 watch-list (review-2026-11-10/watchlist_dr3.parquet)")
    s.add_argument("--known", help="known answers (python -m cosmos_search.slow known)")
    s.add_argument("--abbe", help="the abbe step's abbe.parquet")
    s.add_argument("--out", required=True)
    s.set_defaults(func=list_cmd)
    b = sub.add_parser("abbe", help="release day: stars with the lowest G Abbe values, from all_source_photometry")
    b.add_argument("--out", required=True)
    b.add_argument("--share", type=float, default=0.01, help="share of each G bin selected")
    b.add_argument("--g-max", type=float, default=16.0)
    b.add_argument("--sample", type=int, default=2_000_000, help="random stars that set the cuts")
    b.add_argument("--level", type=int, default=3, help="HEALPix level of the source_id slices")
    b.add_argument("--workers", type=int, default=5, help="jobs at once (ESA allows 5 per user)")
    b.add_argument("--table", default="gaiadr4.all_source_photometry")
    b.add_argument("--limit", type=int, default=0, help="first N slices only")
    b.set_defaults(func=abbe_cmd)
    c = sub.add_parser("check", help="DR4 trends of the watch-list against their predictions")
    c.add_argument("--list", required=True)
    c.add_argument("--summary", required=True, help="summary/ of a pass over the watch-list's light curves")
    c.add_argument("--out", required=True)
    c.set_defaults(func=check_cmd)
    a = ap.parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
