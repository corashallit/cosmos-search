"""Which light curves the day-one stream keeps, and how to fill the gaps afterwards.

    python -m cosmos_search.dayone.keep table --q g3/q.parquet --out keep_table.json [--margin 0.6]
    python -m cosmos_search.dayone.keep missing --q g3/q.parquet --lightcurves stream/lightcurves --out missing.parquet

The stream stores features for every source but light curves only for a few, because DR4's light curves do not
fit (docs/storage.md). Ranking and vetting need light curves for G3's selections, so the keep rule
(`features.keep_light_curve(table=...)`) keeps a source when its G3 |W| reaches a margin times the |W| at which
G3 selected on DR3, per G bin and statistic. On DR3 at margin 0.6 that is 0.37% of sources and every selection,
against 8.25% for the rehearsal's |z| >= 8 rule (docs/dayone.md). DR4's thresholds will differ (more epochs,
other noise), so after G3 runs on DR4 `missing` lists the selections without a light curve, for a DataLink pull
(`dayone.datalink --select`).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

STATS = ("single", "fov")


def _g_bin(g_mag):
    return np.clip(np.floor(np.nan_to_num(np.asarray(g_mag, float), nan=21.0)), 8, 21).astype(int)


def selected(q, level=0.1):
    """Boolean mask of G3 dim-side selections in any statistic."""
    m = np.zeros(len(q), bool)
    for s in STATS:
        m |= ((q[f"q_{s}"] <= level) & (q[f"w_{s}"] > 0)).to_numpy()
    return m


def table(q, level=0.1, margin=0.6):
    """Per statistic and G bin: margin x the smallest |W| G3 selected at q <= level. Bins with no selection are
    left out, so the keep rule keeps nothing there beyond the controls."""
    out = {"level": level, "margin": margin}
    g_bin = _g_bin(q["g_mag"])
    for s in STATS:
        sel = ((q[f"q_{s}"] <= level) & (q[f"w_{s}"] > 0)).to_numpy()
        t = pd.Series(q[f"w_{s}"].to_numpy()[sel], index=g_bin[sel]).groupby(level=0).min()
        # q stores W as float32; shave a ppm so the float64 W in the stream still reaches its own threshold.
        out[s] = {str(int(b)): float(margin * v) * (1 - 1e-6) for b, v in t.items()}
    return out


def missing(q, lc_dir, level=0.1):
    """source_ids G3 selected that have no kept light curve."""
    ids = q["source_id"].to_numpy()[selected(q, level)]
    have = ds.dataset(lc_dir).to_table(columns=["source_id"]).column("source_id").to_numpy()
    return np.setdiff1d(ids, np.unique(have))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.dayone.keep")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--q", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--level", type=float, default=0.1)
    t.add_argument("--margin", type=float, default=0.6)
    m = sub.add_parser("missing")
    m.add_argument("--q", required=True)
    m.add_argument("--lightcurves", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--level", type=float, default=0.1)
    a = ap.parse_args(argv)
    cols = ["source_id", "g_mag"] + [f"{p}_{s}" for s in STATS for p in ("w", "q")]
    q = pq.read_table(a.q, columns=cols).to_pandas()
    if a.cmd == "table":
        Path(a.out).write_text(json.dumps(table(q, a.level, a.margin), indent=1))
        print(f"wrote {a.out}")
    else:
        ids = missing(q, a.lightcurves, a.level)
        # In the prefilter's shape, so dayone.datalink --select can pull them.
        pd.DataFrame({"source_id": ids, "d_faint": 0.0, "d_bright": 0.0, "control": False}).to_parquet(a.out)
        print(f"{len(ids)} selections without a light curve -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
