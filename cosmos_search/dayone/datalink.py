"""The DataLink channel: the prefilter's selection pulled ≤ 50 sources per call, with the same sealed
injections, Stage 0 cleaning and features as the bulk stream, written one chunk per file so a restart
resumes where it stopped.

    python -m cosmos_search.dayone.datalink --select prefilter.parquet --out dl/ --blend aux_blend.parquet \\
        [--inject plan.json.age] [--limit N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

from cosmos_search.gaia import dr3_pass, epoch, features

CHUNK = epoch.DATALINK_MAX_IDS


def order(select):
    """Pull order for day one: the selection by max(d_faint, d_bright), faint and bright alike so the mirror
    stays fair, with the random controls spread evenly through it, so a pull stopped early still has its
    share of controls for measuring the prefilter's recall."""
    df = pq.read_table(select).to_pandas()
    ctrl = df["control"].to_numpy(dtype=bool)
    sel = df[~ctrl].assign(score=df[["d_faint", "d_bright"]].max(axis=1))
    sel = sel.sort_values("score", ascending=False)["source_id"].astype("int64").tolist()
    controls = df.loc[ctrl, "source_id"].astype("int64").tolist()
    out, step = [], max(1, len(sel) // max(1, len(controls)))
    for i, s in enumerate(sel):
        out.append(s)
        if controls and (i + 1) % step == 0:
            out.append(controls.pop(0))
    return out + controls


def pull(ids, out, blend=None, inject=None, limit=None, log=print):
    out = Path(out)
    (out / "features").mkdir(parents=True, exist_ok=True)
    (out / "lightcurves").mkdir(parents=True, exist_ok=True)
    ids = ids[:limit] if limit else ids
    with (out / "timing.jsonl").open("a") as fh:
        for k in range(0, len(ids), CHUNK):
            name = f"chunk-{k // CHUNK:05d}.parquet"
            if (out / "features" / name).exists():
                continue
            chunk = ids[k:k + CHUNK]
            t0 = time.time()
            lc = epoch.datalink_epoch_photometry(chunk, cache=False)
            t1 = time.time()
            if inject:
                from cosmos_search.inject.classes import gaia_stream
                lc = gaia_stream.apply(lc, dr3_pass.injection_plan(inject))
            bl = dr3_pass.blend_for(lc["source_id"].to_numpy(), blend) if blend else None
            feat = features.source_features(lc, blend=bl)
            pq.write_table(lc, out / "lightcurves" / name, compression="zstd")
            pq.write_table(features.to_arrow(feat), out / "features" / name, compression="zstd")
            rec = {"chunk": k // CHUNK, "requested": len(chunk), "returned": len(feat), "rows": lc.num_rows,
                   "fetch_s": round(t1 - t0, 1), "process_s": round(time.time() - t1, 1)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (k // CHUNK) % 10 == 0:
                log(rec)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.dayone.datalink")
    ap.add_argument("--select", required=True, help="prefilter.parquet from cosmos_search.targets run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blend")
    ap.add_argument("--inject")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    pull(order(a.select), a.out, a.blend, a.inject, a.limit, log=lambda r: print(r, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
