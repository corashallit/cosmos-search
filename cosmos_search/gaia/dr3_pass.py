"""Stream all DR3 epoch photometry from the Gaia CDN to per-source features (milestone I2).

    python -m cosmos_search.gaia.dr3_pass --out data/gaia-dr3-epoch --workers 6
    python -m cosmos_search.gaia.dr3_pass --out data/gaia-dr3-epoch --consolidate /mnt/project-files/data/gaia-dr3-epoch

Each bulk file is downloaded to a temp file (MD5-checked), parsed, reduced to features, and deleted. Full
light curves are kept only for sources passing features.keep_light_curve. The run is resumable: files
whose features are already written are skipped.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cosmos_search.net import get_client

from . import epoch, features


def process_file(info, out):
    out = Path(out)
    t0 = time.time()
    feat_path = out / "features" / info["name"].replace(".csv.gz", ".parquet")
    lc_path = out / "lightcurves" / feat_path.name
    for p in (feat_path, lc_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    meta = {b"features_version": features.features_schema_version().encode(), b"source_file": info["name"].encode()}
    tmp_feat, tmp_lc = feat_path.with_suffix(".tmp"), lc_path.with_suffix(".tmp")
    n_src = n_kept = n_rows = 0
    feat_writer = lc_writer = None
    try:
        with get_client().streamed(info["url"], md5=info["md5"]) as path:
            for lc in epoch.iter_epoch_csv(path):
                feat = features.source_features(lc)
                keep = feat["source_id"].to_numpy()[features.keep_light_curve(feat)]
                kept = lc.filter(pc.is_in(lc["source_id"], value_set=pa.array(keep)))
                ft = features.to_arrow(feat)
                if feat_writer is None:
                    feat_writer = pq.ParquetWriter(tmp_feat, ft.schema.with_metadata(meta), compression="zstd")
                    lc_writer = pq.ParquetWriter(tmp_lc, epoch.SCHEMA.with_metadata(meta), compression="zstd")
                feat_writer.write_table(ft.cast(feat_writer.schema))
                lc_writer.write_table(kept.cast(lc_writer.schema))
                n_src, n_kept, n_rows = n_src + len(feat), n_kept + len(keep), n_rows + lc.num_rows
    finally:
        for w in (feat_writer, lc_writer):
            if w is not None:
                w.close()
    os.replace(tmp_lc, lc_path)
    os.replace(tmp_feat, feat_path)  # features last: their presence marks the file done
    return {"name": info["name"], "md5": info["md5"], "sources": n_src, "kept": n_kept,
            "rows": n_rows, "seconds": round(time.time() - t0, 1)}


def run(out, workers, limit=None):
    out = Path(out)
    files = epoch.bulk_files()
    done = {p.name.replace(".parquet", ".csv.gz") for p in (out / "features").glob("*.parquet")}
    todo = [f for f in files if f["name"] not in done][:limit]
    print(f"{len(files)} bulk files, {len(done)} done, {len(todo)} to do", flush=True)
    log = (out / "manifest.jsonl")
    out.mkdir(parents=True, exist_ok=True)
    failures = 0
    with ProcessPoolExecutor(workers) as pool, log.open("a") as fh:
        futures = {pool.submit(process_file, f, out): f for f in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as exc:  # keep going; the file is retried on the next run
                failures += 1
                rec = {"name": futures[fut]["name"], "error": repr(exc)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 20 == 0 or "error" in rec:
                print(f"[{i}/{len(todo)}] {rec}", flush=True)
    return failures


def consolidate(out, dest, per_part=100):
    """Merge per-file outputs into ~per_part-file Parquet parts under dest (e.g. the project file share)."""
    out, dest = Path(out), Path(dest)
    for kind in ("features", "lightcurves"):
        paths = sorted((out / kind).glob("*.parquet"))
        (dest / kind).mkdir(parents=True, exist_ok=True)
        for k in range(0, len(paths), per_part):
            chunk = paths[k:k + per_part]
            first, last = chunk[0].stem.split("_")[1].split("-")[0], chunk[-1].stem.split("-")[-1]
            target = dest / kind / f"{kind}_hpx8_{first}-{last}.parquet"
            tables = [pq.read_table(p) for p in chunk]
            table = pa.concat_tables([t.replace_schema_metadata(None) for t in tables], promote_options="default")
            meta = {b"features_version": features.features_schema_version().encode()}
            pq.write_table(table.replace_schema_metadata(meta), target, compression="zstd", row_group_size=250_000)
            print(f"wrote {target} ({table.num_rows} rows)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--consolidate", metavar="DEST")
    a = ap.parse_args(argv)
    if a.consolidate:
        consolidate(a.out, a.consolidate)
        return 0
    return 1 if run(a.out, a.workers, a.limit) else 0


if __name__ == "__main__":
    sys.exit(main())
