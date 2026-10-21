"""Stream Gaia epoch photometry bulk files to per-source features (milestone I2; the DR4 stream reuses it).

    python -m cosmos_search.gaia.dr3_pass --out data/gaia-dr3-epoch --workers 5
    python -m cosmos_search.gaia.dr3_pass --out /tmp/work --durable /mnt/project-files/data/gaia-dr3-epoch \\
        --shard 0/3 --workers 5
    python -m cosmos_search.gaia.dr3_pass --out data/gaia-dr3-epoch --consolidate /mnt/project-files/data/gaia-dr3-epoch

Each bulk file is downloaded to a temp file (MD5-checked), parsed in chunks, reduced to features, and
deleted. Per file it writes features, the light curves passing features.keep_light_curve, and with --sketch
the per-stratum mirror-null tail sketches (gaia.sketches). Outputs and a done-marker go to --durable when given (local copies are then removed), and files
already marked done there are skipped, so a reclaimed container or a second session resumes cleanly.
--shard i/n takes the files whose stable name hash is i mod n, so sessions split the list without talking.
"""

import argparse
import json
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cosmos_search.net import get_client
from cosmos_search.vetting import stage0

from . import epoch, features, sketches

KINDS = ("features", "lightcurves")


def stem(name):
    return name.replace(".csv.gz", "")


def in_shard(name, shard):
    i, n = shard
    return zlib.crc32(name.encode()) % n == i


def is_done(name, out, durable=None):
    if durable:
        return (Path(durable) / "done" / f"{stem(name)}.json").exists()
    return (Path(out) / "features" / f"{stem(name)}.parquet").exists()


_BLEND = {}


def blend_table(path):
    """Sorted source_id and BP/RP blended-transit fraction arrays from a Parquet of gaia_source's
    stage0.BLEND_COLS (as stage0.blend_fraction: the larger of the BP and RP fractions, NaN if neither is
    known). Loaded once per worker process."""
    if path not in _BLEND:
        t = pq.read_table(path, columns=["source_id", *stage0.BLEND_COLS]).to_pandas()
        fr = []
        for b in ("bp", "rp"):
            n = t[f"phot_{b}_n_obs"].to_numpy(dtype=float, na_value=np.nan)
            k = t[f"phot_{b}_n_blended_transits"].to_numpy(dtype=float, na_value=np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                fr.append(np.where(np.isfinite(n) & (n > 0) & np.isfinite(k), k / n, np.nan))
        sid = t["source_id"].to_numpy()
        order = np.argsort(sid)
        _BLEND[path] = sid[order], np.fmax(*fr)[order]
    return _BLEND[path]


def blend_for(source_ids, path):
    """{source_id: fraction} for the given sources, leaving out those with no known fraction."""
    sid, frac = blend_table(path)
    ids = np.unique(np.asarray(source_ids))
    j = np.clip(np.searchsorted(sid, ids), 0, len(sid) - 1)
    ok = (sid[j] == ids) & np.isfinite(frac[j])
    return dict(zip(ids[ok].tolist(), frac[j][ok].tolist()))


_PLAN = {}


def injection_plan(path):
    """The sealed injection plan (inject.classes.gaia_stream), decrypted in memory once per worker."""
    if path not in _PLAN:
        from cosmos_search.inject.classes import gaia_stream
        _PLAN[path] = gaia_stream.load_plan(path)
    return _PLAN[path]


def process_file(info, out, durable=None, feature_cut=None, sketch=None, blend=None, inject=None, keep=None):
    """Process one bulk file. feature_cut: pandas query string for which feature rows to keep (default all).
    blend: Parquet of gaia_source's stage0.BLEND_COLS; without it Stage 0 leaves quiet sources untested.
    inject: sealed injection plan, applied to every chunk before Stage 0 and features.
    keep: (|z| threshold, control fraction, keep table) for which light curves to keep (features.keep_light_curve).
    sketch: factory for a per-file accumulator with add(features DataFrame) and to_dict() (for example
    sketches.MirrorSketches); it sees every source, before feature_cut, and is written as gzipped JSON."""
    out = Path(out)
    t0 = time.time()
    name = stem(info["name"])
    paths = {k: out / k / f"{name}.parquet" for k in KINDS}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        b"features_version": features.features_schema_version().encode(),
        b"source_file": info["name"].encode(),
        b"control_fraction": str(features.CONTROL_FRACTION).encode(),
        b"control_salt": hex(features.CONTROL_SALT).encode(),
        b"keep_rule": f"features.keep_light_curve{tuple(keep) if keep else ''}".encode(),
        b"injected": b"sealed plan" if inject else b"none",
        b"feature_cut": (feature_cut or "").encode(),
    }
    tmp = {k: p.with_suffix(".tmp") for k, p in paths.items()}
    writers = {}
    acc = sketch() if sketch is not None else None
    keep_rule = keep
    n_src = n_feat = n_kept = n_rows = 0
    try:
        with get_client().streamed(info["url"], md5=info.get("md5")) as path:
            for lc in epoch.iter_epoch_csv(path):
                if inject:
                    from cosmos_search.inject.classes import gaia_stream
                    lc = gaia_stream.apply(lc, injection_plan(inject))
                bl = blend_for(lc["source_id"].to_numpy(), blend) if blend else None
                feat = features.source_features(lc, blend=bl)
                if acc is not None:
                    acc.add(feat)
                keep = feat["source_id"].to_numpy()[features.keep_light_curve(feat, *(keep_rule or ()))]
                kept = lc.filter(pc.is_in(lc["source_id"], value_set=pa.array(keep)))
                n_src, n_kept, n_rows = n_src + len(feat), n_kept + len(keep), n_rows + lc.num_rows
                if feature_cut:
                    feat = feat.query(feature_cut) if len(feat) else feat
                ft = features.to_arrow(feat)
                if not writers:
                    writers["features"] = pq.ParquetWriter(tmp["features"], ft.schema.with_metadata(meta),
                                                           compression="zstd")
                    writers["lightcurves"] = pq.ParquetWriter(tmp["lightcurves"], epoch.SCHEMA.with_metadata(meta),
                                                              compression="zstd")
                writers["features"].write_table(ft.cast(writers["features"].schema))
                writers["lightcurves"].write_table(kept.cast(writers["lightcurves"].schema))
                n_feat += len(feat)
    finally:
        for w in writers.values():
            w.close()
    for k in KINDS:
        if tmp[k].exists():
            os.replace(tmp[k], paths[k])
    rec = {"name": info["name"], "md5": info.get("md5"), "sources": n_src, "feature_rows": n_feat,
           "kept": n_kept, "rows": n_rows, "seconds": round(time.time() - t0, 1)}
    if acc is not None:
        sk = out / "sketches" / f"{name}.json.gz"
        sk.parent.mkdir(parents=True, exist_ok=True)
        sketches.write(sk, acc.to_dict())
        paths["sketches"] = sk
    if durable:
        _publish(paths, Path(durable), name, rec)
    return rec


def _publish(paths, durable, name, rec):
    """Copy outputs to the durable store, then write the done-marker, then drop the local copies."""
    for kind, p in paths.items():
        if not p.exists():
            continue
        dest = durable / kind / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        shutil.copyfile(p, part)
        if part.stat().st_size != p.stat().st_size:
            raise OSError(f"short copy of {p} to {dest}")
        os.replace(part, dest)
    done = durable / "done" / f"{name}.json"
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(rec) + "\n")
    for p in paths.values():
        p.unlink(missing_ok=True)


def run(out, workers, limit=None, durable=None, shard=(0, 1), feature_cut=None, files=None, sketch=None, blend=None,
        inject=None, keep=None):
    out = Path(out)
    files = files if files is not None else epoch.bulk_files()
    mine = [f for f in files if in_shard(f["name"], shard)]
    todo = [f for f in mine if not is_done(f["name"], out, durable)][:limit]
    print(f"{len(files)} bulk files, {len(mine)} in shard {shard[0]}/{shard[1]}, {len(todo)} to do", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    failures = 0
    with ProcessPoolExecutor(workers) as pool, (out / "manifest.jsonl").open("a") as fh:
        futures = {pool.submit(process_file, f, out, durable, feature_cut, sketch, blend, inject, keep): f for f in todo}
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


def consolidate(src, dest, per_part=100, kinds=("features",)):
    """Merge per-file Parquet under src/<kind>/ into ~per_part-file parts under dest/<kind>/."""
    src, dest = Path(src), Path(dest)
    for kind in kinds:
        paths = sorted((src / kind).glob("*.parquet"))
        (dest / kind).mkdir(parents=True, exist_ok=True)
        for k in range(0, len(paths), per_part):
            chunk = paths[k:k + per_part]
            first, last = chunk[0].stem.split("_")[1].split("-")[0], chunk[-1].stem.split("-")[-1]
            target = dest / kind / f"{kind}_hpx8_{first}-{last}.parquet"
            tables = [pq.read_table(p) for p in chunk]
            meta = tables[0].schema.metadata or {}
            meta = {k: v for k, v in meta.items() if k != b"source_file"}
            table = pa.concat_tables([t.replace_schema_metadata(None) for t in tables], promote_options="default")
            pq.write_table(table.replace_schema_metadata(meta), target, compression="zstd", row_group_size=250_000)
            print(f"wrote {target} ({table.num_rows} rows)", flush=True)


def _shard(text):
    i, n = (int(x) for x in text.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError("shard must be i/n with 0 <= i < n")
    return i, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="local working directory")
    ap.add_argument("--durable", help="durable output directory; done-markers live here")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", type=_shard, default=(0, 1))
    ap.add_argument("--feature-cut", help="pandas query selecting feature rows to keep (default: all)")
    ap.add_argument("--sketch", action="store_true", help="write per-stratum mirror-null tail sketches")
    ap.add_argument("--blend", help="Parquet of source_id + stage0.BLEND_COLS from gaia_source, for Stage 0")
    ap.add_argument("--inject", help="sealed injection plan (inject.classes.gaia_stream); needs COSMOS_REVEAL_KEY")
    ap.add_argument("--keep-z", type=float, help="keep light curves with |z| >= this (both sides), not the DR3 rule")
    ap.add_argument("--keep-table", help="keep light curves whose G3 |W| reaches this table's per-G-bin thresholds "
                    "(dayone.keep); the DR4 rule")
    ap.add_argument("--keep-controls", type=float, default=features.CONTROL_FRACTION,
                    help="with --keep-z or --keep-table: fraction of random controls whose light curves are kept")
    ap.add_argument("--consolidate", metavar="DEST")
    a = ap.parse_args(argv)
    if a.consolidate:
        consolidate(a.durable or a.out, a.consolidate)
        return 0
    sketch = sketches.MirrorSketches if a.sketch else None
    keep = (a.keep_z, a.keep_controls, a.keep_table) if (a.keep_z or a.keep_table) else None
    return 1 if run(a.out, a.workers, a.limit, a.durable, a.shard, a.feature_cut, sketch=sketch, blend=a.blend,
                    inject=a.inject, keep=keep) else 0


if __name__ == "__main__":
    sys.exit(main())
