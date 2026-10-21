"""Stream DR3 epoch photometry to a compact, Stage-0-cleaned G table with BP/RP of the same transit.

One Parquet per bulk file under OUT/, long format sorted by (source_id, t). Resumable (skips files already written).
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from cosmos_search.net import get_client
from cosmos_search.gaia import epoch, dr3_pass
from cosmos_search.vetting import stage0

OUT = Path(os.environ.get("POP_LC_DIR", "/home/user/pop/lc"))
BLEND = "/mnt/project-files/data/gaia-dr3-epoch/aux/aux_blend.parquet"
LOCAL_BLEND = os.environ.get("POP_BLEND", BLEND)


def compact(table):
    df = table.to_pandas()
    df["band"] = df["band"].astype(str)
    g = df[df.band == "G"][["source_id", "t", "flux", "err", "flags", "fov", "transit_id"]]
    out = g.copy()
    for b in ("BP", "RP"):
        o = df[df.band == b][["source_id", "transit_id", "flux", "err"]].drop_duplicates(["source_id", "transit_id"], keep="last")
        o = o.rename(columns={"flux": b.lower(), "err": b.lower() + "_err"})
        out = out.merge(o, on=["source_id", "transit_id"], how="left")
    out = out.sort_values(["source_id", "t"]).drop(columns=["transit_id"])
    sid = out.source_id.to_numpy(np.int64)
    cols = {"source_id": sid, "t": out.t.to_numpy(np.float32), "flux": out.flux.to_numpy(np.float32),
            "flags": out["flags"].to_numpy(np.int8),
            "fov": out.fov.to_numpy(np.int8)}
    fg = out.flux.to_numpy(np.float64)
    medg = pd.Series(fg).groupby(sid).transform("median").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cols["err_rel"] = (out.err.to_numpy() / medg).astype(np.float16)   # G error / median G flux
    for b in ("bp", "rp"):
        f = out[b].to_numpy(np.float64)
        med = pd.Series(f).groupby(sid).transform("median").to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            cols["d" + b] = (1 - f / med).astype(np.float16)            # fractional dimming in BP / RP
            cols["e" + b] = (out[b + "_err"].to_numpy() / med).astype(np.float16)
    return pa.table(cols)


def one(info):
    name = dr3_pass.stem(info["name"])
    dest = OUT / f"{name}.parquet"
    if dest.exists():
        return {"name": name, "cached": True}
    t0 = time.time()
    tmp = dest.with_suffix(".tmp")
    w = None
    n = 0
    with get_client().streamed(info["url"], md5=info.get("md5")) as path:
        for lc in epoch.iter_epoch_csv(path):
            bl = dr3_pass.blend_for(lc["source_id"].to_numpy(), LOCAL_BLEND)
            lc, _ = stage0.clean_long(lc, blend=bl)
            c = compact(lc)
            if w is None:
                w = pq.ParquetWriter(tmp, c.schema, compression="zstd")
            w.write_table(c)
            n += c.num_rows
    if w is not None:
        w.close()
        os.replace(tmp, dest)
    return {"name": name, "rows": n, "s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    files = epoch.bulk_files()
    todo = [f for f in files if not (OUT / f"{dr3_pass.stem(f['name'])}.parquet").exists()]
    print(len(files), "files,", len(todo), "to do", flush=True)
    fails = 0
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(one, f): f for f in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                fails += 1
                r = {"name": futs[fu]["name"], "error": repr(e)}
            if i % 25 == 0 or "error" in r:
                print(i, r, flush=True)
    print("done, failures", fails, flush=True)
