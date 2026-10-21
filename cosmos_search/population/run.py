"""Driver for the population test (docs/population.md). Steps, in order:

    python -m cosmos_search.population.run events   --lc LC_DIR --out OUT       # events, A plain / B sealed
    python -m cosmos_search.population.run masks    --out OUT                   # time masks (B's sealed)
    python -m cosmos_search.population.run exposure --lc LC_DIR --out OUT --astrometry DIR
    python -m cosmos_search.population.run assemble --out OUT --aux AUX_DIR --astrometry DIR
    python -m cosmos_search.population.run search   --out OUT                   # half A: GLM, scan, candidates
    python -m cosmos_search.population.run replicate --out OUT --candidates FILE  # opens half B

LC_DIR holds compact G light curves (one Parquet per bulk file: source_id, t, flux, err_rel, flags, fov, dbp, drp,
sorted by source and time). Half-B events and half-B masks are only ever written as age ciphertext to
keys/cora.age.pub and keys/reveal.age.pub; `replicate` refuses to run unless the candidates file is committed.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cosmos_search.inject.sealed import sha256, write_sealed

from . import events, exposure, split

REPO = Path(__file__).resolve().parents[2]
RECIPIENTS = [str(REPO / "keys" / "cora.age.pub"), str(REPO / "keys" / "reveal.age.pub")]
T0 = -200.0                       # BTJD origin of the hour bins
N_HOURS = int((1000.0 - T0) * 24)
NSIDE_MASK_SHIFT = 57             # source_id >> 57 = HEALPix nside-2 (level 1) nested pixel
N_PIX_MASK = 48
MASK_P = 1e-3
EVENT_COLS = ["source_id", "cls", "half", "t_deep", "depth", "z", "n_in", "visit", "dbp", "drp", "fov"]


def hour_bin(t):
    return np.clip(np.floor((np.asarray(t, float) - T0) * 24).astype(np.int64), 0, N_HOURS - 1)


def mask_pixel(source_id):
    return (np.asarray(source_id, np.int64) >> NSIDE_MASK_SHIFT).astype(np.int64)


def _read(path):
    return pq.read_table(path).to_pandas()


# -- events ----------------------------------------------------------------------------------------------------

def _events_one(path, out, keep_rejected):
    name = Path(path).stem
    rows, stars = events.prepare(_read(path), keep_rejected=keep_rejected)
    ev = events.find(rows, stars)
    stars.drop(columns=["first_visit"]).to_parquet(out / "stars" / f"{name}.parquet", index=False)
    ev[ev.half == 0][EVENT_COLS].to_parquet(out / "events_A" / f"{name}.parquet", index=False)
    hist = np.zeros((2, N_HOURS, N_PIX_MASK), np.int64)
    np.add.at(hist, (rows["vhalf"], hour_bin(rows["t"]), mask_pixel(rows["source_id"])), 1)
    b = ev[ev.half == 1][EVENT_COLS]
    return name, hist, b.to_dict(orient="list")


def cmd_events(a):
    out = Path(a.out)
    for d in ("stars", "events_A"):
        (out / d).mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{a.lc}/*.parquet"))[: a.limit]
    hist = np.zeros((2, N_HOURS, N_PIX_MASK), np.int64)
    sealed_b = {k: [] for k in EVENT_COLS}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (name, h, b) in enumerate(ex.map(_events_one, files, [out] * len(files),
                                                [a.keep_rejected] * len(files), chunksize=4), 1):
            hist += h
            for k in EVENT_COLS:
                sealed_b[k] += [v.item() if hasattr(v, "item") else v for v in b[k]]
            if i % 200 == 0:
                print(f"{i}/{len(files)}", flush=True)
    np.save(out / "epoch_hist.npy", hist)
    digest = write_sealed(sealed_b, out / "events_B.json.age", RECIPIENTS)
    (out / "events_B.sha256").write_text(digest + "\n")
    print(f"{len(files)} files; half-B events sealed ({digest[:12]})", flush=True)


def load_sealed(path):
    from pyrage import decrypt, x25519
    secret = os.environ.get("COSMOS_REVEAL_KEY", "").strip()
    if not secret:
        raise RuntimeError("opening sealed half-B products needs COSMOS_REVEAL_KEY")
    return json.loads(decrypt(Path(path).read_bytes(), [x25519.Identity.from_str(secret)]))


def events_A(out):
    return pd.concat([_read(p) for p in sorted(glob.glob(f"{out}/events_A/*.parquet"))], ignore_index=True)


# -- time masks ----------------------------------------------------------------------------------------------

def time_mask(ev, hist_half, p=MASK_P):
    """Masked (hour, nside-2 pixel) cells for one half: whole-sky hours and hour x pixel cells whose event count is
    improbable (binomial, Bonferroni over occupied cells) given the epochs observed there. All event classes count."""
    from scipy import stats
    hb, px = hour_bin(ev.t_deep.to_numpy()), mask_pixel(ev.source_id.to_numpy())
    n = len(ev)
    mask = np.zeros(hist_half.shape, bool)
    if n == 0:
        return mask, {}
    tot = hist_half.sum()
    sky = hist_half.sum(axis=1)
    k_sky = np.bincount(hb, minlength=N_HOURS)
    occ = sky > 0
    p_sky = stats.binom.sf(k_sky - 1, n, sky / tot)
    bad_sky = occ & (p_sky < p / occ.sum()) & (k_sky >= 3)
    mask[bad_sky, :] = True
    k_cell = np.zeros(hist_half.shape, np.int64)
    np.add.at(k_cell, (hb, px), 1)
    occ_c = hist_half > 0
    p_cell = stats.binom.sf(k_cell - 1, n, hist_half / tot)
    bad_cell = occ_c & (p_cell < p / occ_c.sum()) & (k_cell >= 3)
    mask |= bad_cell
    hit = mask[hb, px]
    info = {"events": int(n), "sky_hours_masked": int(bad_sky.sum()), "cells_masked": int(bad_cell.sum()),
            "events_masked": int(hit.sum()), "epochs_masked": int(hist_half[mask].sum()),
            "epochs_total": int(tot)}
    return mask, info


def cmd_masks(a):
    out = Path(a.out)
    hist = np.load(out / "epoch_hist.npy")
    mA, infoA = time_mask(events_A(out), hist[0])
    np.save(out / "mask_A.npy", mA)
    evB = pd.DataFrame(load_sealed(out / "events_B.json.age"))
    mB, infoB = time_mask(evB, hist[1])
    write_sealed({"cells": np.argwhere(mB).tolist(), "info": infoB}, out / "mask_B.json.age", RECIPIENTS)
    (out / "mask_A.json").write_text(json.dumps(infoA, indent=1) + "\n")
    print("half A mask:", infoA, flush=True)


def masks(out):
    """(mask_A, mask_B) boolean arrays; B decrypted in memory."""
    mA = np.load(Path(out) / "mask_A.npy")
    mB = np.zeros_like(mA)
    cells = load_sealed(Path(out) / "mask_B.json.age")["cells"]
    if cells:
        c = np.array(cells)
        mB[c[:, 0], c[:, 1]] = True
    return mA, mB


# -- exposure -----------------------------------------------------------------------------------------------

def _astrometry(path, cols=("source_id", "parallax", "parallax_error")):
    """gaia_source astrometry, Bailer-Jones distances and DPAC class for the epoch-photometry sources (one Parquet,
    fetched by TAP: see docs/population.md §11)."""
    return pq.read_table(path, columns=list(cols)).to_pandas()


_HOST = {}


def _host_lookup(astro_dir, aux_path):
    if "t" not in _HOST:
        ast = _astrometry(astro_dir)
        aux = pq.read_table(aux_path, columns=["source_id", "phot_g_mean_mag", "bp_rp"]).to_pandas()
        t = aux.merge(ast, on="source_id", how="left").sort_values("source_id")
        r, m = exposure.host_params(t.phot_g_mean_mag, t.bp_rp, t.parallax, t.parallax_error)
        _HOST["t"] = (t.source_id.to_numpy(), r, m)
    return _HOST["t"]


def _exposure_one(path, out, astro_dir, aux_path, keep_rejected, n_gen, n_eb):
    name = Path(path).stem
    mA, mB = masks(out)
    rows, stars = events.prepare(_read(path), keep_rejected=keep_rejected)
    # Epochs in a masked (hour, pixel) cell of their half are not evidence for or against a dip: drop them, as
    # the masks drop events there.
    hb, px = hour_bin(rows["t"]), mask_pixel(rows["source_id"])
    bad = np.where(rows["vhalf"] == 0, mA[hb, px], mB[hb, px])
    sid, r, m = _host_lookup(astro_dir, aux_path)
    j = np.clip(np.searchsorted(sid, stars.source_id.to_numpy()), 0, len(sid) - 1)
    ok = sid[j] == stars.source_id.to_numpy()
    radius, mass = np.where(ok, r[j], np.nan), np.where(ok, m[j], np.nan)
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    rows_m = _mask_rows(rows, bad)
    eg = exposure.generic(rows_m, stars, n_draws=n_gen, rng=rng)
    eb = exposure.eclipse(rows_m, stars, radius, mass, n_draws=n_eb, rng=rng)
    df = pd.DataFrame({"source_id": stars.source_id, "E_gen_A": eg[:, 0], "E_gen_B": eg[:, 1],
                       "E_EB_A": eb[:, 0], "E_EB_B": eb[:, 1]})
    df.to_parquet(Path(out) / "exposure" / f"{name}.parquet", index=False)
    return name


def _mask_rows(rows, bad):
    """Rows of masked epochs keep their place but can never be in dip or flat: depth set so z is NaN."""
    rows = dict(rows)
    rows["masked"] = bad
    return rows


def cmd_exposure(a):
    out = Path(a.out)
    (out / "exposure").mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(glob.glob(f"{a.lc}/*.parquet"))[: a.limit]
             if not (out / "exposure" / f"{Path(f).stem}.parquet").exists()]
    n = len(files)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(_exposure_one, files, [out] * n, [a.astrometry] * n, [a.aux_blend_src] * n,
                                     [a.keep_rejected] * n, [a.n_gen] * n, [a.n_eb] * n, chunksize=4), 1):
            if i % 200 == 0:
                print(f"{i}/{n}", flush=True)
    print("exposure done", flush=True)


def _garwood(o, cl=0.90):
    from scipy import stats
    a = (1 - cl) / 2
    lo = 0.0 if o == 0 else stats.chi2.ppf(a, 2 * o) / 2
    return lo, stats.chi2.ppf(1 - a, 2 * o + 2) / 2


def cmd_replicate(a):
    """Open half B and test the frozen half-A candidates (docs/population.md §5)."""
    from scipy import stats
    from . import analysis, glm, scan
    cpath = Path(a.candidates)
    if not _committed(cpath):
        raise SystemExit(f"commit {cpath} before opening half B")
    out = Path(a.out)
    frozen = json.loads(cpath.read_text())
    stars = analysis.load_stars(out)
    codes, _ = glm.covariate_codes(stars)
    terms = [t for t in frozen["model"]["terms"] if t != "cmd"]
    evB = pd.DataFrame(load_sealed(out / "events_B.json.age"))
    _, mB = masks(out)
    yA = analysis.outcome(stars, events_A(out), 0, np.load(out / "mask_A.npy"))
    yB = analysis.outcome(stars, evB, 1, mB)
    lam = {}
    for h, y, e in (("A", yA, stars.E_gen_A.to_numpy()), ("B", yB, stars.E_gen_B.to_numpy())):
        m = glm.fit(codes, y, e, terms)
        lam[h] = m["predict"](codes, e)
    win = scan.Windows()
    res = []
    n = max(len(frozen["candidates"]), 1)
    for c in frozen["candidates"]:
        mem = analysis.window_members(stars, win, c)
        row = dict(c)
        for h, y in (("A", yA), ("B", yB)):
            o, e = float(y[mem].sum()), float(lam[h][mem].sum())
            lo, hi = _garwood(o)
            row[h] = {"O": o, "E": e, "excess_ratio": (o - e) / e, "excess_ratio_90": [(lo - e) / e, (hi - e) / e],
                      "p_one_sided": float(stats.poisson.sf(o - 1, e))}
        row["replicated"] = row["B"]["p_one_sided"] < 0.05 / 5
        row["weak"] = row["replicated"] and row["B"]["excess_ratio"] < 0.25 * row["A"]["excess_ratio"]
        res.append(row)
    rec = {"frozen_candidates": str(cpath), "frozen_sha256": sha256(cpath), "opened_at_commit": _head(),
           "n_events_B": float(yB.sum()), "expected_B": float(lam["B"].sum()), "results": res}
    Path(a.replication).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1), flush=True)


def _committed(path):
    """True if path is tracked by git with no uncommitted changes."""
    p = Path(path).resolve()
    try:
        tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", str(p)],
                                 capture_output=True).returncode == 0
        clean = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", str(p)]).returncode == 0
    except FileNotFoundError:
        return False
    return tracked and clean


def cmd_assemble(a):
    from . import analysis
    df = analysis.assemble(a.out, a.aux, a.astrometry)
    print(f"{len(df)} stars, {int(df.excluded.sum())} in the excluded Magellanic zones", flush=True)


def cmd_search(a):
    from . import analysis
    out = Path(a.out)
    stars = analysis.load_stars(out)
    y = analysis.outcome(stars, events_A(out), 0, np.load(out / "mask_A.npy"))
    cands, info, terms, win = analysis.search(stars, y, stars.E_gen_A.to_numpy(), rng=np.random.default_rng(a.seed))
    rec = {"half": "A", "candidates": cands, "model": info, "code_commit": _head(),
           "plan": "docs/population.md", "n_stars": int(len(stars))}
    Path(a.candidates).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "model"}, indent=1), flush=True)


def _head():
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.population.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("events", "masks", "exposure", "assemble", "search", "replicate"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--lc")
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--limit", type=int)
        s.add_argument("--keep-rejected", action="store_true")
        s.add_argument("--astrometry")
        s.add_argument("--aux-blend-src", default="/home/user/pop/aux/aux_gaia_source.parquet")
        s.add_argument("--n-gen", type=int, default=exposure.N_GEN)
        s.add_argument("--n-eb", type=int, default=exposure.N_EB)
        s.add_argument("--aux", default="/home/user/pop/aux")
        s.add_argument("--candidates", default=str(REPO / "population" / "candidates_A.json"))
        s.add_argument("--seed", type=int, default=20261021)
        s.add_argument("--replication", default=str(REPO / "population" / "replication_B.json"))
    a = ap.parse_args(argv)
    {"events": cmd_events, "masks": cmd_masks, "exposure": cmd_exposure, "assemble": cmd_assemble,
     "search": cmd_search, "replicate": cmd_replicate}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
