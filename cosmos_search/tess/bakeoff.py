"""The T4 comparison run: ~10⁴ QLP light curves through the dimming detector, with nulls and injections.

    python -m cosmos_search.tess.bakeoff sample  --out data/bakeoff/v0 --sectors 60 66 72 78 84 90 --n 10000
    python -m cosmos_search.tess.bakeoff process --out data/bakeoff/v0 --workers 12
    python -m cosmos_search.tess.analyse          --out data/bakeoff/v0

Every light curve is processed three ways:

- real:     as delivered; this is the search;
- mirror:   flux reflected about 1, the empirical null for the dip statistic (statistics.md §2);
- injected: one signal from inject.classes.dimming.FAMILIES (shared library plus dimming extras), with a
            uniformly chosen family and parameters from its prior, multiplied in.

Injections here form the visible tuning set. Their truth goes to injections.parquet in the output
directory, which is fine for tuning and never for final numbers (blinding.md §5). Nothing about an
injection is stored next to its features except a row id, so the analysis joins them deliberately.

Only features and events are kept; light curves are streamed and discarded, so a 10⁴ run needs ~10 MB.
"""

import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.dimming.detect import find_events, scan
from cosmos_search.dimming.features import event_diagnostics, extract
from cosmos_search.inject import lightcurve as injlib
from cosmos_search.inject.classes import dimming as injfam

from . import qlp
from .sample import qlp_sector_sample

LOW_THRESHOLD = 4.0     # events kept for the common-mode test
THRESHOLD = 5.0         # events kept as detections
CHUNK = 250


def cmd_sample(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    per = int(np.ceil(args.n / len(args.sectors)))
    rows = []
    for s in args.sectors:
        rows.extend(qlp_sector_sample(s, per, rng))
        print(f"sector {s}: {len(rows)} so far", flush=True)
    df = pd.DataFrame(rows, columns=["sector", "tic"]).drop_duplicates().reset_index(drop=True)
    df["row"] = np.arange(len(df))
    df.to_parquet(out / "targets.parquet")
    print(f"wrote {len(df)} targets")


def _event_rows(lc, events, variant, row, kind):
    out = []
    for e in events:
        d = e.as_dict()
        d.update(event_diagnostics(lc, e, qlp.REPORT) if kind == "det" else {})
        d.update(row=row, variant=variant, kind=kind)
        out.append(d)
    return out


def process_one(sector, tic, row, seed):
    """Features and events for one (sector, TIC), or None if QLP has no light curve."""
    raw = qlp.fetch(sector, tic)
    if raw is None:
        return None
    lc = qlp.clean(raw)
    if len(lc) < 500:
        return None
    meta = {k: v for k, v in lc.meta.items() if k != "desat_times"}
    rng = np.random.default_rng(seed)
    family = str(rng.choice(injfam.FAMILIES))
    t0, t1 = float(lc.t[0]), float(lc.t[-1])
    sig = injfam.sample(rng, family, t0, t1)
    flux, _ = injlib.inject(lc.t, lc.flux, lc.err, lc.band, sig)
    injected = lc.with_flux(flux)
    # An on-target signal changes every aperture's flux by the same factor.
    k = sig.factor(lc.t, lc.band)
    injected.aux = {n: (np.asarray(x) * k if n.startswith("flux_") else x) for n, x in lc.aux.items()}
    variants = {"real": lc, "mirror": lc.inverted(), "injected": injected}
    feats, events = [], []
    for name, v in variants.items():
        traces = scan(v)
        f, dips, _ = extract(v, traces, threshold=THRESHOLD, flag_bits=qlp.REPORT)
        f.update(meta, row=row, variant=name)
        feats.append(f)
        events += _event_rows(v, dips, name, row, "det")
        # Sub-threshold events of both signs feed the common-mode test.
        for sign, kind in ((-1, "low_dip"), (1, "low_burst")):
            if name == "real":
                events += _event_rows(v, find_events(v, traces, threshold=LOW_THRESHOLD, sign=sign), name, row, kind)
    deficit, excess = injfam.max_effect(sig, lc.t, lc.band)
    truth = {"row": row, "family": family, "max_deficit": deficit, "max_excess": excess,
             "windows": json.dumps(injfam.windows(sig, t0, t1)),
             "signal": json.dumps(injlib.truth(sig), default=_json_default)}
    return feats, events, truth


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _run_chunk(chunk, base_seed):
    feats, events, truths, missing = [], [], [], 0
    for sector, tic, row in chunk:
        try:
            r = process_one(sector, tic, row, base_seed + row)
        except Exception:
            traceback.print_exc()
            r = None
        if r is None:
            missing += 1
            continue
        f, e, t = r
        feats += f
        events += e
        truths.append(t)
    return feats, events, truths, missing


def cmd_process(args):
    out = Path(args.out)
    targets = pd.read_parquet(out / "targets.parquet")
    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    # Injection seeds come from urandom once per run and are written down, so a run is reproducible
    # from its directory but not predictable from the code.
    seed_file = out / "seed.txt"
    if not seed_file.exists():
        seed_file.write_text(str(int.from_bytes(os.urandom(6), "little")))
    base_seed = int(seed_file.read_text())
    todo = []
    for i in range(0, len(targets), CHUNK):
        if not (parts / f"features_{i:06d}.parquet").exists():
            sub = targets.iloc[i:i + CHUNK]
            todo.append((i, list(zip(sub.sector, sub.tic, sub.row))))
    print(f"{len(todo)} chunks to do", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_run_chunk, c, base_seed): i for i, c in todo}
        for fut in as_completed(futs):
            i = futs[fut]
            feats, events, truths, missing = fut.result()
            pd.DataFrame(events).to_parquet(parts / f"events_{i:06d}.parquet")
            pd.DataFrame(truths).to_parquet(parts / f"injections_{i:06d}.parquet")
            pd.DataFrame(feats).to_parquet(parts / f"features_{i:06d}.parquet")
            print(f"chunk {i}: {len(truths)} light curves, {missing} missing", flush=True)
    for name in ("features", "events", "injections"):
        files = sorted(parts.glob(f"{name}_*.parquet"))
        frames = [pd.read_parquet(f) for f in files]
        pd.concat([f for f in frames if len(f)], ignore_index=True).to_parquet(out / f"{name}.parquet")
    print("merged")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cosmos_search.tess.bakeoff")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", required=True)
    s.add_argument("--sectors", type=int, nargs="+", required=True)
    s.add_argument("--n", type=int, default=10000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_sample)
    r = sub.add_parser("process")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=8)
    r.set_defaults(fn=cmd_process)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
