"""Simulate DR4-like epoch astrometry (natural classes and non-gravitational injections) and run the cascade.

    python scripts/astrometry_sim.py --out runs/astrom/nat --n 6000 --kinds natural --seed 1
    python scripts/astrometry_sim.py --out runs/astrom/inj --n 3000 --kinds nongrav --seed 2

Writes one parquet per shard of 200 sources (resumable: existing shards are skipped). Cadence from GOST predictions
for 1,000 real stars (/mnt/project-files/concurrent/gost_dr4.parquet, sample.parquet)."""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.astrometry import detect
from cosmos_search.astrometry import simulate as S

GOST = "/mnt/project-files/concurrent/gost_dr4.parquet"
STARS = "/mnt/project-files/concurrent/sample.parquet"
SHARD = 200
_CAD = None


def _init():
    global _CAD
    _CAD = S.Cadence(pd.read_parquet(GOST), pd.read_parquet(STARS))


def run_shard(job):
    out, seed, k, kinds, amp_range = job
    path = Path(out) / f"shard-{k:04d}.parquet"
    if path.exists():
        return k
    rng = np.random.default_rng([seed, k])
    rows = []
    for i in range(SHARD):
        if kinds == "natural":
            kind = rng.choice(list(S.NATURAL), p=np.array(list(S.NATURAL.values())) / sum(S.NATURAL.values()))
            amp = None
        else:
            kind = rng.choice(S.NONGRAV if kinds == "nongrav" else kinds.split(","))
            amp = float(10 ** rng.uniform(*np.log10(amp_range)))
        data, truth = S.simulate(rng, _CAD, kind, amp=amp)
        try:
            res = detect.cascade(data)
        except Exception as e:  # noqa: BLE001 -- keep the shard; record the failure
            res = {"error": repr(e)[:200]}
        rows.append({**{f"truth_{a}": b for a, b in truth.items()}, **res, "shard": k, "i": i})
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, path)
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--kinds", default="natural")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--amp", type=float, nargs=2, default=(1.0, 50.0))
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = [(a.out, a.seed, k, a.kinds, tuple(a.amp)) for k in range(int(np.ceil(a.n / SHARD)))]
    with Pool(a.procs, initializer=_init) as pool:
        for k in pool.imap_unordered(run_shard, jobs):
            print(f"shard {k} done", flush=True)


if __name__ == "__main__":
    main()
