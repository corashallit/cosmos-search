"""Set astrometry-v1's thresholds from a calibration run, by the procedure pre-registered in docs/arms/astrometry.md.

    python scripts/astrometry_thresholds.py --natural runs/nat6 --nongrav runs/inj6 --instrument runs/ins6 \\
        --out runs/thresholds6.json

Prints the chosen thresholds and the calibration tables (natural false alarms by class, recall by kind and
amplitude, instrument labelling). Nothing here is tuned by hand: each threshold is the first grid value that meets
its criterion.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.astrometry import rule

FRAC_MULTI = 0.10
FOV_GRID = (4.0, 5.0, 6.0, 8.0, 10.0)
REL_GRID = (0.0, 0.01, 0.02, 0.05, 0.1)
S_GRID = (20.0, 25.0, 30.0, 40.0, 50.0, 70.0, 100.0, 150.0, 200.0)
FOV_MAX_RATE = 0.005
REL_MAX_LOSS = 0.02
NAT_MAX_RATE = 0.005


def load(d):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(Path(d) / "*.parquet")))], ignore_index=True)
    df = df.rename(columns={"truth_kind": "kind", "truth_amp": "amp"})
    df["id"] = df["shard"].astype(str) + "-" + df["i"].astype(str)
    return df


def choose(nat, ng):
    both = pd.concat([nat, ng])
    fov_z = next(z for z in FOV_GRID if (both["fov_pair_z"].fillna(0) > z).mean() <= FOV_MAX_RATE)
    hi = ng[ng["amp"] >= 20]

    def recall(rel):
        return rule.apply(hi, s_min=S_GRID[0], rel=rel, fov_z=fov_z, frac_multi=FRAC_MULTI)["flagged"].mean()

    r0 = recall(0.0)
    rel = max(r for r in REL_GRID if r0 - recall(r) <= REL_MAX_LOSS)
    s_min = None
    for s in S_GRID:
        if rule.apply(nat, s_min=s, rel=rel, fov_z=fov_z, frac_multi=FRAC_MULTI)["flagged"].mean() <= NAT_MAX_RATE:
            s_min = s
            break
    return {"FOV_Z": fov_z, "REL": rel, "S_MIN": s_min, "FRAC_MULTI": FRAC_MULTI}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--natural", required=True)
    ap.add_argument("--nongrav", required=True)
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nat, ng, ins = load(a.natural), load(a.nongrav), load(a.instrument)
    th = choose(nat, ng)
    print(json.dumps(th))
    if th["S_MIN"] is None:
        raise SystemExit("no S_MIN on the grid meets the natural false-alarm criterion; the rule is not adopted")
    kw = {"s_min": th["S_MIN"], "rel": th["REL"], "fov_z": th["FOV_Z"], "frac_multi": th["FRAC_MULTI"]}
    m = pd.concat([nat, ng, ins], ignore_index=True)
    m["id"] = m.index.astype(str)
    flags = rule.apply(m, **kw)
    m = m.assign(label=flags["label"].to_numpy(), flagged=flags["flagged"].to_numpy())
    score = rule.score(m)
    print(json.dumps(score, indent=1, default=float))
    Path(a.out).write_text(json.dumps({"thresholds": th, "score": score}, indent=1, default=float))


if __name__ == "__main__":
    main()
