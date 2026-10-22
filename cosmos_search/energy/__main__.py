"""G5 command line (docs/arms/energy-budget.md).

    python -m cosmos_search.energy reference --out ref/
    python -m cosmos_search.energy cal --ref ref/ --half A --out cal/
    python -m cosmos_search.energy candidates --ref ref/ --ranked ranked.parquet --out g5.parquet [--phot phot.parquet]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


def delta_from_ranked(ranked, phot):
    """⟨δ⟩ lower bound from a ranked table without light curves (§6): half the deepest depth times the one-sided 95%
    Clopper-Pearson lower limit on the dimmed fraction of visit groups, N = max(matched_transits / 2, n_episodes)."""
    m = ranked[["source_id", "depth_frac", "n_episodes"]].merge(phot[["source_id", "matched_transits"]], on="source_id",
                                                               how="left")
    k = m["n_episodes"].to_numpy(float)
    n = np.maximum(m["matched_transits"].to_numpy(float) / 2, k)
    lo = np.where((k > 0) & (n > 0), beta_dist.ppf(0.05, k, n - k + 1), 0.0)
    return pd.Series(0.5 * np.abs(m["depth_frac"].to_numpy(float)) * np.nan_to_num(lo), index=m["source_id"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.energy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--out", required=True)
    c = sub.add_parser("cal")
    c.add_argument("--ref", required=True)
    c.add_argument("--half", choices=("A", "B", "C"), required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--z-excess", type=float, help="half B: the thresholds chosen on half A")
    c.add_argument("--z-deficit", type=float)
    k = sub.add_parser("candidates")
    k.add_argument("--ref", required=True)
    k.add_argument("--ranked", required=True)
    k.add_argument("--phot", help="photometry from reference.fetch_ids (fetched if missing)")
    k.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "reference":
        from .reference import fetch
        fetch(a.out)
    elif a.cmd == "cal":
        from . import cal
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        h = {"A": 0, "B": 1, "C": cal.FRESH}[a.half]
        seed = int.from_bytes(os.urandom(8), "little")
        fam, dim, nat = cal.run_half(a.ref, h, seed)
        for name, df in (("fam", fam), ("dim", dim), ("nat", nat)):
            df.to_parquet(out / f"{name}_{a.half}.parquet")
        if a.half == "A":
            z_ex, z_def = cal.choose_thresholds(fam, dim)
        else:
            z_ex, z_def = a.z_excess, a.z_deficit
        rep = cal.summary(fam, dim, nat, z_ex, z_def)
        rep["seed"] = seed
        (out / f"report_{a.half}.json").write_text(json.dumps(rep, indent=2, default=float))
        print(json.dumps(rep, indent=2, default=float))
    elif a.cmd == "candidates":
        from . import run
        from .reference import fetch_ids
        ranked = pd.read_parquet(a.ranked)
        phot = pd.read_parquet(a.phot) if a.phot else fetch_ids(ranked["source_id"])
        phot = phot.drop_duplicates("source_id").set_index("source_id").loc[ranked["source_id"]].reset_index()
        delta = delta_from_ranked(ranked, phot).to_numpy()
        res, _ = run.evaluate(phot, run.load_locus(a.ref), delta=delta)
        keep = [c for c in ("tier", "best_class_name", "depth_frac", "n_episodes", "grey") if c in ranked]
        res = res.merge(ranked[["source_id", *keep]], on="source_id", how="left")
        res.to_parquet(a.out)
        print(f"{len(res)} candidates -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
