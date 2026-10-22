"""Run the residual search on the day-one prefetch list (ranking v1), after the external fetch.

    python -m cosmos_search.residual --queue rank/prefetch.parquet --ranked rank/ranked.parquet \\
        --lightcurves stream/lightcurves [--external ext/external.parquet] [--plan plan.json.age] --out residual.parquet

Inputs and target dip as `python -m cosmos_search.ebfilter` (the same `candidate_dicts`: Stage 0 cleaning, sealed plan
applied in memory). Output: one row per star with the residual verdict (`residual.model.run`) and the EB filter's own
verdict (`eb_explained`), for `python -m cosmos_search.dayone.rank requeue`.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from cosmos_search.ebfilter import decide as D
from cosmos_search.vetting.candidate import LightCurve

from . import model as M

COLUMNS = ("eb_explained", "eb_target_covered", "periodic_unphysical", "delta_bits", "baseline", "period",
           "double_period", "baseline_delta_bits", "target_in_window", "target_residual", "target_residual_sigma",
           "n_residual_dip_epochs", "n_residual_events", "n_residual_events_elsewhere", "residual_testable",
           "residual_aperiodic", "residual_aligned_fraction", "demote", "eb_residual", "n_epochs", "n_dip_epochs",
           "bands")


def verdicts(cands, **kw):
    rows = []
    for c in cands:
        lcs = [LightCurve.from_dict(d) for d in c["lightcurves"]]
        g = next((lc for lc in lcs if lc.band == "G"), None)
        t_target = None
        if g is not None:
            ok = np.isfinite(g.flux)
            if "flags" in g.meta:
                ok &= (g.meta["flags"].astype(int) & 1) == 0
            if ok.any():
                t_target = float(g.time[ok][np.argmin(g.flux[ok])])
        v = M.run(lcs, c.get("source") or {}, t_target, **kw)
        rows.append({"source_id": int(str(c["candidate_id"]).split(":")[-1]), "t_target": t_target,
                     **{k: v.get(k) for k in COLUMNS}})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.residual")
    ap.add_argument("--queue", required=True)
    ap.add_argument("--ranked", required=True)
    ap.add_argument("--lightcurves", required=True)
    ap.add_argument("--external")
    ap.add_argument("--plan", help="sealed injection plan (age), decrypted in memory with COSMOS_REVEAL_KEY")
    ap.add_argument("--bits", type=float, default=D.DELTA_BITS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from cosmos_search.dayone.vet import candidate_dicts

    plan_map = None
    if a.plan:
        from cosmos_search.inject.classes.gaia_stream import load_plan

        plan_map = load_plan(a.plan)
    queue, ranked = pd.read_parquet(a.queue), pd.read_parquet(a.ranked)
    ext = pd.read_parquet(a.external) if a.external else None
    out = verdicts(candidate_dicts(queue, ranked, a.lightcurves, plan_map=plan_map, external=ext),
                   delta_bits=a.bits)
    out.to_parquet(a.out, index=False)
    print(f"{len(out)} stars: {int(out.demote.sum())} demoted (EB filter alone: {int(out.eb_explained.sum())}), "
          f"{int(out.eb_residual.sum())} eclipsing binaries with residual dips -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
