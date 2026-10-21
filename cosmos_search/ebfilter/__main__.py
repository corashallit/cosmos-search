"""Run the EB filter on a day-one vetting queue, before packets are built.

    python -m cosmos_search.ebfilter --queue rank/queue.parquet --ranked rank/ranked.parquet \\
        --lightcurves stream/lightcurves [--external ext/external.parquet] [--plan plan.json.age] --out eb.parquet

Inputs are those of `dayone.vet build`, read through the same `candidate_dicts` (Stage 0 cleaning, sealed plan
applied in memory to Gaia and external epochs alike), so the filter sees exactly what the packets would show. The
target dip is the deepest unflagged Gaia G epoch. Output: one row per queued star with the verdict
(`ebfilter.decide`); `explained` rows move from the vetting queue to the EB control pool.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from cosmos_search.vetting.candidate import LightCurve

from . import decide as D


def verdicts(cands, delta_bits=D.DELTA_BITS):
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
        v = D.decide(lcs, c.get("source") or {}, t_target, delta_bits=delta_bits)
        rows.append({"source_id": int(str(c["candidate_id"]).split(":")[-1]), "t_target": t_target,
                     **{k: v[k] for k in ("explained", "periodic_unphysical", "delta_bits", "period", "target_covered",
                                          "duration_ok", "duration_lo_d", "n_dip_epochs", "n_epochs",
                                          "n_runs_outside", "p_null_bound")},
                     "bands": ",".join(v["bands"])})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.ebfilter")
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
    out = verdicts(candidate_dicts(queue, ranked, a.lightcurves, plan_map=plan_map, external=ext), a.bits)
    out.to_parquet(a.out, index=False)
    print(f"{int(out.explained.sum())} of {len(out)} explained as eclipsing binaries, "
          f"{int(out.periodic_unphysical.sum())} strict-period but unphysical -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
