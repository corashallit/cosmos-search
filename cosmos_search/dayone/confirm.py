"""The confirmatory batch's pre-registered criteria (docs/dayone.md, "Rule v5 and the confirmatory build").

Rule v5 shortlists a packet when the blind vetter's S = P(interesting leaves) >= vet.S_SHORTLIST. No pipeline flag
shortlists on its own (review R16). The rule is adopted only if all four criteria pass; everything else is reported
with intervals whatever the outcome.

    python -m cosmos_search.dayone.confirm --score score/ --hidden score_hidden/ --strata controls_queue.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.inject.lightcurve import HELD_OUT_FAMILIES
from cosmos_search.vetting.power import cp_lower, cp_upper

from .vet import S_SHORTLIST

RECALL_BOUND = 0.30          # held-out decoys: one-sided 95% lower bound on recall >= this
CONTROL_BOUND = 0.15         # reserved controls, pooled: upper bound on the false-shortlist rate <= this
PER_CLASS_MAX = 3            # and at most this many shortlisted in every class (15 each)
QUEUE_BOUND = 0.10           # queue sample: upper bound on the shortlist rate <= this


def _rate(hit):
    k, n = int(hit.sum()), len(hit)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"),
            "lower_95": float(cp_lower(k, n)) if n else float("nan"),
            "upper_95": float(cp_upper(k, n)) if n else float("nan")}


def evaluate(table, strata, s_cut=S_SHORTLIST):
    """Criteria on a scored table (dayone.vet.score_batch's verdicts_table.csv: kind, injected_family, role_in_queue,
    candidate_id, ok, s_interesting). strata: candidate_id -> DPAC stratum for the reserved controls. A packet whose
    verdict failed to parse counts as not shortlisted for decoys and as shortlisted for controls and the queue, so a
    failure never helps the rule."""
    t = table.copy()
    s = t["s_interesting"].astype(float)
    t["hit"] = np.where(t["ok"].astype(bool), s >= s_cut, t["kind"] != "decoy")
    dec = t[(t["kind"] == "decoy") & t["injected_family"].isin(HELD_OUT_FAMILIES)]
    ctl = t[t["kind"] == "control"].assign(stratum=lambda d: d["candidate_id"].map(strata))
    que = t[t["role_in_queue"] == "queue_sample"]
    rec, pooled, queue = _rate(dec["hit"]), _rate(ctl["hit"]), _rate(que["hit"])
    per_class = {k: _rate(g["hit"]) for k, g in ctl.groupby("stratum")}
    crit = {
        "held_out_recall": {**rec, "pass": rec["n"] > 0 and rec["lower_95"] >= RECALL_BOUND},
        "controls_pooled": {**pooled, "pass": pooled["n"] > 0 and pooled["upper_95"] <= CONTROL_BOUND},
        "controls_each_class": {"max_allowed": PER_CLASS_MAX, "per_class": per_class,
                                "pass": bool(per_class) and all(v["k"] <= PER_CLASS_MAX for v in per_class.values())},
        "queue_sample": {**queue, "pass": queue["n"] > 0 and queue["upper_95"] <= QUEUE_BOUND},
    }
    return {"rule": f"v5: real packets with S >= {s_cut}", "adopted": all(c["pass"] for c in crit.values()),
            "criteria": crit, "recall_by_family": {k: _rate(g["hit"]) for k, g in dec.groupby("injected_family")}}


def flags_shown_vs_hidden(shown, hidden, s_cut=S_SHORTLIST):
    """Report only: the same decoys and controls vetted with and without the pipeline's tests in the packet."""
    out = {}
    for name, t in (("shown", shown), ("hidden", hidden)):
        t = t[t["candidate_id"].isin(set(hidden["candidate_id"]))]
        hit = t["ok"].astype(bool) & (t["s_interesting"].astype(float) >= s_cut)
        out[name] = {"held_out_recall": _rate(hit[t["kind"] == "decoy"]), "controls": _rate(hit[t["kind"] == "control"])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.dayone.confirm")
    ap.add_argument("--score", required=True, help="score dir of the main batch (verdicts_table.csv)")
    ap.add_argument("--hidden", help="score dir of the flag-hidden batch")
    ap.add_argument("--strata", required=True, help="parquet with source_id, stratum for the reserved controls")
    a = ap.parse_args(argv)
    st = pd.read_parquet(a.strata, columns=["source_id", "stratum"])
    strata = {f"gaia_dr3:{int(s)}": k for s, k in zip(st["source_id"], st["stratum"])}
    table = pd.read_csv(Path(a.score) / "verdicts_table.csv")
    rep = evaluate(table, strata)
    if a.hidden:
        rep["flags_shown_vs_hidden"] = flags_shown_vs_hidden(table, pd.read_csv(Path(a.hidden) / "verdicts_table.csv"))
    (Path(a.score) / "confirm.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps({k: v["pass"] for k, v in rep["criteria"].items()} | {"adopted": rep["adopted"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
