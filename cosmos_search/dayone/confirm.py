"""The confirmatory batches' pre-registered criteria (docs/dayone.md, "Rule v5 and the confirmatory build" and
"Rule v6").

Rule v5 shortlists a packet when the blind vetter's S = P(interesting leaves) >= vet.S_SHORTLIST. No pipeline flag
shortlists on its own (review R16). The rule is adopted only if all four criteria pass; everything else is reported
with intervals whatever the outcome.

    python -m cosmos_search.dayone.confirm --score score/ --hidden score_hidden/ --strata controls_queue.parquet

Rule v6 shortlists at the lower cut vet.S_SHORTLIST_V6 on packets with the context tests (vetting.context_tests).
Its criteria split the control strata by the ranking's tiers: the activity classes ranking v1 sends to tier 2
(rank.ACTIVE_CLASSES) have their own, looser bound, since they reach Stage A only when tier 1 is exhausted.

    python -m cosmos_search.dayone.confirm --rule v6 --score score/ --compare score_v5packets/ --strata controls.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.inject.lightcurve import HELD_OUT_FAMILIES, HELD_OUT_FAMILIES_V2
from cosmos_search.vetting.power import cp_lower, cp_upper

from .rank import ACTIVE_CLASSES
from .vet import S_SHORTLIST, S_SHORTLIST_V6

RECALL_BOUND = 0.30          # held-out decoys: one-sided 95% lower bound on recall >= this
CONTROL_BOUND = 0.15         # reserved controls, pooled: upper bound on the false-shortlist rate <= this
PER_CLASS_MAX = 3            # and at most this many shortlisted in every class (15 each)
QUEUE_BOUND = 0.10           # queue sample: upper bound on the shortlist rate <= this

# Rule v6 (docs/dayone.md, "Rule v6"). Set for joint power ~0.7 at batch 6's rates seen at S >= 0.2 (post hoc), and
# so that a rule with one tier-1 class shortlisted at 0.4, or every tier-1 class at 0.2, passes under 0.2 of the time.
RECALL_BOUND_V6 = 0.40       # held-out (second set) decoys: one-sided 95% lower bound on recall >= this
TIER1_BOUND_V6 = 0.20        # control strata ranking v1 keeps in tier 1, pooled: upper bound <= this
TIER1_CLASS_MAX_V6 = 4       # and at most this many shortlisted in each of them (15 each)
ACTIVE_BOUND_V6 = 0.40       # activity strata (tier 2 in ranking v1), pooled: upper bound <= this


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


def _hits(table, s_cut):
    t = table.copy()
    s = t["s_interesting"].astype(float)
    t["hit"] = np.where(t["ok"].astype(bool), s >= s_cut, t["kind"] != "decoy")
    return t


def evaluate_v6(table, strata, s_cut=S_SHORTLIST_V6):
    """Rule v6's criteria on a scored table (as `evaluate`), with v5's cut reported on the same verdicts."""
    t = _hits(table, s_cut)
    dec = t[(t["kind"] == "decoy") & t["injected_family"].isin(HELD_OUT_FAMILIES_V2)]
    ctl = t[t["kind"] == "control"].assign(stratum=lambda d: d["candidate_id"].map(strata))
    active = ctl["stratum"].isin(list(ACTIVE_CLASSES))
    que = t[t["role_in_queue"] == "queue_sample"]
    rec, tier1, act, queue = _rate(dec["hit"]), _rate(ctl.loc[~active, "hit"]), _rate(ctl.loc[active, "hit"]), \
        _rate(que["hit"])
    per_class = {k: _rate(g["hit"]) for k, g in ctl.groupby("stratum")}
    tier1_classes = {k: v for k, v in per_class.items() if k not in ACTIVE_CLASSES}
    crit = {
        "held_out_recall": {**rec, "pass": rec["n"] > 0 and rec["lower_95"] >= RECALL_BOUND_V6},
        "controls_tier1_pooled": {**tier1, "pass": tier1["n"] > 0 and tier1["upper_95"] <= TIER1_BOUND_V6},
        "controls_tier1_each_class": {"max_allowed": TIER1_CLASS_MAX_V6, "per_class": tier1_classes,
                                      "pass": bool(tier1_classes)
                                      and all(v["k"] <= TIER1_CLASS_MAX_V6 for v in tier1_classes.values())},
        "controls_active_pooled": {**act, "pass": act["n"] > 0 and act["upper_95"] <= ACTIVE_BOUND_V6},
        "queue_sample": {**queue, "pass": queue["n"] > 0 and queue["upper_95"] <= QUEUE_BOUND},
    }
    v5 = _hits(table, S_SHORTLIST)
    v5_dec = v5[(v5["kind"] == "decoy") & v5["injected_family"].isin(HELD_OUT_FAMILIES_V2)]
    return {"rule": f"v6: real packets (with context tests) with S >= {s_cut}",
            "adopted": all(c["pass"] for c in crit.values()), "criteria": crit,
            "per_class_all": per_class,
            "recall_by_family": {k: _rate(g["hit"]) for k, g in dec.groupby("injected_family")},
            "v5_cut_same_verdicts": {"s_cut": S_SHORTLIST, "held_out_recall": _rate(v5_dec["hit"]),
                                     "controls": _rate(v5.loc[v5["kind"] == "control", "hit"]),
                                     "queue_sample": _rate(v5.loc[v5["role_in_queue"] == "queue_sample", "hit"])}}


def context_shown_vs_not(with_ctx, without, strata):
    """Report only: the same decoys and controls vetted with and without the context tests, at both cuts."""
    ids = set(without["candidate_id"])
    out = {}
    for cut in (S_SHORTLIST_V6, S_SHORTLIST):
        for name, t in (("with_context", with_ctx), ("without_context", without)):
            t = _hits(t[t["candidate_id"].isin(ids)], cut)
            ctl = t[t["kind"] == "control"].assign(stratum=lambda d: d["candidate_id"].map(strata))
            out[f"{name}@{cut}"] = {
                "held_out_recall": _rate(t.loc[(t["kind"] == "decoy")
                                               & t["injected_family"].isin(HELD_OUT_FAMILIES_V2), "hit"]),
                **{f"controls_{k}": _rate(g["hit"]) for k, g in ctl.groupby("stratum")}}
    return out


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
    ap.add_argument("--rule", choices=("v5", "v6"), default="v5")
    ap.add_argument("--score", required=True, help="score dir of the main batch (verdicts_table.csv)")
    ap.add_argument("--hidden", help="v5: score dir of the flag-hidden batch")
    ap.add_argument("--compare", help="v6: score dir of the same decoys and some controls vetted without context")
    ap.add_argument("--strata", required=True, help="parquet with source_id, stratum for the reserved controls")
    a = ap.parse_args(argv)
    st = pd.read_parquet(a.strata, columns=["source_id", "stratum"])
    strata = {f"gaia_dr3:{int(s)}": k for s, k in zip(st["source_id"], st["stratum"])}
    table = pd.read_csv(Path(a.score) / "verdicts_table.csv")
    rep = evaluate_v6(table, strata) if a.rule == "v6" else evaluate(table, strata)
    if a.compare:
        rep["context_shown_vs_not"] = context_shown_vs_not(table, pd.read_csv(Path(a.compare) / "verdicts_table.csv"),
                                                           strata)
    if a.hidden:
        rep["flags_shown_vs_hidden"] = flags_shown_vs_hidden(table, pd.read_csv(Path(a.hidden) / "verdicts_table.csv"))
    (Path(a.score) / "confirm.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps({k: v["pass"] for k, v in rep["criteria"].items()} | {"adopted": rep["adopted"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
