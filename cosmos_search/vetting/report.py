"""Score a revealed batch: calibration, confabulation, sensitivity, recognition, held-out skill.

Key records (one per packet) carry:
  kind        "real", "control" (known answer) or "decoy" (injected)
  label       true class: a leaf or a group from taxonomy (controls and decoys)
  acceptable  classes counted as correct (default: the label's group)
  role        for decoys: "incoherent", "anomaly_probe" or "rare_natural"
  aliases     for controls: names that count as correct recognition
  truths      held-out target answers (all kinds)
"""

import numpy as np

from . import scoring, taxonomy

GROUP_NAMES = list(taxonomy.GROUPS)


def group_of(name):
    return name if name in taxonomy.GROUPS else taxonomy.GROUP_OF[name]


def decision(v):
    """Group with the highest probability; the shortlist decision is 'unexplained' winning."""
    g = v.groups
    return max(g, key=g.get)


def _acceptable_groups(key):
    acc = key.get("acceptable") or [key["label"]]
    return {group_of(a) for a in acc}


def score(verdicts, keys):
    vs = [v for v in verdicts if v.packet_id in keys]
    ok = [v for v in vs if v.ok]
    rep = {"n": len(vs), "format_failure_rate": 1 - len(ok) / len(vs) if vs else float("nan")}

    labelled = [v for v in ok if keys[v.packet_id].get("kind") in ("control", "decoy")]
    if labelled:
        y = np.array([GROUP_NAMES.index(group_of(keys[v.packet_id]["label"])) for v in labelled])
        p = np.array([[v.groups[g] for g in GROUP_NAMES] for v in labelled])
        correct = [decision(v) in _acceptable_groups(keys[v.packet_id]) for v in labelled]
        is_u = (y == GROUP_NAMES.index("unexplained")).astype(float)
        rep["labelled"] = {
            "n": len(labelled), "brier": scoring.brier(p, y), "log_score": scoring.log_score(p, y),
            "accuracy": float(np.mean(correct)),
            "confusion_counts": _counts(y, [GROUP_NAMES.index(decision(v)) for v in labelled]),
            "p_unexplained_reliability": scoring.reliability(p[:, 2], is_u, 5),
            "p_unexplained_ece": scoring.ece(p[:, 2], is_u, 5),
        }

    by_role = {}
    for v in ok:
        k = keys[v.packet_id]
        if k.get("kind") == "decoy":
            by_role.setdefault(k.get("role", "unspecified"), []).append(v)
    if "incoherent" in by_role:
        r = by_role["incoherent"]
        # Confabulation: a confident natural story for a signal no natural scenario produces.
        rep["confabulation_rate"] = _rate(sum(v.groups["astrophysical"] > 0.5 for v in r), len(r))
    if "anomaly_probe" in by_role:
        r = by_role["anomaly_probe"]
        rep["probe_sensitivity"] = _rate(sum(decision(v) == "unexplained" for v in r), len(r))
    if "rare_natural" in by_role:
        r = by_role["rare_natural"]
        rep["rare_natural_accuracy"] = _rate(sum(decision(v) in _acceptable_groups(keys[v.packet_id]) for v in r), len(r))

    controls = [v for v in ok if keys[v.packet_id].get("kind") == "control"]
    if controls:
        named = [v for v in controls if v.recognised_as]
        hits = [v for v in named if _alias_hit(v.recognised_as, keys[v.packet_id].get("aliases", []))]
        rep["recognition"] = {"named": _rate(len(named), len(controls)), "correct": _rate(len(hits), len(controls))}
        # False "unexplained" on known-answer controls is the shortlist's main contaminant.
        rep["control_false_unexplained"] = _rate(sum(decision(v) == "unexplained" for v in controls), len(controls))

    probes = by_role.get("anomaly_probe", [])
    if probes and controls:
        # Stage A's job in the queue is ranking. What matters is whether P(unexplained) puts planted anomalies
        # above known-answer controls, whichever group wins the argmax. In pilot 2, probes got P(u) 0.20-0.40,
        # the largest single class, but "astrophysical" summed over many leaves still won.
        pu = [v.groups["unexplained"] for v in probes + controls]
        lab = [True] * len(probes) + [False] * len(controls)
        top = max(v.groups["unexplained"] for v in controls)
        rep["unexplained_ranking"] = {
            "auc_probes_vs_controls": scoring.auc(pu, lab),
            "auc_p": scoring.auc_permutation_p(pu, lab),
            "max_control_p_unexplained": top,
            "probes_above_every_control": _rate(sum(v.groups["unexplained"] > top for v in probes), len(probes)),
            "probes_unexplained_top_leaf": _rate(sum(max(v.class_probs, key=v.class_probs.get) == "unexplained"
                                                     for v in probes), len(probes)),
        }

    rep["held_out"] = held_out_skill(ok, keys)
    return rep


def held_out_skill(verdicts, keys):
    """Per target: Brier skill (binary) or coverage and interval score (interval), split by kind and confidence."""
    rows = {}
    for v in verdicts:
        k = keys[v.packet_id]
        conf = "confident" if max(v.class_probs.values(), default=0) >= 0.8 else "unsure"
        for tid, truth in (k.get("truths") or {}).items():
            if tid in v.predictions:
                rows.setdefault(tid, []).append((k.get("kind", "real"), conf, v.predictions[tid], truth))
    out = {}
    for tid, rs in rows.items():
        out[tid] = {}
        for split in ("all", "real", "control", "decoy", "confident", "unsure"):
            sub = [r for r in rs if split == "all" or split in r[:2]]
            if not sub:
                continue
            preds, truths = [r[2] for r in sub], [r[3] for r in sub]
            if isinstance(preds[0], list):
                lo, hi = zip(*preds)
                out[tid][split] = {"n": len(sub), "coverage_80": scoring.interval_coverage(lo, hi, truths),
                                   "interval_score": scoring.interval_score(lo, hi, truths)}
                if split == "control":  # known answers: the widening to apply to real candidates' intervals
                    out[tid][split]["conformal_widening_80"] = scoring.conformal_widening(lo, hi, truths)
            else:
                y = np.array(truths, float)
                base = np.mean([float(r[3]) for r in rs])  # base rate over the whole batch, not the split
                out[tid][split] = {"n": len(sub), "brier": scoring.brier_binary(preds, y),
                                   "brier_skill": scoring.brier_skill(preds, y, base)}
    return out


def _counts(y, d):
    m = np.zeros((3, 3), int)
    np.add.at(m, (np.asarray(y), np.asarray(d)), 1)
    return {GROUP_NAMES[i]: dict(zip(GROUP_NAMES, m[i].tolist())) for i in range(3)}


def _rate(k, n):
    lo, hi = scoring.rate_interval(k, n)
    return {"k": int(k), "n": int(n), "rate": k / n if n else float("nan"), "interval_95": [lo, hi]}


def _alias_hit(name, aliases):
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
    n = norm(name)
    return any(norm(a) and (norm(a) in n or n in norm(a)) for a in aliases)
