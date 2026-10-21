"""Build a vetting batch in which real candidates, controls and decoys look alike.

Stage A agents are stateless and see one packet each, so what calibrates them
is per-packet indistinguishability, not a hidden mixing fraction
(docs/vetting.md §4). That puts one rule on batch building: every packet goes
through the same detector, the same field selection and the same holdout. A
decoy that lacks a detector score the real packets carry is a labelled decoy.
"""

import numpy as np

from .candidate import Candidate
from .packet import build, robust_sigma

DIP_FIELDS = ("t_min", "t_start", "t_end", "depth", "duration", "n_points")


def simple_dips(cand, nsigma=3.0, max_dips=10):
    """Runs of consecutive low points in the first light curve. Used only when no detector hook is supplied."""
    lc = cand.lightcurves[0]
    order = np.argsort(lc.time)
    t, f = lc.time[order], lc.flux[order]
    med, s = np.nanmedian(f), robust_sigma(f)
    low = f < med - nsigma * s
    dips, i = [], 0
    while i < len(t):
        if low[i]:
            j = i
            while j + 1 < len(t) and low[j + 1]:
                j += 1
            k = i + int(np.argmin(f[i:j + 1]))
            dips.append({"t_min": float(t[k]), "t_start": float(t[i]), "t_end": float(t[j]),
                         "depth": float(1 - f[k] / med), "duration": float(t[j] - t[i]), "n_points": j - i + 1})
            i = j + 1
        else:
            i += 1
    return sorted(dips, key=lambda d: -d["depth"])[:max_dips]


def harmonise(cands, detect=None):
    """Run the same detector over every candidate and keep only score and test names that all of them have."""
    cands = [c if isinstance(c, Candidate) else Candidate.from_dict(c) for c in cands]
    for c in cands:
        if detect is not None:
            detect(c)
        else:
            c.dips, c.scores, c.tests = simple_dips(c), {}, {}
        c.dips = [{k: d[k] for k in DIP_FIELDS if k in d} for d in c.dips]
    common_scores = set.intersection(*(set(c.scores) for c in cands)) if cands else set()
    common_tests = set.intersection(*(set(c.tests) for c in cands)) if cands else set()
    common_context = set.intersection(*(set(c.context) for c in cands)) if cands else set()
    for c in cands:
        c.scores = {k: v for k, v in c.scores.items() if k in common_scores}
        c.tests = {k: v for k, v in c.tests.items() if k in common_tests}
        c.context = {k: v for k, v in c.context.items() if k in common_context}
    return cands


def build_batch(entries, rng, holdout=(), detect=None, pipeline_flags=True):
    """entries: list of (Candidate, key_extra). Returns (packets, keys) in a shuffled order.

    key_extra carries kind, label, role, acceptable, aliases, and for decoys the
    injection record; it is merged into the packet's key and never into the packet.
    """
    cands = harmonise([e[0] for e in entries], detect)
    packets, keys = [], []
    for cand, (_, extra) in zip(cands, entries):
        packet, key = build(cand, holdout=holdout, pipeline_flags=pipeline_flags)
        key.update(extra)
        packets.append(packet)
        keys.append(key)
    order = rng.permutation(len(packets))
    return [packets[i] for i in order], [keys[i] for i in order]
