"""Stage 0: dips shared between unrelated candidates are instrumental.

A real occulter belongs to one star. An instrument event (a scattered-light
episode, an attitude glitch, a calibration step) touches every star observed at
that moment. So for each dip we ask whether other candidates have dips at the
same time more often than the local dip rate predicts. This is a population
test only the vetting stage can run, because only it sees the whole queue
(and the random-control stream) at once.

For a dip at t, with n dips from other candidates within +-window of t, of
which k are within +-tau, the null is k ~ Binomial(n, tau / window): the local
rate is flat on the scale of the window. Using the local rate rather than a
global one absorbs coverage gaps and season-to-season changes in how many stars
are observed. Groups (camera/CCD for TESS, scan or field of view for Gaia)
restrict the comparison to dips that could share a cause.
"""

import math
from collections import defaultdict

import numpy as np


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + (i * math.log(p) if p > 0 else (0 if i == 0 else -math.inf))
            + ((n - i) * math.log1p(-p) if p < 1 else (0 if i == n else -math.inf))
            for i in range(k, n + 1)]
    m = max(logs)
    return float(min(1.0, math.exp(m) * sum(math.exp(x - m) for x in logs)))


def coincidence(dips, tau, window, group_key=None):
    """Score each dip for coincidence with dips of other candidates.

    dips: iterable of dicts with "candidate_id", "t_min" (common clock, e.g. BJD
    days) and optionally the group_key. Returns one dict per input dip with
    k, n, expected and p (small p: shared with other stars, likely instrumental).
    """
    if not window > tau > 0:
        raise ValueError("need window > tau > 0")
    dips = list(dips)
    groups = defaultdict(list)
    for i, d in enumerate(dips):
        groups[d.get(group_key) if group_key else None].append(i)
    out = [None] * len(dips)
    for members in groups.values():
        t = np.array([float(dips[i]["t_min"]) for i in members])
        cid = np.array([str(dips[i]["candidate_id"]) for i in members])
        order = np.argsort(t)
        ts, cs = t[order], cid[order]
        for j_sorted, i in zip(range(len(order)), np.asarray(members)[order]):
            t0, c0 = ts[j_sorted], cs[j_sorted]
            lo, hi = np.searchsorted(ts, t0 - window), np.searchsorted(ts, t0 + window, side="right")
            dt, other = np.abs(ts[lo:hi] - t0), cs[lo:hi] != c0
            n = int(other.sum())
            k = int((other & (dt <= tau)).sum())
            p_near = tau / window
            out[i] = {"candidate_id": c0, "t_min": float(t0), "k": k, "n": n,
                      "expected": n * p_near, "p": binom_sf(k, n, p_near)}
    return out


def flag_candidates(results, alpha):
    """Candidates with any dip whose coincidence p is below alpha / (number of dips) (Bonferroni over the queue)."""
    m = max(len(results), 1)
    worst = {}
    for r in results:
        worst[r["candidate_id"]] = min(worst.get(r["candidate_id"], 1.0), r["p"])
    return {cid: p for cid, p in worst.items() if p < alpha / m}
