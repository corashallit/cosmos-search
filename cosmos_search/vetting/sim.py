"""Simulated vetters, to test the statistics before any expert time is spent.

A simulated vetter has a true confusion matrix over the three groups and
states probabilities drawn around its decision, with a tunable overconfidence.
study() asks: for a given control budget and queue, how close does the
label-shift estimate of the shortlist's false entries come to the truth, and
does its interval cover? Output feeds the tables in docs/vetting.md §7.

    python -m cosmos_search.vetting.sim
"""

import json

import numpy as np

from .design import neyman_allocation
from .quantify import shortlist_false

U = 2  # index of "unexplained" in (instrumental, astrophysical, unexplained)


class SimulatedVetter:
    def __init__(self, confusion, concentration=20.0, overconfidence=1.0):
        self.C = np.asarray(confusion, float)
        self.conc, self.over = concentration, overconfidence

    def decide(self, true, rng):
        return np.array([rng.choice(3, p=self.C[c]) for c in np.asarray(true)])

    def probs(self, decided, rng):
        """Stated probabilities: a Dirichlet centred on the decision, sharpened by overconfidence."""
        alpha = np.full((len(decided), 3), 1.0)
        alpha[np.arange(len(decided)), decided] += self.conc
        p = np.array([rng.dirichlet(a) for a in alpha]) ** self.over
        return p / p.sum(axis=1, keepdims=True)


def study(vetter, queue_mix, n_queue, control_budget, n_rep=200, rng=None, n_boot=300):
    """Repeat: draw a queue and a control set, estimate shortlist false entries, compare to truth."""
    rng = rng or np.random.default_rng(1)
    alloc = neyman_allocation(control_budget, np.maximum(queue_mix, 0.05), np.diag(1 - vetter.C).clip(0.01))
    errs, covered, widths = [], [], []
    for _ in range(n_rep):
        q_true = rng.choice(3, size=n_queue, p=queue_mix)
        d_queue = vetter.decide(q_true, rng)
        truth = int(np.sum((d_queue == U) & (q_true != U)))
        c_true = np.repeat(np.arange(3), alloc)
        c_dec = vetter.decide(c_true, rng)
        est = shortlist_false(c_true, c_dec, d_queue, 3, U, n_boot=n_boot, rng=rng)
        lo, hi = est["interval"]
        errs.append(est["expected_false"] - truth)
        covered.append(lo <= truth <= hi)
        widths.append(hi - lo)
    return {"allocation": alloc.tolist(), "bias": float(np.mean(errs)), "rmse": float(np.sqrt(np.mean(np.square(errs)))),
            "coverage_90": float(np.mean(covered)), "mean_width_90": float(np.mean(widths))}


def main():
    # A plausible vetter: 3% of instrumental and 2% of astrophysical objects called unexplained; 70% of probes found.
    C = [[0.90, 0.07, 0.03], [0.08, 0.90, 0.02], [0.10, 0.20, 0.70]]
    mix = np.array([0.6, 0.399, 0.001])  # real queue: mostly instrumental, almost nothing truly unexplained
    rows = []
    for budget in (100, 300, 1000):
        r = study(SimulatedVetter(C), mix, 1000, budget, n_rep=100, n_boot=200)
        rows.append({"control_budget": budget, **r})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
