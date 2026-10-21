"""How many controls and decoys to spend, and when a vetter configuration is good enough.

The arithmetic behind docs/vetting.md §7. The budget is ~10^3 expert slots
(10^4 at the edge), so every control has to buy precision on the number we
report: the expected count of false entries on the shortlist.
"""

import math

import numpy as np

from .scoring import betainc


def n_zero_failure(rate_max, conf=0.95):
    """Controls needed, with no failures, to bound a failure rate below rate_max at confidence conf."""
    return math.ceil(math.log(1 - conf) / math.log(1 - rate_max))


def neyman_allocation(budget, mix, rate_guess, floor=10):
    """Split `budget` control slots across classes to minimise the variance of N * sum_c q_c a_c.

    mix: expected class mix q_c of the real queue; rate_guess: guessed rate a_c at
    which class c is wrongly passed. Var is sum_c q_c^2 a_c (1 - a_c) / n_c, so
    the optimum is n_c proportional to q_c sqrt(a_c (1 - a_c)). Every class gets
    at least `floor`, because a class with no controls has an unmeasured rate.
    """
    mix, rate = np.asarray(mix, float), np.clip(np.asarray(rate_guess, float), 1e-4, 0.5)
    w = mix * np.sqrt(rate * (1 - rate))
    n = np.full(len(mix), floor, dtype=float)
    spare = budget - n.sum()
    if spare < 0:
        raise ValueError(f"budget {budget} is below floor x classes = {int(n.sum())}")
    n += spare * w / w.sum()
    out = np.floor(n).astype(int)
    for i in np.argsort(-(n - out))[: budget - out.sum()]:  # largest remainders get the leftover slots
        out[i] += 1
    return out


def posterior_below(k, n, rate_max, a=0.5, b=0.5):
    """P(rate < rate_max | k failures in n), under a Beta(a, b) prior."""
    return betainc(k + a, n - k + b, rate_max)


def sequential_check(k, n, rate_max, conf=0.95):
    """Accept, reject or keep testing a vetter on one control class.

    Bayesian stopping on the failure rate: accept when P(rate < rate_max) >= conf,
    reject when P(rate > rate_max) >= conf, otherwise continue. Checking after
    every batch is fine here because the decision is about the posterior, not a
    p-value; the stated conf is the posterior probability at the time of stopping.
    """
    p = posterior_below(k, n, rate_max)
    if p >= conf:
        return "accept", p
    if 1 - p >= conf:
        return "reject", p
    return "continue", p


def chain_false_pass(rates, joint=None):
    """False-pass rate of filters in series.

    With independent errors it is the product of the stage rates; `joint`, if
    given, is the measured rate at which all stages fail together on the same
    controls, which is what to use once error dependence has been measured.
    """
    return float(joint) if joint is not None else float(np.prod(rates))


def expected_shortlist_false(n_queue, mix, false_pass):
    """Expected false entries on the shortlist for a queue of n_queue with class mix and per-class false-pass rates."""
    return float(n_queue * np.dot(mix, false_pass))
