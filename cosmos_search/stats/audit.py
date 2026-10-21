"""Random audit of what the ranking misses (statistics.md §5).

Audit slots go to sources below the vetting threshold, drawn by Poisson sampling with known
inclusion probabilities. Probabilities lean towards higher scores (that is where misses are most
likely) but never fall below a floor, so every source can be drawn and the Horvitz–Thompson
estimate stays unbiased whatever the ranking's assumptions got wrong.
"""

import numpy as np


def audit_inclusion(scores, n_audit, floor=0.2):
    """Inclusion probabilities summing to n_audit, increasing with score rank, each >= floor * mean.

    floor in (0, 1]: 1 is uniform sampling, smaller values lean harder towards high scores.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if not 0 < n_audit <= n:
        raise ValueError(f"n_audit must be in (0, {n}], got {n_audit}")
    rank = (np.argsort(np.argsort(scores)) + 0.5) / n
    w = floor + (1 - floor) * 2 * rank  # mean 1
    pi = n_audit * w / w.sum()
    # Cap at 1 and spread the excess over the rest until it settles.
    for _ in range(100):
        over = pi > 1
        if not over.any():
            break
        excess = np.sum(pi[over] - 1)
        pi[over] = 1.0
        free = pi < 1
        pi[free] += excess * pi[free] / pi[free].sum()
    return pi


def poisson_sample(rng, pi):
    """Independent draws with probabilities pi. Returns a boolean mask."""
    return rng.random(np.shape(pi)) < pi


def horvitz_thompson(y, pi):
    """Unbiased estimate of sum(y) over the population from a Poisson sample, with its standard error.

    y and pi are for the sampled units only (y = 1 for "real and missed", say).
    """
    y = np.asarray(y, dtype=float)
    pi = np.asarray(pi, dtype=float)
    total = np.sum(y / pi)
    var = np.sum((1 - pi) * y**2 / pi**2)
    return float(total), float(np.sqrt(var))
