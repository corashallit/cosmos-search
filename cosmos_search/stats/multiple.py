"""Combining p-values within a source and controlling errors across sources (statistics.md §1, §3)."""

import numpy as np


def expected_false_alarms(n_tests, p):
    """N * p: the number of null sources expected to pass a per-source threshold p."""
    return n_tests * np.asarray(p, dtype=float)


def cauchy_combine(p, weights=None, axis=-1):
    """Cauchy combination test (Liu & Xie 2020).

    Valid under arbitrary dependence between the inputs in the tail that matters, which suits
    correlated detectors run on the same light curve. Tiny p-values use tan(pi (1/2 - p)) ~ 1/(pi p)
    so nothing underflows.
    """
    p = np.asarray(p, dtype=float)
    if weights is None:
        weights = np.full(p.shape[axis], 1.0 / p.shape[axis])
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    shape = [1] * p.ndim
    shape[axis] = -1
    w = w.reshape(shape)
    small = p < 1e-15
    with np.errstate(divide="ignore"):
        t_i = np.where(small, 1 / (np.pi * np.maximum(p, 1e-300)), np.tan((0.5 - p) * np.pi))
    t = np.sum(w * t_i, axis=axis)
    big = t > 1e15
    return np.where(big, 1 / (np.pi * np.where(big, t, 1.0)), 0.5 - np.arctan(t) / np.pi)


def _bh_adjust(p):
    p = np.asarray(p, dtype=float)
    flat = p.ravel()
    n = flat.size
    order = np.argsort(flat)
    ranked = flat[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out.reshape(p.shape)


def bh(p, q=0.1):
    """Benjamini–Hochberg. Returns (rejected, adjusted q-values)."""
    adj = _bh_adjust(p)
    return adj <= q, adj


def storey_pi0(p, lam=0.5):
    """Storey's estimate of the null fraction: #{p > lam} / ((1 - lam) N), capped at 1."""
    p = np.asarray(p, dtype=float).ravel()
    return float(min(1.0, np.sum(p > lam) / ((1 - lam) * p.size)))


def storey_q(p, lam=0.5):
    """Storey q-values: pi0 times the BH adjustment. Nearly identical to BH when pi0 ~ 1, as here."""
    return storey_pi0(p, lam) * _bh_adjust(p)
