"""Scoring rules, calibration and interval estimates for vetters. Numpy only.

The quantities and why each is here are in docs/vetting.md §5–§7. Everything
takes plain arrays so the same code scores Stage A experts, Stage 0 tests and
the simulated vetters in sim.py.
"""

import math

import numpy as np

# --- Proper scoring rules -------------------------------------------------


def brier(p, y):
    """Multi-class Brier score. p: (n, k) probabilities; y: (n,) integer labels. Lower is better; 0 is perfect."""
    p = np.asarray(p, dtype=float)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(p)), np.asarray(y)] = 1
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def log_score(p, y, floor=1e-4):
    """Mean negative log probability of the true label, with probabilities floored so one zero is not infinite."""
    p = np.asarray(p, dtype=float)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(p)), np.asarray(y)], floor, 1))))


def brier_binary(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def brier_skill(p, y, base=None):
    """1 - Brier / Brier of the base rate. > 0 means the forecaster beats always saying the base rate."""
    y = np.asarray(y, float)
    base = y.mean() if base is None else base
    ref = brier_binary(np.full_like(y, base), y)
    return float("nan") if ref == 0 else 1 - brier_binary(p, y) / ref


def interval_score(lo, hi, y, alpha=0.2):
    """Gneiting & Raftery interval score for central (1 - alpha) intervals. Lower is better."""
    lo, hi, y = (np.asarray(a, float) for a in (lo, hi, y))
    return float(np.mean((hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)))


def interval_coverage(lo, hi, y):
    lo, hi, y = (np.asarray(a, float) for a in (lo, hi, y))
    return float(np.mean((y >= lo) & (y <= hi)))


# --- Calibration ------------------------------------------------------------


def reliability(p, y, n_bins=10):
    """Per-bin mean forecast, observed frequency and count, on equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    which = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = which == b
        if m.any():
            rows.append({"bin": b, "mean_p": float(p[m].mean()), "freq": float(y[m].mean()), "n": int(m.sum())})
    return rows


def ece(p, y, n_bins=10):
    """Expected calibration error: count-weighted mean |forecast - frequency| over bins."""
    rows = reliability(p, y, n_bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["mean_p"] - r["freq"]) for r in rows) / n) if n else float("nan")


class Isotonic:
    """Monotone recalibration map fitted by pool-adjacent-violators.

    Fit on control packets with known outcomes; apply to stated probabilities
    on real packets. Monotone means a vetter that says 0.9 is never mapped
    below one that says 0.8, which is the only structure we can defend with a
    few hundred controls.
    """

    def __init__(self, x, y, prior=(0.5, 1.0)):
        x, y = np.asarray(x, float), np.asarray(y, float)
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
        # A weak prior pulls empty or tiny blocks toward prior[0]; prior[1] is its weight in pseudo-counts per block.
        level, weight, lo = [], [], []
        for xi, yi in zip(x, y):
            level.append(yi)
            weight.append(1.0)
            lo.append(xi)
            while len(level) > 1 and level[-2] > level[-1]:
                w = weight[-2] + weight[-1]
                level[-2] = (level[-2] * weight[-2] + level[-1] * weight[-1]) / w
                weight[-2] = w
                level.pop(), weight.pop(), lo.pop()
        m, a = prior
        self.x = np.array(lo)
        self.y = np.array([(l * w + m * a) / (w + a) for l, w in zip(level, weight)])
        self.y = np.maximum.accumulate(self.y)

    def __call__(self, p):
        p = np.asarray(p, float)
        idx = np.clip(np.searchsorted(self.x, p, side="right") - 1, 0, len(self.y) - 1)
        return self.y[idx]


# --- Intervals for rates ------------------------------------------------------


def _betacf(a, b, x, iters=300, eps=3e-16):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c if abs(1 + aa / c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c if abs(1 + aa / c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def beta_ppf(q, a, b):
    """Quantile of Beta(a, b) by bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def rate_interval(k, n, conf=0.95):
    """Jeffreys interval for a binomial rate k/n; the upper end is 1 - (1-conf)^(1/n)-like when k = 0."""
    if n == 0:
        return (0.0, 1.0)
    tail = (1 - conf) / 2
    lo = 0.0 if k == 0 else beta_ppf(tail, k + 0.5, n - k + 0.5)
    hi = 1.0 if k == n else beta_ppf(1 - tail, k + 0.5, n - k + 0.5)
    return (lo, hi)


def rate_upper(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound; for k = 0 this is the 'rule of three' bound (~3/n at 95%)."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return beta_ppf(conf, k + 1, n - k)


# --- Distinguishability and agreement ----------------------------------------


def auc(score, label):
    """Area under the ROC curve (Mann-Whitney), ties counted half."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_permutation_p(score, label, n_perm=2000, rng=None):
    """Two-sided permutation p-value for AUC != 0.5. Used to test that decoys are indistinguishable from real packets."""
    rng = rng or np.random.default_rng(0)
    label = np.asarray(label, bool)
    obs = abs(auc(score, label) - 0.5)
    hits = sum(abs(auc(score, rng.permutation(label)) - 0.5) >= obs for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


def error_dependence(err_a, err_b):
    """How much more often two filters fail together than independence predicts.

    Returns (P(both), P(a)P(b), ratio). The chain's false-pass rate is the
    product of stage rates only when ratio ~ 1; correlated stages buy less.
    """
    a, b = np.asarray(err_a, bool), np.asarray(err_b, bool)
    both, indep = float(np.mean(a & b)), float(np.mean(a) * np.mean(b))
    return both, indep, (both / indep if indep > 0 else float("nan"))
