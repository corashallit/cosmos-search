"""How many shortlist entries are false, estimated from controls under label shift.

Controls with known answers give the vetter's confusion matrix
C[c, d] = P(decision d | true class c). The real queue has an unknown class mix
q, so its decision counts follow C^T q. If the vetter's behaviour given the true
class is the same on controls and on real candidates (the label-shift
assumption, which the indistinguishability audit tests), q can be estimated
from the real decisions alone (Saerens et al. 2002; Lipton et al. 2018), and
the expected number of explicable objects among the k "unexplained" decisions is

    k * sum_{c != u} q_c C[c, u] / sum_c q_c C[c, u].

That number, with its bootstrap interval, is the shortlist false-positive
figure we report. See docs/vetting.md §5.
"""

import numpy as np


def confusion(true, decided, n_classes, alpha=0.5):
    """Row-normalised confusion matrix with a Jeffreys-like pseudo-count per cell."""
    m = np.full((n_classes, n_classes), alpha, dtype=float)
    np.add.at(m, (np.asarray(true), np.asarray(decided)), 1)
    return m / m.sum(axis=1, keepdims=True)


def em_label_shift(C, decided, n_iter=500, tol=1e-10, q0=None):
    """Maximum-likelihood class mix q of a queue, given its decisions and the confusion matrix C."""
    C = np.asarray(C, float)
    k = C.shape[0]
    counts = np.bincount(np.asarray(decided), minlength=C.shape[1]).astype(float)
    q = np.full(k, 1 / k) if q0 is None else np.asarray(q0, float)
    for _ in range(n_iter):
        joint = q[:, None] * C                      # P(c, d)
        post = joint / joint.sum(axis=0, keepdims=True).clip(1e-300)   # P(c | d)
        q_new = (post * counts[None, :]).sum(axis=1) / counts.sum()
        if np.max(np.abs(q_new - q)) < tol:
            q = q_new
            break
        q = q_new
    return q


def expected_false(C, q, n, u):
    """Expected count of non-u objects that the vetter decides are u, in a queue of n with class mix q."""
    C, q = np.asarray(C, float), np.asarray(q, float)
    others = np.arange(len(q)) != u
    return float(n * np.sum(q[others] * C[others, u]))


def false_fraction(C, q, u):
    """P(true class != u | decision u): the chance that a given shortlist entry is explicable."""
    C, q = np.asarray(C, float), np.asarray(q, float)
    joint = q * C[:, u]
    return float(1 - joint[u] / joint.sum()) if joint.sum() > 0 else float("nan")


def shortlist_false(ctrl_true, ctrl_decided, queue_decided, n_classes, u, n_boot=2000, rng=None, conf=0.9):
    """How many of the queue's u-decisions (the shortlist) are false: estimate and predictive interval.

    Conditional on the observed shortlist size k, the false count is
    Binomial(k, f) with f = false_fraction(C, q, u). The interval resamples
    controls within each class and the queue as a whole, so C and q carry their
    uncertainty, then draws the binomial, so it covers the realised count rather
    than only its expectation.
    """
    rng = rng or np.random.default_rng(0)
    ctrl_true, ctrl_decided, queue_decided = map(np.asarray, (ctrl_true, ctrl_decided, queue_decided))
    n, k = len(queue_decided), int(np.sum(queue_decided == u))
    C = confusion(ctrl_true, ctrl_decided, n_classes)
    q = em_label_shift(C, queue_decided)
    f = false_fraction(C, q, u)
    by_class = [np.flatnonzero(ctrl_true == c) for c in range(n_classes)]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in by_class if len(ix)])
        Cb = confusion(ctrl_true[idx], ctrl_decided[idx], n_classes)
        qd = queue_decided[rng.integers(0, n, n)]
        fb = false_fraction(Cb, em_label_shift(Cb, qd, q0=q), u)
        boots[b] = rng.binomial(k, fb) if np.isfinite(fb) else k
    tail = (1 - conf) / 2
    return {"n_queue": n, "n_shortlist": k, "false_fraction": f, "expected_false": k * f,
            "interval": (float(np.quantile(boots, tail)), float(np.quantile(boots, 1 - tail))),
            "class_mix": q.tolist(), "confusion": C.tolist()}
