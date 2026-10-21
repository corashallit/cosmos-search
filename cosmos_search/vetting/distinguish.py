"""Can packet features alone tell decoys and controls from real candidates?

If they can, calibration measured on decoys and controls does not transfer to
real packets (docs/vetting.md §4). The audit fits an L2 logistic regression on
features that should carry no label information, scores held-out folds, and
tests the out-of-fold AUC against 0.5 by permutation.
"""

import numpy as np

from .scoring import auc, auc_permutation_p


def features(packet):
    """Numeric fingerprint of a packet: cadence, noise, field presence. No class-specific physics on purpose."""
    f = {"n_dips": len(packet["dips"]), "n_tests": len(packet["tests"]), "n_scores": len(packet["scores"]),
         "n_context": len(packet["context"]), "n_bands": len(packet["bands"]), "text_len": len(packet["text"])}
    for band, s in packet["summaries"].items():
        for k in ("n_epochs", "span_days", "median_flux_err", "robust_sigma"):
            v = s.get(k)
            f[f"{band}_{k}"] = float(v) if isinstance(v, (int, float)) else np.nan
    return f


def _matrix(packets):
    rows = [features(p) for p in packets]
    cols = sorted(set().union(*rows))
    X = np.array([[r.get(c, np.nan) for c in cols] for r in rows], float)
    present = np.isfinite(X).astype(float)              # missingness is itself a feature
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    X = np.where(np.isfinite(X), X, np.nan_to_num(med))
    X = np.hstack([X, present])
    sd = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1)
    return X


def _fit_logistic(X, y, l2=1.0, iters=300, lr=0.1):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w / len(y))
        b -= lr * float(np.mean(p - y))
    return w, b


def audit(packets, is_planted, k=5, rng=None, n_perm=1000):
    """Out-of-fold AUC of a planted-vs-real classifier and its permutation p-value."""
    rng = rng or np.random.default_rng(0)
    X, y = _matrix(packets), np.asarray(is_planted, float)
    folds = rng.permutation(len(y)) % k
    score = np.empty(len(y))
    for f in range(k):
        tr, te = folds != f, folds == f
        w, b = _fit_logistic(X[tr], y[tr])
        score[te] = X[te] @ w + b
    return {"auc": auc(score, y.astype(bool)), "p": auc_permutation_p(score, y.astype(bool), n_perm, rng)}
