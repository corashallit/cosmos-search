"""The null model (docs/population.md §3): log lambda = log E_gen + beta . x, a ridge-penalised Poisson GLM.

Covariates are binned, so the fit runs on cells that aggregate stars with identical covariate vectors (sum of
events, sum of E_gen): with main effects plus a colour x absolute-magnitude grid the likelihood is the same as
star by star. None of the covariates reads the dip (§1 point 4): mean photometry, parallax, crowding, robust
light-curve scale, number of visits, and the forward-modelled eclipse share.
"""

import numpy as np
import pandas as pd
from scipy import optimize, sparse, stats

E_FLOOR = 1e-4   # E_gen below this (Monte Carlo zero) is raised to it, so an observed event never has zero rate

BINS = {
    "g": ("phot_g_mean_mag", [11, 12, 13, 14, 15, 16, 17]),
    "c": ("bp_rp", [0.25, 0.5, 0.75, 1.0, 1.25, 1.6, 2.2]),
    "mg": ("m_g", [-1, 1, 3, 5, 8]),
    "x": ("log_excess", [0.1, 0.3, 0.5, 0.8, 1.2]),
    "eb": ("log_eb_share", None),         # sextiles of the population, from the exposure (not the outcome)
    "ipd": ("ipd_frac_multi_peak", [1, 10]),
    "ruwe": ("ruwe", [1.2, 2.0]),
    "blend": ("blend_frac", [0.05, 0.3]),
    "nv": ("n_visits", [12, 16, 22, 30]),
    "ecl": ("abs_ecl_lat", [15, 45]),
}
MAIN = ("g", "c", "mg", "x", "eb", "ipd", "ruwe", "blend", "nv")


def covariate_codes(stars, eb_edges=None):
    """Integer bin per star per covariate (NaN gets its own top code), plus the eb edges used."""
    out = {}
    for key, (col, edges) in BINS.items():
        v = stars[col].to_numpy(dtype=float)
        if key == "eb":
            if eb_edges is None:
                fin = v[np.isfinite(v)]
                eb_edges = np.quantile(fin, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]) if len(fin) else [0] * 5
            edges = eb_edges
        code = np.digitize(v, edges)
        code = np.where(np.isfinite(v), code, len(edges) + 1)
        out[key] = code.astype(np.int16)
    out["cmd"] = (out["c"] * 16 + out["mg"]).astype(np.int16)
    return pd.DataFrame(out, index=stars.index), np.asarray(eb_edges)


def design(codes, terms):
    """Sparse one-hot design (first level of each term dropped) and the column labels."""
    blocks, labels = [], []
    n = len(codes)
    for term in terms:
        v = codes[term].to_numpy()
        levels = np.unique(v)
        idx = np.searchsorted(levels, v)
        m = sparse.csr_matrix((np.ones(n), (np.arange(n), idx)), shape=(n, len(levels)))[:, 1:]
        blocks.append(m)
        labels += [f"{term}={lv}" for lv in levels[1:]]
    blocks.insert(0, sparse.csr_matrix(np.ones((n, 1))))
    return sparse.hstack(blocks).tocsr(), ["intercept"] + labels


def aggregate(codes, y, e, terms):
    """Sum y and E over identical covariate vectors."""
    df = codes[list(terms)].copy()
    df["y"], df["e"] = y, e
    agg = df.groupby(list(terms), sort=False, observed=True).agg(y=("y", "sum"), e=("e", "sum")).reset_index()
    return agg


def fit(codes, y, e, terms, ridge=1.0, agg_terms=None):
    """Fit the Poisson GLM on aggregated cells. Returns a dict with beta, labels, loglik and a predict function.
    agg_terms: aggregate over these (a superset of terms), so nested fits share cells and logliks compare."""
    terms = list(terms)
    agg = aggregate(codes, y, np.maximum(e, E_FLOOR), sorted(set(list(agg_terms or terms) + terms + ["cmd"])))
    use = terms + ["cmd"]
    X, labels = design(agg, use)
    yy, ee = agg.y.to_numpy(float), agg.e.to_numpy(float)
    off = np.log(ee)
    pen = np.r_[0.0, np.full(X.shape[1] - 1, ridge)]

    def f(b):
        eta = X @ b + off
        mu = np.exp(np.clip(eta, -700, 700))
        val = mu.sum() - yy @ eta + 0.5 * (pen * b * b).sum()
        grad = X.T @ (mu - yy) + pen * b
        return val, grad

    b0 = np.zeros(X.shape[1])
    b0[0] = np.log(max(yy.sum(), 1e-9) / ee.sum())
    res = optimize.minimize(f, b0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    beta = res.x
    eta = X @ beta + off
    loglik = float(stats.poisson.logpmf(yy, np.exp(eta)).sum())
    coef = dict(zip(labels, beta))

    def predict(codes_all, e_all):
        lin = np.full(len(codes_all), coef["intercept"])
        for term in use:
            v = codes_all[term].to_numpy()
            lin += np.array([coef.get(f"{term}={lv}", 0.0) for lv in range(int(v.max()) + 1)])[v]
        return np.maximum(e_all, E_FLOOR) * np.exp(lin)

    return {"beta": beta, "labels": labels, "loglik": loglik, "converged": bool(res.success), "terms": use,
            "n_cells": len(agg), "predict": predict}


def lrt(codes, y, e, base_terms, extra_term, ridge=1.0):
    """Likelihood-ratio test for adding one term. Returns (statistic, df, p)."""
    full = list(base_terms) + [extra_term]
    a = fit(codes, y, e, base_terms, ridge, agg_terms=full)
    b = fit(codes, y, e, full, ridge, agg_terms=full)
    df = len(np.unique(codes[extra_term])) - 1
    stat = max(0.0, 2 * (b["loglik"] - a["loglik"]))
    return stat, df, float(stats.chi2.sf(stat, df)) if df > 0 else 1.0
