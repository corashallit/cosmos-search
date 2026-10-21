"""Fit the error model and counterpart prior from the data, and measure false matches on the shifted sky.

Calibration maximises the marginal likelihood of all real targets,

    log L(s, e, pi_b) = sum_units log[(1 - pi_b) + pi_b * sum_j f_j(s, e) / rho_j],

over the survey error scale s, the floor e, and one prior pi_b per stratum (Gaia G magnitude bins by
default). This is the same mixture the posterior uses, so the fitted posterior is calibrated by
construction if the model holds. Two checks say whether it holds:

1. chi^2 of matches, weighted by P, should follow chi^2 with 2 degrees of freedom (mean 2, 1% above 9.21);
2. the false matches measured on the shifted sky should agree with the model's own expected count.

The shifted sky has two channels (summary()). Plain pseudo-targets measure how often a source with no
counterpart is matched to a chance neighbour. Planted pseudo-targets get a synthetic counterpart drawn
from the fitted error model, next to the real neighbours at their shifted position; they measure recall
and how often a chance neighbour beats a true counterpart (confusion), which matters in crowded fields.
Neither number is assumed; both are reported with every run.
"""

import numpy as np
from scipy.optimize import minimize

from .match import ErrorModel, log_lr, posteriors, unit_posteriors

DEFAULT_G_BINS = np.array([-np.inf, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20.5, np.inf])
THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.99)


def strata(gmag, bins=DEFAULT_G_BINS):
    g = np.where(np.isfinite(gmag), gmag, 21.0)
    return np.clip(np.digitize(g, bins) - 1, 0, len(bins) - 2)


MIN_MATCHES_FOR_ERRORS = 50   # effective matches needed to fit scale and floor
MIN_MATCHES_FOR_TAIL = 200    # and to fit the tail component


def fit(cands, stratum, *, n_strata=None, model=None, min_per_stratum=10):
    """Maximum-likelihood (error model, pi per stratum) from real targets only.

    Stage 1 fits pi with the starting error model. If that implies at least MIN_MATCHES_FOR_ERRORS
    matches, stage 2 also fits scale and floor, and with MIN_MATCHES_FOR_TAIL the tail weight and width.
    Strata with fewer than min_per_stratum targets take pi from the nearest populated stratum.
    Returns (ErrorModel, pi per stratum, info dict).
    """
    model = model or ErrorModel(floor_mas=cands.survey.floor_mas)
    n_strata = n_strata or int(stratum.max()) + 1
    real_pair = cands.shift < 0
    real_unit = cands.unit_shift < 0
    u_index = np.full(len(cands.unit_target), -1)
    u_index[real_unit] = np.arange(real_unit.sum())
    pu = u_index[cands.unit[real_pair]]
    us = stratum[cands.unit_target[real_unit]]
    counts = np.bincount(us, minlength=n_strata)
    d, cg, cs, rho = (cands.d[real_pair], cands.cov_gaia[real_pair], cands.cov_survey[real_pair],
                      cands.rho[real_pair])

    def run(free):
        def unpack(x):
            vals = dict(scale=model.scale, floor_mas=model.floor_mas, tail_frac=model.tail_frac,
                        tail_mas=model.tail_mas)
            i = 0
            for name in free:
                vals[name] = 1 / (1 + np.exp(-x[i])) if name == "tail_frac" else np.exp(x[i])
                i += 1
            return ErrorModel(**vals), 1 / (1 + np.exp(-x[i:]))

        def nll(x):
            m, pis = unpack(x)
            llr, _ = log_lr(d, cg, cs, rho, m)
            lr_sum = np.bincount(pu, weights=np.exp(np.minimum(llr, 700)), minlength=len(us))
            pi = pis[us]
            like = (1 - pi) + pi * lr_sum
            # weak Beta(1.05, 1.05) prior keeps empty strata off 0 and 1
            return -(np.sum(np.log(like)) + 0.05 * np.sum(np.log(pis) + np.log1p(-pis)))

        start = {"scale": np.log(model.scale), "floor_mas": np.log(max(model.floor_mas, 1.0)),
                 "tail_frac": np.log(0.02 / 0.98), "tail_mas": np.log(max(model.tail_mas, 5 * model.floor_mas, 1000.0))}
        bounds = {"scale": (np.log(0.2), np.log(10.0)), "floor_mas": (np.log(1.0), np.log(3e4)),
                  "tail_frac": (np.log(1e-4 / (1 - 1e-4)), np.log(0.6 / 0.4)), "tail_mas": (np.log(200.0), np.log(3e4))}
        x0 = np.array([start[f] for f in free] + [0.0] * n_strata)
        b = [bounds[f] for f in free] + [(-12.0, 12.0)] * n_strata
        res = minimize(nll, x0, method="L-BFGS-B", bounds=b)
        m, pis = unpack(res.x)
        return m, pis, res

    m, pis, res = run([])
    # Effective number of matches implied by stage 1.
    llr, _ = log_lr(d, cg, cs, rho, m)
    lr_sum = np.bincount(pu, weights=np.exp(np.minimum(llr, 700)), minlength=len(us))
    pi = pis[us]
    n_eff = float(np.sum(pi * lr_sum / ((1 - pi) + pi * lr_sum)))
    free = []
    if n_eff >= MIN_MATCHES_FOR_ERRORS:
        free += ["scale", "floor_mas"]
    if n_eff >= MIN_MATCHES_FOR_TAIL:
        free += ["tail_frac", "tail_mas"]
    if free:
        model = ErrorModel(scale=model.scale, floor_mas=model.floor_mas, tail_frac=0.02 if "tail_frac" in free else 0.0,
                           tail_mas=model.tail_mas)
        m, pis, res = run(free)
    pis = _fill_small_strata(pis, counts, min_per_stratum)
    info = {"success": bool(res.success), "nll": float(res.fun), "n_units": int(len(us)), "n_eff_matches": n_eff,
            "fitted": free, "counts": counts.tolist(), "message": str(res.message)}
    return m, pis, info


def _fill_small_strata(pis, counts, min_count):
    good = np.nonzero(counts >= min_count)[0]
    if not len(good):
        pooled = float(np.sum(pis * counts) / max(counts.sum(), 1))
        return np.full_like(pis, pooled)
    out = pis.copy()
    for i in np.nonzero(counts < min_count)[0]:
        out[i] = pis[good[np.argmin(np.abs(good - i))]]
    return out


def chance_probability(cands, model, pi_t, threshold=0.9):
    """Per target: the probability that, if it had no counterpart, an unrelated survey source would still be
    accepted at P >= threshold. This is the per-source version of the shifted-sky false-match rate.

    A chance source at offset d is accepted when pi LR / ((1 - pi) + pi LR) >= T, i.e. when its density
    f(d) >= rho_free L* with L* = T (1 - pi) / (pi (1 - T)). That happens inside a region of area A around
    the target, and unrelated sources arrive as a Poisson process of density rho_free, so
    p_chance = 1 - exp(-rho_free A). The error ellipse is approximated by a circle of equal area; the
    target's own most likely candidate supplies the survey errors and density (else the medians).
    """
    n = cands.n_targets
    real = cands.shift < 0
    llr, _ = cands.log_lr(model)
    cg_med = np.median(cands.cov_gaia[real], axis=0) if real.any() else np.eye(2) * 1.0
    cs_med = np.median(cands.cov_survey[real], axis=0) if real.any() else np.zeros((2, 2))
    rho_med = np.median(cands.rho[real]) if real.any() else 1e-9
    cg = np.broadcast_to(cg_med, (n, 2, 2)).copy()
    cs = np.broadcast_to(cs_med, (n, 2, 2)).copy()
    rho = np.full(n, rho_med)
    if real.any():
        idx = np.nonzero(real)[0]
        order = idx[np.lexsort((llr[idx], cands.target[idx]))]
        last = np.r_[cands.target[order][1:] != cands.target[order][:-1], True]
        top = order[last]
        t = cands.target[top]
        cg[t], cs[t], rho[t] = cands.cov_gaia[top], cands.cov_survey[top], cands.rho[top]
    c = model.total_cov(cg, cs)
    s2 = np.sqrt(np.maximum(c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1] ** 2, 1e-12))  # equal-area variance
    free = rho * (1.0 - (cands.claimed if cands.claimed is not None else 0.0))
    pi = np.clip(pi_t, 1e-9, 1 - 1e-9)
    level = free * threshold * (1 - pi) / (pi * (1 - threshold))
    eps, tau2 = model.tail_frac, model.tail_mas**2

    def f(r2):
        core = (1 - eps) * np.exp(-r2 / (2 * s2)) / (2 * np.pi * s2)
        tail = eps * np.exp(-r2 / (2 * (s2 + tau2))) / (2 * np.pi * (s2 + tau2)) if eps > 0 else 0.0
        return core + tail

    # f is decreasing in r^2: bisect for f(r2) = level.
    lo, hi = np.zeros(n), np.full(n, 1e12)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        above = f(mid) >= level
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    area = np.where(f(np.zeros(n)) >= level, np.pi * lo, 0.0)
    return 1.0 - np.exp(-free * area)


def best_per_unit(cands, p_pair, p_none):
    """For each unit: index of the most probable pair (-1 if none) and its P."""
    nu = len(cands.unit_target)
    best_p = np.zeros(nu)
    best_i = np.full(nu, -1)
    if len(p_pair):
        order = np.lexsort((p_pair, cands.unit))
        last = np.r_[cands.unit[order][1:] != cands.unit[order][:-1], True]
        sel = order[last]
        best_i[cands.unit[sel]] = sel
        best_p[cands.unit[sel]] = p_pair[sel]
    return best_i, best_p


def planted(cands, model, prior, rng=0):
    """Plant a synthetic counterpart in every pseudo-target unit whose parent target has a real candidate.

    The template is the parent's most likely real candidate: its Gaia and survey covariances and density.
    The planted offset is drawn from N(0, C) under the fitted model. Returns per planted unit:
    (unit index, P of the planted pair, best P among the real neighbours).
    """
    rng = np.random.default_rng(rng)
    llr, _ = cands.log_lr(model)
    real = cands.shift < 0
    if not real.any():
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    # Template per (target, group): the real pair with the highest LR.
    key = cands.unit_target[cands.unit] * (cands.unit_group.max() + 1) + cands.unit_group[cands.unit]
    order = np.lexsort((llr, key))
    order = order[real[order]]
    last = np.r_[key[order][1:] != key[order][:-1], True]
    tmpl = dict(zip(key[order][last], order[last]))
    fake_units = np.nonzero(cands.unit_shift >= 0)[0]
    ukey = cands.unit_target[fake_units] * (cands.unit_group.max() + 1) + cands.unit_group[fake_units]
    has = np.array([k in tmpl for k in ukey], bool)
    fake_units, ukey = fake_units[has], ukey[has]
    if not len(fake_units):
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    t = np.array([tmpl[k] for k in ukey])
    cg, cs, rho = cands.cov_gaia[t], cands.cov_survey[t], cands.rho[t]
    c = model.total_cov(cg.copy(), cs)
    L = np.linalg.cholesky(c)
    d = np.einsum("nij,nj->ni", L, rng.standard_normal((len(t), 2)))
    llr_plant, _ = log_lr(d, cg, cs, rho, model)
    # Units: the planted units, with their existing shifted pairs plus the planted pair.
    remap = np.full(len(cands.unit_target), -1)
    remap[fake_units] = np.arange(len(fake_units))
    in_pl = remap[cands.unit] >= 0
    all_llr = np.concatenate([llr[in_pl], llr_plant])
    all_unit = np.concatenate([remap[cands.unit[in_pl]], np.arange(len(fake_units))])
    pi_u = prior[cands.unit_target[fake_units]]
    p, _ = unit_posteriors(all_llr, all_unit, pi_u)
    n_bg = in_pl.sum()
    best_bg = np.zeros(len(fake_units))
    np.maximum.at(best_bg, all_unit[:n_bg], p[:n_bg])
    return fake_units, p[n_bg:], best_bg


def summary(cands, model, prior, stratum, *, thresholds=THRESHOLDS, seed=0):
    """False-match accounting at each threshold, overall and per stratum.

    For a threshold T:
      accepted        real units whose best candidate has P >= T
      fmr_shift       plain pseudo-targets with best P >= T: P(false accept | no counterpart)
      confusion       planted pseudo-targets whose best is a chance neighbour with P >= T:
                      P(false accept | counterpart exists)
      recall          planted pseudo-targets whose planted counterpart is accepted (P >= T and best)
      false_shift     expected false accepts from the shifted sky: sum over real units of
                      (1 - pi) fmr_shift + pi confusion
      false_model     sum over accepted of (1 - P): the model's own expected false accepts
    The two false-accept estimates agreeing is the calibration check.
    """
    p_pair, p_none, chi2, _ = posteriors(cands, model, prior)
    _, _, core = cands.log_lr(model, core_resp=True)
    best_i, best_p = best_per_unit(cands, p_pair, p_none)
    pl_units, pl_p, pl_bg = planted(cands, model, prior, seed)
    real = cands.unit_shift < 0
    fake = ~real
    st = stratum[cands.unit_target]
    pi_u = prior[cands.unit_target]
    rows = []
    for T in thresholds:
        acc = real & (best_p >= T)
        fa = fake & (best_p >= T)
        n_real, n_fake = real.sum(), fake.sum()
        fmr = fa.sum() / max(n_fake, 1)
        conf = float(np.mean((pl_bg >= T) & (pl_bg > pl_p))) if len(pl_p) else float("nan")
        rec = float(np.mean((pl_p >= T) & (pl_p >= pl_bg))) if len(pl_p) else float("nan")
        rows.append({
            "threshold": T, "units": int(n_real), "accepted": int(acc.sum()),
            "fmr_shift": float(fmr),
            "fmr_shift_err": float(np.sqrt(max(fa.sum(), 1)) / max(n_fake, 1)),
            "confusion": conf, "recall": rec,
            "false_shift": float(fmr * np.sum(1 - pi_u[real]) + np.nan_to_num(conf) * np.sum(pi_u[real])),
            "false_model": float(np.sum(1 - best_p[acc])),
        })
    per_stratum = []
    T = 0.9
    for s in np.unique(st[real]):
        r, f = real & (st == s), fake & (st == s)
        per_stratum.append({
            "stratum": int(s), "units": int(r.sum()), "pi": float(np.mean(pi_u[r])),
            "accepted_0.9": int((r & (best_p >= T)).sum()),
            "fmr_shift_0.9": float((f & (best_p >= T)).sum() / max(f.sum(), 1)),
            "false_model_0.9": float(np.sum(1 - best_p[r & (best_p >= T)])),
        })
    # chi^2 of real pairs, weighted by match probability and by membership of the Gaussian core, should be
    # chi^2_2: mean 2, and 1% above 9.21.
    w = np.where(cands.shift < 0, p_pair * core, 0.0)
    sw = w.sum()
    chi2_check = {"weight": float(sw),
                  "mean": float(np.sum(w * chi2) / sw) if sw else None, "expected_mean": 2.0,
                  "tail_9.21": float(np.sum(w * (chi2 > 9.21)) / sw) if sw else None, "expected_tail": 0.01}
    return {"thresholds": rows, "per_stratum": per_stratum, "chi2_check": chi2_check,
            "model": {"scale": model.scale, "floor_mas": model.floor_mas, "tail_frac": model.tail_frac,
                      "tail_mas": model.tail_mas}}
