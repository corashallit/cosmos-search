"""Competition between Gaia sources for the same survey source.

The per-target posterior (match.py) asks "which nearby survey source is this star?" as if no other star
could want the same survey source. In crowded fields, and for any survey with a beam much wider than
Gaia's resolution, that is wrong: a faint Gaia star 3 arcsec from a bright one will happily claim the bright
star's WISE detection. The shifted sky catches this (its pseudo-targets do the same), so the false-match
estimates disagree; this module removes the cause.

Model. Each survey source is the counterpart of at most one resolved Gaia source. Claimants are every Gaia
source near the survey source, not just our targets: the targets, plus Gaia neighbours fetched from the
Gaia catalogue itself.

Which claimants compete. A survey source 0.8 arcsec from a faint target and 0.07 arcsec from a bright Gaia
neighbour is the neighbour's, even though 2MASS cannot resolve the pair: its position (good to ~0.1 arcsec)
says whose light dominates it. Letting the target share it would hand the target the neighbour's flux,
which is exactly how a spurious IR excess is made. So two claimants share a row only when its position
cannot tell them apart: their offsets differ by less than SHARE_SIGMA combined positional sigmas (and by
less than half the FWHM, the survey's resolution). Those rows are genuine blends of both, and the blend
columns (blend.py, survey multiplicity) say so; a target merely inside a neighbour's beam gets no match of
its own, and blend.py reports how much of the beam the neighbour fills.

Approximation. Exact marginals of a bipartite matching need a permanent. We use belief propagation for
matching (below), which is exact on trees and close on the small clusters crowded fields produce; mean-field
double-counts and was ~1.7x overconfident on synthetic fields. Pseudo-targets (shifted sky) compete
against the real claimants but are not claimants in the real sky, exactly as a star added to the real sky
would be.
"""

import numpy as np
import pandas as pd

from .match import log_lr, unit_posteriors

DAMPING = 0.5
ITERATIONS = 8
SHARE_SIGMA = 2.0  # claimants closer than this many positional sigmas share a row instead of competing


def competition(cands, nbrs, model, prior_t, prior_nb, blend_radius_mas, *, region=None,
                iterations=ITERATIONS):
    """log competition factor per candidate pair (aligned with cands pairs), and the claimed fraction of the
    background per target.

    prior_t: pi per target; prior_nb: pi per neighbour pair (from its G stratum).
    region: an integer per target (e.g. a coarse HEALPix pixel) over which the claimed fraction of the
    background is averaged; one region if None.

    The background density rho counted every survey source, including those that are other Gaia stars'
    counterparts. Competition now accounts for those explicitly, so the chance-alignment density must
    count only the unclaimed part: rho_free = rho (1 - claimed), where `claimed` is the mean real-sky
    occupancy of rows found by the pseudo-targets, which sample the sky at random.
    """
    real = cands.shift < 0
    base, _ = log_lr(cands.d, cands.cov_gaia, cands.cov_survey, cands.rho, model)
    nb_llr, _ = log_lr(nbrs.d, nbrs.cg, nbrs.cs, nbrs.rho, model) if len(nbrs) else (np.zeros(0), None)

    # Real-sky claimants: real target pairs, then neighbour pairs. Units: (claimant, survey group).
    t_idx, f_idx = np.nonzero(real)[0], np.nonzero(~real)[0]
    n_t, n_n = len(t_idx), len(nbrs)
    rid = np.concatenate([cands.rid[t_idx], nbrs.rid]).astype(str)
    f_rid = cands.rid[f_idx].astype(str)
    codes_all, _ = pd.factorize(np.concatenate([rid, f_rid]))
    codes, fc = codes_all[:len(rid)], codes_all[len(rid):]
    n_codes = int(codes_all.max()) + 1 if len(codes_all) else 0
    d = np.vstack([cands.d[t_idx], nbrs.d]) if n_n else cands.d[t_idx]
    nb_unit_key = (pd.factorize(np.char.add(nbrs.key.astype(str), np.char.add("|", nbrs.gkey.astype(str))))[0]
                   if n_n else np.zeros(0, int))
    t_units, t_inv = np.unique(cands.unit[t_idx], return_inverse=True)
    unit = np.concatenate([t_inv, len(t_units) + nb_unit_key]).astype(np.int64)
    pi_unit = np.concatenate([prior_t[cands.unit_target[t_units]],
                              pd.Series(prior_nb).groupby(nb_unit_key).first().to_numpy() if n_n else np.zeros(0)])
    llr0 = np.concatenate([base[t_idx], nb_llr])

    # Region of each row: that of a target touching it (every touched row has one).
    reg_t = np.zeros(cands.n_targets, np.int64) if region is None else np.asarray(region, np.int64)
    _, reg_codes = np.unique(reg_t, return_inverse=True)
    row_reg = np.zeros(n_codes, np.int64)
    pair_codes = np.concatenate([codes[:n_t], fc])
    pair_t = cands.target[np.concatenate([t_idx, f_idx])]
    row_reg[pair_codes] = reg_codes[pair_t]
    f_reg = reg_codes[cands.target[f_idx]]
    n_reg = int(reg_codes.max()) + 1 if len(reg_codes) else 1

    # Competitor pairs on the same row whose claimants the row's position tells apart.
    cg = np.concatenate([cands.cov_gaia[t_idx], nbrs.cg]) if n_n else cands.cov_gaia[t_idx]
    cs = np.concatenate([cands.cov_survey[t_idx], nbrs.cs]) if n_n else cands.cov_survey[t_idx]
    var = _pos_var(model, cg, cs)
    f_var = _pos_var(model, cands.cov_gaia[f_idx], cands.cov_survey[f_idx])
    ci, cj = _same_row_pairs(codes, codes, d, d, var, var, blend_radius_mas, exclude_self=True)
    fi, mj = _same_row_pairs(fc, codes, cands.d[f_idx], d, f_var, var, blend_radius_mas)

    # Belief propagation for matching with a "no counterpart" option on each claimant:
    #   mu_e  = pi LR_e / ((1 - pi) + pi sum_{k != e in unit} LR_k nu_k)   claimant -> row (odds it takes the row)
    #   nu_e  = 1 / (1 + sum_{competitors e' on the row} mu_e')           row -> claimant (row is free)
    # and the belief P_e = pi LR_e nu_e / ((1 - pi) + pi sum_k LR_k nu_k). On a tree this is exact; mean-field
    # (nu = 1 - sum of competitors' beliefs) double-counts and is overconfident.
    n_e = len(llr0)
    pi_e = pi_unit[unit]
    log_nu = np.zeros(n_e)
    claimed = np.zeros(n_reg)
    log_mu = np.full(n_e, -np.inf)
    for _ in range(iterations):
        bg = -np.log1p(-claimed[row_reg[codes]])
        lw = np.minimum(llr0 + bg + log_nu, 700.0)  # log(LR_e nu_e)
        s_u = np.bincount(unit, weights=np.exp(lw), minlength=len(pi_unit))
        others = np.maximum(s_u[unit] - np.exp(lw), 0.0)
        new_mu = np.log(pi_e) + np.minimum(llr0 + bg, 700.0) - np.log((1 - pi_e) + pi_e * others)
        log_mu = new_mu if not np.isfinite(log_mu).any() else DAMPING * log_mu + (1 - DAMPING) * new_mu
        sum_mu = np.bincount(ci, weights=np.exp(log_mu[cj]), minlength=n_e)
        log_nu = -np.log1p(sum_mu)
        if len(f_idx):
            # Claimed fraction of the background per region: the probability that a row found by a
            # pseudo-target (rows sampled at random) is taken by a real claimant resolved from it. Claims by
            # unresolved claimants are shared, not competed for, so they stay in the background.
            f_sum = np.bincount(fi, weights=np.exp(log_mu[mj]), minlength=len(f_idx))
            taken = f_sum / (1.0 + f_sum)
            num = np.bincount(f_reg, weights=taken, minlength=n_reg)
            den = np.bincount(f_reg, minlength=n_reg)
            new = np.where(den > 0, num / np.maximum(den, 1), 0.0)
            claimed = np.clip(DAMPING * claimed + (1 - DAMPING) * new, 0.0, 0.95)
    bg = -np.log1p(-claimed[row_reg[codes]])

    out = np.zeros(len(cands.d))
    out[t_idx] = (log_nu + bg)[:n_t]
    if len(f_idx):
        f_sum = np.bincount(fi, weights=np.exp(log_mu[mj]), minlength=len(f_idx))
        out[f_idx] = -np.log1p(f_sum) - np.log1p(-claimed[f_reg])
    return out, claimed[reg_codes]


def _pos_var(model, cov_gaia, cov_survey):
    """Per pair: the mean per-axis variance of the offset under the core error model (mas^2)."""
    c = model.total_cov(cov_gaia.copy(), cov_survey)
    return 0.5 * (c[:, 0, 0] + c[:, 1, 1])


def _same_row_pairs(codes_a, codes_b, d_a, d_b, var_a, var_b, max_share, exclude_self=False):
    """Index pairs (i into a, j into b) on the same row whose claimants compete: offsets that differ by at
    least SHARE_SIGMA combined sigmas, or by at least max_share (half the FWHM) whatever the errors."""
    a = pd.DataFrame({"code": codes_a, "i": np.arange(len(codes_a))})
    b = pd.DataFrame({"code": codes_b, "j": np.arange(len(codes_b))})
    a, b = a[a.code >= 0], b[b.code >= 0]
    m = a.merge(b, on="code")
    i, j = m.i.to_numpy(), m.j.to_numpy()
    sep = np.hypot(*(d_a[i] - d_b[j]).T)
    share = np.minimum(max_share, SHARE_SIGMA * np.sqrt(var_a[i] + var_b[j]))
    keep = sep >= share
    if exclude_self:
        keep &= i != j
    return i[keep], j[keep]
