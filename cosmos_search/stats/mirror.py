"""Mirror nulls: using the bright side of every light curve as the null for its faint side (statistics.md §2).

Two ways to use the mirror, with different assumptions:

- **Per-source competition** (knockoff+, Barber & Candès 2015; target–decoy competition in proteomics).
  Each source's dimming statistic competes with its own brightening statistic. Under the null for that
  source (instrument noise symmetric about its median), which side wins is a fair coin, so counting
  bright winners above a threshold estimates the false discoveries among dim winners. FDR control is
  finite-sample, and no exchangeability *between* sources is needed, so magnitude, epoch count and sky
  position cannot leak in through the null. This is the default.
- **Cross-source mirror p-values** (`stats.stratified_p` with the bright side as the null). These give a
  per-source p, which ranking and combination need, but they assume sources in a stratum are exchangeable.

Astrophysical brightening (flares, cosmic rays, pulsators with sharp maxima) only makes the bright side
heavier, so both are conservative for dimming where they are wrong. `tail_ratio` measures how far the
symmetry assumption holds, per stratum, before either is trusted.
"""

import numpy as np


def competition(dim, bright):
    """Signed competition score W: +max if the dim side wins, -max if the bright side wins, 0 on ties."""
    dim = np.asarray(dim, dtype=float)
    bright = np.asarray(bright, dtype=float)
    return np.where(dim > bright, dim, np.where(bright > dim, -bright, 0.0))


def knockoff_q(dim, bright, offset=1):
    """q-value of every source under per-source competition (knockoff+ with offset=1).

    For threshold t, FDR-hat(t) = (offset + #{W <= -t}) / max(1, #{W >= t}); a source's q is the smallest
    FDR-hat over thresholds at or below its own W. Sources whose bright side wins get q = 1. With
    offset=1 no q falls below 1 / (number of dim winners), so q <= 0.01 needs at least 100 discoveries.
    """
    w = competition(dim, bright)
    finite = np.isfinite(w)
    q = np.ones(w.shape)
    pos = np.sort(w[finite & (w > 0)])[::-1]
    neg = np.sort(-w[finite & (w < 0)])[::-1]
    if pos.size == 0:
        return q
    # For each positive W (descending), how many dim winners are >= it and how many bright winners.
    n_pos = np.arange(1, pos.size + 1)
    n_neg = neg.size - np.searchsorted(neg[::-1], pos, side="left")
    fdr = (offset + n_neg) / n_pos
    # q at threshold pos[k] is the min FDR over thresholds <= pos[k], i.e. over indices >= k.
    qs = np.minimum.accumulate(fdr[::-1])[::-1]
    qs = np.minimum(qs, 1.0)
    order = np.argsort(-w[finite & (w > 0)], kind="stable")
    idx = np.flatnonzero(finite & (w > 0))[order]
    q[idx] = qs
    return q


def stratified_knockoff_q(dim, bright, strata, offset=1):
    """knockoff+ q-values computed separately in each stratum; select the union at the target level.

    Pooled competition needs the per-source coin to be fair at the *same threshold* in every source. Where
    the noise is asymmetric in some strata (faint stars: a dip in robust sigma is capped by S/N, while
    cosmic rays and neighbours spike freely), their bright winners swamp the top of the pooled list and cancel
    real dimmers elsewhere. Per-stratum thresholds confine that to the strata where it happens.

    Selecting {q <= alpha} here gives, in each stratum s, FDP-hat_s = (offset + N_s^-) / N_s^+ <= alpha, so the
    pooled estimate sum(offset + N_s^-) / sum(N_s^+) <= alpha too (a ratio of sums is bounded by its largest
    ratio). Under a pure null a stratum selects anything only if its top 1/alpha winners are all dim, with
    probability ~ 2^(-1/alpha), so the union over many null strata stays controlled
    (tests/test_stats.py measures it). Each stratum still pays the knockoff+ floor of 1/alpha discoveries.
    """
    dim = np.asarray(dim, dtype=float)
    bright = np.asarray(bright, dtype=float)
    _, code = np.unique(np.asarray(strata), return_inverse=True)
    q = np.ones(dim.shape)
    order = np.argsort(code, kind="stable")
    bounds = np.flatnonzero(np.diff(code[order])) + 1
    for idx in np.split(order, bounds):
        q[idx] = knockoff_q(dim[idx], bright[idx], offset)
    return q


def tail_ratio(dim, bright, strata, thresholds):
    """Symmetry diagnostic: per stratum and threshold, counts of dim >= t and bright >= t.

    Returns a list of dicts with the ratio and a ±1σ range from Poisson counts. Under a symmetric null
    the ratio is ~1; above 1 is excess dimming (astrophysical or instrumental), below 1 is a heavier bright
    tail, which makes the mirror conservative for dimming.
    """
    dim = np.asarray(dim, dtype=float)
    bright = np.asarray(bright, dtype=float)
    strata = np.asarray(strata)
    rows = []
    for label in np.unique(strata):
        here = strata == label
        d, b = np.sort(dim[here]), np.sort(bright[here])
        for t in thresholds:
            nd = d.size - np.searchsorted(d, t, side="left")
            nb = b.size - np.searchsorted(b, t, side="left")
            ratio = (nd + 0.5) / (nb + 0.5)
            rel = np.sqrt(1 / (nd + 0.5) + 1 / (nb + 0.5))
            rows.append({"stratum": label, "threshold": float(t), "n": int(here.sum()), "n_dim": int(nd),
                         "n_bright": int(nb), "ratio": float(ratio),
                         "ratio_lo": float(ratio * np.exp(-rel)), "ratio_hi": float(ratio * np.exp(rel))})
    return rows
