"""Pass criteria for rule adoption, and the batch sizes that give them power (review finding R4).

Earlier batches adopted a rule on point thresholds over 12-16 decoys and 14 controls ("recall >= 0.5",
"<= 1 control flagged"), which pass a mediocre rule a large share of the time. Here a criterion is a one-sided
Clopper-Pearson bound, so the chance of passing a rule that is actually at the bad level is at most alpha
whatever n is, and n is chosen so a rule at the good level passes with the stated power:

- recall: pass if the lower bound on recall is >= r_bad;
- false shortlisting: pass if the upper bound on the rate is <= f_bad.

    python -m cosmos_search.vetting.power      # prints the design table used in docs/dayone.md
"""

import numpy as np
from scipy import stats


def cp_lower(k, n, alpha=0.05):
    """One-sided Clopper-Pearson lower bound on a binomial rate."""
    k = np.asarray(k)
    return np.where(k > 0, stats.beta.ppf(alpha, np.maximum(k, 1), n - k + 1), 0.0)


def cp_upper(k, n, alpha=0.05):
    """One-sided Clopper-Pearson upper bound on a binomial rate."""
    k = np.asarray(k)
    return np.where(k < n, stats.beta.ppf(1 - alpha, k + 1, np.maximum(n - k, 1)), 1.0)


def pass_lower(n, p, bound, alpha=0.05):
    """P(lower bound >= bound) when the true rate is p: the chance a recall criterion passes."""
    k = np.arange(n + 1)
    return float(stats.binom.pmf(k, n, p)[cp_lower(k, n, alpha) >= bound].sum())


def pass_upper(n, p, bound, alpha=0.05):
    """P(upper bound <= bound) when the true rate is p: the chance a false-rate criterion passes."""
    k = np.arange(n + 1)
    return float(stats.binom.pmf(k, n, p)[cp_upper(k, n, alpha) <= bound].sum())


def _smallest(ok, n_max, margin=20):
    """Smallest n from which ok(n) holds for the next `margin` sizes too: binomial power saws up and down with n,
    and a lucky n is not a design."""
    run = 0
    for n in range(5, n_max):
        run = run + 1 if ok(n) else 0
        if run > margin:
            return n - margin
    return None


def n_for_recall(r_good, r_bad, power=0.8, alpha=0.05, n_max=2000):
    """Smallest n at which a rule with recall r_good passes "lower bound >= r_bad" with the given power."""
    return _smallest(lambda n: pass_lower(n, r_good, r_bad, alpha) >= power, n_max)


def n_for_rate(f_good, f_bad, power=0.8, alpha=0.05, n_max=5000):
    """Smallest n at which a rule with false rate f_good passes "upper bound <= f_bad" with the given power."""
    return _smallest(lambda n: pass_upper(n, f_good, f_bad, alpha) >= power, n_max)


def old_criteria():
    """How often the batch 2-4 point criteria pass, for comparison (docs/dayone.md)."""
    rec = {p: float(stats.binom.sf(5, 12, p)) for p in (0.3, 0.35, 0.4, 0.5, 0.6)}        # >= 6 of 12
    ctl = {f: float(stats.binom.cdf(1, 14, f)) for f in (0.03, 0.05, 0.1, 0.15, 0.2)}       # <= 1 of 14
    return {"recall_ge_6_of_12": rec, "controls_le_1_of_14": ctl}


def main():
    print("old point criteria, pass probability by true rate:")
    for k, v in old_criteria().items():
        print(" ", k, {p: round(x, 2) for p, x in v.items()})
    print("recall criteria (lower 95% bound >= r_bad), n for 80% power at r_good:")
    for good, bad in ((0.5, 0.3), (0.5, 0.35), (0.6, 0.4), (0.4, 0.25)):
        print(f"  r_good {good}, r_bad {bad}: n = {n_for_recall(good, bad)}")
    print("false-rate criteria (upper 95% bound <= f_bad), n for 80% power at f_good:")
    for good, bad in ((0.03, 0.10), (0.05, 0.10), (0.05, 0.15), (0.02, 0.08), (0.10, 0.25)):
        print(f"  f_good {good}, f_bad {bad}: n = {n_for_rate(good, bad)}")


if __name__ == "__main__":
    main()
