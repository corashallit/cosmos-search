"""Population-level statistics: things one light curve cannot tell you about itself.

- Common-mode epochs. An astrophysical dip on one star is independent of the stars around it on the
  detector; scattered light, pointing jitter and calibration steps are not. For each detector group
  (TESS sector/camera/CCD; for Gaia, a scan or a focal-plane region), count how many light curves have an
  event in each time bin and flag bins where the count is improbable under independence.
- (Empirical false-alarm rates live in cosmos_search.stats, shared by every arm.)
- Population anomaly score. Robustly standardised, noise-normalised features, scored by an isolation
  forest and by k-nearest-neighbour distance and averaged by rank. This asks how unusual a light curve is
  relative to everything like it, without a template of what to look for.
"""

import numpy as np
from scipy import stats as sps

ANOMALY_FEATURES_PREFIXES = ("min_stat_", "max_stat_", "log_beta_", "tail_ratio_")
ANOMALY_FEATURES = (
    "qskew_10", "qskew_1", "low_tail_sigma", "high_tail_sigma", "cody_M", "bimodality",
    "von_neumann", "Q", "ls_power", "rms_over_white",
)


def common_mode_epochs(events, n_per_group, bin_width=0.25, alpha=1e-3, min_count=3):
    """Flag (group, time-bin) cells with improbably many light curves in an event.

    events: iterable of (group, lc_id, t_start, t_end); use a low detection threshold here so the test
    sees sub-threshold structure. n_per_group: {group: number of light curves searched in that group}.
    Returns {group: {"bins": set of bin indices, "frac": {bin: fraction of light curves}, "p0": rate}}.
    """
    by_group = {}
    for g, lc_id, t0, t1 in events:
        b0, b1 = int(np.floor(t0 / bin_width)), int(np.floor(t1 / bin_width))
        cells = by_group.setdefault(g, {})
        for b in range(b0, b1 + 1):
            cells.setdefault(b, set()).add(lc_id)
    out = {}
    for g, cells in by_group.items():
        n = int(n_per_group.get(g, 0))
        if n < 20:
            out[g] = {"bins": set(), "frac": {}, "p0": np.nan, "n": n, "tested": False}
            continue
        counts = {b: len(s) for b, s in cells.items()}
        # Background rate: typical occupancy of occupied bins, floored at one light curve per bin.
        p0 = max(np.median(list(counts.values())) / n, 1.0 / n)
        n_bins = max(len(counts), 1)
        flagged = {b for b, k in counts.items()
                   if k >= min_count and sps.binom.sf(k - 1, n, p0) < alpha / n_bins}
        out[g] = {"bins": flagged, "frac": {b: counts[b] / n for b in flagged}, "p0": p0, "n": n, "tested": True}
    return out


def in_common_mode(cm, group, t0, t1, bin_width=0.25):
    """Largest coincidence fraction of any flagged bin overlapping [t0, t1], or 0."""
    g = cm.get(group)
    if not g or not g["bins"]:
        return 0.0
    bins = range(int(np.floor(t0 / bin_width)), int(np.floor(t1 / bin_width)) + 1)
    return max((g["frac"][b] for b in bins if b in g["bins"]), default=0.0)


def anomaly_columns(columns):
    cols = [c for c in columns if c.startswith(ANOMALY_FEATURES_PREFIXES)]
    return cols + [c for c in ANOMALY_FEATURES if c in columns]


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


class AnomalyModel:
    """Isolation forest + kNN distance on robustly standardised features, standardised within strata.

    fit() on the real population; score() any rows (real, injected, mirrored) against it. Scores are
    the mean of the two methods' ranks against the fitted population, so 1.0 means more unusual than
    everything it was fitted on.
    """

    def __init__(self, columns, k=10, seed=0):
        self.columns = list(columns)
        self.k = k
        self.seed = seed

    def _transform(self, X, strata):
        Z = _slog(np.asarray(X, float))
        out = np.empty_like(Z)
        for s in np.unique(strata):
            sel = strata == s
            med, mad = self.loc_.get(s, self.loc_all_)
            out[sel] = (Z[sel] - med) / mad
        return np.nan_to_num(np.clip(out, -50, 50))

    def fit(self, X, strata):
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import NearestNeighbors

        Z = _slog(np.asarray(X, float))
        strata = np.asarray(strata)

        def loc(A):
            med = np.nanmedian(A, axis=0)
            mad = 1.4826 * np.nanmedian(np.abs(A - med), axis=0)
            return np.nan_to_num(med), np.where(np.isfinite(mad) & (mad > 0), mad, 1.0)

        self.loc_all_ = loc(Z)
        self.loc_ = {s: loc(Z[strata == s]) for s in np.unique(strata) if (strata == s).sum() >= 50}
        T = self._transform(X, strata)
        self.forest_ = IsolationForest(n_estimators=300, random_state=self.seed).fit(T)
        self.nn_ = NearestNeighbors(n_neighbors=self.k + 1).fit(T)
        self.ref_if_ = np.sort(-self.forest_.score_samples(T))
        self.ref_knn_ = np.sort(self.nn_.kneighbors(T)[0][:, 1:].mean(axis=1))
        return self

    def score(self, X, strata, in_fit=False):
        T = self._transform(X, np.asarray(strata))
        a = -self.forest_.score_samples(T)
        d, _ = self.nn_.kneighbors(T)
        # A row that was in the fit set is its own nearest neighbour; skip it.
        knn = d[:, 1:].mean(axis=1) if in_fit else d[:, :-1].mean(axis=1)
        return 0.5 * (_rank(self.ref_if_, a) + _rank(self.ref_knn_, knn))


def _rank(ref, x):
    """Quantile of x in ref; beyond the reference maximum, 1 + log(x / max) so extremes stay ordered."""
    r = np.searchsorted(ref, x, side="right") / len(ref)
    top = ref[-1]
    beyond = x > top
    if top > 0:
        r[beyond] = 1 + np.log(x[beyond] / top)
    return r
