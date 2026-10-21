"""Completeness maps from injection-recovery runs (statistics.md §4).

A run gives, per injection, its truth parameters and whether the detector recovered it. Matching
a detection to an injection is detector-specific and happens before this; here we only bin.
"""

from dataclasses import dataclass

import numpy as np


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n trials. Empty bins give (0, 1)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(n > 0, k / n, 0.0)
        denom = 1 + z**2 / n
        centre = (f + z**2 / (2 * n)) / denom
        half = z * np.sqrt(f * (1 - f) / n + z**2 / (4 * n**2)) / denom
    lo = np.where((n > 0) & (k > 0), np.clip(centre - half, 0, 1), 0.0)
    hi = np.where((n > 0) & (k < n), np.clip(centre + half, 0, 1), 1.0)
    return lo, hi


@dataclass
class Completeness:
    names: list
    edges: list
    n: np.ndarray
    k: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    @property
    def fraction(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.n > 0, self.k / self.n, np.nan)

    def rows(self):
        """One dict per bin, for tables and plots."""
        out = []
        for idx in np.ndindex(self.n.shape):
            row = {}
            for d, name in enumerate(self.names):
                row[f"{name}_lo"] = float(self.edges[d][idx[d]])
                row[f"{name}_hi"] = float(self.edges[d][idx[d] + 1])
            row.update(n=int(self.n[idx]), k=int(self.k[idx]), fraction=float(self.fraction[idx]),
                       ci_lo=float(self.lo[idx]), ci_hi=float(self.hi[idx]))
            out.append(row)
        return out


def completeness(params, recovered, edges, z=1.96):
    """Recovered fraction binned over one or more truth parameters.

    params: {name: array}, edges: {name: bin edges}, recovered: bool array. Injections outside the
    edges are dropped, so pick edges that span the family's prior.
    """
    names = list(edges)
    sample = np.column_stack([np.asarray(params[name], dtype=float) for name in names])
    bins = [np.asarray(edges[name], dtype=float) for name in names]
    rec = np.asarray(recovered, dtype=bool)
    n, _ = np.histogramdd(sample, bins=bins)
    k, _ = np.histogramdd(sample[rec], bins=bins)
    lo, hi = wilson(k, n, z)
    return Completeness(names, bins, n.astype(int), k.astype(int), lo, hi)
