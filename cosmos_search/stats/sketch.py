"""Bounded-memory null distributions for streaming passes (arms/gaia-dimmers.md, day one).

A full DR4 stream sees 1.8 × 10⁹ sources, so the mirror null cannot be kept as raw values. A
TailSketch keeps every value above a threshold u exactly and counts the rest in a fixed histogram.
Tail p-values, which are where thresholds are set, stay exact; p-values below u are
conservative (a straddling bin counts as exceeding). When the exact set outgrows max_exact,
u rises and the values below it move into the histogram, so memory stays bounded however many
sources pass through.

Sketches built on different files or sessions merge exactly, provided they share histogram edges.
The GPD extrapolation matches stats.tail_p on the full null as long as the exact set holds the
top tail_fraction of all values, so size max_exact to at least 1% of the stratum's expected count
(or accept a fit on fewer points).
"""

import numpy as np

from .nulls import gpd_tail_p


class TailSketch:
    def __init__(self, edges, max_exact=100_000):
        self.edges = np.asarray(edges, dtype=float)
        if self.edges.ndim != 1 or self.edges.size < 2 or np.any(np.diff(self.edges) <= 0):
            raise ValueError("edges must be a strictly increasing 1-d array")
        self.max_exact = int(max_exact)
        # Bin 0 is (-inf, edges[0]), bin i is [edges[i-1], edges[i]), the last bin is [edges[-1], inf).
        self.counts = np.zeros(self.edges.size + 1, dtype=np.int64)
        self.u = -np.inf
        self._exact = np.empty(0)
        self._pending = []

    @property
    def n(self):
        return int(self.counts.sum()) + self.exact.size

    @property
    def exact(self):
        self._flush()
        return self._exact

    def add(self, values):
        v = np.asarray(values, dtype=float).ravel()
        v = v[np.isfinite(v)]
        above = v >= self.u
        self._bin(v[~above])
        self._pending.append(v[above])
        if sum(a.size for a in self._pending) > self.max_exact:
            self._flush()
        return self

    def merge(self, other):
        if not np.array_equal(self.edges, other.edges):
            raise ValueError("cannot merge sketches with different edges")
        self.counts += other.counts
        u = max(self.u, other.u)
        mine, theirs = self.exact, other.exact
        for v in (mine, theirs):
            self._bin(v[v < u])
        self.u = u
        self._exact = np.sort(np.concatenate([mine[mine >= u], theirs[theirs >= u]]))
        self._shrink()
        return self

    def p(self, stat, min_exceed=10, tail_fraction=0.01, min_tail=100, conservative=True, xi_quantile=0.9):
        """Upper-tail p-values, as stats.tail_p. Returns (p, extrapolated)."""
        s = np.atleast_1d(np.asarray(stat, dtype=float))
        exact, n = self.exact, self.n
        exceed = exact.size - np.searchsorted(exact, s, side="left")
        below_u = s < self.u
        if below_u.any():
            # Every histogrammed value is < u; count whole bins from the one holding s upwards.
            tail = np.cumsum(self.counts[::-1])[::-1]
            j = np.searchsorted(self.edges, s[below_u], side="right")
            exceed[below_u] += tail[j]
        p = (1.0 + exceed) / (1.0 + n)
        far = exceed < min_exceed
        if far.any() and exact.size > min_tail:
            m = min(exact.size - 1, max(min_tail, int(np.ceil(tail_fraction * n))))
            p[far] = gpd_tail_p(s[far], exact[exact.size - m - 1:], n, conservative, xi_quantile)
        elif far.any():
            far[:] = False  # too few exact values to fit a tail; keep the empirical p
        return p.reshape(np.shape(stat)), far.reshape(np.shape(stat))

    def to_dict(self):
        return {"edges": self.edges.tolist(), "max_exact": self.max_exact, "counts": self.counts.tolist(),
                "u": None if np.isinf(self.u) else float(self.u), "exact": self.exact.tolist()}

    @classmethod
    def from_dict(cls, d):
        sk = cls(d["edges"], d["max_exact"])
        sk.counts = np.asarray(d["counts"], dtype=np.int64)
        sk.u = -np.inf if d["u"] is None else float(d["u"])
        sk._exact = np.sort(np.asarray(d["exact"], dtype=float))
        return sk

    def _flush(self):
        if self._pending:
            self._exact = np.sort(np.concatenate([self._exact, *self._pending]))
            self._pending = []
            self._shrink()

    def _bin(self, v):
        if v.size:
            self.counts += np.bincount(np.searchsorted(self.edges, v, side="right"), minlength=self.counts.size)

    def _shrink(self):
        if self._exact.size <= self.max_exact:
            return
        keep = self.max_exact // 2
        cut = self._exact.size - keep
        # Raise u to the smallest kept value; ties with it stay exact so u splits the data cleanly.
        self.u = self._exact[cut]
        cut = np.searchsorted(self._exact, self.u, side="left")
        self._bin(self._exact[:cut])
        self._exact = self._exact[cut:]
