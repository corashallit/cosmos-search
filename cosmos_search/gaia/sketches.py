"""Per-stratum tail sketches of the mirror null, built while streaming bulk files (arms/gaia-dimmers.md).

Each file gets one stats.TailSketch per G stratum and per side: "dim" holds -min_z_rob and "bright" holds
max_z_rob, so the bright side is the mirror null for the dim side and vice versa. Sources with fewer than
MIN_TRANSITS G transits are left out, as in the TAP prefilter.

Within a file, every value is kept until the file ends; then each sketch keeps the top KEEP_FRACTION of its
stratum exactly (at least MIN_EXACT values) and histograms the rest. Merging keeps values above the larger
threshold, so a merged stratum holds about its top KEEP_FRACTION exactly while files are of similar size, which
covers the 1% the GPD fit needs (stats.sketch). Sketches are written gzipped, since most histogram bins
are empty.
"""

import gzip
import json

import numpy as np
import pandas as pd

from cosmos_search.stats import TailSketch

G_ZP = 25.6874
G_BIN = 0.5
EDGES = np.linspace(0.0, 50.0, 1001)
MIN_TRANSITS = 10
MIN_EXACT = 100
KEEP_FRACTION = 0.02
SIDES = ("dim", "bright")


def stratum_key(side, g_lo):
    return f"{side}|G={g_lo:.1f}"


class MirrorSketches:
    """Accumulates one file's features; to_dict() returns {stratum key: TailSketch.to_dict()}."""

    def __init__(self):
        self._parts = []

    def add(self, feat):
        f = feat[feat["n_g"] >= MIN_TRANSITS]
        with np.errstate(divide="ignore", invalid="ignore"):
            g = G_ZP - 2.5 * np.log10(f["med_flux_g"].to_numpy(dtype=float))
        self._parts.append(pd.DataFrame({"g_lo": np.floor(g / G_BIN) * G_BIN,
                                         "dim": -f["min_z_rob"].to_numpy(dtype=float),
                                         "bright": f["max_z_rob"].to_numpy(dtype=float)}))
        return self

    def sketches(self):
        if not self._parts:
            return {}
        df = pd.concat(self._parts, ignore_index=True)
        out = {}
        for g_lo, grp in df[np.isfinite(df["g_lo"])].groupby("g_lo"):
            for side in SIDES:
                # TailSketch halves its exact set when it overflows, so max_exact is twice what we keep.
                sk = TailSketch(EDGES, 2 * max(MIN_EXACT, int(np.ceil(KEEP_FRACTION * len(grp)))))
                out[stratum_key(side, g_lo)] = sk.add(grp[side].to_numpy())
        return out

    def to_dict(self):
        return {k: sk.to_dict() for k, sk in self.sketches().items()}


def write(path, d):
    with gzip.open(path, "wt") as fh:
        json.dump(d, fh)


def read(path):
    with gzip.open(path, "rt") as fh:
        return {k: TailSketch.from_dict(v) for k, v in json.load(fh).items()}


def merge(paths, max_exact=100_000):
    """Merge per-file sketch files into one TailSketch per stratum."""
    merged = {}
    for p in paths:
        for k, sk in read(p).items():
            if k not in merged:
                merged[k] = TailSketch(EDGES, max_exact)
            merged[k].merge(sk)
    return merged
