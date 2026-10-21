"""A survey-agnostic light curve: irregular times, relative flux, errors, flags.

Both arms reduce to this. TESS gives one band at fixed cadence with gaps; Gaia gives three bands at
irregular epochs over years. Everything downstream takes a LightCurve and nothing else, so the detector
never sees which survey it is running on.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LightCurve:
    t: np.ndarray                 # days; BTJD (BJD_TDB - 2457000) for both arms
    flux: np.ndarray              # relative flux, median ~1
    err: np.ndarray               # 1-sigma, same units as flux
    flags: np.ndarray = None      # int bitmask; 0 is good. Bit meanings are per-survey (see FLAG_*)
    band: str = ""                # name as in cosmos_search.inject.lightcurve.BAND_WAVELENGTH_UM
    meta: dict = field(default_factory=dict)
    # Optional per-epoch auxiliary series, aligned with t: background, centroids, alternate apertures.
    aux: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t = np.asarray(self.t, float)
        self.flux = np.asarray(self.flux, float)
        self.err = np.asarray(self.err, float)
        self.flags = np.zeros(len(self.t), np.int64) if self.flags is None else np.asarray(self.flags, np.int64)
        if not (len(self.t) == len(self.flux) == len(self.err) == len(self.flags)):
            raise ValueError("t, flux, err, flags must have equal length")

    def __len__(self):
        return len(self.t)

    def select(self, mask):
        """A new LightCurve keeping only `mask`; aux series follow."""
        return LightCurve(
            self.t[mask], self.flux[mask], self.err[mask], self.flags[mask], self.band, dict(self.meta),
            {k: np.asarray(v)[mask] for k, v in self.aux.items()},
        )

    def clean(self, bad_bits=~0):
        """Keep finite points whose flags share no bit with `bad_bits`, sorted by time, flux normalised to median 1."""
        ok = np.isfinite(self.t) & np.isfinite(self.flux) & ((self.flags & bad_bits) == 0)
        lc = self.select(ok)
        order = np.argsort(lc.t, kind="stable")
        lc = lc.select(order)
        med = np.median(lc.flux) if len(lc) else 1.0
        if med > 0:
            lc.flux = lc.flux / med
            lc.err = lc.err / med
        bad_err = ~np.isfinite(lc.err) | (lc.err <= 0)
        if bad_err.any():
            lc.err[bad_err] = robust_std(np.diff(lc.flux)) / np.sqrt(2) if len(lc) > 2 else 1.0
        return lc

    def with_flux(self, flux):
        """Same epochs and metadata, different flux (used by injection and the inverted null)."""
        return LightCurve(self.t, np.asarray(flux, float), self.err, self.flags, self.band, dict(self.meta),
                          dict(self.aux))

    def inverted(self):
        """Mirror flux about 1. Occultation is one-sided, so dips found in the mirror image are an empirical
        null: they measure how often noise, systematics and stellar variability alone produce 'dips'."""
        out = self.with_flux(2.0 - self.flux)
        out.aux = {k: (2 * np.nanmedian(v) - np.asarray(v) if k.startswith("flux_") else v) for k, v in self.aux.items()}
        return out


def robust_std(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def segments(t, max_gap):
    """Index slices of runs with no gap longer than `max_gap` days."""
    if len(t) == 0:
        return []
    breaks = np.flatnonzero(np.diff(t) > max_gap) + 1
    edges = np.concatenate([[0], breaks, [len(t)]])
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]
