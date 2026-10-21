"""What vetting receives from a detector: one Candidate per object.

Both arms emit plain dicts in this shape (docs/vetting.md §2); from_dict keeps
unknown keys in `extra` so a detector can add fields before vetting uses them.
Arrays are numpy; times are in days on any zero point (the packet builder
re-zeroes them); flux is normalised to a median of about 1.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LightCurve:
    band: str
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    # Per-epoch nuisance channels: fov, ccd, scan_angle, sector, camera, quality, background, centroid_x, ...
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        self.flux_err = np.asarray(self.flux_err, dtype=float)
        self.meta = {k: np.asarray(v) for k, v in self.meta.items()}
        n = len(self.time)
        bad = [k for k, v in [("flux", self.flux), ("flux_err", self.flux_err), *self.meta.items()] if len(v) != n]
        if bad:
            raise ValueError(f"{self.band}: arrays {bad} differ in length from time ({n})")

    @classmethod
    def from_dict(cls, d):
        return cls(band=d["band"], time=d["time"], flux=d["flux"], flux_err=d["flux_err"], meta=dict(d.get("meta", {})))

    def to_dict(self):
        return {"band": self.band, "time": self.time.tolist(), "flux": self.flux.tolist(),
                "flux_err": self.flux_err.tolist(), "meta": {k: v.tolist() for k, v in self.meta.items()}}


# The arm's own bands. A Gaia candidate may also carry ZTF or TESS light curves, named as in
# inject.lightcurve.BAND_WAVELENGTH_UM; those are evidence from other years, not the detection.
PRIMARY_BANDS = {"gaia": ("G", "BP", "RP")}


@dataclass
class Candidate:
    candidate_id: str            # pipeline-internal, never shown to Stage A ("gaia_dr3:123", "tic:456")
    arm: str                     # "gaia" or "tess"
    source: dict                 # ra, dec, phot_g_mean_mag or tmag, bp_rp, parallax, parallax_error, teff, ...
    lightcurves: list
    dips: list = field(default_factory=list)        # dicts: t_min, t_start, t_end, depth, duration, snr, ...
    scores: dict = field(default_factory=dict)      # detector name -> calibrated p or false-alarm rate
    tests: dict = field(default_factory=dict)       # FP test name -> {"statistic": x, "p": p, ...}
    context: dict = field(default_factory=dict)     # IR photometry, neighbours, cross-match, ...
    extra: dict = field(default_factory=dict)

    KNOWN = ("candidate_id", "arm", "source", "lightcurves", "dips", "scores", "tests", "context")

    @classmethod
    def from_dict(cls, d):
        return cls(
            candidate_id=str(d["candidate_id"]), arm=d["arm"], source=dict(d.get("source", {})),
            lightcurves=[lc if isinstance(lc, LightCurve) else LightCurve.from_dict(lc) for lc in d["lightcurves"]],
            dips=list(d.get("dips", [])), scores=dict(d.get("scores", {})), tests=dict(d.get("tests", {})),
            context=dict(d.get("context", {})), extra={k: v for k, v in d.items() if k not in cls.KNOWN},
        )

    def to_dict(self):
        return {"candidate_id": self.candidate_id, "arm": self.arm, "source": self.source,
                "lightcurves": [lc.to_dict() for lc in self.lightcurves], "dips": self.dips,
                "scores": self.scores, "tests": self.tests, "context": self.context, **self.extra}

    def band(self, name):
        for lc in self.lightcurves:
            if lc.band == name:
                return lc
        return None

    def primary(self):
        """The arm's own light curves; other surveys' (ZTF, TESS on a Gaia candidate) are context."""
        own = PRIMARY_BANDS.get(self.arm)
        lcs = [lc for lc in self.lightcurves if own is None or lc.band in own]
        return lcs or self.lightcurves

    def time_span(self):
        """Span of the arm's own epochs, which sets the packet clock and the tail holdout."""
        t = np.concatenate([lc.time for lc in self.primary()])
        return float(t.min()), float(t.max())
