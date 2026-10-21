"""Signals to inject into real light curves, shared by the Gaia and TESS arms.

Every signal is a multiplicative flux factor k(t, band), so injecting into a
real light curve is f' = k f and keeps the real cadence and systematics. Time
is in days; bands are named as in BAND_WAVELENGTH_UM.

Occultation is parametrised physically rather than by a "depth": a screen
covers a fraction c(t) of the stellar disk with optical depth tau(lambda),

    k = 1 - c(t) * (1 - exp(-tau(lambda))),   tau = tau_ref * (lambda / lambda_ref) ** -beta

so beta = 0 or tau -> inf is grey (bodies much larger than the wavelength),
beta ~ 1-4 is dust-like reddening, and beta < 0 makes dips bluer. The families
deliberately include structure no orbit produces (number-theoretic spacings,
steps, chromatic sign flips), so recovery measures how agnostic a detector is
rather than how well it finds what we expected. See docs/statistics.md §4.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

# Effective wavelengths in micrometres, to the precision chromatic scaling needs.
BAND_WAVELENGTH_UM = {
    "BP": 0.51,
    "G": 0.62,
    "G_CCD": 0.62,
    "RP": 0.78,
    "TESS": 0.80,
    "ztf_g": 0.48,
    "ztf_r": 0.64,
    "ztf_i": 0.79,
    "V": 0.55,
    "W1": 3.4,
    "W2": 4.6,
}
REF_UM = BAND_WAVELENGTH_UM["G"]
DAYS_PER_YEAR = 365.25


def wavelength(band):
    try:
        return BAND_WAVELENGTH_UM[band]
    except KeyError:
        raise ValueError(f"unknown band {band!r}; add it to BAND_WAVELENGTH_UM") from None


def _rise_decay(dt, rise, decay):
    """Gaussian rise before dt = 0, exponential decay after; each branch only where it applies."""
    dt = np.asarray(dt, dtype=float)
    flat = np.atleast_1d(dt)
    out = np.empty_like(flat)
    before = flat < 0
    out[before] = np.exp(-0.5 * (flat[before] / rise) ** 2)
    out[~before] = np.exp(-flat[~before] / decay)
    return out.reshape(dt.shape)


def _profile(t, t0, ingress, egress, flat, shape):
    """Covering-fraction time profile in [0, 1]."""
    t = np.asarray(t, dtype=float)
    if shape == "trapezoid":
        start, full, end_full, end = t0, t0 + ingress, t0 + ingress + flat, t0 + ingress + flat + egress
        rise = np.clip((t - start) / ingress, 0, 1) if ingress > 0 else (t >= start).astype(float)
        fall = np.clip((end - t) / egress, 0, 1) if egress > 0 else (t <= end).astype(float)
        out = np.minimum(rise, fall)
        out[(t >= full) & (t <= end_full)] = 1.0
        return out
    if shape == "gaussian":
        sigma = (ingress + egress) / 2
        return np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    if shape == "tail":
        # Fast ingress, exponential egress: a trailing cloud or debris tail.
        return _rise_decay(t - t0, max(ingress, 1e-9), egress)
    raise ValueError(f"unknown shape {shape!r}")


@dataclass
class Dip:
    """One occultation event."""

    t0: float
    covering: float
    tau_ref: float = np.inf
    beta: float = 0.0
    ingress: float = 0.1
    egress: float = 0.1
    flat: float = 0.0
    shape: str = "trapezoid"
    kind: str = field(default="dip", init=False)

    def factor(self, t, band):
        c = self.covering * _profile(t, self.t0, self.ingress, self.egress, self.flat, self.shape)
        tau = self.tau_ref * (wavelength(band) / REF_UM) ** -self.beta
        return 1.0 - c * (1.0 - np.exp(-tau))


@dataclass
class SecularTrend:
    """Linear fractional flux change per year, optionally chromatic."""

    rate_per_year: float
    t_ref: float
    beta: float = 0.0
    kind: str = field(default="trend", init=False)

    def factor(self, t, band):
        scale = (wavelength(band) / REF_UM) ** -self.beta
        return 1.0 + self.rate_per_year * scale * (np.asarray(t, dtype=float) - self.t_ref) / DAYS_PER_YEAR


@dataclass
class Step:
    """Abrupt fractional flux change at t_step."""

    t_step: float
    fraction: float
    beta: float = 0.0
    kind: str = field(default="step", init=False)

    def factor(self, t, band):
        scale = (wavelength(band) / REF_UM) ** -self.beta
        return np.where(np.asarray(t, dtype=float) >= self.t_step, 1.0 + self.fraction * scale, 1.0)


@dataclass
class Flare:
    """Fast-rise, exponential-decay brightening; bluer bands brighten more (gamma > 0)."""

    t0: float
    amplitude: float
    rise: float = 0.002
    decay: float = 0.02
    gamma: float = 2.0
    kind: str = field(default="flare", init=False)

    def factor(self, t, band):
        t = np.asarray(t, dtype=float)
        amp = self.amplitude * (wavelength(band) / REF_UM) ** -self.gamma
        return 1.0 + amp * _rise_decay(t - self.t0, self.rise, self.decay)


@dataclass
class Composite:
    """Product of several signals, e.g. a dip train on top of a fade."""

    parts: list
    kind: str = field(default="composite", init=False)

    def factor(self, t, band):
        k = np.ones(np.shape(t), dtype=float)
        for p in self.parts:
            k = k * p.factor(t, band)
        return k


def truth(signal):
    """JSON-serialisable description of a signal, for the sealed truth table."""
    if isinstance(signal, Composite):
        return {"kind": "composite", "parts": [truth(p) for p in signal.parts]}
    d = asdict(signal)
    return {k: (None if isinstance(v, float) and np.isinf(v) else v) for k, v in d.items()}


def inject(t, flux, err, band, signal):
    """Apply a signal to a real light curve.

    Errors scale with the factor, matching the scaled noise realisation; true
    photon noise would scale as sqrt(k), which differs by < 8% for k > 0.85.
    """
    k = signal.factor(t, band)
    return np.asarray(flux) * k, np.asarray(err) * k


# --- Event times -------------------------------------------------------------

PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])


def event_times(rng, kind, t_min, t_max, **kw):
    """Times of events in a train. Kinds span random, orbital and non-orbital structure."""
    span = t_max - t_min
    if kind == "poisson":
        n = rng.poisson(kw["rate_per_day"] * span)
        return np.sort(rng.uniform(t_min, t_max, n))
    if kind == "burst":
        centre = rng.uniform(t_min, t_max)
        n = int(kw.get("n", 5))
        return np.sort(centre + rng.uniform(0, kw["window"], n))
    if kind == "periodic":
        period = kw["period"]
        start = t_min + rng.uniform(0, period)
        return np.arange(start, t_max, period)
    if kind == "primes":
        # Spacings proportional to successive primes: regular, but not any orbit.
        unit = kw["unit"]
        times = t_min + rng.uniform(0, unit) + unit * np.cumsum(PRIMES)
        return times[times < t_max]
    if kind == "geometric":
        unit, ratio = kw["unit"], kw["ratio"]
        times = t_min + rng.uniform(0, unit) + unit * np.cumsum(ratio ** np.arange(64))
        return times[times < t_max]
    raise ValueError(f"unknown event-time kind {kind!r}")


# --- Sampling families -------------------------------------------------------


def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample(rng, family, t_min, t_max, t_ext=None):
    """Draw one signal from a named family over the light curve's time span.

    With t_ext > t_max, dip trains keep running to t_ext under the same law (same rate, period or spacing),
    so another survey's later epochs see the same process; priors still scale with t_max - t_min."""
    span = t_max - t_min
    t_end = t_max if t_ext is None else max(t_ext, t_max)
    if family == "grey_dip":
        dur = _loguniform(rng, 0.05, 30.0)
        return Dip(t0=rng.uniform(t_min, t_max), covering=_loguniform(rng, 0.005, 0.5),
                   ingress=dur * rng.uniform(0.05, 0.5), egress=dur * rng.uniform(0.05, 0.5),
                   flat=dur * rng.uniform(0, 0.5), shape=str(rng.choice(["trapezoid", "gaussian", "tail"])))
    if family == "dust_dip":
        dur = _loguniform(rng, 0.1, 30.0)
        return Dip(t0=rng.uniform(t_min, t_max), covering=_loguniform(rng, 0.02, 0.9),
                   tau_ref=_loguniform(rng, 0.05, 2.0), beta=rng.uniform(1.0, 4.0),
                   ingress=dur * 0.3, egress=dur * 0.7, shape="tail")
    if family == "blue_dip":
        dur = _loguniform(rng, 0.05, 10.0)
        return Dip(t0=rng.uniform(t_min, t_max), covering=_loguniform(rng, 0.02, 0.5),
                   tau_ref=_loguniform(rng, 0.05, 1.0), beta=rng.uniform(-3.0, -0.5),
                   ingress=dur / 2, egress=dur / 2, shape="gaussian")
    if family in ("dip_train_poisson", "dip_train_periodic", "dip_train_primes", "dip_train_geometric"):
        kind = family.removeprefix("dip_train_")
        # Priors scale with the span, so a 27 d TESS sector and a 2000 d Gaia baseline both get
        # several events per train. Only the chosen kind's prior is drawn.
        kw = {
            "poisson": lambda: {"rate_per_day": _loguniform(rng, 3 / span, 30 / span)},
            "periodic": lambda: {"period": _loguniform(rng, span / 30, span / 3)},
            "primes": lambda: {"unit": _loguniform(rng, span / 2000, span / 200)},
            "geometric": lambda: {"unit": _loguniform(rng, span / 400, span / 20), "ratio": rng.uniform(1.05, 1.6)},
        }[kind]()
        depth, dur = _loguniform(rng, 0.005, 0.3), _loguniform(rng, 0.05, min(5.0, span / 20))
        dips = [Dip(t0=float(t0), covering=depth, ingress=dur / 4, egress=dur / 4, flat=dur / 2)
                for t0 in event_times(rng, kind, t_min, t_end, **kw)]
        return Composite(dips)
    if family in HELD_OUT_FAMILIES:
        return _held_out(rng, family, t_min, t_end, span)
    if family == "secular_fade":
        return SecularTrend(rate_per_year=-_loguniform(rng, 1e-3, 5e-2), t_ref=(t_min + t_max) / 2,
                            beta=float(rng.choice([0.0, rng.uniform(1, 3)])))
    if family == "step":
        return Step(t_step=rng.uniform(t_min, t_max), fraction=float(rng.choice([-1, 1]) * _loguniform(rng, 0.002, 0.1)))
    if family == "chromatic_flip":
        # BP dims while RP brightens: no single screen does this. The brightening scales with the dip so that
        # both halves are present whenever the event is: with independent amplitudes one half usually swamps
        # the other and the result is an ordinary chromatic dip or a red brightening.
        dur = _loguniform(rng, 0.1, 10.0)
        t0 = rng.uniform(t_min, t_max)
        covering = _loguniform(rng, 0.01, 0.2)
        return Composite([
            Dip(t0=t0, covering=covering, tau_ref=1.0, beta=4.0, ingress=dur / 2, egress=dur / 2, shape="gaussian"),
            Flare(t0=t0, amplitude=covering * rng.uniform(0.5, 1.5), rise=dur / 4, decay=dur / 2, gamma=-4.0),
        ])
    if family == "flare":
        return Flare(t0=rng.uniform(t_min, t_max), amplitude=_loguniform(rng, 0.005, 1.0),
                     rise=_loguniform(rng, 5e-4, 5e-3), decay=_loguniform(rng, 5e-3, 0.2))
    raise ValueError(f"unknown family {family!r}")


FAMILIES = (
    "grey_dip", "dust_dip", "blue_dip",
    "dip_train_poisson", "dip_train_periodic", "dip_train_primes", "dip_train_geometric",
    "secular_fade", "step", "chromatic_flip", "flare",
)

# Families no detector, flag or rule was calibrated or tuned on. They exist to measure recall on structure the
# rules were not designed around (docs/plan.md, principle 3; review finding R5), so no calibration script may
# draw them; tests/test_inject.py checks the calibration modules' family lists. Use them only in confirmatory
# batches, and once a rule has been adjusted after seeing their results, they are no longer held out.
HELD_OUT_FAMILIES = ("dip_train_clustered", "dip_train_drifting", "dip_swarm", "dip_irregular")


def _held_out(rng, family, t_min, t_end, span):
    """Dip sequences with structure between the calibrated trains': clustered in time, nearly periodic with a
    drifting period, many dips of scattered depth, or a few complex events (Boyajian's star-like)."""
    def dip(t0, depth, dur, shape="trapezoid"):
        if shape == "tail":
            return Dip(t0=float(t0), covering=depth, ingress=dur * 0.2, egress=dur * 0.8, shape="tail")
        return Dip(t0=float(t0), covering=depth, ingress=dur / 4, egress=dur / 4, flat=dur / 2, shape=shape)

    max_dur = min(5.0, span / 20)
    if family == "dip_train_clustered":
        # Neyman-Scott: a Poisson set of episodes, each a burst of dips within a few days to weeks.
        parents = event_times(rng, "poisson", t_min, t_end, rate_per_day=_loguniform(rng, 1 / span, 8 / span))
        width = _loguniform(rng, 2.0, min(60.0, span / 10))
        times = np.concatenate([p + rng.uniform(0, width, 1 + rng.poisson(_loguniform(rng, 1, 8))) for p in parents]) \
            if len(parents) else np.array([])
        depth, dur = _loguniform(rng, 0.005, 0.3), _loguniform(rng, 0.05, max_dur)
        return Composite([dip(t, depth * rng.uniform(0.5, 1.5), dur) for t in np.sort(times[times < t_end])])
    if family == "dip_train_drifting":
        # Nearly periodic: the period drifts linearly by up to +-40% over the span and each dip is jittered.
        p0 = _loguniform(rng, span / 30, span / 3)
        drift = rng.uniform(-0.4, 0.4) / span
        jitter = rng.uniform(0.01, 0.1)
        t, times = t_min + rng.uniform(0, p0), []
        while t < t_end and len(times) < 500:
            times.append(t)
            p = max(p0 * (1 + drift * (t - t_min)), p0 * 0.2)
            t += p * (1 + jitter * rng.standard_normal())
        depth, dur = _loguniform(rng, 0.005, 0.3), _loguniform(rng, 0.05, max_dur)
        return Composite([dip(t, depth, dur) for t in times])
    if family == "dip_swarm":
        # Many bodies on different orbits: frequent dips whose depths and durations scatter widely.
        times = event_times(rng, "poisson", t_min, t_end, rate_per_day=_loguniform(rng, 10 / span, 60 / span))
        med, scatter = _loguniform(rng, 0.003, 0.05), rng.uniform(0.5, 1.5)
        return Composite([dip(t, float(min(0.9, med * np.exp(scatter * rng.standard_normal()))),
                              _loguniform(rng, 0.05, max_dur), str(rng.choice(["trapezoid", "gaussian", "tail"])))
                          for t in times])
    if family == "dip_irregular":
        # A few complex events, each several overlapping dips of very different depth and duration.
        n = int(rng.integers(3, 16))
        parts = []
        for c in rng.uniform(t_min, t_end, n):
            for _ in range(int(rng.integers(1, 5))):
                parts.append(dip(c + rng.uniform(-5, 5), _loguniform(rng, 0.005, 0.2), _loguniform(rng, 0.2, 20.0),
                                 str(rng.choice(["gaussian", "tail"]))))
        return Composite(parts)
    raise ValueError(f"unknown family {family!r}")
