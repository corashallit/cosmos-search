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
class Ramp:
    """Fractional flux change that starts at t_start, grows linearly over `duration` to `fraction`, then holds.

    A fade that begins partway through the span (fraction < 0), such as a screen or swarm that starts to
    build, rather than one already under way like SecularTrend.
    """

    t_start: float
    duration: float
    fraction: float
    beta: float = 0.0
    power: float = 1.0
    kind: str = field(default="ramp", init=False)

    def factor(self, t, band):
        scale = (wavelength(band) / REF_UM) ** -self.beta
        x = np.clip((np.asarray(t, dtype=float) - self.t_start) / max(self.duration, 1e-9), 0.0, 1.0)
        return 1.0 + self.fraction * scale * x ** self.power


@dataclass
class Cycle:
    """Dimming-only cosine: the flux falls by `amplitude` at t0 + period/2 (mod period) and recovers fully in
    between, like an occulter on a long orbit or a slow pulsation seen only as fades."""

    t0: float
    period: float
    amplitude: float
    beta: float = 0.0
    dpdt: float = 0.0  # the period at time t is period * (1 + dpdt * (t - t0))
    kind: str = field(default="cycle", init=False)

    def factor(self, t, band):
        scale = (wavelength(band) / REF_UM) ** -self.beta
        dt = np.asarray(t, dtype=float) - self.t0
        cycles = dt / self.period if self.dpdt == 0 else np.log1p(self.dpdt * dt) / (self.dpdt * self.period)
        return 1.0 - self.amplitude * scale * (1 - np.cos(2 * np.pi * cycles)) / 2


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


class HeldOutError(ValueError):
    """A held-out family was requested by code that is not drawing a confirmatory batch."""


def sample(rng, family, t_min, t_max, t_ext=None, allow_held_out=False):
    """Draw one signal from a named family over the light curve's time span.

    With t_ext > t_max, dip trains keep running to t_ext under the same law (same rate, period or spacing),
    so another survey's later epochs see the same process; priors still scale with t_max - t_min.
    HELD_OUT_FAMILIES are refused unless allow_held_out is set, which only a confirmatory batch's plan does."""
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
    if family in SPENT_SLOW:
        return _held_out_slow(rng, family, t_min, t_max, span)
    if family in HELD_OUT_FAMILIES + HELD_OUT_FAMILIES_V2 + HELD_OUT_SLOW:
        if not allow_held_out:
            raise HeldOutError(f"{family!r} is held out for the confirmatory batch; calibration and tuning may not "
                               "draw it (docs/dayone.md, review fixes)")
        if family in HELD_OUT_SLOW:
            return _held_out_periodic(rng, family, t_min, t_max, span)
        return _held_out(rng, family, t_min, t_end, span)
    if family == "repeat_long_dip":
        # The same long dip every period, at least 1.5 cycles in the span: a body or swarm on a long orbit.
        period = _loguniform(rng, 40.0, max(41.0, span / 1.5))
        dur = period * _loguniform(rng, 0.05, 0.4)
        ing = dur * rng.uniform(0.1, 0.5)
        beta = _chromatic(rng)
        depth = _loguniform(rng, 0.002, 0.1)
        return Composite(_periodic_dips(rng, t_min, t_max, period, depth, dur, ing, ing, beta))
    if family == "secular_fade":
        return SecularTrend(rate_per_year=-_loguniform(rng, 1e-3, 5e-2), t_ref=(t_min + t_max) / 2,
                            beta=float(rng.choice([0.0, rng.uniform(1, 3)])))
    if family == "long_dip":
        # Weeks to years: disc eclipses, long dust episodes, a large slow occulter. Grey or dust-reddened.
        dur = _loguniform(rng, 20.0, min(1500.0, span))
        dust = rng.random() < 0.5
        return Dip(t0=rng.uniform(t_min - dur / 2, t_max - dur / 2), covering=_loguniform(rng, 0.002, 0.3),
                   tau_ref=_loguniform(rng, 0.1, 2.0) if dust else np.inf, beta=rng.uniform(1.0, 4.0) if dust else 0.0,
                   ingress=dur * rng.uniform(0.1, 0.5), egress=dur * rng.uniform(0.1, 0.5), flat=dur * rng.uniform(0, 0.5),
                   shape=str(rng.choice(["trapezoid", "gaussian", "tail"])))
    if family == "ramp_fade":
        dur = _loguniform(rng, min(30.0, span / 2), span)
        return Ramp(t_start=rng.uniform(t_min - dur / 2, t_max - min(30.0, span / 4)), duration=dur,
                    fraction=-_loguniform(rng, 0.001, 0.1), beta=float(rng.choice([0.0, rng.uniform(1, 3)])))
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
# Slow changes (docs/arms/slow-changes.md). Kept out of FAMILIES so the frozen searches that draw from FAMILIES
# (TESS bake-off, dimming classes) draw exactly what they did before.
SLOW_FAMILIES = ("long_dip", "ramp_fade", "secular_fade")

# Families no detector, flag or rule was calibrated or tuned on. They exist to measure recall on structure the
# rules were not designed around (docs/plan.md, principle 3; review findings R5, R20), so sample() refuses them
# unless the caller is drawing a confirmatory plan, and tests/test_inject_lightcurve.py checks the calibration
# modules' family lists. Use them only in confirmatory
# batches, and once a rule has been adjusted after seeing their results, they are no longer held out.
HELD_OUT_FAMILIES = ("dip_train_clustered", "dip_train_drifting", "dip_swarm", "dip_irregular")
# The first set was spent by the confirmatory batch of rule v5 (docs/dayone.md). This second set is for the batch
# that tests rule v6 (docs/dayone.md, "Rule v6"), under the same conditions: no calibration, flag or rule has used
# them, and nobody looks at how a flag responds to them before that batch is revealed.
HELD_OUT_FAMILIES_V2 = ("dip_intermittent", "dip_depth_trend", "dip_chromatic_train", "dip_doublets")
# Slow shapes that the slow-change detector's first blind test used (docs/arms/slow-changes.md, "Blind test
# (S6)"). That test has been revealed, so they are spent and free for calibration.
SPENT_SLOW = ("staircase_fade", "slow_cycle", "accelerating_fade", "fade_with_dips")
# Repeating slow shapes that the slow period search was not designed or calibrated on, for its blind test
# (docs/arms/slow-changes.md, "Slow period search"). Same conditions as the held-out sets above. Its calibration
# draws PERIODIC_FAMILIES.
HELD_OUT_SLOW = ("sawtooth_dips", "drifting_cycle", "uneven_dips", "paired_dips")
PERIODIC_FAMILIES = ("repeat_long_dip", "slow_cycle")


def _periodic_dips(rng, t_min, t_max, period, depth, dur, ingress, egress, beta=0.0, depths=None):
    """Trapezoid dips every `period` across [t_min, t_max], from a random phase; depths[k] scales dip k."""
    times = np.arange(t_min - rng.uniform(0, period), t_max + period, period)
    scale = np.ones(len(times)) if depths is None else depths(len(times))
    return [Dip(t0=float(t), covering=float(depth * k), ingress=ingress, egress=egress,
                flat=max(dur - ingress - egress, 0.0), beta=beta)
            for t, k in zip(times, scale)]


def _held_out_periodic(rng, family, t_min, t_max, span):
    """Repeating slow changes with a shape other than the calibrated equal, symmetric dips and the sinusoid."""
    p_hi = max(41.0, span / 1.5)
    beta = _chromatic(rng)
    if family == "sawtooth_dips":
        # A fast drop and a slow recovery every cycle: a trailing tail of material.
        period = _loguniform(rng, 40.0, p_hi)
        dur = period * _loguniform(rng, 0.1, 0.5)
        ing = dur * rng.uniform(0.03, 0.15)
        return Composite(_periodic_dips(rng, t_min, t_max, period, _loguniform(rng, 0.003, 0.1), dur, ing,
                                        dur - ing, beta))
    if family == "drifting_cycle":
        # A sinusoidal dimming whose period lengthens or shortens by 15-40% across the span.
        period = _loguniform(rng, 60.0, p_hi)
        change = float(rng.choice([-1, 1]) * rng.uniform(0.15, 0.4))
        return Cycle(t0=float(rng.uniform(t_min, t_max)), period=period, amplitude=_loguniform(rng, 0.003, 0.1),
                     beta=beta, dpdt=change / span)
    if family == "uneven_dips":
        # A long dip every period whose depth changes from cycle to cycle (20-100% of the deepest).
        period = _loguniform(rng, 40.0, p_hi)
        dur = period * _loguniform(rng, 0.05, 0.4)
        ing = dur * rng.uniform(0.1, 0.5)
        return Composite(_periodic_dips(rng, t_min, t_max, period, _loguniform(rng, 0.003, 0.1), dur, ing, ing,
                                        beta, depths=lambda n: rng.uniform(0.2, 1.0, n)))
    if family == "paired_dips":
        # Two long dips per period, the second at phase 0.2-0.8 and 20-100% as deep: two bodies on one orbit, or
        # an eccentric binary seen at a long period.
        period = _loguniform(rng, 60.0, p_hi)
        dur = period * _loguniform(rng, 0.04, 0.2)
        ing = dur * rng.uniform(0.1, 0.5)
        depth = _loguniform(rng, 0.003, 0.1)
        first = _periodic_dips(rng, t_min, t_max, period, depth, dur, ing, ing, beta)
        shift, ratio = period * rng.uniform(0.2, 0.8), rng.uniform(0.2, 1.0)
        second = [Dip(t0=d.t0 + shift, covering=depth * ratio, ingress=ing, egress=ing, flat=d.flat, beta=beta)
                  for d in first]
        return Composite(first + second)
    raise ValueError(f"unknown family {family!r}")


def _chromatic(rng):
    return float(rng.choice([0.0, rng.uniform(1, 3)]))


def _held_out_slow(rng, family, t_min, t_max, span):
    """Slow changes with a shape other than the calibrated long dip, linear ramp and steady fade."""
    if family == "staircase_fade":
        # A few abrupt drops that stay down: pieces added one at a time, or a screen that thickens in jumps.
        n = int(rng.integers(2, 6))
        beta = _chromatic(rng)
        return Composite([Step(t_step=float(t), fraction=-_loguniform(rng, 0.002, 0.03), beta=beta)
                          for t in np.sort(rng.uniform(t_min + 0.05 * span, t_max - 0.05 * span, n))])
    if family == "slow_cycle":
        # Repeated slow fades and full recoveries, a few cycles to under one per span.
        return Cycle(t0=float(rng.uniform(t_min, t_max)), period=_loguniform(rng, 150.0, 1000.0),
                     amplitude=_loguniform(rng, 0.002, 0.1), beta=_chromatic(rng))
    if family == "accelerating_fade":
        # A fade that starts slowly and speeds up (fraction * x**p), still going or levelled off by the end.
        dur = _loguniform(rng, min(100.0, span / 2), 2 * span)
        return Ramp(t_start=float(rng.uniform(t_min - dur / 2, t_max - min(60.0, span / 4))), duration=dur,
                    fraction=-_loguniform(rng, 0.002, 0.1), beta=_chromatic(rng), power=float(rng.uniform(2.0, 4.0)))
    if family == "fade_with_dips":
        # A gradual fade with short dips scattered after it starts: debris or structures building up.
        dur = _loguniform(rng, span / 2, 2 * span)
        t_start = float(rng.uniform(t_min - span / 2, t_min + span / 2))
        parts = [Ramp(t_start=t_start, duration=dur, fraction=-_loguniform(rng, 0.002, 0.05), beta=_chromatic(rng))]
        lo = max(t_start, t_min)
        for t0 in rng.uniform(lo, t_max, int(rng.poisson(_loguniform(rng, 3.0, 20.0)))):
            d = _loguniform(rng, 1.0, 20.0)
            parts.append(Dip(t0=float(t0), covering=_loguniform(rng, 0.005, 0.1), ingress=d / 4, egress=d / 4,
                             flat=d / 2))
        return Composite(parts)
    raise ValueError(f"unknown family {family!r}")


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
    if family == "dip_intermittent":
        # One strict period, but the occulter is there in only some cycles (a clumpy ring, a body that comes and
        # goes): eclipses never skip a cycle.
        period = _loguniform(rng, span / 30, span / 3)
        duty = rng.uniform(0.25, 0.6)
        start = t_min + rng.uniform(0, period)
        times = np.arange(start, t_end, period)
        on = rng.random(len(times)) < duty
        if len(times) >= 2 and on.sum() < 2:
            on[rng.choice(len(times), 2, replace=False)] = True
        depth, dur = _loguniform(rng, 0.005, 0.3), _loguniform(rng, 0.05, max_dur)
        return Composite([dip(t, depth * rng.uniform(0.8, 1.2), dur) for t in times[on]])
    if family == "dip_depth_trend":
        # One strict period with a depth that grows or shrinks by a factor of ~5-20 over the span (something
        # being built, or breaking up): an eclipse keeps its depth.
        period = _loguniform(rng, span / 30, span / 3)
        start = t_min + rng.uniform(0, period)
        times = np.arange(start, t_end, period)
        rate = float(rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)) / span
        d0 = _loguniform(rng, 0.005, 0.15)
        dur = _loguniform(rng, 0.05, max_dur)
        return Composite([dip(t, float(min(0.6, d0 * np.exp(rate * (t - t_min)))), dur) for t in times])
    if family == "dip_chromatic_train":
        # Dips at random times whose colour differs from one event to the next (grey, reddened, or deeper in the
        # red): one occulting population of fixed composition cannot do that.
        times = event_times(rng, "poisson", t_min, t_end, rate_per_day=_loguniform(rng, 3 / span, 30 / span))
        parts = []
        for t in times:
            tau = _loguniform(rng, 0.05, 2.0)
            depth = _loguniform(rng, 0.005, 0.3)
            dur = _loguniform(rng, 0.05, max_dur)
            parts.append(Dip(t0=float(t), covering=float(min(0.95, depth / (1 - np.exp(-tau)))), tau_ref=tau,
                             beta=rng.uniform(-3.0, 4.0), ingress=dur / 4, egress=dur / 4, flat=dur / 2))
        return Composite(parts)
    if family == "dip_doublets":
        # Pairs of dips a fixed interval apart, the pairs at random times: two bodies moving together on a path
        # that is not a closed orbit we see, or one body seen twice.
        parents = event_times(rng, "poisson", t_min, t_end, rate_per_day=_loguniform(rng, 2 / span, 12 / span))
        gap, ratio = _loguniform(rng, 0.2, 5.0), rng.uniform(0.3, 1.0)
        depth, dur = _loguniform(rng, 0.005, 0.3), _loguniform(rng, 0.05, max(0.06, min(max_dur, gap / 2)))
        parts = []
        for p in parents:
            parts += [dip(p, depth, dur), dip(p + gap, depth * ratio, dur)]
        return Composite(parts)
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
