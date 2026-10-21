"""Dimming-arm injection families, on top of the shared library in cosmos_search.inject.lightcurve.

The shared library covers single grey/dust/blue dips, dip trains with random, periodic and
number-theoretic spacing, fades, steps, flares and a chromatic flip. This module adds shapes it lacks:

- boyajian_complex: a cluster of 1-4 overlapping asymmetric dips of different depths (KIC 8462852's
  D1500-like complexes; Boyajian et al. 2016)
- swarm: many shallow, short dips at a high Poisson rate; the signature of a large number of small
  occulters, visible as excess negative skew and flicker rather than as single events
- varperiodic: strictly periodic, with random per-event depth; no Keplerian body changes cross-section
  from transit to transit, a rotating or reconfigurable occulter would
- telegraph: flux switching between two or three discrete levels with exponential dwell times; no stellar
  process produces a few exact states

and `windows`, which says where any signal acts, so recovery can be scored per event.
"""

from dataclasses import dataclass, field

import numpy as np

from cosmos_search.inject import lightcurve as lib
from cosmos_search.inject.lightcurve import Composite, Dip, Flare, SecularTrend, Step


@dataclass
class Levels:
    """Grey piecewise-constant flux: factor levels[i] from switch_times[i] until switch_times[i + 1]."""

    switch_times: list
    levels: list
    kind: str = field(default="levels", init=False)

    def factor(self, t, band):
        t = np.asarray(t, dtype=float)
        idx = np.clip(np.searchsorted(self.switch_times, t, side="right") - 1, 0, len(self.levels) - 1)
        return np.asarray(self.levels, float)[idx]


_logu = lib._loguniform


def _boyajian_complex(rng, t_min, t_max):
    k = int(rng.integers(1, 5))
    centre = rng.uniform(t_min, t_max)
    dips = []
    for _ in range(k):
        dur = _logu(rng, 0.1, 4.0)
        a = rng.uniform(-0.6, 0.6)
        t0 = centre + (rng.normal(0, 1.5 * dur) if k > 1 else 0.0)
        # Gaussian profile's sigma is (ingress + egress) / 2 in the shared model; use a tail shape for
        # asymmetric members so recovery time differs from ingress time.
        shape = "tail" if abs(a) > 0.3 else "gaussian"
        dips.append(Dip(t0=float(t0), covering=_logu(rng, 0.003, 0.2), ingress=dur * (1 - a) / 4,
                        egress=dur * (1 + a) / 2, shape=shape))
    return Composite(dips)


def _swarm(rng, t_min, t_max):
    rate = _logu(rng, 0.5, 5.0)
    n = max(1, rng.poisson(rate * (t_max - t_min)))
    dips = []
    for t0 in rng.uniform(t_min, t_max, n):
        dur = _logu(rng, 0.02, 0.3)
        dips.append(Dip(t0=float(t0), covering=_logu(rng, 0.0005, 0.01), ingress=dur / 2, egress=dur / 2,
                        shape="gaussian"))
    return Composite(dips)


def _varperiodic(rng, t_min, t_max):
    period = _logu(rng, 0.5, 10.0)
    dur = _logu(rng, 0.05, 0.3)
    depth = _logu(rng, 0.002, 0.05)
    times = lib.event_times(rng, "periodic", t_min, t_max, period=period)
    return Composite([Dip(t0=float(t0), covering=depth * rng.uniform(0, 1), ingress=dur / 10, egress=dur / 10,
                          flat=dur * 0.8) for t0 in times])


def _telegraph(rng, t_min, t_max):
    n_levels = int(rng.integers(2, 4))
    depth = _logu(rng, 0.002, 0.05)
    values = 1 - depth * np.arange(n_levels) / (n_levels - 1)
    dwell = _logu(rng, 0.2, 5.0)
    times, levels, tt, state = [t_min - 1.0], [1.0], t_min, 0
    while tt < t_max:
        tt += rng.exponential(dwell)
        state = int(rng.integers(0, n_levels))
        times.append(float(tt))
        levels.append(float(values[state]))
    return Levels(times, levels)


EXTRA = {
    "boyajian_complex": _boyajian_complex,
    "swarm": _swarm,
    "varperiodic": _varperiodic,
    "telegraph": _telegraph,
}
FAMILIES = tuple(lib.FAMILIES) + tuple(EXTRA)


def sample(rng, family, t_min, t_max):
    if family in EXTRA:
        return EXTRA[family](rng, t_min, t_max)
    return lib.sample(rng, family, t_min, t_max)


def windows(signal, t_min, t_max):
    """[(start, end)] intervals inside [t_min, t_max] where the signal departs from 1 appreciably."""
    out = []
    if isinstance(signal, Composite):
        for p in signal.parts:
            out += windows(p, t_min, t_max)
        return out
    if isinstance(signal, Dip):
        if signal.shape == "trapezoid":
            w = (signal.t0, signal.t0 + signal.ingress + signal.flat + signal.egress)
        elif signal.shape == "gaussian":
            s = (signal.ingress + signal.egress) / 2
            w = (signal.t0 - 2 * s, signal.t0 + 2 * s)
        else:
            w = (signal.t0 - 2 * signal.ingress, signal.t0 + 3 * signal.egress)
    elif isinstance(signal, Flare):
        w = (signal.t0 - 5 * signal.rise, signal.t0 + 5 * signal.decay)
    elif isinstance(signal, Step):
        w = (signal.t_step - 0.5, signal.t_step + 0.5)
    elif isinstance(signal, SecularTrend):
        w = (t_min, t_max)
    elif isinstance(signal, Levels):
        for a, b, v in zip(signal.switch_times, signal.switch_times[1:] + [np.inf], signal.levels):
            if v < 1:
                out.append((max(a, t_min), min(b, t_max)))
        return [w for w in out if w[1] > w[0]]
    else:
        raise TypeError(f"no windows for {type(signal).__name__}")
    return [w] if (w[1] >= t_min and w[0] <= t_max) else []


def max_effect(signal, t, band):
    """Largest fractional deficit and excess the signal produces at the sampled times."""
    k = signal.factor(np.asarray(t, float), band)
    return float(max(0.0, 1 - k.min())), float(max(0.0, k.max() - 1))
