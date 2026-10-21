"""Forward models of the expected number of single one-visit dips per star and half (docs/population.md §3).

Both models apply `events.classify`, the rule used on the data, to predicted depths on the star's own epochs,
scale and threshold, so the expectation and the observation share one definition. Noise is left out: an epoch is
in dip when the predicted depth exceeds K sigma, flat when it is under 2 sigma.

Importance sampling. A single one-visit event must overlap the visit that shows it, so each draw is placed over
one of the star's visits, chosen uniformly, and carries weight p / q: p is the event's density in time (the
occulter's rate per day; 1/P per unit phase for an eclipse), q the proposal density at that time, summed over every
visit (and, for an eclipse, every cycle) whose window the draw falls in. Draws that overlap no visit cannot make an
event, so nothing is missed.

- `generic`: a reference one-off occulter at GEN_RATE per day, duration log-uniform GEN_T, depth log-uniform
  GEN_DEPTH, trapezoid with ingress 1-50% of the duration. Its expectation E_gen is the star's exposure to one-off
  dips: cadence, noise, threshold and the scanning law in one number.
- `eclipse`: companions from `vetting.single_event`'s mundane population (stellar binaries with the Raghavan
  period distribution, dark bodies), host radius and mass from Gaia photometry and parallax with 0.15 dex scatter
  (a broad prior when the parallax is poor). E_EB is the expected number of eclipses that show up as a single
  one-visit dip: caught on one visit, with no detectable repeat.
"""

import numpy as np

from cosmos_search.vetting import physics, single_event as se

from . import events

GEN_RATE = 1e-3          # events per day per star (a unit; only ratios between stars matter)
GEN_T = (0.05, 30.0)     # days
GEN_DEPTH = (0.005, 0.5)
EB_LOGP = (np.log10(se.P_MIN), 6.5)
N_GEN, N_EB = 64, 64


def visit_table(rows, stars):
    """Per visit: star, first and last epoch time, half."""
    vid, gid, t = rows["vid"], rows["gid"], rows["t"].astype(np.float64)
    n = int(vid[-1]) + 1 if len(vid) else 0
    a = np.full(n, np.inf)
    b = np.full(n, -np.inf)
    np.minimum.at(a, vid, t)
    np.maximum.at(b, vid, t)
    g = np.zeros(n, np.int64)
    g[vid] = gid
    h = np.zeros(n, np.int8)
    h[vid] = rows["vhalf"]
    return {"gid": g, "a": a, "b": b, "half": h}


def host_params(g, bp_rp, plx, plx_err):
    """Vectorised `physics.stellar_params`: (R, M) in solar units, NaN where the parallax SNR is below 5."""
    g, c, plx, err = (np.asarray(x, float) for x in (g, bp_rp, plx, plx_err))
    ok = np.isfinite(g) & np.isfinite(c) & np.isfinite(plx) & (err > 0) & (plx / np.where(err > 0, err, 1) >= 5)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_g = g + 5 * np.log10(plx) - 10
        cc = np.clip(c, 0.39, 1.6)
        teff = 5040.0 / (0.4929 + 0.5092 * cc - 0.0353 * cc * cc)
        r = np.sqrt(10 ** (-0.4 * (m_g - physics.M_G_SUN))) * (physics.T_SUN / teff) ** 2
    m = np.where(r < 1.5, r ** 1.25, 1.0)
    return np.where(ok, r, np.nan), np.where(ok, m, np.nan)


def _anchor(rng, stars, vt, T_half_pad):
    """Pick a visit per star uniformly and a time uniformly over the window where a draw overlaps it.

    T_half_pad: per-star padding on each side of the visit (T for a start time, T/2 for a mid-time). Returns the
    time, and q's normalisation needs, via `_proposal`."""
    nv = stars["n_visits"]
    k = np.minimum((rng.random(len(nv)) * nv).astype(np.int64), nv - 1)
    v = stars["first_visit"] + k
    lo, hi = vt["a"][v] - T_half_pad[0], vt["b"][v] + T_half_pad[1]
    return lo + rng.random(len(nv)) * (hi - lo)


def _proposal(stars, vt, x, pad_lo, pad_hi, period=None):
    """q(x) per star: (1 / n_visits) * sum over visits (and cycles) whose window [a - pad_lo, b + pad_hi] holds x."""
    g = vt["gid"]
    lo, hi = vt["a"] - pad_lo[g], vt["b"] + pad_hi[g]
    width = hi - lo
    xx = x[g]
    if period is None:
        inside = (xx >= lo) & (xx <= hi)
    else:
        p = period[g]
        mid = 0.5 * (lo + hi)
        n = np.round((mid - xx) / p)
        inside = np.abs(xx + n * p - mid) <= 0.5 * width
    s = np.bincount(g, weights=np.where(inside, 1.0 / width, 0.0), minlength=len(stars["n_visits"]))
    return s / stars["n_visits"]


def _sov_half(rows, stars, depth_rows):
    """Run the event rule on predicted depths. Returns (is SOV, half of its visit) per star."""
    z = depth_rows / stars["sigma"][rows["gid"]]
    if "masked" in rows:                     # epochs in a masked time cell are neither in dip nor flat
        z = np.where(rows["masked"], np.nan, z)
    c = events.classify(rows["gid"], rows["vid"], z, stars["K"], len(stars["K"]))
    sov = (c["n_events"] == 1) & c["one_visit"] & c["bracketed"] & (c["n_in"] >= 2)
    v = np.where(c["visit"] >= 0, c["visit"], 0)
    half = rows["vhalf_by_visit"][v]
    return sov, half


def _arrays(rows, stars_df):
    stars = {k: stars_df[k].to_numpy() for k in ("n_visits", "first_visit", "sigma", "K")}
    vt = visit_table(rows, stars)
    rows = dict(rows)
    rows["vhalf_by_visit"] = vt["half"]
    return rows, stars, vt


def generic(rows, stars_df, n_draws=N_GEN, rng=None):
    """E_gen per star for halves A and B: shape (n_stars, 2)."""
    rng = rng or np.random.default_rng(0)
    rows, stars, vt = _arrays(rows, stars_df)
    n = len(stars["n_visits"])
    t = rows["t"].astype(np.float64)
    gid = rows["gid"]
    out = np.zeros((n, 2))
    for _ in range(n_draws):
        T = np.exp(rng.uniform(*np.log(GEN_T), n))
        depth = np.exp(rng.uniform(*np.log(GEN_DEPTH), n))
        tau = T * rng.uniform(0.01, 0.5, n)
        t0 = _anchor(rng, stars, vt, (T, np.zeros(n)))
        q = _proposal(stars, vt, t0, T, np.zeros(n))
        w = GEN_RATE / np.maximum(q, 1e-300)
        mid = t0 + T / 2
        d = se._trapezoid(t - mid[gid], T[gid], tau[gid], depth[gid])
        sov, half = _sov_half(rows, stars, d)
        for h in (0, 1):
            out[:, h] += np.where(sov & (half == h), w, 0.0)
    return out / n_draws


def eclipse(rows, stars_df, radius, mass, n_draws=N_EB, rng=None):
    """E_EB per star for halves A and B: shape (n_stars, 2). radius, mass: per star (solar units), NaN for the
    broad prior."""
    rng = rng or np.random.default_rng(1)
    rows, stars, vt = _arrays(rows, stars_df)
    n = len(stars["n_visits"])
    t = rows["t"].astype(np.float64)
    gid = rows["gid"]
    known = np.isfinite(radius) & np.isfinite(mass)
    out = np.zeros((n, 2))
    span_logp = EB_LOGP[1] - EB_LOGP[0]
    for _ in range(n_draws):
        r1 = np.where(known, radius * 10 ** (se.R_SCATTER_DEX * rng.standard_normal(n)),
                      np.exp(rng.uniform(*np.log(se.BROAD_R), n)))
        m1 = np.where(known, mass * 10 ** (0.5 * se.R_SCATTER_DEX * rng.standard_normal(n)),
                      np.exp(rng.uniform(*np.log(se.BROAD_M), n)))
        k, lr, mtot, dark = se._companions(r1, m1, rng)
        logp = rng.uniform(*EB_LOGP, n)
        P = 10 ** logp
        e = rng.uniform(0, se.E_MAX, n)
        w_peri = rng.uniform(0, 2 * np.pi, n)
        b = rng.uniform(0, 1 + k)
        gm = se.G_SI * se.M_SUN * mtot
        a = (gm * (P * physics.DAY) ** 2 / (4 * np.pi**2)) ** (1 / 3)
        r1_m = r1 * physics.R_SUN
        chord = np.sqrt(np.clip((1 + k) ** 2 - b**2, 0, None))
        ecc = np.sqrt(1 - e**2) / (1 + e * np.sin(w_peri))
        T = P / np.pi * (r1_m * chord / a) * ecc                        # days
        ok = (a > 2 * r1_m * (1 + k)) & (chord > 0) & (T > 0)
        flat_chord = np.sqrt(np.clip((1 - k) ** 2 - b**2, 0, None))
        tau = 0.5 * T * (1 - flat_chord / np.maximum(chord, 1e-12))
        depth = se.overlap_fraction(np.where(b < np.abs(1 - k), 0.0, b), k) / (1 + lr)
        p_tr = r1_m * (1 + k) / a * (1 + e * np.sin(w_peri)) / (1 - e**2)
        # A draw with b uniform on [0, 1 + k] stands for every inclination that transits: weight p_tr.
        prior = se._logp_density(logp, dark) * span_logp * np.clip(p_tr, 0, 1) * ok
        tc = _anchor(rng, stars, vt, (T / 2, T / 2))
        q = _proposal(stars, vt, tc, T / 2, T / 2, period=P)
        w = prior * (1 / P) / np.maximum(q, 1e-300)
        dt = t - tc[gid]
        dph = dt - np.round(dt / P[gid]) * P[gid]
        d = se._trapezoid(dph, T[gid], tau[gid], depth[gid])
        sov, half = _sov_half(rows, stars, d)
        for h in (0, 1):
            out[:, h] += np.where(sov & (half == h) & np.isfinite(w), w, 0.0)
    return out / n_draws
