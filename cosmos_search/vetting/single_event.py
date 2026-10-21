"""Could one eclipse explain a dip that was seen once? A test for stars whose epochs show a single dip event.

A dip seen once cannot be tested for recurrence, and a vetter reading it calls it an eclipsing binary, which on
the packet alone is fair (docs/dayone.md, batches 2-4: 0 of 7 single events kept). But "eclipsing binary" makes a
prediction. The event's duration, with the star's size and mass, fixes the orbit's speed, so it fixes the period
to within the spread that impact parameter, eccentricity and companion size allow: by Kepler, an eclipse of
duration T on a star of radius R and mass M has

    T ~ (P / pi) (R (1 + k) / a) sqrt(1 - b^2 / (1 + k)^2) sqrt(1 - e^2) / (1 + e sin w),   a^3 = G M_tot P^2 / 4 pi^2

so T ~ P^(1/3): a 0.2 d eclipse on the Sun comes back every ~10 d, a 1 d eclipse every few years. Grazing
geometry and periastron passages shorten T for a given P, which lets longer periods in, but each costs prior
weight. With the period, the binary predicts more eclipses at t_c + nP. Every other epoch the star has, in
every survey, is a chance to have caught one. When the epochs are dense enough that almost every orbit that fits
the observed event would have put another eclipse on a detectable epoch, and none is there, "eclipsing binary"
is the wrong explanation: the dip did not recur where an orbit says it must.

The test is a Monte Carlo over that hypothesis, conditioned on the observed event:

1. Epochs and the event, as `recurrence` defines them (in-dip: >= 4 robust sigma low and deeper than the band's
   largest brightening; Gaia BP/RP left out; events merged within 1 d). The test applies only when all surveys
   together show exactly one event. Its in-dip epochs bound the eclipse duration from below; the nearest epochs
   at normal brightness (< 2 sigma low) on either side bound it from above.
2. Draw companions from a mundane population: stars (period log-normal, mean 10^5.03 d, sigma 2.28 dex, for
   solar-type binaries; mass ratio uniform 0.1-1; main-sequence radius and G-band light from the mass) and dark
   bodies (planets and brown dwarfs, 0.03-0.2 Rsun, log-uniform period 0.5-10^4 d), eccentricity uniform to 0.7,
   argument of periastron uniform, impact parameter uniform across the transiting range. The host's radius and
   mass come from `physics.stellar_params` with 0.15 dex scatter, or from a broad prior if the parallax is poor.
3. Each draw's duration is drawn log-uniformly over what the bracketing epochs allow, the period follows from
   Kepler, and the draw carries the weight that the population, the geometric transit probability, and the
   chance that the observed epochs fall inside its eclipse give it. The eclipse's mid-time is placed so that it
   covers the observed in-dip epochs.
4. A trapezoid eclipse of the draw's depth (covered area / (1 + companion light)) predicts each epoch's depth.
   A draw fits the event if every in-dip epoch of the event is within 25% (+3 sigma) of the depth it predicts
   there, and every normal-brightness epoch in the same eclipse is predicted shallower than 5 sigma. Among
   fitting draws, the ones that predict a detectable depth (above the band's in-dip threshold, x1.2) at any
   epoch of another cycle have been refuted by the data.

    p_eb = weight of fitting draws that predict no detectable repeat / weight of all fitting draws

is the probability, under a mundane eclipsing companion, of seeing what the data show: one event and nothing
else. Small p_eb means an eclipse would almost surely have been seen again. The secondary eclipse is ignored
(conservative: it can only add repeats), and so is limb darkening (it shortens the detectable part of an
eclipse; the 1.2 margin covers it). The test cannot tell a one-off occulter from an artefact that happened
once: confirmation (both fields of view, BP/RP consistency, Stage 0) is a separate question, and `n_in_dip`
reports how many epochs carry the event.

Other numbers in the result: the duration bounds, the period range of the fitting draws that survive (the
eclipse ephemeris follow-up should test if the star is an EB after all), the expected number of detectable
repeats, the dip's colour (RP depth / BP depth, from Gaia epochs within the event), and the transverse speed
the fastest flux change in the event needs against the host's surface escape speed (no bound occulter moves
faster; a dip that needs more is an artefact or something in the foreground).
"""

import numpy as np

from . import physics, recurrence

G_SI = 6.674e-11
M_SUN = 1.989e30
LOGP_MEAN, LOGP_SIGMA = 5.03, 2.28          # Raghavan et al. 2010, solar-type binaries, log10(P / d)
F_BINARY, F_DARK = 0.46, 0.3                 # companions per star: stellar (all periods), dark bodies (0.5-1e4 d)
DARK_P = (0.5, 1e4)
DARK_R = (0.03, 0.2)                         # Rsun: Earth-ish to inflated hot Jupiter / brown dwarf
E_MAX = 0.7
P_MIN = 0.2                                  # days; anything faster is a contact binary, not an isolated dip
R_SCATTER_DEX = 0.15
BROAD_R, BROAD_M = (0.3, 30.0), (0.3, 3.0)   # host prior when the parallax is too poor for stellar_params
DETECT_MARGIN = 1.2
FIT_REL, FIT_SIGMA = 0.25, 3.0
FLAT_TOL_SIGMA = 5.0
MAX_T_DAYS = 3000.0
N_DRAWS = 40000
B_SHORTLIST = 2.5                            # single-dip flag: log10(one-off / eclipse) at least this (rule v4)
MIN_IN_DIP = 2                               # and the event on at least two epochs
CHUNK = 2000
T_FLOOR = 0.01                               # days: shortest eclipse drawn
TESS_DILUTION_UNKNOWN = 0.3


def _band_epochs(lcs, dilution=None):
    """Per epoch over the usable bands: t, band name, robust sigma (relative flux), in-dip threshold, z, depth.

    Same in-dip rule as `recurrence.epochs`; dilution (band -> fraction of the aperture's flux from other stars,
    e.g. TESS contamination) scales the depth an eclipse would show in that band."""
    dilution = dilution or {}
    out = {k: [] for k in ("t", "band", "sigma", "thresh", "z", "depth", "dil")}
    for lc in lcs:
        if lc.band in recurrence.SKIP_BANDS:
            continue
        ok = np.isfinite(lc.time) & np.isfinite(lc.flux)
        t, f = lc.time[ok], lc.flux[ok]
        if len(t) < 10:
            continue
        med = np.median(f)
        s = recurrence._robust_sigma(f / med)
        if not s > 0:
            continue
        z = (1 - f / med) / s
        k = max(recurrence.DIP_SIGMA, float(np.max(-z)))
        d = float(dilution.get(lc.band, 0.0) or 0.0)
        n = len(t)
        out["t"].append(t)
        out["band"].append(np.full(n, lc.band, dtype=object))
        out["sigma"].append(np.full(n, s))
        out["thresh"].append(np.full(n, k * s))
        out["z"].append(z)
        out["depth"].append(1 - f / med)
        out["dil"].append(np.full(n, min(max(d, 0.0), 0.99)))
    if not out["t"]:
        return None
    ep = {k: np.concatenate(v) for k, v in out.items()}
    o = np.argsort(ep["t"])
    return {k: v[o] for k, v in ep.items()}


def overlap_fraction(d, k):
    """Fraction of a unit disk covered by a disk of radius k whose centre is d away."""
    d, k = np.broadcast_arrays(np.asarray(d, float), np.asarray(k, float))
    out = np.zeros(d.shape)
    inside = d <= np.abs(1 - k)
    out[inside] = np.minimum(k[inside], 1.0) ** 2
    part = (d < 1 + k) & ~inside
    dd, kk = d[part], k[part]
    a1 = np.arccos(np.clip((dd**2 + 1 - kk**2) / (2 * dd), -1, 1))
    a2 = np.arccos(np.clip((dd**2 + kk**2 - 1) / (2 * dd * kk), -1, 1))
    tri = 0.5 * np.sqrt(np.clip((-dd + 1 + kk) * (dd + 1 - kk) * (dd - 1 + kk) * (dd + 1 + kk), 0, None))
    out[part] = (a1 + kk**2 * a2 - tri) / np.pi
    return out


def _host(source, n, rng):
    """Host radius and mass per draw (solar units)."""
    sp = physics.stellar_params(source or {})
    if sp is None:
        r = np.exp(rng.uniform(np.log(BROAD_R[0]), np.log(BROAD_R[1]), n))
        m = np.exp(rng.uniform(np.log(BROAD_M[0]), np.log(BROAD_M[1]), n))
        return r, m, False
    r0, m0 = sp
    r = r0 * 10 ** (R_SCATTER_DEX * rng.standard_normal(n))
    m = m0 * 10 ** (0.5 * R_SCATTER_DEX * rng.standard_normal(n))
    return r, m, True


def _companions(r1, m1, rng):
    """Companion radius ratio k, G-band light ratio L2/L1, total mass, period prior density per dex, and the
    period range the population allows (as log10 days)."""
    n = len(r1)
    dark = rng.random(n) < F_DARK / (F_DARK + F_BINARY)
    q = rng.uniform(0.1, 1.0, n)
    m2 = np.where(dark, 0.0, q * m1)
    r2_star = np.where(r1 < 1.5, r1 * q ** 0.8, np.clip(m2, 0.08, None) ** 0.8)   # evolved host: dwarf companion
    r2 = np.where(dark, np.exp(rng.uniform(np.log(DARK_R[0]), np.log(DARK_R[1]), n)), r2_star)
    # G-band light: main-sequence L ~ M^4 against the host's (from its own mass if a dwarf, else large).
    l1 = np.where(r1 < 1.5, np.clip(m1, 0.08, None) ** 4, r1 ** 2 * 1.0)
    l2 = np.where(dark, 0.0, np.clip(m2, 0.08, None) ** 4)
    return r2 / r1, l2 / l1, m1 + m2, dark


def _logp_density(logp, dark):
    """Companions per star per dex of period."""
    star = F_BINARY * np.exp(-0.5 * ((logp - LOGP_MEAN) / LOGP_SIGMA) ** 2) / (LOGP_SIGMA * np.sqrt(2 * np.pi))
    lo, hi = np.log10(DARK_P[0]), np.log10(DARK_P[1])
    planet = np.where((logp >= lo) & (logp <= hi), F_DARK / (hi - lo), 0.0)
    return np.where(dark, planet, star)


def _event(ep):
    """The single event: indices of its in-dip epochs and the duration bounds from the bracketing flat epochs.

    In-dip epochs belong to one event unless an epoch at normal brightness that would have shown the dip lies
    between them; a gap in the data does not split an event (a dip lasting days, seen on Gaia visits days apart,
    is one event). Returns (event or None, number of events)."""
    t = ep["t"]
    in_dip = ep["z"] > ep["thresh"] / ep["sigma"]
    idx = np.flatnonzero(in_dip)
    if not len(idx):
        return None, 0
    deep = float(np.max(ep["depth"][idx]))
    # A flat epoch bounds the event only if the observed depth would have shown there.
    flat = (ep["z"] < recurrence.FLAT_SIGMA) & (deep * (1 - ep["dil"]) >= FLAT_TOL_SIGMA * ep["sigma"])
    n_flat_before = np.cumsum(flat)
    splits = np.flatnonzero(n_flat_before[idx[1:]] - n_flat_before[idx[:-1]] > 0)
    if len(splits):
        return None, len(splits) + 1
    t_a, t_b = t[idx].min(), t[idx].max()
    before, after = flat & (t < t_a), flat & (t > t_b)
    t_lo = t[before].max() if before.any() else -np.inf
    t_hi = t[after].min() if after.any() else np.inf
    return {"idx": idx, "t_a": float(t_a), "t_b": float(t_b), "t_before": float(t_lo), "t_after": float(t_hi),
            "t_deep": float(t[idx[np.argmax(ep["depth"][idx])]]), "depth": deep}, 1


def _trapezoid(dt, T, tau, depth):
    """Depth at time offsets dt from mid-eclipse for a trapezoid of full duration T and ingress tau."""
    x = T / 2 - np.abs(dt)
    return np.where(x <= 0, 0.0, depth * np.clip(x / np.maximum(tau, 1e-9), 0, 1))


def eb_probability(ep, event, source, n_draws=N_DRAWS, rng=None):
    """Monte Carlo of the eclipsing-companion hypothesis given the event. Returns a dict of results."""
    rng = rng or np.random.default_rng(0)
    t, depth_obs, sig, thresh, dil = ep["t"], ep["depth"], ep["sigma"], ep["thresh"], ep["dil"]
    idx = event["idx"]
    span_lo = max(event["t_b"] - event["t_a"], 1e-3)
    span_hi = min(event["t_after"] - event["t_before"], MAX_T_DAYS)
    r1, m1, known = _host(source, n_draws, rng)
    k, lr, mtot, dark = _companions(r1, m1, rng)
    e = rng.uniform(0, E_MAX, n_draws)
    w_peri = rng.uniform(0, 2 * np.pi, n_draws)
    b = rng.uniform(0, 1 + k)
    # The event's in-dip epochs may sit in different cycles of a short orbit (gaps without data do not split an
    # event), so durations below the event's span are drawn too.
    t_floor = min(T_FLOOR, span_lo)
    logT = rng.uniform(np.log(t_floor), np.log(span_hi), n_draws)
    T = np.exp(logT)
    ecc = np.sqrt(1 - e**2) / (1 + e * np.sin(w_peri))
    chord = np.sqrt(np.clip((1 + k) ** 2 - b**2, 0, None))
    # T = P / pi * (R1 chord / a) * ecc with a = (G M P^2 / 4 pi^2)^(1/3)  =>  T = C P^(1/3)
    r1_m = r1 * physics.R_SUN
    gm = G_SI * M_SUN * mtot
    c = (r1_m * chord * ecc / np.pi) * (4 * np.pi**2 / gm) ** (1 / 3)        # seconds^(2/3)
    P = (T * physics.DAY / np.maximum(c, 1e-30)) ** 3 / physics.DAY
    a = (gm * (P * physics.DAY) ** 2 / (4 * np.pi**2)) ** (1 / 3)
    r_sum = r1_m * (1 + k)
    ok = (P >= P_MIN) & (a > 2 * r_sum) & (chord > 0)
    logp = np.log10(np.where(ok, P, 1.0))
    # Flat part of the eclipse (second to third contact) for the trapezoid.
    flat_chord = np.sqrt(np.clip((1 - k) ** 2 - b**2, 0, None))
    tau = 0.5 * T * (1 - flat_chord / np.maximum(chord, 1e-12))
    depth = overlap_fraction(np.where(b < np.abs(1 - k), 0.0, b), k) / (1 + lr)
    p_tr = r_sum / a * (1 + e * np.sin(w_peri)) / (1 - e**2)
    # The deepest epoch is in eclipse: t_c within T/2 of it, which a random phase gives with probability T / P.
    tc = event["t_deep"] - T / 2 + rng.random(n_draws) * T
    w = _logp_density(logp, dark) * 3.0 * p_tr * T / np.maximum(P, 1e-9) * ok
    w[~np.isfinite(w)] = 0.0
    fit = np.zeros(n_draws, bool)
    repeat = np.zeros(n_draws, bool)
    n_rep = np.zeros(n_draws)
    flat_near = (ep["z"] < recurrence.FLAT_SIGMA)
    inside = (t >= event["t_before"]) & (t <= event["t_after"])
    live = np.flatnonzero(w > 0)
    for s in range(0, len(live), CHUNK):
        j = live[s:s + CHUNK]
        dt = t[None, :] - tc[j, None]
        cyc = np.round(dt / P[j, None])
        dphase = dt - cyc * P[j, None]
        pred = _trapezoid(dphase, T[j, None], tau[j, None], depth[j, None]) * (1 - dil[None, :])
        same = inside[None, :]
        # Fit: in-dip epochs match the predicted depth; flat epochs between the bracketing ones stay shallow.
        d_obs = depth_obs[idx][None, :]
        tol = FIT_REL * d_obs + FIT_SIGMA * sig[idx][None, :]
        good_dip = np.all(np.abs(pred[:, idx] - d_obs) <= tol, axis=1)
        bad_flat = np.any(same & flat_near[None, :] & (pred > FLAT_TOL_SIGMA * sig[None, :]), axis=1)
        fit[j] = good_dip & ~bad_flat
        det = (~same) & (pred >= DETECT_MARGIN * thresh[None, :])
        repeat[j] = det.any(axis=1)
        n_rep[j] = det.sum(axis=1)
    wf = w * fit
    tot = wf.sum()
    log_range = np.log(span_hi / t_floor)
    out = {"n_draws": int(n_draws), "n_fit": int(fit.sum()), "host_known": bool(known),
           "duration_min_days": round(float(span_lo), 4),
           "duration_max_days": round(float(span_hi), 4) if np.isfinite(span_hi) else None}
    if not tot > 0 or fit.sum() < 20:
        out["p_eb"] = None
        return out
    surv = wf * ~repeat
    out["p_eb"] = float(surv.sum() / tot)
    # Per-star probability that a mundane companion produces exactly this: the event and no repeat. With no
    # surviving draw, half a draw of the typical fitting weight stands in (an upper bound at this resolution).
    s_surv = surv.sum() if surv.sum() > 0 else 0.5 * float(np.median(wf[fit]))
    out["eb_rate"] = float(s_surv / n_draws * log_range)
    out["expected_repeat_epochs"] = float((wf * n_rep).sum() / tot)
    out["eb_period_median_days"] = float(_wquantile(P, wf, 0.5))
    if surv.sum() > 0:
        out["surviving_period_5_95_days"] = [float(_wquantile(P, surv, 0.05)), float(_wquantile(P, surv, 0.95))]
    out["dark_companion_share"] = float((wf * dark).sum() / tot)
    return out


ONEOFF_T = (0.01, 1000.0)       # days: the one-off occulter's duration prior, log-uniform
ONEOFF_DEPTH = (1e-3, 1.0)
ONEOFF_RATE = 1e-3              # events per day per star; only the ratio's ranking matters


def oneoff_rate(ep, event, n_draws=N_DRAWS, rng=None):
    """Per-star probability that a one-off occulter of unknown duration, depth and ingress (log-uniform priors,
    at ONEOFF_RATE per day) produces this event, scored with the same fit rules as the eclipse draws."""
    rng = rng or np.random.default_rng(1)
    t, depth_obs, sig = ep["t"], ep["depth"], ep["sigma"]
    idx = event["idx"]
    span_lo = max(event["t_b"] - event["t_a"], 1e-3)
    span_hi = min(event["t_after"] - event["t_before"], MAX_T_DAYS)
    lo, hi = max(span_lo, ONEOFF_T[0]), min(span_hi, ONEOFF_T[1])
    if not hi > lo:
        return 0.0
    T = np.exp(rng.uniform(np.log(lo), np.log(hi), n_draws))
    depth = np.exp(rng.uniform(np.log(ONEOFF_DEPTH[0]), np.log(ONEOFF_DEPTH[1]), n_draws))
    tau = T * rng.uniform(0.01, 0.5, n_draws)
    shape = rng.integers(0, 2, n_draws)          # 0 trapezoid, 1 gaussian-like (tau = T / 2)
    tau = np.where(shape == 1, T / 2, tau)
    win = np.clip(T - (event["t_b"] - event["t_a"]), 0, None)
    tc = event["t_b"] - T / 2 + rng.random(n_draws) * win
    near = np.flatnonzero((t >= event["t_before"]) & (t <= event["t_after"]))
    pred = _trapezoid(t[near][None, :] - tc[:, None], T[:, None], tau[:, None], depth[:, None]) * (1 - ep["dil"][near][None, :])
    pos = {int(i): n for n, i in enumerate(near)}
    cols = np.array([pos[int(i)] for i in idx])
    d_obs = depth_obs[idx][None, :]
    tol = FIT_REL * d_obs + FIT_SIGMA * sig[idx][None, :]
    good = np.all(np.abs(pred[:, cols] - d_obs) <= tol, axis=1)
    flat = (ep["z"][near] < recurrence.FLAT_SIGMA)[None, :]
    bad = np.any(flat & (pred > FLAT_TOL_SIGMA * sig[near][None, :]), axis=1)
    prior = np.log(hi / lo) / np.log(ONEOFF_T[1] / ONEOFF_T[0])
    return float(ONEOFF_RATE * np.mean(win * (good & ~bad)) * prior)


def _wquantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][min(np.searchsorted(c, q * c[-1]), len(x) - 1)]


def colour(lcs, t_a, t_b, pad=0.01):
    """RP depth / BP depth over the Gaia epochs of the event, with a 1-sigma error, or None."""
    by = {lc.band: lc for lc in lcs}
    if "BP" not in by or "RP" not in by:
        return None
    d, e = {}, {}
    for band in ("BP", "RP"):
        lc = by[band]
        ok = np.isfinite(lc.flux) & np.isfinite(lc.time)
        f, tt = lc.flux[ok], lc.time[ok]
        med = np.median(f)
        s = recurrence._robust_sigma(f / med)
        m = (tt >= t_a - pad) & (tt <= t_b + pad)
        if not m.any() or not s > 0:
            return None
        d[band] = float(np.mean(1 - f[m] / med))
        e[band] = s / np.sqrt(m.sum())
    if not abs(d["BP"]) > 2 * e["BP"]:
        return None
    r = d["RP"] / d["BP"]
    err = abs(r) * np.hypot(e["RP"] / max(abs(d["RP"]), 1e-9), e["BP"] / abs(d["BP"]))
    return round(r, 3), round(float(err), 3)


def speed_check(ep, event, source):
    """Transverse speed the fastest flux change between the event and its neighbours needs, and the host's
    surface escape speed (km/s). A rigid opaque occulter moving at v changes the covered fraction of a uniform
    disk at most 2 v / (pi R); limb darkening raises the centre by up to ~1.3."""
    sp = physics.stellar_params(source or {})
    if sp is None:
        return None
    r, m = sp
    t, depth, sig = ep["t"], ep["depth"], ep["sigma"]
    lo, hi = max(event["t_before"], event["t_a"] - 5.0), min(event["t_after"], event["t_b"] + 5.0)
    near = np.flatnonzero((t >= lo - 1e-9) & (t <= hi + 1e-9))
    dt = t[near][None, :] - t[near][:, None]
    dd = np.abs(depth[near][None, :] - depth[near][:, None]) - 3 * np.hypot(sig[near][None, :], sig[near][:, None])
    rate = np.where((dt > 0) & (dd > 0), dd / np.where(dt > 0, dt, 1.0), 0.0)
    best = float(rate.max()) if rate.size else 0.0
    v_req = np.pi * r * physics.R_SUN * best / (2 * 1.3 * physics.DAY) / 1e3
    v_esc = np.sqrt(2 * G_SI * M_SUN * m / (r * physics.R_SUN)) / 1e3
    return round(float(v_req), 1), round(float(v_esc), 1)


def test(lcs, source=None, context=None, n_draws=N_DRAWS, rng=None):
    """Packet test dict for a list of visible light curves, or None when there is not exactly one dip event."""
    # TESS's 21" pixels dilute a change on the target by the neighbours' share of the aperture; without the
    # contamination ratio, assume a sizeable one, which can only make repeats harder to see (conservative).
    c = (context or {}).get("tess_contamination")
    dil = {"TESS": float(c) if c is not None and np.isfinite(float(c)) else TESS_DILUTION_UNKNOWN}
    ep = _band_epochs(lcs, dil)
    if ep is None:
        return None
    event, n_ev = _event(ep)
    if event is None:
        return None
    res = eb_probability(ep, event, source, n_draws, rng)
    if res.get("eb_rate"):
        one = oneoff_rate(ep, event, n_draws, rng)
        res["log10_oneoff_vs_eb"] = round(float(np.log10(max(one, 1e-300) / res["eb_rate"])), 2) if one > 0 else None
    n_in = int(len(event["idx"]))
    out = {"n_in_dip": n_in, "bands_in_dip": sorted(set(ep["band"][event["idx"]].tolist())),
           "depth": round(event["depth"], 4), **res}
    out["bands_in_dip"] = ",".join(out["bands_in_dip"])
    col = colour(lcs, event["t_a"], event["t_b"])
    if col:
        out["colour_ratio_rp_bp"], out["colour_ratio_err"] = col
    sp = speed_check(ep, event, source)
    if sp:
        out["v_required_km_s"], out["v_escape_km_s"] = sp
    bf = out.get("log10_oneoff_vs_eb")
    out["single_dip_flag"] = bool(bf is not None and bf >= B_SHORTLIST and n_in >= MIN_IN_DIP)
    out["description"] = (
        "Single-event test, for stars whose epochs in all surveys show exactly one dip event. An eclipse of the "
        "observed duration on this star implies an orbital period (Kepler), and that period predicts more eclipses. "
        "p_eb = probability, under a population of mundane eclipsing companions (binary stars and dark bodies, any "
        "impact parameter, eccentricity up to 0.7) that fit this event's depth and duration bounds, that none of the "
        "star's other epochs in any survey would have caught a detectable repeat. Small p_eb: an eclipsing companion "
        "would almost surely have been seen again. expected_repeat_epochs: mean number of epochs that should have "
        "shown a repeat. surviving_period_5_95_days: periods of the eclipse draws the data do not refute (a follow-up "
        "ephemeris). n_in_dip: epochs carrying the event (one epoch alone is often an artefact). colour_ratio_rp_bp: "
        "RP depth / BP depth (1 grey, < 1 dust-like reddening, < 0 one band brightened). v_required_km_s: transverse "
        "speed the fastest flux change needs; v_escape_km_s: the host's surface escape speed, which no bound occulter "
        "exceeds. log10_oneoff_vs_eb: log10 of how much better a one-off occulter of unknown duration, depth and "
        "ingress (log-uniform priors) explains the event and the absence of repeats than the eclipsing-companion "
        "population does (large: an eclipse is a poor explanation, because eclipses that fit are rarely caught "
        "once and never again; it says nothing about which non-eclipse cause, astrophysical or instrumental). "
        "single_dip_flag: log10_oneoff_vs_eb >= 2.5 with the event on at least two epochs.")
    out["visible_only"] = True
    return out
