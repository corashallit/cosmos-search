"""Baseline model, flux-level eclipse prediction and residuals for one star (see the package docstring).

    v = run(lightcurves, source, t_target)

`v["demote"]` is the ranking decision: the eclipse baseline explains the selected dip at flux level and leaves no
residual event anywhere, so the star moves from the vetting queue to the EB control pool. `v["eb_residual"]` marks an
eclipsing binary with something left over, which ranking treats as a flag in the star's favour.
"""

import numpy as np

from cosmos_search.ebfilter import decide as D
from cosmos_search.ebfilter import search as S
from cosmos_search.vetting import physics, recurrence
from cosmos_search.vetting.candidate import LightCurve

RES_SIGMA = 4.0             # an in-window epoch deeper than the eclipse allows by this many combined sigma (as DIP_SIGMA)
NEIGHBOUR_PHASE = 0.125     # same eclipse phase: within this share of the window's phase width (the period grid's drift)
TOL = 0.2                   # relative depth tolerance, same band; placeholder until set on half A (docs/arms/residual.md)
CROSS_FACTOR = 2.0          # tolerance multiplier when the prediction comes from another band or the whole window
MIN_EVENT_EPOCHS = 1        # residual events elsewhere need this many in-dip epochs; placeholder until set on half A
PAIR_BANDS = ("BP", "RP")   # Gaia's colour bands share G's transits: they follow G's decision
MIN_RATIO_PAIRS = 2        # other-cycle epochs of a band needed to measure its eclipse depth ratio to the other bands
RATIO_SIGMA = 5.0          # ... each where the other bands predict an eclipse at least this deep (their sigma)
RATIO_MAX = 3.0            # the ratio is clipped to [1/RATIO_MAX, RATIO_MAX]
MARGIN = 1.0               # window widths either side of a window over which the eclipse is predicted
REFINE_STEPS = 81          # trial frequencies across two grid steps when refining the period on depths
PAIR_MATCH_D = 0.01         # days; a BP/RP epoch this close to a G epoch is the same transit

# how each in-window prediction was made
OUTSIDE, SAME_BAND, OTHER_BAND, WINDOW, NONE = 0, 1, 2, 3, 4


def baseline_solution(labels, res, sp, delta_bits=D.DELTA_BITS):
    """The eclipse baseline: the first solution among the good ones (`ebfilter.decide.good_solutions`) whose primary
    window is pure and whose dips fit a stellar orbit. A secondary window that is not pure is dropped. None: the
    baseline is constant."""
    t, lab, bi = labels[:3]
    keep = lab >= 0
    for s in D.good_solutions(res, delta_bits):
        if D.window_purity(t, lab, bi, s["period"], s["s1"], s["q1"]) < D.MIN_PURITY:
            continue
        if D.duration_fits(D.duration_lo(t[keep], lab[keep], s), s["period"], sp) is False:
            continue
        s = dict(s)
        if s["q2"] > 0 and D.window_purity(t, lab, bi, s["period"], s["s2"], s["q2"]) < D.MIN_PURITY:
            s["s2"], s["q2"] = 0.0, -1.0
        return s
    return None


def windows(sol, double=False):
    """(period, [(start, width), ...]) of a solution, or of the same solution read at twice its period, where each
    window splits into two (cycles of alternate parity: a primary and a secondary eclipse folded together)."""
    w = [(sol["s1"], sol["q1"])] + ([(sol["s2"], sol["q2"])] if sol["q2"] > 0 else [])
    if not double:
        return sol["period"], w
    return 2 * sol["period"], [((s + k) / 2, q / 2) for s, q in w for k in (0, 1)]


def _covered(t, period, wins):
    ph = (t / period) % 1.0
    out = np.zeros(len(t), bool)
    for s, q in wins:
        out |= ((ph - s) % 1.0) <= q
    return out


def refine(t, bi, frac, sig, period, wins, n=REFINE_STEPS):
    """The period refined on the depths, not the labels: the search's grid can leave a phase drift of up to an
    eighth of a window across the span, which smears the eclipse profile that predictions read. Over the frequencies
    within two grid steps either side (windows kept centred on the in-window epochs' mean time), take the one with
    the smoothest profile: the least mean squared depth difference between phase-adjacent in-window epochs of the
    same band, in units of the band's sigma (string length)."""
    f0, q = 1.0 / period, max(w for _, w in wins)
    step = 2 * q / (S.PHASE_STEPS * max(float(t[-1] - t[0]), 1e-9))
    inw = _covered(t, period, [(s - MARGIN * w, (2 * MARGIN + 1) * w) for s, w in wins])
    if inw.sum() < 4:
        return period, wins
    t_ref = float(np.mean(t[inw]))
    best, best_l = (period, wins), np.inf
    for d in np.linspace(-step, step, n):
        f = f0 + d
        ws = [((s + t_ref * d) % 1.0, w * f / f0) for s, w in wins]
        m_all = _covered(t, 1.0 / f, [(s - MARGIN * w, (2 * MARGIN + 1) * w) for s, w in ws])
        total, cnt = 0.0, 0
        for b in np.unique(bi[m_all]):
            m = np.flatnonzero(m_all & (bi == b))
            if len(m) < 2:
                continue
            o = m[np.argsort((t[m] * f) % 1.0)]
            total += float(np.sum((np.diff(frac[o]) / sig[o][1:]) ** 2))
            cnt += len(o) - 1
        if cnt and total / cnt < best_l:
            best_l, best = total / cnt, (1.0 / f, ws)
    return best


def _band_sigma(frac, z, bi):
    """Robust sigma (relative flux) per epoch: z = frac / sigma within each band."""
    sig = np.full(len(z), np.nan)
    for b in np.unique(bi):
        m = bi == b
        ok = m & (np.abs(z) > 1e-9)
        sig[m] = np.median(frac[ok] / z[ok]) if ok.any() else np.nan
    return sig


def _neighbours(i, other, x, bi, h, same_band):
    """The nearest other-cycle epoch at or before epoch i's phase and the nearest at or after, within h, from i's own
    band (same_band) or from the other bands."""
    dx = x[other] - x[i]
    sel = (np.abs(dx) <= h) & ((bi[other] == bi[i]) if same_band else (bi[other] != bi[i]))
    out = []
    for m, pick in ((sel & (dx <= 0), np.argmax), (sel & (dx >= 0), np.argmin)):
        if m.any():
            out.append(int(other[m][pick(dx[m])]))
    return out


def predict(t, lab, bi, frac, sig, period, wins, tol=TOL, cross=CROSS_FACTOR):
    """Per epoch: in_window, d_hi (deepest depth the eclipse allows there, tolerance included), sig_n (the noise of
    the epoch the prediction came from) and how (OUTSIDE, SAME_BAND, OTHER_BAND, WINDOW, NONE).

    The eclipse is predicted over each window widened by MARGIN window widths either side, because the label
    search's windows come from a coarse grid and an eclipse's ingress can lie outside them. An epoch's prediction is
    the deeper of its two phase neighbours from other cycles (more than SAME_EVENT_D away, so a dip never explains
    itself), the nearest at or before its phase and the nearest at or after, each within NEIGHBOUR_PHASE of the
    window width: on a monotonic stretch of the profile the true depth lies between them.

    - SAME_BAND: both neighbours from the epoch's own band; tolerance tol.
    - OTHER_BAND: a same-band neighbour on one side only (cross x tol), or the other bands' neighbours (one side is
      enough; the deeper of the two readings is used), times the star's own depth ratio between this band and
      the others, measured on this band's other in-eclipse epochs (median of at least MIN_RATIO_PAIRS, each from
      another cycle, where the other bands predict at least RATIO_SIGMA sigma): tolerance tol. Eclipse depths differ
      between bands by colour and by blending (ZTF's PSF takes in neighbours Gaia resolves), so without a measured
      ratio the tolerance is cross x tol.
    - WINDOW: no neighbour at that phase: the deepest other-cycle epoch in the window itself (an eclipse is nowhere
      deeper than its deepest point); cross x tol.
    period, wins: from `windows`."""
    n = len(t)
    inw = np.zeros(n, bool)
    d_hi, sig_n = np.zeros(n), np.zeros(n)
    how = np.zeros(n, np.int8)
    ph = (t / period) % 1.0
    for s, q in wins:
        x = (ph - s + MARGIN * q) % 1.0                  # phase from the widened window's start
        core = (x >= MARGIN * q) & (x <= (MARGIN + 1) * q)
        reg = x <= (2 * MARGIN + 1) * q
        part = np.isin(bi, np.unique(bi[core & (lab == 1)]))
        idx = np.flatnonzero(reg & part & ~inw)
        h = NEIGHBOUR_PHASE * q
        other_of = {i: idx[np.abs(t[idx] - t[i]) > D.SAME_EVENT_D] for i in idx}
        # Cross-band prediction for every epoch, for the per-band depth ratio.
        xb = {}
        for i in idx:
            nb = _neighbours(i, other_of[i], x, bi, h, False)
            if nb:
                j = nb[int(np.argmax(frac[nb]))]
                xb[i] = (float(frac[j]), float(sig[j]))
        for i in idx:
            inw[i] = True
            other = other_of[i]
            nb = _neighbours(i, other, x, bi, h, True)
            if len(nb) == 2:
                j = nb[int(np.argmax(frac[nb]))]
                d_hi[i], sig_n[i], how[i] = max(float(frac[j]), 0.0) * (1.0 + tol), sig[j], SAME_BAND
                continue
            one = nb[0] if nb else None                    # a same-band neighbour on one side only
            if i in xb:
                pairs = [frac[k] / xb[k][0] for k in other if bi[k] == bi[i] and k in xb
                         and xb[k][0] >= RATIO_SIGMA * xb[k][1]]
                if len(pairs) >= MIN_RATIO_PAIRS:
                    r, f = float(np.clip(np.median(pairs), 1 / RATIO_MAX, RATIO_MAX)), 1.0
                else:
                    r, f = 1.0, cross
                d, sn = max(xb[i][0], 0.0) * r, xb[i][1] * r
                if one is not None and frac[one] > d:
                    d, sn = float(frac[one]), sig[one]
                d_hi[i], sig_n[i], how[i] = d * (1.0 + f * tol), sn, OTHER_BAND
                continue
            if one is not None:
                d_hi[i], sig_n[i], how[i] = max(float(frac[one]), 0.0) * (1.0 + cross * tol), sig[one], OTHER_BAND
                continue
            win = other[core[other]]
            if not len(win):
                how[i] = NONE
                continue
            j = win[int(np.argmax(frac[win]))]
            d_hi[i], sig_n[i], how[i] = max(float(frac[j]), 0.0) * (1.0 + cross * tol), sig[j], WINDOW
    return inw, d_hi, sig_n, how


def residual_lightcurves(lcs, idx, t, drop, excess):
    """Copies of lcs with the explained in-window epochs removed and residual dips reduced to their excess depth.

    idx: (light-curve index, position) per labelled epoch (`search.labels(with_index=True)`); drop: epochs to remove;
    excess: per epoch, the depth to subtract from the flux (relative to the band median), 0 to leave it. BP and RP
    epochs follow the G transit they share."""
    removed = {}
    scale = {}
    for k in np.flatnonzero(drop):
        removed.setdefault(int(idx[k, 0]), []).append(int(idx[k, 1]))
    for k in np.flatnonzero(excess != 0):
        scale.setdefault(int(idx[k, 0]), {})[int(idx[k, 1])] = float(excess[k])
    g_drop = np.array([t[k] for k in np.flatnonzero(drop) if lcs[int(idx[k, 0])].band == "G"])
    out = []
    for li, lc in enumerate(lcs):
        keep = np.ones(len(lc.time), bool)
        flux = lc.flux.astype(float).copy()
        if li in removed:
            keep[removed[li]] = False
        if lc.band in PAIR_BANDS and len(g_drop):
            near = np.min(np.abs(lc.time[:, None] - g_drop[None, :]), axis=1) <= PAIR_MATCH_D
            keep &= ~near
        if li in scale:
            med = np.median(flux[np.isfinite(flux)])
            for pos, d in scale[li].items():
                flux[pos] = flux[pos] + d * med
        out.append(LightCurve(lc.band, lc.time[keep], flux[keep], lc.flux_err[keep],
                              {k: v[keep] for k, v in lc.meta.items()}))
    return out


def _events_with(rec_epochs, min_epochs):
    """Events (recurrence.events) on the residual epochs, and how many have at least min_epochs in-dip epochs."""
    t, d, z = rec_epochs
    ev = recurrence.events(t, d, z)
    if not len(ev):
        return ev, 0
    td = t[d]
    n_in = np.array([int(np.sum((td >= a) & (td <= b))) for _, a, b, _ in ev])
    return ev, int(np.sum(n_in >= min_epochs))


def prepare(lcs, source=None, t_target=None, delta_bits=D.DELTA_BITS):
    """The parts of `run` that do not depend on the tolerances: labels, the period search, the EB filter's verdict
    and the eclipse baseline (period and windows refined on the depths, at the solution's period and at twice it)."""
    labs = S.labels(lcs, with_index=True)
    t, lab, bi, z, names, frac, idx = labs
    res = S.search(t, lab, bi, names)
    eb = D.judge(labs, res, source, t_target, delta_bits)
    sp = physics.stellar_params(source) if source else None
    base = eb["solution"] or baseline_solution(labs, res, sp, delta_bits)
    ctx = {"lcs": lcs, "labels": labs, "res": res, "eb": eb, "base": base, "t_target": t_target}
    if base:
        sig = _band_sigma(frac, z, bi)
        ctx["sig"] = sig
        ctx["models"] = [(parity, *refine(t, bi, frac, sig, *windows(base, parity))) for parity in (False, True)]
    return ctx


def run(lcs, source=None, t_target=None, delta_bits=D.DELTA_BITS, **kw):
    """The residual verdict for one star (dict). lcs: light curves of every survey (vetting.candidate.LightCurve);
    source: phot_g_mean_mag, bp_rp, parallax, parallax_error; t_target: time of the selected Gaia dip. kw: tol,
    cross, min_event_epochs, recurrence_test (`evaluate`)."""
    return evaluate(prepare(lcs, source, t_target, delta_bits), **kw)


def evaluate(ctx, tol=TOL, cross=CROSS_FACTOR, min_event_epochs=MIN_EVENT_EPOCHS, recurrence_test=True):
    """Residuals and the verdict under one setting of the tolerances, from `prepare`."""
    t, lab, bi, z, names, frac, idx = ctx["labels"]
    res, eb, base, t_target, lcs = ctx["res"], ctx["eb"], ctx["base"], ctx["t_target"], ctx["lcs"]
    out = {"n_epochs": int(len(t)), "n_dip_epochs": int((lab == 1).sum()), "bands": ",".join(names),
           "delta_bits": float(res["delta_bits"]), "eb_explained": bool(eb["explained"]),
           "eb_target_covered": eb["target_covered"], "periodic_unphysical": bool(eb["periodic_unphysical"]),
           "baseline": "eclipse" if base else "constant", "period": np.nan,
           "baseline_delta_bits": base["delta_bits"] if base else np.nan}
    if base:
        sig = ctx["sig"]
        # The eclipse model at the solution's period or at twice it, whichever leaves fewer residual dips: both are
        # the same strict-period description of the dip labels, and only the depths can tell them apart.
        best = None
        for parity, period, wins in ctx["models"]:
            inw, d_hi, sig_n, how = predict(t, lab, bi, frac, sig, period, wins, tol, cross)
            zr = np.where(inw & (how != NONE), (frac - d_hi) / np.hypot(sig, sig_n), np.inf)
            res_dip = (lab == 1) & (~inw | (zr >= RES_SIGMA))
            if best is None or res_dip.sum() < best[-1].sum():
                best = (parity, period, inw, d_hi, sig_n, how, zr, res_dip)
        parity, period, inw, d_hi, sig_n, how, zr, res_dip = best
        out.update({"double_period": bool(parity), "period": float(period)})
        drop = inw & ~res_dip
        excess = np.where(inw & res_dip & (how != NONE), d_hi, 0.0)
        out.update({"n_in_window": int(inw.sum()), "n_predicted_same_band": int((how == SAME_BAND).sum()),
                    "n_predicted_other_band": int((how == OTHER_BAND).sum()),
                    "n_predicted_window": int((how == WINDOW).sum()), "n_unpredicted": int((how == NONE).sum())})
    else:
        inw = np.zeros(len(t), bool)
        zr = np.full(len(t), np.inf)
        how = np.zeros(len(t), np.int8)
        res_dip = lab == 1
    out["n_residual_dip_epochs"] = int(res_dip.sum())
    if t_target is not None and len(t):
        k = int(np.argmin(np.abs(t - t_target)))
        out.update({"target_in_window": bool(inw[k]), "target_residual": not bool(inw[k] and zr[k] < RES_SIGMA),
                    "target_residual_sigma": float(zr[k]) if np.isfinite(zr[k]) else np.nan,
                    "target_prediction": int(how[k])})
    else:
        out.update({"target_in_window": None, "target_residual": None, "target_residual_sigma": np.nan,
                    "target_prediction": None})
    rlcs = residual_lightcurves(lcs, idx, t, drop, excess) if base else lcs
    rec_ep = recurrence.epochs(rlcs)
    ev, n_ev = _events_with(rec_ep, min_event_epochs)
    # The selected dip's own event does not count here: target_residual covers it.
    if t_target is not None and len(ev):
        own = (ev[:, 1] - D.SAME_EVENT_D <= t_target) & (t_target <= ev[:, 2] + D.SAME_EVENT_D)
        if own.any():
            td = rec_ep[0][rec_ep[1]]
            n_own = [int(np.sum((td >= a) & (td <= b))) for _, a, b, _ in ev[own]]
            n_ev -= sum(n >= min_event_epochs for n in n_own)
    out["n_residual_events"] = int(len(ev))
    out["n_residual_events_elsewhere"] = int(n_ev)
    if recurrence_test:
        r = recurrence.search(*rec_ep)
        out.update({"residual_testable": bool(r["testable"]), "residual_aperiodic": bool(r.get("aperiodic", False)),
                    "residual_aligned_fraction": r.get("aligned_fraction", np.nan)})
    out["demote"] = bool(eb["explained"] and base and n_ev == 0 and out["target_residual"] is False)
    out["eb_residual"] = bool(base) and not out["demote"] and bool(out["target_residual"] or n_ev > 0)
    out["residual_lightcurves"] = rlcs
    return out
