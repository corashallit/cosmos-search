"""The filter's verdict for one queued star: is its selected dip an eclipse of a binary?

    verdict = decide(lightcurves, source, t_target)

A star is **explained** (removed from the vetting queue and kept as a known-answer control) only when all of these
hold:

1. **A strict period describes the dips** (`search.search`): the eclipse code saves at least DELTA_BITS bits over
   placing the dip runs at random among the epochs. Under that null P(saving >= k) <= 2^-k on the star's own
   cadence, so the threshold is a false-alarm bound per star, not a tuned score.
2. **The selected dip is one of the eclipses.** The epoch that put the star in the queue (`t_target`, the deepest
   Gaia G epoch) falls inside one of the solution's windows, and that window holds dips from other cycles at least
   as deep as it, within DEPTH_FACTOR x their depth + DEPTH_SLACK (in magnitudes, as ranking v0's DPAC check).
   An eclipsing binary with an extra dip at another phase, or deeper than its eclipses (a circumbinary occulter, a
   disk, debris passing during an eclipse), is not explained by its eclipses, so it stays in the queue.
3. **The eclipse duration fits an orbit at that period.** The dip epochs inside the primary window span at least
   duration_lo of the orbit. Two stars on an orbit of period P with radii R1 + R2 <= 2 R* (R* from Gaia photometry
   and parallax, as vetting.physics) and total mass >= M* cannot eclipse for longer than
   D_max = (P / pi) asin(min(1, 2 R* / a)) sqrt((1 + e) / (1 - e)), with e <= E_MAX; the solution's own period and
   twice it (equal eclipses folded together) are tried. A strict period whose dips last longer than any stellar
   companion allows is not an eclipsing binary; it stays in the queue flagged `periodic_unphysical`. With no usable
   parallax this check is skipped (`duration_ok` None).

Any solution among the best few periods (aliases) that meets 1-3 explains the star. The verdict also reports dips
outside the windows (`n_runs_outside`), so an explained EB with extra dips is visible to whoever reads the control.
"""

import math

import numpy as np

from cosmos_search.vetting import physics

from . import search as S

DELTA_BITS = 10.0
DEPTH_FACTOR = 1.5
DEPTH_SLACK = 0.05      # mag
SAME_EVENT_D = 1.0      # dips this close to the target are its own event, not another cycle
ALIAS_MARGIN = 10.0     # bits: periods this much worse than the best are 2^-10 as likely; not used to explain
EVENT_GAP_D = 1.5       # dip epochs with no flat between them and closer than this (and than P/2) are one eclipse       # pre-registered in docs/arms/eb-filter.md before the held-out half was scored
E_MAX = 0.9             # as vetting.physics.min_period_days
G_SI = 6.674e-11
M_SUN = 1.989e30
DAY = 86400.0


def max_duration_days(period_d, r_star, m_star, e_max=E_MAX):
    """Longest eclipse two stars with R1 + R2 <= 2 r_star (Rsun) and M >= m_star (Msun) can make at this period."""
    a = (G_SI * m_star * M_SUN * (period_d * DAY) ** 2 / (4 * math.pi ** 2)) ** (1 / 3)
    x = min(1.0, 2 * r_star * physics.R_SUN / a)
    return period_d / math.pi * math.asin(x) * math.sqrt((1 + e_max) / (1 - e_max))


def decide(lcs, source=None, t_target=None, delta_bits=DELTA_BITS, **search_kw):
    """Verdict dict for one star. lcs: light curves (vetting.candidate.LightCurve, any surveys; BP/RP skipped);
    source: dict with phot_g_mean_mag, bp_rp, parallax, parallax_error (for the duration check); t_target: time of
    the selected dip (None: not required)."""
    t, lab, bi, z, names, frac = S.labels(lcs)
    res = S.search(t, lab, bi, names, **search_kw)
    sp = physics.stellar_params(source) if source else None
    out = {k: v for k, v in res.items() if k != "solutions"}
    out.update({"explained": False, "periodic_unphysical": False, "duration_ok": None, "target_covered": None,
                "r_star": sp[0] if sp else np.nan, "m_star": sp[1] if sp else np.nan, "p_null_bound":
                float(min(1.0, 2.0 ** -res["delta_bits"])), "bands": names})
    sols = res.get("solutions", [])
    floor = max(delta_bits, sols[0]["delta_bits"] - ALIAS_MARGIN) if sols else delta_bits
    good = [s for s in sols if s["delta_bits"] >= floor]
    if not good:
        return out
    keep = lab >= 0
    tk, labk = t[keep], lab[keep]
    for s in good:
        cov_target = None
        if t_target is not None:
            cov_target = target_explained(t, lab, frac, t_target, s)
        dur_lo = duration_lo(tk, labk, s)
        dur_ok = None
        if sp:
            dmax = max(max_duration_days(s["period"], *sp), max_duration_days(2 * s["period"], *sp))
            dur_ok = bool(dur_lo <= dmax)
        if cov_target is not False and dur_ok is not False:
            out.update({"explained": True, "period": s["period"], "delta_bits": s["delta_bits"],
                        "duration_ok": dur_ok, "target_covered": cov_target, "duration_lo_d": dur_lo,
                        "solution_rank": res["solutions"].index(s)})
            return out
        if cov_target is not False and dur_ok is False:
            out["periodic_unphysical"] = True
    # Nothing explains it: report the best solution's target and duration status.
    s = good[0]
    out["target_covered"] = None if t_target is None else target_explained(t, lab, frac, t_target, s)
    return out


def duration_lo(t, lab, sol):
    """Lower bound on the eclipse duration under a solution: the longest run of dip epochs (consecutive among the
    labelled epochs, no flat between, gaps under min(EVENT_GAP_D, P/2)) that touches a window, or the phase extent
    of the dips in the primary window, whichever is longer."""
    p = sol["period"]
    cov = S.covered(t, p, sol["s1"], sol["q1"], sol["s2"], sol["q2"])
    ph = (t / p) % 1.0
    x = ((ph - sol["s1"]) % 1.0)[(lab == 1) & (((ph - sol["s1"]) % 1.0) <= sol["q1"])]
    best = float((x.max() - x.min()) * p) if len(x) > 1 else 0.0
    gap = min(EVENT_GAP_D, 0.5 * p)
    i, n = 0, len(t)
    while i < n:
        if lab[i] != 1:
            i += 1
            continue
        j = i
        while j + 1 < n and lab[j + 1] == 1 and t[j + 1] - t[j] <= gap:
            j += 1
        if cov[i:j + 1].any():
            best = max(best, float(t[j] - t[i]))
        i = j + 1
    return best


def _mag(frac):
    return -2.5 * np.log10(np.clip(1 - np.asarray(frac, float), 1e-3, None))


def target_explained(t, lab, frac, t_target, sol):
    """True if the epoch at t_target lies in one of the solution's windows and that window holds dips from other
    cycles at least 1 / DEPTH_FACTOR as deep (after DEPTH_SLACK)."""
    k = int(np.argmin(np.abs(t - t_target)))
    windows = [(sol["s1"], sol["q1"])] + ([(sol["s2"], sol["q2"])] if sol["q2"] > 0 else [])
    ph = (t / sol["period"]) % 1.0
    for s, q in windows:
        inw = ((ph - s) % 1.0) <= q
        if not inw[k]:
            continue
        other = inw & (lab == 1) & (np.abs(t - t[k]) > SAME_EVENT_D)
        if not other.any():
            return False
        return bool(_mag(frac[k]) <= DEPTH_FACTOR * _mag(frac[other]).max() + DEPTH_SLACK)
    return False
