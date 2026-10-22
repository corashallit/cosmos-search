"""Three tests that tell a blind vetter what an eclipse explanation predicts, which it cannot work out by eye.

The confirmatory batch of rule v5 (docs/dayone.md, "Rule v6") lost most held-out decoys to readings of "eclipsing
binary" held at 0.3-0.45, and shortlisted one real star that turned out to be a known eclipsing binary (WY Cen).
Each test below adds evidence in one direction or the other, and none of them quotes a calibration figure:

- `strict_period`: the EB filter's compression test (`ebfilter.search`) on the visible epochs. delta_bits is how
  much a strict-period eclipse description saves over placing the same dip epochs at random on the star's own
  cadence; P(delta_bits >= k | random placement) <= 2^-k. The recurrence test reports which period lines up the
  most events, but not how often chance does that as well on sparse epochs; this does.
- `catalogue_ephemeris`: for a star in a variable-star catalogue that is not built from Gaia (VSX entries not named
  after a Gaia release), the catalogue's type and period, and how many detected dip events fall in two narrow
  phase windows at that period.
- `orbit_conjunction`: for a star with a Gaia DR3 two-body orbit from radial velocities or astrometry (SB1, SB2,
  AstroSpectroSB1, Orbital), the times the two stars are lined up with us, and how far each dip event is from the
  nearest of them. Eclipses can only happen there. DPAC's own EclipsingBinary solutions are not used: they are
  fitted to the photometry the packet already shows.

All three run on the epochs the packet shows (tail holdout applied) and on absolute times.
"""

import math

import numpy as np

from cosmos_search.ebfilter import decide as D
from cosmos_search.ebfilter import search as S

from . import physics, recurrence

PHASE_WINDOW = 0.05          # catalogue ephemeris: least full width of each eclipse window, in phase
MAX_WINDOW = 0.25            # and most, whatever the longest eclipse the star's size allows
E_CATALOGUE = 0.5            # eccentricity allowed for that longest eclipse (catalogues rarely give one)
ORBIT_TYPES = ("SB1", "SB2", "SB2C", "AstroSpectroSB1", "Orbital", "OrbitalAlternative", "OrbitalTargetedSearch")
GAIA_REF_BTJD = 389.0        # J2016.0 = JD 2457389.0; Gaia's t_periastron is in days from it
TIMING_SIGMA = 2.0           # an event is at a conjunction within half the longest eclipse plus this many sigma
MAX_EVENTS = 12


def _events(lcs):
    t, d, z = recurrence.epochs(lcs)
    return recurrence.events(t, d, z), (t, d, z)


def strict_period(lcs, source=None, t_target=None):
    labs = S.labels(lcs)
    t, lab = labs[0], labs[1]
    if not len(t) or not (lab == 1).any():
        return None
    res = S.search(*labs[:3], labs[4])
    eb = D.judge(labs, res, source, t_target)
    delta = float(res["delta_bits"])
    per = res.get("period")
    out = {"delta_bits": round(delta, 1), "p_random_upper": float(f"{min(1.0, 2.0 ** -delta):.2g}"),
           "period": float(f"{per:.5g}") if per is not None and np.isfinite(per) else None,
           "n_dip_epochs": int((lab == 1).sum()), "filter_explains_selected_dip": bool(eb["explained"]),
           "duration_too_long_for_orbit": bool(eb["periodic_unphysical"]), "visible_only": True}
    out["description"] = (
        "Strict-period test (minimum description length) on every shown epoch of every survey except BP/RP. Each "
        "epoch is labelled dip, normal or neither; delta_bits = how many bits a description with one strict period "
        "(one or two eclipse windows) saves over placing the same dip epochs at random among this star's own "
        "epochs. If the dips were placed at random, delta_bits >= k happens with probability at most 2^-k "
        "(p_random_upper), whatever the cadence, so a period found at a sampling alias earns nothing unless it also "
        "keeps the normal epochs out of its window. Large delta_bits: a strict period is real. delta_bits near or "
        "below 0: no period stands out from chance, which is what dips that do not recur give, and also what an "
        "eclipsing binary seen too few times gives; it does not rule eclipses out. "
        "filter_explains_selected_dip: the deepest shown Gaia G dip sits in a pure eclipse window with comparable "
        "dips from other cycles and a duration a stellar orbit at that period allows. "
        "duration_too_long_for_orbit: the best period's dips last longer than two stars of this size could eclipse.")
    return out


def known_variable(row):
    """Catalogue type and period from the cross-match context row (xmatch's vsx_* columns), or None. Entries named
    after a Gaia release are skipped: VSX imports Gaia's own classifications, which use the photometry shown."""
    name = row.get("vsx_Name")
    if not isinstance(name, str) or not name or name.startswith("Gaia"):
        return None
    p = row.get("vsx_p_cal")
    if p is not None and np.isfinite(p) and p < 0.5:
        return None
    per = row.get("vsx_Period")
    return {"type": str(row.get("vsx_Type") or "VAR"),
            "period_d": float(per) if per is not None and np.isfinite(per) and per > 0 else None}


def _size(source):
    sp = physics.stellar_params(source) if source else None
    return sp if sp else (10.0, 1.0)            # unknown host: a giant's radius, so windows err wide


def catalogue_ephemeris(lcs, period, source=None):
    """Detected dip events folded at a catalogue period: how many fall in two windows, each as wide as the longest
    eclipse the star allows at that period (at least PHASE_WINDOW, at most MAX_WINDOW in phase)."""
    ev, _ = _events(lcs)
    if not len(ev) or not period:
        return None
    r, m = _size(source)
    width = min(MAX_WINDOW, max(PHASE_WINDOW, D.max_duration_days(period, r, m, E_CATALOGUE) / period))
    ph = (ev[:, 0] / period) % 1.0
    half = width / 2

    def inside(x, c):
        return np.abs((x - c + 0.5) % 1.0 - 0.5) <= half

    counts = np.array([inside(ph, c).sum() for c in ph])
    c1 = ph[int(np.argmax(counts))]
    rest = ~inside(ph, c1)
    n2, c2 = 0, None
    if rest.any():
        c2s = ph[rest]
        k2 = np.array([(inside(ph, c) & rest).sum() for c in c2s])
        n2, c2 = int(k2.max()), c2s[int(np.argmax(k2))]
    n1 = int(inside(ph, c1).sum())
    d1 = (ph[inside(ph, c1)] - c1 + 0.5) % 1.0 - 0.5
    return {"catalogue_period_d": float(f"{period:.6g}"), "n_events": int(len(ev)),
            "n_events_on_ephemeris": n1 + (n2 if n2 >= 2 else 0), "n_primary": n1,
            "n_secondary": n2 if n2 >= 2 else 0,
            "secondary_phase": round(float((c2 - c1) % 1.0), 3) if c2 is not None and n2 >= 2 else None,
            "primary_spread_phase": round(float(d1.max() - d1.min()), 3),
            "window_width_phase": round(width, 3), "chance_fraction": round(2 * width, 3), "visible_only": True,
            "description": (
                "The star is in a variable-star catalogue that is not built from Gaia (context known_variable_type, "
                "known_variable_period_d). Dip events (as in the recurrence test) are folded at the catalogue period: "
                "n_primary events fall in the best window of width window_width_phase (the longest eclipse a star "
                "of this size allows at that period, at least 0.05), n_secondary (if 2 or more) in a second one at "
                "secondary_phase. An eclipsing binary at that period puts all its dips in those two windows; dips "
                "at other phases are not its eclipses. chance_fraction: share of the orbit the two windows cover, so "
                "about that share of unrelated dips land in them by chance. primary_spread_phase: the phase range "
                "the primary window's events actually span; eclipses at the right period line up within about one "
                "eclipse duration. Catalogue periods can be aliases or out of date.")}


def _num(x, default=0.0):
    """A finite float, or default (circular orbits have no eccentricity or periastron angle in the catalogue)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def conjunctions(orbit, t_lo, t_hi):
    """BTJD times (sorted) in [t_lo, t_hi] when the two stars of a Gaia two-body orbit are lined up with us
    (true anomaly + omega = 90 or 270 deg), with the cycle count from periastron for each."""
    P, e = float(orbit["period"]), _num(orbit.get("eccentricity"))
    w = math.radians(_num(orbit.get("arg_periastron")))
    tp = float(orbit["t_periastron"]) + GAIA_REF_BTJD
    out = []
    for target in (math.pi / 2, 3 * math.pi / 2):
        nu = (target - w) % (2 * math.pi)
        E = 2 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2), math.sqrt(1 + e) * math.cos(nu / 2))
        M = (E - e * math.sin(E)) % (2 * math.pi)
        t0 = tp + M / (2 * math.pi) * P
        n0, n1 = math.floor((t_lo - t0) / P) - 1, math.ceil((t_hi - t0) / P) + 1
        out += [(t0 + n * P, n + (t0 - tp) / P) for n in range(n0, n1 + 1)]
    return sorted(out)


def orbit_conjunction(lcs, orbit, source=None):
    """Distance of each dip event from the nearest conjunction of a Gaia two-body orbit, or None."""
    if orbit is None or str(orbit.get("nss_solution_type")) not in ORBIT_TYPES or not orbit.get("period"):
        return None
    if not np.isfinite(_num(orbit.get("t_periastron"), np.nan)) or not _num(orbit.get("period")) > 0:
        return None
    ev, _ = _events(lcs)
    if not len(ev):
        return None
    P, e = float(orbit["period"]), _num(orbit.get("eccentricity"))
    r, m = _size(source)
    half = 0.5 * D.max_duration_days(P, r, m, e_max=min(max(e, 0.0), 0.95))
    sp_err, tp_err = _num(orbit.get("period_error")), _num(orbit.get("t_periastron_error"))
    order = np.argsort(-ev[:, 3])[:MAX_EVENTS]
    conj = conjunctions(orbit, float(ev[:, 0].min()), float(ev[:, 0].max()))
    tc = np.array([c[0] for c in conj])
    ncyc = np.array([c[1] for c in conj])
    offs, at = [], 0
    for i in order:
        j = int(np.argmin(np.abs(tc - ev[i, 0])))
        sig = math.hypot(tp_err, abs(ncyc[j]) * sp_err)
        off = float(ev[i, 0] - tc[j])
        offs.append((off, sig))
        at += abs(off) <= half + TIMING_SIGMA * sig
    off0, sig0 = offs[0]
    return {"orbit_period_d": float(f"{P:.6g}"), "eccentricity": round(e, 3),
            "n_events": int(len(order)), "n_events_at_conjunction": int(at),
            "deepest_event_offset_days": round(off0, 3), "timing_sigma_days": round(sig0, 3),
            "max_eclipse_half_duration_days": round(half, 3),
            "chance_fraction": round(min(1.0, 2 * (2 * half + 2 * TIMING_SIGMA * sig0) / P), 3), "visible_only": True,
            "description": (
                f"The star has a Gaia DR3 two-body orbit ({orbit.get('nss_solution_type')}) from radial velocities or "
                "astrometry, not from photometry. Eclipses can only happen when the two stars are lined up with us, "
                "twice per orbit. For the deepest dip events (n_events), n_events_at_conjunction fall within "
                "max_eclipse_half_duration_days (half the longest eclipse two stars of this size could make on this "
                "orbit) plus 2 timing_sigma_days (the orbit's period and periastron errors carried to that epoch) of "
                "a conjunction. deepest_event_offset_days: the deepest event's time minus its nearest conjunction. "
                "chance_fraction: share of the orbit those windows cover, so about that share of unrelated dips land "
                "in them by chance. Dips well away from every conjunction are not eclipses by this companion "
                "(another, unseen companion could still eclipse).")}
