"""Vetting batches: real candidates, known-answer controls and injected decoys, made indistinguishable.

Runs in the truth repo's Actions job like every generator here. Config:

    class: vetting_batch
    real: candidates.jsonl          # Candidate dicts to vet (may be absent for a calibration batch)
    controls: controls.jsonl        # Candidate dicts plus "label" and optional "aliases", "acceptable"
    hosts: hosts.jsonl              # ordinary stars from the same sample, for decoys
    holdout: [tail, colour]         # applied identically to every packet
    decoys: {grey_primes: 20, chromatic_flip: 10, ...}   # counts per DECOY_SPECS name

Products are the packets. Truth (Cora's key) holds host ids and injection
parameters. Reveal (reveal key) holds the per-packet scoring key, in which a
decoy's candidate_id is just "decoy".
"""

import json
from pathlib import Path

import numpy as np

from cosmos_search.inject import lightcurve as lcinj
from cosmos_search.vetting.batch import build_batch, simple_dips
from cosmos_search.vetting.candidate import Candidate, LightCurve
from cosmos_search.vetting.packet import HOLDOUT_TAIL, robust_sigma

# name: (family, role, label, acceptable, band mode)
# Roles: anomaly_probe = coherent but no natural counterpart (scores sensitivity to real anomalies);
# incoherent = no single physical screen does this (scores confabulation); rare_natural = physically
# consistent natural scenario (scores over-calling "unexplained"); ambiguous = not scored for correctness.
DECOY_SPECS = {
    "grey_primes": ("dip_train_primes", "anomaly_probe", "unexplained", ["unexplained"], "all"),
    "grey_geometric": ("dip_train_geometric", "anomaly_probe", "unexplained", ["unexplained"], "all"),
    "chromatic_flip": ("chromatic_flip", "incoherent", "unexplained", ["unexplained", "instrumental"], "all"),
    "single_band_dip": ("grey_dip", "incoherent", "other_instrumental", ["instrumental", "unexplained"], "one"),
    "step": ("step", "incoherent", "calibration_offset", ["instrumental", "unexplained"], "all"),
    "dust_dip": ("dust_dip", "rare_natural", "yso_dipper", ["astrophysical"], "all"),
    "periodic_grey": ("dip_train_periodic", "rare_natural", "eclipsing_binary", ["astrophysical"], "all"),
    "grey_dip": ("grey_dip", "ambiguous", "unexplained", ["astrophysical", "unexplained"], "all"),
    "secular_fade": ("secular_fade", "ambiguous", "unexplained", ["astrophysical", "unexplained", "instrumental"], "all"),
}


def _read_jsonl(path):
    if not path:
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _photon_scaled(flux, err, k, rng):
    """A real light curve multiplied by k as a real change in the source's brightness would be.

    Multiplying a noisy flux scales its noise realisation by k, but photon noise
    scales as sqrt(k). The difference is added back as fresh noise, and the quoted
    errors scale as sqrt(k). Errors falling in proportion to the flux inside a dip
    gave decoys away in the TESS pilot. Brightening (k > 1) cannot remove noise, so
    it is left at k.
    """
    k = np.asarray(k, float)
    f = np.asarray(flux, float) * k
    if rng is not None:
        f = f + rng.normal(0.0, 1.0, len(f)) * np.asarray(err, float) * np.sqrt(np.clip(k - k ** 2, 0, None))
    return f, np.asarray(err, float) * np.sqrt(k)


def _band(lc):
    return lc.band if lc.band in lcinj.BAND_WAVELENGTH_UM else "G"


def inject_candidate(host, signal, bands=None, rng=None):
    """A copy of host with signal applied to the named bands (all bands if None).

    Every flux channel is injected: the main flux and any flux* arrays in meta
    (alternate apertures), with their *err* companions scaled to match. A dip
    present in one aperture and missing from another is a giveaway.
    """
    lcs = []
    for lc in host.lightcurves:
        meta = {k: v.copy() for k, v in lc.meta.items()}
        if bands is None or lc.band in bands:
            k = signal.factor(lc.time, _band(lc))
            f, e = _photon_scaled(lc.flux, lc.flux_err, k, rng)
            for name, v in lc.meta.items():
                if not name.startswith("flux") or np.shape(v) != np.shape(lc.time) or "err" in name:
                    continue
                err = lc.meta.get(f"{name}_err", lc.flux_err)
                meta[name], e2 = _photon_scaled(v, err, k, rng)
                if f"{name}_err" in lc.meta:
                    meta[f"{name}_err"] = e2
        else:
            f, e = lc.flux.copy(), lc.flux_err.copy()
        lcs.append(LightCurve(lc.band, lc.time.copy(), f, e, meta))
    return Candidate(host.candidate_id, host.arm, dict(host.source), lcs, [], dict(host.scores),
                     dict(host.tests), dict(host.context))


def _deviations(host, signal, bands, t_cut):
    """Per-band (times, (k - 1) / scatter): the signal in units of the host's own robust scatter, over the shown epochs."""
    out = {}
    for lc in host.lightcurves:
        if bands is not None and lc.band not in bands:
            continue
        keep = lc.time < t_cut
        k = signal.factor(lc.time[keep], _band(lc))
        scale = robust_sigma(lc.flux[keep]) or float(np.nanmedian(lc.flux_err[keep]))
        out[lc.band] = (lc.time[keep], (k - 1) / scale)
    return out


def _events(t, dev, thresh=3.0, gap=1.0):
    """Times of distinct visible events: epochs above thresh, grouped when closer than gap days."""
    hit = np.sort(t[dev >= thresh])
    if len(hit) == 0:
        return hit
    starts = np.concatenate([[0], np.flatnonzero(np.diff(hit) > gap) + 1])
    return hit[starts]


def _window_fits(a, b, out, width):
    """Can an eclipse window of duration `width` contain phases [a, b] and no out-of-dip phase?"""
    if b - a > width or np.any((out >= a) & (out <= b)):
        return False
    left = out[out < a].max(initial=-np.inf)
    right = out[out > b].min(initial=np.inf)
    return max(b - width, left) < min(a, right - width)


def _free_window_fits(a, b, out, max_width):
    """A window of any duration up to max_width: the shortest one, [a, b] itself, is the easiest to fit."""
    return b - a <= max_width and not np.any((out >= a) & (out <= b))


def single_period_fits(t_in, t_out, width, span):
    """Could one periodic eclipsing system explain which epochs are in and out of a dip?

    Brute force over trial periods. The in-dip epochs must fall in at most two
    phase windows: one eclipse of the signal's duration, placed anywhere that
    avoids the out-of-dip epochs, and a second eclipse at any phase whose
    duration is free up to 2 * width (an eccentric binary's secondary). This is
    the observer's test: pilot-2 vetters tried eccentric primary-plus-secondary
    solutions, so a decoy that one of those fits is not an anomaly probe at this
    cadence.
    """
    t_in, t_out = np.asarray(t_in, float), np.asarray(t_out, float)
    if len(t_in) < 2:
        return True
    ref = t_in[0]
    p, p_max = max(2 * width, 0.05), span
    step = 1 + 0.25 * width / span
    while p <= p_max:
        ph_in = ((t_in - ref) / p + 0.5) % 1 * p - 0.5 * p  # signed, the first in-dip epoch at 0
        ph_out = ((t_out - ref) / p + 0.5) % 1 * p - 0.5 * p
        near = np.abs(ph_in) <= width
        a, b = ph_in[near].min(), ph_in[near].max()
        rest = ph_in[~near]
        if len(rest) == 0:
            if _window_fits(a, b, ph_out, width):
                return True
        else:
            # Re-centre the second window's phases on its own epochs so it cannot straddle the wrap.
            c = rest[0]
            r = ((rest - c) / p + 0.5) % 1 * p - 0.5 * p
            o = ((ph_out - c) / p + 0.5) % 1 * p - 0.5 * p
            fixed_first = _window_fits(a, b, ph_out, width) and _free_window_fits(r.min(), r.max(), o, 2 * width)
            fixed_second = _free_window_fits(a, b, ph_out, 2 * width) and _window_fits(r.min(), r.max(), o, width)
            if fixed_first or fixed_second:
                return True
        p *= step
    return False


def _width(signal):
    parts = signal.parts if isinstance(signal, lcinj.Composite) else [signal]
    dips = [d for d in parts if isinstance(d, lcinj.Dip)]
    return max((d.ingress + d.flat + d.egress for d in dips), default=1.0)


def _flip_visible(devs):
    """A chromatic flip is incoherent only if the packet shows both halves: BP down and RP up at the same epochs."""
    if "BP" not in devs or "RP" not in devs:
        return False
    (t_b, d_b), (t_r, d_r) = devs["BP"], devs["RP"]
    up = t_r[d_r >= 3]
    if len(up) == 0:
        return False
    # Same transit: BP and RP epochs carry slightly different times, so match within ~15 min.
    near = np.abs(t_b[:, None] - up[None, :]).min(axis=1) < 0.01
    return bool(np.any(near & (d_b <= -3)))


# Specs whose label rests on more than one band: the feature that makes them what they are must be visible too.
IDENTIFIABLE = {"chromatic_flip": _flip_visible}


def visibility(host, signal, bands, t_cut, role, spec=None):
    """Whether a packet showing epochs before t_cut would show this signal, and as what.

    Visible: at least two epochs >= 3 and one >= 4 times the host's own robust
    scatter (the star's variability is noise here, not only the photometric
    errors). Specs in IDENTIFIABLE must also show their defining feature. Anomaly
    probes must be identifiable as anomalies from the sampled data: >= 4
    distinct events that no periodic eclipsing system fits (pilot 2: with 3
    events vetters found eclipse solutions, often with an unseen secondary).
    """
    devs = _deviations(host, signal, bands, t_cut)
    if not devs:
        return False, {}
    ref_band = max(devs, key=lambda b: np.abs(devs[b][1]).max())
    t, dev = devs[ref_band][0], np.abs(devs[ref_band][1])
    info = {"max_dev": float(dev.max()), "n_dev3": int((dev >= 3).sum())}
    if not (dev.max() >= 4 and (dev >= 3).sum() >= 2):
        return False, info
    if spec in IDENTIFIABLE and not IDENTIFIABLE[spec](devs):
        info["identifiable"] = False
        return False, info
    if role == "anomaly_probe":
        ev = _events(t, dev)
        info["n_events"] = len(ev)
        if len(ev) < 4:
            return False, info
        t_in, t_out = t[dev >= 3], t[dev < 0.5]
        if single_period_fits(t_in, t_out, _width(signal), float(t.max() - t.min())):
            return False, info
    return True, info


def _detected(cand, signal, t_cut, detect):
    """The candidate detector fires on the planted signal before t_cut, as it must for a real candidate."""
    dips = detect(cand)
    dips = cand.dips if dips is None else dips  # batch.harmonise hooks set cand.dips; simple_dips returns them
    g = cand.lightcurves[0]
    for d in dips:
        if d["t_min"] < t_cut and signal.factor(np.array([d["t_min"]]), _band(g))[0] < 1 - 1e-3:
            return True
    return False


def make_decoy(host, spec, rng, t_cut=np.inf, max_tries=500, detect=simple_dips):
    """Draw a decoy that the packet will actually show, and that is what its label says at this cadence.

    Sparse cadences (Gaia: ~40-150 epochs over years) miss most short events, and
    a decoy whose signal falls between epochs, inside the star's own variability,
    or in a withheld tail is a no-signal packet with the wrong label. The first
    pilot (docs/vetting.md, Log) had several. So signals are redrawn until
    visibility() accepts them on the epochs before t_cut and the candidate
    detector fires on them; the accepted distribution is "family conditional on
    visible and detected", which is also what real candidates are.

    Signals are drawn over the whole span, tail included, so a decoy's withheld
    tail behaves as a real candidate's would: a repeating signal continues into
    it. (Pilot 2 drew them before the cut only, so no decoy ever dipped in its
    tail.) Single-band decoys go into the detector's band, since a dip the
    detector cannot see never reaches vetting. detect=None skips the detector
    requirement (for families a dip detector is not meant to find, such as steps
    and slow fades).
    """
    family, role, label, acceptable, mode = DECOY_SPECS[spec]
    t0, t1 = host.time_span()
    for _ in range(max_tries):
        signal = lcinj.sample(rng, family, t0, t1)
        bands = [host.lightcurves[0].band] if mode == "one" else None
        ok, info = visibility(host, signal, bands, t_cut, role, spec)
        if ok:
            cand = inject_candidate(host, signal, bands, rng)
            if detect is None or _detected(cand, signal, t_cut, detect):
                break
    else:
        raise RuntimeError(f"no visible {spec} decoy in {max_tries} draws on {host.candidate_id}")
    extra = {"kind": "decoy", "candidate_id": "decoy", "role": role, "label": label, "acceptable": acceptable,
             "spec": spec}
    truth = {"spec": spec, "host": host.candidate_id, "bands": bands, "visibility": info,
             "signal": lcinj.truth(signal)}
    return cand, extra, truth


def vetting_batch(cfg, rng, detect=None):
    """Packets, truth rows and scoring keys. detect is the batch's candidate detector (simple_dips if None),
    run on every packet and required to fire on every decoy."""
    detect = detect or simple_dips
    entries, truths = [], []
    for d in _read_jsonl(cfg.get("real")):
        entries.append((Candidate.from_dict(d), {"kind": "real"}))
        truths.append(None)
    for d in _read_jsonl(cfg.get("controls")):
        extra = {"kind": "control", "label": d.pop("label"), "aliases": d.pop("aliases", []),
                 "acceptable": d.pop("acceptable", None)}
        entries.append((Candidate.from_dict(d), extra))
        truths.append(None)
    hosts = [Candidate.from_dict(d) for d in _read_jsonl(cfg.get("hosts"))]
    holdout = tuple(cfg.get("holdout", ()))
    # Identifiable anomaly probes are rare at sparse cadence (~1 in 600 draws on DR3 hosts), and draws are cheap.
    max_tries = int(cfg.get("max_tries", 3000))
    # Spread decoys evenly over hosts (a shuffled round robin), so no host carries several decoys while others idle.
    order = []
    for spec, n in (cfg.get("decoys") or {}).items():
        if spec not in DECOY_SPECS:
            raise SystemExit(f"unknown decoy spec {spec!r}")
        order += [spec] * int(n)
    host_cycle = np.concatenate([rng.permutation(len(hosts)) for _ in range(len(order) // max(len(hosts), 1) + 1)])
    for spec, h in zip(order, host_cycle):
        # A host whose cadence or variability cannot show this decoy hands it to the next host.
        for j in range(len(hosts)):
            host = hosts[(int(h) + j) % len(hosts)]
            t0, t1 = host.time_span()
            t_cut = t0 + (1 - HOLDOUT_TAIL) * (t1 - t0) if "tail" in holdout else np.inf
            try:
                cand, extra, truth = make_decoy(host, spec, rng, t_cut=t_cut, max_tries=max_tries, detect=detect)
                break
            except RuntimeError:
                continue
        else:
            raise SystemExit(f"no host can show a visible {spec} decoy")
        entries.append((cand, extra))
        truths.append(truth)

    # build_batch shuffles; carry the truth through by tagging each entry's extra with its index.
    for i, (_, extra) in enumerate(entries):
        extra["_i"] = i
    packets, keys = build_batch(entries, rng, holdout=holdout, detect=None if detect is simple_dips else detect)
    truth_rows = []
    for k in keys:
        i = k.pop("_i")
        if truths[i] is not None:
            truth_rows.append({"packet_id": k["packet_id"], **truths[i]})
        if k["kind"] == "decoy":
            k["candidate_id"] = "decoy"
    return packets, truth_rows, keys
