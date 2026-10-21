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
from cosmos_search.vetting.batch import build_batch
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


def inject_candidate(host, signal, bands=None):
    """A copy of host with signal multiplied into the named bands (all bands if None)."""
    lcs = []
    for lc in host.lightcurves:
        if bands is None or lc.band in bands:
            band = lc.band if lc.band in lcinj.BAND_WAVELENGTH_UM else "G"
            f, e = lcinj.inject(lc.time, lc.flux, lc.flux_err, band, signal)
        else:
            f, e = lc.flux.copy(), lc.flux_err.copy()
        lcs.append(LightCurve(lc.band, lc.time.copy(), f, e, {k: v.copy() for k, v in lc.meta.items()}))
    return Candidate(host.candidate_id, host.arm, dict(host.source), lcs, [], dict(host.scores),
                     dict(host.tests), dict(host.context))


def _deviations(host, signal, bands, t_cut):
    """Per-band arrays of |1 - k| in units of the host's own robust scatter, over the epochs a packet will show."""
    out = {}
    for lc in host.lightcurves:
        if bands is not None and lc.band not in bands:
            continue
        keep = lc.time < t_cut
        band = lc.band if lc.band in lcinj.BAND_WAVELENGTH_UM else "G"
        k = signal.factor(lc.time[keep], band)
        scale = robust_sigma(lc.flux[keep]) or float(np.nanmedian(lc.flux_err[keep]))
        out[lc.band] = (lc.time[keep], np.abs(1 - k) / scale)
    return out


def _events(t, dev, thresh=3.0, gap=1.0):
    """Times of distinct visible events: epochs above thresh, grouped when closer than gap days."""
    hit = np.sort(t[dev >= thresh])
    if len(hit) == 0:
        return hit
    starts = np.concatenate([[0], np.flatnonzero(np.diff(hit) > gap) + 1])
    return hit[starts]


def single_period_fits(t_in, t_out, width, span):
    """Could one strictly periodic event of the given width explain which epochs are in and out of a dip?

    Brute force over trial periods, phased on the first in-dip epoch. This is the
    observer's test: if some period fits, the sampled data cannot tell a non-orbital
    spacing from an orbit, so the decoy is not an anomaly probe at this cadence.
    """
    t_in, t_out = np.asarray(t_in, float), np.asarray(t_out, float)
    if len(t_in) < 2:
        return True
    ref = t_in[0]
    p, p_max = max(2 * width, 0.05), span
    step = 1 + 0.25 * width / span
    while p <= p_max:
        ph_in = np.abs(((t_in - ref) / p + 0.5) % 1 - 0.5) * p
        if np.all(ph_in <= width):
            ph_out = np.abs(((t_out - ref) / p + 0.5) % 1 - 0.5) * p
            if not np.any(ph_out <= 0.5 * width):
                return True
        p *= step
    return False


def _width(signal):
    parts = signal.parts if isinstance(signal, lcinj.Composite) else [signal]
    dips = [d for d in parts if isinstance(d, lcinj.Dip)]
    return max((d.ingress + d.flat + d.egress for d in dips), default=1.0)


def visibility(host, signal, bands, t_cut, role):
    """Whether a packet showing epochs before t_cut would show this signal, and as what.

    Visible: at least two epochs >= 3 and one >= 4 times the host's own robust
    scatter (the star's variability is noise here, not only the photometric
    errors). Anomaly probes must also be identifiable as anomalies from the
    sampled data: >= 3 distinct events that no single period fits.
    """
    devs = _deviations(host, signal, bands, t_cut)
    if not devs:
        return False, {}
    ref_band = max(devs, key=lambda b: devs[b][1].max())
    t, dev = devs[ref_band]
    info = {"max_dev": float(dev.max()), "n_dev3": int((dev >= 3).sum())}
    if not (dev.max() >= 4 and (dev >= 3).sum() >= 2):
        return False, info
    if role == "anomaly_probe":
        ev = _events(t, dev)
        info["n_events"] = len(ev)
        if len(ev) < 3:
            return False, info
        t_in, t_out = t[dev >= 3], t[dev < 0.5]
        if single_period_fits(t_in, t_out, _width(signal), float(t.max() - t.min())):
            return False, info
    return True, info


def make_decoy(host, spec, rng, t_cut=np.inf, max_tries=500):
    """Draw a decoy that the packet will actually show, and that is what its label says at this cadence.

    Sparse cadences (Gaia: ~40-150 epochs over years) miss most short events, and
    a decoy whose signal falls between epochs, inside the star's own variability,
    or in a withheld tail is a no-signal packet with the wrong label. The first
    pilot (docs/vetting.md, Log) had several. So signals are redrawn until
    visibility() accepts them on the epochs before t_cut; the accepted
    distribution is "family conditional on visible", which is also what real
    candidates are.
    """
    family, role, label, acceptable, mode = DECOY_SPECS[spec]
    t0, t1 = host.time_span()
    t_hi = min(t1, t_cut)
    for _ in range(max_tries):
        signal = lcinj.sample(rng, family, t0, t_hi)
        bands = None
        if mode == "one":
            bands = [host.lightcurves[int(rng.integers(len(host.lightcurves)))].band]
        ok, info = visibility(host, signal, bands, t_cut, role)
        if ok:
            break
    else:
        raise RuntimeError(f"no visible {spec} decoy in {max_tries} draws on {host.candidate_id}")
    cand = inject_candidate(host, signal, bands)
    extra = {"kind": "decoy", "candidate_id": "decoy", "role": role, "label": label, "acceptable": acceptable,
             "spec": spec}
    truth = {"spec": spec, "host": host.candidate_id, "bands": bands, "visibility": info,
             "signal": lcinj.truth(signal)}
    return cand, extra, truth


def vetting_batch(cfg, rng):
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
                cand, extra, truth = make_decoy(host, spec, rng, t_cut=t_cut)
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
    packets, keys = build_batch(entries, rng, holdout=holdout)
    truth_rows = []
    for k in keys:
        i = k.pop("_i")
        if truths[i] is not None:
            truth_rows.append({"packet_id": k["packet_id"], **truths[i]})
        if k["kind"] == "decoy":
            k["candidate_id"] = "decoy"
    return packets, truth_rows, keys
