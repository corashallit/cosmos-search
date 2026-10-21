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


def injected_snr(host, signal, bands=None):
    """Matched-filter S/N of the signal as sampled: sqrt(sum ((1 - k) / err)^2) over the injected epochs."""
    total = 0.0
    for lc in host.lightcurves:
        if bands is None or lc.band in bands:
            band = lc.band if lc.band in lcinj.BAND_WAVELENGTH_UM else "G"
            k = signal.factor(lc.time, band)
            total += float(np.sum(((1 - k) * lc.flux / lc.flux_err) ** 2))
    return total ** 0.5


def make_decoy(host, spec, rng, min_snr=7.0, max_tries=200):
    """Draw a decoy that the sampling actually shows.

    Sparse cadences (Gaia: ~100 epochs over years) miss most short events, and a
    decoy whose signal falls between epochs is a no-signal packet with the wrong
    label. So signals are redrawn until their sampled S/N reaches min_snr; the
    accepted distribution is therefore "family conditional on visible", which is
    also what real candidates are.
    """
    family, role, label, acceptable, mode = DECOY_SPECS[spec]
    t0, t1 = host.time_span()
    for _ in range(max_tries):
        signal = lcinj.sample(rng, family, t0, t1)
        bands = None
        if mode == "one":
            bands = [host.lightcurves[int(rng.integers(len(host.lightcurves)))].band]
        snr = injected_snr(host, signal, bands)
        if snr >= min_snr:
            break
    else:
        raise RuntimeError(f"no visible {spec} decoy in {max_tries} draws on {host.candidate_id}")
    cand = inject_candidate(host, signal, bands)
    extra = {"kind": "decoy", "candidate_id": "decoy", "role": role, "label": label, "acceptable": acceptable,
             "spec": spec}
    truth = {"spec": spec, "host": host.candidate_id, "bands": bands, "snr": snr, "signal": lcinj.truth(signal)}
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
    for spec, n in (cfg.get("decoys") or {}).items():
        if spec not in DECOY_SPECS:
            raise SystemExit(f"unknown decoy spec {spec!r}")
        for _ in range(int(n)):
            cand, extra, truth = make_decoy(hosts[int(rng.integers(len(hosts)))], spec, rng)
            entries.append((cand, extra))
            truths.append(truth)

    # build_batch shuffles; carry the truth through by tagging each entry's extra with its index.
    for i, (_, extra) in enumerate(entries):
        extra["_i"] = i
    packets, keys = build_batch(entries, rng, holdout=tuple(cfg.get("holdout", ())))
    truth_rows = []
    for k in keys:
        i = k.pop("_i")
        if truths[i] is not None:
            truth_rows.append({"packet_id": k["packet_id"], **truths[i]})
        if k["kind"] == "decoy":
            k["candidate_id"] = "decoy"
    return packets, truth_rows, keys
