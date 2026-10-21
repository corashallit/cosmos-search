"""Turn a Candidate into a blind Stage A packet and a key record.

The packet is what the expert sees: anonymised, re-zeroed in time, rendered as
text. The key record is what scoring needs and the expert must not see: the
candidate id and the true answers to the held-out prediction targets. In a
sealed run the key goes into the reveal map (encrypted); in a tuning run it is
written in plaintext next to the packets.

Held-out targets (docs/vetting.md §6) are the direct test for confabulation: an
explanation that cannot predict data it was not shown is a story, not a model.
"""

import json
import math
import re
import secrets

import numpy as np

from . import taxonomy
from .anonymise import leak_scan, scrub_source, sig
from .candidate import Candidate, LightCurve

MAX_ROWS = 400        # epochs per band shown in full; longer light curves are binned
N_BINS = 150
ZOOM_POINTS = 60      # per dip window
HOLDOUT_TAIL = 0.2    # fraction of the time span withheld by the "tail" holdout
DIP_SIGMA = 4.0       # a held-out "dip" is a point this many robust sigma below the visible median


class LeakError(ValueError):
    pass


# No digits and no hex letters, so an id can never look like a catalogue number ("cd0f...", "b243172978...").
ID_ALPHABET = "ghjkmnpqrstvwxyz"


def new_packet_id(n=16):
    return "".join(secrets.choice(ID_ALPHABET) for _ in range(n))


def robust_sigma(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def band_summary(lc):
    f = lc.flux[np.isfinite(lc.flux)]
    if len(f) == 0:
        return {"n_epochs": 0}
    med, s = float(np.median(f)), robust_sigma(f)
    z = (f - med) / s if s > 0 else np.zeros_like(f)
    return {
        "n_epochs": int(len(f)),
        "span_days": sig(lc.time.max() - lc.time.min(), 3),
        "median_flux": sig(med, 5),
        "median_flux_err": sig(np.nanmedian(lc.flux_err), 2),
        "robust_sigma": sig(s, 2),
        "min_flux": sig(f.min(), 4),
        "max_flux": sig(f.max(), 4),
        "n_below_3sigma": int((z < -3).sum()),
        "n_above_3sigma": int((z > 3).sum()),
    }


def _fmt(x, nd):
    if isinstance(x, (float, np.floating)):
        return "nan" if not math.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def _rows(lc, idx):
    cols = ["t", "flux", "err"] + sorted(lc.meta)
    lines = [" ".join(cols)]
    for i in idx:
        vals = [_fmt(lc.time[i], 4), _fmt(lc.flux[i], 5), _fmt(lc.flux_err[i], 5)]
        vals += [_fmt(lc.meta[k][i], 2) for k in sorted(lc.meta)]
        lines.append(" ".join(vals))
    return lines


def _binned(lc, n_bins):
    edges = np.linspace(lc.time.min(), lc.time.max(), n_bins + 1)
    which = np.clip(np.searchsorted(edges, lc.time, side="right") - 1, 0, n_bins - 1)
    lines = ["t_mid n median_flux min_flux"]
    for b in range(n_bins):
        m = which == b
        if m.any():
            f = lc.flux[m]
            lines.append(f"{(edges[b] + edges[b + 1]) / 2:.3f} {int(m.sum())} {np.nanmedian(f):.5f} {np.nanmin(f):.5f}")
    return lines


def _zoom(lc, dip):
    half = max(2.0 * float(dip.get("duration", 0) or 0), 0.5)
    t0 = float(dip["t_min"])
    idx = np.flatnonzero(np.abs(lc.time - t0) <= half)
    if len(idx) > ZOOM_POINTS:
        idx = idx[np.linspace(0, len(idx) - 1, ZOOM_POINTS).astype(int)]
    return idx


def _subset(lc, keep):
    return LightCurve(lc.band, lc.time[keep], lc.flux[keep], lc.flux_err[keep], {k: v[keep] for k, v in lc.meta.items()})


# --- Held-out targets ------------------------------------------------------


def _tail_targets(cand, t_cut):
    """Withhold everything after t_cut; ask whether further dimming appears there, and how deep."""
    targets, truths = [], {}
    ref = cand.lightcurves[0]
    vis, hid = ref.time < t_cut, ref.time >= t_cut
    if vis.sum() < 10 or hid.sum() < 3:
        return targets, truths, None
    med, s = float(np.nanmedian(ref.flux[vis])), robust_sigma(ref.flux[vis])
    thresh = med - DIP_SIGMA * s
    window = f"The last {HOLDOUT_TAIL:.0%} of the time span ({hid.sum()} {ref.band} epochs after t = {t_cut:.1f} d) is withheld."
    # Stars that vary by tens of per cent (R CrB, Miras) would need a dip below zero flux; ask only the interval there.
    if thresh > 0:
        targets.append({"id": "tail_dip", "kind": "binary",
                        "question": f"{window} Probability that at least one withheld {ref.band} epoch has flux below "
                                    f"{thresh:.4f} ({DIP_SIGMA:g} robust sigma below the visible median)?"})
        truths["tail_dip"] = bool(np.nanmin(ref.flux[hid]) < thresh)
    targets.append({"id": "tail_min_flux", "kind": "interval",
                    "question": f"{window} Central 80% interval for the minimum {ref.band} flux among the withheld epochs."})
    truths["tail_min_flux"] = float(np.nanmin(ref.flux[hid]))
    return targets, truths, t_cut


def _colour_targets(cand):
    """Withhold BP and RP; ask for the colour of the deepest dip seen in G (Gaia arm)."""
    g, bp, rp = cand.band("G"), cand.band("BP"), cand.band("RP")
    if g is None or bp is None or rp is None or len(g.time) < 10:
        return [], {}, ()
    i = int(np.nanargmin(g.flux))
    t_dip = g.time[i]
    jb, jr = int(np.argmin(np.abs(bp.time - t_dip))), int(np.argmin(np.abs(rp.time - t_dip)))
    if abs(bp.time[jb] - t_dip) > 0.01 or abs(rp.time[jr] - t_dip) > 0.01:
        return [], {}, ()
    depth_bp = 1 - bp.flux[jb] / np.nanmedian(bp.flux)
    depth_rp = 1 - rp.flux[jr] / np.nanmedian(rp.flux)
    if not depth_bp > 0:
        return [], {}, ()
    targets = [{"id": "dip_colour_ratio", "kind": "interval",
                "question": f"BP and RP photometry are withheld. Central 80% interval for R = (fractional depth in RP) / "
                            f"(fractional depth in BP) at the deepest G epoch (t = {t_dip:.3f} d). R = 1 is grey; "
                            f"interstellar-like dust gives R < 1."}]
    return targets, {"dip_colour_ratio": float(depth_rp / depth_bp)}, ("BP", "RP")


def _context_targets(cand):
    """Withhold the infrared excess; ask whether there is one (energy budget, docs/arms/gaia-dimmers.md)."""
    sigma = cand.context.get("ir_excess_sigma")
    if sigma is None or not math.isfinite(float(sigma)):
        return [], {}, ()
    targets = [{"id": "ir_excess", "kind": "binary",
                "question": "Infrared photometry is withheld. Probability that the source shows a mid-infrared (W3 or W4) "
                            "excess over its photosphere at more than 3 sigma?"}]
    return targets, {"ir_excess": bool(float(sigma) > 3)}, ("ir_excess_sigma",)


HOLDOUTS = ("tail", "colour", "ir")


# --- Building --------------------------------------------------------------


def build(candidate, holdout=(), packet_id=None):
    """Return (packet, key). `holdout` is a subset of HOLDOUTS; the targets that apply are added."""
    cand = candidate if isinstance(candidate, Candidate) else Candidate.from_dict(candidate)
    packet_id = packet_id or new_packet_id()
    t0, t1 = cand.time_span()

    targets, truths = [], {}
    hidden_bands, hidden_context, t_cut = (), (), None
    if "tail" in holdout:
        tg, tr, t_cut = _tail_targets(cand, t0 + (1 - HOLDOUT_TAIL) * (t1 - t0))
        targets += tg
        truths.update(tr)
    visible = cand if t_cut is None else Candidate(
        cand.candidate_id, cand.arm, cand.source, [_subset(lc, lc.time < t_cut) for lc in cand.lightcurves],
        cand.dips, cand.scores, cand.tests, cand.context)
    if "colour" in holdout:
        tg, tr, hidden_bands = _colour_targets(visible)
        targets += tg
        truths.update(tr)
    if "ir" in holdout:
        tg, tr, hidden_context = _context_targets(visible)
        targets += tg
        truths.update(tr)

    lcs = []
    for lc in visible.lightcurves:
        if lc.band not in hidden_bands:
            lc = _subset(lc, np.ones(len(lc.time), bool))
            lc.time = lc.time - t0
            lcs.append(lc)

    dips = [{k: (sig(v - t0, 6) if k in ("t_min", "t_start", "t_end") else sig(v, 3) if isinstance(v, (int, float)) else v)
             for k, v in d.items() if isinstance(v, (int, float, str)) and k not in ("id", "source_id")}
            for d in cand.dips if t_cut is None or float(d.get("t_min", t0)) < t_cut]

    packet = {
        "packet_id": packet_id,
        "arm": cand.arm,
        "source": scrub_source(cand.source),
        "bands": [lc.band for lc in lcs],
        "summaries": {lc.band: band_summary(lc) for lc in lcs},
        "dips": dips,
        # Detector outputs were computed on the full light curve, so a tail holdout would leak through them.
        "tests": {} if t_cut is not None else
                 {k: {kk: (sig(vv, 3) if isinstance(vv, (int, float)) else vv) for kk, vv in v.items()
                      if isinstance(vv, (int, float)) or (kk == "description" and isinstance(vv, str))}
                  for k, v in cand.tests.items()},
        "scores": {} if t_cut is not None else
                  {k: sig(v, 3) for k, v in cand.scores.items() if isinstance(v, (int, float))},
        "context": {k: (sig(v, 3) if isinstance(v, (int, float)) else v) for k, v in cand.context.items()
                    if k not in hidden_context and isinstance(v, (int, float, str))},
        "targets": [_rezero_question(t, t0) for t in targets],
        "withheld": {"bands": list(hidden_bands), "context": list(hidden_context),
                     "after_t": None if t_cut is None else sig(t_cut - t0, 5)},
    }
    packet["text"] = render(packet, lcs)
    # Random hex ids can contain long digit runs; they are ours, so take them out before scanning.
    leaks = leak_scan((json.dumps({k: v for k, v in packet.items() if k != "text"}) + packet["text"]).replace(packet_id, ""))
    if leaks:
        raise LeakError(f"packet {packet_id}: identifier-like content {leaks}")
    key = {"packet_id": packet_id, "candidate_id": cand.candidate_id, "arm": cand.arm, "truths": truths}
    return packet, key


def _rezero_question(target, t0):
    """Target questions are written with candidate-clock times; rewrite 't = X d' on the packet clock."""
    q = re.sub(r"t = (-?\d+\.?\d*) d", lambda m: f"t = {float(m.group(1)) - t0:.3f} d", target["question"])
    return {**target, "question": q}


def render(packet, lcs):
    """The packet as the text the expert reads."""
    out = [f"# Packet {packet['packet_id']}", "", f"Arm: {packet['arm']}. Times are days since the first epoch.", ""]
    out += ["## Source", json.dumps(packet["source"], sort_keys=True), ""]
    if packet["context"]:
        out += ["## Context", json.dumps(packet["context"], sort_keys=True), ""]
    if packet["tests"] or packet["scores"]:
        out += ["## Detector scores and false-positive tests",
                "Scores are calibrated false-alarm probabilities unless named otherwise. Tests give a statistic, a "
                "description of it where the pipeline supplies one, and, where available, the p-value of the null "
                "'this false-positive class does not explain the dip'.",
                json.dumps({"scores": packet["scores"], "tests": packet["tests"]}, sort_keys=True), ""]
    if packet["dips"]:
        out += ["## Detected dips", *(json.dumps(d, sort_keys=True) for d in packet["dips"]), ""]
    out += ["## Light curves"]
    for lc in lcs:
        out += [f"### {lc.band}", json.dumps(packet["summaries"][lc.band], sort_keys=True)]
        if len(lc.time) <= MAX_ROWS:
            out += _rows(lc, range(len(lc.time)))
        else:
            out += [f"Binned to {N_BINS} bins:"] + _binned(lc, N_BINS)
            for j, d in enumerate(packet["dips"]):
                if "t_min" in d:
                    out += [f"Zoom on dip {j}:"] + _rows(lc, _zoom(lc, d))
        out.append("")
    w = packet["withheld"]
    if w["bands"] or w["context"] or w["after_t"] is not None:
        out += ["## Withheld data", json.dumps(w, sort_keys=True), ""]
    if packet["targets"]:
        out += ["## Prediction targets", "Answer each from your explanation, before and independent of the class probabilities."]
        out += [f"- {t['id']} ({t['kind']}): {t['question']}" for t in packet["targets"]]
        out.append("")
    out += ["## Classes", taxonomy.describe()]
    return "\n".join(out)
