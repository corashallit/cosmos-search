"""Injection test of the Stage 0 band-consistency check on real Gaia DR3 light curves.

    python scripts/stage0_recovery.py PILOT_DIR/controls.jsonl PILOT_DIR/hosts.jsonl [--sources S.json] [--trials 40]

Each star is cleaned (stage0.clean) and then given single-transit signals at random G epochs: a G-only dip (the
fault Stage 0 should catch) and changes the source's light can make, which it must pass: grey, dust-reddened,
chromatic flip, and "diluted", a grey dip that BP and RP see at a fraction k ~ U(0.1, 1) of its depth, as a
neighbour inside the BP/RP window makes them, on the same star marked as blended in every transit. Depths are in
units of the star's G scatter. --sources (JSON, source_id -> gaia_source columns) supplies the blended-transit
counts, without which quiet stars are left untested. Prints the band_inconsistent rate per kind, depth and
magnitude bin: recall for g_only, false-veto rate for the rest.
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from cosmos_search.inject import lightcurve as lcinj
from cosmos_search.inject.classes.vetting_decoys import inject_candidate
from cosmos_search.vetting.candidate import Candidate
from cosmos_search.vetting.packet import robust_sigma
from cosmos_search.vetting.stage0 import band_consistency, clean

Z = (3, 5, 10, 20)
MAG_BINS = (14, 16, 18)
HALF = 0.02  # days: one transit's G, BP and RP, not the other field of view's (~0.07 d later)


def signal(kind, t, depth):
    """A single-transit change with G deficit `depth`."""
    win = dict(t0=t - HALF, ingress=0.0, egress=0.0, flat=2 * HALF)
    if kind in ("g_only", "grey"):
        return lcinj.Dip(covering=depth, **win)
    if kind == "dust":
        return lcinj.Dip(covering=1.0, tau_ref=-np.log(1 - depth), beta=2.0, **win)
    if kind == "flip":  # BP down, RP up, G net down by about `depth`
        tau = 1.0
        cover = depth / ((1 - np.exp(-tau)) - 0.5)
        return lcinj.Composite([lcinj.Dip(covering=min(cover, 0.95), tau_ref=tau, beta=4.0, **win),
                                lcinj.Flare(t0=t, amplitude=0.5 * min(cover, 0.95), rise=0.1, decay=0.1, gamma=-4.0)])
    if kind == "diluted":
        return lcinj.Dip(covering=depth, **win)
    raise ValueError(kind)


KINDS = ("g_only", "grey", "dust", "flip", "diluted")


def inject(cand, kind, t, depth, rng):
    if kind == "g_only":
        return inject_candidate(cand, signal(kind, t, depth), ["G"], rng)
    if kind != "diluted":
        return inject_candidate(cand, signal(kind, t, depth), None, rng)
    k = rng.uniform(0.1, 1.0)
    out = inject_candidate(cand, signal(kind, t, depth), ["G"], rng)
    out = inject_candidate(out, signal(kind, t, k * depth), ["BP", "RP"], rng)
    for b in ("bp", "rp"):  # blended in every transit
        n = out.source.get(f"phot_{b}_n_obs") or len(out.band(b.upper()).time)
        out.source.update({f"phot_{b}_n_obs": n, f"phot_{b}_n_blended_transits": n})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", help="JSON: source_id -> gaia_source columns (blended-transit counts)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    extra = json.load(open(a.sources)) if a.sources else {}
    hits = defaultdict(lambda: [0, 0])
    for path in a.files:
        for line in open(path):
            d = json.loads(line)
            for k in ("label", "aliases", "acceptable", "dpac_class"):
                d.pop(k, None)
            d["source"].update(extra.get(d["candidate_id"].split(":")[-1], {}))
            cand, _ = clean(Candidate.from_dict(d))
            g = cand.lightcurves[0]
            sigma = robust_sigma(g.flux) / np.median(g.flux)
            mag = cand.source.get("phot_g_mean_mag", np.nan)
            mbin = int(np.searchsorted(MAG_BINS, mag))
            for _ in range(a.trials):
                t = float(rng.choice(g.time))
                for z in Z:
                    depth = min(z * sigma, 0.6)
                    for kind in KINDS:
                        inj = inject(cand, kind, t, depth, rng)
                        (row,) = band_consistency(inj, [{"t_min": t}])
                        for key in ((kind, z, mbin), (kind, z, None)):
                            hits[key][0] += row["verdict"] == "band_inconsistent"
                            hits[key][1] += 1
    labels = [f"G<{MAG_BINS[0]}"] + [f"{lo}-{hi}" for lo, hi in zip(MAG_BINS, MAG_BINS[1:])] + [f"G>{MAG_BINS[-1]}"]
    print("band_inconsistent rate (n trials); g_only = recall, others = false veto")
    print(f"{'kind':8s} {'z':>3s} {'all':>12s} " + " ".join(f"{s:>12s}" for s in labels))
    for kind in KINDS:
        for z in Z:
            cells = []
            for mbin in [None] + list(range(len(labels))):
                h, n = hits.get((kind, z, mbin), (0, 0))
                cells.append(f"{h / n:6.3f} ({n:4d})" if n else f"{'-':>12s}")
            print(f"{kind:8s} {z:3d} " + " ".join(cells))


if __name__ == "__main__":
    main()
