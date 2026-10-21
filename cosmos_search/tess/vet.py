"""Feed bake-off candidates into the shared vetting package (cosmos_search.vetting) for Stage A.

    python -m cosmos_search.tess.vet build --out data/bakeoff/v0 --n-real 24 --controls cosmos_search/tess/controls.csv
    python -m cosmos_search.vetting prompts --packets data/bakeoff/v0/vet/products.json --out data/bakeoff/v0/vet/prompts
    ... run the no-tool experts, then `vetting commit` and `vetting score` with vet/key.json

Every packet, real, control or decoy, goes through the same detector hook (`Detector`), so no field tells a
decoy apart (vetting/batch.py). Decoys use the vetting thread's DECOY_SPECS that make sense for one band.
This is a visible tuning batch: the key is written in plaintext next to the packets. The experts never
see it because they have no tools.
"""

import argparse
import json
import secrets
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.dimming.detect import find_events, scan
from cosmos_search.dimming.features import event_diagnostics
from cosmos_search.dimming.lightcurve import LightCurve as DimLC
from cosmos_search.inject.classes.vetting_decoys import make_decoy
from cosmos_search.vetting.batch import build_batch
from cosmos_search.vetting.candidate import Candidate, LightCurve

from . import qlp
from .analyse import SHORT_SCALE, calibrated_p, tmag_stratum

# Candidate meta channel <-> dimming aux name.
CHANNELS = {"background": "bkg", "centroid_x": "x", "centroid_y": "y", "flux_small_aperture": "flux_sml",
            "flux_large_aperture": "flux_lag"}
# One-band decoys only: chromatic and single-band specs need several bands.
TESS_DECOYS = ("grey_primes", "grey_geometric", "step", "dust_dip", "periodic_grey", "grey_dip")
TEST_DEFINITIONS = (
    "Tests describe the strongest dip. placement_z: dip mean minus surroundings, against the same window placed "
    "elsewhere in this light curve (more negative = more unusual for this star). explained_total: fraction of the "
    "dip predicted by the flux's coupling to background and centroid outside the dip (near 0 = systematics do not "
    "explain it; near 1 = they do). depth_ratio_sml/lag: fractional depth in a smaller/larger aperture over the "
    "primary, only for dips under 0.3 d (near 1 = on target). edge_frac/edge_sigma: largest single-cadence flux "
    "step in the dip as a fraction of its depth / in noise units. kinematic_limit: deepest flux change an occulter "
    "moving at twice the star's escape speed can make in one cadence; a step larger than this cannot be an "
    "occultation. z_bkg, z_x, z_y: in-dip shift of background and centroids against random placements. "
    "dip_false_alarm_p: how often the mirror-image (brightening) light curves of 10^4 stars of similar brightness "
    "give a dip this strong, x2 for the two duration classes searched."
)
TEST_KEYS = ("placement_z", "explained_total", "depth_ratio_sml", "depth_ratio_lag", "edge_frac", "edge_sigma",
             "kinematic_limit", "z_bkg", "z_x", "z_y")


def to_candidate(lc, cid):
    meta = {name: np.asarray(lc.aux[aux], float) for name, aux in CHANNELS.items() if aux in lc.aux}
    m = lc.meta
    source = {"tmag": m.get("tmag"), "teff": m.get("teff"), "radius_rsun": m.get("radius"), "mass_msun": m.get("mass"),
              "ra": m.get("ra"), "dec": m.get("dec")}
    return Candidate(cid, "tess", {k: v for k, v in source.items() if v is not None and np.isfinite(v)},
                     [LightCurve("TESS", lc.t, lc.flux, lc.err, meta)],
                     context={"pipeline_test_definitions": TEST_DEFINITIONS,
                              "photometry": "MIT QLP simple-aperture flux, 200 s cadence, scattered-light cadences "
                                            "removed; one TESS sector"})


def to_dimming(cand):
    lc = cand.band("TESS")
    aux = {CHANNELS[k]: np.asarray(v, float) for k, v in lc.meta.items() if k in CHANNELS}
    meta = {"radius": cand.source.get("radius_rsun"), "mass": cand.source.get("mass_msun")}
    return DimLC(lc.time, lc.flux, lc.flux_err, band="TESS", meta=meta, aux=aux)


class Detector:
    """The bake-off detector as a vetting hook: dips, a mirror-null p-value, and per-dip systematics tests."""

    def __init__(self, null):
        self.null = null

    def __call__(self, cand):
        lc = to_dimming(cand)
        traces = scan(lc)
        events = sorted(find_events(lc, traces, threshold=5.0), key=lambda e: -e.snr)[:6]
        cand.dips = [{"t_min": e.t_peak, "t_start": e.t_start, "t_end": e.t_end, "depth": e.depth,
                      "duration": e.t_end - e.t_start, "n_points": e.n_points} for e in events]
        cand.scores, cand.tests = {}, {}
        if not events:
            cand.scores["dip_false_alarm_p"] = 1.0
            return cand
        e = events[0]
        cls = "short" if e.scale <= SHORT_SCALE else "long"
        stratum = tmag_stratum([cand.source.get("tmag", 12.0)])
        p = calibrated_p([e.snr], stratum, self.null[cls].values, self.null.stratum.values)[0]
        cand.scores["dip_false_alarm_p"] = float(min(1.0, 2 * p))
        diag = event_diagnostics(lc, e)
        for k in TEST_KEYS:
            v = diag.get(k)
            if v is not None and np.isfinite(v):
                cand.tests[k] = {"statistic": float(v)}
        return cand


def cmd_build(args):
    run = Path(args.out)
    vet = run / "vet"
    vet.mkdir(exist_ok=True)
    rng = np.random.default_rng(secrets.randbits(64))
    null = pd.read_parquet(run / "null.parquet")
    cand = pd.read_parquet(run / "candidates.parquet")
    targets = pd.read_parquet(run / "targets.parquet")
    entries = []
    for r in cand.head(args.n_real).itertuples():
        lc = qlp.fetch(int(r.sector), int(r.tic))
        if lc is not None:
            entries.append((to_candidate(qlp.clean(lc), f"tic:{int(r.tic)}:s{int(r.sector)}"), {"kind": "real"}))
    if args.controls:
        ctl = pd.read_csv(args.controls)
        ctl = ctl[ctl.qlp_sector.notna()] if "qlp_sector" in ctl else ctl
        for r in ctl.sample(min(args.n_controls, len(ctl)), random_state=int(rng.integers(1 << 31))).itertuples():
            lc = qlp.fetch(int(r.qlp_sector), int(r.tic))
            if lc is not None:
                entries.append((to_candidate(qlp.clean(lc), f"tic:{int(r.tic)}:s{int(r.qlp_sector)}"),
                                {"kind": "control", "label": r.label, "aliases": [str(r.name)],
                                 "acceptable": [r.group] if "group" in ctl else None}))
    n_decoy = int(round(args.decoy_frac * len(entries) / (1 - args.decoy_frac)))
    hosts = targets[~targets.tic.isin(cand.tic)].sample(4 * n_decoy + 10, random_state=int(rng.integers(1 << 31)))
    truths = []
    for r in hosts.itertuples():
        if len(truths) >= n_decoy:
            break
        lc = qlp.fetch(int(r.sector), int(r.tic))
        if lc is None:
            continue
        host = to_candidate(qlp.clean(lc), f"tic:{int(r.tic)}:s{int(r.sector)}")
        spec = str(rng.choice(TESS_DECOYS))
        try:
            dec, extra, truth = make_decoy(host, spec, rng)
        except RuntimeError:
            continue
        entries.append((dec, extra))
        truths.append(truth)
    packets, keys = build_batch(entries, rng, detect=Detector(null))
    for k in keys:
        if k.get("kind") == "decoy":
            k["candidate_id"] = "decoy"
    (vet / "products.json").write_text(json.dumps(packets))
    (vet / "key.json").write_text(json.dumps(keys, indent=1, default=str))
    (vet / "decoy_truth.json").write_text(json.dumps(truths, indent=1, default=str))
    kinds = pd.Series([k["kind"] for k in keys]).value_counts().to_dict()
    print(f"wrote {len(packets)} packets: {kinds}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cosmos_search.tess.vet")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--n-real", type=int, default=24)
    b.add_argument("--controls")
    b.add_argument("--n-controls", type=int, default=4)
    b.add_argument("--decoy-frac", type=float, default=0.15)
    b.set_defaults(fn=cmd_build)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
