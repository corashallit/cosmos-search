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
from .analyse import CLASSES, calibrated_p, event_class, tmag_stratum
from .controls import jobs

# Candidate meta channel <-> dimming aux name.
CHANNELS = {"background": "bkg", "centroid_x": "x", "centroid_y": "y", "flux_small_aperture": "flux_sml",
            "flux_large_aperture": "flux_lag"}
# One-band decoys only: chromatic and single-band specs need several bands.
TESS_DECOYS = ("grey_primes", "grey_geometric", "step", "dust_dip", "periodic_grey", "grey_dip")
# Rendered with each test in the packet (vetting/packet.py); all refer to the strongest dip.
TEST_DESCRIPTIONS = {
    "placement_z": "dip mean minus surroundings, against the same window placed elsewhere in this light curve; "
                   "more negative = more unusual for this star",
    "explained_bkg": "fraction of the dip predicted by the flux's coupling to the background outside the dip; "
                     "near 0 = the background does not explain it, near 1 = it does",
    "depth_ratio_sml": "fractional depth in a smaller aperture over the primary (dips under 0.1 d only); near 1 = on target",
    "depth_ratio_lag": "fractional depth in a larger aperture over the primary (dips under 0.1 d only); near 1 = on target",
    "edge_frac": "largest single-cadence flux step in the dip as a fraction of its depth",
    "edge_sigma": "largest single-cadence flux step in the dip in units of the point-to-point noise",
    "kinematic_limit": "deepest flux change an occulter at twice the star's escape speed can make in one cadence; "
                       "a larger step cannot be an occultation",
    "z_bkg": "in-dip background shift against random placements of the same window",
    "z_x": "in-dip column centroid shift against random placements of the same window",
    "z_y": "in-dip row centroid shift against random placements of the same window",
}
# The packet renders test statistics only, so their meanings travel in the context.
TEST_CONTEXT = "; ".join(f"{k}: {v}" for k, v in TEST_DESCRIPTIONS.items())
MAX_DIPS = 3  # dips listed and zoomed in each packet
TEST_KEYS = ("placement_z", "explained_bkg", "depth_ratio_sml", "depth_ratio_lag", "edge_frac", "edge_sigma",
             "kinematic_limit", "z_bkg", "z_x", "z_y")

# controls.csv class -> (taxonomy label, groups counted as correct). Aperiodic deep dimmers are the class this
# arm is looking for: the literature explains most of them (debris, disks), but "unexplained" from a packet
# alone is a fair answer, so both groups count.
CONTROL_LABELS = {
    "1_yso_dipper": ("yso_dipper", ["astrophysical"]),
    "2_aperiodic_deep_dimmer": ("debris_transit", ["astrophysical", "unexplained"]),
    "3_rcb": ("rcrb_dust", ["astrophysical"]),
    "4_uxor_haebe": ("yso_dipper", ["astrophysical"]),
    "5_long_or_disk_eclipse": ("disk_occultation", ["astrophysical"]),
    "5b_detached_eb_reference": ("eclipsing_binary", ["astrophysical"]),
}


def control_label(cls, notes=""):
    if cls in CONTROL_LABELS:
        return CONTROL_LABELS[cls]
    notes = str(notes).lower()
    if "scattered" in notes:
        return "scattered_light", ["instrumental"]
    if "blend" in notes or "neighbour" in notes or "contamination" in notes:
        return "blend_or_crowding", ["instrumental"]
    return "processing", ["instrumental", "astrophysical"]


def to_candidate(lc, cid):
    meta = {name: np.asarray(lc.aux[aux], float) for name, aux in CHANNELS.items() if aux in lc.aux}
    m = lc.meta
    source = {"tmag": m.get("tmag"), "teff": m.get("teff"), "radius_rsun": m.get("radius"), "mass_msun": m.get("mass"),
              "ra": m.get("ra"), "dec": m.get("dec")}
    return Candidate(cid, "tess", {k: v for k, v in source.items() if v is not None and np.isfinite(v) and (k in ('ra', 'dec') or v > 0)},
                     [LightCurve("TESS", lc.t, lc.flux, lc.err, meta)],
                     context={"dip_false_alarm_p": "how often the mirror-image (brightening) light curves of 10^4 "
                                                   "stars of similar brightness give a dip this strong, x2 for the "
                                                   "two duration classes searched (up to 3 h, longer)",
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
        events = sorted(find_events(lc, traces, threshold=5.0), key=lambda e: -e.snr)[:MAX_DIPS]
        cand.dips = [{"t_min": e.t_peak, "t_start": e.t_start, "t_end": e.t_end, "depth": e.depth,
                      "duration": e.t_end - e.t_start, "n_points": e.n_points} for e in events]
        # Every packet carries every test name (None when it cannot be computed), because the batch keeps only
        # names all packets share and a test missing from some packets would be dropped from all.
        cand.scores, cand.tests = {}, {k: {"statistic": None} for k in TEST_KEYS}
        cand.context["tests"] = TEST_CONTEXT
        if not events:
            cand.scores["dip_false_alarm_p"] = 1.0
            return cand
        stratum = tmag_stratum([cand.source.get("tmag", 12.0)])
        ps = [calibrated_p([e.snr], stratum, self.null[c].values, self.null.stratum.values)[0]
              for e, c in zip(events, event_class([e.scale for e in events]))]
        e = events[int(np.argmin(ps))]
        cand.scores["dip_false_alarm_p"] = float(min(1.0, len(CLASSES) * min(ps)))
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
        ev = [(i, s) for i, s, state in jobs(ctl, max_event=3, max_quiet=0) if state == "event"]
        # Stage A calibration only transfers if controls pass the same detector as real candidates, so use
        # event sectors in which the detector fired (controls.evaluate writes controls_results.parquet).
        res = run / "controls_results.parquet"
        if res.exists():
            r = pd.read_parquet(res)
            r = r[r.state.eq("event") & r.flag_either]
            fired = set(zip(r.ctl, r.sector))
            ev = [(i, s) for i, s in ev if (i, s) in fired]
        else:
            print("no controls_results.parquet: run `python -m cosmos_search.tess.controls` first; using all event sectors")
        for k in rng.permutation(len(ev))[:args.n_controls]:
            i, sector = ev[k]
            r = ctl.loc[i]
            lc = qlp.fetch(int(sector), int(r.tic))
            if lc is None:
                continue
            label, acceptable = control_label(r["class"], r["notes"])
            entries.append((to_candidate(qlp.clean(lc), f"tic:{int(r.tic)}:s{int(sector)}"),
                            {"kind": "control", "label": label, "acceptable": acceptable,
                             "aliases": [str(r["name"])], "control_class": r["class"]}))
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
    b.add_argument("--n-controls", type=int, default=12)
    b.add_argument("--decoy-frac", type=float, default=0.15)
    b.set_defaults(fn=cmd_build)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
