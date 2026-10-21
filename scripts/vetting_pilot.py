"""Small Stage A pilot on Gaia DR3: known-answer controls from DPAC's classifier, decoys injected into quiet stars.

    python scripts/vetting_pilot.py fetch --out PILOT_DIR      # TAP + DataLink, writes controls.jsonl, hosts.jsonl
    python scripts/vetting_pilot.py build --out PILOT_DIR      # packets (products.json) and plaintext key.json

The key is a tuning key: written to disk, never printed. Read it only through
`python -m cosmos_search.vetting score`, after the batch is committed.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cosmos_search.gaia.archive import tap
from cosmos_search.gaia.epoch import datalink_epoch_photometry
from cosmos_search.inject.classes.vetting_decoys import vetting_batch

# DPAC class -> (our label, acceptable groups, how many)
CONTROLS = {
    "ECL": ("eclipsing_binary", ["astrophysical"], 2),
    "YSO": ("yso_dipper", ["astrophysical"], 2),
    "RCB": ("rcrb_dust", ["astrophysical"], 2),
    "LPV": ("pulsation_or_irregular", ["astrophysical"], 2),
    "RS": ("rotation_or_spots", ["astrophysical"], 2),
    "SOLAR_LIKE": ("rotation_or_spots", ["astrophysical"], 2),
    "AGN": ("extragalactic", ["astrophysical"], 2),
}
HOST_CLASSES = {"SOLAR_LIKE": 6, "RS": 6}
DECOYS = {"grey_primes": 2, "grey_geometric": 2, "chromatic_flip": 2, "single_band_dip": 2, "dust_dip": 2,
          "periodic_grey": 2}
SOURCE_COLS = "s.source_id, s.ra, s.dec, s.phot_g_mean_mag, s.bp_rp, s.parallax, s.parallax_error, s.ruwe, " \
              "s.ipd_frac_multi_peak, s.phot_bp_rp_excess_factor"


def select(cls, n, extra="", after=0):
    """n sources of a DPAC class in random_index order; `after` skips earlier pilots' sources."""
    q = f"""SELECT TOP {n} {SOURCE_COLS} FROM gaiadr3.vari_classifier_result c
            JOIN gaiadr3.gaia_source s USING (source_id)
            WHERE c.best_class_name = '{cls}' AND c.best_class_score > 0.9 AND s.random_index > {int(after)} {extra}
            ORDER BY s.random_index"""
    return tap().query(q)


def candidates(rows, lc):
    out = []
    for r in rows:
        sid = int(r["source_id"])
        sub = lc.filter(lc["source_id"].to_numpy() == sid).to_pandas()
        lcs = []
        for band in ("G", "BP", "RP"):
            b = sub[(sub.band == band) & np.isfinite(sub.flux) & (sub.flux > 0)]
            if len(b) < 10:
                continue
            med = float(np.median(b.flux))
            lcs.append({"band": band, "time": b["t"].tolist(), "flux": (b["flux"] / med).tolist(),
                        "flux_err": (b["err"] / med).tolist(), "meta": {"fov": b["fov"].tolist(), "flags": b["flags"].tolist()}})
        if len(lcs) == 3:
            src = {k: (None if np.ma.is_masked(r[k]) else float(r[k])) for k in r.colnames if k != "source_id"}
            out.append({"candidate_id": f"gaia_dr3:{sid}", "arm": "gaia",
                        "source": {k: v for k, v in src.items() if v is not None}, "lightcurves": lcs})
    return out


def fetch(out, after=0):
    out.mkdir(parents=True, exist_ok=True)
    controls = []
    for cls, (label, acceptable, n) in CONTROLS.items():
        rows = select(cls, n + 2, after=after)
        lc = datalink_epoch_photometry([int(s) for s in rows["source_id"]])
        for c in candidates(rows, lc)[:n]:
            controls.append({**c, "label": label, "acceptable": acceptable, "aliases": [], "dpac_class": cls})
    # Hosts from classes that also appear among the controls, so planted packets are not a separate population.
    hosts = []
    for cls, n in HOST_CLASSES.items():
        rows = select(cls, n + 3, "AND s.phot_g_mean_mag BETWEEN 12 AND 17", after=after)
        hosts += candidates(rows, datalink_epoch_photometry([int(s) for s in rows["source_id"]]))[:n]
    (out / "controls.jsonl").write_text("\n".join(json.dumps(c) for c in controls))
    (out / "hosts.jsonl").write_text("\n".join(json.dumps(h) for h in hosts))
    print(f"{len(controls)} controls, {len(hosts)} hosts")


def build(out):
    cfg = {"controls": str(out / "controls.jsonl"), "hosts": str(out / "hosts.jsonl"), "holdout": ["tail"],
           "decoys": DECOYS}
    rng = np.random.default_rng(int.from_bytes(__import__("os").urandom(16), "little"))
    packets, truth, keys = vetting_batch(cfg, rng)
    (out / "products.json").write_text(json.dumps(packets))
    (out / "key.json").write_text(json.dumps(keys))
    (out / "truth.json").write_text(json.dumps(truth))
    tokens = [len(p["text"]) // 4 for p in packets]
    print(f"{len(packets)} packets; ~{int(np.median(tokens))} tokens median, {max(tokens)} max")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "build"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--after", type=int, default=0, help="random_index to start after (fresh sources per pilot)")
    a = ap.parse_args()
    fetch(a.out, a.after) if a.step == "fetch" else build(a.out)
