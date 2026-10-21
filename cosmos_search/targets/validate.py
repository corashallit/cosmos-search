"""Check the TAP prefilter against features computed from full DR3 light curves.

    python -m cosmos_search.targets.validate data/targets/dr3 data/gaia-dr3-epoch/features

The prefilter only sees vari_summary statistics, which DPAC computes from non-rejected transits. The full
light curves see everything, so they measure what the prefilter misses, by candidate class.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLUMNS = ["source_id", "med_flux_g", "min_z_rob", "max_z_rob", "min_partner_z", "max_partner_z",
           "n_rej_low3", "n_low_episodes", "control"]

# G = 25.6874 - 2.5 log10(flux), the DR3 G zero point (Riello et al. 2021)
G_ZP = 25.6874


def classes(f):
    deep = f["min_z_rob"] <= -8
    bright = f["max_z_rob"] >= 8
    return {
        "deep dip, confirmed in other FoV": deep & (f["min_partner_z"] <= -4),
        "deep dip, partner disagrees": deep & (f["min_partner_z"] > -1),
        "deep dip, no partner transit": deep & f["min_partner_z"].isna(),
        "only DPAC-rejected faint points": (f["n_rej_low3"] >= 1) & (f["min_z_rob"] > -3),
        "mirror: bright spike, confirmed": bright & (f["max_partner_z"] >= 4),
        "random controls": f["control"],
    }


def validate(prefilter_dir, features_dir):
    sel = pq.read_table(Path(prefilter_dir) / "prefilter.parquet", columns=["source_id"]).to_pandas()
    paths = sorted(Path(features_dir).glob("*.parquet"))
    f = pd.concat([pq.read_table(p, columns=COLUMNS).to_pandas() for p in paths], ignore_index=True)
    f["selected"] = f["source_id"].isin(sel["source_id"])
    f["gmag"] = G_ZP - 2.5 * np.log10(f["med_flux_g"])
    out = {"files": len(paths), "sources": len(f), "selected_fraction": float(f["selected"].mean()),
           "classes": {}}
    for name, m in classes(f).items():
        row = {"n": int(m.sum()), "recall": float(f.loc[m, "selected"].mean()) if m.any() else None}
        b = m & (f["gmag"] < 13)
        row["n_G<13"] = int(b.sum())
        row["recall_G<13"] = float(f.loc[b, "selected"].mean()) if b.any() else None
        out["classes"][name] = row
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    print(json.dumps(validate(*argv[:2]), indent=2))


if __name__ == "__main__":
    main()
