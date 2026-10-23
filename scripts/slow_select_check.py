"""How much of what the detector finds survives the whole-sky selection on DR4's summary columns?

    python scripts/slow_detect_check.py ... --out slow_detect/     # first: statistics and fakes
    python scripts/slow_select_check.py --detect slow_detect/ --out slow_select/

DR4's light curves come through DataLink, so the search sees only stars that pass an archive-side selection
first (cosmos_search.slow.select). From the DR3 light curves, slow_detect_check.py also computed that selection's
score (slow_z, from the epochs' unit-weight variance and Abbé value, as DR4 publishes them) for every star and
fake. Here thresholds on slow_z are set per G bin so that a given share of quiet half-B stars is selected, and
for each share the script reports which fakes the detector finds at its 0.1% thresholds and how many of those
the selection kept.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.slow import drift, select

SHARES = (0.001, 0.003, 0.01, 0.03, 0.1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d, out = Path(a.detect), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = pd.read_parquet(d / "stats.parquet")
    fakes = pd.read_parquet(d / "fakes.parquet")
    thr = json.loads((d / "summary.json").read_text())["thresholds"]
    quiet = stats[stats["quiet_b"] & stats["slow_z"].notna()]
    host = quiet.set_index("source_id")[["gbin"]]
    f = fakes.join(host, on="host")
    f = f[f["slow_z"].notna() & f["gbin"].notna()]
    f["gbin"] = f["gbin"].astype(int)
    f["mag"] = np.where(f["gbin"] <= 7, "G<19", "G>=19")
    th = pd.DataFrame({g: v["0.001"] for g, v in thr.items()}).T.reindex(f["mag"])
    f["found"] = (f["s_dim"].to_numpy() > th["s"].to_numpy()) | (f["slope_z"].to_numpy() < -th["z"].to_numpy())

    rows = []
    for share in SHARES:
        cuts = select.thresholds(quiet, share)
        c = cuts.set_index("gbin")["slow_z_min"]
        kept = f["slow_z"].to_numpy() > c.reindex(f["gbin"]).to_numpy()
        for (mag, fam), g in f.assign(kept=kept).groupby(["mag", "family"]):
            hit = g[g["found"]]
            rows.append({"share": share, "mag": mag, "family": fam, "fakes": len(g), "found": len(hit),
                         "found_and_selected": int(hit["kept"].sum()),
                         "selected_given_found": round(hit["kept"].mean(), 3) if len(hit) else np.nan,
                         "selected_any": round(g["kept"].mean(), 3)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "selection_recall.csv", index=False)
    print(res.to_string(index=False))

    # The cuts as they would be set on DR4's own random sample, here from DR3's quiet stars, for the record.
    cuts = select.thresholds(quiet, 0.01)
    cuts["g_bin"] = [f"{drift.G_EDGES[g]:g}-{drift.G_EDGES[g + 1]:g}" for g in cuts["gbin"]]
    cuts.to_csv(out / "cuts_share_0.01.csv", index=False)
    print(cuts.to_string(index=False))
    (out / "example_selection.adql").write_text(select.adql(cuts) + "\n")
    ab = quiet.groupby("gbin")[["uwv_g", "abbe_g", "slow_z"]].median()
    print(ab)


if __name__ == "__main__":
    main()
