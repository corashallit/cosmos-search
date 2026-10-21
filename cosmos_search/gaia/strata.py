"""Rates of a feature-defined class across strata of magnitude, colour and crowding.

    python -m cosmos_search.gaia.strata FEATURES_DIR AUX_PARQUET

Used to size the random full-light-curve channel (docs/plan.md): if the rate of "quiet except for a
DPAC-rejected faint point" varies strongly across strata, those strata can be oversampled with known
inclusion probabilities (cosmos_search.stats audit sampling) and the estimate stays unbiased.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

STRATA = {
    "G": ("phot_g_mean_mag", [0, 11, 13, 15, 17, 18, 19, 20, 21, 25]),
    "BP-RP": ("bp_rp", [-5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 10]),
    "ipd_frac_multi_peak": ("ipd_frac_multi_peak", [-1, 0, 2, 10, 30, 101]),
    "|b| (deg)": ("abs_b", [0, 5, 10, 20, 40, 91]),
}


def quiet_rejected(f):
    return (f["n_rej_low3"] >= 1) & (f["min_z_rob"] > -3)


def load(features_dir, aux_path):
    cols = ["source_id", "n_rej_low3", "min_z_rob", "n_g", "n_rej"]
    f = pd.concat([pq.read_table(p, columns=cols).to_pandas() for p in sorted(Path(features_dir).glob("*.parquet"))],
                  ignore_index=True)
    aux = pq.read_table(aux_path).to_pandas()
    aux["abs_b"] = aux["b"].abs()
    return f.merge(aux, on="source_id", how="inner")


def rates(df, flag):
    rows = []
    for name, (col, edges) in STRATA.items():
        cut = pd.cut(df[col], edges, right=False)
        g = df.assign(flag=flag).groupby(cut, observed=True)["flag"]
        for interval, n, k in zip(g.size().index, g.size().to_numpy(), g.sum().to_numpy()):
            rows.append({"stratum": name, "bin": str(interval), "n": int(n), "k": int(k),
                         "rate": k / n if n else np.nan})
    return pd.DataFrame(rows)


def to_markdown(r):
    lines = ["| Stratum | Bin | Sources | Quiet with rejected faint point | Rate |", "|---|---|---|---|---|"]
    for row in r.itertuples():
        lines.append(f"| {row.stratum} | {row.bin} | {row.n:,} | {row.k:,} | {row.rate:.1%} |")
    return "\n".join(lines)


def main(argv=None):
    argv = argv or sys.argv[1:]
    df = load(*argv[:2])
    flag = quiet_rejected(df)
    print(f"{len(df):,} sources; overall rate {flag.mean():.2%}\n")
    print(to_markdown(rates(df, flag)))


if __name__ == "__main__":
    main()
