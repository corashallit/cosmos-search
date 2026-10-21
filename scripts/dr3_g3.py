"""G3 on the DR3 rehearsal: mirror-null symmetry, per-source competition q-values, first ranked list.

    python scripts/dr3_g3.py FEATURES_DIR AUX_PARQUET OUT_DIR

Reads the per-source features of the DR3 epoch-photometry pass (docs/infra/gaia-epoch.md) and writes:

- OUT_DIR/symmetry.csv: stats.tail_ratio of dim vs bright, per G × n_g stratum and threshold;
- OUT_DIR/q.parquet: per source, the competition score W and knockoff+ q for each statistic;
- OUT_DIR/summary.md: tables for docs/arms, and the top of the ranked list.

Two statistics, both mirrored so the bright side is the null for the dim side:

- `single`: the deepest G point, dim = -min_z_rob, bright = max_z_rob.
- `fov`: the deepest point that its partner transit in the other field of view (~106.5 min later) also
  shows, dim = min(-min_z_rob, -min_partner_z). A dip lasting hours appears in both fields of view;
  cosmic rays, CCD defects and FoV calibration offsets mostly do not. This is the invariance test of
  docs/arms/gaia-dimmers.md, applied symmetrically.
- `single_raw`: as `single`, on the extremes before Stage 0 cleaning (`min_z_raw`, `max_z_raw`), when the
  features carry them (features-v1-stage0). Comparing it with `single` measures what cleaning buys.

DR3 epoch photometry covers DPAC's variables, so eclipsing binaries and other known dimmers will dominate
the dim winners. Competition q says "more dimming than this source's own bright side explains", not
"unexplained by astrophysics"; class labels come next.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosmos_search.stats import knockoff_q, tail_ratio  # noqa: E402

G_ZP = 25.6874
MIN_TRANSITS = 10
N_EDGES = (10, 20, 40, 80)
THRESHOLDS = (3.0, 5.0, 8.0, 12.0, 20.0, 30.0)
FEATURE_COLS = ["source_id", "n_g", "med_flux_g", "min_z_rob", "max_z_rob", "min_partner_z", "max_partner_z",
                "min_frac", "min_frac_bp", "min_frac_rp", "n_low_episodes", "n_rej_low3", "slope_g", "control"]
OPTIONAL_COLS = ["min_z_raw", "max_z_raw", "stage0_n_band_inconsistent", "stage0_n_colour_fault",
                 "stage0_n_inflated_error", "stage0_n_untested"]


def load(features_dir, aux_path):
    parts = sorted(Path(features_dir).glob("*.parquet"))
    if not parts:
        raise SystemExit(f"no feature parts in {features_dir}")
    names = set(pq.read_schema(parts[0]).names)
    cols = FEATURE_COLS + [c for c in OPTIONAL_COLS if c in names]
    f = pd.concat([pq.read_table(p, columns=cols).to_pandas() for p in parts], ignore_index=True)
    aux = pq.read_table(aux_path).to_pandas()
    return f.merge(aux, on="source_id", how="left")


def statistics(f):
    """Dim and bright statistics per source, NaN where undefined (no partner, too few transits)."""
    ok = f["n_g"] >= MIN_TRANSITS
    dim_single = np.where(ok, -f["min_z_rob"], np.nan)
    bright_single = np.where(ok, f["max_z_rob"], np.nan)
    dim_fov = np.where(ok, np.minimum(-f["min_z_rob"], -f["min_partner_z"]), np.nan)
    bright_fov = np.where(ok, np.minimum(f["max_z_rob"], f["max_partner_z"]), np.nan)
    out = {"single": (dim_single, bright_single), "fov": (dim_fov, bright_fov)}
    if "min_z_raw" in f:
        out["single_raw"] = (np.where(ok, -f["min_z_raw"], np.nan), np.where(ok, f["max_z_raw"], np.nan))
    return out


def strata(f):
    with np.errstate(divide="ignore", invalid="ignore"):
        g = G_ZP - 2.5 * np.log10(f["med_flux_g"].to_numpy(dtype=float))
    g_bin = np.clip(np.floor(g), 8, 21).astype(int)
    n_lo = np.asarray(N_EDGES)[np.clip(np.searchsorted(N_EDGES, f["n_g"].to_numpy(), side="right") - 1, 0, None)]
    return np.array([f"G={a}|n>={b}" for a, b in zip(g_bin, n_lo)]), g


def run(f):
    labels, g = strata(f)
    stats = statistics(f)
    sym, q_cols = [], {"source_id": f["source_id"].to_numpy()}
    for name, (dim, bright) in stats.items():
        fin = np.isfinite(dim) & np.isfinite(bright)
        for row in tail_ratio(dim[fin], bright[fin], labels[fin], THRESHOLDS):
            sym.append({"statistic": name, **row})
        w = np.where(fin, np.where(dim > bright, dim, np.where(bright > dim, -bright, 0.0)), np.nan)
        q = np.ones(dim.size)
        q[fin] = knockoff_q(dim[fin], bright[fin])
        q_cols[f"w_{name}"] = w
        q_cols[f"q_{name}"] = q
    out = pd.DataFrame(q_cols)
    out["g_mag"] = g
    return pd.DataFrame(sym), out


def ranked(f, q, top=50):
    """Sources selected by the FoV-confirmed statistic at q <= 0.01, deepest first, with colour of the dip."""
    df = f.merge(q, on="source_id")
    sel = df[(df["q_fov"] <= 0.01) & (df["w_fov"] > 0)].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        sel["dip_colour_R"] = sel["min_frac_rp"] / sel["min_frac_bp"]
    cols = ["source_id", "g_mag", "bp_rp", "n_g", "w_fov", "q_fov", "min_frac", "dip_colour_R", "n_low_episodes",
            "ruwe", "ipd_frac_multi_peak", "stage0_n_band_inconsistent"]
    cols = [c for c in cols if c in sel.columns]
    return sel.sort_values("w_fov", ascending=False)[cols].head(top), len(sel)


def _markdown(df):
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)  # noqa: E731
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def summary(sym, q, f):
    lines = ["# DR3 G3 summary", ""]
    for name in [c[2:] for c in q.columns if c.startswith("q_")]:
        qq = q[f"q_{name}"]
        lines.append(f"- `{name}`: {np.sum(qq <= 0.01):,} sources at q ≤ 0.01, {np.sum(qq <= 0.1):,} at q ≤ 0.1"
                     f" (of {np.isfinite(q[f'w_{name}']).sum():,} with the statistic defined).")
    lines += ["", "## Symmetry (dim / bright exceedances), pooled over strata", "",
              "| Statistic | Threshold | n_dim | n_bright | ratio |", "|---|---|---|---|---|"]
    pooled = sym.groupby(["statistic", "threshold"])[["n_dim", "n_bright"]].sum().reset_index()
    for r in pooled.itertuples():
        lines.append(f"| {r.statistic} | {r.threshold:g} | {r.n_dim:,} | {r.n_bright:,} | "
                     f"{(r.n_dim + 0.5) / (r.n_bright + 0.5):.2f} |")
    top, n_sel = ranked(f, q)
    lines += ["", f"## FoV-confirmed dimmers at q ≤ 0.01: {n_sel:,} sources; deepest 50", "",
              _markdown(top)]
    return "\n".join(lines) + "\n"


def main(argv=None):
    argv = argv or sys.argv[1:]
    features_dir, aux_path, out_dir = argv[:3]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = load(features_dir, aux_path)
    sym, q = run(f)
    sym.to_csv(out / "symmetry.csv", index=False)
    q.to_parquet(out / "q.parquet", index=False)
    (out / "summary.md").write_text(summary(sym, q, f))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
