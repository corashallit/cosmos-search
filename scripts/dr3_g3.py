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

Selection is per-source competition (knockoff+) run separately in each G-magnitude bin, and the union is
taken (`stats.stratified_knockoff_q`). Pooled over all sources, it found nothing on DR3. Faint stars cannot dip
by many robust sigma, but they spike freely, so their bright winners filled the top of the pooled list and
cancelled the bright stars' dips. The pooled q is kept as `qpool_*` for comparison. G × n_g strata are used for
the symmetry table, where the number of epochs changes the scale of an extreme.

DR3 epoch photometry covers DPAC's variables, so eclipsing binaries and other known dimmers will dominate
the dim winners. Competition q says "more dimming than this source's own bright side explains", not
"unexplained by astrophysics"; class labels come next.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosmos_search.stats import knockoff_q, stratified_knockoff_q, tail_ratio  # noqa: E402

G_ZP = 25.6874
MIN_TRANSITS = 10
N_EDGES = (10, 20, 40, 80)
THRESHOLDS = (3.0, 5.0, 8.0, 12.0, 20.0, 30.0)
FEATURE_COLS = ["source_id", "n_g", "med_flux_g", "min_z_rob", "max_z_rob", "min_partner_z", "max_partner_z",
                "min_frac", "min_frac_bp", "min_frac_rp", "n_low_episodes", "n_rej_low3", "slope_g", "control"]
AUX_COLS = ["bp_rp", "ruwe", "ipd_frac_multi_peak"]
OPTIONAL_COLS = ["min_z_raw", "max_z_raw", "stage0_n_band_inconsistent", "stage0_n_colour_fault",
                 "stage0_n_inflated_error", "stage0_n_untested"]


def load(features_dir):
    """Features only, floats as float32: all 11.75 M DR3 sources then fit in ~1 GB. Aux joins the selection only."""
    parts = sorted(Path(features_dir).glob("*.parquet"))
    if not parts:
        raise SystemExit(f"no feature parts in {features_dir}")
    names = set(pq.read_schema(parts[0]).names)
    cols = FEATURE_COLS + [c for c in OPTIONAL_COLS if c in names]
    tables = []
    for p in parts:
        t = pq.read_table(p, columns=cols)
        tables.append(t.cast(pa.schema([pa.field(x.name, pa.float32()) if pa.types.is_floating(x.type) else x
                                        for x in t.schema])))
    return pa.concat_tables(tables).to_pandas()


def with_aux(sel, aux_path):
    if aux_path is None or sel.empty:
        return sel
    names = set(pq.read_schema(aux_path).names)
    cols = ["source_id"] + [c for c in AUX_COLS if c in names]
    aux = pq.read_table(aux_path, columns=cols, filters=[("source_id", "in", sel["source_id"].tolist())]).to_pandas()
    return sel.merge(aux, on="source_id", how="left")


def statistics(f):
    """Dim and bright statistics per source, NaN where undefined (no partner, too few transits).

    Yields one statistic at a time, so only one pair of 10⁷-element arrays is alive at once."""
    ok = (f["n_g"] >= MIN_TRANSITS).to_numpy()
    col = lambda c: f[c].to_numpy()  # noqa: E731
    yield "single", np.where(ok, -col("min_z_rob"), np.nan), np.where(ok, col("max_z_rob"), np.nan)
    yield ("fov", np.where(ok, np.minimum(-col("min_z_rob"), -col("min_partner_z")), np.nan),
           np.where(ok, np.minimum(col("max_z_rob"), col("max_partner_z")), np.nan))
    if "min_z_raw" in f:
        yield "single_raw", np.where(ok, -col("min_z_raw"), np.nan), np.where(ok, col("max_z_raw"), np.nan)


def strata(f):
    """Integer G bin and G × n_g code per source (strings for 10⁷ sources cost GBs), plus G itself."""
    with np.errstate(divide="ignore", invalid="ignore"):
        g = G_ZP - 2.5 * np.log10(f["med_flux_g"].to_numpy(dtype=float))
    g_bin = np.clip(np.floor(np.nan_to_num(g, nan=21.0)), 8, 21).astype(np.int16)
    n_idx = np.clip(np.searchsorted(N_EDGES, f["n_g"].to_numpy(), side="right") - 1, 0, None).astype(np.int16)
    return g_bin, g_bin * len(N_EDGES) + n_idx, g


def _label(code):
    return f"G={code // len(N_EDGES)}|n>={N_EDGES[code % len(N_EDGES)]}"


def run(f):
    g_bin, codes, g = strata(f)
    sym, q_cols = [], {"source_id": f["source_id"].to_numpy()}
    for name, dim, bright in statistics(f):
        fin = np.isfinite(dim) & np.isfinite(bright)
        for row in tail_ratio(dim[fin], bright[fin], codes[fin], THRESHOLDS):
            sym.append({"statistic": name, **row, "stratum": _label(int(row["stratum"]))})
        w = np.where(fin, np.where(dim > bright, dim, np.where(bright > dim, -bright, 0.0)), np.nan)
        q, qpool = np.ones(dim.size, np.float32), np.ones(dim.size, np.float32)
        q[fin] = stratified_knockoff_q(dim[fin], bright[fin], g_bin[fin])
        qpool[fin] = knockoff_q(dim[fin], bright[fin])
        q_cols[f"w_{name}"] = w.astype(np.float32)
        q_cols[f"q_{name}"] = q
        q_cols[f"qpool_{name}"] = qpool
    out = pd.DataFrame(q_cols)
    out["g_mag"] = g
    return pd.DataFrame(sym), out


def ranked(f, q, top=50, level=0.1, aux_path=None):
    """Sources selected by the FoV-confirmed statistic at q <= level, deepest first, with colour of the dip.

    0.1 because this list feeds vetting, which is the next filter, and because each G bin pays the knockoff+
    floor of 1/level discoveries: at 0.01 a bin needs 100 dimmers before it can report any."""
    sel = with_aux(f.merge(q[(q["q_fov"] <= level) & (q["w_fov"] > 0)], on="source_id"), aux_path)
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


def summary(sym, q, f, aux_path=None):
    lines = ["# DR3 G3 summary", ""]
    for name in [c[2:] for c in q.columns if c.startswith("q_")]:
        qq, qp = q[f"q_{name}"], q[f"qpool_{name}"]
        lines.append(f"- `{name}`: {np.sum(qq <= 0.01):,} sources at q ≤ 0.01, {np.sum(qq <= 0.1):,} at q ≤ 0.1"
                     f" (pooled competition: {np.sum(qp <= 0.1):,} at q ≤ 0.1; of"
                     f" {np.isfinite(q[f'w_{name}']).sum():,} with the statistic defined).")
    by_g = q.assign(g_bin=np.clip(np.floor(np.nan_to_num(q["g_mag"], nan=21.0)), 8, 21).astype(int))
    by_g = by_g[by_g["w_fov"].notna()].groupby("g_bin").agg(
        n=("w_fov", "size"), dim_wins=("w_fov", lambda w: int((w > 0).sum())),
        selected=("q_fov", lambda v: int((v <= 0.1).sum()))).reset_index()
    lines += ["", "## `fov` selections at q ≤ 0.1 by G bin", "", _markdown(by_g)]
    lines += ["", "## Symmetry (dim / bright exceedances), pooled over strata", "",
              "| Statistic | Threshold | n_dim | n_bright | ratio |", "|---|---|---|---|---|"]
    pooled = sym.groupby(["statistic", "threshold"])[["n_dim", "n_bright"]].sum().reset_index()
    for r in pooled.itertuples():
        lines.append(f"| {r.statistic} | {r.threshold:g} | {r.n_dim:,} | {r.n_bright:,} | "
                     f"{(r.n_dim + 0.5) / (r.n_bright + 0.5):.2f} |")
    top, n_sel = ranked(f, q, aux_path=aux_path)
    lines += ["", f"## FoV-confirmed dimmers at q ≤ 0.1: {n_sel:,} sources; deepest 50", "",
              _markdown(top)]
    return "\n".join(lines) + "\n"


def main(argv=None):
    argv = argv or sys.argv[1:]
    features_dir, aux_path, out_dir = argv[:3]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = load(features_dir)
    sym, q = run(f)
    sym.to_csv(out / "symmetry.csv", index=False)
    q.to_parquet(out / "q.parquet", index=False)
    (out / "summary.md").write_text(summary(sym, q, f, aux_path))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
