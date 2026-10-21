"""Score a bake-off run: vetoes, empirical false-alarm rates, injection recovery, anomaly ranking.

    python -m cosmos_search.tess.analyse --out data/bakeoff/v0

Writes summary.json, recovery.csv and candidates.parquet into the run directory.

Recovery counts an injected event only when the detection overlaps an injection window *and* the real
light curve has no comparable event there, so a host's own dips are not credited to the injection. The
anomaly score's recovery is likewise counted only for hosts that were not already above the threshold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.dimming.population import AnomalyModel, anomaly_columns, common_mode_epochs, in_common_mode
from cosmos_search.stats import bh, stratified_p, tail_p

GROUP = ["sector", "camera", "ccd"]
FAR_LEVELS = (1e-2, 1e-3)
ANOMALY_QUANTILES = (0.99, 0.999)
SHORT_SCALE = 0.125     # d; events found at this scale or shorter are calibrated separately

# Veto thresholds, v0. Each is checked against injection loss in summary["veto_loss"].
VETO = {
    "explained": 0.5,      # fraction of the dip predicted by background/centroid coupling (features._explained)
    "depth_ratio": (0.5, 2.0),  # alternative-aperture depth ratio outside this range: not on target
    "edge_dist": 0.1,      # d from a segment edge (residual ramps after scattered-light removal)
    "dump_dist": 0.05,     # d from a momentum dump, for events shorter than dump_max_dur
    "dump_max_dur": 0.3,
}
VETO_REASONS = ("common_mode", "systematics", "aperture", "segment_edge", "momentum_dump")


TMAG_EDGES = (10.0, 12.0, 13.0, 14.0)    # coarse enough that each stratum holds ~10^3 nulls in a 10^4 run


def tmag_stratum(tmag):
    return np.digitize(np.asarray(tmag, float), TMAG_EDGES)


def calibrated_p(stat, strata, null, null_strata):
    """Mirror-null p-value within the Tmag stratum (cosmos_search.stats); whole-null tail where a stratum
    is too small."""
    p, _ = stratified_p(stat, strata, null, null_strata)
    missing = ~np.isfinite(p)
    if missing.any():
        p[missing], _ = tail_p(np.asarray(stat, float)[missing], null)
    return p


def load(out):
    out = Path(out)
    return (pd.read_parquet(out / "features.parquet"), pd.read_parquet(out / "events.parquet"),
            pd.read_parquet(out / "injections.parquet"))


def veto_reasons(ev, cm_dip, cm_burst):
    """One string per event: '' if it survives, else ';'-joined reasons."""
    reasons = []
    for r in ev.itertuples(index=False):
        why = []
        cm = cm_burst if r.variant == "mirror" else cm_dip
        frac = in_common_mode(cm, (r.sector, r.camera, r.ccd), r.t_start, r.t_end)
        if frac > 0:
            why.append("common_mode")
        if _get(r, "explained_total") > VETO["explained"]:
            why.append("systematics")
        lo, hi = VETO["depth_ratio"]
        ratios = [_get(r, c, np.nan) for c in ("depth_ratio_sml", "depth_ratio_lag")]
        if any(np.isfinite(x) and not lo <= x <= hi for x in ratios):
            why.append("aperture")
        if r.edge_dist < VETO["edge_dist"]:
            why.append("segment_edge")
        if _get(r, "dump_dist", np.inf) < VETO["dump_dist"] and (r.t_end - r.t_start) < VETO["dump_max_dur"]:
            why.append("momentum_dump")
        reasons.append(";".join(why))
    return reasons


def _get(r, name, default=0.0):
    v = getattr(r, name, default)
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else v


def analyse(out):
    out = Path(out)
    feats, events, inj = load(out)
    real = feats[feats.variant == "real"].set_index("row")
    meta = real[GROUP + ["tmag", "tic"]]
    n_per_group = real.groupby(GROUP).size().to_dict()

    ev = events.join(meta[GROUP], on="row")
    low = ev[ev.kind.isin(["low_dip", "low_burst"])]
    cm_dip = common_mode_epochs(
        [(tuple(g), r, a, b) for g, r, a, b in zip(low[low.kind == "low_dip"][GROUP].values,
                                                     low[low.kind == "low_dip"].row, low[low.kind == "low_dip"].t_start,
                                                     low[low.kind == "low_dip"].t_end)], n_per_group)
    cm_burst = common_mode_epochs(
        [(tuple(g), r, a, b) for g, r, a, b in zip(low[low.kind == "low_burst"][GROUP].values,
                                                     low[low.kind == "low_burst"].row, low[low.kind == "low_burst"].t_start,
                                                     low[low.kind == "low_burst"].t_end)], n_per_group)

    det = ev[ev.kind == "det"].copy()
    det["veto"] = veto_reasons(det, cm_dip, cm_burst)
    det["ok"] = det.veto == ""

    rows = real.index.values
    strata = tmag_stratum(meta.loc[rows, "tmag"])
    det["stratum"] = tmag_stratum(det.row.map(meta.tmag))
    # Occultation channel: surviving events whose edges are not kinematically too fast for an occulter
    # (features.edge_kinematics). Too-fast events form a separate "sudden change" channel.
    det["fast_edge"] = det.get("fast_edge", pd.Series(False, index=det.index)).fillna(False).astype(bool)
    det["occ"] = det.ok & ~det.fast_edge
    # Short events are where mirrored flares live; long events are where Boyajian-like dips live.
    # Calibrating each against its own null keeps flares from setting the bar for day-long dips.
    det["cls"] = np.where(det.scale <= SHORT_SCALE, "short", "long")
    det["far"] = 1.0
    null_best = {}
    for cls in ("short", "long"):
        m = det[det.occ & (det.variant == "mirror") & (det.cls == cls)].groupby("row").snr.max()
        null_best[cls] = m.reindex(rows).fillna(0.0).values
        sel = det.occ & (det.cls == cls)
        far = calibrated_p(det.loc[sel, "snr"].values, det.loc[sel, "stratum"].values, null_best[cls], strata)
        det.loc[sel, "far"] = np.minimum(1.0, 2 * far)   # x2: two scale classes searched
    pd.DataFrame({"row": rows, "stratum": strata, **null_best}).to_parquet(out / "null.parquet")
    far_row = {v: det[det.occ & (det.variant == v)].groupby("row").far.min().reindex(rows).fillna(1.0).values
               for v in ("real", "mirror", "injected")}
    far_real = far_row["real"]
    thresholds = {f"{cls}_{a:g}": float(np.quantile(null_best[cls], 1 - a / 2))
                  for cls in ("short", "long") for a in FAR_LEVELS}

    # --- Injection recovery (dip detector) ---
    inj = inj.set_index("row").reindex(rows)
    real_det = det[(det.variant == "real")]
    rec_rows = []
    for row in rows:
        tr = inj.loc[row]
        if not isinstance(tr.family, str):
            continue
        wins = json.loads(tr.windows)
        cand = det[(det.variant == "injected") & (det.row == row)]
        host = real_det[real_det.row == row]
        best_snr, best_far, sudden = 0.0, 1.0, False
        for e in cand.itertuples(index=False):
            if not any(e.t_start <= b + 0.05 and a - 0.05 <= e.t_end for a, b in wins):
                continue
            # Not credited if the host already has an event here at comparable strength.
            clash = host[(host.t_start <= e.t_end) & (e.t_start <= host.t_end) & (host.snr >= 0.8 * e.snr)]
            if len(clash):
                continue
            best_snr = max(best_snr, e.snr)
            if e.occ:
                best_far = min(best_far, e.far)
            sudden |= bool(e.ok and e.fast_edge)
        rec_rows.append({"row": row, "family": tr.family, "max_deficit": tr.max_deficit,
                         "sigma_white": real.loc[row, "sigma_white"],
                         "max_excess": tr.max_excess, "tmag": meta.loc[row, "tmag"],
                         "n_windows": len(wins), "snr_any": best_snr, "far": best_far, "sudden": sudden})
    rec = pd.DataFrame(rec_rows)
    # Peak deficit in units of the host's per-cadence noise: the one number that sets detectability.
    rec["depth_over_noise"] = rec.max_deficit / rec.sigma_white
    rec["dnr_bin"] = pd.cut(rec.depth_over_noise, [0, 0.3, 1, 3, 10, np.inf], right=False,
                            labels=["<0.3", "0.3-1", "1-3", "3-10", ">=10"])
    for a in FAR_LEVELS:
        rec[f"dip_rec_{a:g}"] = rec.far <= a

    # --- Anomaly score ---
    cols = anomaly_columns(feats.columns)
    model = AnomalyModel(cols).fit(real.loc[rows, cols].values, strata)
    anom_real = model.score(real.loc[rows, cols].values, strata, in_fit=True)
    by_var = {v: feats[feats.variant == v].set_index("row").reindex(rows) for v in ("mirror", "injected")}
    anom_inj = model.score(by_var["injected"][cols].values, strata)
    anom_mir = model.score(by_var["mirror"][cols].values, strata)
    q_thr = {q: float(np.quantile(anom_real, q)) for q in ANOMALY_QUANTILES}
    host_anom = pd.Series(anom_real, index=rows)
    rec["anom_injected"] = anom_inj[np.searchsorted(rows, rec.row)] if len(rec) else []
    rec["anom_host"] = host_anom.loc[rec.row].values if len(rec) else []
    for q, s in q_thr.items():
        rec[f"anom_rec_{q:g}"] = (rec.anom_injected >= s) & (rec.anom_host < s)

    # --- Veto loss on injections: fraction of overlapping injected detections removed by each veto ---
    inj_det = det[det.variant == "injected"].merge(inj[["windows"]], left_on="row", right_index=True)
    overl = inj_det[[any(e.t_start <= b and a <= e.t_end for a, b in json.loads(w))
                     for e, w in zip(inj_det.itertuples(index=False), inj_det.windows)]]
    veto_loss = {}
    for reason in VETO_REASONS:
        veto_loss[reason] = {
            "injected_frac": float(overl.veto.str.contains(reason).mean()) if len(overl) else np.nan,
            "real_frac": float(det[det.variant == "real"].veto.str.contains(reason).mean()),
            "mirror_frac": float(det[det.variant == "mirror"].veto.str.contains(reason).mean()),
        }

    # --- Candidates: dip detections passing FDR on the mirror-calibrated FAR, plus top anomalies ---
    fdr, _ = bh(far_real, q=0.1)
    cand = pd.DataFrame({
        "row": rows, "tic": meta.loc[rows, "tic"].values, "sector": meta.loc[rows, "sector"].values,
        "camera": meta.loc[rows, "camera"].values, "ccd": meta.loc[rows, "ccd"].values,
        "tmag": meta.loc[rows, "tmag"].values, "dip_snr": det[det.occ & (det.variant == "real")].groupby("row").snr.max().reindex(rows).fillna(0.0).values,
        "dip_far": far_real,
        "sudden": det[det.ok & det.fast_edge & (det.variant == "real")].groupby("row").size().reindex(rows).fillna(0).values > 0,
        "dip_fdr_pass": fdr, "anomaly": anom_real,
    })
    cand["anomaly_top"] = cand.anomaly >= q_thr[0.99]
    cand["dip_pass"] = cand.dip_far <= FAR_LEVELS[0]
    keep = cand[cand.dip_pass | cand.anomaly_top].copy()
    # Value, v0: evidence (larger of the two tail strengths) x verifiability (bright stars first).
    p_anom = 1 - np.clip(np.searchsorted(np.sort(anom_real), keep.anomaly, side="left") / len(anom_real), 0, 1 - 1e-6)
    evidence = np.maximum(-np.log10(keep.dip_far.clip(1e-12)), -np.log10(p_anom))
    verif = np.clip(1 - (keep.tmag - 11) / 6, 0.5, 1.0)
    keep["value"] = evidence * verif
    keep = keep.sort_values("value", ascending=False)
    best_ev = det[(det.variant == "real") & det.ok].sort_values("snr").groupby("row").tail(1).set_index("row")
    keep = keep.join(best_ev[["t_peak", "t_start", "t_end", "depth", "scale", "asym", "eqw"]], on="row")
    keep.to_parquet(out / "candidates.parquet")

    summary = {
        "n_light_curves": int(len(rows)),
        "n_groups": int(len(n_per_group)),
        "common_mode_bins": {"dip": int(sum(len(g["bins"]) for g in cm_dip.values())),
                             "burst": int(sum(len(g["bins"]) for g in cm_burst.values()))},
        "detections": {v: int(((det.variant == v)).sum()) for v in ("real", "mirror", "injected")},
        "surviving": {v: int(((det.variant == v) & det.ok).sum()) for v in ("real", "mirror", "injected")},
        "rows_with_occultation_event": {v: int((far_row[v] < 1).sum()) for v in far_row},
        "rows_with_sudden_event": {v: int(det[det.ok & det.fast_edge & (det.variant == v)].row.nunique())
                                   for v in far_row},
        "snr_threshold_at_far_by_class": thresholds,
        "anomaly_threshold_at_quantile": {f"{q:g}": s for q, s in q_thr.items()},
        "mirror_anomaly_above_0.99": float(np.mean(anom_mir >= q_thr[0.99])),
        "veto_loss": veto_loss,
        "n_candidates": {"dip_far_1e-2": int(cand.dip_pass.sum()), "dip_fdr_0.1": int(fdr.sum()),
                         "anomaly_top_1pct": int(cand.anomaly_top.sum()), "union": int(len(keep))},
        "recovery_by_family": _recovery_table(rec).to_dict(orient="index"),
        "recovery_by_family_and_depth": _recovery_by_depth(rec),
    }
    rec.to_csv(out / "recovery.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary, rec, keep


def _recovery_table(rec):
    cols = [c for c in rec.columns if c.startswith(("dip_rec_", "anom_rec_"))]
    g = rec.groupby("family")
    t = g[cols].mean()
    t["either_1e-2_0.99"] = g.apply(lambda d: (d["dip_rec_0.01"] | d["anom_rec_0.99"]).mean(), include_groups=False)
    t["n"] = g.size()
    t["median_max_deficit"] = g.max_deficit.median()
    return t.round(3)


def _recovery_by_depth(rec):
    """Dip-detector recall at FAR 1e-2 (or anomaly top 1%), by family and depth-to-noise bin."""
    either = rec["dip_rec_0.01"] | rec["anom_rec_0.99"]
    t = either.groupby([rec.family, rec.dnr_bin], observed=False).agg(["mean", "size"])
    out = {}
    for (fam, b), r in t.iterrows():
        out.setdefault(fam, {})[str(b)] = {"recall": None if r["size"] == 0 else round(float(r["mean"]), 3),
                                           "n": int(r["size"])}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="cosmos_search.tess.analyse")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    summary, rec, keep = analyse(args.out)
    print(json.dumps({k: v for k, v in summary.items() if k != "recovery_by_family"}, indent=2, default=float))
    print(pd.DataFrame(summary["recovery_by_family"]).T.to_string())
    d = summary["recovery_by_family_and_depth"]
    print(pd.DataFrame({f: {b: (f"{v['recall']:.2f} ({v['n']})" if v["n"] else "-") for b, v in bins.items()}
                        for f, bins in d.items()}).T.to_string())


if __name__ == "__main__":
    main()
