"""Known-answer controls: does the detector find documented dimmers, and leave their quiet sectors alone?

    python -m cosmos_search.tess.controls --run data/bakeoff/v0 [--controls cosmos_search/tess/controls.csv]

Uses controls.csv (literature-backed TESS dimmers by class, with the sectors in which the dimming is
documented; see controls_notes.md) and the bake-off run's mirror null and anomaly population, so the
numbers are on the same scale as the bake-off. Writes controls_results.parquet and controls_summary.json
into the run directory.

Each star contributes up to three event sectors and up to two quiet sectors. Quiet sectors of the same star
are the best negatives there are: same star, same pipeline, different state.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.dimming.detect import find_events, scan
from cosmos_search.dimming.features import event_diagnostics, extract
from cosmos_search.dimming.population import AnomalyModel, anomaly_columns

from . import qlp
from .analyse import CLASSES, VETO_REASONS, calibrated_p, event_class, tmag_stratum, veto_reasons

DEFAULT = Path(__file__).with_name("controls.csv")


def _sectors(s):
    if not isinstance(s, str) or not s.strip():
        return []
    return [int(x) for x in s.split(";") if x.strip().isdigit() and int(x) < 1000]


def jobs(ctl, max_event=3, max_quiet=2):
    out = []
    for r in ctl.itertuples():
        qlp_s = set(_sectors(r.qlp_sectors))
        ev = [s for s in _sectors(r.qlp_event_sectors) if s in qlp_s]
        persistent = isinstance(r.event_basis, str) and r.event_basis.startswith(("persistent", "class-only"))
        quiet = [] if persistent else sorted(qlp_s - set(_sectors(r.event_sectors)))
        out += [(r.Index, s, "event") for s in ev[:max_event]]
        out += [(r.Index, s, "quiet") for s in quiet[-max_quiet:]]
    return out


def run_one(tic, sector):
    raw = qlp.fetch(sector, tic)
    if raw is None:
        return None
    lc = qlp.clean(raw)
    if len(lc) < 500:
        return None
    traces = scan(lc)
    f, _, _ = extract(lc, traces, flag_bits=qlp.REPORT)
    dips = find_events(lc, traces, threshold=5.0)
    rows = []
    for e in dips:
        d = e.as_dict()
        d.update(event_diagnostics(lc, e, qlp.REPORT))
        rows.append(d)
    meta = {k: lc.meta[k] for k in ("camera", "ccd", "tmag")}
    return f, rows, meta


def evaluate(run, controls=DEFAULT, workers=16):
    run = Path(run)
    ctl = pd.read_csv(controls)
    null = pd.read_parquet(run / "null.parquet")
    todo = jobs(ctl)
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda j: run_one(int(ctl.loc[j[0], "tic"]), j[1]), todo))

    feats, events = [], []
    for k, ((i, sector, state), res) in enumerate(zip(todo, results)):
        if res is None:
            continue
        f, rows, meta = res
        base = {"job": k, "ctl": i, "cls": ctl.loc[i, "class"], "name": ctl.loc[i, "name"],
                "tic": int(ctl.loc[i, "tic"]), "sector": sector, "state": state, **meta}
        feats.append({**base, **f})
        events += [{**base, **r, "variant": "real", "row": k} for r in rows]
    F = pd.DataFrame(feats)
    E = pd.DataFrame(events)

    # Vetoes as in the bake-off, minus common mode (needs the star's CCD neighbours, which we don't have here).
    if len(E):
        E["veto"] = veto_reasons(E, {}, {})
        E["fast_edge"] = E.get("fast_edge", pd.Series(False, index=E.index)).fillna(False).astype(bool)
        E["occ"] = (E.veto == "") & ~E.fast_edge
        E["far"] = 1.0
        E["cls_event"] = event_class(E.scale)
        for cls in CLASSES:
            sel = E.occ & (E.cls_event == cls)
            if sel.any() and cls in null:
                p = calibrated_p(E.loc[sel, "snr"].values, tmag_stratum(E.loc[sel, "tmag"]), null[cls].values,
                                 null.stratum.values)
                E.loc[sel, "far"] = np.minimum(1.0, len(CLASSES) * p)
        best = E.groupby("job").far.min()
        F["dip_far"] = F.job.map(best).fillna(1.0)
        F["n_dips_any"] = F.job.map(E.groupby("job").size()).fillna(0).astype(int)
        F["veto_reasons"] = F.job.map(E[E.veto != ""].groupby("job").veto.agg(lambda v: ";".join(sorted(set(";".join(v).split(";"))))))
    else:
        F["dip_far"], F["n_dips_any"], F["veto_reasons"] = 1.0, 0, None

    # Population anomaly score against the bake-off's real light curves.
    bf = pd.read_parquet(run / "features.parquet")
    real = bf[bf.variant == "real"]
    cols = anomaly_columns(real.columns)
    model = AnomalyModel(cols).fit(real[cols].values, tmag_stratum(real.tmag))
    F["anomaly"] = model.score(F[cols].values, tmag_stratum(F.tmag))
    thr = float(np.quantile(model.score(real[cols].values, tmag_stratum(real.tmag), in_fit=True), 0.99))
    F["flag_dip"] = F.dip_far <= 1e-2
    F["flag_anom"] = F.anomaly >= thr
    F["flag_either"] = F.flag_dip | F.flag_anom

    F.drop(columns=[c for c in F.columns if c.startswith(("min_stat", "max_stat", "log_beta", "tail_ratio"))]) \
        .to_parquet(run / "controls_results.parquet")
    summary = (F.groupby(["cls", "state"])
               .agg(n=("job", "size"), stars=("tic", "nunique"), dip=("flag_dip", "mean"),
                    anomaly=("flag_anom", "mean"), either=("flag_either", "mean"))
               .round(3).reset_index())
    veto_counts = {}
    if len(E):
        ev_event = E[E.state == "event"]
        for reason in VETO_REASONS[1:]:
            veto_counts[reason] = float(ev_event.veto.str.contains(reason).mean())
        veto_counts["fast_edge"] = float(ev_event.fast_edge.mean())
    out = {"by_class_and_state": summary.to_dict(orient="records"),
           "event_sector_dip_veto_fractions": veto_counts, "anomaly_threshold": thr}
    (run / "controls_summary.json").write_text(json.dumps(out, indent=2))
    return summary, out


def main(argv=None):
    p = argparse.ArgumentParser(prog="cosmos_search.tess.controls")
    p.add_argument("--run", required=True)
    p.add_argument("--controls", default=str(DEFAULT))
    args = p.parse_args(argv)
    summary, out = evaluate(args.run, args.controls)
    print(summary.to_string())
    print(json.dumps(out["event_sector_dip_veto_fractions"], indent=2))


if __name__ == "__main__":
    main()
