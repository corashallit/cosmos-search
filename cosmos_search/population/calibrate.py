"""Injection universes for the half-A search, and their scoring (docs/population.md §8).

    python -m cosmos_search.population.calibrate run   --out OUT --lc LC_DIR    # plan (sealed), universes, candidates
    python -m cosmos_search.population.calibrate score --out OUT                # after the candidates are committed

`run` writes population/universes_candidates.json (candidates only) and seals the plan and the per-universe truth
(which half-A SOV events are injected) to OUT. `score` refuses to open them unless the candidates file is committed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.inject.sealed import write_sealed

from . import analysis, inject, scan
from .run import RECIPIENTS, REPO, _committed, _head, events_A, hour_bin, load_sealed, mask_pixel

CANDS = REPO / "population" / "universes_candidates.json"
RECOVER_SHARE = 0.5


load_stars = analysis.load_stars


def _a_counts(ev, mask, source_ids):
    e = ev[(ev.half == 0)]
    e = e[~mask[hour_bin(e.t_deep.to_numpy()), mask_pixel(e.source_id.to_numpy())]]
    c = e.groupby("source_id").size()
    return pd.Series(source_ids).map(c).fillna(0).to_numpy(float), e


def cmd_run(a):
    out = Path(a.out)
    stars = load_stars(out)
    mask = np.load(out / "mask_A.npy")
    y_real = analysis.outcome(stars, events_A(out), 0, mask)
    terms = json.loads(Path(a.candidates).read_text())["model"]["terms"]
    terms = [t for t in terms if t != "cmd"]
    rng = np.random.default_rng(inject.secret_seed())
    plan = inject.draw_plan(stars, rng)
    write_sealed(plan, out / "universes_plan.json.age", RECIPIENTS)
    index = inject.file_index(a.lc)
    win = scan.Windows()
    results, truth = [], []
    for uni in plan[: a.limit]:
        y = y_real.copy()
        tr = []
        for reg in uni["regions"]:
            mem = inject.region_members(stars, reg["ra"], reg["dec"], reg["radius_deg"], reg["shells"])
            ids = stars.source_id.to_numpy()[mem]
            lc = inject.load_light_curves(a.lc, ids, index)
            if lc.empty:
                tr.append({"source_ids": [], "n_injected_sov_A": 0, "n_dips": 0})
                continue
            lc_inj, dips = inject.inject_dips(lc, reg["rate_per_day"], np.random.default_rng(reg["seed"]))
            ev, found = inject.injected_events(lc_inj, dips)
            counts, ev_a = _a_counts(ev, mask, ids)
            y[mem] = counts
            inj = ev_a[ev_a.injected]
            tr.append({"source_ids": inj.source_id.astype(int).tolist(), "n_injected_sov_A": int(len(inj)),
                       "n_dips": int(len(dips))})
        cands, info, _, _ = analysis.search(stars, y, stars.E_gen_A.to_numpy(), win=win, n_null=a.n_null,
                                            rng=np.random.default_rng(a.seed + uni["universe"]), terms=terms)
        info.pop("_lam", None)
        results.append({"universe": uni["universe"], "candidates": cands, "max_llr": info["max_llr"],
                        "n_events": info["n_events"], "null_q95": info["null_quantiles"][0.95]})
        truth.append({"universe": uni["universe"], "regions": tr})
        print(f"universe {uni['universe']}: {len(cands)} candidates", flush=True)
    write_sealed(truth, out / "universes_truth.json.age", RECIPIENTS)
    Path(a.universes).write_text(json.dumps({"code_commit": _head(), "universes": results}, indent=1) + "\n")


def cmd_score(a):
    cands_path = Path(a.universes)
    if not _committed(cands_path) and not a.dry:
        raise SystemExit(f"commit {cands_path} before scoring")
    out = Path(a.out)
    stars = load_stars(out)
    plan = load_sealed(out / "universes_plan.json.age")
    truth = {t["universe"]: t for t in load_sealed(out / "universes_truth.json.age")}
    cands = {r["universe"]: r for r in json.loads(cands_path.read_text())["universes"]}
    win = scan.Windows()
    rows = []
    for uni in plan:
        u = uni["universe"]
        if u not in cands:
            continue
        cs = cands[u]["candidates"]
        members = [analysis.window_members(stars, win, c) for c in cs]
        matched = set()
        for reg, tr in zip(uni["regions"], truth[u]["regions"]):
            ids = np.asarray(tr["source_ids"], np.int64)
            best, best_k = 0.0, None
            for k, m in enumerate(members):
                share = np.isin(ids, stars.source_id.to_numpy()[m]).mean() if len(ids) else 0.0
                if share > best:
                    best, best_k = share, k
            rec = best >= RECOVER_SHARE
            if rec:
                matched.add(best_k)
            rows.append({"universe": u, "radius_deg": reg["radius_deg"], "shells": reg["shells"],
                         "n_members": reg["n_members"], "n_target": reg["n_target"],
                         "n_injected_sov_A": tr["n_injected_sov_A"], "best_share": best, "recovered": rec,
                         "candidate": cs[best_k] if best_k is not None else None})
        for k, c in enumerate(cs):
            if k not in matched:
                rows.append({"universe": u, "unmatched_candidate": c})
    res = {"regions": [r for r in rows if "recovered" in r], "unmatched": [r for r in rows if "unmatched_candidate" in r],
           "plan": plan}
    cands_path.with_name("universes_scored.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    reg = pd.DataFrame(res["regions"])
    if len(reg):
        print(reg[["universe", "radius_deg", "n_members", "n_injected_sov_A", "best_share", "recovered"]].to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.population.calibrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "score"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--lc")
        s.add_argument("--limit", type=int)
        s.add_argument("--n-null", type=int, default=scan.N_NULL)
        s.add_argument("--seed", type=int, default=7)
        s.add_argument("--candidates", default=str(REPO / "population" / "candidates_A.json"))
        s.add_argument("--universes", default=str(CANDS))
        s.add_argument("--dry", action="store_true", help="score without the commit check (dry runs only)")
    a = ap.parse_args(argv)
    {"run": cmd_run, "score": cmd_score}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
