"""From the vetting queue to Stage A packets, and from committed verdicts to a scored shortlist.

    python -m cosmos_search.dayone.vet build --queue rank/queue.parquet --ranked rank/ranked.parquet \\
        --lightcurves stream/lightcurves [--context xm/context.parquet] --out vet/
    python -m cosmos_search.dayone.vet score --out vet/ --batch vet/batch [--plan plan.json.age]

`build` writes products.json (packets), prompts/ and key.json. The key maps packet ids to candidates and
roles; it carries no injection truth, because the vetting stage does not know which sources were injected.
`score` opens it only after `cosmos_search.vetting commit`, then (with --plan) decrypts the sealed injection
plan to label injected sources as decoys, and writes the report and the shortlist.

Every packet goes through the same path (docs/vetting.md §4): Stage 0 cleaning, the batch detector
(`batch.simple_dips` on G), the cross-candidate coincidence test over the whole ranked list, the
solar-system check at the deepest dip, cross-match context, and the tail holdout.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.vetting import coincidence, sso, stage0
from cosmos_search.vetting.batch import build_batch, simple_dips

from .rank import _blend_map, read_lightcurves

SOURCE_KEYS = ("ra", "dec", "phot_g_mean_mag", "bp_rp", "parallax", "parallax_error", "ruwe", "ipd_frac_multi_peak",
               "phot_bp_rp_excess_factor", *stage0.BLEND_COLS)
COINCIDENCE_TAU = 0.05      # days: one Gaia transit pair is 0.074 d; same-scan stars share times within ~minutes
COINCIDENCE_WINDOW = 5.0    # days
CONTEXT_KEYS = {"allwise_w1mpro": "W1", "allwise_w2mpro": "W2", "allwise_w3mpro": "W3", "allwise_w4mpro": "W4",
                "allwise_p_cal": "wise_match_p", "2mass_k_m": "Ks", "2mass_p_cal": "twomass_match_p"}
P_U_SHORTLIST = 0.25        # pilots 2-3: probes 4/8 and controls 0/26 at P(unexplained) >= 0.25


def candidate_dicts(queue, ranked, lc_dir, wide=None):
    """Candidate dicts (vetting.candidate shape) for the queue, from the stream's kept light curves."""
    ids = queue["source_id"].astype("int64").to_numpy()
    lc = read_lightcurves(lc_dir, ids)
    lc, _ = stage0.clean_long(lc, blend=_blend_map(queue))
    df = lc.to_pandas()
    df["band"] = df["band"].astype(str)
    ctx = wide.set_index("source_id") if wide is not None else None
    out = []
    for _, row in queue.iterrows():
        sid = int(row["source_id"])
        sub = df[df.source_id == sid]
        lcs = []
        for band in ("G", "BP", "RP"):
            b = sub[(sub.band == band) & np.isfinite(sub.flux) & (sub.flux > 0)]
            if len(b) < 10:
                continue
            med = float(np.median(b.flux))
            lcs.append({"band": band, "time": b["t"].tolist(), "flux": (b["flux"] / med).tolist(),
                        "flux_err": (b["err"] / med).tolist(),
                        "meta": {"fov": b["fov"].tolist(), "flags": b["flags"].tolist()}})
        if not lcs or lcs[0]["band"] != "G":
            continue
        source = {k: float(row[k]) for k in SOURCE_KEYS if k in row and pd.notna(row[k])}
        context = {}
        if ctx is not None and sid in ctx.index:
            for col, name in CONTEXT_KEYS.items():
                v = ctx.at[sid, col] if col in ctx.columns else np.nan
                context[name] = float(v) if pd.notna(v) else float("nan")
        out.append({"candidate_id": f"gaia_dr3:{sid}", "arm": "gaia", "source": source, "lightcurves": lcs,
                    "context": context, "role": row["role"]})
    return out


def coincidence_tests(ranked):
    """Coincidence of each ranked source's deepest epoch with other ranked sources' (Stage 0, docs/vetting.md §3)."""
    dips = [{"candidate_id": f"gaia_dr3:{int(s)}", "t_min": float(t)}
            for s, t in zip(ranked["source_id"], ranked["t_deepest"]) if np.isfinite(t)]
    res = coincidence.coincidence(dips, COINCIDENCE_TAU, COINCIDENCE_WINDOW)
    flagged = coincidence.flag_candidates(res, 0.01)
    by_id = {r["candidate_id"]: r for r in res}
    return by_id, flagged


def add_tests(cands, coinc, sso_check=True, holdout_tail=True):
    """Coincidence and solar-system tests at the deepest G epoch that Stage A will see. Under the tail holdout
    that is the deepest epoch before the cut, and the tests are marked visible_only so the packet keeps them."""
    from cosmos_search.vetting.candidate import Candidate
    from cosmos_search.vetting.packet import HOLDOUT_TAIL

    for c in cands:
        cand = Candidate.from_dict(c)
        t0, t1 = cand.time_span()
        cut = t0 + (1 - HOLDOUT_TAIL) * (t1 - t0) if holdout_tail else np.inf
        g = cand.band("G")
        vis = np.flatnonzero(g.time < cut)
        if not len(vis):
            c["tests"] = {}
            continue
        k = vis[int(np.argmin(g.flux[vis]))]
        d = {"t_min": float(g.time[k]), "depth": float(1 - g.flux[k])}
        r = coinc(d["t_min"], c["candidate_id"])
        c["tests"] = {"coincidence": {
            "statistic": r["k"] if r else 0, "p": r["p"] if r else 1.0, "visible_only": True,
            "description": "number of other queued or ranked stars whose deepest epoch lies within 0.05 d of this "
                           "star's deepest shown epoch; p under a flat local rate over +-5 d (small p: shared with "
                           "unrelated stars, so instrumental)"}}
        if sso_check:
            try:
                res = sso.check(cand, [d])
                verdict = res[0]["verdict"] if res else "clear"
            except Exception as exc:  # the check is evidence, not a gate; record that it failed
                verdict = f"failed ({type(exc).__name__})"
            c["tests"]["solar_system"] = {
                "statistic": {"clear": 0, "sso_faint": 1, "sso_bright": 2}.get(verdict, -1), "visible_only": True,
                "description": f"known solar-system bodies within 15 arcsec at the deepest shown epoch: {verdict}"}
    return cands


def coincidence_at(ranked):
    """A function (t, candidate_id) -> coincidence record for that star's dip at t against the other ranked stars'
    deepest epochs."""
    r = ranked.dropna(subset=["t_deepest"])
    t_all = r["t_deepest"].to_numpy(dtype=float)
    order = np.argsort(t_all)
    t_all = t_all[order]
    own_t = dict(zip((f"gaia_dr3:{int(s)}" for s in r["source_id"]), r["t_deepest"].astype(float)))

    def at(t, cid):
        lo, hi = np.searchsorted(t_all, t - COINCIDENCE_WINDOW), np.searchsorted(t_all, t + COINCIDENCE_WINDOW, "right")
        dt = np.abs(t_all[lo:hi] - t)
        mine = own_t.get(cid)
        own_n = int(mine is not None and abs(mine - t) <= COINCIDENCE_WINDOW)
        own_k = int(mine is not None and abs(mine - t) <= COINCIDENCE_TAU)
        n, k = hi - lo - own_n, int(np.sum(dt <= COINCIDENCE_TAU)) - own_k
        p_near = COINCIDENCE_TAU / COINCIDENCE_WINDOW
        return {"k": k, "n": n, "p": coincidence.binom_sf(k, n, p_near)}
    return at


def build(queue, ranked, lc_dir, out, wide=None, rng=None, sso_check=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = rng or np.random.default_rng(int.from_bytes(__import__("os").urandom(16), "little"))
    cands = candidate_dicts(queue, ranked, lc_dir, wide)
    _, flagged = coincidence_tests(ranked)
    cands = add_tests(cands, coincidence_at(ranked), sso_check)
    entries = [(c, {"kind": "control" if c["role"] == "control" else "real", "role_in_queue": c["role"],
                    **({"label": "eclipsing_binary", "acceptable": ["astrophysical"]} if c["role"] == "control" else {})})
               for c in cands]
    # The batch detector, as a hook: with no hook, build_batch also clears the tests and scores.
    packets, keys = build_batch(entries, rng, holdout=("tail",), detect=lambda c: setattr(c, "dips", simple_dips(c)))
    (out / "products.json").write_text(json.dumps(packets))
    (out / "key.json").write_text(json.dumps(keys))
    (out / "coincidence_flagged.json").write_text(json.dumps(flagged))
    return packets, keys


def score_batch(out, batch, plan=None):
    """Reveal the key (after commit), mark injected sources as decoys, score, and write the shortlist."""
    from cosmos_search.vetting.commit import load_verdicts, reveal
    from cosmos_search.vetting.report import decision, score

    out = Path(out)
    keys = reveal(batch, out / "key.json")
    injected = {}
    if plan:
        from cosmos_search.inject.classes import gaia_stream
        injected = gaia_stream.load_plan(plan)
    for k in keys.values():
        sid = int(k["candidate_id"].split(":")[1])
        if sid in injected:
            fam = injected[sid]["family"]
            k.update(kind="decoy", injected_family=fam, **DECOY_LABELS.get(fam, DECOY_LABELS["grey_dip"]))
    verdicts = load_verdicts(batch)
    rep = score(verdicts, keys)
    rows = []
    for v in verdicts:
        k = keys[v.packet_id]
        rows.append({"packet_id": v.packet_id, "candidate_id": k["candidate_id"], "kind": k.get("kind"),
                     "role_in_queue": k.get("role_in_queue"), "injected_family": k.get("injected_family"),
                     "ok": v.ok, "p_unexplained": v.groups["unexplained"] if v.ok else None,
                     "decision": decision(v) if v.ok else None,
                     "top_leaf": max(v.class_probs, key=v.class_probs.get) if v.ok else None})
    table = pd.DataFrame(rows).sort_values("p_unexplained", ascending=False)
    short = table[(table.kind == "real") & (table.p_unexplained >= P_U_SHORTLIST)]
    rep["shortlist"] = short.to_dict("records")
    rep["shortlist_rule"] = f"real packets with P(unexplained) >= {P_U_SHORTLIST}"
    ctrl = table[table.kind == "control"]
    rep["controls_flagged"] = {"k": int((ctrl.p_unexplained >= P_U_SHORTLIST).sum()), "n": len(ctrl)}
    table.to_csv(out / "verdicts_table.csv", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=2, sort_keys=True, default=str))
    return rep, table


# Decoy labels by injected family, as vetting_decoys.DECOY_SPECS reads them (what an ideal observer could conclude
# is not checked here: in-stream injections are scored by family, and the report says so).
DECOY_LABELS = {
    "grey_dip": {"label": "unexplained", "acceptable": ["astrophysical", "unexplained"], "role": "ambiguous"},
    "dust_dip": {"label": "yso_dipper", "acceptable": ["astrophysical"], "role": "rare_natural"},
    "blue_dip": {"label": "unexplained", "acceptable": ["astrophysical", "unexplained"], "role": "ambiguous"},
    "dip_train_poisson": {"label": "unexplained", "acceptable": ["astrophysical", "unexplained"], "role": "ambiguous"},
    "dip_train_periodic": {"label": "eclipsing_binary", "acceptable": ["astrophysical"], "role": "rare_natural"},
    "dip_train_primes": {"label": "unexplained", "acceptable": ["unexplained"], "role": "anomaly_probe"},
    "dip_train_geometric": {"label": "unexplained", "acceptable": ["unexplained"], "role": "anomaly_probe"},
    "secular_fade": {"label": "unexplained", "acceptable": ["astrophysical", "unexplained", "instrumental"],
                     "role": "ambiguous"},
    "step": {"label": "calibration_offset", "acceptable": ["instrumental", "unexplained"], "role": "incoherent"},
    "chromatic_flip": {"label": "unexplained", "acceptable": ["unexplained", "instrumental"], "role": "incoherent"},
    "g_only": {"label": "other_instrumental", "acceptable": ["instrumental"], "role": "incoherent"},
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.dayone.vet")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--queue", required=True)
    b.add_argument("--ranked", required=True)
    b.add_argument("--lightcurves", required=True)
    b.add_argument("--context")
    b.add_argument("--out", required=True)
    b.add_argument("--no-sso", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--batch", required=True)
    s.add_argument("--plan")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        wide = pd.read_parquet(a.context) if a.context else None
        packets, _ = build(pd.read_parquet(a.queue), pd.read_parquet(a.ranked), a.lightcurves, a.out, wide,
                           sso_check=not a.no_sso)
        from cosmos_search.vetting.__main__ import main as vmain
        vmain(["prompts", "--packets", str(Path(a.out) / "products.json"), "--out", str(Path(a.out) / "prompts")])
        print(f"{len(packets)} packets")
    else:
        rep, table = score_batch(a.out, a.batch, a.plan)
        print(json.dumps({k: rep[k] for k in ("n", "shortlist_rule", "controls_flagged") if k in rep}, indent=2))
        print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
