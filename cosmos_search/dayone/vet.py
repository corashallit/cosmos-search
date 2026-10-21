"""From the vetting queue to Stage A packets, and from committed verdicts to a scored shortlist.

    python -m cosmos_search.dayone.vet build --queue rank/queue.parquet --ranked rank/ranked.parquet \\
        --lightcurves stream/lightcurves [--context xm/context.parquet] [--external ext.parquet] \\
        [--plan decoys.json.age] --out vet/
    python -m cosmos_search.dayone.vet score --out vet/ --batch vet/batch [--plan plan.json.age]
    # (commit with --archive and push first: score refuses to reveal until the archived verdicts are on origin)

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

from cosmos_search.inject import lightcurve as lc_families
from cosmos_search.vetting import coincidence, sso, stage0
from cosmos_search.vetting.batch import build_batch, simple_dips

from .rank import _blend_map, read_lightcurves

SOURCE_KEYS = ("ra", "dec", "phot_g_mean_mag", "bp_rp", "parallax", "parallax_error", "ruwe", "ipd_frac_multi_peak",
               "phot_bp_rp_excess_factor", *stage0.BLEND_COLS)
COINCIDENCE_TAU = 0.05      # days: one Gaia transit pair is 0.074 d; same-scan stars share times within ~minutes
COINCIDENCE_WINDOW = 5.0    # days
CONTEXT_KEYS = {"allwise_w1mpro": "W1", "allwise_w2mpro": "W2", "allwise_w3mpro": "W3", "allwise_w4mpro": "W4",
                "allwise_p_cal": "wise_match_p", "2mass_k_m": "Ks", "2mass_p_cal": "twomass_match_p"}
EXTERNAL_CONTEXT_KEYS = {"blend_tess_contamination": "tess_contamination"}
EXTERNAL_BANDS = ("ztf_g", "ztf_r", "ztf_i", "TESS")
P_U_SHORTLIST = 0.25        # pilots 2-3: probes 4/8 and controls 0/26 at P(unexplained) >= 0.25
# Shortlist rule v1 (docs/dayone.md, pre-registered before batch 2): for a technosignature search the rare
# astrophysical leaves are as interesting as "unexplained", so the score is their sum. YSO dippers stay common.
INTERESTING = ("disk_occultation", "debris_transit", "other_astrophysical", "unexplained")
S_SHORTLIST = 0.35
# The shortlist rule in force for the D2 freeze (docs/dayone.md). score_batch's headline keys, the printed summary
# and shortlist.csv come from this rule; every other rule is reported under report["rules"] as a comparison.
ADOPTED_RULE = "v3b"
# Injected families an ideal v1 shortlist should keep; the others should read as EB, YSO or instrumental.
TRAIN_FAMILIES = ("dip_train_poisson", "dip_train_primes", "dip_train_geometric")
SINGLE_FAMILIES = ("grey_dip", "blue_dip", "chromatic_flip")
HELD_OUT_FAMILIES = lc_families.HELD_OUT_FAMILIES
TARGET_FAMILIES = ("grey_dip", "blue_dip", "dip_train_poisson", "dip_train_primes", "dip_train_geometric",
                   "chromatic_flip", "secular_fade")


def candidate_dicts(queue, ranked, lc_dir, wide=None, plan_map=None, external=None):
    """Candidate dicts (vetting.candidate shape) for the queue, from the stream's kept light curves.

    plan_map (a decrypted sealed plan) injects into the Gaia epochs here, before Stage 0, as the stream would have,
    and into the external epochs (dayone.external) with the same signal. external adds ZTF and TESS bands after
    Gaia's, plus the TESS contamination ratio to the context."""
    ids = queue["source_id"].astype("int64").to_numpy()
    lc = read_lightcurves(lc_dir, ids)
    if plan_map:
        from cosmos_search.inject.classes.gaia_stream import apply
        lc = apply(lc, plan_map)
    lc, _ = stage0.clean_long(lc, blend=_blend_map(queue))
    ctx = wide.set_index("source_id") if wide is not None else None
    if external is not None:
        from .external import apply as apply_external
        contam = {} if ctx is None or "blend_tess_contamination" not in ctx else \
            ctx["blend_tess_contamination"].dropna().astype(float).to_dict()
        external = apply_external(external[external["source_id"].isin(ids)], plan_map, contam)
        ext_by_id = {int(k): v for k, v in external.groupby("source_id")}
    df = lc.to_pandas()
    df["band"] = df["band"].astype(str)
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
        if external is not None and sid in ext_by_id:
            e = ext_by_id[sid]
            for band in EXTERNAL_BANDS:
                b = e[(e["band"] == band) & np.isfinite(e["flux"]) & (e["flux"] > 0)].sort_values("t")
                if len(b) >= 10:
                    med = float(np.median(b.flux))
                    lcs.append({"band": band, "time": b["t"].tolist(), "flux": (b["flux"] / med).tolist(),
                                "flux_err": (b["err"] / med).tolist(), "meta": {}})
        source = {k: float(row[k]) for k in SOURCE_KEYS if k in row and pd.notna(row[k])}
        context = {}
        if ctx is not None and sid in ctx.index:
            for col, name in {**CONTEXT_KEYS, **(EXTERNAL_CONTEXT_KEYS if external is not None else {})}.items():
                v = ctx.at[sid, col] if col in ctx.columns else np.nan
                context[name] = float(v) if pd.notna(v) else float("nan")
        out.append({"candidate_id": f"gaia_dr3:{sid}", "arm": "gaia", "source": source, "lightcurves": lcs,
                    "context": context, "role": row["role"],
                    "control_label": row["control_label"] if "control_label" in row and pd.notna(row.get("control_label"))
                    else "eclipsing_binary"})
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


def _control_label(c):
    """Known-answer label for a control. Tier-3 EB controls are labelled at the dip. Other DPAC-class controls are
    labelled by star, so an instrumental reading of their dip is acceptable too (pilot 3: labels must be dip-level)."""
    label = c.get("control_label", "eclipsing_binary")
    acceptable = ["astrophysical"] if label == "eclipsing_binary" else ["astrophysical", "instrumental"]
    return {"label": label, "acceptable": acceptable}


def interesting_score(class_probs):
    """Shortlist score v1: the probability mass on the rare astrophysical leaves plus unexplained."""
    return float(sum(class_probs.get(k, 0.0) for k in INTERESTING))


def build(queue, ranked, lc_dir, out, wide=None, rng=None, sso_check=True, plan_map=None, external=None,
          pipeline_flags=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = rng or np.random.default_rng(int.from_bytes(__import__("os").urandom(16), "little"))
    cands = candidate_dicts(queue, ranked, lc_dir, wide, plan_map, external)
    _, flagged = coincidence_tests(ranked)
    cands = add_tests(cands, coincidence_at(ranked), sso_check)
    entries = [(c, {"kind": "control" if c["role"] == "control" else "real", "role_in_queue": c["role"],
                    **(_control_label(c) if c["role"] == "control" else {})})
               for c in cands]
    # The batch detector, as a hook: with no hook, build_batch also clears the tests and scores.
    packets, keys = build_batch(entries, rng, holdout=("tail",), detect=lambda c: setattr(c, "dips", simple_dips(c)),
                                pipeline_flags=pipeline_flags)
    (out / "products.json").write_text(json.dumps(packets))
    (out / "key.json").write_text(json.dumps(keys))
    (out / "coincidence_flagged.json").write_text(json.dumps(flagged))
    return packets, keys


def score_batch(out, batch, plan=None, remote="origin"):
    """Reveal the key (after commit, and once the archived verdicts are on `remote`), mark injected sources as
    decoys, score, and write the shortlist under the adopted rule. remote=None skips the remote check (tuning)."""
    from cosmos_search.vetting.commit import load_verdicts, reveal
    from cosmos_search.vetting.report import decision, score

    out = Path(out)
    keys = reveal(batch, out / "key.json", remote=remote)
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
                     "s_interesting": interesting_score(v.class_probs) if v.ok else None,
                     "top_leaf": max(v.class_probs, key=v.class_probs.get) if v.ok else None})
    table = pd.DataFrame(rows).sort_values("p_unexplained", ascending=False)
    ctrl = table[table.kind == "control"]
    rules = {"v0": {"rule": f"real packets with P(unexplained) >= {P_U_SHORTLIST}",
                    "shortlist": table[(table.kind == "real") & (table.p_unexplained >= P_U_SHORTLIST)].to_dict("records"),
                    "controls_flagged": {"k": int((ctrl.p_unexplained >= P_U_SHORTLIST).sum()), "n": len(ctrl)}},
             "v1": score_v1(table)}
    rep["v1"] = rules["v1"]
    # Rule v3 (docs/dayone.md, batch 4): the recurrence test's flag, and recall on the dip trains (the families
    # the test exists for) and on single-event families separately. The flag comes from the key, which holds the
    # pipeline's tests whether or not the packet showed them; batches built before that read it from the packet.
    packets = json.loads((out / "products.json").read_text())
    flags = {p["packet_id"]: bool(keys[p["packet_id"]].get("pipeline", p["tests"]).get("recurrence", {})
                                  .get("aperiodic", False)) for p in packets}
    table["aperiodic"] = table.packet_id.map(flags).fillna(False)
    table["pipeline_shown"] = table.packet_id.map(lambda pid: keys[pid].get("pipeline_shown", True))
    rules["v3a"] = score_v1(table, targets=TRAIN_FAMILIES)
    rules["v3b"] = score_v1(table, flag="aperiodic", targets=TRAIN_FAMILIES)
    rep["v3"] = {"v3a": rules["v3a"], "v3b": rules["v3b"],
                 "single_events_v3b": score_v1(table, flag="aperiodic", targets=SINGLE_FAMILIES)["target_decoy_recall"]}
    held = (table.kind == "decoy") & table.injected_family.isin(HELD_OUT_FAMILIES)
    if held.any():
        rep["held_out_recall"] = {r: score_v1(table, flag="aperiodic" if r == "v3b" else None,
                                              targets=HELD_OUT_FAMILIES)["target_decoy_recall"] for r in ("v3a", "v3b")}
    rep["rules"] = rules
    adopted = rules[ADOPTED_RULE]
    rep["adopted_rule"] = ADOPTED_RULE
    rep["shortlist_rule"] = adopted["rule"]
    rep["shortlist"] = adopted["shortlist"]
    rep["controls_flagged"] = adopted["controls_flagged"]
    pd.DataFrame(adopted["shortlist"]).to_csv(out / "shortlist.csv", index=False)
    table.to_csv(out / "verdicts_table.csv", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=2, sort_keys=True, default=str))
    return rep, table


def score_v1(table, threshold=S_SHORTLIST, flag=None, targets=TARGET_FAMILIES):
    """Shortlist rule v1 and its pre-registered measures (docs/dayone.md): recall on target-family decoys, the
    false-interesting rate on controls and on the other decoys, and AUCs of S and P(u) for decoys vs controls.
    flag: a boolean column of the table that also shortlists (rule v3b's pipeline flag)."""
    from cosmos_search.inject.recovery import wilson

    ok = table[table.ok]
    hit = ok.s_interesting >= threshold
    if flag is not None:
        hit = hit | ok[flag].fillna(False).astype(bool)
    target = (ok.kind == "decoy") & ok.injected_family.isin(targets)
    other = (ok.kind == "decoy") & ~ok.injected_family.isin(TARGET_FAMILIES + HELD_OUT_FAMILIES)
    ctrl = ok.kind == "control"

    def rate(mask):
        k, n = int((hit & mask).sum()), int(mask.sum())
        lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
        return {"k": k, "n": n, "rate": k / n if n else float("nan"), "interval_95": [float(lo), float(hi)]}

    def auc(col, pos, neg):
        a, b = ok.loc[pos, col].to_numpy(float), ok.loc[neg, col].to_numpy(float)
        if not len(a) or not len(b):
            return float("nan")
        return float(((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))

    short = ok[(ok.kind == "real") & hit]
    rule = f"real packets with S = P({' + '.join(INTERESTING)}) >= {threshold}" + (f" or {flag}" if flag else "")
    return {"rule": rule,
            "shortlist": short.to_dict("records"),
            "target_decoy_recall": rate(target), "other_decoys_flagged": rate(other), "controls_flagged": rate(ctrl),
            "real_flagged": rate(ok.kind == "real"),
            "auc_s_target_vs_controls": auc("s_interesting", target, ctrl),
            "auc_pu_target_vs_controls": auc("p_unexplained", target, ctrl)}


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
    **{f: {"label": "unexplained", "acceptable": ["astrophysical", "unexplained"], "role": "ambiguous"}
       for f in HELD_OUT_FAMILIES},
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
    b.add_argument("--external", help="dayone.external fetch output: ZTF and TESS bands for the packets")
    b.add_argument("--plan", help="sealed decoy plan to inject at build time (dayone.decoys; needs COSMOS_REVEAL_KEY)")
    b.add_argument("--hide-pipeline-flags", action="store_true",
                   help="leave the pipeline's tests (recurrence) out of the packets; they stay in the key for scoring")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--batch", required=True)
    s.add_argument("--plan")
    s.add_argument("--unpublished", action="store_true",
                   help="reveal without checking that the archived verdicts are on origin (tuning batches only)")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        wide = pd.read_parquet(a.context) if a.context else None
        from cosmos_search.inject.classes.gaia_stream import load_plan
        packets, _ = build(pd.read_parquet(a.queue), pd.read_parquet(a.ranked), a.lightcurves, a.out, wide,
                           sso_check=not a.no_sso, plan_map=load_plan(a.plan) if a.plan else None,
                           external=pd.read_parquet(a.external) if a.external else None,
                           pipeline_flags=not a.hide_pipeline_flags)
        from cosmos_search.vetting.__main__ import main as vmain
        vmain(["prompts", "--packets", str(Path(a.out) / "products.json"), "--out", str(Path(a.out) / "prompts")])
        print(f"{len(packets)} packets")
    else:
        rep, table = score_batch(a.out, a.batch, a.plan, remote=None if a.unpublished else "origin")
        adopted = rep["rules"][rep["adopted_rule"]]
        summary = {"adopted_rule": rep["adopted_rule"], "shortlist_rule": rep["shortlist_rule"],
                   "shortlist": [r["candidate_id"] for r in rep["shortlist"]],
                   **{k: adopted[k] for k in ("target_decoy_recall", "controls_flagged", "real_flagged") if k in adopted}}
        print(json.dumps(summary, indent=2, default=str))
        print("comparisons:", ", ".join(f"{r}: {len(v['shortlist'])} shortlisted" for r, v in rep["rules"].items()))
        print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
