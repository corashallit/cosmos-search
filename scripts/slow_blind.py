"""Sealed blind test of the slow-change detector and rule slow-v1 (docs/arms/slow-changes.md, "Blind test (S6)").

    python scripts/slow_blind.py make   --dir D --run /home/user/slow/dr3b --aux AUX   # draw; truth sealed with age
    python scripts/slow_blind.py run    --dir D                                        # detector + rule -> D/flags.csv
    python scripts/slow_blind.py commit --dir D --batch slow-blind-1                   # archive flags in git
    git push
    python scripts/slow_blind.py reveal --dir D --batch slow-blind-1                   # refuses unless pushed

--run is a dr3_pass output that kept every light curve and shares no stars with the one the rule was calibrated
on (shard 1/10 or 2/10 against shard 0/10). The batch holds quiet stars as they are, and quiet stars of G < 19
carrying one fake each: the rule's new shapes, the shapes it was calibrated on, and brightenings (`_families`).
slow-v1 (slow-blind-1) tested the detector; slow-v2 adds the slow period search and tests it on new repeating
shapes (inject.lightcurve.HELD_OUT_SLOW). Every light curve gets an anonymous id and the order is shuffled; each host star is
used once and never also appears as a natural star. Nobody sees the truth until the flags are committed and on
origin."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cosmos_search.inject import lightcurve as inj
from cosmos_search.inject.sealed import write_sealed
from cosmos_search.slow import detect, drift, period, rule, rule_v2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slow_detect_check import BAND_NAMES, corrected, depth_g, scored

ROOT = Path(__file__).resolve().parents[1]
RECIPIENTS = [ROOT / "keys" / "cora.age.pub", ROOT / "keys" / "reveal.age.pub"]
LC_COLUMNS = ["source_id", "band", "t", "flux", "err", "flags", "fov"]

_STATE = {}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()


def _read(name):
    t = pq.read_table(Path(_STATE["run"]) / "lightcurves" / name, columns=LC_COLUMNS).to_pandas()
    t["band"] = t["band"].astype(str)
    return t[t["source_id"].isin(_STATE["wanted"])]


def _band_codes(lc):
    """Band names to drift.BANDS codes (0 G, 1 BP, 2 RP), as drift.normalise reads them from a DataFrame."""
    lc["band"] = lc["band"].map({b: i for i, b in enumerate(BAND_NAMES)}).astype(np.int8)
    return lc


RULES = {rule.NAME: rule, rule_v2.NAME: rule_v2}


def _signal(rng, family, t_min, t_max):
    if family == "ramp_brighten":
        sig = inj.sample(rng, "ramp_fade", t_min, t_max)
        sig.fraction = -sig.fraction
        return sig
    if family == "repeat_brighten":
        sig = inj.sample(rng, "repeat_long_dip", t_min, t_max)
        for p in sig.parts:
            p.covering = -p.covering
        return sig
    return inj.sample(rng, family, t_min, t_max, allow_held_out=family in inj.HELD_OUT_SLOW)


def _families(a):
    """(family, group, count) for the batch of each rule's blind test."""
    if a.rule == rule.NAME:  # slow-blind-1: its new shapes are spent since (inject.lightcurve.SPENT_SLOW)
        return [(f, "held_out", a.n_held_out) for f in inj.SPENT_SLOW] + \
            [(f, "known", a.n_known) for f in inj.SLOW_FAMILIES] + [("ramp_brighten", "brightening", a.n_bright)]
    if a.rule == rule_v2.NAME:  # slow-blind-2: its new shapes are spent since (inject.lightcurve.SPENT_PERIODIC)
        return [(f, "held_out", a.n_held_out) for f in inj.SPENT_PERIODIC] + \
            [(f, "known", a.n_known) for f in inj.PERIODIC_FAMILIES] + [("repeat_brighten", "brightening", a.n_bright)]
    return [(f, "held_out", a.n_held_out) for f in inj.HELD_OUT_SLOW] + \
        [(f, "known", a.n_known) for f in inj.PERIODIC_FAMILIES] + [("repeat_brighten", "brightening", a.n_bright)]


def _table(a):
    return RULES[a.rule].load()[-1]


def make(a):
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    if (d / "batch.parquet").exists():
        raise SystemExit("batch exists; a blind batch is drawn once")
    table = _table(a)
    rng = np.random.default_rng(int.from_bytes(os.urandom(16), "little"))

    names = sorted(p.name for p in (Path(a.run) / "lightcurves").glob("*.parquet"))
    feat = pd.concat([pq.read_table(Path(a.run) / "features" / n,
                                    columns=["source_id", "n_g", "sigma_rob_g", "err_med_g", "min_z_rob",
                                             "max_z_rob"]).to_pandas() for n in names], ignore_index=True)
    quiet = feat["source_id"].to_numpy()[drift.reference_mask(feat)]
    aux = pd.read_parquet(a.aux, columns=["source_id", "phot_g_mean_mag", "bp_rp"])
    cells = drift.star_cells(aux[aux["source_id"].isin(quiet)])
    quiet = cells.index.to_numpy()
    bright = rng.permutation(quiet[cells["gbin"].to_numpy() <= rule.FAINT_GBIN - 1])
    faint = rng.permutation(quiet[cells["gbin"].to_numpy() >= rule.FAINT_GBIN])

    fams = _families(a)
    n_hosts = sum(n for *_, n in fams)
    if a.n_nat + n_hosts > len(bright) or a.n_nat_faint > len(faint):
        raise SystemExit(f"not enough quiet stars: {len(bright)} at G < 19, {len(faint)} at G >= 19")
    plan_rows = [(int(s), "natural", "") for s in bright[:a.n_nat]] + [(int(s), "natural", "") for s in faint[:a.n_nat_faint]]
    hosts = bright[a.n_nat:a.n_nat + n_hosts]
    k = 0
    for f, g, n in fams:
        plan_rows += [(int(s), g, f) for s in hosts[k:k + n]]
        k += n

    _STATE.update(run=a.run, wanted={r[0] for r in plan_rows})
    with Pool(a.procs) as pool:  # forked workers inherit _STATE
        lcs = pd.concat(pool.map(_read, names, chunksize=4), ignore_index=True)
    by_star = dict(tuple(lcs.groupby("source_id", sort=False)))

    order = rng.permutation(len(plan_rows))
    out, truth, cell_rows = [], [], []
    for j, i in enumerate(order):
        sid, group, fam = plan_rows[i]
        lc = by_star.get(sid)
        if lc is None:
            continue
        anon = j + 1
        row = {"id": anon, "group": group, "family": fam, "star": sid}
        lc = lc.copy()
        if fam:
            tg = np.sort(lc.loc[lc["band"] == "G", "t"].to_numpy())
            sig = _signal(rng, fam, float(tg.min()), float(tg.max()))
            for b in BAND_NAMES:
                m = (lc["band"] == b).to_numpy()
                lc.loc[m, "flux"], lc.loc[m, "err"] = inj.inject(lc["t"].to_numpy()[m], lc["flux"].to_numpy()[m],
                                                               lc["err"].to_numpy()[m], b, sig)
            tv, _ = detect.visits(tg, tg)
            row["depth_g"], row["visits_in"] = depth_g(sig, tv)
            row["signal"] = inj.truth(sig)
        out.append(lc.assign(source_id=anon))
        truth.append(row)
        cell_rows.append((anon, *cells.loc[sid, ["gbin", "cbin"]]))
    batch = _band_codes(pd.concat(out, ignore_index=True))
    batch.to_parquet(d / "batch.parquet", index=False)
    pd.DataFrame(cell_rows, columns=["source_id", "gbin", "cbin"]).to_parquet(d / "cells.parquet", index=False)

    # The hosts' own noise, from the uninjected light curves, for the expected signal-to-noise at reveal.
    host_ids = [t["star"] for t in truth if t["family"]]
    host = scored(corrected(_band_codes(lcs[lcs["source_id"].isin(host_ids)].copy()), cells, table))
    host = host.set_index("source_id")["sigma_visit"]
    for t in truth:
        if t["family"]:
            t["host_sigma_visit"] = float(host.get(t["star"], np.nan))
    truth_sha = write_sealed(truth, d / "truth.json.age", RECIPIENTS)
    tt = pd.DataFrame(truth)
    tt["mag"] = rule.mag_group(cells.loc[tt["star"], "gbin"].to_numpy())
    plan = {"n": len(truth), "counts": {f"{g}/{f or '-'}/{m}": int(n)
                                        for (g, f, m), n in tt.groupby(["group", "family", "mag"]).size().items()},
            "run": str(a.run),
            "batch_sha256": _sha(d / "batch.parquet"), "cells_sha256": _sha(d / "cells.parquet"),
            "truth_sha256": truth_sha, "code_commit": _git("rev-parse", "HEAD"), "rule": a.rule,
            "rule_files": {p.name: _sha(p) for p in sorted(RULES[a.rule].RULE_DIR.iterdir())}}
    (d / "plan.json").write_text(json.dumps(plan, indent=1))
    print(json.dumps(plan, indent=1))


def run(a):
    d = Path(a.dir)
    rule_name = json.loads((d / "plan.json").read_text())["rule"]
    batch = pd.read_parquet(d / "batch.parquet")
    cells = pd.read_parquet(d / "cells.parquet").set_index("source_id")
    if rule_name == rule.NAME:
        thr, table = rule.load()
        stats = scored(corrected(batch, cells, table)).join(cells, on="source_id")
        flags = rule.apply(stats, thr)
    else:
        thr, thr_p, table = rule_v2.load()
        ep = corrected(batch, cells, table)
        stats = scored(ep).merge(period.star_features(ep), on="source_id", how="left").join(cells, on="source_id")
        flags = rule_v2.apply(stats, thr, thr_p)
    stats.to_parquet(d / "stats.parquet", index=False)
    missing = sorted(set(cells.index) - set(flags["id"]))
    if missing:  # too few visits to score: recorded as not flagged
        flags = pd.concat([flags, pd.DataFrame({"id": missing, "gbin": cells.loc[missing, "gbin"].to_numpy(),
                                                "flagged": False, "by": "unscored"})], ignore_index=True)
    flags.sort_values("id").to_csv(d / "flags.csv", index=False)
    print(flags["flagged"].value_counts().to_dict(), flags["by"].value_counts().to_dict())


def commit(a):
    d = Path(a.dir)
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked files are modified; commit the code first so code_commit names it")
    arch = ROOT / "prereg" / "verdicts" / a.batch
    arch.mkdir(parents=True, exist_ok=True)
    (arch / "flags.csv").write_bytes((d / "flags.csv").read_bytes())
    plan = json.loads((d / "plan.json").read_text())
    rec = {"batch": a.batch, "flags_sha256": _sha(arch / "flags.csv"), "plan": plan,
           "code_commit": _git("rev-parse", "HEAD"), "rule": plan["rule"]}
    (arch / "commit.json").write_text(json.dumps(rec, indent=1))
    _git("add", str(arch / "flags.csv"), str(arch / "commit.json"))
    _git("commit", "-m", f"Archive {a.batch} flags before reveal")
    print("archived at", _git("rev-parse", "HEAD"), "- push before revealing")


def reveal(a):
    d = Path(a.dir)
    arch = ROOT / "prereg" / "verdicts" / a.batch
    rec = json.loads((arch / "commit.json").read_text())
    _git("fetch", "--quiet", "origin")
    sha = _git("log", "-n", "1", "--format=%H", "--", str(arch / "commit.json"))
    if not [r for r in _git("branch", "-r", "--contains", sha).split() if r.startswith("origin/")]:
        raise SystemExit(f"archive commit {sha[:12]} is not on origin; push it first")
    if _sha(arch / "flags.csv") != rec["flags_sha256"] or _sha(d / "flags.csv") != rec["flags_sha256"]:
        raise SystemExit("flags do not match the archived hash")
    if _sha(d / "truth.json.age") != rec["plan"]["truth_sha256"]:
        raise SystemExit("sealed truth does not match the plan")
    from pyrage import decrypt, x25519

    ident = x25519.Identity.from_str(os.environ["COSMOS_REVEAL_KEY"].strip())
    truth = pd.DataFrame(json.loads(decrypt((d / "truth.json.age").read_bytes(), [ident])))
    flags = pd.read_csv(d / "flags.csv")
    m = flags.merge(truth, on="id")
    m["snr"] = m["depth_g"] * np.sqrt(np.maximum(m["visits_in"].fillna(1), 1)) / m["host_sigma_visit"]
    score = RULES[rec["rule"]].score(m)
    (d / "score.json").write_text(json.dumps(score, indent=1, default=str))
    m.drop(columns=["signal"]).to_parquet(d / "revealed.parquet")
    print(json.dumps(score["criteria"], indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["make", "run", "commit", "reveal"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--batch", default="slow-blind-1")
    ap.add_argument("--rule", default=rule.NAME, choices=sorted(RULES), help="make only; later steps read the plan")
    ap.add_argument("--run")
    ap.add_argument("--aux", default="/mnt/project-files/data/gaia-dr3-epoch/aux/aux_gaia_source.parquet")
    ap.add_argument("--n-nat", type=int, default=5000)
    ap.add_argument("--n-nat-faint", type=int, default=2000)
    ap.add_argument("--n-held-out", type=int, default=600)
    ap.add_argument("--n-known", type=int, default=300)
    ap.add_argument("--n-bright", type=int, default=300)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    {"make": make, "run": run, "commit": commit, "reveal": reveal}[a.step](a)


if __name__ == "__main__":
    main()
