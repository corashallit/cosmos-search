"""Sealed blind test of the astrometric rule (docs/arms/astrometry.md, "Blind test").

    python scripts/astrometry_blind.py make   --dir D     # draw the batch; truth sealed with age, seed from urandom
    python scripts/astrometry_blind.py run    --dir D     # cascade + the frozen rule -> D/flags.csv
    python scripts/astrometry_blind.py commit --dir D --batch astrometry-blind-1   # archive flags in git
    git push
    python scripts/astrometry_blind.py reveal --dir D --batch astrometry-blind-1   # refuses unless pushed

The batch mixes natural sources (the calibration mix), non-gravitational injections and instrument injections, in
shuffled order under anonymous ids. Nobody sees the truth until the flags are committed and on origin."""

import argparse
import hashlib
import json
import os
import pickle
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.astrometry import detect, rule
from cosmos_search.astrometry import simulate as S
from cosmos_search.inject.sealed import write_sealed

ROOT = Path(__file__).resolve().parents[1]
GOST = "/mnt/project-files/concurrent/gost_dr4.parquet"
STARS = "/mnt/project-files/concurrent/sample.parquet"
RECIPIENTS = [ROOT / "keys" / "cora.age.pub", ROOT / "keys" / "reveal.age.pub"]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()


def make(a):
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    if (d / "batch.pkl").exists():
        raise SystemExit("batch exists; a blind batch is drawn once")
    rng = np.random.default_rng(int.from_bytes(os.urandom(16), "little"))
    cad = S.Cadence(pd.read_parquet(GOST), pd.read_parquet(STARS))
    names, p = list(S.NATURAL), np.array(list(S.NATURAL.values()))
    kinds = list(rng.choice(names, size=a.n_nat, p=p / p.sum()))
    kinds += list(rng.choice(S.NONGRAV, size=a.n_ng)) + ["fov_jump"] * a.n_inst
    order = rng.permutation(len(kinds))
    batch, truth = [], []
    for j, i in enumerate(order):
        kind = kinds[i]
        amp = float(10 ** rng.uniform(np.log10(a.amp_lo), np.log10(a.amp_hi))) if kind in S.NONGRAV + S.INSTRUMENT \
            else None
        data, tr = S.simulate(rng, cad, kind, amp=amp)
        sid = f"A{j:04d}"
        batch.append({"id": sid, **data})
        truth.append({"id": sid, **{k: (v.item() if hasattr(v, "item") else v) for k, v in tr.items()}})
    with open(d / "batch.pkl", "wb") as fh:
        pickle.dump(batch, fh)
    truth_sha = write_sealed(truth, d / "truth.json.age", RECIPIENTS)
    plan = {"n": len(batch), "n_nat": a.n_nat, "n_nongrav": a.n_ng, "n_instrument": a.n_inst,
            "amp_range": [a.amp_lo, a.amp_hi], "batch_sha256": _sha(d / "batch.pkl"), "truth_sha256": truth_sha,
            "code_commit": _git("rev-parse", "HEAD"), "rule": rule.ADOPTED_NAME}
    (d / "plan.json").write_text(json.dumps(plan, indent=1))
    print(json.dumps(plan, indent=1))


def _one(src):
    try:
        out = detect.cascade(src)
    except Exception as e:  # noqa: BLE001 -- recorded, scored as not flagged
        out = {"error": repr(e)[:200]}
    return {"id": src["id"], **out}


def run(a):
    d = Path(a.dir)
    with open(d / "batch.pkl", "rb") as fh:
        batch = pickle.load(fh)
    with Pool(a.procs) as pool:
        rows = pool.map(_one, batch, chunksize=4)
    res = pd.DataFrame(rows)
    res.to_parquet(d / "results.parquet")
    flags = rule.apply(res)
    flags.to_csv(d / "flags.csv", index=False)
    print(flags["flagged"].value_counts().to_dict(), flags["label"].value_counts().to_dict())


def commit(a):
    d = Path(a.dir)
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked files are modified; commit the code first so code_commit names it")
    arch = ROOT / "prereg" / "verdicts" / a.batch
    arch.mkdir(parents=True, exist_ok=True)
    (arch / "flags.csv").write_bytes((d / "flags.csv").read_bytes())
    plan = json.loads((d / "plan.json").read_text())
    rec = {"batch": a.batch, "flags_sha256": _sha(arch / "flags.csv"), "plan": plan,
           "code_commit": _git("rev-parse", "HEAD"), "rule": rule.ADOPTED_NAME}
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
    score = rule.score(m)
    (d / "score.json").write_text(json.dumps(score, indent=1))
    m.to_parquet(d / "revealed.parquet")
    print(json.dumps(score, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["make", "run", "commit", "reveal"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--batch", default="astrometry-blind-1")
    ap.add_argument("--n-nat", type=int, default=400)
    ap.add_argument("--n-ng", type=int, default=160)
    ap.add_argument("--n-inst", type=int, default=40)
    ap.add_argument("--amp-lo", type=float, default=3.0)
    ap.add_argument("--amp-hi", type=float, default=50.0)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    {"make": make, "run": run, "commit": commit, "reveal": reveal}[a.step](a)


if __name__ == "__main__":
    main()
