"""Run the two-sided dimmer prefilter on a release and record exactly what ran.

Rule (frozen; see docs/infra/targets.md):
1. Histogram both tails of vari_summary per magnitude bin (two aggregate queries).
2. Per bin, threshold T_b = the smallest d at which the bright tail (the mirror null) keeps a fraction
   ≤ alpha of the bin. Bins with fewer than min_count sources use the pooled threshold.
3. Select every source whose faint *or* bright d is ≥ T_b, plus MOD(source_id, 4093) = 0 controls.

alpha is set from the budget: about 2 · alpha · N sources pass (both tails), plus any real faint excess.
"""

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import queries
from .releases import tables

RULE_VERSION = 1


@dataclass(frozen=True)
class Rule:
    alpha: float = 1.2e-3  # DR4: ~2 × 1.2e-3 × 4.0e8 ≈ 1e6 sources, the DataLink budget
    min_count: int = 2000
    min_transits: int = 10
    mag_bin: float = queries.MAG_BIN
    d_step: float = queries.D_STEP
    control_modulus: int = queries.CONTROL_MODULUS
    version: int = RULE_VERSION


def survival(hist, mag_bin=None):
    """(d_edges, S) where S[i] is the fraction with d >= d_edges[i], for one bin or pooled."""
    h = hist if mag_bin is None else hist[hist["mag_bin"] == mag_bin]
    d = np.asarray(h["d_bin"], dtype=float)
    n = np.asarray(h["n"], dtype=float)
    ok = np.isfinite(d)
    d, n = d[ok], n[ok]
    order = np.argsort(d)
    d, n = d[order], n[order]
    s = np.cumsum(n[::-1])[::-1] / n.sum()
    return d * queries.D_STEP, s


def threshold(hist, alpha, mag_bin=None):
    edges, s = survival(hist, mag_bin)
    passing = np.flatnonzero(s <= alpha)
    return float(edges[passing[0]]) if passing.size else float(edges[-1] + queries.D_STEP)


def thresholds(bright_hist, rule):
    pooled = threshold(bright_hist, rule.alpha)
    out = {}
    totals = {}
    for b, n in zip(bright_hist["mag_bin"], bright_hist["n"]):
        if b is not None and np.isfinite(b):
            totals[int(b)] = totals.get(int(b), 0) + int(n)
    for b, total in sorted(totals.items()):
        out[b] = threshold(bright_hist, rule.alpha, b) if total >= rule.min_count else pooled
    return out, pooled


def template_hashes():
    return {name: hashlib.sha256(q.template.encode()).hexdigest() for name, q in queries.REGISTRY.items()}


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(release, tap, out, rule=Rule(), extra_where=""):
    """Run the prefilter; write histograms, selection Parquet and a run manifest under out."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"release": release, "tables": tables(release), "rule": asdict(rule),
                "templates_sha256": template_hashes(), "git_commit": git_commit(),
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "queries": {}}
    hists = {}
    for side in ("faint", "bright"):
        adql = queries.render_histogram(release, side, min_transits=rule.min_transits, extra_where=extra_where)
        hists[side] = tap.query(adql)
        hists[side].write(out / f"histogram_{side}.ecsv", overwrite=True)
        manifest["queries"][f"histogram_{side}"] = {"adql": adql, "sha256": _sha(adql), "rows": len(hists[side])}
    thr, pooled = thresholds(hists["bright"], rule)
    manifest["thresholds"] = {str(k): v for k, v in thr.items()}
    manifest["pooled_threshold"] = pooled
    adql = queries.render_prefilter(release, thr, min_transits=rule.min_transits, extra_where=extra_where)
    sel = tap.query(adql)
    sel["control"] = sel["control_mod"] == 0
    sel.write(out / "prefilter.parquet", overwrite=True)
    manifest["queries"]["prefilter"] = {"adql": adql, "sha256": _sha(adql), "rows": len(sel)}
    manifest["selected"] = {
        "total": len(sel), "controls": int(sel["control"].sum()),
        "faint": int(np.sum(_over(sel, thr, "d_faint"))), "bright": int(np.sum(_over(sel, thr, "d_bright"))),
    }
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return sel, manifest


def _over(sel, thr, col):
    t = np.array([thr.get(int(b), np.inf) for b in sel["mag_bin"]])
    return np.asarray(sel[col], dtype=float) >= t


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def freeze(tag, path, rule=Rule()):
    """Write a freeze manifest: the rule, template hashes and code version, without data."""
    manifest = {"tag": tag, "rule": asdict(rule), "templates_sha256": template_hashes(),
                "git_commit": git_commit(), "frozen": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify(path):
    """Return a list of differences between a freeze manifest and the current code (empty = unchanged)."""
    frozen = json.loads(Path(path).read_text())
    diffs = []
    now = template_hashes()
    for name, h in frozen["templates_sha256"].items():
        if now.get(name) != h:
            diffs.append(f"template {name} changed")
    for name in set(now) - set(frozen["templates_sha256"]):
        diffs.append(f"template {name} added")
    if frozen["rule"] != asdict(Rule()):
        diffs.append(f"rule changed: {frozen['rule']} -> {asdict(Rule())}")
    return diffs
