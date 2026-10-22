"""A control pool no rule has been tuned on, with the variable classes that will crowd the DR4 queue (review R3).

Batches 2-5 reused the same 14 controls and 16 real packets, and those stars were also in the recurrence
calibration, so every control false-flag rate quoted so far is in-sample. The controls were also nearly all
eclipsing binaries, while the DR4 queue will be dominated by spotted and other active stars. This module draws a
fresh pool, stratified by DPAC class, from stars no batch, calibration or review has touched, and reserves it:

- the pool (source_id, stratum, known-answer label) is sealed with age to the reveal key and Cora's key, so the
  list with its strata and labels is not something one reads by accident;
- sha256 hashes of its ids with a fixed, public salt are in the repo, so any calibration or decoy-host draw can
  leave the pool out without the key (`reserved`); dayone.decoys.hosts does.

Membership is therefore public by design: anyone can hash a list of candidate ids (the ranked list, say) and see
which are in the pool. The salt only stops a hash from matching other tables of Gaia ids. What keeps the pool
untouched is procedure, not secrecy: nothing may be tuned or calibrated on a reserved star, every draw filters
with `reserved`, and the confirmatory batch reports only from this pool (review finding R19).

The pool is used once, for the confirmatory batch that decides the D2 rule (docs/dayone.md). Control rates are
quoted only from it; numbers measured on the old controls are in-sample.

    python -m cosmos_search.dayone.controls draw --ranked rank/ranked.parquet --exclude rank/queue*.parquet ... \\
        --name dayone-controls-v1
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "prereg" / "controls"
KEYS = (ROOT / "keys" / "reveal.age.pub", ROOT / "keys" / "cora.age.pub")
SALT = "cosmos-search-controls:"

# DPAC class -> the known-answer label Stage A should give. The first three are the activity classes the review
# found the recurrence flag fires on (R1); the rest are the classes already in the queue's tiers 2 and 3.
STRATA = {
    "SOLAR_LIKE": "rotation_or_spots",
    "RS": "rotation_or_spots",
    "ACV|CP|MCP|ROAM|ROAP|SXARI": "rotation_or_spots",
    "ECL": "eclipsing_binary",
    "LPV": "pulsation_or_irregular",
    "DSCT|GDOR|SXPHE": "pulsation_or_irregular",
    "YSO": "yso_dipper",
    "BE|GCAS|SDOR|WR": "eruptive_or_be",
    "AGN": "extragalactic",
}
N_PER_STRATUM = 15
G_RANGE = (10.0, 15.5)         # the real candidates' range in batches 2-4 (G 7.7-14.8, median 12.5)
G_RANGE_BY_STRATUM = {"AGN": (14.0, 18.0)}     # hardly any AGN are brighter; DR4's will be fainter still
RANDOM_FRACTION = {"AGN": 1.0, "ACV|CP|MCP|ROAM|ROAP|SXARI": 1.0, "BE|GCAS|SDOR|WR": 0.5, "YSO": 0.3}


def _hash(source_id):
    return hashlib.sha256(f"{SALT}{int(source_id)}".encode()).hexdigest()


def reserved(source_ids, pool_dir=None):
    """Boolean mask: which source ids are in any reserved control pool."""
    hashes = set()
    for f in Path(pool_dir or PREREG).glob("*.sha256"):
        hashes |= {h.strip() for h in f.read_text().splitlines() if h.strip() and not h.startswith("#")}
    return np.array([_hash(s) in hashes for s in source_ids], dtype=bool)


def draw(ranked, exclude, rng, extra=None, n=N_PER_STRATUM, g_range=G_RANGE):
    """n stars per stratum: from the ranked list (G3-selected, as candidates are) where it has enough, otherwise
    topped up from `extra` (a DPAC-classified sample not selected by G3; marked selected=False)."""
    excl = set(int(x) for x in exclude)
    rows = []
    pools = [(ranked, True)] + ([(extra, False)] if extra is not None else [])
    for cls, label in STRATA.items():
        got = []
        for df, selected in pools:
            need = n - len(got)
            if need <= 0:
                break
            g = df["phot_g_mean_mag"].astype(float)
            sid = df["source_id"].astype("int64")
            lo, hi = G_RANGE_BY_STRATUM.get(cls, g_range)
            ok = df[(df["best_class_name"] == cls) & g.between(lo, hi) & ~sid.isin(excl) & ~reserved(sid)]
            ok = ok[~ok["source_id"].astype("int64").isin([r["source_id"] for r in got])]
            pick = ok.iloc[rng.permutation(len(ok))[:need]]
            got += [{"source_id": int(s), "stratum": cls, "label": label, "selected": selected,
                     "tier": int(t) if pd.notna(t) else None}
                    for s, t in zip(pick["source_id"], pick["tier"] if "tier" in pick else [None] * len(pick))]
        rows += got
    return rows


def queue_sample(ranked, n, rng, exclude=(), tier=1):
    """The confirmatory batch's queue arm: n untouched stars of the frozen ranking's top tier, drawn at build time
    (rng from os.urandom), never in a reserved pool or used before. Unlabelled: their shortlist rate is the load
    the rule puts on the DR4 expert queue, an upper bound on its false-positive rate."""
    sid = ranked["source_id"].astype("int64")
    ok = ranked[(ranked["tier"] == tier) & ~sid.isin(set(int(x) for x in exclude)) & ~reserved(sid)]
    return ok.iloc[rng.permutation(len(ok))[:n]].assign(role="queue_sample").reset_index(drop=True)


def extra_sample(tap, classes, per_class=200, g_range=G_RANGE):
    """DPAC-classified stars of the given classes from the archive, for strata the ranked list lacks. One query
    per class over a random_index window sized to the class's rarity, so TOP does not just return the first
    rows in storage order."""
    parts = []
    for cls in classes:
        lo, hi = G_RANGE_BY_STRATUM.get(cls, g_range)
        frac = RANDOM_FRACTION.get(cls, 0.1)
        adql = (f"SELECT TOP {int(per_class)} c.source_id, c.best_class_name, s.phot_g_mean_mag "
                "FROM gaiadr3.vari_classifier_result AS c JOIN gaiadr3.gaia_source AS s ON s.source_id = c.source_id "
                f"WHERE c.best_class_name = '{cls}' AND s.random_index < {int(frac * 1_811_709_771)} "
                f"AND s.phot_g_mean_mag BETWEEN {lo} AND {hi}")
        parts.append(tap.query(adql).to_pandas())
    return pd.concat(parts, ignore_index=True)


def seal(rows, name, pool_dir=PREREG, recipients=KEYS):
    """Write <name>.json.age (sealed rows) and <name>.sha256 (public salted id hashes). Refuses to overwrite."""
    from cosmos_search.inject.sealed import write_sealed

    pool_dir = Path(pool_dir)
    sealed, hashes = pool_dir / f"{name}.json.age", pool_dir / f"{name}.sha256"
    if sealed.exists() or hashes.exists():
        raise SystemExit(f"{name} exists; a reserved pool is drawn once")
    sha = write_sealed(rows, sealed, [str(r) for r in recipients])
    hashes.write_text(f"# sha256('{SALT}' + source_id) for each star in {name}.json.age; see dayone/controls.py\n"
                      + "\n".join(sorted(_hash(r["source_id"]) for r in rows)) + "\n")
    counts = pd.Series([r["stratum"] for r in rows]).value_counts().to_dict()
    return {"name": name, "n": len(rows), "per_stratum": counts, "sealed_sha256": sha,
            "selected": int(sum(r["selected"] for r in rows))}


def load(name, pool_dir=PREREG):
    """Decrypt a pool (needs COSMOS_REVEAL_KEY). Only the confirmatory batch build should call this."""
    from pyrage import decrypt, x25519

    secret = os.environ.get("COSMOS_REVEAL_KEY", "").strip()
    if not secret:
        raise SystemExit("the control pool is sealed; it needs COSMOS_REVEAL_KEY")
    raw = decrypt((Path(pool_dir) / f"{name}.json.age").read_bytes(), [x25519.Identity.from_str(secret)])
    return json.loads(raw)


def _ids(paths):
    out = set()
    for p in paths:
        p = Path(p)
        if p.suffix == ".parquet":
            out |= set(pd.read_parquet(p, columns=["source_id"])["source_id"].astype("int64"))
        elif p.suffix == ".csv":
            out |= set(pd.read_csv(p, usecols=["source_id"])["source_id"].astype("int64"))
        else:
            out |= {int(x) for x in p.read_text().split() if x.strip().isdigit()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.dayone.controls")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--ranked", required=True)
    d.add_argument("--exclude", nargs="*", default=[],
                   help="parquet/csv with source_id, or text files of ids: every star already used anywhere")
    d.add_argument("--exclude-plan", nargs="*", default=[], help="sealed plans whose hosts are excluded")
    d.add_argument("--name", required=True)
    d.add_argument("--n", type=int, default=N_PER_STRATUM)
    a = ap.parse_args(argv)
    from cosmos_search.gaia.archive import tap
    from cosmos_search.inject.classes.gaia_stream import load_plan

    excl = _ids(a.exclude)
    for p in a.exclude_plan:
        excl |= {int(k) for k in load_plan(p)}
    ranked = pd.read_parquet(a.ranked, columns=["source_id", "best_class_name", "phot_g_mean_mag", "tier"])
    have = ranked[ranked["phot_g_mean_mag"].between(*G_RANGE)].groupby("best_class_name").size()
    short = [c for c in STRATA if have.get(c, 0) < 3 * a.n]
    extra = extra_sample(tap(), short) if short else None
    rng = np.random.default_rng(int.from_bytes(os.urandom(16), "little"))
    rows = draw(ranked, excl, rng, extra, a.n)
    print(json.dumps({**seal(rows, a.name), "excluded": len(excl), "topped_up_from_archive": short}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
