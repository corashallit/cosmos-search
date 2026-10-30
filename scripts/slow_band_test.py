"""Test of the G-against-BP-and-RP check on fresh fakes and on DR3's natural fades (docs/arms/slow-changes.md,
"G against BP and RP"; groups and pass marks written there before the draw).

    python scripts/slow_band_test.py fakes   --sample S --rule RULE --out D [--workers 4]
    python scripts/slow_band_test.py natural --run R --lists L --rule RULE --out D
    python scripts/slow_band_test.py report  --out D

S: the trial's random sky patches (`slow_search.py sample`), whose quiet stars are the hosts and whose other stars
are their neighbours for the shared-date test; RULE: the thresholds and drift table set from them; R and L: a
scored run (kept light curves) and its lists, for the natural fades.
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import slow_search  # noqa: E402

from cosmos_search.inject import lightcurve as inj  # noqa: E402
from cosmos_search.slow import checks, detect, drift, rule_v3, search, vet  # noqa: E402
from cosmos_search.vetting import power  # noqa: E402

AUX_DR3 = "/mnt/project-files/data/gaia-dr3-epoch/aux/aux_gaia_source.parquet"
BLEND = "/mnt/project-files/data/gaia-dr3-epoch/aux/aux_blend.parquet"
EXCLUDE = "/mnt/project-files/dayone/rehearsal-inputs/exclude/all_used_ids.txt"
CLASSES = "/home/user/trial/classes/classes.parquet"
SHAPES = ("long_dip", "ramp_fade", "secular_fade", "staircase_fade", "accelerating_fade", "repeat_long_dip")
REAL = ("grey", "dust", "reemit", "temperature")
# (group, flavour, number, hosts, lowest and highest S/N), as written in the doc before the draw
GROUPS = [(f, f, 75, "any", 20, np.inf) for f in REAL] + [
    ("grey_low", "grey", 100, "any", 6, 20),
    ("g_only", "g_only", 100, "unblended", 20, np.inf),
    ("g_only_blended", "g_only", 50, "blended", 20, np.inf),
    ("neighbour", "neighbour", 50, "any", 20, np.inf)]
TRIES = 4            # candidate (host, shape) draws per open slot in each round
SEED = 20261030

_S = {}


def _init(cells, table):
    _S.update(cells=cells, table=table)


def signal(tr, t_min, t_max, bp_rp):
    """The fake's signal: the shape drawn with its seed, carried into each band by the flavour's ratios. A
    neighbour's change is the base in BP and RP and a fraction r of it in G."""
    rng = np.random.default_rng(int(tr["seed"]))
    base = inj.sample(rng, tr["shape"], t_min, t_max)
    if tr["flavour"] == "neighbour":
        return inj.BandScaled(base, {"G": float(rng.uniform(0.1, 0.6)), "BP": 1.0, "RP": 1.0})
    beta, t_shell = float(rng.uniform(1.0, 2.5)), float(rng.choice(vet.SHELL_TEMPS_K))
    ratios = vet.ratios(tr["flavour"], vet.teff(bp_rp), beta, t_shell)
    return inj.BandScaled(base, ratios, default=0.0 if tr["flavour"] == "g_only" else 1.0)


def _inject(df, sig):
    df = df.copy()
    flux, err = df["flux"].to_numpy(float).copy(), df["err"].to_numpy(float).copy()
    for b in ("G", "BP", "RP"):
        m = (df["band"] == b).to_numpy()
        flux[m], err[m] = inj.inject(df["t"].to_numpy(float)[m], flux[m], err[m], b, sig)
    df["flux"], df["err"] = flux, err
    df["band"] = df["band"].map({"G": 0, "BP": 1, "RP": 2}).astype(np.int8)
    return df


def _eval_file(job):
    """Scores of each trial fake whose host is in one patch file."""
    path, trials = job
    t = pq.read_table(path, filters=[("source_id", "in", [int(x["host"]) for x in trials])]).to_pandas()
    t["band"] = t["band"].astype(str)
    out = []
    for tr in trials:
        d = t[t["source_id"] == tr["host"]]
        tg = np.sort(d.loc[d["band"] == "G", "t"].to_numpy(float))
        if len(tg) < detect.MIN_VISITS:
            continue
        sig = signal(tr, float(tg.min()), float(tg.max()), tr["bp_rp"])
        s = search.score(_inject(d, sig), _S["cells"], _S["table"])
        if not len(s):
            continue
        tv, _ = detect.visits(tg, tg)
        out.append(s.assign(trial=tr["trial"], snr=vet.snr(sig, tv, float(tr["sigma_visit"]))))
    return pd.concat(out, ignore_index=True) if out else None


def cmd_fakes(a):
    sample, out = Path(a.sample), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    thr, thr_p, table = rule_v3.load(a.rule)
    nb = pd.read_parquet(Path(a.rule) / "stats.parquet")          # every sample star: the hosts' neighbours
    aux = pd.read_parquet(sample / "aux.parquet")
    cells = drift.star_cells(aux)
    q = pd.read_parquet(a.flags)
    files = sorted((sample / "lightcurves").glob("*.parquet"))
    where = pd.concat([pd.DataFrame({"source_id": pq.read_table(f, columns=["source_id"])["source_id"].to_numpy(),
                                     "file": str(f)}).drop_duplicates("source_id") for f in files])
    bl = pd.read_parquet(BLEND)
    bl = bl[bl["source_id"].isin(nb["source_id"])]
    with np.errstate(invalid="ignore", divide="ignore"):
        bl["blend"] = np.fmax(bl["phot_bp_n_blended_transits"] / bl["phot_bp_n_obs"],
                              bl["phot_rp_n_blended_transits"] / bl["phot_rp_n_obs"])
    used = set(int(x) for x in Path(EXCLUDE).read_text().split()) if Path(EXCLUDE).exists() else set()
    hosts = (nb[["source_id", "gbin", "sigma_visit"]].merge(q, on="source_id").merge(aux, on="source_id")
             .merge(bl[["source_id", "blend"]], on="source_id", how="left").merge(where, on="source_id"))
    hosts = hosts[hosts["quiet"] & (hosts["gbin"] <= 7) & ~hosts["source_id"].isin(used)].reset_index(drop=True)
    print(f"{len(hosts)} quiet hosts at G < 19 ({int(hosts.gaps.sum())} in the M31 field); "
          f"unblended {int((hosts.blend < checks.BLEND_MAX).sum())}", flush=True)
    rng = np.random.default_rng(SEED)
    free = np.ones(len(hosts), bool)
    slots = [{"group": g, "flavour": f, "hosts": h, "lo": lo, "hi": hi, "m31": k % 2 == 0}
             for g, f, n, h, lo, hi in GROUPS for k in range(n)]
    kept, n_trials, rounds = {}, 0, 0
    with Pool(a.workers, initializer=_init, initargs=(cells, table)) as pool:
        while len(kept) < len(slots) and rounds < 40:
            rounds += 1
            trials = []
            for si, sl in enumerate(slots):
                if si in kept:
                    continue
                ok = free & (hosts["gaps"].to_numpy() == sl["m31"])
                b = hosts["blend"].to_numpy()
                if sl["hosts"] == "unblended":
                    ok &= b < checks.BLEND_MAX
                elif sl["hosts"] == "blended":
                    ok &= b >= checks.BLEND_MAX
                for i in rng.choice(np.flatnonzero(ok), size=min(TRIES, ok.sum()), replace=False):
                    h = hosts.iloc[i]
                    trials.append({"trial": n_trials, "slot": si, "row": int(i), "host": int(h.source_id),
                                   "file": h.file, "bp_rp": float(h.bp_rp), "sigma_visit": float(h.sigma_visit),
                                   "flavour": sl["flavour"], "shape": str(rng.choice(SHAPES)),
                                   "seed": int(rng.integers(2 ** 62))})
                    n_trials += 1
            by_file = {}
            for tr in trials:
                by_file.setdefault(tr["file"], []).append(tr)
            parts = [x for x in pool.imap_unordered(_eval_file, list(by_file.items())) if x is not None]
            if not parts:
                continue
            st = pd.concat(parts, ignore_index=True).join(cells, on="source_id")
            pix = drift.sky_pixel(st["source_id"].to_numpy(), rule_v3.SHARED_LEVEL)
            sh = rule_v3.shared(nb, targets=st, target_patch=pix)
            f = search.flag(st, a.rule, shared_flags=sh.reset_index(drop=True))
            f["trial"], f["snr"] = st["trial"].to_numpy(), st["snr"].to_numpy()
            info = {tr["trial"]: tr for tr in trials}
            for r in f.itertuples(index=False):
                tr = info[r.trial]
                sl = slots[tr["slot"]]
                if tr["slot"] in kept or not free[tr["row"]] or not r.flagged or not (sl["lo"] <= r.snr < sl["hi"]):
                    continue
                kept[tr["slot"]] = {**tr, **sl, **r._asdict()}
                free[tr["row"]] = False
            print(f"round {rounds}: {len(trials)} draws, {len(kept)} of {len(slots)} slots filled", flush=True)
    res = pd.DataFrame(kept.values())
    res = res.merge(hosts[["source_id", "blend", "gaps", "phot_g_mean_mag"]].rename(columns={"source_id": "host"}),
                    on="host", how="left")
    res = pd.concat([res, checks.band_check(res, res["blend"])], axis=1)
    res.to_parquet(out / "fakes.parquet", index=False)
    print(res.groupby("group")["band_verdict"].value_counts().unstack(fill_value=0))


def cmd_natural(a):
    """Band verdicts of DR3's natural fades: 3,000 at random from the fading list and its 2,000 strongest (no
    still stars), rescored from the kept light curves with the current detector and period search."""
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fad = pd.read_parquet(Path(a.lists) / "fading.parquet", columns=["id", "strength", "still", "gbin"])
    fad = fad[~fad["still"]]
    rng = np.random.default_rng(SEED + 1)
    pick = pd.concat([fad.sample(3000, random_state=rng.integers(2 ** 31)).assign(set="random"),
                      fad.nlargest(2000, "strength").assign(set="strongest")])
    _, _, table = rule_v3.load(a.rule)
    aux = pd.read_parquet(AUX_DR3, columns=["source_id", "phot_g_mean_mag", "bp_rp"])
    aux = aux[aux["source_id"].isin(pick["id"])]
    cells = drift.star_cells(aux)
    stats = sorted((run / "stats").glob("*.parquet"))
    ranges = slow_search._id_ranges([str(p) for p in stats])
    ids = np.sort(pick["id"].unique())
    rows = []
    for p, (lo, hi) in zip(stats, ranges):
        mine = ids[(ids >= lo) & (ids <= hi)]
        lcf = run / "lightcurves" / p.name
        if not len(mine) or not lcf.exists():
            continue
        lc = pq.read_table(lcf, filters=[("source_id", "in", [int(x) for x in mine])])
        s = search.score(lc, cells, table)
        if len(s):
            rows.append(s)
    st = pd.concat(rows, ignore_index=True).join(cells, on="source_id")
    none = pd.DataFrame({"shared_fade": np.zeros(len(st), bool), "shared_rise": np.zeros(len(st), bool)})
    f = search.flag(st, a.rule, shared_flags=none)
    bl = pd.read_parquet(BLEND)
    bl = bl[bl["source_id"].isin(f["id"])]
    with np.errstate(invalid="ignore", divide="ignore"):
        bl["blend"] = np.fmax(bl["phot_bp_n_blended_transits"] / bl["phot_bp_n_obs"],
                              bl["phot_rp_n_blended_transits"] / bl["phot_rp_n_obs"])
    f = f.merge(bl[["source_id", "blend"]].rename(columns={"source_id": "id"}), on="id", how="left")
    f = pd.concat([f, checks.band_check(f, f["blend"])], axis=1)
    cls = pd.read_parquet(CLASSES).rename(columns={"source_id": "id"})[["id", "best_class_name"]]
    f = f.merge(cls, on="id", how="left").merge(pick[["id", "set"]], on="id")
    f.to_parquet(out / "natural.parquet", index=False)
    print(f.groupby("set")["band_verdict"].value_counts().unstack(fill_value=0))


def _cp(k, n):
    g = lambda x: float(np.asarray(x).ravel()[0])  # noqa: E731
    return {"k": int(k), "n": int(n), "rate": k / n if n else float("nan"),
            "lower_95": g(power.cp_lower(k, n)) if n else float("nan"),
            "upper_95": g(power.cp_upper(k, n)) if n else float("nan")}


def cmd_report(a):
    out = Path(a.out)
    fk = pd.read_parquet(out / "fakes.parquet")
    cut = fk["band_verdict"].isin(["g_only", "bp_rp_only"])
    real = fk["group"].isin(REAL)
    go = fk["group"] == "g_only"
    rep = {"B1_real_cut": _cp(cut[real].sum(), real.sum()),
           "B2_g_only_marked": _cp((fk.loc[go, "band_verdict"] == "g_only").sum(), go.sum())}
    rep["B1_pass"] = rep["B1_real_cut"]["upper_95"] <= 0.02
    rep["B2_pass"] = rep["B2_g_only_marked"]["lower_95"] >= 0.30
    rep["verdicts_by_group"] = {g: d["band_verdict"].value_counts().to_dict() for g, d in fk.groupby("group")}
    for key in ("shape", "strength_by", "gaps"):
        rep[f"by_{key}"] = {f"{g} {k}": d["band_verdict"].value_counts().to_dict()
                            for (g, k), d in fk.groupby(["group", key])}
    rep["by_magnitude"] = {f"{g} G{'<17' if b else '>=17'}": d["band_verdict"].value_counts().to_dict()
                           for (g, b), d in fk.groupby(["group", fk["phot_g_mean_mag"] < 17])}
    nat = out / "natural.parquet"
    if nat.exists():
        n = pd.read_parquet(nat)
        n["cls"] = n["best_class_name"].fillna("none")
        rep["natural"] = {s: d["band_verdict"].value_counts().to_dict() for s, d in n.groupby("set")}
        rep["natural_by_class"] = {f"{s} {c}": d["band_verdict"].value_counts().to_dict()
                                   for (s, c), d in n.groupby(["set", "cls"]) if len(d) >= 20}
        rep["natural_by_score"] = {f"{s} {c}": d["band_verdict"].value_counts().to_dict()
                                   for (s, c), d in n.groupby(["set", "strength_by"])}
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fakes")
    f.add_argument("--sample", required=True)
    f.add_argument("--flags", required=True, help="Parquet with source_id, quiet, gaps (M31 field) per sample star")
    f.add_argument("--rule", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--workers", type=int, default=4)
    f.set_defaults(func=cmd_fakes)
    n = sub.add_parser("natural")
    n.add_argument("--run", required=True)
    n.add_argument("--lists", required=True)
    n.add_argument("--rule", required=True)
    n.add_argument("--out", required=True)
    n.set_defaults(func=cmd_natural)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_report)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
