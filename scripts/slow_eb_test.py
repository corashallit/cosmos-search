"""Test of the eclipsing-binary check on slow-change candidates (docs/arms/slow-changes.md, "Eclipsing binaries
among the candidates"; E1 written there before the draw).

    python scripts/slow_eb_test.py hosts --sample S --flags F --rule RULE --exclude D/fakes.parquet --out E
    python scripts/slow_eb_test.py fakes --sample S --rule RULE --out E
    python scripts/slow_eb_test.py real  --lists L --lightcurves R/lightcurves --checked C --out E
    python scripts/slow_eb_test.py report --out E

hosts: quiet stars at G < 19 of the trial's sky patches, their positions and parallaxes from gaia_source, and ZTF
epochs for those north of -30 deg until enough have >= 100. fakes: grey slow changes injected into Gaia and ZTF,
kept when the trial thresholds flag them at S/N >= 20, then `checks.eclipse`. real: the same check on DPAC
eclipsing binaries among the strongest fades and on the trial's checked non-eclipsing fades.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import slow_band_test as bt  # noqa: E402
import slow_cal  # noqa: E402
import slow_checks  # noqa: E402

from cosmos_search.dayone import external  # noqa: E402
from cosmos_search.inject import lightcurve as inj  # noqa: E402
from cosmos_search.slow import checks, detect, drift, rule_v3, search, vet  # noqa: E402

N_REPEAT, N_OTHER = 50, 50
OTHER_SHAPES = ("long_dip", "ramp_fade", "secular_fade", "staircase_fade", "accelerating_fade")
MIN_ZTF = 100
SEED = 20261031
SRC_COLS = "source_id, ra, dec, parallax, parallax_error, phot_g_mean_mag, bp_rp"
_S = {}


def _gaia(ids):
    from astropy.table import Table

    from cosmos_search.gaia.archive import tap

    d = tap().query(f"SELECT {', '.join('g.' + c.strip() for c in SRC_COLS.split(','))} FROM TAP_UPLOAD.ids AS u "
                    "JOIN gaiadr3.gaia_source AS g ON g.source_id = u.source_id",
                    uploads={"ids": Table({"source_id": np.asarray(ids, np.int64)})}).to_pandas()
    d.columns = [c.lower() for c in d.columns]
    return d


def cmd_hosts(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nb = pd.read_parquet(Path(a.rule) / "stats.parquet", columns=["source_id", "gbin", "sigma_visit"])
    q = pd.read_parquet(a.flags)
    used = set(pd.read_parquet(a.exclude)["host"]) if a.exclude else set()
    excl = Path(bt.EXCLUDE)
    used |= {int(x) for x in excl.read_text().split()} if excl.exists() else set()
    h = nb.merge(q, on="source_id")
    h = h[h["quiet"] & (h["gbin"] <= 7) & ~h["source_id"].isin(used)]
    h = h.sample(frac=1.0, random_state=SEED).head(a.n)
    g = _gaia(h["source_id"])
    h = h.merge(g, on="source_id")
    h = h[h["dec"] > -30].reset_index(drop=True)
    print(f"{len(h)} candidate hosts north of -30; fetching ZTF", flush=True)
    (out / "ztf").mkdir(exist_ok=True)

    def fetch(r):
        p = out / "ztf" / f"{int(r.source_id)}.parquet"
        if not p.exists():
            try:
                vet.ztf_raw(float(r.ra), float(r.dec)).to_parquet(p, index=False)
            except Exception as e:  # noqa: BLE001
                return int(r.source_id), -1, repr(e)[:100]
        return int(r.source_id), len(pd.read_parquet(p)), ""

    good, n = [], 0
    with ThreadPoolExecutor(4) as ex:
        for sid, k, err in ex.map(fetch, h.itertuples(index=False)):
            n += 1
            good.append({"source_id": sid, "n_ztf": k, "error": err})
            if n % 25 == 0:
                print(f"{n} fetched, {sum(x['n_ztf'] >= MIN_ZTF for x in good)} with >= {MIN_ZTF} epochs", flush=True)
    h = h.merge(pd.DataFrame(good), on="source_id")
    h.to_parquet(out / "hosts.parquet", index=False)
    print(f"{int((h.n_ztf >= MIN_ZTF).sum())} hosts with >= {MIN_ZTF} ZTF epochs")


def _init(cells, table, ztf_dir):
    _S.update(cells=cells, table=table, ztf_dir=ztf_dir)


def _eval_file(job):
    """Scores and the eclipse verdict of each trial fake whose host is in one patch file."""
    path, trials = job
    t = pq.read_table(path, filters=[("source_id", "in", [int(x["host"]) for x in trials])]).to_pandas()
    t["band"] = t["band"].astype(str)
    out = []
    for tr in trials:
        d = t[t["source_id"] == tr["host"]]
        tg = np.sort(d.loc[d["band"] == "G", "t"].to_numpy(float))
        if len(tg) < detect.MIN_VISITS:
            continue
        rng = np.random.default_rng(int(tr["seed"]))
        sig = inj.sample(rng, tr["shape"], float(tg.min()), float(tg.max()))
        if tr["shape"] == "repeat_long_dip":
            sig = slow_cal._extend(sig)
        rows = bt._inject(d, sig)
        s = search.score(rows, _S["cells"], _S["table"])
        if not len(s):
            continue
        ep = drift.normalise(rows)
        ep = ep[ep["band"] == 0]
        z = vet.ztf_normalised(pd.read_parquet(Path(_S["ztf_dir"]) / f"{tr['host']}.parquet"), sig)
        src = {k: tr[k] for k in ("phot_g_mean_mag", "bp_rp", "parallax", "parallax_error")}
        v = checks.eclipse(pd.DataFrame({"t": ep["t"], "flux": 1 + ep["frac"]}), z, src)
        tv, _ = detect.visits(tg, tg)
        out.append(s.assign(trial=tr["trial"], snr=vet.snr(sig, tv, float(tr["sigma_visit"])), **v))
    return pd.concat(out, ignore_index=True) if out else None


def cmd_fakes(a):
    sample, out = Path(a.sample), Path(a.out)
    _, _, table = rule_v3.load(a.rule)
    nb = pd.read_parquet(Path(a.rule) / "stats.parquet")
    aux = pd.read_parquet(sample / "aux.parquet")
    cells = drift.star_cells(aux)
    h = pd.read_parquet(out / "hosts.parquet")
    h = h[h["n_ztf"] >= MIN_ZTF].reset_index(drop=True)
    files = sorted((sample / "lightcurves").glob("*.parquet"))
    where = pd.concat([pd.DataFrame({"source_id": pq.read_table(f, columns=["source_id"])["source_id"].to_numpy(),
                                     "file": str(f)}).drop_duplicates("source_id") for f in files])
    h = h.merge(where, on="source_id")
    rng = np.random.default_rng(SEED)
    slots = ["repeat_long_dip"] * N_REPEAT + ["other"] * N_OTHER
    free = np.ones(len(h), bool)
    kept, n_trials = {}, 0
    with Pool(4, initializer=_init, initargs=(cells, table, str(out / "ztf"))) as pool:
        for rnd in range(30):
            trials = []
            for si, sl in enumerate(slots):
                if si in kept:
                    continue
                for i in rng.choice(np.flatnonzero(free), size=min(3, free.sum()), replace=False):
                    r = h.iloc[i]
                    trials.append({"trial": n_trials, "slot": si, "row": int(i), "host": int(r.source_id),
                                   "file": r.file, "sigma_visit": float(r.sigma_visit),
                                   "phot_g_mean_mag": float(r.phot_g_mean_mag), "bp_rp": float(r.bp_rp),
                                   "parallax": float(r.parallax), "parallax_error": float(r.parallax_error),
                                   "shape": sl if sl != "other" else str(rng.choice(OTHER_SHAPES)),
                                   "seed": int(rng.integers(2 ** 62))})
                    n_trials += 1
            if not trials:
                break
            by_file = {}
            for tr in trials:
                by_file.setdefault(tr["file"], []).append(tr)
            parts = [x for x in pool.imap_unordered(_eval_file, list(by_file.items())) if x is not None]
            if parts:
                st = pd.concat(parts, ignore_index=True).join(cells, on="source_id")
                pix = drift.sky_pixel(st["source_id"].to_numpy(), rule_v3.SHARED_LEVEL)
                sh = rule_v3.shared(nb, targets=st, target_patch=pix)
                f = search.flag(st, a.rule, shared_flags=sh.reset_index(drop=True))
                info = {tr["trial"]: tr for tr in trials}
                for k, r in enumerate(f.itertuples(index=False)):
                    tr = info[int(st["trial"].iloc[k])]
                    if tr["slot"] in kept or not free[tr["row"]] or not r.flagged or not st["snr"].iloc[k] >= 20:
                        continue
                    kept[tr["slot"]] = {**tr, "by": r.by, "strength": r.strength, "strength_by": r.strength_by,
                                        "snr": float(st["snr"].iloc[k]),
                                        **{c: st[c].iloc[k] for c in st.columns if c.startswith("eb_")}}
                    free[tr["row"]] = False
            print(f"round {rnd + 1}: {len(trials)} draws, {len(kept)} of {len(slots)} kept, "
                  f"{int(free.sum())} hosts free", flush=True)
    res = pd.DataFrame(kept.values())
    res.to_parquet(out / "fakes.parquet", index=False)
    print(res.groupby("shape")[["eb_explained", "eb_periodic_unphysical"]].agg(["sum", "size"]))


def cmd_real(a):
    out = Path(a.out)
    fad = pd.read_parquet(Path(a.lists) / "fading.parquet")
    cls = pd.read_parquet(bt.CLASSES).rename(columns={"source_id": "id"})[["id", "best_class_name"]]
    fad = fad[~fad["still"]].merge(cls, on="id", how="left")
    chk = pd.read_csv(Path(a.checked) / "checks.csv")
    ecl = fad[(fad["best_class_name"] == "ECL") & (fad["phot_g_mean_mag"] < 17)].sort_values("strength",
                                                                                          ascending=False)
    g = _gaia(np.r_[ecl["id"].head(200).to_numpy(), chk["source_id"].to_numpy()])
    ecl = ecl.merge(g.rename(columns={"source_id": "id"}).drop(columns=["phot_g_mean_mag", "bp_rp"]), on="id")
    ecl = ecl[ecl["dec"] > -30].head(60)
    ztf_known = pd.read_parquet(Path(a.checked) / "ztf.parquet")

    def ztf_of(r):
        if r.id in set(ztf_known["source_id"]):
            return ztf_known[ztf_known["source_id"] == r.id]
        if r.dec <= -30:
            return external._empty()
        return external.ztf(r.id, r.ra, r.dec)

    stars = pd.concat([ecl.assign(set="DPAC eclipsing binary"),
                       chk.rename(columns={"source_id": "id"})[["id"]].merge(
                           g.rename(columns={"source_id": "id"}), on="id").assign(set="checked, not DPAC EB")],
                      ignore_index=True)
    gep = slow_checks.g_epochs(a.lightcurves, stars["id"])
    g_of = dict(tuple(gep.groupby("source_id")))

    def one(r):
        z = ztf_of(r)
        src = {k: getattr(r, k) for k in ("phot_g_mean_mag", "bp_rp", "parallax", "parallax_error")}
        v = checks.eclipse(g_of.get(r.id, pd.DataFrame(columns=["t", "flux"])), z, src)
        return {"id": r.id, "set": r.set, "dec": r.dec, "n_ztf": len(z), **v}

    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(one, stars.itertuples(index=False)))
    res = pd.DataFrame(rows)
    res.to_parquet(out / "real.parquet", index=False)
    print(res.groupby("set")[["eb_explained", "eb_periodic_unphysical"]].agg(["sum", "size"]))


def cmd_report(a):
    out = Path(a.out)
    fk = pd.read_parquet(out / "fakes.parquet")
    rep = {"E1_fakes_explained": bt._cp(fk["eb_explained"].astype(bool).sum(), len(fk))}
    rep["E1_pass"] = rep["E1_fakes_explained"]["upper_95"] <= 0.05
    rep["fakes_by_shape"] = {s: {"n": int(len(d)), "explained": int(d["eb_explained"].astype(bool).sum()),
                                 "periodic_unphysical": int(d["eb_periodic_unphysical"].astype(bool).sum())}
                             for s, d in fk.groupby("shape")}
    if (out / "real.parquet").exists():
        r = pd.read_parquet(out / "real.parquet")
        rep["real"] = {s: {"n": int(len(d)), "with_ztf": int((d["n_ztf"] > 0).sum()),
                           "explained": int(d["eb_explained"].astype(bool).sum()),
                           "periodic_unphysical": int(d["eb_periodic_unphysical"].astype(bool).sum())}
                       for s, d in r.groupby("set")}
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hosts")
    h.add_argument("--sample", required=True)
    h.add_argument("--flags", required=True)
    h.add_argument("--rule", required=True)
    h.add_argument("--exclude", help="fakes.parquet of the G-against-BP-and-RP test (its hosts are not reused)")
    h.add_argument("--n", type=int, default=700, help="candidate hosts looked up")
    h.add_argument("--out", required=True)
    h.set_defaults(func=cmd_hosts)
    f = sub.add_parser("fakes")
    f.add_argument("--sample", required=True)
    f.add_argument("--rule", required=True)
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_fakes)
    r = sub.add_parser("real")
    r.add_argument("--lists", required=True)
    r.add_argument("--lightcurves", required=True)
    r.add_argument("--checked", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_real)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_report)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
