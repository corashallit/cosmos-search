"""Calibration run (V3) of the candidate checks and the blind review on slow-change candidates (docs/vetting.md §14).

    python scripts/slow_cal.py score-shard --run /home/user/cal/dr3e --frame frame.txt --out flag/     # every star
    python scripts/slow_cal.py draw  --run /home/user/cal/dr3e --stats flag/stats.parquet --dir D        # sealed plan
    python scripts/slow_cal.py build --run /home/user/cal/dr3e --stats flag/stats.parquet --dir D        # packets
    # Stage A: one blind subagent per D/prompts/<packet_id>.md writes D/replies/<packet_id>.txt
    python -m cosmos_search.vetting commit --packets D/products.json --replies D/replies --vetter <label> \\
        --batch D/batch --archive prereg/verdicts/slow-cal-1
    git push
    python scripts/slow_cal.py score --dir D                                                             # reveal

score-shard runs rule slow-v3 (frozen files in prereg/rules/slow-v3/) over a dr3_pass output that kept every light
curve: the detector for every star, the period search for every star of the sampling frame (a list of bulk-file
names) and for quiet stars everywhere, then the shared-date test. draw picks the natural stars (DPAC class strata,
Neyman allocation) from the frame, every flagged quiet star, and the fakes (each in its own quiet host outside the
frame, kept when slow-v3 flags it at the slot's signal-to-noise), and seals the fake plan to Cora's key and the
reveal key. build fetches NEOWISE, ZTF and the infrared context, injects each fake's signal into its host's Gaia,
ZTF and NEOWISE epochs, and writes the packets (`slow.vet`), the prompts and the sealed key. score opens the key
only after the verdicts are committed and on origin.
"""

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cosmos_search.inject import lightcurve as inj
from cosmos_search.inject.sealed import write_sealed
from cosmos_search.slow import checks, detect, drift, period, release, rule_v3, vet
from cosmos_search.vetting import design
from cosmos_search.vetting import packet as packet_mod
from cosmos_search.vetting.__main__ import prompt

ROOT = Path(__file__).resolve().parents[1]
RECIPIENTS = [ROOT / "keys" / "cora.age.pub", ROOT / "keys" / "reveal.age.pub"]
AUX = "/mnt/project-files/data/gaia-dr3-epoch/aux/aux_gaia_source.parquet"
EXCLUDE = "/mnt/project-files/dayone/rehearsal-inputs/exclude/all_used_ids.txt"
BAND_NAMES = ("G", "BP", "RP")
N_NATURAL = 280
FLOOR = 10
MIN_STRATUM = 10
RATE_GUESS = {"LPV": 0.02, "ECL": 0.02, "RS": 0.10, "SOLAR_LIKE": 0.10, "YSO": 0.10, "other": 0.10}
RATE_DEFAULT = 0.05
SHAPES = ("long_dip", "ramp_fade", "secular_fade", "staircase_fade", "accelerating_fade", "repeat_long_dip")
# (flavour, number, lowest and highest signal-to-noise); docs/vetting.md §14.
SLOTS = (("grey", 25, 20, np.inf), ("dust", 15, 20, np.inf), ("reemit", 10, 20, np.inf),
         ("temperature", 25, 20, np.inf), ("g_only", 25, 20, np.inf), ("grey", 10, 6, 20))
INTERESTING_FLAVOURS = ("grey", "dust", "reemit")
DRAWS_PER_SLOT = 6           # candidate hosts tried per open slot in each round
EXT_END_BTJD = 3700.0        # repeating dips continue to the end of the ZTF and NEOWISE data (2027)
GAIA_COLS = ("source_id, ra, dec, ra_error, dec_error, ref_epoch, parallax, parallax_error, pmra, pmra_error, pmdec, "
             "pmdec_error, ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr, "
             "dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr, "
             "astrometric_params_solved, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, ruwe, "
             "ipd_frac_multi_peak, phot_bp_rp_excess_factor, l, b, ecl_lat")
SOURCE_KEYS = ("ra", "dec", "phot_g_mean_mag", "bp_rp", "parallax", "parallax_error", "ruwe", "ipd_frac_multi_peak",
               "phot_bp_rp_excess_factor")

_S = {}


# --- scoring every star of the shard ---------------------------------------------------------------------------


def _init_score(cells, frame, table):
    _S.update(cells=cells, frame=frame, table=table)


def corrected(lc, cells, table):
    """Drift-corrected epochs (frac_corr) of a light-curve table, for stars that have drift cells."""
    ep = drift.normalise(lc)
    ep = ep[ep["source_id"].isin(cells.index)]
    return ep.assign(frac_corr=ep["frac"].to_numpy() - drift.drift_at(ep, cells, table))


def _score_file(path):
    ep = corrected(pq.read_table(path), _S["cells"], _S["table"])
    if not len(ep):
        return None
    f = detect.star_features(ep)
    if not len(f):
        return None
    q = release.quiet_ids(ep)
    in_frame = path.name in _S["frame"]
    # The period search costs ~40 ms a star: every star of the sampling frame, and quiet stars everywhere.
    pe = ep if in_frame else ep[ep["source_id"].isin(q)]
    p = period.star_features(pe) if len(pe) else pd.DataFrame({"source_id": pd.Series(dtype="int64")})
    f = f.merge(p, on="source_id", how="left")
    return f.assign(quiet=f["source_id"].isin(q), frame=in_frame, file=path.name)


def flag(stats, thr, thr_p, targets=None, target_patch=None):
    """rule_v3 on `stats` (or on `targets`, fakes tested against the neighbours in `stats`)."""
    if targets is None:
        sh = rule_v3.shared(stats)
        return rule_v3.apply(stats, thr, thr_p, shared_flags=sh), sh
    sh = rule_v3.shared(stats, targets=targets, target_patch=target_patch)
    return rule_v3.apply(targets, thr, thr_p, shared_flags=sh), sh


def cmd_score_shard(a):
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    thr, thr_p, table = rule_v3.load()
    files = sorted((run / "lightcurves").glob("*.parquet"))
    frame = set(Path(a.frame).read_text().split())
    ids = np.unique(np.concatenate([pq.read_table(f, columns=["source_id"])["source_id"].to_numpy() for f in files]))
    aux = pd.read_parquet(AUX)
    aux = aux[aux["source_id"].isin(ids)]
    cells = drift.star_cells(aux)
    t = time.time()
    with Pool(a.workers, initializer=_init_score, initargs=(cells, frame, table)) as pool:
        parts = [x for x in pool.imap_unordered(_score_file, files, chunksize=1) if x is not None]
    stats = pd.concat(parts, ignore_index=True).join(cells, on="source_id").merge(aux, on="source_id", how="left")
    res, sh = flag(stats, thr, thr_p)
    stats["shared_fade"], stats["shared_rise"] = sh["shared_fade"].to_numpy(), sh["shared_rise"].to_numpy()
    for c in ("flagged", "early", "by"):
        stats[c] = res[c].to_numpy()
    stats.to_parquet(out / "stats.parquet", index=False)
    print(f"{len(stats)} stars in {time.time() - t:.0f} s; flagged {int(stats.flagged.sum())} "
          f"(frame {int((stats.flagged & stats.frame).sum())}), quiet {int(stats.quiet.sum())}, "
          f"quiet flagged {int((stats.quiet & stats.flagged).sum())}")


# --- fakes ----------------------------------------------------------------------------------------------------


def _extend(sig, t_end=EXT_END_BTJD):
    """A repeating long dip keeps repeating after Gaia's span, so ZTF and NEOWISE see the same process."""
    if not isinstance(sig, inj.Composite) or len(sig.parts) < 2:
        return sig
    parts = sorted(sig.parts, key=lambda p: p.t0)
    p = parts[1].t0 - parts[0].t0
    last, k, extra = parts[-1], 1, []
    while last.t0 + k * p < t_end:
        extra.append(inj.Dip(t0=last.t0 + k * p, covering=last.covering, tau_ref=last.tau_ref, beta=last.beta,
                             ingress=last.ingress, egress=last.egress, flat=last.flat, shape=last.shape))
        k += 1
    return inj.Composite(parts + extra)


def make_signal(spec, t_min, t_max, bp_rp):
    """The fake's signal from its plan entry: shape drawn with the entry's seed, its G change carried into every
    band by the flavour's ratios (`slow.vet.ratios`)."""
    rng = np.random.default_rng(int(spec["seed"]))
    base = inj.sample(rng, spec["shape"], t_min, t_max)
    if spec["shape"] == "repeat_long_dip":
        base = _extend(base)
    beta = float(rng.uniform(1.0, 2.5))
    t_shell = float(rng.choice(vet.SHELL_TEMPS_K))
    ratios = vet.ratios(spec["flavour"], vet.teff(bp_rp), beta, t_shell)
    return inj.BandScaled(base, ratios, default=0.0 if spec["flavour"] == "g_only" else 1.0), \
        {"beta": beta, "t_shell": t_shell}


def inject_rows(df, sig):
    """Raw light-curve rows (band as G/BP/RP names) with the signal multiplied in, band coded 0/1/2."""
    df = df.copy()
    flux, err = df["flux"].to_numpy(float).copy(), df["err"].to_numpy(float).copy()
    for b in BAND_NAMES:
        m = (df["band"] == b).to_numpy()
        flux[m], err[m] = inj.inject(df["t"].to_numpy(float)[m], flux[m], err[m], b, sig)
    df["flux"], df["err"] = flux, err
    df["band"] = df["band"].map({b: i for i, b in enumerate(BAND_NAMES)}).astype(np.int8)
    return df


def read_rows(path, ids):
    t = pq.read_table(path, filters=[("source_id", "in", [int(x) for x in ids])])
    df = t.to_pandas()
    df["band"] = df["band"].astype(str)
    return df


def _eval_file(job):
    """Detector and period-search statistics of each trial fake in one bulk file."""
    path, trials = job
    cells, table = _S["cells"], _S["table"]
    rows = read_rows(path, [t["host"] for t in trials])
    out = []
    for tr in trials:
        d = rows[rows["source_id"] == tr["host"]]
        tg = np.sort(d.loc[d["band"] == "G", "t"].to_numpy(float))
        if len(tg) < detect.MIN_VISITS:
            continue
        sig, _ = make_signal(tr, float(tg.min()), float(tg.max()), tr["bp_rp"])
        ep = corrected(inject_rows(d, sig), cells, table)
        f = detect.star_features(ep)
        if not len(f):
            continue
        f = f.merge(period.star_features(ep), on="source_id", how="left")
        tv, _ = detect.visits(tg, tg)
        f["snr"] = vet.snr(sig, tv, float(tr["sigma_visit"]))
        f["trial"] = tr["trial"]
        out.append(f)
    return pd.concat(out, ignore_index=True) if out else None


def _strata(flagged):
    cls = flagged["best_class_name"].fillna("none")
    counts = cls.value_counts()
    big = set(counts[counts >= MIN_STRATUM].index) - {"none"}
    return cls.where(cls.isin(big), "other")


def allocation(sizes, budget=N_NATURAL, floor=FLOOR):
    """Neyman allocation over strata (mix = share of flagged frame stars, guessed rates RATE_GUESS), with floor
    `floor`; a stratum smaller than its share is taken whole and its spare slots go to the others by weight."""
    names = list(sizes.index)
    size = sizes.to_numpy(int)
    mix = size / size.sum()
    rate = np.array([RATE_GUESS.get(n, RATE_DEFAULT) for n in names])
    n = np.minimum(design.neyman_allocation(min(budget, int(size.sum())), mix, rate, floor=floor), size)
    w = mix * np.sqrt(rate * (1 - rate))
    while n.sum() < min(budget, size.sum()):
        open_ = n < size
        spare = min(budget, size.sum()) - n.sum()
        add = np.floor(spare * np.where(open_, w, 0) / w[open_].sum()).astype(int)
        if add.sum() == 0:
            add[np.argmax(np.where(open_, w, -1))] = 1
        n = np.minimum(n + add, size)
    return pd.Series(n, index=names)


def classes(ids):
    from cosmos_search.gaia.archive import tap

    t, parts = tap(), []
    ids = [int(x) for x in ids]
    for i in range(0, len(ids), 500):
        chunk = ",".join(map(str, ids[i:i + 500]))
        parts.append(t.query("SELECT source_id, best_class_name, best_class_score FROM "
                             f"gaiadr3.vari_classifier_result WHERE source_id IN ({chunk})", sync=True).to_pandas())
    return pd.concat(parts, ignore_index=True)


def cmd_draw(a):
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    rnd = secrets.SystemRandom()
    thr, thr_p, table = rule_v3.load()
    stats_all = pd.read_parquet(a.stats)
    excl = {int(x) for x in Path(EXCLUDE).read_text().split()}
    stats = stats_all[~stats_all["source_id"].isin(excl)]

    # Natural stars: flagged, not quiet, in the sampling frame, by DPAC class.
    nat = stats[stats["flagged"] & stats["frame"] & ~stats["quiet"]].copy()
    nat = nat.merge(classes(nat["source_id"]), on="source_id", how="left")
    nat["stratum"] = _strata(nat)
    sizes = nat["stratum"].value_counts()
    alloc = allocation(sizes)
    picks = []
    for s, n in alloc.items():
        pool = nat[nat["stratum"] == s]["source_id"].tolist()
        picks += [{"source_id": int(x), "stratum": s} for x in rnd.sample(pool, int(n))]
    natural = pd.DataFrame(picks).merge(nat[["source_id", "best_class_name", "best_class_score"]], on="source_id")
    strata = pd.DataFrame({"flagged_in_frame": sizes, "drawn": alloc})
    print(strata)

    # Chance passes: every flagged quiet star in the shard.
    chance = stats[stats["flagged"] & stats["quiet"]][["source_id"]].assign(stratum="quiet")
    print(len(chance), "chance passes")

    # Fakes: quiet hosts at G < 19 outside the frame, shuffled; each slot takes the first host that passes.
    hosts = stats[stats["quiet"] & ~stats["flagged"] & ~stats["early"] & ~stats["frame"] & (stats["gbin"] < 8)
                  & ~stats["source_id"].isin(chance["source_id"])]
    order = hosts.sample(frac=1.0, random_state=rnd.randrange(2 ** 32)).reset_index(drop=True)
    slots = [{"flavour": f, "snr_lo": lo, "snr_hi": hi} for f, n, lo, hi in SLOTS for _ in range(n)]
    for i, s in enumerate(slots):
        s["slot"] = i
    accepted, next_host, rnd_no = {}, 0, 0
    files = {r.source_id: r.file for r in order.itertuples()}
    cells = drift.star_cells(pd.read_parquet(AUX).merge(order[["source_id"]], on="source_id"))
    _S.update(cells=cells, table=table)
    while len(accepted) < len(slots) and next_host < len(order):
        rnd_no += 1
        trials = []
        for s in slots:
            if s["slot"] in accepted:
                continue
            for _ in range(DRAWS_PER_SLOT):
                if next_host >= len(order):
                    break
                h = order.iloc[next_host]
                next_host += 1
                trials.append({"trial": len(trials), "slot": s["slot"], "host": int(h["source_id"]),
                               "flavour": s["flavour"], "shape": rnd.choice(SHAPES), "seed": rnd.getrandbits(63),
                               "bp_rp": float(h["bp_rp"]), "sigma_visit": float(h["sigma_visit"]),
                               "snr_lo": s["snr_lo"], "snr_hi": s["snr_hi"], "order": next_host - 1})
        jobs = {}
        for t in trials:
            jobs.setdefault(files[t["host"]], []).append(t)
        with Pool(a.workers, initializer=_init_score, initargs=(cells, set(), table)) as pool:
            res = [x for x in pool.map(_eval_file, [(Path(a.run) / "lightcurves" / f, j) for f, j in jobs.items()])
                   if x is not None]
        fs = pd.concat(res, ignore_index=True).join(cells, on="source_id")
        flags, _ = flag(stats_all, thr, thr_p, targets=fs,
                        target_patch=drift.sky_pixel(fs["source_id"].to_numpy(), 6))
        fs["flagged"], fs["by"] = flags["flagged"].to_numpy(), flags["by"].to_numpy()
        tr = pd.DataFrame(trials).set_index("trial")
        fs = fs.join(tr[["slot", "order", "snr_lo", "snr_hi"]], on="trial")
        ok = fs[fs["flagged"] & (fs["snr"] >= fs["snr_lo"]) & (fs["snr"] < fs["snr_hi"])].sort_values("order")
        for slot, g in ok.groupby("slot"):
            if slot not in accepted:
                t = tr.loc[int(g.iloc[0]["trial"])].to_dict()
                accepted[slot] = {**t, "snr": float(g.iloc[0]["snr"]), "by": g.iloc[0]["by"]}
        print(f"round {rnd_no}: {len(trials)} trials, {len(ok)} passed, {len(accepted)}/{len(slots)} slots filled")
    plan = [{k: (v if not isinstance(v, (np.integer, np.floating)) else v.item()) for k, v in accepted[s].items()
             if k not in ("trial", "order", "snr_lo", "snr_hi")} for s in sorted(accepted)]
    natural.to_parquet(d / "natural.parquet", index=False)
    chance.to_parquet(d / "chance.parquet", index=False)
    strata.to_csv(d / "strata.csv")
    sha = write_sealed({"fakes": plan, "natural": natural["source_id"].tolist(),
                        "chance": chance["source_id"].tolist()}, d / "plan.json.age", RECIPIENTS)
    (d / "plan.json").write_text(json.dumps({
        "fakes": len(plan), "natural": len(natural), "chance": len(chance), "plan_sha256": sha,
        "hosts_tried": next_host, "rounds": rnd_no, "rule": rule_v3.NAME,
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                                      cwd=ROOT).stdout.strip()}, indent=1))
    print(f"plan sealed: {len(plan)} fakes, {len(natural)} natural, {len(chance)} chance passes")


# --- building packets ------------------------------------------------------------------------------------------


def _open_sealed(path):
    from pyrage import decrypt, x25519

    secret = os.environ.get("COSMOS_REVEAL_KEY", "").strip()
    if not secret:
        raise SystemExit("the sealed plan needs COSMOS_REVEAL_KEY")
    return json.loads(decrypt(Path(path).read_bytes(), [x25519.Identity.from_str(secret)]))


def gaia_rows(ids):
    from cosmos_search.gaia.archive import tap

    t, parts = tap(), []
    ids = [int(x) for x in ids]
    for i in range(0, len(ids), 300):
        chunk = ",".join(map(str, ids[i:i + 300]))
        parts.append(t.query(f"SELECT {GAIA_COLS} FROM gaiadr3.gaia_source WHERE source_id IN ({chunk})",
                             sync=True).to_pandas())
    return pd.concat(parts, ignore_index=True)


def at_2019(r):
    dt = 2019.0 - r["ref_epoch"]
    pmra = 0.0 if pd.isna(r["pmra"]) else r["pmra"]
    pmdec = 0.0 if pd.isna(r["pmdec"]) else r["pmdec"]
    return r["ra"] + pmra * dt / 3.6e6 / np.cos(np.radians(r["dec"])), r["dec"] + pmdec * dt / 3.6e6


def fetch_external(gaia, out):
    """NEOWISE single exposures and raw ZTF epochs per star, one Parquet each under out/, skipped when present."""
    from cosmos_search.net.tap import Tap

    out.mkdir(parents=True, exist_ok=True)
    irsa = Tap(checks.IRSA_TAP)

    def neowise(r):
        p = out / f"neowise_{int(r['source_id'])}.parquet"
        if p.exists():
            return
        ra, dec = at_2019(r)
        for attempt in range(4):
            try:
                rows = irsa.query(checks.neowise_adql(ra, dec), sync=True).to_pandas()
                checks.neowise_clean(rows).to_parquet(p, index=False)
                return
            except Exception as e:  # noqa: BLE001 (retry, then record the failure)
                err = e
                time.sleep(5 * 2 ** attempt)
        print("NEOWISE failed", int(r["source_id"]), err)

    def ztf(r):
        p = out / f"ztf_{int(r['source_id'])}.parquet"
        if p.exists() or r["dec"] < -30:
            return
        for attempt in range(4):
            try:
                vet.ztf_raw(r["ra"], r["dec"]).to_parquet(p, index=False)
                return
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(5 * 2 ** attempt)
        print("ZTF failed", int(r["source_id"]), err)

    rows = [r for _, r in gaia.iterrows()]
    with ThreadPoolExecutor(10) as ex:
        list(ex.map(neowise, rows))
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(ztf, rows))


def _context(xm, sid):
    from cosmos_search.dayone.vet import CONTEXT_KEYS, wise_limits

    if xm is None or sid not in xm.index:
        return {}
    row = xm.loc[sid]
    ctx = {name: (float(row[col]) if col in row.index and pd.notna(row[col]) else float("nan"))
           for col, name in CONTEXT_KEYS.items()}
    return wise_limits(ctx, row)


def cmd_build(a):
    d = Path(a.dir)
    plan = _open_sealed(d / "plan.json.age")
    natural = pd.read_parquet(d / "natural.parquet")
    chance = pd.read_parquet(d / "chance.parquet")
    fakes = plan["fakes"]
    stats = pd.read_parquet(a.stats)
    sidx = stats.set_index("source_id")
    ids = sorted(set(natural["source_id"]) | set(chance["source_id"]) | {f["host"] for f in fakes})
    gaia_path = d / "work" / "gaia.parquet"
    gaia_path.parent.mkdir(parents=True, exist_ok=True)
    if not gaia_path.exists():
        gaia_rows(ids).to_parquet(gaia_path, index=False)
    gaia = pd.read_parquet(gaia_path).set_index("source_id", drop=False)
    print(len(gaia), "Gaia rows")
    if not a.skip_fetch:
        fetch_external(gaia, d / "work" / "ext")
    xm_path = d / "work" / "xm" / "context.parquet"
    if not xm_path.exists():
        subprocess.run([sys.executable, "-m", "cosmos_search.xmatch", str(gaia_path), "--surveys", "allwise", "2mass",
                        "--context", "--out", str(xm_path.parent)], check=True, cwd=ROOT)
    xm = pd.read_parquet(xm_path).set_index("source_id")

    thr, thr_p, table = rule_v3.load()
    aux = pd.read_parquet(AUX).merge(pd.DataFrame({"source_id": ids}), on="source_id")
    cells = drift.star_cells(aux)
    objs = [("natural", int(r.source_id), r.stratum, None) for r in natural.itertuples()] + \
           [("chance", int(r.source_id), "quiet", None) for r in chance.itertuples()] + \
           [("fake", int(f["host"]), f["flavour"], f) for f in fakes]
    by_file = {}
    for kind, sid, label, spec in objs:
        by_file.setdefault(sidx.at[sid, "file"], []).append((kind, sid, label, spec))

    packets, keys = [], []
    for fname, items in sorted(by_file.items()):
        rows = read_rows(Path(a.run) / "lightcurves" / fname, [x[1] for x in items])
        for kind, sid, label, spec in items:
            g = gaia.loc[sid]
            d_rows = rows[rows["source_id"] == sid]
            tg = np.sort(d_rows.loc[d_rows["band"] == "G", "t"].to_numpy(float))
            sig, extra = (make_signal(spec, float(tg.min()), float(tg.max()), spec["bp_rp"]) if spec else (None, {}))
            raw = inject_rows(d_rows, sig) if sig is not None else \
                d_rows.assign(band=d_rows["band"].map({b: i for i, b in enumerate(BAND_NAMES)}).astype(np.int8))
            ep = corrected(raw, cells, table)
            if sig is None:
                row = sidx.loc[sid].to_dict()
            else:
                f = detect.star_features(ep).merge(period.star_features(ep), on="source_id", how="left")
                f = f.join(cells, on="source_id")
                fl, _ = flag(stats, thr, thr_p, targets=f, target_patch=drift.sky_pixel(f["source_id"].to_numpy(), 6))
                row = {**f.iloc[0].to_dict(), "by": fl["by"].iloc[0], "flagged": bool(fl["flagged"].iloc[0])}
            row["source_id"] = sid
            nw_path, zt_path = d / "work" / "ext" / f"neowise_{sid}.parquet", d / "work" / "ext" / f"ztf_{sid}.parquet"
            nw_clean = pd.read_parquet(nw_path) if nw_path.exists() else checks.neowise_clean([])
            if sig is not None and len(nw_clean):
                k = np.ones(len(nw_clean))
                for b in ("W1", "W2"):
                    m = (nw_clean["band"] == b).to_numpy()
                    k[m] = sig.factor(nw_clean["mjd"].to_numpy(float)[m] - checks.BTJD_T0_MJD, b)
                nw_clean = nw_clean.assign(flux=nw_clean["flux"].to_numpy() * k, err=nw_clean["err"].to_numpy() * k)
                for b in ("W1", "W2"):
                    m = nw_clean["band"] == b
                    med = np.median(nw_clean.loc[m, "flux"]) if m.any() else 1.0
                    nw_clean.loc[m, ["flux", "err"]] = nw_clean.loc[m, ["flux", "err"]] / med
            zt_raw = pd.read_parquet(zt_path) if zt_path.exists() else pd.DataFrame(
                columns=["oid", "band", "t", "mag", "magerr"])
            ztf = vet.ztf_normalised(zt_raw, sig)
            changes = vet.band_changes(ep, row)
            tests = vet.tests(row, changes, nw_clean, ztf, g["bp_rp"])
            source = {k: float(g[k]) for k in SOURCE_KEYS if k in g.index and pd.notna(g[k])}
            cand = vet.candidate(f"gaia_dr3:{sid}", row, ep, source, _context(xm, sid),
                                 neowise=vet.neowise_visits(nw_clean), ztf=ztf, tests_=tests)
            pk, key = packet_mod.build(cand, holdout=(), pipeline_flags=False, pipeline_tests=False)
            truth = {}
            if sig is not None:
                truth = expected_checks(sig, ep, row, nw_clean)
                truth.update(flavour=spec["flavour"], shape=spec["shape"], snr=spec["snr"], **extra,
                             signal=inj.truth(sig))
            key.update(kind=kind, label=label, source_id=sid, flagged_by=row.get("by"),
                       flagged=bool(row.get("flagged", True)), g=float(g["phot_g_mean_mag"]),
                       measured={"colour": tests["colour_of_change"], "neowise": {
                           k: v for k, v in tests["neowise_change"].items() if k != "description"},
                           "ztf": {k: v for k, v in tests["ztf_after"].items() if k != "description"}},
                       truth=truth)
            packets.append(pk)
            keys.append(key)
        print(fname, len(packets))
    order = list(range(len(packets)))
    secrets.SystemRandom().shuffle(order)
    packets = [packets[i] for i in order]
    keys = [keys[i] for i in order]
    (d / "products.json").write_text(json.dumps(packets, default=_json))
    (d / "prompts").mkdir(exist_ok=True)
    for p in packets:
        system, user = prompt(p)
        (d / "prompts" / f"{p['packet_id']}.md").write_text(f"{system}\n\n---\n\n{user}\n")
    write_sealed(json.loads(json.dumps(keys, default=_json)), d / "key.json.age", RECIPIENTS)
    (d / "packet_ids.txt").write_text("\n".join(p["packet_id"] for p in packets) + "\n")
    print(f"{len(packets)} packets")


def _json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


def expected_checks(sig, ep, row, nw_clean):
    """What the colour and NEOWISE checks would give on this fake with no noise: the same estimators on the
    signal alone, sampled at the host's own epochs."""
    clean = ep.assign(frac_corr=0.0)
    for code, name in ((0, "G"), (1, "BP"), (2, "RP")):
        m = (clean["band"] == code).to_numpy()
        clean.loc[m, "frac_corr"] = sig.factor(clean.loc[m, "t"].to_numpy(float), name) - 1.0
    ch = vet.band_changes(clean, row)
    out = {"depth_g": ch["G"][0], "depth_bp": ch["BP"][0], "depth_rp": ch["RP"][0]}
    out["ratio_bp_over_rp"] = ch["BP"][0] / ch["RP"][0] if ch["RP"][0] and np.isfinite(ch["RP"][0]) and \
        abs(ch["RP"][0]) > 1e-9 else float("nan")
    t0, t1, kind = vet.change_window(row)
    if len(nw_clean):
        nz = nw_clean.copy()
        for b in ("W1", "W2"):
            m = (nz["band"] == b).to_numpy()
            nz.loc[m, "flux"] = sig.factor(nz.loc[m, "mjd"].to_numpy(float) - checks.BTJD_T0_MJD, b)
        mjd0, mjd1 = t0 + checks.BTJD_T0_MJD, t1 + checks.BTJD_T0_MJD
        if kind == "trend":
            ws = checks.wise_slope(nz, mjd0, mjd1)
            out.update({f"{b}_slope_per_yr": ws[b][0] for b in ("W1", "W2")})
        else:
            wc = checks.wise_change(nz, mjd0, mjd1)
            out.update({f"{b}_change": wc[b][0] for b in ("W1", "W2")})
    return out


# --- scoring ---------------------------------------------------------------------------------------------------


def _cp(k, n):
    from cosmos_search.vetting import power

    f = lambda x: float(np.asarray(x).ravel()[0])
    return {"k": int(k), "n": int(n), "rate": k / n if n else float("nan"),
            "lower_95_one_sided": f(power.cp_lower(k, n)) if n else float("nan"),
            "upper_95_one_sided": f(power.cp_upper(k, n)) if n else float("nan")}


def _within(z):
    z = np.asarray([x for x in z if x is not None and np.isfinite(x)], float)
    return {"n": len(z), "within_2": int(np.sum(np.abs(z) <= 2)),
            "share_within_2": float(np.mean(np.abs(z) <= 2)) if len(z) else float("nan"),
            "median_abs_z": float(np.median(np.abs(z))) if len(z) else float("nan")}


def table(verdicts, keys):
    from cosmos_search.dayone.vet import INTERESTING, S_SHORTLIST

    rows = []
    for v in verdicts:
        k = keys[v.packet_id]
        t, m = k.get("truth") or {}, k["measured"]
        s = sum(v.class_probs.get(c, 0.0) for c in INTERESTING) if v.ok else float("nan")
        hit = bool(v.ok and s >= S_SHORTLIST)
        col, nw = m["colour"], m["neowise"]
        trend = "W1_slope_per_yr" in nw
        meas_w1 = nw.get("W1_slope_per_yr" if trend else "W1_change")
        err_w1 = nw.get("W1_slope_err" if trend else "W1_err")
        exp_w1 = t.get("W1_slope_per_yr" if trend else "W1_change")
        visits_in = nw.get("W1_visits" if trend else "W1_visits_in") or 0
        z_col = (col["ratio_bp_over_rp"] - t["ratio_bp_over_rp"]) / col["ratio_err"] \
            if t and _fin(col.get("ratio_bp_over_rp")) and _fin(t.get("ratio_bp_over_rp")) and _fin(col.get("ratio_err")) \
            and col["ratio_err"] > 0 else None
        z_w1 = (meas_w1 - exp_w1) / err_w1 if t and _fin(meas_w1) and _fin(exp_w1) and _fin(err_w1) and err_w1 > 0 \
            and visits_in >= (4 if trend else 1) else None
        z_w1_pred = {}
        if _fin(meas_w1) and _fin(err_w1) and err_w1 > 0 and not trend:
            for name in ("grey", "dust_in_front", "star_temperature"):
                pred = nw.get(f"W1_pred_{name}")
                z_w1_pred[name] = (meas_w1 - pred) / err_w1 if _fin(pred) else None
        rows.append({
            "packet_id": v.packet_id, "kind": k["kind"], "label": k["label"], "ok": v.ok, "error": v.error,
            "s": s, "shortlisted": hit, "p_unexplained": v.p_unexplained if v.ok else float("nan"),
            "top": v.top() if v.ok else None, "recognised_as": v.recognised_as, "g": k["g"],
            "flagged_by": k.get("flagged_by"), "flavour": t.get("flavour"), "shape": t.get("shape"),
            "snr": t.get("snr"), "z_grey": col.get("z_grey"), "z_dust": col.get("z_dust"),
            "ratio": col.get("ratio_bp_over_rp"), "ratio_true": t.get("ratio_bp_over_rp"), "z_colour_vs_truth": z_col,
            "w1_visits_in": visits_in, "w1": meas_w1, "w1_err": err_w1, "w1_true": exp_w1, "z_w1_vs_truth": z_w1,
            **{f"z_w1_vs_{n}": z for n, z in z_w1_pred.items()},
            "ztf_bands": sum(1 for kk in m["ztf"] if kk.endswith("_change_after_2020")),
            "candidate_id": k["candidate_id"]})
    return pd.DataFrame(rows)


def _fin(x):
    return x is not None and isinstance(x, (int, float)) and np.isfinite(x)


def report(df, strata):
    from cosmos_search.vetting import taxonomy

    out = {}
    fake = df[df["kind"] == "fake"]
    good = fake[fake["flavour"].isin(INTERESTING_FLAVOURS) & (fake["snr"] >= 20)]
    nat = df[df["kind"].isin(["natural", "chance"])]
    out["R1_recall"] = {**_cp(good["shortlisted"].sum(), len(good)), "bar": 0.5}
    out["R1_recall"]["pass"] = out["R1_recall"]["lower_95_one_sided"] >= 0.5
    out["R2_natural_false_shortlist"] = {**_cp(nat["shortlisted"].sum(), len(nat)), "bar": 0.10}
    out["R2_natural_false_shortlist"]["pass"] = out["R2_natural_false_shortlist"]["upper_95_one_sided"] <= 0.10
    c1 = _within(fake["z_colour_vs_truth"])
    out["C1_colour_errors"] = {**c1, "pass": c1["share_within_2"] >= 0.85}
    c2 = _within(fake["z_w1_vs_truth"])
    out["C2_neowise_errors"] = {**c2, "pass": c2["share_within_2"] >= 0.85}
    out["fakes_by_flavour"] = {f"{f} snr>=20" if lo else f"{f} snr<20": _cp(g["shortlisted"].sum(), len(g))
                               for (f, lo), g in fake.groupby([fake["flavour"], fake["snr"] >= 20])}
    out["fakes_by_shape"] = {s: _cp(g["shortlisted"].sum(), len(g)) for s, g in good.groupby("shape")}
    bins = pd.cut(fake["snr"], [6, 20, 40, 100, 1e9])
    out["interesting_fakes_by_snr"] = {str(b): _cp(g["shortlisted"].sum(), len(g)) for b, g in
                                       fake[fake["flavour"].isin(INTERESTING_FLAVOURS)].groupby(
                                           bins[fake["flavour"].isin(INTERESTING_FLAVOURS)], observed=True)}
    out["natural_by_stratum"] = {s: _cp(g["shortlisted"].sum(), len(g)) for s, g in nat.groupby("label")}
    # Weighted back to the frame's flagged mix: per-stratum rates, bootstrap within strata.
    w = strata["flagged_in_frame"] / strata["flagged_in_frame"].sum()
    nn = nat[nat["kind"] == "natural"]
    rng = np.random.default_rng(0)
    rates = {s: nn.loc[nn["label"] == s, "shortlisted"].to_numpy(float) for s in w.index}
    est = sum(w[s] * rates[s].mean() for s in w.index if len(rates[s]))
    boot = [sum(w[s] * rng.choice(rates[s], len(rates[s])).mean() for s in w.index if len(rates[s]))
            for _ in range(2000)]
    out["natural_weighted_to_frame"] = {"rate": float(est), "lo_95": float(np.quantile(boot, 0.025)),
                                        "hi_95": float(np.quantile(boot, 0.975))}
    out["top_group"] = {k: g["top"].map(lambda x: taxonomy.GROUP_OF.get(x) if x else None).value_counts().to_dict()
                        for k, g in df.groupby("kind")}
    out["top_class"] = {lab: g["top"].value_counts().head(5).to_dict() for lab, g in df.groupby(
        df["label"].where(df["kind"] != "fake", "fake:" + df["flavour"].astype(str)))}
    out["checks_on_natural"] = {s: {
        "n": len(g), "colour_measured": int(g["z_grey"].notna().sum()),
        "grey_within_2": int((g["z_grey"].abs() <= 2).sum()),
        "neowise_visit_inside": int((g["w1_visits_in"] >= 1).sum()),
        "w1_consistent_grey": int((g.get("z_w1_vs_grey", pd.Series(dtype=float)).abs() <= 2).sum()),
        "w1_consistent_dust": int((g.get("z_w1_vs_dust_in_front", pd.Series(dtype=float)).abs() <= 2).sum()),
        "ztf_any_band": int((g["ztf_bands"] > 0).sum())} for s, g in nat.groupby("label")}
    out["coverage_all"] = {"n": len(df), "colour_measured": int(df["z_grey"].notna().sum()),
                           "neowise_visit_inside": int((df["w1_visits_in"] >= 1).sum()),
                           "ztf_any_band": int((df["ztf_bands"] > 0).sum())}
    out["format_failures"] = int((~df["ok"]).sum())
    out["recognised"] = int(df["recognised_as"].notna().sum())
    from sklearn.metrics import roc_auc_score

    pos, neg = good["s"].dropna(), nat["s"].dropna()
    out["auc_s_interesting_fakes_vs_natural"] = float(roc_auc_score(
        np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])) if len(pos) and len(neg) else None
    return out


def cmd_score(a):
    from cosmos_search.vetting.commit import load_verdicts, reveal

    d = Path(a.dir)
    keys = reveal(d / "batch", d / "key.json.age", remote=None if a.unpublished else "origin")
    df = table(load_verdicts(d / "batch"), keys)
    strata = pd.read_csv(d / "strata.csv", index_col=0)
    rep = report(df, strata)
    df.to_csv(d / "verdicts_table.csv", index=False)
    (d / "report.json").write_text(json.dumps(rep, indent=1, default=_json))
    print(json.dumps({k: rep[k] for k in ("R1_recall", "R2_natural_false_shortlist", "C1_colour_errors",
                                          "C2_neowise_errors")}, indent=1, default=_json))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score-shard")
    s.add_argument("--run", required=True)
    s.add_argument("--frame", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=os.cpu_count())
    s.set_defaults(fn=cmd_score_shard)
    s = sub.add_parser("draw")
    s.add_argument("--run", required=True)
    s.add_argument("--stats", required=True)
    s.add_argument("--dir", required=True)
    s.add_argument("--workers", type=int, default=os.cpu_count())
    s.set_defaults(fn=cmd_draw)
    s = sub.add_parser("build")
    s.add_argument("--run", required=True)
    s.add_argument("--stats", required=True)
    s.add_argument("--dir", required=True)
    s.add_argument("--skip-fetch", action="store_true")
    s.set_defaults(fn=cmd_build)
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--unpublished", action="store_true", help="tuning only: skip the check that verdicts are on origin")
    s.set_defaults(fn=cmd_score)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
