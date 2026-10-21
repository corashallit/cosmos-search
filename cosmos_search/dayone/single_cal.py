"""Calibration of the single-event test (vetting.single_event) before rule v4 is pre-registered (docs/dayone.md).

    python -m cosmos_search.dayone.single_cal sample --features FEATURES_DIR --aux aux_gaia_source.parquet \\
        --ids extra_ids.csv --out cal1/sample.parquet
    python -m cosmos_search.dayone.single_cal fetch --sample cal1/sample.parquet --out cal1/
    python -m cosmos_search.dayone.single_cal run --dir cal1/ --rounds 3 --out cal1/results.parquet

Populations (column `population`):

- `background`: random DR3 sources whose cleaned G light curve has exactly one low episode (n_low_episodes == 1,
  min_z_rob <= -5, no bright 5-sigma epoch): what a single dip usually is in the DR3 variable sample.
- `long_eb`: Gaia DR3 eclipsing binaries (vari_eclipsing_binary) with periods over 20 d: the false-positive class
  the test is built against.
- `host`: quiet stars (G 10-16, robust scatter under 2x the errors, no 3-sigma epoch, not in a DPAC class that
  dips) for injected single dips (grey, blue, chromatic flip), drawn over DR3's span and redrawn until a visible
  G epoch moves by >= 5 sigma, as decoys are.
- anything in --ids (a csv of source_id, population): the rehearsal's real candidates, controls and tier-1
  single-episode stars.

Every star goes through the packet path (Stage 0, normalisation, ZTF and TESS bands, tail holdout), and the
test is read from the packet, so calibration measures what Stage A would see.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from cosmos_search.gaia import epoch
from cosmos_search.inject import lightcurve as lcinj
from cosmos_search.inject.classes import gaia_stream
from cosmos_search.vetting.candidate import Candidate

from .rank import DIP_CLASSES

SINGLE_FAMILIES = ("grey_dip", "blue_dip", "chromatic_flip")
N_BACKGROUND, N_HOSTS, N_EB = 150, 120, 120
EB_MIN_PERIOD = 20.0
G_RANGE = (10.0, 16.5)
MIN_SIGMA = 5.0


def sample(features_dir, aux, rng, n_background=N_BACKGROUND, n_hosts=N_HOSTS, n_files=8):
    """Background single-episode sources and quiet hosts from a random subset of the features parts."""
    files = sorted(Path(features_dir).glob("part-*.parquet"))
    pick = [files[i] for i in rng.choice(len(files), min(n_files, len(files)), replace=False)]
    cols = ["source_id", "n_g", "sigma_rob_g", "err_med_g", "n_low3", "n_high3", "n_high5", "min_z_rob",
            "n_low_episodes"]
    f = pd.concat([pq.read_table(p, columns=cols).to_pandas() for p in pick], ignore_index=True)
    g = pq.read_table(aux, columns=["source_id", "phot_g_mean_mag"]).to_pandas()
    f = f.merge(g, on="source_id")
    f = f[(f.phot_g_mean_mag >= G_RANGE[0]) & (f.phot_g_mean_mag < G_RANGE[1]) & (f.n_g >= 20)]
    bg = f[(f.n_low_episodes == 1) & (f.min_z_rob <= -5) & (f.n_high5 == 0)]
    quiet = f[(f.sigma_rob_g < 2 * f.err_med_g) & (f.n_low3 == 0) & (f.n_high3 == 0) & (f.phot_g_mean_mag < 16)]
    bg = bg.sample(min(n_background, len(bg)), random_state=rng.integers(2**31))
    quiet = quiet.sample(min(3 * n_hosts, len(quiet)), random_state=rng.integers(2**31))
    return pd.concat([pd.DataFrame({"source_id": bg.source_id, "population": "background"}),
                      pd.DataFrame({"source_id": quiet.source_id, "population": "host"})], ignore_index=True)


def long_ebs(tap, n=N_EB):
    q = (f"SELECT TOP {n} e.source_id FROM gaiadr3.vari_eclipsing_binary AS e JOIN gaiadr3.gaia_source AS s "
         f"ON s.source_id = e.source_id WHERE e.frequency < {1 / EB_MIN_PERIOD} AND e.frequency > 0 "
         f"AND s.phot_g_mean_mag BETWEEN {G_RANGE[0]} AND {G_RANGE[1]} ORDER BY s.random_index")
    ids = tap.query(q).to_pandas()["source_id"].astype("int64")
    return pd.DataFrame({"source_id": ids, "population": "long_eb"})


def fetch(sample_df, out, log=print, n_hosts=N_HOSTS, external=True):
    """Context (gaia_source, DPAC class, EB model), Gaia light curves through DataLink, ZTF and TESS."""
    from cosmos_search.gaia.archive import tap

    from . import external as ext_mod
    from .rank import context

    out = Path(out)
    (out / "lightcurves").mkdir(parents=True, exist_ok=True)
    ctx = context(sample_df["source_id"].to_numpy(), tap())
    ctx = sample_df.merge(ctx, on="source_id", how="inner")
    # Hosts must not be in a class that dips; keep the first n_hosts that pass.
    host = ctx.population == "host"
    bad = host & ctx["best_class_name"].isin(DIP_CLASSES)
    keep = ctx[~bad]
    keep = pd.concat([keep[keep.population != "host"], keep[keep.population == "host"].head(n_hosts)])
    keep.to_parquet(out / "context.parquet")
    ids = keep["source_id"].astype("int64").tolist()
    done = set()
    for p in (out / "lightcurves").glob("*.parquet"):
        done |= set(pq.read_table(p, columns=["source_id"])["source_id"].to_pylist())
    todo = [s for s in ids if s not in done]
    for k in range(0, len(todo), epoch.DATALINK_MAX_IDS):
        chunk = todo[k:k + epoch.DATALINK_MAX_IDS]
        lc = epoch.datalink_epoch_photometry(chunk, cache=False)
        pq.write_table(lc, out / "lightcurves" / f"chunk-{len(done) + k:06d}.parquet", compression="zstd")
        log(f"gaia {k + len(chunk)}/{len(todo)}")
    if external and not (out / "external.parquet").exists():
        e = ext_mod.fetch(keep[["source_id", "ra", "dec"]], max_sectors=6, log=log)
        e.to_parquet(out / "external.parquet")
    return keep


def _draw_plan(hosts, lc_dir, blend, rng, families=SINGLE_FAMILIES):
    """One single-dip injection per host, visible in some G epoch before the tail cut by >= MIN_SIGMA."""
    from .decoys import _visible_g
    from .rank import read_lightcurves

    lc = read_lightcurves(lc_dir, hosts)
    plan = {}
    for i, sid in enumerate(hosts):
        vis = _visible_g(lc, int(sid), blend)
        if vis is None:
            continue
        t, _, s, _ = vis
        fam = families[i % len(families)]
        for _ in range(500):
            sig = lcinj.sample(rng, fam, *gaia_stream.DR3_SPAN)
            if np.max(np.abs(1 - sig.factor(t, "G"))) >= MIN_SIGMA * s:
                break
        else:
            continue
        frac = blend.get(int(sid))
        r = float(rng.uniform(0.2, 1.0)) if frac is not None and frac >= gaia_stream.BLEND_MAX else 1.0
        plan[int(sid)] = {"source_id": int(sid), "family": fam, "bands": None, "bp_rp_dilution": r,
                          "seed": int(rng.integers(2**62)), "signal": lcinj.truth(sig)}
    return plan


def visible(cand):
    """The candidate's light curves as a tail-holdout packet shows them (vetting.packet.build)."""
    from cosmos_search.vetting.packet import HOLDOUT_TAIL, _subset

    t0, t1 = cand.time_span()
    cut = t0 + (1 - HOLDOUT_TAIL) * (t1 - t0)
    return [_subset(lc, (lc.time < cut) | (lc.time > t1)) for lc in cand.lightcurves]


def evaluate(ctx, lc_dir, external, plan_map=None, n_draws=None):
    """The single-event test on every star in ctx, on the epochs a tail-holdout packet would show."""
    from cosmos_search.vetting import single_event

    from .vet import candidate_dicts

    queue = ctx.assign(role="ranked")
    cands = candidate_dicts(queue, None, lc_dir, ctx, plan_map, external)
    pop = dict(zip(ctx.source_id.astype("int64"), ctx.population))
    rows = []
    for c in cands:
        sid = int(c["candidate_id"].split(":")[1])
        cand = Candidate.from_dict(c)
        lcs = visible(cand)
        se = single_event.test(lcs, cand.source, cand.context, **({"n_draws": n_draws} if n_draws else {})) or {}
        rows.append({"source_id": sid, "population": pop.get(sid), "injected": bool(plan_map and sid in plan_map),
                     "family": plan_map[sid]["family"] if plan_map and sid in plan_map else None,
                     "single": bool(se), **{f"se_{k}": (str(v) if isinstance(v, list) else v)
                                            for k, v in se.items() if k != "description"},
                     "ext_bands": ",".join(sorted({lc.band for lc in cand.lightcurves} - {"G", "BP", "RP"}))})
    return pd.DataFrame(rows)


def run(d, rounds, rng):
    from .rank import _blend_map

    d = Path(d)
    ctx = pd.read_parquet(d / "context.parquet")
    lc_dir = d / "lightcurves"
    ext = pd.read_parquet(d / "external.parquet") if (d / "external.parquet").exists() else None
    parts = [evaluate(ctx, lc_dir, ext)]
    hosts = ctx.loc[ctx.population == "host", "source_id"].astype("int64").to_numpy()
    blend = _blend_map(ctx)
    hctx = ctx[ctx.population == "host"]
    for r in range(rounds):
        plan = _draw_plan(rng.permutation(hosts), lc_dir, blend, rng)
        res = evaluate(hctx, lc_dir, ext, plan)
        parts.append(res.assign(round=r))
    return pd.concat(parts, ignore_index=True)


def joint(lc_path, ctx_path, work, n=None, rng=None, batch=300, log=print):
    """The single-event test on a joint Gaia + ZTF set (the EB-filter thread's /mnt/project-files/ebfilter/):
    real stars with their DPAC class, and DPAC's eclipsing-binary period where it has one."""
    work = Path(work)
    (work / "lightcurves").mkdir(parents=True, exist_ok=True)
    from cosmos_search.vetting import stage0

    ctx = pd.read_parquet(ctx_path).drop_duplicates("source_id")
    for col in stage0.BLEND_COLS:
        if col not in ctx:
            ctx[col] = np.nan
    if n:
        ctx = ctx.sample(min(n, len(ctx)), random_state=(rng or np.random.default_rng()).integers(2**31))
    period = 1 / ctx["frequency"].where(ctx["frequency"] > 0)
    ctx = ctx.assign(population=np.where(period > EB_MIN_PERIOD, "joint_long_eb",
                                         "joint_" + ctx["best_class_name"].fillna("none").astype(str)),
                     eb_period=period)
    lc = pq.read_table(lc_path, filters=[("source_id", "in", ctx["source_id"].astype("int64").tolist())]).to_pandas()
    gaia = lc[lc.band.isin(["G", "BP", "RP"])].assign(band=lambda x: pd.Categorical(x.band, categories=["G", "BP", "RP"]))
    pq.write_table(pa.Table.from_pandas(gaia, preserve_index=False), work / "lightcurves" / "joint.parquet")
    ext = lc[~lc.band.isin(["G", "BP", "RP"])][["source_id", "band", "t", "flux", "err"]]
    out = []
    for k in range(0, len(ctx), batch):
        out.append(evaluate(ctx.iloc[k:k + batch], work / "lightcurves", ext))
        log(f"joint {min(k + batch, len(ctx))}/{len(ctx)}")
    res = pd.concat(out, ignore_index=True)
    return res.merge(ctx[["source_id", "eb_period", "phot_g_mean_mag"]], on="source_id", how="left")


# Batch 5 (docs/dayone.md, rule v4): single-event families are the primary measure; dust dips are other decoys
# (an ideal observer reads them as dust, not as an eclipse).
COUNTS5 = {"grey_dip": 8, "blue_dip": 4, "chromatic_flip": 4, "dust_dip": 2}
MAX_DRAWS5 = 400


def single_event_count(ctx_row, lc_dir, external, rec, contamination=None):
    """(number of dip events, epochs in the event) the single-event test sees for one host with this signal."""
    from cosmos_search.vetting import single_event

    from .vet import candidate_dicts

    sid = int(ctx_row["source_id"])
    q = pd.DataFrame([ctx_row]).assign(role="ranked")
    ext = None if external is None else external[external["source_id"] == sid]
    if ext is not None and contamination:
        q["blend_tess_contamination"] = contamination.get(sid, np.nan)
    c = candidate_dicts(q, None, lc_dir, q, {sid: rec}, ext)
    if not c:
        return 0, 0
    cand = Candidate.from_dict(c[0])
    ep = single_event._band_epochs(visible(cand))
    if ep is None:
        return 0, 0
    ev, n = single_event._event(ep)
    return n, (len(ev["idx"]) if ev else 0)


def plan5(host_ctx, lc_dir, external, rng, counts=COUNTS5, min_in_dip=1, half="batch5", contamination=None):
    """Sealed-plan records for batch 5: single-event families on hosts with external data, each redrawn until a
    visible G epoch moves by >= MIN_SIGMA and the packet's epochs show exactly one dip event on at least
    min_in_dip epochs (dust dips: visibility only)."""
    from .decoys import MIN_EXTERNAL, _visible_g
    from .rank import _blend_map, read_lightcurves

    n_ext = external.groupby(["source_id", "band"]).size().unstack(fill_value=0)
    covered = set(n_ext.index[(n_ext >= MIN_EXTERNAL).any(axis=1)])
    host_ctx = host_ctx[host_ctx["source_id"].isin(covered)].reset_index(drop=True)
    order = rng.permutation(len(host_ctx))
    lc = read_lightcurves(lc_dir, host_ctx["source_id"].to_numpy())
    blend = _blend_map(host_ctx)
    t_ext = float(external["t"].max())
    out, used = [], 0
    for fam in [f for f, k in counts.items() for _ in range(k)]:
        while used < len(order):
            row = host_ctx.iloc[order[used]]
            used += 1
            sid = int(row["source_id"])
            vis = _visible_g(lc, sid, blend)
            if vis is None:
                continue
            t, _, sg, _ = vis
            frac = blend.get(sid)
            r = float(rng.uniform(0.2, 1.0)) if frac is not None and frac >= gaia_stream.BLEND_MAX else 1.0
            for _ in range(MAX_DRAWS5):
                sig = lcinj.sample(rng, fam, *gaia_stream.DR3_SPAN, t_ext=t_ext)
                if np.max(np.abs(1 - sig.factor(t, "G"))) < MIN_SIGMA * sg:
                    continue
                rec = {"source_id": sid, "family": fam, "half": half, "bands": None, "bp_rp_dilution": r,
                       "seed": int(rng.integers(2**62)), "signal": lcinj.truth(sig)}
                if fam in SINGLE_FAMILIES:
                    n, n_in = single_event_count(row, lc_dir, external, rec, contamination)
                    if n != 1 or n_in < min_in_dip:
                        continue
                out.append(rec)
                break
            else:
                continue
            break
    return out


def summary(res, b_min=2.5):
    """Single-event share and flag rates per population (and per injected family)."""
    res = res.copy()
    res["group"] = np.where(res.injected, "inj:" + res.family.fillna(""), res.population)
    bf = pd.to_numeric(res.get("se_log10_oneoff_vs_eb"), errors="coerce")
    n2 = pd.to_numeric(res.get("se_n_in_dip"), errors="coerce") >= 2
    res["flag_bf"] = res.single & n2 & (bf >= b_min)
    p = pd.to_numeric(res.get("se_p_eb"), errors="coerce")
    res["flag_p"] = res.single & n2 & (p <= 0.01)
    g = res.groupby("group")
    return pd.DataFrame({"n": g.size(), "single": g.single.sum(), "n2": (res.single & n2).groupby(res.group).sum(),
                         "flag_bf": g.flag_bf.sum(), "flag_p": g.flag_p.sum()})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.dayone.single_cal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--features", required=True)
    s.add_argument("--aux", required=True)
    s.add_argument("--ids", help="csv with source_id, population")
    s.add_argument("--out", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--sample", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--no-external", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--dir", required=True)
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--out", required=True)
    j = sub.add_parser("joint")
    j.add_argument("--lightcurves", required=True)
    j.add_argument("--context", required=True)
    j.add_argument("--work", required=True)
    j.add_argument("--n", type=int)
    j.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(int.from_bytes(os.urandom(16), "little"))
    if a.cmd == "sample":
        from cosmos_search.gaia.archive import tap

        parts = [sample(a.features, a.aux, rng), long_ebs(tap())]
        if a.ids:
            parts.append(pd.read_csv(a.ids, dtype={"source_id": "int64"}))
        df = pd.concat(parts, ignore_index=True).drop_duplicates("source_id")
        df.to_parquet(a.out)
        print(df.population.value_counts().to_string())
    elif a.cmd == "fetch":
        fetch(pd.read_parquet(a.sample), a.out, log=lambda m: print(m, flush=True), external=not a.no_external)
    elif a.cmd == "joint":
        res = joint(a.lightcurves, a.context, a.work, a.n, rng, log=lambda m: print(m, flush=True))
        res.to_parquet(a.out)
        print(summary(res).to_string())
    else:
        res = run(a.dir, a.rounds, rng)
        res.to_parquet(a.out)
        print(summary(res).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
