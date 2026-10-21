"""Calibration and held-out test of the residual search (docs/arms/residual.md).

    python -m cosmos_search.residual.cal run --lc joint.parquet --ctx ctx.parquet --half A --out res_A.parquet
    python -m cosmos_search.residual.cal choose res_A.parquet
    python -m cosmos_search.residual.cal report res_B.parquet --tol 0.1 --min-event-epochs 2

Same data, halves and injection machinery as the EB filter's calibration (`ebfilter.cal`): Gaia DR3 G and ZTF DR22
epochs for 7,565 stars, halves by a hash of source_id, every injection redrawn until its deepest G epoch would put
the star in the queue. Joint (G + ZTF) data only: the eclipse baseline needs the external epochs.

Populations: `real_eb` (DPAC eclipsing binaries as observed; should be demoted), `host` (queued non-EBs as
observed), `eb_synth` (synthetic binaries in non-EB hosts; should be demoted), `eb_host:<family>` (an aperiodic dip
or train injected into a real binary; must stay), `anomaly:<family>` (the target families in non-EB hosts; must
stay) and `anomaly:dip_train_periodic` (reported).

Each job runs the EB filter and the residual search once (`model.prepare`) and evaluates every tolerance in TOLS and
every MIN_EVENT_EPOCHS in MIN_EVENTS on it. The recurrence test runs on the raw epochs (rule v3b's flag) and, for
the populations where recurrence matters (trains, real binaries, hosts), on the residual epochs at every tolerance.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from cosmos_search.ebfilter import cal as EC
from cosmos_search.vetting import recurrence

from . import model as M

TOLS = (0.05, 0.1, 0.2, 0.4)
MIN_EVENTS = (1, 2)
TRAIN_FAMILIES = ("dip_train_poisson", "dip_train_primes", "dip_train_geometric")
RECURRENCE_KINDS = ("real_eb", "host")
R_FRACTION = 0.75           # pre-registered: demotion on real EBs at least this share of the filter's


def _job(job):
    kind, family, sid, src, df, seed, _ = job
    rng = np.random.default_rng(seed)
    t_lo, t_hi = float(df.t.min()), float(df.t.max())
    if kind in ("host", "real_eb"):
        g = df[df.band == "G"]
        tt = EC._target(g.t.to_numpy(), g.flux.to_numpy())
        if tt is None:
            return []
        inj = df
    else:
        sig, tt = EC._draw(rng, "eb_synth" if kind == "eb_synth" else "family", family, src, df, t_lo, t_hi)
        if sig is None:
            return [{"kind": kind, "family": family, "source_id": sid, "drawn": False}]
        inj = EC._apply(df, sig)
    lcs = EC._lcs(inj, EC.GAIA_BANDS + EC.ZTF_BANDS)
    ctx = M.prepare(lcs, src, tt)
    raw = recurrence.search(*recurrence.epochs(lcs))
    rec = kind in RECURRENCE_KINDS or family in TRAIN_FAMILIES or family == "dip_train_periodic"
    base = {"kind": kind, "family": family, "source_id": sid, "drawn": True, "t_target": tt,
            "raw_testable": bool(raw["testable"]), "raw_aperiodic": bool(raw.get("aperiodic", False)),
            "raw_n_events": int(raw["n_events"]), "g_mag": src.get("phot_g_mean_mag")}
    rows = []
    for tol in TOLS:
        for m in MIN_EVENTS:
            # With a constant baseline the residual epochs are the raw ones: reuse the raw recurrence result.
            v = M.evaluate(ctx, tol=tol, min_event_epochs=m,
                           recurrence_test=rec and m == MIN_EVENTS[0] and ctx["base"] is not None)
            v.pop("residual_lightcurves")
            if rec and ctx["base"] is None:
                v.update({"residual_testable": bool(raw["testable"]), "residual_aperiodic": bool(raw.get("aperiodic", False)),
                          "residual_aligned_fraction": raw.get("aligned_fraction", np.nan)})
            rows.append({**base, "tol": tol, "min_event_epochs": m, **v})
    # recurrence on residuals does not depend on min_event_epochs: copy it across
    for i, r in enumerate(rows):
        if "residual_aperiodic" not in r and rec:
            src_row = rows[i - (i % len(MIN_EVENTS))]
            for k in ("residual_testable", "residual_aperiodic", "residual_aligned_fraction"):
                r[k] = src_row.get(k)
    return rows


def jobs(lc, ctx, which, n_synth, n_family, n_host_family, rng):
    """EB filter calibration jobs (`ebfilter.cal.jobs`) for one half, with this test's population sizes."""
    js, counts = EC.jobs(lc, ctx, which, n_synth, n_family, n_host_family, rng)
    return js, counts


def wilson(k, n):
    return EC.wilson(k, n)


def _rate(sub, col):
    k, n = int(sub[col].fillna(False).astype(bool).sum()), len(sub)
    lo, hi = wilson(k, n)
    return k, n, lo, hi


def table(df, tol, m):
    """Per population at one setting: filter-explained (current), demoted, eb_residual, and recurrence flags."""
    d = df[df.drawn & (df.tol == tol) & (df.min_event_epochs == m)].copy()
    d["pop"] = np.where(d.kind.isin(["anomaly", "eb_host"]), d.kind + ":" + d.family, d.kind)
    d.loc[d.kind.eq("anomaly") & d.family.isin(EC.TARGET_FAMILIES), "group"] = "anomaly (target families)"
    d.loc[d.kind.eq("eb_host"), "group"] = "eb_host (all families)"
    rows = []
    for name, sub in list(d.groupby("pop")) + list(d.dropna(subset=["group"]).groupby("group")):
        row = {"population": name, "n": len(sub)}
        for col in ("eb_explained", "demote", "eb_residual", "raw_aperiodic", "residual_aperiodic"):
            k, n, lo, hi = _rate(sub, col) if col in sub else (0, 0, np.nan, np.nan)
            row[col] = f"{k} ({k / n:.1%}; {lo:.1%}-{hi:.1%})" if n else ""
        if "raw_testable" in sub:
            row["testable raw/resid"] = f"{int(sub.raw_testable.sum())}/{int(sub.residual_testable.fillna(False).sum())}"
        rows.append(row)
    return pd.DataFrame(rows)


def choose(df):
    """The pre-registered choice on half A: among TOLS x MIN_EVENTS, the smallest eb_host loss (demoted) such that
    demotion of real EBs is at least R_FRACTION of the EB filter's; ties go to more real-EB demotion. None if no
    setting qualifies."""
    d = df[df.drawn]
    r0 = d[(d.kind == "real_eb") & (d.tol == TOLS[0]) & (d.min_event_epochs == MIN_EVENTS[0])].eb_explained.mean()
    best = None
    out = []
    for tol in TOLS:
        for m in MIN_EVENTS:
            s = d[(d.tol == tol) & (d.min_event_epochs == m)]
            r = s[s.kind == "real_eb"].demote.mean()
            loss = s[s.kind == "eb_host"].demote.mean()
            ok = r >= R_FRACTION * r0
            out.append({"tol": tol, "min_event_epochs": m, "real_eb_demoted": r, "eb_host_loss": loss,
                        "filter_real_eb": r0, "filter_eb_host_loss": s[s.kind == "eb_host"].eb_explained.mean(),
                        "qualifies": ok})
            if ok and (best is None or (loss, -r) < (best[2], -best[3])):
                best = (tol, m, loss, r)
    return (best[:2] if best else None), pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.residual.cal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--lc", required=True)
    r.add_argument("--ctx", required=True)
    r.add_argument("--half", choices=("A", "B"), required=True)
    r.add_argument("--n-synth", type=int, default=300)
    r.add_argument("--n-family", type=int, default=60)
    r.add_argument("--n-host-family", type=int, default=120)
    r.add_argument("--workers", type=int, default=os.cpu_count())
    r.add_argument("--seed", type=int)
    r.add_argument("--limit", type=int, help="first N jobs only (smoke test)")
    r.add_argument("--out", required=True)
    c = sub.add_parser("choose")
    c.add_argument("file")
    p = sub.add_parser("report")
    p.add_argument("file")
    p.add_argument("--tol", type=float, required=True)
    p.add_argument("--min-event-epochs", type=int, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "choose":
        best, grid = choose(pd.read_parquet(a.file))
        print(grid.to_markdown(index=False, floatfmt=".3f"))
        print(f"\nchosen: {best}")
        return 0
    if a.cmd == "report":
        print(table(pd.read_parquet(a.file), a.tol, a.min_event_epochs).to_markdown(index=False))
        return 0
    seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(8), "little")
    rng = np.random.default_rng(seed)
    lc = pd.read_parquet(a.lc)
    lc["band"] = lc["band"].astype(str)
    ctx = pd.read_parquet(a.ctx)
    js, counts = jobs(lc, ctx, a.half, a.n_synth, a.n_family, a.n_host_family, rng)
    if a.limit:
        js = js[: a.limit]
    print(f"half {a.half}: {counts}, {len(js)} jobs, seed {seed}", flush=True)
    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for i, res in enumerate(pool.map(_job, js, chunksize=1)):
            rows += res
            if i % 100 == 0:
                print(f"{i}/{len(js)}", flush=True)
    out = pd.DataFrame(rows)
    out.attrs["seed"] = seed
    out["half"] = a.half
    out["seed"] = str(seed)
    out.to_parquet(a.out)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
