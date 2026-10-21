"""Decoys for a vetting batch whose packets carry other surveys' light curves (docs/dayone.md, batch 3).

    python -m cosmos_search.dayone.decoys hosts --lightcurves stream/lightcurves --ranked rank/ranked.parquet \\
        --like rank/queue2.parquet --exclude-plan plan.json.age --aux aux_gaia_source.parquet --n 60 \\
        --out ext/hosts.parquet
    python -m cosmos_search.dayone.decoys plan --hosts ext/hosts.parquet --lightcurves stream/lightcurves \\
        --external ext/external.parquet --out ext/plan3.json.age --recipient-file keys/cora.age.pub ...

The rehearsal's injections were drawn over DR3's span only, so they stop where ZTF and TESS begin. Here every
dip train keeps running under its own law (same rate, period or spacing) to the end of the external data, so a
decoy's ZTF and TESS epochs show what that process would show. Families and counts are fixed in advance
(COUNTS); signals are drawn blind with os.urandom and sealed like the stream's plan, and the scoring step labels
decoys from that plan, so the build never writes which packets are injected.

A decoy must be visible in what Stage A sees: some Gaia G epoch before the tail cut must move by at least
MIN_SIGMA robust sigma. Signals are redrawn until that holds (or the host is dropped), which is the vetting-stage
analogue of being selected.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from cosmos_search.inject import lightcurve as lcinj
from cosmos_search.inject.classes import gaia_stream
from cosmos_search.inject.sealed import write_sealed
from cosmos_search.vetting import stage0
from cosmos_search.vetting.packet import HOLDOUT_TAIL, robust_sigma

from .rank import DIP_CLASSES

# Target families (rule v1/v2 should keep them) and the rest (should read as EB, YSO or instrumental).
COUNTS = {"dip_train_poisson": 3, "dip_train_primes": 3, "dip_train_geometric": 3, "grey_dip": 3, "blue_dip": 1,
          "chromatic_flip": 1, "secular_fade": 2, "dip_train_periodic": 2, "dust_dip": 1, "step": 1}
MIN_SIGMA = 5.0
MAX_DRAWS = 200
MIN_EXTERNAL = 100          # epochs in at least one external band
G_MATCH = 0.25              # mag: hosts are matched in G to the batch's real candidates


def kept_ids(lc_dir, n_files=400, rng=None):
    """source_ids with a kept light curve, from a random subset of the stream's files."""
    rng = rng or np.random.default_rng()
    files = sorted(ds.dataset(str(lc_dir), format="parquet").files)
    pick = rng.choice(len(files), min(n_files, len(files)), replace=False)
    ids = [pc.unique(ds.dataset(files[i], format="parquet").to_table(columns=["source_id"])["source_id"]).to_numpy()
           for i in pick]
    return np.unique(np.concatenate(ids))


def hosts(lc_dir, ranked, like, exclude, n, rng, tap, g_mag=None):
    """n hosts with kept light curves, not ranked, not injected before, not in a DPAC class that dips, matched in G
    to the real candidates in `like` (one host per real, cycling). g_mag (source_id -> G, e.g. the stream's aux
    table) narrows the pool to the G range before the TAP query, since most kept stars are fainter than the reals."""
    from .rank import context

    ids = np.setdiff1d(kept_ids(lc_dir, rng=rng), np.fromiter(set(ranked) | set(exclude), np.int64))
    targets = np.asarray(like, float)
    if g_mag is not None:
        g = g_mag.reindex(ids).to_numpy(float)
        ids = ids[np.any(np.abs(g[:, None] - targets[None, :]) <= G_MATCH, axis=1)]
    ids = rng.permutation(ids)[: 40 * n]
    ctx = context(ids, tap)
    ctx = ctx[~ctx["best_class_name"].isin(DIP_CLASSES)]
    g = ctx["phot_g_mean_mag"].to_numpy(float)
    chosen = []
    for i in range(n):
        gt = targets[i % len(targets)]
        cand = np.flatnonzero((np.abs(g - gt) <= G_MATCH) & ~np.isin(np.arange(len(g)), chosen))
        if len(cand):
            chosen.append(int(rng.choice(cand)))
    return ctx.iloc[chosen].reset_index(drop=True)


def _visible_g(lc, sid, blend):
    """Stage-0-cleaned G epochs of one host before the tail cut: (t, flux, robust sigma)."""
    clean, _ = stage0.clean_long(lc.filter(pc.equal(lc["source_id"], sid)), blend=blend)
    df = clean.to_pandas()
    g = df[(df["band"].astype(str) == "G") & np.isfinite(df["flux"]) & (df["flux"] > 0)]
    if len(g) < 10:
        return None
    t0, t1 = g["t"].min(), g["t"].max()
    vis = g[g["t"] < t0 + (1 - HOLDOUT_TAIL) * (t1 - t0)]
    f = vis["flux"].to_numpy() / np.median(vis["flux"])
    return vis["t"].to_numpy(), f, robust_sigma(f), (t0, t1)


def plan(host_ctx, lc_dir, external, rng, counts=COUNTS):
    """One sealed-plan record per decoy, in gaia_stream's format, with signals that continue past DR3."""
    from .rank import _blend_map, read_lightcurves

    n_ext = external.groupby(["source_id", "band"]).size().unstack(fill_value=0)
    covered = set(n_ext.index[(n_ext >= MIN_EXTERNAL).any(axis=1)])
    host_ctx = host_ctx[host_ctx["source_id"].isin(covered)]
    order = rng.permutation(len(host_ctx))
    lc = read_lightcurves(lc_dir, host_ctx["source_id"].to_numpy())
    blend = _blend_map(host_ctx)
    t_ext = float(external["t"].max()) if len(external) else None
    fams = [f for f, k in counts.items() for _ in range(k)]
    out, used = [], 0
    for fam in fams:
        while used < len(order):
            row = host_ctx.iloc[order[used]]
            used += 1
            sid = int(row["source_id"])
            vis = _visible_g(lc, sid, blend)
            if vis is None:
                continue
            t, _, s, span = vis
            for _ in range(MAX_DRAWS):
                sig = lcinj.sample(rng, fam, *gaia_stream.DR3_SPAN, t_ext=t_ext)
                if np.max(np.abs(1 - sig.factor(t, "G"))) >= MIN_SIGMA * s:
                    break
            else:
                continue
            frac = blend.get(sid)
            r = float(rng.uniform(0.2, 1.0)) if frac is not None and frac >= gaia_stream.BLEND_MAX else 1.0
            out.append({"source_id": sid, "family": fam, "half": "batch3", "bands": None, "bp_rp_dilution": r,
                        "seed": int(rng.integers(2**62)), "signal": lcinj.truth(sig)})
            break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.dayone.decoys")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hosts")
    h.add_argument("--lightcurves", required=True)
    h.add_argument("--ranked", required=True)
    h.add_argument("--like", required=True, help="queue whose role == ranked rows set the G distribution")
    h.add_argument("--exclude-plan", help="sealed plan whose hosts are excluded (needs COSMOS_REVEAL_KEY)")
    h.add_argument("--aux", help="parquet with source_id, phot_g_mean_mag for the sample (the stream's aux table)")
    h.add_argument("--n", type=int, required=True)
    h.add_argument("--out", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--hosts", required=True)
    p.add_argument("--lightcurves", required=True)
    p.add_argument("--external", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--recipient-file", action="append", required=True)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(int.from_bytes(os.urandom(32), "little"))
    if a.cmd == "hosts":
        from cosmos_search.gaia.archive import tap

        ranked = pd.read_parquet(a.ranked, columns=["source_id"])["source_id"].astype("int64")
        like = pd.read_parquet(a.like)
        exclude = gaia_stream.load_plan(a.exclude_plan).keys() if a.exclude_plan else ()
        g_mag = None
        if a.aux:
            aux = pd.read_parquet(a.aux, columns=["source_id", "phot_g_mean_mag"])
            g_mag = aux.set_index("source_id")["phot_g_mean_mag"]
        out = hosts(a.lightcurves, set(ranked), like.loc[like["role"] == "ranked", "phot_g_mean_mag"], exclude,
                    a.n, rng, tap(), g_mag)
        out.to_parquet(a.out)
        print(f"{len(out)} hosts -> {a.out}")
    else:
        records = plan(pd.read_parquet(a.hosts), a.lightcurves, pd.read_parquet(a.external), rng)
        digest = write_sealed(records, a.out, a.recipient_file)
        # Counts only. Never print anything derived from the plan beyond its size.
        print(f"sealed {len(records)} decoys to {a.out} (sha256 {digest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
