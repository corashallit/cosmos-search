"""Calibration of the EB filter: recall on eclipsing binaries, and how many anomalies it would remove.

    python -m cosmos_search.ebfilter.cal run --lc joint.parquet --ctx ctx.parquet --half A --out cal_A.parquet
    python -m cosmos_search.ebfilter.cal report cal_A.parquet [cal_B.parquet] [--bits 10]

Input: a long light-curve table (source_id, band, t, flux, err, flags) holding Gaia G (BP/RP ignored) and ZTF
epochs for the same stars, and a context table (source_id, phot_g_mean_mag, bp_rp, parallax, parallax_error,
best_class_name, frequency = DPAC EB model frequency or NaN). Stars split into halves A and B by a hash of
source_id: thresholds are chosen on A and reported on B.

Populations (each run on Gaia G alone, on ZTF alone where it makes sense, and on both):

- `eb_synth`: physical eclipsing binaries (inject.classes.eclipsing_binary) in non-EB hosts. Truth is exact.
- `anomaly:<family>`: inject.lightcurve's dip families in the same hosts. Removing one of these is anomaly loss.
- `eb_host:<family>`: an aperiodic dip injected into a real DPAC eclipsing binary. The EB is real, the extra dip is
  the anomaly, and the filter must not explain it by the binary's eclipses.
- `real_eb`: DPAC eclipsing binaries as observed. DPAC's labels come from the Gaia epochs, so only the ZTF-only run
  measures recall without leakage; the Gaia and joint runs show agreement.
- `host`: the non-EB hosts as observed, for the rate at which real queued non-EBs are explained.

Every injected star must look like a queued candidate: after injection its deepest Gaia G epoch is >= SELECT_SIGMA
robust sigma low, lower than its largest brightening is high, and the injection moved that epoch by
>= SELECT_SIGMA. That epoch is the target dip. Signals are redrawn (new phase, new parameters) until the condition
holds, so recall is conditional on Gaia having seen the dip, as it is for every star in the queue.
"""

import argparse
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from cosmos_search.inject import lightcurve as lcinj
from cosmos_search.inject.classes import eclipsing_binary as ebinj
from cosmos_search.vetting.candidate import LightCurve

from . import decide as D

SELECT_SIGMA = 5.0
GAIA_BANDS = ("G",)
ZTF_BANDS = ("ztf_g", "ztf_r", "ztf_i")
EB_CLASSES = ("ECL", "ELL")
TARGET_FAMILIES = ("grey_dip", "dust_dip", "blue_dip", "chromatic_flip", "dip_train_poisson", "dip_train_primes",
                   "dip_train_geometric", "secular_fade", "step")
OTHER_FAMILIES = ("dip_train_periodic",)
HOST_FAMILIES = ("grey_dip", "dip_train_poisson", "dip_train_primes", "dip_train_geometric")
MAX_TRIES = 400
G_SPAN = (-140.0, 905.0)        # DR3 epoch photometry (gaia_stream.DR3_SPAN); trains keep running to the last epoch
SOURCE_KEYS = ("phot_g_mean_mag", "bp_rp", "parallax", "parallax_error")


def half(source_id):
    return "AB"[hashlib.sha256(str(int(source_id)).encode()).digest()[0] & 1]


def _robust(f):
    med = np.median(f)
    s = 1.4826 * np.median(np.abs(f / med - 1))
    return med, s


def _target(t, f_after, f_before=None):
    """Time of the deepest G epoch if the star would be queued (and, with f_before, if the injection made it)."""
    if len(t) < 5:
        return None
    med, s = _robust(f_after)
    if not s > 0:
        return None
    z = (1 - f_after / med) / s
    k = int(np.argmax(z))
    if z[k] < SELECT_SIGMA or z[k] <= np.max(-z):
        return None
    if f_before is not None and (f_before[k] - f_after[k]) / med / s < SELECT_SIGMA:
        return None
    return float(t[k])


def _lcs(df, bands):
    return [LightCurve(b, g.t.to_numpy(), g.flux.to_numpy(), g.err.to_numpy())
            for b, g in df.groupby("band") if b in bands]


def _apply(df, signal):
    # By position: label lookups mix bands whose rows share index labels (dayone.external.apply).
    out = df.copy()
    band, t = out["band"].astype(str).to_numpy(), out["t"].to_numpy(float)
    flux, err = out["flux"].to_numpy(float).copy(), out["err"].to_numpy(float).copy()
    for b in np.unique(band):
        m = band == b
        k = np.clip(signal.factor(t[m], b), 0, None)
        flux[m], err[m] = flux[m] * k, err[m] * np.sqrt(k)
    out["flux"], out["err"] = flux, err
    return out


def _draw(rng, kind, family, src, df, t_lo, t_hi):
    """A signal and the target time, redrawn until the injected star would be queued; (None, None) if never."""
    g = df[df.band == "G"]
    t, f = g.t.to_numpy(), g.flux.to_numpy()
    for _ in range(MAX_TRIES):
        if kind == "eb_synth":
            sig = ebinj.sample(rng, src, t_lo, t_hi)
        else:
            sig = lcinj.sample(rng, family, *G_SPAN, t_ext=t_hi)
        tt = _target(t, f * np.clip(sig.factor(t, "G"), 0, None), f)
        if tt is not None:
            return sig, tt
    return None, None



def _run_modes(df, src, t_target, modes):
    rows = {}
    for mode in modes:
        bands = {"gaia": GAIA_BANDS, "ztf": ZTF_BANDS, "joint": GAIA_BANDS + ZTF_BANDS}[mode]
        lcs = _lcs(df, bands)
        if not lcs:
            continue
        v = D.decide(lcs, src, t_target if mode != "ztf" else None, delta_bits=0.0)
        rows[mode] = v
    return rows


def _job(job):
    kind, family, sid, src, df, seed, modes = job
    rng = np.random.default_rng(seed)
    t_lo, t_hi = float(df.t.min()), float(df.t.max())
    truth = {}
    if kind in ("host", "real_eb"):
        g = df[df.band == "G"]
        tt = _target(g.t.to_numpy(), g.flux.to_numpy())
        if tt is None:
            return []
        inj = df
    else:
        sig, tt = _draw(rng, "eb_synth" if kind == "eb_synth" else "family", family, src, df, t_lo, t_hi)
        if sig is None:
            return [{"kind": kind, "family": family, "source_id": sid, "mode": "none", "drawn": False}]
        inj = _apply(df, sig)
        if kind == "eb_synth":
            truth = {"true_period": sig.period, "true_ecc": sig.ecc, "true_third": sig.third_light,
                     "true_depth_g": float(1 - np.min(sig.factor(np.linspace(0, sig.period, 20001) + sig.t_peri, "G")))}
    out = []
    for mode, v in _run_modes(inj, src, tt, modes).items():
        out.append({"kind": kind, "family": family, "source_id": sid, "mode": mode, "drawn": True, "t_target": tt,
                    **truth, **{k: v.get(k) for k in (
                        "n_epochs", "n_dip_epochs", "delta_bits", "period", "explained", "periodic_unphysical",
                        "duration_ok", "target_covered", "duration_lo_d", "n_cycles", "n_runs_outside",
                        "n_flat_in_window", "secondary", "r_star")}})
    return out


def jobs(lc, ctx, which, n_synth, n_family, n_host_family, rng, min_ztf=100):
    ctx = ctx.set_index("source_id")
    lc = lc[(lc.band.isin(GAIA_BANDS + ZTF_BANDS)) & ((lc["flags"] & 1) == 0)]
    groups = {int(s): g[["band", "t", "flux", "err"]].reset_index(drop=True) for s, g in lc.groupby("source_id")}
    ids = [s for s in groups if half(s) == which and s in ctx.index]
    n_ztf = {s: int(groups[s].band.isin(ZTF_BANDS).sum()) for s in ids}
    is_eb = {s: (ctx.loc[s, "best_class_name"] in EB_CLASSES) or np.isfinite(ctx.loc[s, "frequency"]) for s in ids}
    hosts = [s for s in ids if not is_eb[s] and n_ztf[s] >= min_ztf]
    ebs = [s for s in ids if is_eb[s] and n_ztf[s] >= min_ztf]

    def src(s):
        r = ctx.loc[s]
        return {k: (None if pd.isna(r.get(k)) else float(r.get(k))) for k in SOURCE_KEYS}

    def seed():
        return int(rng.integers(2 ** 63))

    out = []
    for s in hosts:
        out.append(("host", "none", s, src(s), groups[s], seed(), ("gaia", "joint")))
    for s in ebs:
        out.append(("real_eb", "none", s, src(s), groups[s], seed(), ("gaia", "ztf", "joint")))
    for s in rng.choice(hosts, n_synth):
        out.append(("eb_synth", "eclipsing_binary", int(s), src(s), groups[int(s)], seed(), ("gaia", "ztf", "joint")))
    for fam in TARGET_FAMILIES + OTHER_FAMILIES:
        for s in rng.choice(hosts, n_family):
            out.append(("anomaly", fam, int(s), src(s), groups[int(s)], seed(), ("gaia", "joint")))
    for fam in HOST_FAMILIES:
        for s in rng.choice(ebs, n_host_family):
            out.append(("eb_host", fam, int(s), src(s), groups[int(s)], seed(), ("gaia", "joint")))
    return out, {"hosts": len(hosts), "ebs": len(ebs)}


def score(df, bits):
    """Recall and loss tables at a threshold (bits): explained = solution with delta >= bits meets the checks."""
    d = df[df.drawn].copy()
    d["hit"] = d.explained.fillna(False).astype(bool) & (d.delta_bits >= bits)
    return d


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def threshold(df, k_min=7, k_max=60, pooled=0.01, per_family=0.03):
    """The pre-registered rule: smallest integer k >= k_min with joint-mode target-family loss <= pooled overall
    and <= per_family in every target family."""
    d = df[(df.kind == "anomaly") & (df["mode"] == "joint") & df.family.isin(TARGET_FAMILIES)]
    for k in range(k_min, k_max + 1):
        s = score(d, k)
        if s.hit.mean() <= pooled and s.groupby("family").hit.mean().max() <= per_family:
            return k
    return None


def report(df, bits):
    """Markdown tables of recall and loss at a threshold."""
    s = score(df, bits)
    lines = []

    def row(label, sub):
        k, n = int(sub.hit.sum()), len(sub)
        lo, hi = wilson(k, n)
        return f"| {label} | {k}/{n} | {k / n if n else float('nan'):.3f} | {lo:.3f}-{hi:.3f} |"

    head = "| population | explained | rate | 95% CI |\n|---|---|---|---|"
    lines.append(f"Threshold {bits} bits.\n\n" + head)
    for kind, fams in (("eb_synth", None), ("real_eb", None), ("host", None), ("anomaly", TARGET_FAMILIES),
                       ("anomaly", OTHER_FAMILIES), ("eb_host", HOST_FAMILIES)):
        for mode in ("gaia", "ztf", "joint"):
            sub = s[(s.kind == kind) & (s["mode"] == mode)]
            if fams is not None:
                sub = sub[sub.family.isin(fams)]
            if len(sub):
                name = kind if fams is None else f"{kind} ({', '.join(fams) if len(fams) < 3 else 'target'})"
                lines.append(row(f"{name}, {mode}", sub))
    lines.append("\nPer anomaly family (joint):\n\n" + head)
    for fam in TARGET_FAMILIES + OTHER_FAMILIES:
        lines.append(row(fam, s[(s.kind == "anomaly") & (s["mode"] == "joint") & (s.family == fam)]))
    eb = s[(s.kind == "eb_synth") & (s["mode"] == "joint")]
    if len(eb):
        lines.append("\nSynthetic EBs by true period (joint):\n\n" + head)
        bins = [0.3, 1, 3, 10, 30, 100, 300]
        for a, b in zip(bins[:-1], bins[1:]):
            lines.append(row(f"{a}-{b} d", eb[(eb.true_period >= a) & (eb.true_period < b)]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.ebfilter.cal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--lc", required=True)
    r.add_argument("--ctx", required=True)
    r.add_argument("--half", choices=("A", "B"), required=True)
    r.add_argument("--n-synth", type=int, default=1000)
    r.add_argument("--n-family", type=int, default=150)
    r.add_argument("--n-host-family", type=int, default=75)
    r.add_argument("--workers", type=int, default=os.cpu_count())
    r.add_argument("--seed", type=int)
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    p.add_argument("--bits", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "report":
        df = pd.concat([pd.read_parquet(f) for f in a.files], ignore_index=True)
        bits = a.bits if a.bits is not None else threshold(df)
        print(report(df, bits))
        return 0
    seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(8), "little")
    rng = np.random.default_rng(seed)
    lc = pd.read_parquet(a.lc)
    lc["band"] = lc["band"].astype(str)
    ctx = pd.read_parquet(a.ctx)
    js, counts = jobs(lc, ctx, a.half, a.n_synth, a.n_family, a.n_host_family, rng)
    print(f"half {a.half}: {counts}, {len(js)} jobs, seed {seed}", flush=True)
    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for i, res in enumerate(pool.map(_job, js, chunksize=2)):
            rows += res
            if i % 200 == 0:
                print(f"{i}/{len(js)}", flush=True)
    out = pd.DataFrame(rows)
    out.attrs["seed"] = seed
    out["half"] = a.half
    out.to_parquet(a.out)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
