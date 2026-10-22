"""Non-gravitational injections into real DR4 pre-release epoch astrometry (plan A3).

The 12 pre-release sources are the only real per-transit abscissae before DR4. Each source is run through the cascade
as published (the null), then with step / kick / thrust / square signals added to its abscissae at a few amplitudes
(units of the source's median per-transit sigma). The real noise, cadence, outliers and any unmodelled systematics
stay in; only the signal is synthetic (and the flux; see source_data).

    python scripts/astrometry_real_inject.py --transits /mnt/project-files/astrometry/prerelease/transits.parquet \\
        --out runs/real_inject.parquet
"""

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

from cosmos_search.astrometry import detect
from cosmos_search.astrometry import simulate as S


def source_data(g, rng):
    """Per-transit data dict for the cascade: formal sigma inflated by the intra-transit chi2 and a 0.02 mas floor.

    The pre-release has no per-transit flux, and DR4 will. Without it a step hides inside a microlensing fit whose
    magnification nothing can check, so each source gets the flux of a constant star with DR4-like noise."""
    k = max(1.0, np.sqrt(g.chi2_intra.sum() / g.dof_intra.sum()))
    sig = np.sqrt((g.sigma_w.to_numpy() * k) ** 2 + 0.02 ** 2)
    fe = S.flux_noise(float(g.g_mag.iloc[0]))
    return {"t": g.t.to_numpy(), "psi": g.psi.to_numpy(), "f_al": g.f_al.to_numpy(), "w": g.w.to_numpy(),
            "sigma": sig, "fov": g.fov.to_numpy(), "multi_source": g.multi_source.to_numpy(),
            "flux": 1 + rng.normal(0, fe, len(g)), "flux_err": np.full(len(g), fe)}


def _one(job):
    sid, kind, amp, seed, data = job
    rng = np.random.default_rng(seed)
    truth = {"source_id": sid, "kind": kind, "amp": amp}
    if kind != "null":
        a = amp * float(np.median(data["sigma"]))
        dx, dy, extra = S.nongrav_offset(rng, kind, data["t"], a)
        data = {**data, "w": data["w"] + S.al(dx, dy, data["psi"])}
        truth.update(amp_mas=a, **extra)
    try:
        out = detect.cascade(data)
    except Exception as e:  # noqa: BLE001 -- recorded, scored as not flagged
        out = {"error": repr(e)[:200]}
    return {**truth, **out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4, help="injections per source, kind and amplitude")
    ap.add_argument("--amps", type=float, nargs="+", default=[5.0, 10.0, 20.0, 50.0])
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    tr = pd.read_parquet(a.transits).sort_values(["source_id", "t"])
    jobs, seed = [], a.seed * 100000
    rng = np.random.default_rng(a.seed)
    for sid, g in tr.groupby("source_id"):
        data = source_data(g, rng)
        jobs.append((int(sid), "null", 0.0, seed, data))
        for kind in S.NONGRAV:
            for amp in a.amps:
                for _ in range(a.n):
                    seed += 1
                    jobs.append((int(sid), kind, amp, seed, data))
    with Pool(a.procs) as pool:
        rows = pool.map(_one, jobs, chunksize=2)
    pd.DataFrame(rows).to_parquet(a.out)
    print(len(rows), "rows ->", a.out)


if __name__ == "__main__":
    main()
