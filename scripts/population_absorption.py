"""Does the DR4 null absorb real excesses? (docs/population.md §9.) Catalogue-level injections into DR3 half A.

Each universe adds Poisson(f * lambda_i) events to the stars of one region (a sky disc x shell window), so the
excess has the null's own make-up; then the frozen null and the DR4 null are refitted and the scan rerun. A region
counts as recovered when a family-wise significant window overlaps it. Regions avoid the real half-A windows. The
light-curve injections (calibrate.py) test the event rule; this tests only the null and scan, so it injects counts.
Universe draws use urandom; the scoring is mechanical."""
import argparse, json, os
import healpy as hp
import numpy as np, pandas as pd
from cosmos_search.population import analysis, glm, run, scan

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/home/user/pop/full")
ap.add_argument("--density", default="/home/user/pop/aux/density_hpx6.parquet")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--write", default="/home/user/pop/dr4null/absorption.json")
a = ap.parse_args()
rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))

stars = analysis.load_stars(a.out, density=a.density)
evA = run.events_A(a.out); mA = np.load(f"{a.out}/mask_A.npy")
y0 = analysis.outcome(stars, evA, 0, mA)
y1 = analysis.outcome(stars, evA, 0, mA, cls="SOV1")
E = stars.E_gen_A.to_numpy()
codes, _ = glm.covariate_codes(stars)
frozen = [t for t in json.load(open("population/candidates_A.json"))["model"]["terms"] if t != "cmd"]
MODELS = {"frozen": frozen, "dr4": list(glm.DR4)}
win = scan.Windows()
px, sh = stars.pixel.to_numpy(), stars.shell.to_numpy()
real = json.load(open("population/candidates_A.json"))["candidates"]
base_lam = glm.fit(codes, y0, E, frozen)["predict"](codes, E)
nulls, phi_mon = {}, {}
for k, t in MODELS.items():
    lam = glm.fit(codes, y0, E, t)["predict"](codes, E)
    nulls[k] = scan.null_max(win, scan.cell_sums(px, sh, lam), rng=np.random.default_rng(11))
    phi_mon[k] = analysis.monitor_dispersion(stars, y1, E, t, win, curve=True)
print("nulls ready", {k: float(np.quantile(v, 0.95)) for k, v in nulls.items()}, flush=True)

vec = np.array(hp.pix2vec(scan.NSIDE_CELL, np.arange(hp.nside2npix(scan.NSIDE_CELL)), nest=True)).T
shells = scan.shell_windows()


def draw():
    while True:
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        lon, lat = hp.vec2ang(v, lonlat=True)
        if any(analysis._angdist(lon[0], lat[0], c["ra"], c["dec"]) < 8 + c["radius_deg"] + 2 for c in real):
            continue
        radius = float(rng.choice([2.0, 4.0, 8.0, 15.0]))
        lo, hi = shells[rng.integers(len(shells))]
        pix = hp.query_disc(scan.NSIDE_CELL, v, np.radians(radius), nest=True)
        mem = np.isin(px, pix) & (sh >= lo) & (sh < hi)
        e_in = base_lam[mem].sum()
        if e_in < 20:
            continue
        n_add = float(np.exp(rng.uniform(np.log(10), np.log(600))))
        return dict(ra=float(lon[0]), dec=float(lat[0]), radius=radius, shells=[int(lo), int(hi)], e_in=float(e_in),
                    n_add_target=n_add, frac=n_add / e_in), mem, pix


out = []
for u in range(a.n):
    reg, mem, pix = draw()
    y = y0.copy()
    add = rng.poisson(reg["frac"] * base_lam[mem])
    y[mem] += add
    reg["n_added"] = int(add.sum())
    for k, t in MODELS.items():
        lam = glm.fit(codes, y, E, t)["predict"](codes, E)
        obs, lc = scan.cell_sums(px, sh, y), scan.cell_sums(px, sh, lam)
        if k == "frozen":        # the DR3 rule: one phi per scale from SOV alone
            variants = {"phi_sov": scan.dispersion(win, obs, lc)[0]}
        else:                    # the DR4 rule: SOV and SOV-1 curves in E, the larger at each window's E
            variants = {"dr4_rule": scan.phi_from_curves(win, lc, scan.dispersion_curve(win, obs, lc), phi_mon[k])}
        for vname, ph in variants.items():
            cands, _ = scan.candidates(win, obs, lc, nulls[k], k_max=10, phi=ph)
            hit = [c for c in cands if np.intersect1d(win.M[c["disc"]].indices, pix).size
                   and c["shells"][0] < reg["shells"][1] and reg["shells"][0] < c["shells"][1]]
            reg[f"{k}/{vname}"] = {"recovered": bool(hit), "p": hit[0]["p_fw"] if hit else None,
                                   "O_in": float(y[mem].sum()), "E_in": float(lam[mem].sum())}
    out.append(reg)
    print(json.dumps(reg), flush=True)
    json.dump(out, open(a.write, "w"), indent=1)
print("done", flush=True)
