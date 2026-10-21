"""Sealed excess populations for calibrating the half-A search (docs/population.md §8).

A plan draws, per universe, 0-2 regions: a sky disc (centre uniform on the sphere outside the Magellanic zones,
radius log-uniform 1-6 deg) crossed with one of the scan's distance windows. Every star in a region gets one-off
grey dips as a Poisson process over DR3's span, duration log-uniform 0.1-10 d, depth log-uniform 1-30%, trapezoid
with ingress 1-50% of the duration. The rate is set so the expected extra half-A SOV count, from E_gen, is
log-uniform 5-300 (the realised count differs, since the injected durations and depths are not E_gen's).

Dips go into the compact G light curves before the event rule runs, with photon-scaled noise: a flux scaled by
(1 - d) keeps the measured noise scaled by (1 - d), so extra noise of variance sigma^2 ((1 - d) - (1 - d)^2)
brings it to the photon-noise sigma^2 (1 - d).

The plan is drawn from os.urandom and written only as age ciphertext. `apply` decrypts it in memory; the truth of
which events are injected is sealed per universe, and only `score`, run after the candidates are committed,
opens it.
"""

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cosmos_search.vetting import single_event as se

from . import events, scan

N_UNIVERSES = 12
N_REGIONS_P = (0.15, 0.5, 0.35)
RADIUS = (1.0, 6.0)
N_TARGET = (5.0, 300.0)
T_RANGE = (0.1, 10.0)
DEPTH = (0.01, 0.3)
SPAN = (-140.0, 905.0)
_RANGE = re.compile(r"EpochPhotometry_(\d+)-(\d+)")


def _unit_vectors(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]


def draw_plan(stars, rng):
    """stars: assembled table (excluded zones removed). Returns a JSON-able plan."""
    from .analysis import LMC, SMC, _angdist
    shells = scan.shell_windows()
    plan = []
    for u in range(N_UNIVERSES):
        regions = []
        for _ in range(rng.choice(3, p=N_REGIONS_P)):
            while True:
                ra, dec = rng.uniform(0, 360), np.degrees(np.arcsin(rng.uniform(-1, 1)))
                if _angdist(ra, dec, LMC[0], LMC[1]) > LMC[2] + 6 and _angdist(ra, dec, SMC[0], SMC[1]) > SMC[2] + 6:
                    break
            radius = float(np.exp(rng.uniform(*np.log(RADIUS))))
            lo, hi = shells[rng.integers(len(shells))]
            members = region_members(stars, ra, dec, radius, (lo, hi))
            e = float(stars.E_gen_A.to_numpy()[members].sum())
            n_target = float(np.exp(rng.uniform(*np.log(N_TARGET))))
            rate = 1e-3 * n_target / max(e, 1e-9)                   # E_gen is per 1e-3 / day
            regions.append({"ra": ra, "dec": dec, "radius_deg": radius, "shells": [int(lo), int(hi)],
                            "n_target": n_target, "rate_per_day": rate, "n_members": int(members.sum()),
                            "seed": int(rng.integers(2**62))})
        plan.append({"universe": u, "regions": regions})
    return plan


def region_members(stars, ra, dec, radius_deg, shells):
    from .analysis import _angdist
    lo, hi = shells
    sh = stars.shell.to_numpy()
    return (_angdist(stars.ra.to_numpy(), stars.dec.to_numpy(), ra, dec) <= radius_deg) & (sh >= lo) & (sh < hi)


def file_index(lc_dir):
    """(lo, hi, path) of each compact file's HEALPix level-8 range."""
    out = []
    for p in sorted(glob.glob(f"{lc_dir}/*.parquet")):
        m = _RANGE.search(Path(p).name)
        out.append((int(m.group(1)), int(m.group(2)), p))
    return out


def load_light_curves(lc_dir, source_ids, index=None):
    """Compact light curves of the given sources, sorted by (source_id, t)."""
    index = index or file_index(lc_dir)
    ids = np.unique(np.asarray(source_ids, np.int64))
    pix8 = ids >> 43
    frames = []
    for lo, hi, p in index:
        sel = ids[(pix8 >= lo) & (pix8 <= hi)]
        if len(sel):
            t = pq.read_table(p).to_pandas()
            frames.append(t[t.source_id.isin(sel)])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["source_id", "t"], kind="stable").reset_index(drop=True)


def inject_dips(lc, rate_per_day, rng):
    """Add grey one-off dips to compact light curves. Returns the modified copy and the dips (source_id, t0, T)."""
    lc = lc.copy()
    sids = lc.source_id.unique()
    n = rng.poisson(rate_per_day * (SPAN[1] - SPAN[0]), len(sids))
    rec = []
    flux = lc.flux.to_numpy(np.float64).copy()
    err = lc.err_rel.to_numpy(np.float64)
    t = lc.t.to_numpy(np.float64)
    gid = np.searchsorted(sids, lc.source_id.to_numpy())   # sids from unique() of a sorted column are sorted
    med = pd.Series(flux).groupby(gid).transform("median").to_numpy()
    for s, k in zip(range(len(sids)), n):
        if k == 0:
            continue
        rows = np.flatnonzero(gid == s)
        for _ in range(k):
            T = float(np.exp(rng.uniform(*np.log(T_RANGE))))
            d = float(np.exp(rng.uniform(*np.log(DEPTH))))
            tau = T * rng.uniform(0.01, 0.5)
            t0 = rng.uniform(SPAN[0] - T, SPAN[1])
            prof = se._trapezoid(t[rows] - (t0 + T / 2), T, tau, d)
            hit = prof > 0
            if hit.any():
                keep = 1 - prof[hit]
                sig = err[rows][hit] * med[rows][hit]
                extra = np.sqrt(np.clip(keep - keep**2, 0, None)) * np.nan_to_num(sig)
                flux[rows[hit]] = flux[rows[hit]] * keep + extra * rng.standard_normal(hit.sum())
            rec.append((int(sids[s]), t0, T, d))
    lc["flux"] = flux.astype(np.float32)
    return lc, pd.DataFrame(rec, columns=["source_id", "t0", "T", "depth"])


def injected_events(lc_inj, dips):
    """SOV events (all halves) on injected light curves, each marked injected if its deepest epoch is in a dip."""
    rows, st = events.prepare(lc_inj)
    ev = events.find(rows, st)
    ev = ev[ev.cls == "SOV"].copy()
    ev["injected"] = False
    if len(dips) and len(ev):
        m = ev.merge(dips, on="source_id", how="left")
        inside = (m.t_deep >= m.t0) & (m.t_deep <= m.t0 + m["T"])
        key = set(zip(m.loc[inside, "source_id"], m.loc[inside, "t_deep"]))
        ev["injected"] = [(s, t) in key for s, t in zip(ev.source_id, ev.t_deep)]
    return ev, st.source_id.to_numpy()


def secret_seed():
    return int.from_bytes(os.urandom(8), "little") >> 2
