"""Build the local solar-system ephemeris that cosmos_search.vetting.sso_local reads when SkyBoT is unavailable.

    uv venv .venv-sso && uv pip install --python .venv-sso/bin/python assist rebound numpy pandas pyarrow requests
    ulimit -n 4096   # one memory map per snapshot
    .venv-sso/bin/python -I scripts/build_sso.py elements    # MPCORB -> bodies.parquet + initial states
    .venv-sso/bin/python -I scripts/build_sso.py integrate   # ASSIST, every body, a snapshot every STEP days (~2 h)
    .venv-sso/bin/python -I scripts/build_sso.py observers   # Sun, Earth (DE440) and Gaia, TESS (Horizons), hourly

ASSIST and REBOUND are build-time only (a separate venv); sso_local needs numpy and astropy.

Inputs in ~/cosmos-data/mirror/sso (download them there first): MPCORB.DAT.gz (minorplanetcenter.net), linux_p1550p2650.440 and
sb441-n16.bsp (ssd.jpl.nasa.gov, the files ASSIST needs). Every MPCORB orbit is integrated with ASSIST (Sun,
planets, Moon and Pluto from DE440, the 16 largest asteroids, GR) from its epoch to each snapshot, and the
heliocentric ICRF state is stored per snapshot. sso_local propagates the nearest snapshot two-body (at most
STEP/2 days), which the check step measures against Horizons.

Outputs, under ~/cosmos-data/mirror/sso/ephemeris:
    bodies.parquet        number, name, H, G, U, a, e, q, i, class (one row per MPCORB orbit, same order)
    epochs.npy            snapshot times, JD TDB
    states_<k>.npy        (n_bodies, 6) float64 heliocentric ICRF state at epochs[k], au and au/day
    observers.npz         jd_tdb (hourly) and barycentric ICRF positions (au) of sun, earth, gaia, tess
    manifest.json
"""

import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("COSMOS_DATA_DIR", Path.home() / "cosmos-data")).expanduser() / "mirror" / "sso"
OUT = ROOT / "ephemeris"
WORK = ROOT / "work"
START_JD = 2456809.5   # 2014-06-01 TDB, before Gaia's first science observations (2014-07-25)
END_JD = 2461406.5     # 2027-01-01
STEP = 20.0            # days between snapshots (two-body from 10 d away: < 0.15" against Horizons)
GM_SUN = 0.01720209895 ** 2  # au^3/day^2 (Gaussian constant, as MPC elements use)
EPS = np.radians(84381.448 / 3600)  # J2000 obliquity: MPC elements are ecliptic J2000
BATCH = 256
UA = "cosmos-search (mailto:cora.shallit@gmail.com)"


def epochs():
    n = int(round((END_JD - START_JD) / STEP)) + 1
    return START_JD + STEP * np.arange(n)


# --- MPCORB -------------------------------------------------------------------------------------------

def unpack_epoch(s):
    """MPC packed date (e.g. K2555) -> JD TT at 0h."""
    cent = {"I": 1800, "J": 1900, "K": 2000}[s[0]]
    year = cent + int(s[1:3])
    dig = lambda c: int(c) if c.isdigit() else ord(c) - ord("A") + 10
    month, day = dig(s[3]), dig(s[4])
    a = (14 - month) // 12
    y, m = year + 4800 - a, month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return jdn - 0.5


def parse_mpcorb(path):
    rows = []
    with gzip.open(path, "rt", encoding="latin-1") as f:
        started = False
        for line in f:
            if not started:
                started = line.startswith("-----")
                continue
            if len(line) < 103 or not line[:7].strip():
                continue
            try:
                h = float(line[8:13]) if line[8:13].strip() else np.nan
                g = float(line[14:19]) if line[14:19].strip() else 0.15
                rows.append((line[0:7].strip(), line[166:194].strip(), h, g, line[105:106].strip(),
                             unpack_epoch(line[20:25]), float(line[26:35]), float(line[37:46]),
                             float(line[48:57]), float(line[59:68]), float(line[70:79]), float(line[92:103])))
            except (ValueError, KeyError):
                continue
    return rows


def elements_to_state(a, e, i, node, peri, M):
    """Heliocentric ecliptic J2000 elements (degrees) -> ICRF (equatorial) state, au and au/day."""
    i, node, peri, M = (np.radians(x) for x in (i, node, peri, M))
    E = M.copy()
    for _ in range(50):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < 1e-14:
            break
    cosE, sinE = np.cos(E), np.sin(E)
    b = a * np.sqrt(1 - e**2)
    xp, yp = a * (cosE - e), b * sinE
    n = np.sqrt(GM_SUN / a**3)
    edot = n / (1 - e * cosE)
    vxp, vyp = -a * sinE * edot, b * cosE * edot
    co, so, cw, sw, ci, si = np.cos(node), np.sin(node), np.cos(peri), np.sin(peri), np.cos(i), np.sin(i)
    P = np.array([co * cw - so * sw * ci, so * cw + co * sw * ci, sw * si])
    Q = np.array([-co * sw - so * cw * ci, -so * sw + co * cw * ci, cw * si])
    r = P * xp + Q * yp
    v = P * vxp + Q * vyp
    ce, se = np.cos(EPS), np.sin(EPS)
    rot = lambda x: np.array([x[0], ce * x[1] - se * x[2], se * x[1] + ce * x[2]])
    return np.vstack([rot(r), rot(v)]).T


def orbit_class(a, e, q, i):
    Q = a * (1 + e)
    return np.select(
        [q < 1.3, (a > 5.05) & (a < 5.35), (a > 3.7) & (a < 4.2) & (e < 0.3), a > 30.0, a > 5.5,
         (a > 1.78) & (a < 2.0) & (q > 1.3), (a >= 2.0) & (a <= 3.3) & (q > 1.3), (a > 3.3) & (a <= 3.7)],
        ["NEA", "Trojan", "Hilda", "TNO", "Centaur", "Hungaria", "MB", "Cybele"], default="Other")


def cmd_elements(a):
    import pandas as pd

    rows = parse_mpcorb(ROOT / "MPCORB.DAT.gz")
    df = pd.DataFrame(rows, columns=["packed", "name", "H", "G", "U", "epoch_jd", "M", "peri", "node", "i", "e", "a"])
    df = df[(df.e < 1) & (df.a > 0)].reset_index(drop=True)
    df["number"] = df["name"].str.extract(r"^\((\d+)\)")[0].astype("Int64")
    df["name"] = df["name"].str.replace(r"^\(\d+\)\s*", "", regex=True)
    df["q"] = df.a * (1 - df.e)
    df["class"] = orbit_class(df.a.to_numpy(), df.e.to_numpy(), df.q.to_numpy(), df.i.to_numpy())
    # Stored sorted by epoch, then perihelion: integration batches are contiguous rows that share an epoch,
    # with similar orbits together so IAS15's step suits the whole batch.
    df = df.sort_values(["epoch_jd", "q"], kind="stable").reset_index(drop=True)
    state = elements_to_state(*(df[c].to_numpy() for c in ("a", "e", "i", "node", "peri", "M")))
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    # The 16 asteroids ASSIST itself uses as perturbers would attract themselves: take them from sb441 instead.
    ephem = _ephem()
    df["ephem_index"] = -1
    for k in range(11, 27):
        t = df.epoch_jd.mode()[0]
        p, sun = ephem.get_particle(k, t - ephem.jd_ref), ephem.get_particle("sun", t - ephem.jd_ref)
        d = np.linalg.norm(state[:, :3] - np.array([p.x - sun.x, p.y - sun.y, p.z - sun.z]), axis=1)
        d[df.epoch_jd.to_numpy() != t] = np.inf
        j = int(np.argmin(d))
        if d[j] < 1e-3:
            df.loc[j, "ephem_index"] = k
    print("perturbers from the ephemeris:", df.loc[df.ephem_index >= 0, ["number", "name"]].to_dict("records"))
    np.save(WORK / "initial_states.npy", state)
    np.save(WORK / "initial_epochs.npy", df["epoch_jd"].to_numpy())  # TT ~ TDB to 2 ms
    df[["number", "name", "packed", "H", "G", "U", "a", "e", "q", "i", "class", "epoch_jd", "ephem_index"]].to_parquet(
        OUT / "bodies.parquet", index=False)
    print(f"{len(df):,} orbits; epochs: {df.epoch_jd.value_counts().head(3).to_dict()}; "
          f"classes: {df['class'].value_counts().to_dict()}")


# --- integration ---------------------------------------------------------------------------------------

def _ephem():
    import assist

    return assist.Ephem(str(ROOT / "linux_p1550p2650.440"), str(ROOT / "sb441-n16.bsp"))


def integrate_batch(args):
    """Integrate bodies idx (sharing one epoch) to every snapshot; returns (idx, states (n_snap, n, 6))."""
    import assist
    import rebound

    idx, epoch_jd, state, eidx = args
    ephem = _ephem()
    snaps = epochs()
    out = np.full((len(snaps), len(idx), 6), np.nan)
    free = eidx < 0
    for j in np.nonzero(~free)[0]:
        for k, t in enumerate(snaps):
            p, sun = ephem.get_particle(int(eidx[j]), t - ephem.jd_ref), ephem.get_particle("sun", t - ephem.jd_ref)
            out[k, j] = [p.x - sun.x, p.y - sun.y, p.z - sun.z, p.vx - sun.vx, p.vy - sun.vy, p.vz - sun.vz]
    if not free.any():
        return idx, out
    for direction in (-1, 1):
        sim = rebound.Simulation()
        sim.t = epoch_jd - ephem.jd_ref
        sun = ephem.get_particle("sun", sim.t)
        for s in state[free]:
            sim.add(x=s[0] + sun.x, y=s[1] + sun.y, z=s[2] + sun.z,
                    vx=s[3] + sun.vx, vy=s[4] + sun.vy, vz=s[5] + sun.vz)
        sim.ri_ias15.min_dt = 1e-3
        extras = assist.Extras(sim, ephem)
        order = np.nonzero(snaps < epoch_jd)[0][::-1] if direction < 0 else np.nonzero(snaps >= epoch_jd)[0]
        for k in order:
            t = snaps[k] - ephem.jd_ref
            sim.integrate(t)
            sun = ephem.get_particle("sun", t)
            xyz = np.zeros((sim.N, 6))
            sim.serialize_particle_data(xyzvxvyvz=xyz)
            xyz -= np.array([sun.x, sun.y, sun.z, sun.vx, sun.vy, sun.vz])
            out[k, free] = xyz
        del extras
    return idx, out


def cmd_integrate(a):
    state = np.load(WORK / "initial_states.npy")
    ep = np.load(WORK / "initial_epochs.npy")
    import pandas as pd

    eidx = pd.read_parquet(OUT / "bodies.parquet", columns=["ephem_index"])["ephem_index"].to_numpy()
    snaps = epochs()
    n = len(state)
    if a.limit:
        n = a.limit
    jobs = []
    for e in np.unique(ep[:n]):
        ids = np.nonzero(ep[:n] == e)[0]  # contiguous (bodies.parquet is sorted by epoch)
        for s in range(0, len(ids), BATCH):
            b = ids[s:s + BATCH]
            jobs.append((b, float(e), state[b], eidx[b]))
    done_file = WORK / "done.npy"
    done = np.load(done_file) if done_file.exists() and not a.limit else np.zeros(n, bool)
    jobs = [j for j in jobs if not done[j[0]].all()]
    mm = [np.lib.format.open_memmap(WORK / f"states_{k:04d}.npy", mode="r+" if (WORK / f"states_{k:04d}.npy").exists() else "w+",
                                    dtype=np.float64, shape=(len(state), 6)) for k in range(len(snaps))]
    print(f"{len(jobs)} batches to integrate over {len(snaps)} snapshots with {a.workers} workers", flush=True)
    t0, nb = time.time(), 0
    with ProcessPoolExecutor(a.workers) as ex:
        for idx, out in ex.map(integrate_batch, jobs, chunksize=1):
            for k in range(len(snaps)):
                mm[k][idx[0]:idx[-1] + 1] = out[k]
            done[idx] = True
            nb += 1
            if nb % 50 == 0 or nb == len(jobs):
                for m in mm:
                    m.flush()
                np.save(done_file, done)
                el = time.time() - t0
                print(f"  {nb}/{len(jobs)} batches, {done.sum():,} bodies, {el / 60:.1f} min, "
                      f"eta {el / nb * (len(jobs) - nb) / 60:.0f} min", flush=True)
    for m in mm:
        m.flush()
    np.save(done_file, done)
    if not a.limit and done.all():
        np.save(OUT / "epochs.npy", snaps)
        for k in range(len(snaps)):
            os.replace(WORK / f"states_{k:04d}.npy", OUT / f"states_{k:04d}.npy")
        write_manifest()
        print("integration complete", flush=True)


# --- observers -----------------------------------------------------------------------------------------

def horizons_vectors(target, start_jd, stop_jd, step="1h"):
    """Barycentric ICRF positions (au) of a spacecraft from JPL Horizons, as (jd_tdb, xyz)."""
    import requests

    jds, xyz = [], []
    chunk = 3000 / 24  # Horizons returns at most ~90k lines; 125-day chunks at 1 h stay far below
    t = start_jd
    while t < stop_jd:
        t1 = min(t + chunk, stop_jd)
        params = {"format": "json", "COMMAND": f"'{target}'", "EPHEM_TYPE": "VECTORS", "CENTER": "'500@0'",
                  "START_TIME": f"'JD{t:.5f}'", "STOP_TIME": f"'JD{t1:.5f}'", "STEP_SIZE": f"'{step}'",
                  "REF_PLANE": "FRAME", "REF_SYSTEM": "ICRF", "VEC_TABLE": "1", "OUT_UNITS": "AU-D",
                  "CSV_FORMAT": "YES", "TIME_TYPE": "TDB", "OBJ_DATA": "NO"}
        for attempt in range(6):
            r = requests.get("https://ssd.jpl.nasa.gov/api/horizons.api", params=params, headers={"User-Agent": UA},
                             timeout=300)
            if r.ok:
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        text = r.json()["result"]
        if "$$SOE" not in text:
            raise RuntimeError(f"Horizons {target} {t}-{t1}: {text[-500:]}")
        body = text.split("$$SOE")[1].split("$$EOE")[0]
        for line in body.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            jds.append(float(p[0]))
            xyz.append([float(p[2]), float(p[3]), float(p[4])])
        t = t1
    jds, idx = np.unique(np.asarray(jds), return_index=True)
    return jds, np.asarray(xyz)[idx]


def cmd_observers(a):
    ephem = _ephem()
    grid = np.arange(START_JD, END_JD + 1e-9, 1 / 24)
    out = {"jd_tdb": grid}
    for body in ("sun", "earth"):
        p = np.array([[q.x, q.y, q.z] for q in (ephem.get_particle(body, t - ephem.jd_ref) for t in grid)])
        out[body] = p
    # Gaia's science phase ended 2025-01-15 and its ephemeris stops in 2025; TESS launched 2018-04-18.
    for name, target, lo, hi in (("gaia", "-139479", 2456700.5, 2460700.5), ("tess", "-95", 2458250.5, END_JD)):
        jd, xyz = horizons_vectors(target, max(lo, START_JD), min(hi, END_JD))
        pos = np.full((len(grid), 3), np.nan)
        ok = (grid >= jd[0]) & (grid <= jd[-1])
        for c in range(3):
            pos[ok, c] = np.interp(grid[ok], jd, xyz[:, c])
        out[name] = pos
        print(f"{name}: {len(jd)} Horizons rows, {jd[0]:.1f}-{jd[-1]:.1f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "observers.npz", **out)
    write_manifest()


def write_manifest():
    m = {"start_jd_tdb": START_JD, "end_jd_tdb": END_JD, "step_days": STEP,
         "orbits": "MPCORB.DAT (minorplanetcenter.net), file of " + time.strftime(
             "%Y-%m-%d", time.gmtime((ROOT / "MPCORB.DAT.gz").stat().st_mtime)),
         "integrator": "ASSIST (DE440 + sb441-n16), IAS15",
         "observers": "Sun, Earth from DE440; Gaia (-139479), TESS (-95) from JPL Horizons; hourly",
         "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / "manifest.json").write_text(json.dumps(m, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["elements", "integrate", "observers"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    {"elements": cmd_elements, "integrate": cmd_integrate, "observers": cmd_observers}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
