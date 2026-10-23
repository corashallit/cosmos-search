"""Known solar-system bodies near a line of sight, computed locally: the fallback for sso.py when SkyBoT is down.

The ephemeris is built once by scripts/build_sso.py (docs/storage.md): every MPCORB orbit (~1.6 M asteroids, no
comets) integrated with ASSIST (DE440 plus the 16 largest asteroids) and stored as heliocentric ICRF states every
`step_days`, with hourly barycentric positions of the Sun, the Earth, Gaia and TESS. A query takes the nearest
snapshot, moves every body two-body to the light-emission time, and keeps those within the radius as seen from
the spacecraft itself, so no parallax correction is needed (contrast sso.nearby).

Rows have the keys sso.parse gives, plus sep_arcsec (observer-centric) and source="local". v_mag is the H, G
magnitude; pos_err_arcsec is NaN (MPCORB gives no positional uncertainty, only the U parameter, kept as "U").
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

GM_SUN = 0.01720209895 ** 2   # au^3/day^2, as the MPC elements and build_sso use
C_AU_DAY = 173.1446326742403  # speed of light, au/day
OBSERVERS = {"gaia": "gaia", "tess": "tess", "500": "earth", "geocentre": "earth"}


def default_dir():
    root = os.environ.get("COSMOS_SSO_DIR")
    if root:
        return Path(root)
    from cosmos_search.net.local import data_dir

    d = data_dir()
    return d / "mirror" / "sso" / "ephemeris" if d else None


class Ephemeris:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.epochs = np.load(self.root / "epochs.npy")
        obs = np.load(self.root / "observers.npz")
        self.obs_jd = obs["jd_tdb"]
        self.obs = {k: obs[k] for k in obs.files if k != "jd_tdb"}
        self._bodies = None

    @property
    def bodies(self):
        if self._bodies is None:
            import pandas as pd

            self._bodies = pd.read_parquet(self.root / "bodies.parquet")
        return self._bodies

    def covers(self, jd_tdb, observer="earth"):
        half = (self.epochs[1] - self.epochs[0]) / 2 if len(self.epochs) > 1 else 0.0
        if not (self.epochs[0] - half <= jd_tdb <= self.epochs[-1] + half):
            return False
        p = self.position(observer, jd_tdb)
        return bool(np.all(np.isfinite(p)))

    def position(self, name, jd_tdb):
        """Barycentric ICRF position (au) of sun, earth, gaia or tess at jd_tdb, interpolated from the hourly grid."""
        p = self.obs[name]
        return np.array([np.interp(jd_tdb, self.obs_jd, p[:, c], left=np.nan, right=np.nan) for c in range(3)])

    def states(self, k):
        return np.load(self.root / f"states_{k:04d}.npy", mmap_mode="r")

    def heliocentric(self, jd_tdb):
        """Heliocentric ICRF positions and velocities (au, au/day) of every body at jd_tdb (scalar or per body)."""
        k = int(np.argmin(np.abs(self.epochs - np.median(jd_tdb))))
        s = np.asarray(self.states(k))
        return propagate(s[:, :3], s[:, 3:], np.asarray(jd_tdb) - self.epochs[k])


def propagate(r0, v0, dt, mu=GM_SUN):
    """Two-body propagation of elliptic states (N, 3) by dt days (scalar or (N,)), vectorised; f and g functions in
    the eccentric-anomaly difference. Rows that are not elliptic come back NaN."""
    r0, v0 = np.asarray(r0, float), np.asarray(v0, float)
    dt = np.broadcast_to(np.asarray(dt, float), (len(r0),))
    rn = np.linalg.norm(r0, axis=1)
    v2 = np.einsum("ij,ij->i", v0, v0)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = 1.0 / (2.0 / rn - v2 / mu)
        ok = a > 0
        a = np.where(ok, a, np.nan)
        sa = np.sqrt(a)
        n = np.sqrt(mu / a**3)
        sigma0 = np.einsum("ij,ij->i", r0, v0) / np.sqrt(mu)
        M = n * dt
        dE = M.copy()
        for _ in range(30):
            s, c = np.sin(dE), np.cos(dE)
            f = dE - (1 - rn / a) * s + sigma0 / sa * (1 - c) - M
            fp = 1 - (1 - rn / a) * c + sigma0 / sa * s
            step = f / fp
            dE = dE - step
            if np.nanmax(np.abs(step)) < 1e-13:
                break
        s, c = np.sin(dE), np.cos(dE)
        r = a + (rn - a) * c + sigma0 * sa * s
        F = 1 - a / rn * (1 - c)
        G = dt + (s - dE) / n
        Fd = -np.sqrt(mu * a) / (r * rn) * s
        Gd = 1 - a / r * (1 - c)
    pos = F[:, None] * r0 + G[:, None] * v0
    vel = Fd[:, None] * r0 + Gd[:, None] * v0
    return pos, vel


def _unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _radec(v):
    v = np.atleast_2d(v)
    d = np.linalg.norm(v, axis=1)
    return np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360, np.degrees(np.arcsin(v[:, 2] / d))


def hg_mag(H, G, r, delta, alpha):
    """V magnitude from the H, G system (Bowell et al. 1989)."""
    t = np.tan(alpha / 2)
    phi1 = np.exp(-3.33 * t**0.63)
    phi2 = np.exp(-1.87 * t**1.22)
    with np.errstate(divide="ignore", invalid="ignore"):
        return H + 5 * np.log10(r * delta) - 2.5 * np.log10((1 - G) * phi1 + G * phi2)


def jd_utc_to_tdb(jd_utc):
    from astropy.time import Time

    return float(Time(jd_utc, format="jd", scale="utc").tdb.jd)


def cone(jd_utc, ra, dec, radius_arcsec, observer="earth", eph=None, prefilter_deg=5.0):
    """Bodies within radius_arcsec of (ra, dec) seen from observer (gaia, tess, earth) at jd_utc, astrometric
    (light-time corrected) ICRF positions. Same keys as sso.parse, plus sep_arcsec, U and source."""
    eph = eph or load()
    obs_name = OBSERVERS.get(observer, observer)
    t = jd_utc_to_tdb(jd_utc)
    if not eph.covers(t, obs_name):
        raise LookupError(f"local ephemeris does not cover JD {jd_utc:.3f} for {obs_name}")
    o = eph.position(obs_name, t)
    sun = eph.position("sun", t)
    u0 = _unit(ra, dec)
    # Geometric positions first, to drop everything far from the line of sight before the light-time iteration.
    p, v = eph.heliocentric(t)
    d = p + sun - o
    dn = np.linalg.norm(d, axis=1)
    cosang = d @ u0 / dn
    near = np.nonzero(cosang >= np.cos(np.radians(prefilter_deg)))[0]
    if not len(near):
        return []
    k = int(np.argmin(np.abs(eph.epochs - t)))
    s = np.asarray(eph.states(k))[near]
    tau = dn[near] / C_AU_DAY
    for _ in range(3):
        te = t - tau
        pe, ve = propagate(s[:, :3], s[:, 3:], te - eph.epochs[k])
        sun_e = np.array([np.interp(te, eph.obs_jd, eph.obs["sun"][:, c]) for c in range(3)]).T
        d = pe + sun_e - o
        tau = np.linalg.norm(d, axis=1) / C_AU_DAY
    dn = np.linalg.norm(d, axis=1)
    sep = np.degrees(np.arccos(np.clip(d @ u0 / dn, -1, 1))) * 3600
    hit = np.nonzero(sep <= radius_arcsec)[0]
    if not len(hit):
        return []
    rows = eph.bodies.iloc[near[hit]]
    r_h = np.linalg.norm(pe[hit], axis=1)
    alpha = np.arccos(np.clip(np.einsum("ij,ij->i", pe[hit], d[hit]) / (r_h * dn[hit]), -1, 1))
    vmag = hg_mag(rows["H"].to_numpy(float), rows["G"].to_numpy(float), r_h, dn[hit], alpha)
    # Apparent rate: the sky-plane part of the relative velocity (observer velocity from the hourly grid).
    o_vel = (eph.position(obs_name, t + 1 / 48) - eph.position(obs_name, t - 1 / 48)) * 24
    sun_vel = (eph.position("sun", t + 1 / 48) - eph.position("sun", t - 1 / 48)) * 24
    rel_v = ve[hit] + sun_vel - o_vel
    los = d[hit] / dn[hit, None]
    perp = rel_v - np.einsum("ij,ij->i", rel_v, los)[:, None] * los
    rate = np.linalg.norm(perp, axis=1) / dn[hit] * 206264.806 / 24  # arcsec/h
    ras, decs = _radec(d[hit])
    out = []
    for j, (_, b) in enumerate(rows.iterrows()):
        num = "" if b["number"] is None or (isinstance(b["number"], float) and np.isnan(b["number"])) else str(b["number"])
        out.append({"name": str(b["name"]), "number": num if num not in ("<NA>", "nan") else "", "class": str(b["class"]),
                    "v_mag": float(vmag[j]), "ra": float(ras[j]), "dec": float(decs[j]), "dist_au": float(dn[hit][j]),
                    "pos_err_arcsec": float("nan"), "U": str(b["U"]), "geo_sep_arcsec": float("nan"),
                    "rate_arcsec_h": float(rate[j]), "sep_arcsec": float(sep[hit][j]), "source": "local"})
    return sorted(out, key=lambda x: x["sep_arcsec"])


@lru_cache(maxsize=1)
def _load(root):
    return Ephemeris(root)


def load(root=None):
    root = Path(root) if root else default_dir()
    if root is None or not (root / "manifest.json").exists() or not (root / "epochs.npy").exists():
        raise FileNotFoundError(f"no local solar-system ephemeris at {root} (build it with scripts/build_sso.py)")
    return _load(str(root))


def available(root=None):
    try:
        load(root)
        return True
    except FileNotFoundError:
        return False
