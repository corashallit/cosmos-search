"""The window scan (docs/population.md §4): sky discs x distance windows, Poisson likelihood ratio, parametric null.

Counts live on cells: HEALPix nside-64 pixels (nested; a source's pixel is source_id >> 47) x distance shells.
A disc is the set of nside-64 pixels whose centres lie within its radius of an nside-16 pixel centre; a distance
window is one to three contiguous shells or all distances (which also holds stars without a distance). Under the
null the count in each cell is Poisson with the summed lambda, so a null realisation is one Poisson draw per cell.
"""

import healpy as hp
import numpy as np
from scipy import sparse

NSIDE_CELL, NSIDE_CENTRE = 64, 16
RADII_DEG = (1.0, 2.0, 4.0, 8.0)
SHELL_EDGES_PC = (0, 250, 500, 1000, 2000, 4000, 8000, np.inf)
N_SHELLS = len(SHELL_EDGES_PC) - 1           # plus one column for "no distance"
MAX_SHELLS = 3
N_NULL = 199


def cell_pixel(source_id):
    return (np.asarray(source_id, np.int64) >> 47).astype(np.int64)


def shell_of(distance_pc):
    d = np.asarray(distance_pc, float)
    s = np.digitize(d, SHELL_EDGES_PC[1:-1])
    return np.where(np.isfinite(d) & (d > 0), s, N_SHELLS)


def cell_sums(pixel, shell, values):
    """(npix, N_SHELLS + 1) array of summed values."""
    npix = hp.nside2npix(NSIDE_CELL)
    out = np.zeros((npix, N_SHELLS + 1))
    np.add.at(out, (pixel, shell), values)
    return out


def shell_windows():
    """List of (lo, hi) shell index ranges (hi exclusive); (0, N_SHELLS + 1) is all distances."""
    w = [(lo, lo + k) for k in range(1, MAX_SHELLS + 1) for lo in range(N_SHELLS - k + 1)]
    return w + [(0, N_SHELLS + 1)]


class Windows:
    """Disc membership (sparse discs x pixels) and the window grid."""

    def __init__(self, exclude_pixels=None):
        npix = hp.nside2npix(NSIDE_CELL)
        ncen = hp.nside2npix(NSIDE_CENTRE)
        cen = np.array(hp.pix2vec(NSIDE_CENTRE, np.arange(ncen), nest=True)).T
        rows, cols, self.disc = [], [], []
        for ri, r in enumerate(RADII_DEG):
            for c in range(ncen):
                pix = hp.query_disc(NSIDE_CELL, cen[c], np.radians(r), nest=True)
                rows.append(np.full(len(pix), len(self.disc)))
                cols.append(pix)
                self.disc.append((c, ri))
        self.M = sparse.csr_matrix((np.ones(sum(len(x) for x in cols)), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(len(self.disc), npix))
        self.disc = np.array(self.disc)
        lon, lat = hp.pix2ang(NSIDE_CENTRE, np.arange(ncen), nest=True, lonlat=True)
        self.centre_lonlat = np.c_[lon, lat]
        self.shells = shell_windows()
        if exclude_pixels is not None:
            keep = np.ones(npix)
            keep[exclude_pixels] = 0
            self.M = self.M @ sparse.diags(keep)

    def totals(self, cells):
        """(n_discs, n_shell_windows) sums over windows of a cell array."""
        d = np.asarray(self.M @ cells)                       # discs x (shells + unknown)
        cum = np.concatenate([np.zeros((len(d), 1)), np.cumsum(d, axis=1)], axis=1)
        return np.stack([cum[:, hi] - cum[:, lo] for lo, hi in self.shells], axis=1)

    def describe(self, i, j):
        c, ri = self.disc[i]
        lo, hi = self.shells[j]
        dist = "all" if hi > N_SHELLS else f"{SHELL_EDGES_PC[lo]:g}-{SHELL_EDGES_PC[hi]:g} pc"
        lon, lat = self.centre_lonlat[c]
        return {"disc": int(i), "shell_window": int(j), "ra": float(lon), "dec": float(lat),
                "radius_deg": RADII_DEG[ri], "distance": dist, "shells": [int(lo), int(hi)]}


def llr(O, E):
    with np.errstate(divide="ignore", invalid="ignore"):
        v = O * np.log(O / E) - (O - E)
    return np.where((O > E) & (E > 0), v, 0.0)


def null_max(win, lam_cells, n=N_NULL, rng=None):
    """Maximum LLR over windows in n Poisson realisations of the cells."""
    rng = rng or np.random.default_rng()
    E = win.totals(lam_cells)
    out = np.empty(n)
    for k in range(n):
        O = win.totals(rng.poisson(lam_cells).astype(float))
        out[k] = llr(O, E).max()
    return out


def _overlap(win, a, b):
    """Do windows a and b (disc, shell-window) share any cell?"""
    pa, pb = win.M[a[0]].indices, win.M[b[0]].indices
    if not np.intersect1d(pa, pb).size:
        return False
    (la, ha), (lb, hb) = win.shells[a[1]], win.shells[b[1]]
    return la < hb and lb < ha


def candidates(win, obs_cells, lam_cells, null, k_max=5, alpha=0.05):
    """Greedy non-overlapping windows with family-wise p < alpha (p from the max-null)."""
    O, E = win.totals(obs_cells), win.totals(lam_cells)
    L = llr(O, E)
    order = np.argsort(L, axis=None)[::-1]
    picked = []
    for flat in order[:200000]:
        i, j = np.unravel_index(flat, L.shape)
        p = (1 + np.sum(null >= L[i, j])) / (1 + len(null))
        if p >= alpha or len(picked) >= k_max:
            break
        if any(_overlap(win, (i, j), (q["disc"], q["shell_window"])) for q in picked):
            continue
        d = win.describe(i, j)
        d.update(O=float(O[i, j]), E=float(E[i, j]), llr=float(L[i, j]), p_fw=float(p))
        picked.append(d)
    return picked, float(L.max())
