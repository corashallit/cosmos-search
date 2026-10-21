"""Propagate Gaia astrometry, with its full covariance, to another survey's epoch.

The state is the five-parameter Gaia solution in tangent-plane form about the reference position:
p = (xi, eta, parallax, pmra*, pmdec) in mas and mas/yr, with covariance C5 built from the errors and the
ten correlations. At epoch t the predicted offset is J p with

    J = [[1, 0, f_xi(t),  dt, 0 ],
         [0, 1, f_eta(t), 0,  dt]],   dt = t - ref_epoch (Julian years)

and its covariance J C5 J^T. The parallax factors f depend on how the other survey's position was formed
(`parallax_mode`):

- "exact":   a single-epoch position (2MASS, NEOWISE single exposures). f from the Earth's barycentric
             position at t.
- "unknown": a position averaged over epochs at unknown parallactic phases (WISE coadds, eROSITA). f = 0,
             and the parallax adds (plx^2 + sigma_plx^2) / 2 of variance per axis, the mean of sin^2 over phase.
- "none":    a catalogue that already removed parallax (TIC, SIMBAD, VSX positions at J2000). f = 0.

Sources without a parallax or proper motion (Gaia two-parameter solutions) get a prior: the proper motion is
drawn from the local population of faint five-parameter sources in the same table (robust mean and scatter),
and the parallax from a zero-mean prior. Both priors are deliberately broad; the calibration step measures
whether they are broad enough.

Ignored, and why: perspective acceleration (matters only for a few dozen stars within ~10 pc over decades)
and orbital motion of binaries (shows up as excess offset; the calibration's error floor absorbs the typical
case, and outliers are worth seeing).
"""

from dataclasses import dataclass

import numpy as np

JULIAN_YEAR = 365.25
J2000_JD = 2451545.0
MJD_OFFSET = 2400000.5

# Correlation columns in Gaia's order for (ra, dec, parallax, pmra, pmdec).
_PARAMS = ("ra", "dec", "parallax", "pmra", "pmdec")
_CORR = {
    (0, 1): "ra_dec_corr", (0, 2): "ra_parallax_corr", (0, 3): "ra_pmra_corr", (0, 4): "ra_pmdec_corr",
    (1, 2): "dec_parallax_corr", (1, 3): "dec_pmra_corr", (1, 4): "dec_pmdec_corr",
    (2, 3): "parallax_pmra_corr", (2, 4): "parallax_pmdec_corr", (3, 4): "pmra_pmdec_corr",
}

DEFAULT_PM_PRIOR_SIGMA = 10.0  # mas/yr per axis, used when a table has too few faint 5-p sources to measure it
DEFAULT_PLX_PRIOR_SIGMA = 1.0  # mas


def mjd_to_jyear(mjd):
    return 2000.0 + (np.asarray(mjd, dtype=float) + MJD_OFFSET - J2000_JD) / JULIAN_YEAR


def jd_to_jyear(jd):
    return 2000.0 + (np.asarray(jd, dtype=float) - J2000_JD) / JULIAN_YEAR


def jyear_to_jd(jyear):
    return J2000_JD + (np.asarray(jyear, dtype=float) - 2000.0) * JULIAN_YEAR


def earth_barycentric_au(jyear):
    """Barycentric position of the Earth (ICRS axes, au) at Julian-year epochs, from ERFA's epv00."""
    import erfa

    jd = np.atleast_1d(jyear_to_jd(jyear))
    _, pvb = erfa.epv00(np.floor(jd), jd - np.floor(jd))
    return pvb["p"] if pvb.dtype.names else pvb[..., 0, :]


def parallax_factors(ra_deg, dec_deg, jyear):
    """(f_xi, f_eta): the apparent offset in mas per mas of parallax, seen from the Earth at jyear."""
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    x, y, z = np.moveaxis(earth_barycentric_au(jyear), -1, 0)
    f_xi = x * np.sin(a) - y * np.cos(a)
    f_eta = x * np.cos(a) * np.sin(d) + y * np.sin(a) * np.sin(d) - z * np.cos(d)
    return f_xi, f_eta


@dataclass
class Astrometry:
    """Vectorised Gaia astrometry for N sources."""

    ra: np.ndarray        # deg, at ref_epoch
    dec: np.ndarray
    ref_epoch: np.ndarray  # Julian year
    mean: np.ndarray      # (N, 5): 0, 0, parallax, pmra, pmdec (offsets are about ra/dec, so start at 0)
    cov: np.ndarray       # (N, 5, 5)
    solved: np.ndarray    # bool: True if parallax and proper motion were measured

    def __len__(self):
        return len(self.ra)

    @classmethod
    def from_table(cls, t, *, pm_prior=None, plx_prior_sigma=DEFAULT_PLX_PRIOR_SIGMA, ref_epoch=2016.0):
        """Build from a table with Gaia gaia_source column names (astropy Table, pandas, or dict of arrays).

        pm_prior: (mean_pmra, mean_pmdec, sigma) for unsolved sources; measured from the table when None.
        """
        col = lambda name, default=np.nan: _column(t, name, default)
        ra, dec = col("ra"), col("dec")
        n = len(ra)
        epoch = col("ref_epoch", ref_epoch)
        plx, pmra, pmdec = col("parallax"), col("pmra"), col("pmdec")
        solved = np.isfinite(plx) & np.isfinite(pmra) & np.isfinite(pmdec)

        sig = np.column_stack([col("ra_error"), col("dec_error"), col("parallax_error"),
                               col("pmra_error"), col("pmdec_error")])
        corr = np.broadcast_to(np.eye(5), (n, 5, 5)).copy()
        for (i, j), name in _CORR.items():
            r = np.nan_to_num(col(name, 0.0))
            corr[:, i, j] = corr[:, j, i] = r
        sig = np.where(np.isfinite(sig), sig, 0.0)
        cov = corr * sig[:, :, None] * sig[:, None, :]

        if pm_prior is None:
            pm_prior = local_pm_prior(pmra, pmdec, col("phot_g_mean_mag"), solved)
        mu_a, mu_d, mu_sig = pm_prior
        mean = np.zeros((n, 5))
        mean[:, 2] = np.where(solved, plx, 0.0)
        mean[:, 3] = np.where(solved, pmra, mu_a)
        mean[:, 4] = np.where(solved, pmdec, mu_d)
        # Unsolved: keep the position block, replace the rest with independent priors.
        u = ~solved
        cov[u, 2:, :] = 0.0
        cov[u, :, 2:] = 0.0
        cov[u, 2, 2] = plx_prior_sigma**2
        cov[u, 3, 3] = cov[u, 4, 4] = mu_sig**2
        return cls(ra=ra, dec=dec, ref_epoch=epoch, mean=mean, cov=cov, solved=solved)

    def subset(self, idx):
        return Astrometry(self.ra[idx], self.dec[idx], self.ref_epoch[idx], self.mean[idx], self.cov[idx],
                          self.solved[idx])

    def predict(self, jyear, parallax_mode="unknown", idx=None):
        """Predicted tangent-plane offsets at jyear and their covariance.

        jyear is a scalar or one value per row; idx selects rows (e.g. one per candidate pair).
        Returns (offset (M, 2) mas, cov (M, 2, 2) mas^2).
        """
        a = self if idx is None else self.subset(idx)
        m = len(a.ra)
        t = np.broadcast_to(np.asarray(jyear, dtype=float), (m,))
        dt = t - a.ref_epoch
        J = np.zeros((m, 2, 5))
        J[:, 0, 0] = J[:, 1, 1] = 1.0
        J[:, 0, 3] = J[:, 1, 4] = dt
        if parallax_mode == "exact":
            J[:, 0, 2], J[:, 1, 2] = parallax_factors(a.ra, a.dec, t)
        off = np.einsum("mij,mj->mi", J, a.mean)
        cov = np.einsum("mij,mjk,mlk->mil", J, a.cov, J)
        if parallax_mode == "unknown":
            v = 0.5 * (a.mean[:, 2] ** 2 + a.cov[:, 2, 2])
            cov[:, 0, 0] += v
            cov[:, 1, 1] += v
        elif parallax_mode not in ("exact", "none"):
            raise ValueError(f"parallax_mode {parallax_mode!r}")
        return off, cov

    def total_pm(self):
        return np.hypot(self.mean[:, 3], self.mean[:, 4])


def local_pm_prior(pmra, pmdec, gmag, solved, min_sources=30):
    """Robust (mean_pmra, mean_pmdec, sigma) of the faintest solved sources: the best stand-in for the
    unknown motion of a faint unsolved source in the same field."""
    ok = solved & np.isfinite(gmag)
    if ok.sum() >= min_sources:
        faint = ok & (gmag >= np.nanpercentile(gmag[ok], 75))
        if faint.sum() >= min_sources:
            ok = faint
    if ok.sum() < min_sources:
        return 0.0, 0.0, DEFAULT_PM_PRIOR_SIGMA
    a, d = np.median(pmra[ok]), np.median(pmdec[ok])
    mad = lambda x, c: 1.4826 * np.median(np.abs(x - c))
    sigma = float(np.hypot(mad(pmra[ok], a), mad(pmdec[ok], d)) / np.sqrt(2.0))
    return float(a), float(d), max(sigma, 1.0)


def _column(t, name, default):
    try:
        v = t[name]
    except (KeyError, ValueError, IndexError):
        n = len(t["ra"])
        return np.full(n, default, dtype=float)
    v = np.ma.filled(np.ma.asarray(v, dtype=float), np.nan) if np.ma.isMaskedArray(v) else np.asarray(v, dtype=float)
    return v
