"""What the matcher needs to know about each survey: epoch, position errors, beam, where to read it.

Every number here is a starting value. The error scale and systematic floor are re-fitted from the data
on every run (calibrate.py), and docs/infra/xmatch.md records the fitted values per survey.
"""

from dataclasses import dataclass, field

import numpy as np

from .astrometry import jd_to_jyear, mjd_to_jyear
from .geometry import MAS_PER_ARCSEC

HATS = "https://data.lsdb.io/hats"
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP"
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap"


@dataclass(frozen=True)
class Survey:
    name: str
    ra: str
    dec: str
    id: str
    error: tuple                 # (kind, *columns), see covariance()
    epoch: float                 # nominal epoch, Julian year
    epoch_col: str | None = None  # per-row epoch, if the catalogue has one
    epoch_unit: str = "jyear"    # unit of epoch_col: "mjd", "jd", "jyear"
    epoch_span: float = 0.0      # years: half-range of contributing epochs (sets the search slack for fast movers)
    parallax_mode: str = "unknown"
    floor_mas: float = 50.0      # starting systematic floor per axis
    search_radius_mas: float = 5000.0
    shift_annulus_arcsec: tuple = (60.0, 120.0)  # shifted-sky offsets and the local-density annulus
    fwhm_arcsec: float = 2.0
    group_by: tuple = ()         # columns whose values each hold at most one counterpart (e.g. NEOWISE frame)
    group_fn: object = None      # or a function rows -> key array, when the key is derived (ZTF objectid)
    keep: tuple = ()             # survey columns to carry into the match table
    hats: str | None = None
    tap: tuple | None = None     # (service url, table)
    tap_pos: tuple = ("ra", "dec")  # the TAP table's indexed position columns, for cone queries
    prior: float = 0.5           # P(a Gaia source has a counterpart) before calibration
    notes: str = ""

    @property
    def columns(self):
        cols = [self.ra, self.dec, self.id, *(c for c in self.error[1:] if isinstance(c, str)),
                *self.group_by, *self.keep]
        if self.epoch_col:
            cols.append(self.epoch_col)
        return list(dict.fromkeys(c for c in cols if c))

    def epochs(self, rows):
        if not self.epoch_col:
            return np.full(len(rows), self.epoch)
        v = _float(rows[self.epoch_col])
        conv = {"mjd": mjd_to_jyear, "jd": jd_to_jyear, "jyear": lambda x: x}[self.epoch_unit]
        v = conv(v)
        return np.where(np.isfinite(v), v, self.epoch)

    def covariance(self, rows):
        """Catalogue position covariance (N, 2, 2) in mas^2, axes (RA*cos dec, Dec)."""
        kind, *cols = self.error
        n = len(rows)
        cov = np.zeros((n, 2, 2))
        if kind == "radec":          # sigra, sigdec, sigradec (WISE family; co-sigma = sign * sqrt|cov|)
            sx, sy, sxy = (_float(rows[c]) for c in cols)
            unit = MAS_PER_ARCSEC
            cov[:, 0, 0], cov[:, 1, 1] = (sx * unit) ** 2, (sy * unit) ** 2
            cov[:, 0, 1] = cov[:, 1, 0] = np.sign(sxy) * (sxy * unit) ** 2
        elif kind == "ellipse":      # semi-major, semi-minor, position angle E of N; cols + (unit_mas,)
            a, b, pa = (_float(rows[c]) for c in cols[:3])
            unit = float(cols[3]) if len(cols) > 3 else MAS_PER_ARCSEC
            th = np.radians(pa)
            # major axis direction in (east, north): (sin pa, cos pa)
            ux, uy = np.sin(th), np.cos(th)
            a2, b2 = (a * unit) ** 2, (b * unit) ** 2
            cov[:, 0, 0] = a2 * ux**2 + b2 * uy**2
            cov[:, 1, 1] = a2 * uy**2 + b2 * ux**2
            cov[:, 0, 1] = cov[:, 1, 0] = (a2 - b2) * ux * uy
        elif kind == "axes":         # sigma_ra*, sigma_dec in the given unit (mas per unit)
            sx, sy = _float(rows[cols[0]]), _float(rows[cols[1]])
            unit = float(cols[2])
            cov[:, 0, 0], cov[:, 1, 1] = (sx * unit) ** 2, (sy * unit) ** 2
        elif kind == "circular":     # one per-axis sigma column, in the given unit
            s = _float(rows[cols[0]]) * float(cols[1])
            cov[:, 0, 0] = cov[:, 1, 1] = s**2
        elif kind == "none":         # no per-source errors: the floor carries it
            pass
        else:
            raise ValueError(f"error kind {kind!r}")
        return np.where(np.isfinite(cov), cov, 0.0)

    def group_keys(self, rows):
        """Raw group key per row (comparable across fetches), as strings; "" when ungrouped."""
        if self.group_fn is not None:
            return np.asarray(self.group_fn(rows)).astype(str)
        if not self.group_by:
            return np.full(len(rows), "", dtype=object)
        return np.array(["|".join(map(str, t)) for t in zip(*(np.asarray(rows[c]) for c in self.group_by))],
                        dtype=object)

    def groups(self, rows):
        if self.group_fn is not None:
            _, inv = np.unique(np.asarray(self.group_fn(rows)), return_inverse=True)
            return inv.astype(np.int64)
        if not self.group_by:
            return np.zeros(len(rows), dtype=np.int64)
        keys = [np.asarray(rows[c]) for c in self.group_by]
        if len(keys) == 1:
            _, inv = np.unique(keys[0], return_inverse=True)
        else:
            _, inv = np.unique(np.rec.fromarrays(keys), return_inverse=True)
        return inv.astype(np.int64)


GAIA_ASTROMETRY = (
    "ref_epoch", "parallax", "pmra", "pmdec", "ra_error", "dec_error", "parallax_error", "pmra_error",
    "pmdec_error", "ra_dec_corr", "ra_parallax_corr", "ra_pmra_corr", "ra_pmdec_corr", "dec_parallax_corr",
    "dec_pmra_corr", "dec_pmdec_corr", "parallax_pmra_corr", "parallax_pmdec_corr", "pmra_pmdec_corr",
)

GAIA_DR3 = Survey(
    "gaia_dr3", ra="ra", dec="dec", id="source_id", error=("none",), epoch=2016.0, parallax_mode="none",
    keep=(*GAIA_ASTROMETRY, "phot_g_mean_mag", "phot_rp_mean_mag", "phot_bp_mean_mag", "bp_rp"),
    hats=f"{HATS}/gaia_dr3/gaia",
    notes="Gaia itself: the claimants that compete for survey sources (compete.py) and blending (blend.py).",
)


def _ztf_quadrant(objectid):
    """ZTF objectid is field (3-4 digits), readout channel (2), filter (1), then a 7-digit serial."""
    return np.asarray(objectid, dtype=np.int64) // 10**7


def _float(v):
    v = np.ma.filled(np.ma.asarray(v, dtype=float), np.nan) if np.ma.isMaskedArray(v) else np.asarray(v, dtype=float)
    return v


SURVEYS = {s.name: s for s in [
    Survey(
        "2mass", ra="ra", dec="decl", id="designation", error=("ellipse", "err_maj", "err_min", "err_ang"),
        epoch=1999.0, epoch_col="jdate", epoch_unit="jd", parallax_mode="exact",
        floor_mas=40.0, search_radius_mas=3000.0, shift_annulus_arcsec=(45.0, 90.0), fwhm_arcsec=2.5,
        keep=("j_m", "h_m", "k_m", "j_msigcom", "h_msigcom", "k_msigcom", "ph_qual", "cc_flg", "bl_flg", "gal_contam"),
        hats=f"{HATS}/two_mass/two_mass",
        notes="Single-epoch positions (1997-2001); jdate per source, so parallax is applied exactly.",
    ),
    Survey(
        "allwise", ra="ra", dec="dec", id="designation", error=("radec", "sigra", "sigdec", "sigradec"),
        epoch=2010.55, epoch_span=0.6,
        floor_mas=50.0, search_radius_mas=4000.0, shift_annulus_arcsec=(60.0, 120.0), fwhm_arcsec=6.1,
        keep=("w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro", "w3mpro", "w3sigmpro", "w4mpro", "w4sigmpro",
              "w1snr", "w3snr", "w4snr", "cc_flags", "ext_flg", "ph_qual", "nb", "na", "w3rchi2", "w4rchi2"),
        hats=f"{HATS}/wise/allwise",
        notes="Coadd of 2010.0-2011.1. W3/W4 for energy-budget tests; W3/W4 detections near bright stars "
              "are often spurious, so keep cc_flags and the rchi2 columns. Positions sit at a per-source epoch "
              "(39 stars faster than 1.5 arcsec/yr: 2010.51 +- 0.12); w1mjdmean tracks it only roughly (off by "
              "0.2-0.3 yr where the W1 dates span one visit), so a fixed epoch is kept. The data.lsdb.io copy "
              "was column-shifted from use_src on until its 2026-10-23 re-import (astronomy-commons/lsdb#1660); "
              "the new one agrees with IRSA in every column.",
    ),
    Survey(
        "catwise2020", ra="ra_pm", dec="dec_pm", id="source_name",
        error=("radec", "sigra_pm", "sigdec_pm", "sigradec_pm"),
        epoch=2015.405, parallax_mode="unknown",
        floor_mas=50.0, search_radius_mas=4000.0, shift_annulus_arcsec=(60.0, 120.0), fwhm_arcsec=6.1,
        keep=("w1mpro_pm", "w2mpro_pm", "w1snr_pm", "w2snr_pm", "pmra", "pmdec", "sigpmra", "sigpmdec",
              "ab_flags", "cc_flags", "nb", "na", "meanobsmjd"),
        tap=(IRSA_TAP, "catwise_2020"),
        notes="Motion-fit positions at MJD 57170 (2015.405), close to Gaia's epoch, so proper motion barely "
              "matters. Deeper than AllWISE in W1/W2; no W3/W4.",
    ),
    Survey(
        "unwise", ra="ra", dec="dec", id="unwise_objid", error=("axes", "dx_1", "dy_1", 2750.0),
        epoch=2014.0, epoch_span=3.5, floor_mas=100.0, search_radius_mas=4000.0,
        shift_annulus_arcsec=(60.0, 120.0), fwhm_arcsec=6.1,
        keep=("flux_1", "dflux_1", "flux_2", "dflux_2", "fracflux_1", "fracflux_2", "qf_1", "qf_2",
              "flags_unwise_1", "flags_info_1"),
        tap=(IRSA_TAP, "unwise_2019"),
        notes="Crowded-field forced photometry, 2010-2017 coadds; fracflux is unWISE's own blending measure. "
              "Positional error is pixel dx/dy (2.75 arcsec/pix), band 1.",
    ),
    Survey(
        "neowise", ra="RA", dec="DEC", id="SOURCE_ID", error=("radec", "SIGRA", "SIGDEC", "SIGRADEC"),
        epoch=2019.0, epoch_col="MJD", epoch_unit="mjd", epoch_span=5.5, parallax_mode="exact",
        floor_mas=100.0, search_radius_mas=4000.0, shift_annulus_arcsec=(60.0, 120.0), fwhm_arcsec=6.1,
        group_by=("FRAME_NUM", "SCAN_ID"),
        keep=("MJD", "W1MPRO", "W1SIGMPRO", "W2MPRO", "W2SIGMPRO", "QUAL_FRAME", "CC_FLAGS", "PH_QUAL",
              "SAA_SEP", "MOON_MASKED", "QI_FACT", "NB"),
        hats=f"{HATS}/wise/neowise",
        notes="Single exposures 2013-2024; each frame holds at most one counterpart, so frames are matched "
              "independently and the result is a light curve.",
    ),
    Survey(
        "tic", ra="ra", dec="dec", id="ID", error=("axes", "e_RA", "e_Dec", 1.0),
        epoch=2000.0, parallax_mode="none", floor_mas=20.0, search_radius_mas=3000.0,
        shift_annulus_arcsec=(45.0, 90.0), fwhm_arcsec=21.0,
        keep=("GAIA", "Tmag", "objType", "disposition", "duplicate_id", "contratio", "numcont", "POSflag",
              "TWOMASS", "HIP", "TYC"),
        hats=f"{HATS}/tess/tic",
        notes="TIC v8.2 positions are at epoch J2000 (Gaia DR2 propagated). Most rows carry their Gaia DR2 "
              "id, so ids.py checks the positional match against dr2_neighbourhood. fwhm is TESS's pixel "
              "(21 arcsec) for blending.",
    ),
    Survey(
        "erosita_dr2", ra="RA", dec="DEC", id="IAUNAME", error=("circular", "POS_ERR", 1000.0),
        epoch=2020.5, epoch_span=0.75, floor_mas=1000.0, search_radius_mas=30000.0,
        shift_annulus_arcsec=(300.0, 900.0), fwhm_arcsec=26.0, prior=0.01,
        keep=("ML_FLUX_1", "ML_FLUX_ERR_1", "DET_LIKE_0", "EXT", "EXT_LIKE", "RADEC_ERR", "FLAG_SP_SNR",
              "FLAG_SP_BPS", "FLAG_SP_SCL", "FLAG_SP_LGA", "FLAG_OPT"),
        hats=f"{HATS}/erosita_dr2_erass3/erass3_main",
        notes="eRASS:3 cumulative (Dec 2019 - mid 2021), western Galactic hemisphere only. Positional errors "
              "of seconds of arc, so the density term dominates: most Gaia stars have no X-ray counterpart "
              "and the prior is low.",
    ),
    Survey(
        "ztf_dr22", ra="objra", dec="objdec", id="objectid", error=("none",),
        epoch=2020.5, epoch_span=2.5, parallax_mode="unknown", floor_mas=100.0, search_radius_mas=3000.0,
        shift_annulus_arcsec=(45.0, 90.0), fwhm_arcsec=2.0, group_fn=lambda rows: _ztf_quadrant(rows["objectid"]),
        keep=("filterid", "nepochs"),
        hats=f"{HATS}/ztf_dr22/ztf_lc",
        notes="One object per field, CCD, quadrant and filter (the leading digits of objectid), so each of "
              "those is matched independently: a star in two overlapping fields has two objects per filter.",
    ),
    Survey(
        "vsx", ra="RAdeg", dec="DEdeg", id="OID", error=("none",),
        epoch=2000.0, parallax_mode="none", floor_mas=1000.0, search_radius_mas=10000.0,
        shift_annulus_arcsec=(60.0, 180.0), fwhm_arcsec=2.0, prior=0.05,
        keep=("Name", "Type", "Period", "MagMax", "MagMin", "MaxPassband", "VarFlag"),
        hats=f"{HATS}/vsx/2026-06",
        notes="Heterogeneous positions from many sources; no errors given, so the fitted floor sets them.",
    ),
    Survey(
        "asassn_vars", ra="RAJ2000", dec="DEJ2000", id="ID", error=("none",),
        epoch=2001.5, epoch_span=4.0, parallax_mode="none", floor_mas=500.0, search_radius_mas=10000.0,
        shift_annulus_arcsec=(60.0, 180.0), fwhm_arcsec=16.0, prior=0.02,
        keep=("ASASSN-V", "Type", "Per", "Amp", "Vmag", "Prob", "GaiaDR3", "TIC"),
        tap=(VIZIER_TAP, '"II/366/catv2021"'), tap_pos=("RAJ2000", "DEJ2000"),
        notes="ASAS-SN variable stars (2021 version, 687k). Positions have mixed epochs (1998-2005, measured "
              "from high-proper-motion stars against Gaia DR3), so the span is wide. Rows carry GaiaDR3 and "
              "TIC ids: ids.check_ids compares them with the positional match. ASAS-SN light curves for any "
              "star come from Sky Patrol v2 by coordinates or by Gaia DR2/TIC id (ids.previous_release_ids).",
    ),
    Survey(
        "simbad", ra="ra", dec="dec", id="oid", error=("ellipse", "coo_err_maj", "coo_err_min", "coo_err_angle", 1.0),
        epoch=2000.0, parallax_mode="none", floor_mas=100.0, search_radius_mas=10000.0,
        shift_annulus_arcsec=(60.0, 180.0), fwhm_arcsec=2.0, prior=0.1,
        keep=("main_id", "otype", "sp_type", "coo_bibcode", "nbref"),
        tap=(SIMBAD_TAP, "basic"),
        notes="Identifiers, object types and bibliography counts. CDS blacklists > 6 queries/s per IP; the "
              "shared client keeps every CDS host at 2 concurrent and 0.5 s apart.",
    ),
]}
