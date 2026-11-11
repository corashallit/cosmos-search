"""Photometry of one star from surveys other than Gaia, as one long table on Gaia's clock.

    from cosmos_search import surveys
    lc = surveys.fetch("skymapper", source_id, ra, dec)       # ra, dec at the survey's own epoch (`at_year`)
    lc = surveys.fetch_all(star, ("ztf", "skymapper", "noirlab"))

A Gaia fade can be an instrument or calibration effect, and Gaia DR4 stops on 2020-01-20. Another telescope that
watched the star over the same years, or since, answers both: did it see the same change on the same dates, and
did the change go on. `slow.independent` makes that comparison; this package only reads.

Every reader returns COLUMNS:

- `survey`, `band`: the survey and its passband, named as in `inject.lightcurve.BAND_WAVELENGTH_UM` so that a sealed
  signal can be injected into these epochs like any others (`dayone.external.apply`).
- `group`: the unit that has its own zero point (an ASAS-SN camera, a ZTF field and CCD quadrant, a SkyMapper or
  DECam filter, an ATLAS site and filter). Levels are only comparable inside a group, so `flux` is relative to the
  group's own median and a fit gives every group its own level.
- `t`: BTJD (BJD_TDB - 2457000), as Gaia's epochs. The surveys give MJD or HJD at the observatory, which differs
  from BJD_TDB by at most ~8.3 min: far below what a change lasting weeks or longer needs.
- `flux`, `err`: flux over the group's median, and its error from the survey's own error column. Those errors
  leave out each survey's calibration drift from night to night and year to year; `slow.independent` adds floors
  measured on constant stars (`scripts/survey_calibrate.py`).
- `mag`: the survey's calibrated magnitude where it gives one, for a reader who wants the level itself.

| Survey | Years | Sky | Useful G | Where it answers from |
|---|---|---|---|---|
| ztf (IRSA) | 2018-03 on | dec > -30 | 13-20.5 | anywhere |
| asassn (Sky Patrol v2) | 2012 on (V to 2018, g from 2017) | all | 10.5-16.5 | only where port 9006 is open: not cloud containers |
| atlas (forced photometry) | 2018-10 on (read from then); south of -50 from 2022 | all | 12.5-19 | anywhere, with ATLAS_TOKEN |
| skymapper (DR4) | 2014-03 to 2021-09 | dec < +16 | 9-18.5 | anywhere |
| noirlab (NSC DR2) | 2012-09 to 2019-11 | mostly dec < +30 | 14.5-21 | anywhere |

TESS is read separately (`surveys.tess_activity`): its light curves are normalised sector by sector, so they say
how variable a star is within a month, not whether it faded over years.
"""

import numpy as np
import pandas as pd

COLUMNS = ["source_id", "survey", "band", "group", "t", "flux", "err", "mag"]
_DTYPES = ("int64", "object", "object", "object", "float64", "float64", "float64", "float64")
MJD_TO_BTJD = 2400000.5 - 2457000.0   # BTJD = MJD + MJD_TO_BTJD

# Mean observing year of each survey, for propagating a Gaia position to it; point-spread FWHM in arcsec, for the
# share of each measurement that comes from Gaia neighbours (`slow.independent.dilution`).
MEAN_YEAR = {"ztf": 2021.5, "asassn": 2019.0, "atlas": 2020.5, "skymapper": 2017.5, "noirlab": 2016.5, "tess": 2021.5}
FWHM_ARCSEC = {"ztf": 2.0, "asassn": 16.0, "atlas": 5.0, "skymapper": 2.9, "noirlab": 1.2, "tess": 42.0}
# Declination range (deg) each survey observes; a star outside it is not asked for.
DEC_RANGE = {"ztf": (-30.0, 90.0), "asassn": (-90.0, 90.0), "atlas": (-90.0, 90.0), "skymapper": (-90.0, 16.0),
             "noirlab": (-90.0, 90.0)}
# G range in which each survey's photometry of a star is usable: brighter stars saturate, fainter ones are below
# its limit. ZTF's bright end is measured: of 56 constant stars of G 11-13, 4 read as changing against a made-up
# trend where at most 1 in 100 should (2026-11-10). The others are the surveys' own stated limits.
G_RANGE = {"ztf": (13.0, 20.5), "asassn": (10.5, 16.5), "atlas": (12.5, 19.0), "skymapper": (9.0, 18.5),
           "noirlab": (14.5, 21.0)}
ALL = ("ztf", "asassn", "atlas", "skymapper", "noirlab")
# Read unless asked otherwise. ASAS-SN is left out until its reader has run against the live service (asassn.py).
DEFAULT = ("ztf", "atlas", "skymapper", "noirlab")


class MissingCredential(RuntimeError):
    """A survey needs an account or token that this session does not have."""


class Unreachable(RuntimeError):
    """A survey's service cannot be reached from this machine (blocked port or host), as opposed to being down."""


def empty():
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(COLUMNS, _DTYPES)})


def table(source_id, survey, band, group, t, flux, err, mag=None):
    """Rows for one star in COLUMNS, with each group's flux and error divided by the group's median flux."""
    n = len(t)
    df = pd.DataFrame({"source_id": np.full(n, int(source_id), np.int64), "survey": survey,
                       "band": np.asarray(band, object), "group": np.asarray(group, object),
                       "t": np.asarray(t, float), "flux": np.asarray(flux, float), "err": np.asarray(err, float),
                       "mag": np.full(n, np.nan) if mag is None else np.asarray(mag, float)})
    df = df[np.isfinite(df["t"]) & np.isfinite(df["flux"]) & np.isfinite(df["err"]) & (df["err"] > 0)]
    if not len(df):
        return empty()
    med = df.groupby(["band", "group"])["flux"].transform("median")
    df = df[med > 0].copy()
    med = med[med > 0]
    df["flux"], df["err"] = df["flux"] / med, df["err"] / med
    return df.sort_values(["band", "group", "t"], kind="stable").reset_index(drop=True)[COLUMNS]


def mag_to_flux(mag, magerr):
    """Flux (arbitrary zero point) and its error from a magnitude and its error."""
    mag, magerr = np.asarray(mag, float), np.asarray(magerr, float)
    flux = 10 ** (-0.4 * (mag - np.nanmedian(mag))) if len(mag) else mag
    return flux, flux * magerr * np.log(10) / 2.5


def at_year(ra, dec, pmra, pmdec, ref_epoch, year):
    """(ra, dec) in degrees propagated from ref_epoch to year by the proper motion (mas/yr; missing counts as 0)."""
    dt = float(year) - float(ref_epoch)
    pmra = 0.0 if pmra is None or not np.isfinite(pmra) else float(pmra)
    pmdec = 0.0 if pmdec is None or not np.isfinite(pmdec) else float(pmdec)
    return (float(ra) + pmra * dt / 3.6e6 / np.cos(np.radians(float(dec)))) % 360.0, float(dec) + pmdec * dt / 3.6e6


# Readers that move the position to each image's date themselves (from the position at MEAN_YEAR and the proper
# motion); the others get one position, at MEAN_YEAR.
PROPER_MOTION = ("atlas",)


def placed(survey, ra, dec, pmra, pmdec, ref_epoch):
    """(ra, dec, options) for one survey's reader: the position moved from ref_epoch to the survey's MEAN_YEAR, and
    for PROPER_MOTION readers also the proper motion and that year."""
    ref_epoch = 2016.0 if ref_epoch is None or not np.isfinite(ref_epoch) else ref_epoch
    at = at_year(ra, dec, pmra, pmdec, ref_epoch, MEAN_YEAR[survey])
    return (*at, {"pm": (pmra, pmdec), "epoch": MEAN_YEAR[survey]} if survey in PROPER_MOTION else {})


def covers(survey, dec, g_mag=None):
    """Whether the survey observes this declination and, when g_mag is given and finite, a star this bright."""
    lo, hi = DEC_RANGE[survey]
    if not lo <= float(dec) <= hi:
        return False
    if g_mag is None or not np.isfinite(g_mag):
        return True
    bright, faint = G_RANGE[survey]
    return bright <= float(g_mag) <= faint


def reader(survey):
    """The fetch function of one survey: f(source_id, ra, dec, **options) -> COLUMNS table."""
    if survey == "ztf":
        return _ztf
    if survey == "asassn":
        from . import asassn
        return asassn.fetch
    if survey == "atlas":
        from . import atlas
        return atlas.fetch
    if survey == "skymapper":
        from . import skymapper
        return skymapper.fetch
    if survey == "noirlab":
        from . import noirlab
        return noirlab.fetch
    raise ValueError(f"unknown survey {survey!r}; one of {ALL}")


def usable(names=DEFAULT, **options):
    """(the surveys of `names` that this machine can read now, {survey: why not}). ASAS-SN needs its port and
    ATLAS a token; each is tried once here so that a run reports a missing survey once and not once per star.
    options: {survey: {keyword: value}} for that survey's check (a test server's address)."""
    ok, why = [], {}
    for s in names:
        try:
            if s == "asassn":
                from . import asassn
                asassn.check_reachable(**options.get(s, {}))
            elif s == "atlas":
                from . import atlas
                atlas.check_token(**options.get(s, {}))
            ok.append(s)
        except (MissingCredential, Unreachable) as e:
            why[s] = str(e)
    return ok, why


def fetch(survey, source_id, ra, dec, **options):
    return reader(survey)(source_id, ra, dec, **options)


def fetch_all(star, surveys=DEFAULT, log=None, **options):
    """Every listed survey for one star. star: a mapping or row with source_id, ra, dec and, where known, pmra,
    pmdec, ref_epoch and phot_g_mean_mag. A survey that does not cover the star's declination or brightness is
    skipped; one that fails is reported in the returned errors and the rest go on. Returns (table, {survey:
    error text}). options: {survey: {keyword: value}} passed to that survey's reader."""
    get = (lambda k, d=None: star.get(k, d)) if hasattr(star, "get") else (lambda k, d=None: getattr(star, k, d))
    parts, errors = [], {}
    for s in surveys:
        if not covers(s, get("dec"), get("phot_g_mean_mag")):
            continue
        ra, dec, moves = placed(s, get("ra"), get("dec"), get("pmra"), get("pmdec"), get("ref_epoch", 2016.0))
        try:
            parts.append(fetch(s, int(get("source_id")), ra, dec, **{**moves, **options.get(s, {})}))
        except Exception as e:  # noqa: BLE001 (one survey failing for one star is missing data, not a stop)
            errors[s] = f"{type(e).__name__}: {e}"[:200]
            if log:
                log(f"{get('source_id')}: {s} failed ({errors[s]})")
    parts = [p for p in parts if len(p)]
    return (pd.concat(parts, ignore_index=True) if parts else empty()), errors


def _ztf(source_id, ra, dec):
    from cosmos_search.dayone import external

    z = external.ztf(source_id, ra, dec, groups=True)
    if not len(z):
        return empty()
    return z.assign(survey="ztf", mag=np.nan)[COLUMNS]
