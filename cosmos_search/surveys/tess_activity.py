"""How variable a star is within a month, from TESS full-frame images: a measure of activity.

Why it bears on a fade. On a spotted star the brightness changes as spots rotate into view (days to weeks) and
as the spot coverage itself grows and shrinks (months to years). Both come from the same spots, so the size of the
rotational modulation says how large a slow change activity alone can make. `slow.independent.activity` reports
Gaia's change as a multiple of the amplitude measured here.

Source: TGLC light curves on MAST's S3 bucket (`tess.tglc`), one file per star and sector, keyed by Gaia DR3 id.
TGLC fits the field with Gaia positions and magnitudes, so its fluxes are the target's own, not the 21-arcsec
pixel's. The file's `cal_*` columns are detrended with a 1-day biweight filter (header WOTAN_WL = 1), which
removes rotation slower than a day, so this module reads the undetrended `aperture_flux` and `psf_flux` with
the background already subtracted. TGLC covers sectors 1 to 55 (2018-07 to 2022-09); it has no file for stars fainter
than its limit (Tmag ~16) or too near a CCD edge.

Per sector (`variability`): cadences with clean TESS and TGLC flags, divided by the median, averaged in
BIN_DAYS bins; `range` is the 5th to 95th percentile spread of the bins; `noise` is what that spread would be
from the bins' point-to-point scatter alone (3.29 sigma for a Gaussian); `amp` is the range with the noise removed
in quadrature (0 when the range is not above it). `period` is the highest Lomb-Scargle peak between MIN_PERIOD
and half the sector and `power` its normalised height. A sector is at most 27 days, so rotation slower than
about 13 days shows as a drift, counted in the range but without a period.

Two fluxes, the smaller amplitude. Scattered light and pointing changes leave trends in the undetrended fluxes
that differ between the aperture and the PSF fit, while a real change in the star is in both. Each sector's
`amp` is the smaller of the two. What a constant star still shows is in docs/arms/slow-changes.md ("Other
surveys"): amplitudes below that floor mean "quiet", not a measured rotation.

`activity` picks up to max_sectors sectors spread over the available ones and returns one summary row: the
median amplitude, its smallest and largest sector values, the best period, and the amplitude's change from the
first to the last sector read (a spot cycle changes it over years).
"""

import io

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.timeseries import LombScargle

BIN_DAYS = 0.125
MIN_PERIOD = 0.3
MIN_BINS = 60
GAUSS_RANGE = 3.29          # 5th to 95th percentile of a Gaussian, in sigma
MAX_SECTORS = 6
TGLC_LAST_SECTOR = 55     # the last sector under mast/hlsp/tglc/ on 2026-11-10 (2022-09)
SECTOR_COLUMNS = ["source_id", "sector", "t", "n_bins", "range", "noise", "amp", "amp_aper", "amp_psf", "period",
                  "power", "tmag"]


def read_raw(data):
    """(t, aperture flux, PSF flux, meta) of the unflagged cadences of one TGLC file, undetrended."""
    with fits.open(io.BytesIO(data), memmap=False) as h:
        hdr, d = h[0].header, h[1].data
        ok = (np.asarray(d["TESS_flags"]) == 0) & (np.asarray(d["TGLC_flags"]) == 0)
        t = np.asarray(d["time"], float)[ok]
        aper = np.asarray(d["aperture_flux"], float)[ok]
        psf = np.asarray(d["psf_flux"], float)[ok]
        meta = {"sector": int(hdr["SECTOR"]), "tmag": float(hdr.get("TESSMAG", np.nan)),
                "tic": hdr.get("TICID")}
    return t, aper, psf, meta


def variability(t, flux, bin_days=BIN_DAYS):
    """dict(n_bins, range, noise, amp, period, power) of one sector's flux (module notes); None when fewer than
    MIN_BINS bins have data or the median flux is not positive."""
    ok = np.isfinite(t) & np.isfinite(flux)
    t, flux = t[ok], flux[ok]
    if len(t) < MIN_BINS or not np.median(flux) > 0:
        return None
    f = flux / np.median(flux)
    k = np.floor((t - t.min()) / bin_days).astype(np.int64)
    n = np.bincount(k)
    filled = n >= max(1, int(0.5 * np.median(n[n > 0])))
    tb = (np.bincount(k, t)[filled] / n[filled])
    fb = (np.bincount(k, f)[filled] / n[filled])
    if len(fb) < MIN_BINS:
        return None
    lo, hi = np.percentile(fb, [5, 95])
    d = np.diff(fb)
    sigma = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    rng, noise = float(hi - lo), float(GAUSS_RANGE * sigma)
    span = tb.max() - tb.min()
    period, power = np.nan, np.nan
    if span > 4 * MIN_PERIOD:
        freq = np.linspace(2 / span, 1 / MIN_PERIOD, 4000)
        p = LombScargle(tb, fb - fb.mean()).power(freq)
        i = int(np.argmax(p))
        period, power = float(1 / freq[i]), float(p[i])
    return {"n_bins": len(fb), "range": rng, "noise": noise, "amp": float(np.sqrt(max(rng ** 2 - noise ** 2, 0))),
            "period": period, "power": power}


def sector_row(source_id, data):
    """One row of SECTOR_COLUMNS from a TGLC file's bytes, or None when the sector has too little clean data."""
    t, aper, psf, meta = read_raw(data)
    va, vp = variability(t, aper), variability(t, psf)
    if va is None and vp is None:
        return None
    best = min((v for v in (va, vp) if v is not None), key=lambda v: v["amp"])
    return {"source_id": int(source_id), "sector": meta["sector"], "t": float(np.median(t)),
            "n_bins": best["n_bins"], "range": best["range"], "noise": best["noise"], "amp": best["amp"],
            "amp_aper": va["amp"] if va else np.nan, "amp_psf": vp["amp"] if vp else np.nan,
            "period": best["period"], "power": best["power"], "tmag": meta["tmag"]}


def spread(sectors, n=MAX_SECTORS):
    """Up to n of the sectors, evenly spaced through the list (first and last always kept)."""
    sectors = sorted(sectors)
    if len(sectors) <= n:
        return sectors
    idx = np.unique(np.round(np.linspace(0, len(sectors) - 1, n)).astype(int))
    return [sectors[i] for i in idx]


def summarise(rows):
    """One summary dict from the per-sector rows: tess_sectors, tess_amp (median), tess_amp_min, tess_amp_max,
    tess_noise (median), tess_period and tess_power (of the sector with the highest peak), tess_amp_change (last
    sector's amplitude minus the first's), tess_first and tess_last (BTJD), tess_tmag."""
    if not len(rows):
        return {"tess_sectors": 0}
    r = rows.sort_values("t")
    best = r.loc[r["power"].fillna(-1).idxmax()]
    return {"tess_sectors": len(r), "tess_amp": float(r["amp"].median()), "tess_amp_min": float(r["amp"].min()),
            "tess_amp_max": float(r["amp"].max()), "tess_noise": float(r["noise"].median()),
            "tess_period": float(best["period"]), "tess_power": float(best["power"]),
            "tess_amp_change": float(r["amp"].iloc[-1] - r["amp"].iloc[0]) if len(r) > 1 else np.nan,
            "tess_first": float(r["t"].iloc[0]), "tess_last": float(r["t"].iloc[-1]),
            "tess_tmag": float(r["tmag"].median())}


def activity(source_id, ra, dec, max_sectors=MAX_SECTORS):
    """(summary dict, per-sector DataFrame) for one star: TGLC sectors found through TESScut's sector list at
    (ra, dec), up to max_sectors of them spread over time."""
    from cosmos_search.dayone.external import tess_sectors
    from cosmos_search.tess import tglc
    from cosmos_search.tess.http import get_bytes

    rows, missing = [], 0
    listed = sorted(tess_sectors(ra, dec))
    todo = [s for s in listed if s[0] <= TGLC_LAST_SECTOR]
    for sector, cam, ccd in spread(todo, max_sectors):
        data = get_bytes(tglc.url(sector, cam, ccd, int(source_id)), missing_ok=True)
        if data is None:
            missing += 1
            continue
        row = sector_row(source_id, data)
        if row is not None:
            rows.append(row)
    table = pd.DataFrame(rows, columns=SECTOR_COLUMNS)
    out = summarise(table)
    out["tess_listed"], out["tess_missing"] = len(todo), missing
    return out, table
