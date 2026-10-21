"""Light curves from other surveys for the stars in a vetting queue: ZTF and TESS (TGLC), on the Gaia clock.

    python -m cosmos_search.dayone.external fetch --queue rank/queue3.parquet --out ext/external.parquet

Gaia's ~35-70 epochs show most dip trains as one or two dips, which an eclipsing binary explains as well, so a
blind vetter rightly calls them EBs (docs/dayone.md, batch 2). Denser photometry from other years makes that
testable: an EB repeats at a fixed period with a fixed depth, other occulters need not. Times are BTJD
(BJD_TDB - 2457000), as in the Gaia stream; ZTF's HJD differs from BJD by under a minute. ZTF (IRSA's
light-curve service, latest data release) objects are normalised to their own median per field, CCD quadrant
and filter; TGLC per sector. ZTF keeps catflags == 0 epochs; TGLC keeps unflagged cadences (`tess.tglc.clean`).

Which TESS sectors: a dip Gaia caught on one visit can only be argued with photometry taken during or near it, so
sectors are ranked by how close they come to the dip (`t_deepest` in the queue; the whole Gaia span when the
queue has no dip time) before `max_sectors` cuts the list. DR4 ends 2020-01-20 (BTJD 1868.5), so its concurrent
sectors are 1-20; TESScut lists sectors in numerical order, which put them first only by luck. Sector times are
in `tess/sectors.csv`, measured from one light curve per sector.

The output is a long table (source_id, band, t, flux, err), with bands named as in
`inject.lightcurve.BAND_WAVELENGTH_UM` (ztf_g, ztf_r, ztf_i, TESS), so `apply` can inject the same sealed signal
into them that the Gaia epochs received.
"""

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ZTF_LC = ("https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE%20{ra:.6f}%20{dec:.6f}%20{r:.6f}"
          "&FORMAT=csv")
ZTF_BANDS = {"zg": "ztf_g", "zr": "ztf_r", "zi": "ztf_i"}
ZTF_RADIUS_ARCSEC = 1.5     # ZTF's PSF is ~2"; Gaia positions are good to mas, so anything further is a neighbour
ZTF_MIN_EPOCHS = 20
TESSCUT_SECTORS = "https://mast.stsci.edu/tesscut/api/v0.1/sector?ra={ra:.6f}&dec={dec:.6f}"
TESS_BIN_DAYS = 1 / 48       # 30 min: TGLC's slowest FFI cadence, so every sector reads the same
COLUMNS = ["source_id", "band", "t", "flux", "err"]
GAIA_DR4_SPAN = (-136.06, 1868.5)   # BTJD: 2014-07-25 10:31 to 2020-01-20 (DR4's epoch-photometry span)
SECTORS_CSV = Path(__file__).resolve().parent.parent / "tess" / "sectors.csv"


def _empty():
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                         zip(COLUMNS, ("int64", "object", "float64", "float64", "float64"))})


def ztf(source_id, ra, dec):
    """ZTF epochs within ZTF_RADIUS_ARCSEC of (ra, dec) from IRSA's light-curve service, one block per ZTF object
    (field, CCD quadrant, filter), each normalised to its own median.

    Not the HATS copy: its partitions are single row groups of whole light curves, so a cone search reads ~10 GB."""
    import io

    from cosmos_search.net import get_client

    body = get_client().get_bytes(ZTF_LC.format(ra=ra, dec=dec, r=ZTF_RADIUS_ARCSEC / 3600), cache=False)
    rows = pd.read_csv(io.BytesIO(body)) if body else pd.DataFrame()
    out = []
    for _, r in rows.groupby("oid") if len(rows) else ():
        ok = (r["catflags"] == 0) & np.isfinite(r["mag"]) & np.isfinite(r["magerr"])
        band = ZTF_BANDS.get(str(r["filtercode"].iloc[0]))
        if ok.sum() < ZTF_MIN_EPOCHS or band is None:
            continue
        mag, magerr = r.loc[ok, "mag"].to_numpy(float), r.loc[ok, "magerr"].to_numpy(float)
        flux = 10 ** (-0.4 * (mag - np.median(mag)))
        out.append(pd.DataFrame({"source_id": int(source_id), "band": band,
                                 "t": r.loc[ok, "hjd"].to_numpy(float) - 2457000.0,
                                 "flux": flux, "err": flux * magerr * np.log(10) / 2.5}))
    return pd.concat(out, ignore_index=True) if out else _empty()


def tess_sectors(ra, dec):
    """(sector, camera, ccd) that observed (ra, dec), from TESScut."""
    from cosmos_search.tess.http import get_bytes

    body = get_bytes(TESSCUT_SECTORS.format(ra=ra, dec=dec), missing_ok=True)
    if not body:
        return []
    return [(int(r["sector"]), int(r["camera"]), int(r["ccd"])) for r in json.loads(body)["results"]]


@lru_cache(maxsize=1)
def sector_times():
    """{sector: (btjd_start, btjd_end)} for every sector in tess/sectors.csv."""
    df = pd.read_csv(SECTORS_CSV)
    return {int(r.sector): (float(r.btjd_start), float(r.btjd_end)) for r in df.itertuples()}


def sector_gap(sector, times):
    """Days between a sector's observing interval and the nearest of times (0 if one falls inside); inf for a
    sector not in the table (newer than it)."""
    span = sector_times().get(int(sector))
    if span is None:
        return np.inf
    t = np.atleast_1d(np.asarray(times, float))
    t = t[np.isfinite(t)]
    if not len(t):
        return np.inf
    return float(np.min(np.clip(span[0] - t, 0, None) + np.clip(t - span[1], 0, None)))


def rank_sectors(sectors, near=None, span=GAIA_DR4_SPAN):
    """(sector, camera, ccd) ordered for testing a Gaia dip: nearest to the dip epochs `near` first; without dip
    times, those overlapping the Gaia span first; ties (and sectors far from both) by sector number."""
    near = np.atleast_1d(np.asarray(near, float)) if near is not None else np.array([])
    near = near[np.isfinite(near)]

    def key(s):
        if len(near):
            return (sector_gap(s[0], near), s[0])
        lo, hi = span
        mid = np.clip(np.mean(sector_times().get(int(s[0]), (np.inf, np.inf))), lo, hi)
        return (sector_gap(s[0], mid), s[0])

    return sorted(sectors, key=key)


def _bin(t, f, width):
    """Mean flux in fixed-width bins, with error = per-point scatter / sqrt(n), the per-point scatter taken from
    point-to-point differences (robust, so dips and trends barely move it). TGLC gives no per-point errors.

    Bins with a single point are kept: sectors 1-26 (all of DR4's concurrent TESS data) have 30-min FFIs, one
    point per 30-min bin, and a rule that needed two points per bin dropped every one of them."""
    key = np.floor((t - t.min()) / width).astype(np.int64)
    df = pd.DataFrame({"k": key, "t": t, "f": f}).groupby("k")
    agg = df.agg(t=("t", "mean"), f=("f", "mean"), n=("f", "size"))
    d = np.diff(f)
    s = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) if len(d) else np.nan
    return agg.t.to_numpy(), agg.f.to_numpy(), (s / np.sqrt(agg.n)).to_numpy()


def tess(source_id, ra, dec, max_sectors=None, near=None):
    """TGLC aperture photometry, binned to TESS_BIN_DAYS and normalised per sector, for up to max_sectors sectors
    ranked by `rank_sectors` (closest to the dip epochs `near` first)."""
    from cosmos_search.tess import tglc

    out = []
    for sector, cam, ccd in rank_sectors(tess_sectors(ra, dec), near)[:max_sectors]:
        lc = tglc.fetch(sector, cam, ccd, int(source_id))
        if lc is None:
            continue
        lc = tglc.clean(lc)
        ok = np.isfinite(lc.flux) & np.isfinite(lc.t)
        if ok.sum() < 100:
            continue
        t, f, e = _bin(lc.t[ok], lc.flux[ok] / np.median(lc.flux[ok]), TESS_BIN_DAYS)
        # Flat blocks (one value repeated) come from failed fits; the robust scatter of a real star is never 0.
        if not np.isfinite(e).any() or np.nanmedian(e) == 0 or len(t) < 100:
            continue
        out.append(pd.DataFrame({"source_id": int(source_id), "band": "TESS", "t": t, "flux": f, "err": e}))
    return pd.concat(out, ignore_index=True) if out else _empty()


def fetch(queue, max_sectors=None, log=print):
    """External light curves for every star in the queue (source_id, ra, dec, optional t_deepest), as one long
    table."""
    parts = []
    near = queue["t_deepest"].astype(float) if "t_deepest" in queue else pd.Series(np.nan, index=queue.index)
    # Not iterrows: on an all-numeric frame it casts source_id to float64 and corrupts it.
    for sid, ra, dec, td in zip(queue["source_id"].astype("int64"), queue["ra"].astype(float),
                                queue["dec"].astype(float), near):
        sid, ra, dec = int(sid), float(ra), float(dec)
        z, s = _empty(), _empty()
        try:
            z = ztf(sid, ra, dec)
        except Exception as exc:  # a survey that fails for one star is missing data, not a reason to stop
            log(f"{sid}: ZTF failed ({type(exc).__name__})")
        try:
            s = tess(sid, ra, dec, max_sectors, near=td)
        except Exception as exc:
            log(f"{sid}: TESS failed ({type(exc).__name__})")
        parts += [z, s]
        log(f"{len(parts) // 2}/{len(queue)} ztf {len(z)} tess {len(s)}")
    df = pd.concat(parts, ignore_index=True) if parts else _empty()
    return df.astype({"source_id": "int64", "band": "object"})


def coverage(ext):
    """Epoch counts per star and band."""
    return ext.groupby(["source_id", "band"]).size().unstack(fill_value=0)


def apply(ext, plan_map, contamination=None):
    """Inject the sealed plan's signals into external epochs, as `gaia_stream.apply` does for Gaia's.

    TESS pixels are 21": a change on the target is diluted by the flux of its neighbours, so the TESS factor is
    1 - (1 - c) (1 - k) with c the target's TESS contamination ratio (xmatch blend context). ZTF is PSF photometry
    at ~2" and is not diluted. A g_only plan (an instrument fault in Gaia's G) leaves other surveys alone.
    Noise is scaled with the factor as for Gaia; the extra photon-noise draw is left out because external errors
    are scatter estimates, not photon counts.
    """
    from cosmos_search.inject.classes.gaia_stream import signal_from_truth

    if not plan_map or ext.empty:
        return ext
    ext = ext.copy()
    contamination = contamination or {}
    for sid, rows in ext.groupby("source_id").groups.items():
        rec = plan_map.get(int(sid))
        if rec is None or rec.get("bands"):
            continue
        sig = signal_from_truth(rec["signal"])
        for band, idx in ext.loc[rows].groupby("band").groups.items():
            k = sig.factor(ext.loc[idx, "t"].to_numpy(), band)
            if band == "TESS":
                c = contamination.get(int(sid), 0.0)
                k = 1 - (1 - (c if np.isfinite(c) else 0.0)) * (1 - k)
            k = np.clip(k, 0, None)
            ext.loc[idx, "flux"] = ext.loc[idx, "flux"].to_numpy() * k
            ext.loc[idx, "err"] = ext.loc[idx, "err"].to_numpy() * np.sqrt(k)
    return ext


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.dayone.external")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--queue", required=True, help="parquet with source_id, ra, dec")
    f.add_argument("--out", required=True)
    f.add_argument("--max-sectors", type=int)
    a = ap.parse_args(argv)
    q = pd.read_parquet(a.queue)
    q = q[[c for c in ("source_id", "ra", "dec", "t_deepest") if c in q]].drop_duplicates("source_id")
    ext = fetch(q, a.max_sectors)
    ext.to_parquet(a.out)
    cov = coverage(ext) if len(ext) else pd.DataFrame()
    print(f"{len(ext)} external epochs for {ext.source_id.nunique()} of {len(q)} stars -> {a.out}")
    if len(cov):
        print((cov > 0).sum().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
