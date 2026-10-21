"""Light curves from other surveys for the stars in a vetting queue: ZTF DR22 and TESS (TGLC), on the Gaia clock.

    python -m cosmos_search.dayone.external fetch --queue rank/queue3.parquet --out ext/external.parquet

Gaia's ~35-70 epochs show most dip trains as one or two dips, which an eclipsing binary explains as well, so a
blind vetter rightly calls them EBs (docs/dayone.md, batch 2). Denser photometry from other years makes that
testable: an EB repeats at a fixed period with a fixed depth, other occulters need not. Times are BTJD
(BJD_TDB - 2457000), as in the Gaia stream. ZTF objects are normalised to their own median per field, CCD and
filter; TGLC per sector. ZTF keeps catflags == 0 epochs; TGLC keeps unflagged cadences (`tess.tglc.clean`).

The output is a long table (source_id, band, t, flux, err), with bands named as in
`inject.lightcurve.BAND_WAVELENGTH_UM` (ztf_g, ztf_r, ztf_i, TESS), so `apply` can inject the same sealed signal
into them that the Gaia epochs received.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from cosmos_search.xmatch.surveys import HATS

ZTF_HATS = f"{HATS}/ztf_dr22/ztf_lc"
ZTF_BANDS = {1: "ztf_g", 2: "ztf_r", 3: "ztf_i"}
ZTF_RADIUS_ARCSEC = 1.5     # ZTF's PSF is ~2"; Gaia positions are good to mas, so anything further is a neighbour
ZTF_MIN_EPOCHS = 20
MJD_TO_BTJD = 2400000.5 - 2457000.0
TESSCUT_SECTORS = "https://mast.stsci.edu/tesscut/api/v0.1/sector?ra={ra:.6f}&dec={dec:.6f}"
TESS_BIN_DAYS = 1 / 48       # 30 min: TGLC's slowest FFI cadence, so every sector reads the same
COLUMNS = ["source_id", "band", "t", "flux", "err"]


def _empty():
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                         zip(COLUMNS, ("int64", "object", "float64", "float64", "float64"))})


def ztf(source_id, ra, dec, catalog=None):
    """ZTF DR22 epochs within ZTF_RADIUS_ARCSEC of (ra, dec), one block per ZTF object (field, CCD, filter)."""
    import lsdb

    cat = catalog if catalog is not None else lsdb.open_catalog(ZTF_HATS)
    rows = cat.cone_search(ra, dec, ZTF_RADIUS_ARCSEC).compute()
    out = []
    for _, r in rows.iterrows():
        mag, magerr = np.asarray(r["mag"], float), np.asarray(r["magerr"], float)
        hmjd, flags = np.asarray(r["hmjd"], float), np.asarray(r["catflags"], np.int64)
        ok = (flags == 0) & np.isfinite(mag) & np.isfinite(magerr)
        if ok.sum() < ZTF_MIN_EPOCHS:
            continue
        band = ZTF_BANDS.get(int(r["filterid"]))
        if band is None:
            continue
        flux = 10 ** (-0.4 * (mag[ok] - np.median(mag[ok])))
        out.append(pd.DataFrame({"source_id": int(source_id), "band": band, "t": hmjd[ok] - MJD_TO_BTJD,
                                 "flux": flux, "err": flux * magerr[ok] * np.log(10) / 2.5}))
    return pd.concat(out, ignore_index=True) if out else _empty()


def tess_sectors(ra, dec):
    """(sector, camera, ccd) that observed (ra, dec), from TESScut."""
    from cosmos_search.tess.http import get_bytes

    body = get_bytes(TESSCUT_SECTORS.format(ra=ra, dec=dec), missing_ok=True)
    if not body:
        return []
    return [(int(r["sector"]), int(r["camera"]), int(r["ccd"])) for r in json.loads(body)["results"]]


def _bin(t, f, width):
    """Mean flux in fixed-width bins with the scatter of the mean as the error (TGLC gives no per-point errors)."""
    key = np.floor((t - t.min()) / width).astype(np.int64)
    df = pd.DataFrame({"k": key, "t": t, "f": f}).groupby("k")
    agg = df.agg(t=("t", "mean"), f=("f", "mean"), s=("f", "std"), n=("f", "size"))
    agg = agg[agg.n >= 2]
    return agg.t.to_numpy(), agg.f.to_numpy(), (agg.s / np.sqrt(agg.n)).to_numpy()


def tess(source_id, ra, dec, max_sectors=None):
    """TGLC aperture photometry for every sector that has it, binned to TESS_BIN_DAYS and normalised per sector."""
    from cosmos_search.tess import tglc

    out = []
    for sector, cam, ccd in tess_sectors(ra, dec)[:max_sectors]:
        lc = tglc.fetch(sector, cam, ccd, int(source_id))
        if lc is None:
            continue
        lc = tglc.clean(lc)
        ok = np.isfinite(lc.flux) & np.isfinite(lc.t)
        if ok.sum() < 100:
            continue
        t, f, e = _bin(lc.t[ok], lc.flux[ok] / np.median(lc.flux[ok]), TESS_BIN_DAYS)
        # Flat blocks (one value repeated) come from failed fits; the robust scatter of a real star is never 0.
        if not np.isfinite(e).any() or np.nanmedian(e) == 0:
            continue
        out.append(pd.DataFrame({"source_id": int(source_id), "band": "TESS", "t": t, "flux": f, "err": e}))
    return pd.concat(out, ignore_index=True) if out else _empty()


def fetch(queue, max_sectors=None, log=print):
    """External light curves for every star in the queue (source_id, ra, dec), as one long table."""
    import lsdb

    cat = lsdb.open_catalog(ZTF_HATS, columns=["filterid", "objra", "objdec", "hmjd", "mag", "magerr", "catflags"])
    parts = []
    # Not iterrows: on an all-numeric frame it casts source_id to float64 and corrupts it.
    for sid, ra, dec in zip(queue["source_id"].astype("int64"), queue["ra"].astype(float), queue["dec"].astype(float)):
        sid, ra, dec = int(sid), float(ra), float(dec)
        z, s = _empty(), _empty()
        try:
            z = ztf(sid, ra, dec, cat)
        except Exception as exc:  # a survey that fails for one star is missing data, not a reason to stop
            log(f"{sid}: ZTF failed ({type(exc).__name__})")
        try:
            s = tess(sid, ra, dec, max_sectors)
        except Exception as exc:
            log(f"{sid}: TESS failed ({type(exc).__name__})")
        parts += [z, s]
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
    q = pd.read_parquet(a.queue, columns=["source_id", "ra", "dec"]).drop_duplicates("source_id")
    ext = fetch(q, a.max_sectors)
    ext.to_parquet(a.out)
    cov = coverage(ext) if len(ext) else pd.DataFrame()
    print(f"{len(ext)} external epochs for {ext.source_id.nunique()} of {len(q)} stars -> {a.out}")
    if len(cov):
        print((cov > 0).sum().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
