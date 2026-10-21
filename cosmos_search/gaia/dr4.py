"""DR4 adapters (milestone G7), written against the draft data model of 2026-06-26.

DR4 epoch photometry is one row per FoV transit (DR3 packed each source's light curve into arrays), with
DPAC flag words instead of DR3's booleans, and a separate per-CCD G table. These functions turn either
into the shared light-curve schema (epoch.SCHEMA), so features and detectors run unchanged on DR4.

Flag mapping into `flags` (bits 1, 2, 4 keep their DR3 meaning; the rest are DR4-only):

| bit | set when | DR4 source |
|---|---|---|
| 1 | VariPipe did not use the observation | vari_epoch_flags (optional join) |
| 2 | BP/RP transit unrecoverable after truncation or gating | phot_ccd_proc_flags[BP or RP] bit 13 |
| 8 | saturation correction applied on any AF CCD | phot_ccd_proc_flags[AF1–9] bit 0 |
| 16 | observation affected by a satellite event | phot_ccd_proc_flags[any] bit 8 |
| 32 | dead columns or anomalous samples in some windows | transit_proc_flags bits 11–13 |
| 64 | multiple-source image parameters (a neighbour in the window) | transit_proc_flags & 0xF in 1–3 |
| 128 | attitude problem | transit_proc_flags bit 15 |

Bit 4 (DR3's "noisy across CCDs") has no direct DR4 equivalent; the per-CCD table measures it instead
(`ccd_consistency`). Column names may change by release; `COLUMNS` is the one place to fix them.
"""

import numpy as np
import pyarrow as pa

from .epoch import (BANDS, FLAG_PHOT_REJECT, FLAG_VARI_REJECT, SCHEMA, decode_transit_id,
                    gaia_time_to_btjd)

FLAG_SATURATION = 8
FLAG_SATELLITE_EVENT = 16
FLAG_COSMETIC = 32
FLAG_MULTI_SOURCE = 64
FLAG_ATTITUDE = 128

COLUMNS = {
    "G": ("g_obs_time", "g_flux", "g_flux_error"),
    "BP": ("bp_obs_time", "bp_flux", "bp_flux_error"),
    "RP": ("rp_obs_time", "rp_flux", "rp_flux_error"),
}
_PHOT_INDEX = {"SM": 0, "AF": slice(1, 10), "BP": 10, "RP": 11}

CCD_SCHEMA = pa.schema([
    ("source_id", pa.int64()), ("transit_id", pa.int64()), ("ccd", pa.int8()), ("t", pa.float64()),
    ("flux", pa.float64()), ("err", pa.float64()),
])


def _col(table, name, dtype=np.float64):
    col = table[name]
    if hasattr(col, "filled"):  # astropy masked column
        return np.asarray(col.filled(np.nan if np.dtype(dtype).kind == "f" else 0), dtype=dtype)
    return np.asarray(col, dtype=dtype)


def transit_flags(table):
    """Per-transit flag bits (before band-specific ones) from DR4 transit_proc_flags and phot flags."""
    n = len(table)
    flags = np.zeros(n, np.int32)
    if "transit_proc_flags" in table.colnames:
        proc = _col(table, "transit_proc_flags", np.int64) & 0xFFFF
        ipd = proc & 0xF
        flags |= np.where((ipd >= 1) & (ipd <= 3), FLAG_MULTI_SOURCE, 0)
        flags |= np.where(proc & 0x3800, FLAG_COSMETIC, 0)
        flags |= np.where(proc & 0x8000, FLAG_ATTITUDE, 0)
    phot = _phot_flags(table)
    if phot is not None:
        flags |= np.where((phot[:, _PHOT_INDEX["AF"]] & 0x1).any(axis=1), FLAG_SATURATION, 0)
        flags |= np.where((phot & 0x100).any(axis=1), FLAG_SATELLITE_EVENT, 0)
    return flags


def _phot_flags(table):
    if "phot_ccd_proc_flags" not in table.colnames:
        return None
    raw = table["phot_ccd_proc_flags"]
    arr = np.array([np.asarray(v, dtype=np.int64) if v is not None else np.zeros(12, np.int64) for v in raw])
    return arr.reshape(len(table), 12) & 0xFFFF


def epoch_photometry_to_lc(table, vari_rejected=None):
    """DR4 epoch_photometry rows (astropy Table, one per transit) -> light-curve table (epoch.SCHEMA).

    vari_rejected: optional dict {(transit_id, band): True} from vari_epoch_flags.
    """
    sid = _col(table, "source_id", np.int64)
    tid = _col(table, "transit_id", np.int64)
    _, fov, row, _ = decode_transit_id(tid)
    base = transit_flags(table)
    phot = _phot_flags(table)
    n_obs = _col(table, "g_n_obs", np.int64).astype(np.int8) if "g_n_obs" in table.colnames else np.zeros(len(table), np.int8)
    parts = []
    for code, band in enumerate(BANDS):
        tcol, fcol, ecol = COLUMNS[band]
        t, f, e = _col(table, tcol), _col(table, fcol), _col(table, ecol)
        keep = np.isfinite(t) & np.isfinite(f)
        flags = base.copy()
        if band != "G" and phot is not None:
            flags |= np.where(phot[:, _PHOT_INDEX[band]] & 0x2000, FLAG_PHOT_REJECT, 0)
        if vari_rejected:
            flags |= np.array([FLAG_VARI_REJECT if vari_rejected.get((int(x), band)) else 0 for x in tid], np.int32)
        k = int(keep.sum())
        parts.append({
            "source_id": sid[keep], "band": np.full(k, code, np.int8), "t": gaia_time_to_btjd(t[keep]),
            "flux": f[keep], "err": e[keep], "flags": flags[keep].astype(np.int32), "transit_id": tid[keep],
            "fov": fov[keep], "ccd_row": row[keep],
            "n_obs": n_obs[keep] if band == "G" else np.zeros(k, np.int8),
        })
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.lexsort((merged["t"], merged["band"], merged["source_id"]))
    arrays = []
    for field in SCHEMA:
        v = merged[field.name][order]
        if field.name == "band":
            arrays.append(pa.DictionaryArray.from_arrays(pa.array(v, pa.int8()), pa.array(BANDS)))
        else:
            arrays.append(pa.array(v, field.type))
    return pa.Table.from_arrays(arrays, schema=SCHEMA)


def epoch_photometry_ccd_to_long(table):
    """DR4 epoch_photometry_ccd rows (af_* arrays of 9) -> one row per AF CCD observation (CCD_SCHEMA)."""
    n = len(table)
    sid = np.repeat(_col(table, "source_id", np.int64), 9)
    tid = np.repeat(_col(table, "transit_id", np.int64), 9)
    ccd = np.tile(np.arange(1, 10, dtype=np.int8), n)

    def flat(name):
        return np.array([np.asarray(v, dtype=np.float64) if v is not None else np.full(9, np.nan)
                         for v in table[name]]).reshape(-1)

    t, f, e = flat("af_obs_time"), flat("af_flux"), flat("af_flux_error")
    keep = np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
    return pa.table({"source_id": sid[keep], "transit_id": tid[keep], "ccd": ccd[keep],
                     "t": gaia_time_to_btjd(t[keep]), "flux": f[keep], "err": e[keep]}, schema=CCD_SCHEMA)


def ccd_consistency(ccd_long):
    """Per transit: reduced chi² of the AF CCD fluxes about their weighted mean, and the CCD count.

    A dip lasting hours is common to all nine CCDs (≈ 40 s apart), so it leaves chi² ≈ 1; a cosmic ray, a
    CCD defect or a mis-centred window inflates it. Returns pandas with transit_id, n_ccd, chi2_red.
    """
    import pandas as pd

    df = ccd_long.select(["transit_id", "flux", "err"]).to_pandas()
    w = 1.0 / df["err"] ** 2
    df["w"], df["wf"] = w, w * df["flux"]
    g = df.groupby("transit_id")
    mean = g["wf"].sum() / g["w"].sum()
    df["r2"] = ((df["flux"] - df["transit_id"].map(mean)) / df["err"]) ** 2
    g = df.groupby("transit_id")
    n = g.size()
    chi2 = g["r2"].sum() / (n - 1).where(n > 1)
    return pd.DataFrame({"transit_id": n.index, "n_ccd": n.to_numpy(), "chi2_red": chi2.to_numpy()})


def ccd_at_extremes(feat, ccd_long, nsigma=3.0, trim=2):
    """Per-CCD tests at each source's faintest and brightest transits (features' tid_min, tid_max).

    An occultation lasting more than the ~45 s AF crossing dims all nine AF CCDs; a cosmic ray, a CCD
    defect or a neighbour entering one window moves one or two. At the faintest transit, frac_ccd_low is the
    fraction of AF CCDs below that source's own baseline on that CCD by more than nsigma; at the brightest,
    frac_ccd_high is the mirror. Baseline and scale are per source and CCD (median, and the larger of the
    quoted error and 1.4826 MAD), so CCD calibration offsets cancel. Both sides are computed the same way,
    so the mirror null stays symmetric. chi2_ccd_* is ccd_consistency's reduced chi-squared at that transit.

    Each CCD carries about a third of a transit's S/N, so a dip of Z sigma per transit is only ~Z/3 per CCD
    and the nsigma count stays coarse until Z ~ 9. z_trim_min is the coherent shift with the trim most
    negative CCD z dropped, sum(rest) / sqrt(n - trim) (negative for a dip); z_trim_max drops the trim most
    positive (positive for a spike). A cosmic ray or neighbour sits in the dropped CCDs, while an
    occultation shifts them all. NaN where n <= trim.

    ccd_long: epoch_photometry_ccd_to_long rows covering all transits of the sources (the baseline needs
    them). Returns pandas indexed like feat's source_id, NaN where the transit has no CCD rows.
    """
    import pandas as pd

    df = ccd_long.select(["source_id", "transit_id", "ccd", "flux", "err"]).to_pandas()
    grp = df.groupby(["source_id", "ccd"])["flux"]
    df["base"] = grp.transform("median")
    df["mad"] = (df["flux"] - df["base"]).abs().groupby([df["source_id"], df["ccd"]]).transform("median")
    df["z"] = (df["flux"] - df["base"]) / np.fmax(df["err"], 1.4826 * df["mad"])
    chi = ccd_consistency(ccd_long).set_index("transit_id")
    df["low"], df["high"] = (df["z"] < -nsigma).astype(float), (df["z"] > nsigma).astype(float)
    per = df.groupby(["source_id", "transit_id"]).agg(n=("z", "size"), low=("low", "mean"), high=("high", "mean"))
    df = df.sort_values(["source_id", "transit_id", "z"], kind="stable")
    rank = df.groupby(["source_id", "transit_id"]).cumcount().to_numpy()
    size = df.groupby(["source_id", "transit_id"])["z"].transform("size").to_numpy()
    keys = [df["source_id"], df["transit_id"]]
    per["sum_lo"] = df["z"].where(rank >= trim, 0.0).groupby(keys).sum()  # drop the trim lowest
    per["sum_hi"] = df["z"].where(rank < size - trim, 0.0).groupby(keys).sum()  # drop the trim highest
    kept = (per["n"] - trim).where(per["n"] > trim)
    per["trim_min"], per["trim_max"] = per["sum_lo"] / np.sqrt(kept), per["sum_hi"] / np.sqrt(kept)
    out = pd.DataFrame({"source_id": feat["source_id"].to_numpy()})
    for side, frac in (("min", "low"), ("max", "high")):
        key = pd.MultiIndex.from_arrays([out["source_id"], feat[f"tid_{side}"].to_numpy()])
        row = per.reindex(key)
        out[f"n_ccd_{side}"] = row["n"].to_numpy()
        out[f"frac_ccd_{frac}_{side}"] = row[frac].to_numpy()
        out[f"z_trim_{side}"] = row[f"trim_{side}"].to_numpy()
        out[f"chi2_ccd_{side}"] = chi["chi2_red"].reindex(feat[f"tid_{side}"].to_numpy()).to_numpy()
    return out
