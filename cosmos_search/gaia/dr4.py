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
(`ccd_consistency`). Bit 1 is never set from epoch_photometry alone: the draft model has no variability-rejection
column (DR3's `variability_flag_*_reject`), so the 2-5% of epochs DPAC's variability processing rejected in DR3
arrive unmarked unless `vari_epoch_flags` (a separate DataLink product) is joined.

Column names may change by release. `COLUMNS` and `ALIASES` are the one place to fix them: `resolve` matches names
without regard to case and accepts DR3's spellings (`g_transit_time`, `g_transit_flux`, ...). Fluxes are float32
in the draft model and are widened here. A table with one row per source and array-valued transit columns (DR3's
layout, and how AIP served DR3) is unpacked by `explode`.
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .epoch import (
    BANDS,
    FLAG_PHOT_REJECT,
    FLAG_VARI_REJECT,
    SCHEMA,
    decode_transit_id,
    gaia_time_to_btjd,
)

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

# Other spellings a release (or a mirror) may use for a column, tried in order after the name itself.
ALIASES = {
    "g_obs_time": ("g_transit_time",), "g_flux": ("g_transit_flux",), "g_flux_error": ("g_transit_flux_error",),
    "g_n_obs": ("g_transit_n_obs",),
}
# What epoch_photometry_to_lc reads; only REQUIRED must be present.
READ_COLUMNS = ("source_id", "transit_id", "g_n_obs", "transit_proc_flags", "phot_ccd_proc_flags",
                *(c for cols in COLUMNS.values() for c in cols))
REQUIRED = ("source_id", "transit_id", *COLUMNS["G"])

CCD_SCHEMA = pa.schema([
    ("source_id", pa.int64()), ("transit_id", pa.int64()), ("ccd", pa.int8()), ("t", pa.float64()),
    ("flux", pa.float64()), ("err", pa.float64()),
])


def _names(table):
    """Column names of an astropy Table, a pyarrow Table (bulk Parquet files) or a pandas frame (DataLink csv)."""
    if hasattr(table, "colnames"):
        return table.colnames
    return table.column_names if hasattr(table, "column_names") else list(table.columns)


def resolve(names):
    """{name in READ_COLUMNS: the table's own spelling of it}, for the columns the table has."""
    lower = {str(n).lower(): n for n in names}
    out = {}
    for c in READ_COLUMNS:
        for cand in (c, *ALIASES.get(c, ())):
            if cand in lower:
                out[c] = lower[cand]
                break
    return out


def missing_required(names):
    return [c for c in REQUIRED if c not in resolve(names)]


def _col(table, name, dtype=np.float64):
    fill = np.nan if np.dtype(dtype).kind == "f" else 0
    col = table[name]
    if isinstance(col, (pa.ChunkedArray, pa.Array)):
        return np.asarray(col.fill_null(fill) if col.null_count else col, dtype=dtype)
    if hasattr(col, "filled"):  # astropy masked column
        return np.asarray(col.filled(fill), dtype=dtype)
    if hasattr(col, "to_numpy") and hasattr(col, "isna"):  # pandas
        return col.to_numpy(dtype=np.float64, na_value=np.nan).astype(dtype) if np.dtype(dtype).kind == "f" \
            else col.fillna(0).to_numpy(dtype=dtype)
    return np.asarray(col, dtype=dtype)


def _is_list(table, name):
    if isinstance(table, pa.Table):
        t = table.schema.field(name).type
        return pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)
    return False


def explode(table):
    """One row per transit from a pyarrow table with one row per source and array-valued transit columns: every
    list column as long as transit_id's is unpacked, the others are repeated."""
    tid = resolve(table.column_names)["transit_id"]
    ref = pc.list_value_length(table[tid]).fill_null(0)
    parent = pc.list_parent_indices(table[tid].combine_chunks())
    cols = {}
    for name in table.column_names:
        col = table[name]
        if _is_list(table, name) and pc.list_value_length(col).fill_null(0).equals(ref):
            cols[name] = pc.list_flatten(col.combine_chunks())
        else:
            cols[name] = col.take(parent)
    return pa.table(cols)


def transit_flags(table):
    """Per-transit flag bits (before band-specific ones) from DR4 transit_proc_flags and phot flags."""
    n = len(table)
    flags = np.zeros(n, np.int32)
    names = resolve(_names(table))
    if "transit_proc_flags" in names:
        proc = _col(table, names["transit_proc_flags"], np.int64) & 0xFFFF
        ipd = proc & 0xF
        flags |= np.where((ipd >= 1) & (ipd <= 3), FLAG_MULTI_SOURCE, 0)
        flags |= np.where(proc & 0x3800, FLAG_COSMETIC, 0)
        flags |= np.where(proc & 0x8000, FLAG_ATTITUDE, 0)
    phot = _phot_flags(table)
    if phot is not None:
        flags |= np.where((phot[:, _PHOT_INDEX["AF"]] & 0x1).any(axis=1), FLAG_SATURATION, 0)
        flags |= np.where((phot & 0x100).any(axis=1), FLAG_SATELLITE_EVENT, 0)
    return flags


def _flag_cell(v):
    """One row's phot_ccd_proc_flags as 12 ints: an array, a list, or text such as "[0, 8192, ...]" (csv)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.zeros(12, np.int64)
    if isinstance(v, (str, bytes)):
        text = v.decode() if isinstance(v, bytes) else v
        parts = text.strip().strip("[](){}").replace(",", " ").split()
        v = [int(float(x)) if x.lower() not in ("null", "nan", "") else 0 for x in parts]
    out = np.zeros(12, np.int64)
    arr = np.asarray(v, dtype=np.int64).ravel()[:12]
    out[:len(arr)] = arr
    return out


def _phot_flags(table):
    name = resolve(_names(table)).get("phot_ccd_proc_flags")
    if name is None:
        return None
    raw = table[name]
    if isinstance(raw, pa.ChunkedArray) and not raw.null_count and _is_list(table, name):
        if raw.num_chunks:
            flat = np.concatenate([np.asarray(c.flatten(), dtype=np.int64) for c in raw.chunks])
            if len(flat) == 12 * len(raw):  # vectorised path for Parquet list columns
                return flat.reshape(len(raw), 12) & 0xFFFF
    cells = raw.to_pylist() if isinstance(raw, (pa.ChunkedArray, pa.Array)) else raw
    if not len(table):
        return np.zeros((0, 12), np.int64)
    return np.array([_flag_cell(v) for v in cells]).reshape(len(table), 12) & 0xFFFF


def epoch_photometry_to_lc(table, vari_rejected=None, max_rows=1 << 18):
    """DR4 epoch_photometry rows (pyarrow or astropy Table or pandas frame, one per transit) -> light-curve table
    (epoch.SCHEMA), sorted by source, band and time.

    vari_rejected: optional dict {(transit_id, band): True} from vari_epoch_flags.

    A pyarrow table longer than `max_rows` whose stars are in source_id order is converted a slice of whole stars
    at a time (the same result; the working memory of the conversion is about 0.7 kB per transit row otherwise).
    """
    names = resolve(_names(table))
    missing = [c for c in REQUIRED if c not in names]
    if missing:
        raise KeyError(f"no column(s) {missing}; columns are {sorted(map(str, _names(table)))} "
                       "(update gaia.dr4.COLUMNS or ALIASES)")
    if _is_list(table, names["g_flux"]):
        table = explode(table)
    if isinstance(table, pa.Table) and table.num_rows > max_rows:
        sid = _col(table, names["source_id"], np.int64)
        starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
        if np.all(np.diff(sid[starts]) > 0):
            cuts = [0]
            while cuts[-1] < len(sid):
                k = np.searchsorted(starts, cuts[-1] + max_rows, side="right") - 1
                nxt = starts[k] if starts[k] > cuts[-1] else (starts[k + 1] if k + 1 < len(starts) else len(sid))
                cuts.append(int(nxt) if cuts[-1] + max_rows < len(sid) else len(sid))
            del sid
            return pa.concat_tables([_to_lc(table.slice(a, b - a), names, vari_rejected)
                                     for a, b in zip(cuts, cuts[1:])])
    return _to_lc(table, names, vari_rejected)


def _to_lc(table, names, vari_rejected):
    sid = _col(table, names["source_id"], np.int64)
    tid = _col(table, names["transit_id"], np.int64)
    _, fov, row, _ = decode_transit_id(tid)
    base = transit_flags(table)
    phot = _phot_flags(table)
    n_obs = np.clip(_col(table, names["g_n_obs"], np.int64), 0, 127).astype(np.int8) if "g_n_obs" in names \
        else np.zeros(len(table), np.int8)
    parts = []
    for code, band in enumerate(BANDS):
        tcol, fcol, ecol = COLUMNS[band]
        if not all(c in names for c in (tcol, fcol)):
            continue
        t, f = _col(table, names[tcol]), _col(table, names[fcol])
        e = _col(table, names[ecol]) if ecol in names else np.full(len(t), np.nan)
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
