"""Brightness offsets shared by stars in the same patch of sky at the same visit, measured from quiet neighbours.

Gaia observes all stars in a small patch of sky in the same scans, so a calibration error of one scan (one
visit) is shared by its neighbours and differs from patch to patch. The drift correction (`drift`) removes what
stars of the same magnitude and colour share across the whole sky; what is left on DR3 shows up as a sky pattern
in the stars' trends (docs/arms/slow-changes.md, S2). This module measures it per patch and visit from quiet
neighbouring stars, as differential photometry against comparison stars does:

    offset(band, patch, visit) = sum_k w_k ybar_k / sum_k w_k,   w_k = 1 / sigma_k^2

over comparison stars k in the patch with epochs in that visit, ybar_k the mean drift-corrected fractional flux of
star k's epochs in the visit, and sigma_k that star's robust epoch scatter. A star whose ybar_k lies beyond
CLIP sigma_k is left out (a comparison star with its own event). Each offset is then shrunk towards zero by
tau^2 / (tau^2 + err^2), with tau^2 the variance of the true offsets estimated from all of them
(`shared_scatter`), so a patch with few comparison stars adds little noise.

A "patch" is a HEALPix pixel (`drift.sky_pixel`) when every star is corrected from the stars around it, as on DR3;
for one candidate on DR4 it is that candidate and the quiet neighbours whose light curves were fetched for it.
Comparison stars must never include the star being corrected: on DR3 each half of the stars (`drift.half`) is
corrected from the quiet stars of the other half.

    off = offsets(ref_epochs.assign(patch=...), sigma=...)   # per band, patch, visit
    epochs["frac_sky"] = epochs["frac_corr"] - offset_at(epochs.assign(patch=...), off)
"""

import numpy as np
import pandas as pd

VISIT_GAP_DAYS = 1.0   # a patch's epochs closer than this belong to one visit (the scans of one passage)
MATCH_DAYS = 0.5       # an epoch belongs to a visit if it lies within this of the visit's first or last epoch
CLIP = 4.0
MIN_STARS = 3


def star_sigma(epochs, col="frac_corr"):
    """Robust epoch scatter per (source_id, band): 1.4826 x MAD of `col`."""
    g = epochs.groupby(["source_id", "band"])[col]
    med = g.transform("median")
    mad = (epochs[col] - med).abs().groupby([epochs["source_id"], epochs["band"]]).median()
    return (1.4826 * mad).clip(lower=1e-5).rename("sigma")


def offsets(ref, sigma=None, col="frac_corr", min_stars=MIN_STARS):
    """Offset per (band, patch, visit) from comparison-star epochs (source_id, band, t, `col`, patch).

    Returns band, patch, t_lo, t_hi, offset (shrunk), raw, err, n_stars; tau per band is in attrs["tau"]."""
    sigma = star_sigma(ref, col) if sigma is None else sigma
    e = ref[["source_id", "band", "patch", "t", col]].sort_values(["band", "patch", "t"], kind="stable")
    band, patch, t = e["band"].to_numpy(), e["patch"].to_numpy(), e["t"].to_numpy()
    new = np.r_[True, (band[1:] != band[:-1]) | (patch[1:] != patch[:-1]) | (np.diff(t) > VISIT_GAP_DAYS)]
    e = e.assign(visit=np.cumsum(new) - 1)
    per = e.groupby(["visit", "source_id"]).agg(band=("band", "first"), patch=("patch", "first"),
                                                 y=(col, "mean")).reset_index()
    per = per.join(sigma, on=["source_id", "band"])
    per = per[np.abs(per["y"]) <= CLIP * per["sigma"]]
    per["w"] = 1.0 / per["sigma"] ** 2
    per["wy"] = per["w"] * per["y"]
    v = per.groupby("visit").agg(band=("band", "first"), patch=("patch", "first"), sw=("w", "sum"),
                                 swy=("wy", "sum"), n_stars=("y", "size"))
    v["raw"] = v["swy"] / v["sw"]
    # Error from the comparison stars' own scatter about the offset (scaled by their reduced chi^2 when > 1).
    per = per.join(v["raw"], on="visit")
    chi2 = (per["w"] * (per["y"] - per["raw"]) ** 2).groupby(per["visit"]).sum()
    dof = (v["n_stars"] - 1).clip(lower=1)
    v["err"] = np.sqrt(np.maximum(chi2 / dof, 1.0) / v["sw"])
    span = e.groupby("visit")["t"].agg(t_lo="min", t_hi="max")
    v = v.join(span)
    v = v[v["n_stars"] >= min_stars].reset_index(drop=True)
    tau = {int(b): shared_scatter(d["raw"].to_numpy(), d["err"].to_numpy()) for b, d in v.groupby("band")}
    t2 = v["band"].map(tau).to_numpy() ** 2
    v["offset"] = v["raw"] * np.where(t2 > 0, t2 / (t2 + v["err"] ** 2), 0.0)
    out = v[["band", "patch", "t_lo", "t_hi", "offset", "raw", "err", "n_stars"]]
    out.attrs["tau"] = tau
    return out


def shared_scatter(raw, err):
    """Scatter tau of the true offsets behind measured ones (raw, with errors err), by maximum likelihood with
    raw ~ N(0, err^2 + tau^2), on a grid. Most visits of a patch are measured on faint stars with errors far above
    tau; the likelihood weights the precise ones as it should, where an unweighted mean of raw^2 - err^2 is lost
    in the noise."""
    raw, err = np.asarray(raw, float), np.asarray(err, float)
    grid = np.r_[0.0, np.geomspace(1e-5, 0.1, 161)]
    var = err[None, :] ** 2 + grid[:, None] ** 2
    ll = -0.5 * np.sum(np.log(var) + raw[None, :] ** 2 / var, axis=1)
    return float(grid[int(np.argmax(ll))])


def offset_at(epochs, off):
    """Offset for each epoch (source_id, band, t, patch) from `offsets`: its patch's visit containing it, else 0."""
    if not len(off):
        return np.zeros(len(epochs))
    e = epochs[["band", "patch", "t"]].reset_index(drop=True)
    e["_row"] = np.arange(len(e))
    e = e.sort_values("t", kind="stable")
    o = off.sort_values("t_lo", kind="stable")
    keys = ["band", "patch"]
    m = pd.merge_asof(e, o[keys + ["t_lo", "t_hi", "offset"]], left_on="t", right_on="t_lo", by=keys,
                      tolerance=None, direction="backward", allow_exact_matches=True)
    late = pd.merge_asof(e, o[keys + ["t_lo", "t_hi", "offset"]], left_on="t", right_on="t_lo", by=keys,
                         direction="forward")
    ok = (m["t"] <= m["t_hi"] + MATCH_DAYS).to_numpy()
    ok_late = (late["t_lo"] - late["t"] <= MATCH_DAYS).to_numpy()
    val = np.where(ok, m["offset"].to_numpy(), np.where(ok_late, late["offset"].to_numpy(), 0.0))
    val = np.nan_to_num(val, nan=0.0)
    out = np.empty(len(e))
    out[e["_row"].to_numpy()] = val
    return out


def correct_by_halves(epochs, ref_ids, level, col="frac_corr", half=None, pix=None):
    """DR3-style correction of every star from the quiet stars (`ref_ids`) of the other half, in HEALPix
    pixels at `level`. half / pix: optional Series by source_id overriding drift.half / drift.sky_pixel (for
    fakes, which take their host's). Returns the offset per epoch and the two offset tables."""
    from cosmos_search.slow import drift

    sid = epochs["source_id"].to_numpy()
    h = drift.half(sid) if half is None else half.reindex(sid).to_numpy()
    p = drift.sky_pixel(np.abs(sid), level) if pix is None else pix.reindex(sid).to_numpy()
    e = epochs.assign(patch=p, _half=h)
    is_ref = np.isin(sid, np.fromiter(ref_ids, np.int64)) if not isinstance(ref_ids, np.ndarray) \
        else np.isin(sid, ref_ids)
    out = np.zeros(len(e))
    tables = {}
    for target in (0, 1):
        ref = e[is_ref & (e["_half"].to_numpy() == 1 - target)]
        off = offsets(ref, col=col)
        tables[1 - target] = off
        m = e["_half"].to_numpy() == target
        out[m] = offset_at(e[m], off)
    return out, tables


SHARED_WINDOW_DAYS = 1.5
SHARED_P = 1e-3


def shared_changes(stats, patch, targets=None, target_patch=None, date_cols=("dim_t0", "rise_t0"),
                   score_cols=("s_fade", "s_rise"), s_min=4.0, window=SHARED_WINDOW_DAYS, p_max=SHARED_P):
    """How many other stars in the same patch changed on the same date, and whether that is more than chance.

    In crowded DR3 fields hundreds of stars in one level-6 pixel step down at the same visit (BTJD 269.2, 2015-09-07,
    in one bulge pixel: 383 of 6,103 stars with s_fade > 4): an artefact of that scan (likely crowding, which
    depends on scan direction), invisible in a single light curve and mirror-asymmetric, so the brightening
    mirror does not measure it. A real occulter changes one star.

    For each star and each direction (fade, rise), its change date is when it entered its faint (bright) window.
    Among the other stars in its patch whose score in either direction exceeds s_min, n_same counts those whose
    change date in either direction lies within `window` days. Under chance, change dates spread over the
    patch's distinct dates, so the expected count is (strong stars - 1) / (distinct dates); `shared` is True when
    a Poisson count that large has probability below p_max and n_same >= 2.

    stats: one row per star with the score and date columns; patch: its patch ids (e.g. drift.sky_pixel(source_id,
    6), or a candidate's neighbour group). targets / target_patch: stars to test against those neighbours instead
    of `stats` itself (fakes placed in their hosts' patches); their own events are then not in the counts.
    Returns a DataFrame indexed like the tested stars: n_same_fade, n_same_rise, expected, shared_fade,
    shared_rise."""
    from scipy.stats import poisson

    s = stats.reset_index(drop=True)
    patch = np.asarray(patch)
    own_in = targets is None
    tg = s if own_in else targets.reset_index(drop=True)
    tpatch = patch if own_in else np.asarray(target_patch)
    events = []
    for d, c in zip(date_cols, score_cols):
        m = (s[c].to_numpy() > s_min) & np.isfinite(s[d].to_numpy())
        events.append(pd.DataFrame({"row": np.flatnonzero(m), "patch": patch[m], "t": s[d].to_numpy()[m]}))
    ev = pd.concat(events, ignore_index=True).drop_duplicates(["row", "t"])
    out = pd.DataFrame(index=tg.index, data={"n_same_fade": 0, "n_same_rise": 0, "expected": 0.0})
    for p, e in ev.groupby("patch"):
        m = tpatch == p
        if not m.any():
            continue
        t = np.sort(e["t"].to_numpy())
        dates = t[np.r_[True, np.diff(t) > window]]
        n_strong = e["row"].nunique()
        for k, (d, c) in enumerate(zip(date_cols, score_cols)):
            mine = tg.loc[m, d].to_numpy()
            ok = np.isfinite(mine)
            cnt = np.where(ok, np.searchsorted(t, np.where(ok, mine, 0) + window, "right")
                           - np.searchsorted(t, np.where(ok, mine, 0) - window, "left"), 0)
            if own_in:  # leave the star's own event out (in the list when its score passed s_min)
                cnt = np.maximum(cnt - (tg.loc[m, c].to_numpy() > s_min).astype(int), 0)
            out.loc[m, ["n_same_fade", "n_same_rise"][k]] = cnt
        out.loc[m, "expected"] = max(n_strong - (1 if own_in else 0), 0) / max(len(dates), 1)
    for k in ("fade", "rise"):
        n = out[f"n_same_{k}"].to_numpy()
        out[f"shared_{k}"] = (n >= 2) & (poisson.sf(n - 1, np.maximum(out["expected"].to_numpy(), 1e-9)) < p_max)
    return out
