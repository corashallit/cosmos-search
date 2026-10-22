"""The event rule of docs/population.md §2, vectorised over every star in a long G light-curve table.

Rows must be sorted by (source_id, t). The same functions run on real, injected and mirrored light curves, and the
forward models in `exposure` apply the rule to predicted depths, so all three see one definition.
"""

import numpy as np
import pandas as pd

from . import split

MIN_EPOCHS = 15
DIP_FLOOR = 4.0          # in-dip: z above max(this, the second-largest excursion of the other sign)
FLAT_SIGMA = 2.0         # an epoch less than this many robust sigma off is at normal brightness
FLAG_VARI_REJECT = 1
CLASSES = ("SOV", "SOV1", "MIR", "MIR1")   # dips on >= 2 epochs, on 1 epoch; the mirrored (brightening) pair


def group_index(sid):
    """For sorted ids: group per row, the unique ids, and each group's first row."""
    new = np.ones(len(sid), bool)
    new[1:] = sid[1:] != sid[:-1]
    start = np.flatnonzero(new)
    return np.cumsum(new) - 1, sid[start], start


def group_median(gid, x, n):
    """Median of x within each of n groups (rows of a group need not be contiguous); NaN for empty groups."""
    order = np.lexsort((x, gid))
    g, v = gid[order], x[order]
    c = np.bincount(g, minlength=n)
    start = np.concatenate([[0], np.cumsum(c)[:-1]])
    out = np.full(n, np.nan)
    ok = c > 0
    lo = start[ok] + (c[ok] - 1) // 2
    hi = start[ok] + c[ok] // 2
    out[ok] = 0.5 * (v[lo] + v[hi])
    return out


def group_kth(gid, x, n, k):
    """k-th smallest x per group (k may be negative: -1 the largest), NaN where a group has too few rows."""
    order = np.lexsort((x, gid))
    g, v = gid[order], x[order]
    c = np.bincount(g, minlength=n)
    start = np.concatenate([[0], np.cumsum(c)[:-1]])
    out = np.full(n, np.nan)
    kk = k if k >= 0 else None
    ok = c > (k if k >= 0 else -k - 1)
    pos = start[ok] + (kk if kk is not None else c[ok] + k)
    out[ok] = v[pos]
    return out


def prepare(table, keep_rejected=False, min_epochs=MIN_EPOCHS):
    """Good G epochs of stars with enough of them, with per-star scale. table: dict-like of arrays or a DataFrame
    with source_id, t, flux, err_rel, flags (and optionally fov, dbp, drp). Returns a dict of row arrays and a
    DataFrame of stars."""
    cols = {k: np.asarray(table[k]) for k in ("source_id", "t", "flux", "err_rel", "flags")}
    extra = {k: np.asarray(table[k]) for k in ("fov", "dbp", "drp") if k in table}
    good = np.isfinite(cols["flux"]) & (cols["flux"] > 0) & np.isfinite(cols["t"])
    if not keep_rejected:
        good &= (cols["flags"] & FLAG_VARI_REJECT) == 0
    rows = {k: v[good] for k, v in {**cols, **extra}.items()}
    gid, uniq, _ = group_index(rows["source_id"])
    n = np.bincount(gid, minlength=len(uniq))
    ok = n[gid] >= min_epochs
    rows = {k: v[ok] for k, v in rows.items()}
    gid, uniq, start = group_index(rows["source_id"])
    ng = len(uniq)
    f = rows["flux"].astype(np.float64)
    med = group_median(gid, f, ng)
    r = f / med[gid]
    sigma = 1.4826 * group_median(gid, np.abs(r - 1), ng)
    vid, vhalf = split.visits(gid, rows["t"])
    rows.update(gid=gid, r=r, vid=vid, vhalf=vhalf[vid])
    first_v = vid[start]
    n_vis = np.bincount(gid[np.r_[True, vid[1:] != vid[:-1]]], minlength=ng)
    vis_gid = gid[np.r_[True, vid[1:] != vid[:-1]]]
    n_vis_b = np.bincount(vis_gid, weights=vhalf.astype(float), minlength=ng)
    stars = pd.DataFrame({
        "source_id": uniq, "n_good": np.bincount(gid, minlength=ng), "n_visits": n_vis,
        "n_visits_A": n_vis - n_vis_b.astype(int), "n_visits_B": n_vis_b.astype(int),
        "sigma": sigma, "err_med": group_median(gid, rows["err_rel"].astype(np.float64), ng),
        "t_first": rows["t"][start], "t_last": rows["t"][np.r_[start[1:], len(gid)] - 1], "first_visit": first_v,
    })
    with np.errstate(divide="ignore", invalid="ignore"):
        stars["excess"] = stars.sigma / stars.err_med
    z = (1 - r) / sigma[gid]
    rows["z"] = z
    stars["K"] = np.fmax(DIP_FLOOR, -group_kth(gid, z, ng, 1))        # second-largest brightening
    stars["K_mir"] = np.fmax(DIP_FLOOR, group_kth(gid, z, ng, -2))    # second-largest dip
    return rows, stars


def classify(gid, vid, z, K, n_groups):
    """The event rule on per-epoch excursions z (positive = the direction searched) with per-group threshold K.

    Returns per group: n_events, n_in (in-dip epochs), one_visit, bracketed, the event's visit and its deepest row
    (-1 where there is no in-dip epoch)."""
    indip = z > K[gid]
    flat = z < FLAT_SIGMA
    cumflat = np.cumsum(flat)
    idx = np.flatnonzero(indip)
    g = gid[idx]
    newev = np.ones(len(idx), bool)
    newev[1:] = (g[1:] != g[:-1]) | (cumflat[idx[1:]] - cumflat[idx[:-1]] > 0)
    n_events = np.bincount(g[newev], minlength=n_groups)
    n_in = np.bincount(g, minlength=n_groups)
    big = np.iinfo(np.int64).max
    vmin = np.full(n_groups, big)
    vmax = np.full(n_groups, -1)
    np.minimum.at(vmin, g, vid[idx])
    np.maximum.at(vmax, g, vid[idx])
    fidx = np.flatnonzero(flat)
    fmin = np.full(n_groups, big)
    fmax = np.full(n_groups, -1)
    np.minimum.at(fmin, gid[fidx], vid[fidx])
    np.maximum.at(fmax, gid[fidx], vid[fidx])
    deep = np.full(n_groups, -1, np.int64)
    if len(idx):
        o = np.lexsort((-z[idx], g))
        first = np.r_[True, g[o][1:] != g[o][:-1]]
        deep[g[o][first]] = idx[o][first]
    has = n_in > 0
    return {"n_events": n_events, "n_in": n_in, "one_visit": has & (vmin == vmax),
            "bracketed": has & (fmin < vmin) & (fmax > vmax), "visit": np.where(has, vmin, -1), "deep": deep}


def find(rows, stars):
    """Single one-visit events of every class (docs/population.md §2) as a DataFrame, one row per event."""
    ng = len(stars)
    gid, vid = rows["gid"], rows["vid"]
    out = []
    for sign, kcol, (multi, single) in ((1, "K", ("SOV", "SOV1")), (-1, "K_mir", ("MIR", "MIR1"))):
        c = classify(gid, vid, sign * rows["z"], stars[kcol].to_numpy(), ng)
        base = (c["n_events"] == 1) & c["one_visit"] & c["bracketed"]
        for name, m in ((multi, base & (c["n_in"] >= 2)), (single, base & (c["n_in"] == 1))):
            j = np.flatnonzero(m)
            d = c["deep"][j]
            ev = pd.DataFrame({
                "source_id": stars.source_id.to_numpy()[j], "cls": name, "half": rows["vhalf"][d],
                "t_deep": rows["t"][d], "depth": sign * (1 - rows["r"][d]), "z": sign * rows["z"][d],
                "n_in": c["n_in"][j], "visit": c["visit"][j] - stars.first_visit.to_numpy()[j],
            })
            for k in ("dbp", "drp"):
                if k in rows:
                    ev[k] = rows[k][d].astype(np.float64)
            if "fov" in rows:
                ev["fov"] = rows["fov"][d]
            out.append(ev)
    return pd.concat(out, ignore_index=True)
