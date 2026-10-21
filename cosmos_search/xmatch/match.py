"""Candidate retrieval and likelihood scoring.

For each Gaia target the question is: which of the nearby survey sources, if any, is the same object?
Answering it needs three numbers per candidate, and nothing else:

1. f, the density of the observed offset if the candidate is the counterpart: a 2-D Gaussian with the
   Gaia position propagated to the candidate's epoch (astrometry.py) plus the survey's catalogue error,
   scaled by s, plus a floor e in quadrature: C = C_gaia(t) + s^2 C_survey + e^2 I.
2. rho, the local sky density of unrelated survey sources around this target, counted in an annulus that
   excludes the counterpart and shrunk toward the regional density when counts are small.
3. pi, the prior probability that this target has a counterpart in the survey at all (strongly dependent on
   magnitude and colour; fitted per stratum by calibrate.py).

With Poisson background, the posterior that candidate j is the counterpart is

    P_j = pi LR_j / ((1 - pi) + pi sum_k LR_k),   LR_j = f_j / rho,

and P_none = (1 - pi) / (same denominator). Magnitudes and colours are deliberately not used: a prior on
the counterpart's photometry would bias the match against exactly the objects the energy-budget test
looks for (infrared excess or its absence).

Shifted sky: every target also gets K pseudo-targets displaced by a random offset in the same annulus.
They have Gaia's motion and errors but no counterpart by construction, so the rate at which they pass a
threshold is the false-match rate for a source with no counterpart, measured in the same density,
crowding and artefact environment as the real one.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .astrometry import Astrometry
from .geometry import (MAS_PER_ARCSEC, MAS_PER_DEG, chord, healpix_groups, healpix_order_for_side,
                       offset_position, random_annulus, separation_mas, tangent_offsets, unit_vectors)

DENSITY_PSEUDOCOUNT = 2.0


@dataclass
class ErrorModel:
    """Offset distribution of true counterparts: a Gaussian core with a broader Gaussian tail.

    core: C = C_gaia(t) + scale^2 C_survey + floor^2 I
    tail: the core plus tail_mas^2 I, with weight tail_frac. Real offsets have tails (unresolved binaries,
    blends that pull a survey centroid toward a neighbour, orbital motion, bad epochs); without a tail term
    the fitted floor inflates to cover them and every match loses discriminating power.
    """

    scale: float = 1.0     # multiplies the survey's catalogue errors
    floor_mas: float = 50.0  # added in quadrature per axis
    tail_frac: float = 0.0
    tail_mas: float = 0.0

    def total_cov(self, cov_gaia, cov_survey):
        c = cov_gaia + self.scale**2 * cov_survey
        c[:, 0, 0] += self.floor_mas**2
        c[:, 1, 1] += self.floor_mas**2
        return c


@dataclass
class Candidates:
    """Every (target or pseudo-target, survey row) pair within the search radius, with what scoring needs."""

    survey: object
    n_targets: int
    n_shifts: int
    # one entry per pair
    target: np.ndarray        # target index
    shift: np.ndarray         # -1 for the real target, 0..K-1 for pseudo-targets
    row: np.ndarray           # index into rows
    unit: np.ndarray          # hypothesis unit: (target, shift, survey group); at most one counterpart each
    d: np.ndarray             # (P, 2) residual offset, mas
    cov_gaia: np.ndarray      # (P, 2, 2)
    cov_survey: np.ndarray    # (P, 2, 2)
    rho: np.ndarray           # background density for the pair's unit, per mas^2
    # one entry per unit
    unit_target: np.ndarray
    unit_shift: np.ndarray
    unit_group: np.ndarray
    rows: pd.DataFrame = field(repr=False)  # survey rows referenced by real pairs, plus _epoch
    n_local: np.ndarray = None  # per target: survey rows within the outer annulus radius (0 = outside footprint)
    rid: np.ndarray = None      # per pair: the survey row's id (real and shifted pairs)
    touched: pd.DataFrame = field(default=None, repr=False)  # every row any pair touches: rid, ra, dec, epoch, cov
    log_comp: np.ndarray = None  # per pair: log of the competition factor (compete.py), or None
    nbrs: object = field(default=None, repr=False)  # Neighbours, when gathered with neighbours=
    claimed: np.ndarray = None  # per target: fraction of the background claimed by Gaia neighbours
    stats: dict = field(default_factory=dict)

    @property
    def real(self):
        return self.shift < 0

    def log_lr(self, model, core_resp=False):
        out = log_lr(self.d, self.cov_gaia, self.cov_survey, self.rho, model, core_resp=core_resp)
        if self.log_comp is None:
            return out
        return (out[0] + self.log_comp, *out[1:])


def log_lr(d, cov_gaia, cov_survey, rho, model, *, core_resp=False):
    """log(f / rho) and the core chi^2 for each pair under an error model.

    With core_resp, also the probability that a true counterpart at this offset belongs to the core.
    """
    c = model.total_cov(cov_gaia.copy(), cov_survey)
    lf, chi2 = _log_gauss(d, c)
    if model.tail_frac > 0:
        ct = c.copy()
        ct[:, 0, 0] += model.tail_mas**2
        ct[:, 1, 1] += model.tail_mas**2
        lt, _ = _log_gauss(d, ct)
        a, b = np.log1p(-model.tail_frac) + lf, np.log(model.tail_frac) + lt
        lf_all = np.logaddexp(a, b)
        resp = np.exp(a - lf_all)
    else:
        lf_all, resp = lf, np.ones(len(lf))
    out = lf_all - np.log(rho)
    return (out, chi2, resp) if core_resp else (out, chi2)


def _log_gauss(d, c):
    det = c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1] ** 2
    dx, dy = d[:, 0], d[:, 1]
    chi2 = (c[:, 1, 1] * dx**2 - 2 * c[:, 0, 1] * dx * dy + c[:, 0, 0] * dy**2) / det
    return -0.5 * chi2 - np.log(2 * np.pi) - 0.5 * np.log(det), chi2


def gather(gaia, survey, source, *, n_shifts=2, seed=0, group_side_deg=0.5, astrometry=None, workers=4,
           neighbours=None):
    """Fetch survey rows around every target and form candidate pairs for real and shifted targets.

    gaia: table with gaia_source columns (source_id, ra, dec, ref_epoch, parallax, pm*, errors, corrs).
    source: object with fetch(survey, ra, dec, radius_deg) -> DataFrame (sources.py).
    Targets are batched by HEALPix pixel (about group_side_deg across) into one cone fetch each; up to
    `workers` fetches run at once. Shifts are seeded per pixel, so results do not depend on fetch order.
    neighbours: (source, gaia_spec, target_ids) to also pair every non-target Gaia source near a touched
    survey row with all survey rows around it, for compete.py. Fetched from the same cone, so each
    neighbour's candidate list is complete.
    """
    from concurrent.futures import ThreadPoolExecutor

    astro = astrometry or Astrometry.from_table(gaia)
    n = len(astro)
    rmin, rmax = (r * MAS_PER_ARCSEC for r in survey.shift_annulus_arcsec)

    # Search centre: Gaia propagated to the survey's nominal epoch. Slack covers per-row epochs that differ
    # from the nominal one, parallax, and 5 sigma of the propagated position.
    off0, cov0 = astro.predict(survey.epoch, "none")
    ra_c, dec_c = offset_position(astro.ra, astro.dec, off0[:, 0], off0[:, 1])
    slack = (astro.total_pm() * survey.epoch_span + np.abs(astro.mean[:, 2])
             + 5 * np.sqrt(cov0[:, 0, 0] + cov0[:, 1, 1]))
    r_search = survey.search_radius_mas + slack
    r_fetch = rmax + r_search

    pix = healpix_groups(ra_c, dec_c, healpix_order_for_side(group_side_deg))
    groups = []
    for p in np.unique(pix):
        idx = np.nonzero(pix == p)[0]
        v = unit_vectors(ra_c[idx], dec_c[idx]).sum(axis=0)
        v /= np.linalg.norm(v)
        cra, cdec = np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(v[2]))
        radius = np.max(separation_mas(cra, cdec, ra_c[idx], dec_c[idx]) + r_fetch[idx]) / MAS_PER_DEG
        groups.append((int(p), idx, cra, cdec, radius))

    parts, kept, compacts = [], [], []
    n_local = np.zeros(n, dtype=np.int64)
    stats = {"groups": len(groups), "rows_fetched": 0}
    nb_parts = []
    target_set = set(np.asarray(neighbours[2]).tolist()) if neighbours else set()

    def fetch(cra, cdec, radius):
        rows = source.fetch(survey, cra, cdec, radius)
        nb = neighbours[0].fetch(neighbours[1], cra, cdec, radius) if neighbours and len(rows) else None
        return rows, nb

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fetch, cra, cdec, radius) for _, _, cra, cdec, radius in groups]
        for (p, idx, _, _, radius), fut in zip(groups, futures):
            rows, nb = fut.result()
            stats["rows_fetched"] += len(rows)
            if len(rows) == 0:
                parts.append(_empty_group(idx, n_shifts))
                continue
            rng = np.random.default_rng([seed, p])
            cat, units, rows_kept, n_local[idx] = _pairs_for_group(idx, astro, survey, rows, ra_c, dec_c, r_search,
                                                                   rmin, rmax, n_shifts, rng, radius * MAS_PER_DEG)
            if nb is not None and cat is not None and len(nb):
                nb = nb[~nb[neighbours[1].id].isin(target_set)].reset_index(drop=True)
                nb_parts.append(_neighbour_pairs(nb, survey, rows, rows_kept[1], rmin, rmax,
                                                 radius * MAS_PER_DEG))
            if cat is not None:
                offset = sum(len(k) for k in kept)
                cat["row"] = np.where(cat["row"] >= 0, cat["row"] + offset, -1)
                kept.append(rows_kept[0])
                compacts.append(rows_kept[1])
            parts.append((cat, units))

    # Rows near group edges are fetched twice; keep the first copy of each survey id.
    rows_all = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    if len(rows_all):
        codes, _ = pd.factorize(rows_all[survey.id])
        first = np.unique(codes, return_index=True)[1]
        rows_out = rows_all.iloc[first].reset_index(drop=True)
        for cat, _ in parts:
            if cat is not None:
                cat["row"] = np.where(cat["row"] >= 0, codes[np.maximum(cat["row"], 0)], -1)
    else:
        rows_out = pd.DataFrame(columns=[*survey.columns, "_epoch"])
    cands = _assemble(parts, survey, n, n_shifts, rows_out, stats)
    cands.n_local = n_local
    cands.touched = (pd.concat(compacts, ignore_index=True).drop_duplicates("rid").reset_index(drop=True)
                     if compacts else pd.DataFrame(columns=["rid", "ra", "dec", "epoch", "c00", "c01", "c11", "gkey"]))
    if neighbours:
        cands.nbrs = Neighbours.concat([q for q in nb_parts if q is not None])
        stats["neighbour_pairs"] = len(cands.nbrs)
    return cands


class Neighbours:
    """Pairs between non-target Gaia sources and survey rows, with what scoring needs."""

    FIELDS = ("key", "rid", "gkey", "d", "cg", "cs", "rho", "gmag")

    def __init__(self, **kw):
        for k in self.FIELDS:
            setattr(self, k, kw[k])

    def __len__(self):
        return len(self.key)

    @classmethod
    def concat(cls, parts):
        if not parts:
            return cls(key=np.zeros(0, np.int64), rid=np.zeros(0, object), gkey=np.zeros(0, object),
                       d=np.zeros((0, 2)), cg=np.zeros((0, 2, 2)), cs=np.zeros((0, 2, 2)), rho=np.zeros(0),
                       gmag=np.zeros(0))
        cat = {k: np.concatenate([q[k] for q in parts]) for k in cls.FIELDS}
        # A neighbour-row pair seen from two overlapping fetch groups is kept once.
        _, first = np.unique(pd.Series(cat["key"]).astype(str).str.cat(pd.Series(cat["rid"]).astype(str), sep="|"),
                             return_index=True)
        return cls(**{k: v[first] for k, v in cat.items()})


def _neighbour_pairs(nb, survey, rows, touched, rmin, rmax, cone_radius):
    """Pair Gaia neighbours near any touched row with every survey row within their search radius."""
    rows = rows.reset_index(drop=True)
    s_ra, s_dec = np.asarray(rows[survey.ra], float), np.asarray(rows[survey.dec], float)
    good = np.isfinite(s_ra) & np.isfinite(s_dec)
    rows, s_ra, s_dec = rows[good].reset_index(drop=True), s_ra[good], s_dec[good]
    astro = Astrometry.from_table(nb)
    off0, cov0 = astro.predict(survey.epoch, "none")
    ra_c, dec_c = offset_position(astro.ra, astro.dec, off0[:, 0], off0[:, 1])
    slack = (astro.total_pm() * survey.epoch_span + np.abs(astro.mean[:, 2])
             + 5 * np.sqrt(cov0[:, 0, 0] + cov0[:, 1, 1]))
    r_search = survey.search_radius_mas + slack
    # Only neighbours that could claim a touched row matter.
    t_tree = cKDTree(unit_vectors(touched.ra.to_numpy(float), touched.dec.to_numpy(float)))
    near = np.array([len(h) > 0 for h in t_tree.query_ball_point(unit_vectors(ra_c, dec_c), chord(r_search))], bool)
    if not near.any():
        return None
    sel = np.nonzero(near)[0]
    astro = astro.subset(sel)
    ra_c, dec_c, r_search = ra_c[sel], dec_c[sel], r_search[sel]
    gmag = np.asarray(nb.get("phot_g_mean_mag", pd.Series(np.nan, index=nb.index)), float)[sel]
    keys = np.asarray(nb["source_id"])[sel]

    epochs, s_cov, s_grp = survey.epochs(rows), survey.covariance(rows), survey.groups(rows)
    gkeys = survey.group_keys(rows)
    tree = cKDTree(unit_vectors(s_ra, s_dec))
    n_grp = int(s_grp.max()) + 1 if len(s_grp) else 1
    regional = np.maximum(np.bincount(s_grp, minlength=n_grp), 1) / (np.pi * cone_radius**2)
    xyz = unit_vectors(ra_c, dec_c)
    outer = tree.query_ball_point(xyz, chord(rmax))
    inner = tree.query_ball_point(xyz, chord(rmin))
    ann = np.pi * (rmax**2 - rmin**2)
    rho = np.array([(np.bincount(s_grp[o], minlength=n_grp) - np.bincount(s_grp[i], minlength=n_grp)
                     + DENSITY_PSEUDOCOUNT) / (ann + DENSITY_PSEUDOCOUNT / regional) for o, i in zip(outer, inner)])
    hits = tree.query_ball_point(xyz, chord(r_search))
    a = np.repeat(np.arange(len(sel)), [len(h) for h in hits])
    if not len(a):
        return None
    r = np.fromiter((j for h in hits for j in h), dtype=np.int64, count=len(a))
    xi, eta = tangent_offsets(astro.ra[a], astro.dec[a], s_ra[r], s_dec[r])
    pred, cg = astro.predict(epochs[r], survey.parallax_mode, idx=a)
    return {"key": keys[a], "rid": np.asarray(rows[survey.id])[r], "gkey": gkeys[r],
            "d": np.column_stack([xi - pred[:, 0], eta - pred[:, 1]]), "cg": cg, "cs": s_cov[r],
            "rho": rho[a, s_grp[r]], "gmag": gmag[a]}


def _pairs_for_group(idx, astro, survey, rows, ra_c, dec_c, r_search, rmin, rmax, n_shifts, rng,
                     cone_radius):
    rows = rows.reset_index(drop=True)
    s_ra, s_dec = np.asarray(rows[survey.ra], float), np.asarray(rows[survey.dec], float)
    good = np.isfinite(s_ra) & np.isfinite(s_dec)
    rows, s_ra, s_dec = rows[good].reset_index(drop=True), s_ra[good], s_dec[good]
    epochs = survey.epochs(rows)
    s_cov = survey.covariance(rows)
    s_grp = survey.groups(rows)
    tree = cKDTree(unit_vectors(s_ra, s_dec))
    n_grp = int(s_grp.max()) + 1 if len(s_grp) else 1

    # Regional density per survey group, the target of the shrinkage.
    cone_area = np.pi * cone_radius**2
    regional = np.maximum(np.bincount(s_grp, minlength=n_grp), 1) / cone_area

    # Local density per (target, group) from the annulus around the propagated position.
    xyz_c = unit_vectors(ra_c[idx], dec_c[idx])
    outer = tree.query_ball_point(xyz_c, chord(rmax))
    inner = tree.query_ball_point(xyz_c, chord(rmin))
    ann_area = np.pi * (rmax**2 - rmin**2)
    rho = np.empty((len(idx), n_grp))
    present = np.zeros((len(idx), n_grp), bool)
    for a, (o, i) in enumerate(zip(outer, inner)):
        cnt = np.bincount(s_grp[o], minlength=n_grp) - np.bincount(s_grp[i], minlength=n_grp)
        rho[a] = (cnt + DENSITY_PSEUDOCOUNT) / (ann_area + DENSITY_PSEUDOCOUNT / regional)
        present[a] = np.bincount(s_grp[o], minlength=n_grp) > 0

    out = {k: [] for k in ("target", "shift", "row", "group", "d", "cg", "cs", "rho")}
    units = []  # (target, shift, group) for every group present around the target
    gkeys = survey.group_keys(rows)
    for k in range(-1, n_shifts):
        if k < 0:
            sx = sy = np.zeros(len(idx))
            qra, qdec = ra_c[idx], dec_c[idx]
        else:
            sx, sy = random_annulus(len(idx), rmin, rmax, rng)
            qra, qdec = offset_position(ra_c[idx], dec_c[idx], sx, sy)
        hits = tree.query_ball_point(unit_vectors(qra, qdec), chord(r_search[idx]))
        a_idx = np.repeat(np.arange(len(idx)), [len(h) for h in hits])
        r_idx = np.fromiter((j for h in hits for j in h), dtype=np.int64, count=len(a_idx))
        pa, pg = np.nonzero(present)
        units.append(np.column_stack([idx[pa], np.full(len(pa), k), pg]))
        if len(a_idx) == 0:
            continue
        t_idx = idx[a_idx]
        xi, eta = tangent_offsets(astro.ra[t_idx], astro.dec[t_idx], s_ra[r_idx], s_dec[r_idx])
        pred, cg = astro.predict(epochs[r_idx], survey.parallax_mode, idx=t_idx)
        d = np.column_stack([xi - pred[:, 0] - sx[a_idx], eta - pred[:, 1] - sy[a_idx]])
        out["target"].append(t_idx)
        out["shift"].append(np.full(len(a_idx), k))
        out["row"].append(r_idx)
        out["group"].append(s_grp[r_idx])
        out["d"].append(d)
        out["cg"].append(cg)
        out["cs"].append(s_cov[r_idx])
        out["rho"].append(rho[a_idx, s_grp[r_idx]])

    # Keep only rows referenced by real pairs; shifted pairs need only their numbers (row = -1).
    n_local = np.array([len(o) for o in outer], dtype=np.int64)
    if not out["target"]:
        return None, np.vstack(units), None, n_local
    cat = {k: np.concatenate(v) for k, v in out.items()}
    # Compact description of every row any pair touches (real or shifted), for the competition step.
    touched = np.unique(cat["row"])
    compact = pd.DataFrame({"rid": np.asarray(rows[survey.id])[touched], "ra": s_ra[touched], "dec": s_dec[touched],
                            "epoch": epochs[touched], "c00": s_cov[touched, 0, 0], "c01": s_cov[touched, 0, 1],
                            "c11": s_cov[touched, 1, 1], "gkey": gkeys[touched]})
    cat["rid"] = np.asarray(rows[survey.id])[cat["row"]]
    real_rows = np.unique(cat["row"][cat["shift"] < 0])
    local = np.full(len(rows), -1, dtype=np.int64)
    local[real_rows] = np.arange(len(real_rows))
    cat["row"] = np.where(cat["shift"] < 0, local[cat["row"]], -1)
    kept = rows.iloc[real_rows].reset_index(drop=True)
    kept["_epoch"] = epochs[real_rows]
    return cat, np.vstack(units), (kept, compact), n_local


def _empty_group(idx, n_shifts):
    t, k = np.meshgrid(idx, np.arange(-1, n_shifts), indexing="ij")
    return None, np.column_stack([t.ravel(), k.ravel(), np.zeros(t.size, np.int64)])


def _assemble(parts, survey, n, n_shifts, rows, stats):
    cats = [c for c, _ in parts if c is not None]
    if cats:
        cat = {k: np.concatenate([c[k] for c in cats]) for k in cats[0]}
    else:
        cat = {"target": np.zeros(0, np.int64), "shift": np.zeros(0, np.int64), "row": np.zeros(0, np.int64),
               "group": np.zeros(0, np.int64), "d": np.zeros((0, 2)), "cg": np.zeros((0, 2, 2)),
               "cs": np.zeros((0, 2, 2)), "rho": np.zeros(0), "rid": np.zeros(0, object)}
    # Hypothesis units (target, shift, group), encoded as one integer. Group indices are local to the fetch
    # a target belongs to, which is all a unit needs. A target or pseudo-target with no survey source
    # nearby still gets one empty unit, so that calibration counts its "no counterpart" outcome.
    span = n_shifts + 1
    found = [u for _, u in parts if len(u)]
    gmax = int(max([1, *(u[:, 2].max() + 1 for u in found), *(cat["group"] + 1)]))
    code = lambda t, k, g: (np.asarray(t, np.int64) * span + (np.asarray(k, np.int64) + 1)) * gmax + g
    real = np.unique(np.concatenate([np.zeros(0, np.int64), *(code(*u.T) for u in found),
                                     code(cat["target"], cat["shift"], cat["group"])]))
    t_all, k_all = np.meshgrid(np.arange(n), np.arange(-1, n_shifts), indexing="ij")
    empty = code(t_all.ravel(), k_all.ravel(), 0)
    empty = empty[~np.isin(empty // gmax, real // gmax)]
    ucodes = np.union1d(real, empty)
    tk, g = ucodes // gmax, ucodes % gmax
    unit = np.searchsorted(ucodes, code(cat["target"], cat["shift"], cat["group"]))
    stats["pairs_real"] = int((cat["shift"] < 0).sum())
    stats["pairs_shifted"] = int((cat["shift"] >= 0).sum())
    return Candidates(
        survey=survey, n_targets=n, n_shifts=n_shifts, target=cat["target"], shift=cat["shift"],
        row=cat["row"], unit=unit, d=cat["d"], cov_gaia=cat["cg"], cov_survey=cat["cs"], rho=cat["rho"],
        unit_target=tk // span, unit_shift=tk % span - 1, unit_group=g,
        rows=rows, rid=cat["rid"], stats=stats,
    )


def posteriors(cands, model, prior):
    """Per-pair posterior P(match) and per-unit P(no counterpart).

    prior: array of pi per target (length n_targets).
    Returns (p_pair, p_none_unit, chi2_pair, log_lr_pair).
    """
    llr, chi2 = cands.log_lr(model)
    p_pair, p_none = unit_posteriors(llr, cands.unit, prior[cands.unit_target])
    return p_pair, p_none, chi2, llr


def unit_posteriors(llr, unit, pi_unit):
    """P_j = pi LR_j / ((1 - pi) + pi sum_k LR_k) within each unit, computed in log space."""
    pi = np.clip(pi_unit, 1e-12, 1 - 1e-12)
    nu = len(pi)
    m = np.full(nu, -np.inf)
    np.maximum.at(m, unit, llr)
    s = np.zeros(nu)
    np.add.at(s, unit, np.exp(llr - m[unit]))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_alt = np.where(s > 0, np.log(pi) + m + np.log(np.where(s > 0, s, 1.0)), -np.inf)
    log_den = np.logaddexp(np.log1p(-pi), log_alt)
    p_none = np.exp(np.log1p(-pi) - log_den)
    p_pair = np.exp(np.log(pi[unit]) + llr - log_den[unit])
    return p_pair, p_none
