"""Stage 0 checks that need no model: could the source's light have done what an epoch shows?

Band consistency. G spans the BP and RP passbands, so any change in the light reaching the instrument, grey or
chromatic, gives a fractional G change close to a flux-weighted mean of the BP and RP changes at the same
transit, d_G ~ w d_BP + (1 - w) d_RP, with w the BP share of the G signal. w depends on colour; bounding it in
W_RANGE (wide enough for M dwarfs to blue stars) keeps the test free of a spectral model, and a chromatic sign
flip (BP down, RP up) still passes. A G change outside that interval by more than the noise is a measurement
problem in one instrument. Faults are excursions, so the instrument at fault is the one whose change is larger
in units of its own scatter: "band_inconsistent" when that is G (the common DR3 case is a G-only dip with BP
and RP flat; pilot 3 found about half the detector's non-EB triggers to be of this kind, docs/vetting.md §11),
"colour_fault" when it is BP or RP (DR3 has BP or RP values tens of times the median at otherwise good transits).

Noise per epoch is the larger of the quoted errors and the star's own scatter in the band-consistency
residual (DR3 errors are often underestimated, and a variable star's colour changes add scatter that is no
evidence of a fault). The test is only made where BP and RP are measured well enough to have shown the change.

Error inflation. An epoch whose G error is many times the star's median comes from a degraded transit (bad CCD
samples, gating, window problems), so its flux is not evidence of anything.
"""

from dataclasses import replace

import numpy as np

from .packet import robust_sigma

MATCH_DAYS = 0.01  # BP and RP epochs carry slightly different times from G's within one transit
W_RANGE = (0.1, 0.9)  # BP share of G: ~0.2 for M dwarfs, ~0.6 for A stars
ARTEFACTS = ("band_inconsistent", "inflated_error")  # G epochs that are no evidence about the star


def _matched(lc, t):
    """Fractional change (1 - f / median) and its error at each time in t, NaN where lc has no epoch in the transit."""
    d, e = np.full(len(t), np.nan), np.full(len(t), np.nan)
    if lc is None or len(lc.time) == 0:
        return d, e
    order = np.argsort(lc.time)
    ts = lc.time[order]
    j = np.clip(np.searchsorted(ts, t), 0, len(ts) - 1)
    prev = np.clip(j - 1, 0, len(ts) - 1)
    j = np.where(np.abs(ts[prev] - t) < np.abs(ts[j] - t), prev, j)
    ok = np.abs(ts[j] - t) <= MATCH_DAYS
    med = float(np.median(lc.flux))
    idx = order[j[ok]]
    d[ok] = 1 - lc.flux[idx] / med
    e[ok] = lc.flux_err[idx] / med
    return d, e


VERDICTS = ("consistent", "band_inconsistent", "colour_fault", "inflated_error", "untested")


def _robust(x):
    x = x[np.isfinite(x)]
    return robust_sigma(x) if len(x) >= 5 else 0.0


PARTNER_DAYS = 0.1  # the other field of view's transit follows ~0.074 d later
BLEND_MAX = 0.05  # blended-transit fraction below which a quiet source is taken as unblended
RATIO_SIG = 10.0  # G changes (in G errors) used to measure the BP/RP dilution ratio


def _partner(t):
    """Index of the nearest other G epoch within PARTNER_DAYS (the other field of view's transit), else -1."""
    t = np.asarray(t, float)
    order = np.argsort(t)
    ts = t[order]
    best = np.full(len(t), -1)
    if len(t) < 2:
        return best
    gap_prev = np.r_[np.inf, np.diff(ts)]
    gap_next = np.r_[np.diff(ts), np.inf]
    use_next = gap_next < gap_prev
    nearest = np.where(use_next, np.arange(len(ts)) + 1, np.arange(len(ts)) - 1)
    ok = np.minimum(gap_prev, gap_next) <= PARTNER_DAYS
    best[order[ok]] = order[nearest[ok]]
    return best


def dilution_floor(d_g, e_g, d_bp, d_rp, blend_frac=None, min_epochs=5):
    """Lower bound on r = (BP/RP response) / (G response) to a change in the source's light, for one source.

    BP and RP are measured in a 3.5 x 2.1 arcsec window, G in a 0.7 x 2.1 arcsec one, so a neighbour adds
    constant light to BP and RP and dilutes the source's changes there by a factor r <= 1. A real G dip then sits
    outside the unblended interval by about 1/r (DR3 eclipsing binaries in the LMC: dips removed with BP/RP
    blended in every transit). Where the source changes significantly in G at >= min_epochs epochs, r is
    measured from them: a generous fraction of the lower quintile of mix / d_G for a source known to be
    unblended (blend_frac < BLEND_MAX), of the minimum otherwise, because a neighbour enters the BP/RP window
    only at some scan angles and so dilutes only some transits. The minimum includes the epoch under test:
    leaving it out gains ~4 points of G-only recall but vetoed 2-8% of diluted real changes in the injection
    test (docs/vetting.md §12). A quiet source gives no measurement: r = 1 if
    it is known to be unblended, else 0, which leaves every G change untested rather than vetoed. On a
    blended source the test is therefore lenient by design.
    """
    unblended = blend_frac is not None and np.isfinite(blend_frac) and blend_frac < BLEND_MAX
    mix = 0.5 * (d_bp + d_rp)
    sig = np.isfinite(mix) & (np.abs(d_g) > RATIO_SIG * e_g)
    if sig.sum() >= min_epochs:
        ratio = mix[sig] / d_g[sig]
        return float(np.clip(0.7 * (np.quantile(ratio, 0.2) if unblended else ratio.min()), 0.0, 1.0))
    return 1.0 if unblended else 0.0


def _assess(d_g, e_g, d_bp, e_bp, d_rp, e_rp, nsigma, err_ratio, w_range, r_lo=1.0, partner=None):
    """Verdicts from per-G-epoch fractional changes and errors, with BP/RP NaN where the transit has none.

    r_lo (per source; scalar or per-epoch array) is dilution_floor's bound: the allowed G interval is the
    BP-RP mixing interval scaled by every r in [r_lo, 1], and BP/RP noise is scaled by 1 / r_lo. partner is
    _partner's index array; an inflated G error is only a fault when neither BP/RP nor the partner transit
    corroborates the change.
    """
    r_lo = np.broadcast_to(np.asarray(r_lo, float), np.shape(d_g))
    inv = np.divide(1.0, r_lo, out=np.full(np.shape(d_g), np.inf), where=r_lo > 0)
    ends = np.array([w * d_bp + (1 - w) * d_rp for w in w_range])
    lo0, hi0 = ends.min(axis=0), ends.max(axis=0)
    with np.errstate(invalid="ignore"):
        lo = np.where(lo0 < 0, lo0 * inv, lo0)  # dilution only shrinks BP/RP changes towards zero
        hi = np.where(hi0 > 0, hi0 * inv, hi0)
    outside = np.where(d_g > hi, d_g - hi, np.where(d_g < lo, d_g - lo, 0.0))
    outside[~np.isfinite(lo0)] = np.nan
    e_other = np.fmax(e_bp, e_rp)
    with np.errstate(invalid="ignore"):
        e_scaled = e_other * inv
    tested = np.isfinite(outside)
    noise = np.fmax(np.hypot(e_g, e_scaled), _robust(d_g - 0.5 * (d_bp + d_rp)))
    g_err_ratio = e_g / np.median(e_g)

    def z(d, e):  # a band's change in units of its own scatter (or the epoch's error, if larger)
        return np.abs(d) / np.fmax(e, _robust(d))

    g_at_fault = z(d_g, e_g) > np.fmax(np.nan_to_num(z(d_bp, e_bp)), np.nan_to_num(z(d_rp, e_rp)))
    verdict = np.full(len(d_g), "untested", dtype=object)
    with np.errstate(invalid="ignore"):
        verdict[tested & (nsigma * e_scaled < np.abs(d_g))] = "consistent"
        bad = tested & (np.abs(np.nan_to_num(outside)) > nsigma * noise)
    verdict[bad & g_at_fault] = "band_inconsistent"
    verdict[bad & ~g_at_fault] = "colour_fault"
    inflated = g_err_ratio > err_ratio
    # Seen by an independent measurement: BP or RP at this transit, or G at the other field of view's, changed
    # the same way by more than 3 of its own errors. Lenient on purpose: this rule has no physics behind it.
    with np.errstate(invalid="ignore"):
        corroborated = np.zeros(len(d_g), bool)
        for d, e in ((d_bp, e_bp), (d_rp, e_rp)):
            corroborated |= (np.abs(d) > 3 * e) & (np.sign(d) == np.sign(d_g))
    if partner is not None:
        p = np.asarray(partner)
        q = np.clip(p, 0, None)
        corroborated |= ((p >= 0) & ~inflated[q] & (np.abs(d_g[q]) > 3 * e_g[q])
                         & (np.sign(d_g[q]) == np.sign(d_g)))
    verdict[inflated & ~corroborated] = "inflated_error"
    return {"verdict": verdict, "d_g": d_g, "d_bp": d_bp, "d_rp": d_rp, "outside": outside, "noise": noise,
            "g_err_ratio": g_err_ratio, "r_lo": r_lo}


BLEND_COLS = ("phot_bp_n_obs", "phot_bp_n_blended_transits", "phot_rp_n_obs", "phot_rp_n_blended_transits")


def blend_fraction(source):
    """Largest of the BP and RP blended-transit fractions from gaia_source columns, or None if absent."""
    out = []
    for b in ("bp", "rp"):
        n, k = source.get(f"phot_{b}_n_obs"), source.get(f"phot_{b}_n_blended_transits")
        if n and k is not None and np.isfinite(float(k)):
            out.append(float(k) / float(n))
    return max(out) if out else None


def epoch_verdicts(cand, nsigma=4.0, err_ratio=5.0, w_range=W_RANGE):
    """Per G epoch: a verdict and the numbers behind it, as a dict of arrays aligned with the G light curve.

    Verdicts: "consistent"; "band_inconsistent" (G outside the BP-RP mixing interval, allowing for BP/RP
    dilution, by more than nsigma, G at fault); "colour_fault" (the same, BP or RP at fault); "inflated_error"
    (G error > err_ratio times the median, not corroborated by BP/RP or the partner transit); "untested" (no
    BP/RP in that transit, BP/RP too noisy or too diluted to have shown a change this size).
    """
    g = cand.band("G")
    if g is None:
        return None
    med = float(np.median(g.flux))
    d_g, e_g = 1 - g.flux / med, g.flux_err / med
    (d_bp, e_bp), (d_rp, e_rp) = _matched(cand.band("BP"), g.time), _matched(cand.band("RP"), g.time)
    r_lo = dilution_floor(d_g, e_g, d_bp, d_rp, blend_fraction(cand.source))
    ev = _assess(d_g, e_g, d_bp, e_bp, d_rp, e_rp, nsigma, err_ratio, w_range, r_lo, _partner(g.time))
    return {"time": g.time, **ev}


def verdict_counts(verdicts):
    """Per-source feature: how many G epochs got each verdict. Dropped epochs are counted, never silently lost:
    the rate of faults per source and per sky region is an instrument diagnostic in its own right."""
    v = np.asarray(verdicts, dtype=object)
    return {f"stage0_n_{name}": int((v == name).sum()) for name in VERDICTS}


def clean_long(table, nsigma=4.0, err_ratio=5.0, w_range=W_RANGE, blend=None):
    """Stage 0 on gaia.epoch's long light-curve table (one row per observation, G/BP/RP sharing transit_id).

    Returns (table without the faulty measurements, pandas frame of per-source verdict_counts). Matching is
    by transit_id, which is exact. blend maps source_id to its BP/RP blended-transit fraction (blend_fraction
    of the gaia_source row); a quiet source missing from it is left untested (dilution_floor). For the statistics downstream to stay valid, run this before every
    statistic that reads the light curves (dim and bright alike, so a mirrored null stays symmetric) and on
    injected light curves too, so completeness is measured after cleaning.
    """
    import pandas as pd
    import pyarrow as pa

    df = table.to_pandas() if hasattr(table, "to_pandas") else table.copy()
    band = df["band"].astype(str).to_numpy()
    flux, err, tid = (df[c].to_numpy() for c in ("flux", "err", "transit_id"))
    drop = np.zeros(len(df), bool)
    counts = []
    for sid, idx in df.groupby("source_id", sort=False).indices.items():
        rows = {b: idx[(band[idx] == b) & np.isfinite(flux[idx])] for b in ("G", "BP", "RP")}
        g = rows["G"]
        if len(g) == 0:
            counts.append({"source_id": sid, **verdict_counts([])})
            continue
        def changes(r):
            f, e = flux[r], err[r]
            med = float(np.median(f)) if len(f) else np.nan
            return 1 - f / med, e / med

        d_g, e_g = changes(g)
        cols = []
        for b in ("BP", "RP"):
            d, e = changes(rows[b])
            if len(d) == 0:
                cols += [np.full(len(g), np.nan)] * 2
                continue
            at = {t: i for i, t in enumerate(tid[rows[b]])}
            j = np.array([at.get(t, -1) for t in tid[g]])
            cols += [np.where(j >= 0, d[np.clip(j, 0, None)], np.nan), np.where(j >= 0, e[np.clip(j, 0, None)], np.nan)]
        r_lo = dilution_floor(d_g, e_g, cols[0], cols[2], (blend or {}).get(sid))
        v = _assess(d_g, e_g, *cols, nsigma, err_ratio, w_range, r_lo, _partner(df["t"].to_numpy()[g]))["verdict"]
        drop[g[np.isin(v, ARTEFACTS)]] = True
        faulty = set(tid[g][v == "colour_fault"])
        for b in ("BP", "RP"):
            drop[rows[b][np.isin(tid[rows[b]], list(faulty))]] = True
        counts.append({"source_id": sid, **verdict_counts(v)})
    keep = ~drop
    out = table.filter(pa.array(keep)) if isinstance(table, pa.Table) else df[keep]
    return out, pd.DataFrame(counts)


def band_consistency(cand, dips, **kw):
    """The epoch verdict at each detected dip's deepest point, with the band changes there."""
    ev = epoch_verdicts(cand, **kw)
    if ev is None:
        return []
    out = []
    for d in dips:
        i = int(np.argmin(np.abs(ev["time"] - d["t_min"])))
        row = {"t_min": d["t_min"], "verdict": str(ev["verdict"][i])}
        if abs(ev["time"][i] - d["t_min"]) > MATCH_DAYS:
            row["verdict"] = "untested"
        for k in ("d_g", "d_bp", "d_rp", "outside", "noise", "g_err_ratio", "r_lo"):
            row[k] = float(ev[k][i])
        out.append(row)
    return out


def clean(cand, **kw):
    """A copy of cand without the measurements Stage 0 marks as faults, and the G times dropped.

    G epochs in ARTEFACTS are dropped from G; BP and RP are dropped at colour_fault transits, where G is kept.
    This works like DPAC's own rejected-epoch flags: a measurement fault is removed before anything looks for
    dips, so a candidate's dips, and a control's label, are about the star.
    """
    ev = epoch_verdicts(cand, **kw)
    if ev is None:
        return cand, []
    drop = {"G": ev["time"][np.isin(ev["verdict"], ARTEFACTS)]}
    drop["BP"] = drop["RP"] = ev["time"][ev["verdict"] == "colour_fault"]
    if not any(len(v) for v in drop.values()):
        return cand, []
    lcs = []
    for lc in cand.lightcurves:
        bad = drop.get(lc.band, ())
        if not len(bad):
            lcs.append(lc)
            continue
        keep = np.min(np.abs(lc.time[:, None] - np.asarray(bad)[None, :]), axis=1) > MATCH_DAYS
        meta = {k: (v[keep] if np.shape(v) == np.shape(lc.time) else v) for k, v in lc.meta.items()}
        lcs.append(replace(lc, time=lc.time[keep], flux=lc.flux[keep], flux_err=lc.flux_err[keep], meta=meta))
    return replace(cand, lightcurves=lcs), [float(b) for b in drop["G"]]


def veto(cand, dips=None):
    """Reasons to drop a candidate before vetting: its strongest detected dip is a Stage 0 artefact.

    Usable as the `veto` hook of vetting_batch / make_decoy (an empty list passes).
    """
    if dips is None:
        from .batch import simple_dips
        dips = simple_dips(cand)
    if not dips:
        return []
    strongest = max(dips, key=lambda d: d.get("depth", 0))
    rows = band_consistency(cand, [strongest])
    return [rows[0]["verdict"]] if rows and rows[0]["verdict"] in ARTEFACTS else []
