"""From G3 selections to a vetting queue (milestone G6, first version, pre-registered for the DR3 rehearsal).

    python -m cosmos_search.dayone.rank --q g3/q.parquet --features stream/features --lightcurves stream/lightcurves \\
        --blend aux_blend.parquet --out rank/ [--level 0.1] [--n-vet 30] [--n-audit 6] [--n-controls 8]

Steps, and why:

1. **Selections.** Sources with q ≤ level and W > 0 in `fov` (dip confirmed by the other field of view) or
   in `single` (deepest point alone). `fov` is the invariance-tested channel and ranks first.
2. **Context** for the selections in three TAP upload queries: astrometry and photometry from gaia_source,
   DPAC's best class, and DPAC's eclipsing-binary model.
3. **Dip-level explanation.** DPAC labels the star, not the dip (docs/vetting.md §11). A selection counts as
   explained only when the dip is: its deepest epoch falls inside the primary or secondary eclipse of DPAC's
   eclipsing-binary model (tier 3). A star whose DPAC class could make dips but whose dip is not checked
   against a model is tier 2. Everything else is tier 1. Tier 3 is not vetted; a random handful of them are
   the batch's known-answer controls, labelled at the dip.
4. **Per-event physics** from the Stage-0-cleaned light curve (vetting.physics). Episodes are runs of
   ≤ −3σ epochs separated by > 1 d. An episode seen in both fields of view lasted ≥ 106.5 min; one spanning
   epochs further apart lasted at least that long. The shortest period any orbit allows for that duration
   (e ≤ 0.9) is compared with the spacing of episodes: recurrence faster than a third of that floor is
   `tension`. Dip colour R = (ΔF/F)_RP / (ΔF/F)_BP at the deepest epoch; `grey` when both bands dip ≥ 3σ
   and 0.75 ≤ R ≤ 1.33.
5. **Priority** within a tier: the count of flags (fov channel, grey, tension, G < 13 for cheap follow-up),
   then W. Flags are weighted equally until vetting outcomes and injection recovery can fit weights
   (docs/arms/gaia-dimmers.md, Ranking).
6. **Queue**: the top n_vet of tiers 1–2, an audit sample of n_audit drawn below the cut with known inclusion
   probabilities (stats.audit), and n_controls tier-3 controls.

Ranking v1 (for the D2 freeze, docs/arms/residual.md) adds a second pass once external light curves exist:

    python -m cosmos_search.dayone.rank prefetch --ranked rank/ranked.parquet --n 2000 --out rank/prefetch.parquet
    python -m cosmos_search.dayone.external fetch --queue rank/prefetch.parquet --out ext/external.parquet
    python -m cosmos_search.residual --queue rank/prefetch.parquet --ranked rank/ranked.parquet \
        --lightcurves stream/lightcurves --external ext/external.parquet --out rank/residual.parquet
    python -m cosmos_search.dayone.rank requeue --ranked rank/ranked.parquet --residual rank/residual.parquet \
        --out rank/ [--rule residual|filter] [--flag-eb-residual]

7. **Demotion.** A star whose selected dip the eclipse baseline explains (`residual.model`: `demote`, or with
   `--rule filter` the EB filter's `explained` as in PR #45) moves to tier 3 as a known-answer control
   (`explained_by`). It is not deleted: the audit draw (step 6) runs over every star below the vetting cut,
   demoted ones included, so the cost of demotion is measured on DR4 itself.
8. **eb_residual.** An eclipsing binary with an unexplained dip left over. With `--flag-eb-residual` (adopted only if
   docs/arms/residual.md's D2 passes) it counts as a priority flag like fov, grey, tension and bright.
9. **Active stars** (red-team review R1). On real DR3 stars with ZTF, the recurrence flag fires on 79% of testable
   RS CVn and 67% of testable SOLAR_LIKE stars: spotted stars' low states are aperiodic dips. Ranking v0 put those
   classes in tier 1. In v1 they and AGN are tier 2 (ACTIVE_CLASSES), the recurrence flag (on residuals where the
   residual search ran) is a priority flag, not a route around the vetter, and the queue carries n_active_controls
   active-class stars as controls labelled `rotation_or_spots` (AGN: `extragalactic`), so the shortlist's false-flag
   rate is measured on the class that will crowd it.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from astropy.table import Table

from cosmos_search.gaia.epoch import gaia_time_to_btjd
from cosmos_search.stats import audit
from cosmos_search.vetting import physics, stage0

G_ZP = 25.6874
FOV_PAIR_DAYS = 106.5 / 1440
EPISODE_GAP = 1.0
LOW_Z = -3.0
TENSION_FACTOR = 3.0

# DPAC classes whose stars dip for known reasons. Without a model check at the dip epoch they only lower the tier.
DIP_CLASSES = {"ECL", "ELL", "EP", "YSO", "RCB", "LPV", "CEP", "RR", "DSCT|GDOR|SXPHE", "BE|GCAS|SDOR|WR", "SYST",
               "CV", "AGN", "BCEP", "SPB", "ACYG"}
SOURCE_COLS = ("ra", "dec", "ra_error", "dec_error", "ref_epoch", "parallax", "parallax_error", "pmra", "pmra_error",
               "pmdec", "pmdec_error", "ra_dec_corr", "ra_parallax_corr", "ra_pmra_corr", "ra_pmdec_corr",
               "dec_parallax_corr", "dec_pmra_corr", "dec_pmdec_corr", "parallax_pmra_corr", "parallax_pmdec_corr",
               "pmra_pmdec_corr", "astrometric_params_solved", "phot_g_mean_mag", "phot_bp_mean_mag",
               "phot_rp_mean_mag", "bp_rp", "ruwe", "ipd_frac_multi_peak", "phot_bp_rp_excess_factor", "l", "b",
               "ecl_lat", *stage0.BLEND_COLS)
EB_COLS = ("reference_time", "frequency", "derived_primary_ecl_phase", "derived_primary_ecl_duration",
           "derived_primary_ecl_depth", "derived_secondary_ecl_phase", "derived_secondary_ecl_duration",
           "derived_secondary_ecl_depth", "global_ranking")


def selections(q, level=0.1):
    q = pd.read_parquet(q) if not isinstance(q, pd.DataFrame) else q
    fov = (q["q_fov"] <= level) & (q["w_fov"] > 0)
    single = (q["q_single"] <= level) & (q["w_single"] > 0)
    sel = q[fov | single].copy()
    sel["channel"] = np.where(fov[fov | single], "fov", "single")
    sel["w"] = np.where(sel["channel"] == "fov", sel["w_fov"], sel["w_single"])
    sel["q"] = np.where(sel["channel"] == "fov", sel["q_fov"], sel["q_single"])
    return sel[["source_id", "channel", "w", "q", "g_mag"]].reset_index(drop=True)


def context(ids, tap):
    """gaia_source, best DPAC class and DPAC EB model for ids, as one DataFrame (TAP uploads)."""
    up = {"ids": Table({"source_id": np.asarray(ids, dtype=np.int64)})}
    src = tap.query("SELECT u.source_id, " + ", ".join(f"s.{c}" for c in SOURCE_COLS) +
                    " FROM TAP_UPLOAD.ids AS u JOIN gaiadr3.gaia_source AS s ON s.source_id = u.source_id",
                    uploads=up).to_pandas()
    cls = tap.query("SELECT u.source_id, c.best_class_name, c.best_class_score FROM TAP_UPLOAD.ids AS u "
                    "JOIN gaiadr3.vari_classifier_result AS c ON c.source_id = u.source_id", uploads=up).to_pandas()
    eb = tap.query("SELECT u.source_id, " + ", ".join(f"e.{c}" for c in EB_COLS) +
                   " FROM TAP_UPLOAD.ids AS u JOIN gaiadr3.vari_eclipsing_binary AS e ON e.source_id = u.source_id",
                   uploads=up).to_pandas()
    for t in (cls, eb):
        if "best_class_name" in t:
            t["best_class_name"] = t["best_class_name"].astype(str)
    return src.merge(cls, on="source_id", how="left").merge(eb, on="source_id", how="left")


def in_eclipse(t, ctx, depth_mag=None, core=0.25, depth_factor=1.5, depth_slack=0.05):
    """True where time t (BTJD) falls in the core of DPAC's primary or secondary eclipse, |Δφ| ≤ core × duration
    (the derived duration is the full width, so ±duration/2 would explain a random epoch ~50% of the time),
    and, when depth_mag is given, the dip is no deeper than depth_factor × the model's eclipse + depth_slack."""
    t = np.asarray(t, float)
    ref = gaia_time_to_btjd(ctx["reference_time"].to_numpy(dtype=float))
    phase = np.mod((t - ref) * ctx["frequency"].to_numpy(dtype=float), 1.0)
    out = np.zeros(len(t), bool)
    for which in ("primary", "secondary"):
        ph = ctx[f"derived_{which}_ecl_phase"].to_numpy(dtype=float)
        half = core * ctx[f"derived_{which}_ecl_duration"].to_numpy(dtype=float)
        d = np.abs(np.mod(phase - ph + 0.5, 1.0) - 0.5)
        ok = np.isfinite(d) & np.isfinite(half) & (d <= np.maximum(half, 0.01))
        if depth_mag is not None:
            model = ctx[f"derived_{which}_ecl_depth"].to_numpy(dtype=float)
            ok &= np.asarray(depth_mag, float) <= depth_factor * np.nan_to_num(model, nan=0.0) + depth_slack
        out |= ok
    return out


def read_lightcurves(lc_dir, ids):
    """The stream's kept light curves for ids (raw: injected, before Stage 0)."""
    dataset = ds.dataset(str(lc_dir), format="parquet")
    return dataset.to_table(filter=ds.field("source_id").isin(pa.array(np.asarray(ids, dtype=np.int64))))


def _blend_map(ctx):
    rows = ctx[["source_id", *stage0.BLEND_COLS]].to_dict("records")
    out = {}
    for r in rows:
        f = stage0.blend_fraction({c: (None if pd.isna(r[c]) else r[c]) for c in stage0.BLEND_COLS})
        if f is not None and np.isfinite(f):
            out[int(r["source_id"])] = f
    return out


def episodes(t, z, fov):
    """Runs of low epochs (z ≤ LOW_Z) separated by > EPISODE_GAP days: (centre, duration lower bound, n)."""
    order = np.argsort(t)
    t, z, fov = t[order], z[order], fov[order]
    low = np.flatnonzero(np.isfinite(z) & (z <= LOW_Z))
    out = []
    if not len(low):
        return out
    start = 0
    for i in range(1, len(low) + 1):
        if i == len(low) or t[low[i]] - t[low[i - 1]] > EPISODE_GAP:
            idx = low[start:i]
            span = float(t[idx].max() - t[idx].min())
            both = len(set(fov[idx].tolist())) > 1
            out.append((float(t[idx].mean()), max(span, FOV_PAIR_DAYS if both else 0.0), len(idx)))
            start = i
    return out


def lc_physics(lc, ctx):
    """Per source: episodes, duration bound, orbital period floor, recurrence tension, dip colour, grey."""
    cleaned, _ = stage0.clean_long(lc, blend=_blend_map(ctx))
    df = cleaned.to_pandas()
    df["band"] = df["band"].astype(str)
    params = ctx.set_index("source_id")
    rows = []
    for sid, g in df.groupby("source_id"):
        G = g[(g.band == "G") & ((g["flags"] & 1) == 0) & (g.flux > 0)]
        if len(G) < 5:
            continue
        med = np.median(G.flux)
        sig = 1.4826 * np.median(np.abs(G.flux - med))
        z = (G.flux.to_numpy() - med) / sig if sig > 0 else np.full(len(G), np.nan)
        eps = episodes(G.t.to_numpy(), z, G.fov.to_numpy())
        src = params.loc[sid] if sid in params.index else None
        sp = physics.stellar_params({k: (None if src is None or pd.isna(src.get(k)) else float(src.get(k)))
                                     for k in ("phot_g_mean_mag", "bp_rp", "parallax", "parallax_error")})
        dur = max((e[1] for e in eps), default=0.0)
        floor = physics.min_period_days(dur, *sp) if sp and dur > 0 else np.nan
        spacing = float(np.min(np.diff(sorted(e[0] for e in eps)))) if len(eps) >= 2 else np.nan
        k = int(np.nanargmin(z)) if np.isfinite(z).any() else None
        colour = {}
        if k is not None:
            tid = G.transit_id.iloc[k]
            for band in ("BP", "RP"):
                b = g[(g.band == band) & ((g["flags"] & 3) == 0) & (g.flux > 0)]
                if len(b) >= 5:
                    bm = np.median(b.flux)
                    bs = 1.4826 * np.median(np.abs(b.flux - bm))
                    at = b[b.transit_id == tid]
                    if len(at):
                        colour[band] = ((at.flux.iloc[0] / bm - 1), (at.flux.iloc[0] - bm) / bs if bs > 0 else np.nan)
        r = colour["RP"][0] / colour["BP"][0] if {"BP", "RP"} <= set(colour) and colour["BP"][0] != 0 else np.nan
        grey = bool({"BP", "RP"} <= set(colour) and colour["BP"][1] <= -3 and colour["RP"][1] <= -3
                    and 0.75 <= r <= 1.33)
        rows.append({"source_id": sid, "n_episodes": len(eps), "duration_lo_d": dur, "period_floor_d": floor,
                     "min_spacing_d": spacing, "tension": bool(np.isfinite(floor) and np.isfinite(spacing)
                                                               and spacing < floor / TENSION_FACTOR),
                     "dip_colour_R": r, "grey": grey, "t_deepest": float(G.t.iloc[k]) if k is not None else np.nan,
                     "depth_frac": float(G.flux.iloc[k] / med - 1) if k is not None else np.nan,
                     "r_star": sp[0] if sp else np.nan, "m_star": sp[1] if sp else np.nan})
    return pd.DataFrame(rows)


def rank(sel, ctx, phys):
    df = sel.merge(ctx, on="source_id", how="left").merge(phys, on="source_id", how="left")
    df["eb_dip"] = False
    has_eb = df["frequency"].notna() & df["t_deepest"].notna()
    depth = -2.5 * np.log10(np.clip(1 + df.loc[has_eb, "depth_frac"].to_numpy(dtype=float), 1e-3, None))
    df.loc[has_eb, "eb_dip"] = in_eclipse(df.loc[has_eb, "t_deepest"], df.loc[has_eb], depth)
    dip_class = df["best_class_name"].isin(DIP_CLASSES)
    df["tier"] = np.where(df["eb_dip"], 3, np.where(dip_class, 2, 1))
    df["bright"] = df["phot_g_mean_mag"] < 13
    flags = ["fov", "grey", "tension", "bright"]
    df["fov"] = df["channel"] == "fov"
    df[flags] = df[flags].fillna(False).astype(bool)
    df["n_flags"] = df[flags].sum(axis=1)
    return df.sort_values(["tier", "n_flags", "w"], ascending=[True, False, False]).reset_index(drop=True)


def queue(ranked, n_vet, n_audit, n_controls, rng):
    """Top n_vet of tiers 1-2, an audit draw below them with known inclusion probabilities, and tier-3 controls."""
    vettable = ranked[ranked["tier"] < 3]
    top = vettable.head(n_vet).assign(role="ranked", inclusion=1.0)
    rest = vettable.iloc[n_vet:]
    parts = [top]
    if len(rest) and n_audit:
        pi = audit.audit_inclusion(rest["w"].to_numpy(), n_audit)
        take = audit.poisson_sample(rng, pi)
        parts.append(rest[take].assign(role="audit", inclusion=pi[take]))
    ctrl = ranked[ranked["tier"] == 3]
    if len(ctrl) and n_controls:
        pick = rng.choice(len(ctrl), min(n_controls, len(ctrl)), replace=False)
        parts.append(ctrl.iloc[np.sort(pick)].assign(role="control", inclusion=n_controls / len(ctrl)))
    return pd.concat(parts, ignore_index=True)


def prefetch(ranked, n):
    """The top n of tiers 1-2: the stars that get external light curves and the residual search before requeue."""
    return ranked[ranked["tier"] < 3].head(n).reset_index(drop=True)


FLAGS_V0 = ("fov", "grey", "tension", "bright")
# Classes whose dips are activity, not occultation (review R1): RS CVn, spotted (SOLAR_LIKE), short-timescale, AGN.
ACTIVE_CLASSES = {"RS": "rotation_or_spots", "SOLAR_LIKE": "rotation_or_spots", "S": "rotation_or_spots",
                  "AGN": "extragalactic"}


def apply_residual(ranked, verdicts, rule="residual", flag_eb_residual=False):
    """Ranking v1: demote the stars the eclipse baseline explains to tier 3 and, optionally, count eb_residual as a
    priority flag. verdicts: `python -m cosmos_search.residual` output (source_id, demote, eb_explained,
    eb_residual, ...). Stars without a verdict keep their tier."""
    cols = ["source_id", "demote", "eb_explained", "eb_residual", "baseline", "period", "target_residual",
            "n_residual_events_elsewhere", "residual_aperiodic"]
    v = verdicts[[c for c in cols if c in verdicts]].rename(columns={"period": "baseline_period"})
    df = ranked.drop(columns=[c for c in v.columns if c != "source_id" and c in ranked], errors="ignore") \
        .merge(v, on="source_id", how="left")
    hit = df["demote" if rule == "residual" else "eb_explained"].fillna(False).astype(bool) & (df["tier"] < 3)
    df["explained_by"] = np.where(df["tier"] == 3, "dpac_model", None)
    df.loc[hit, "explained_by"] = "residual" if rule == "residual" else "ebfilter"
    df.loc[hit, "tier"] = 3
    df["eb_residual"] = df["eb_residual"].fillna(False).astype(bool)
    if "best_class_name" in df:
        df.loc[(df["tier"] == 1) & df["best_class_name"].isin(list(ACTIVE_CLASSES)), "tier"] = 2
    df["aperiodic"] = df["residual_aperiodic"].fillna(False).astype(bool) if "residual_aperiodic" in df else False
    flags = list(FLAGS_V0) + ["aperiodic"] + (["eb_residual"] if flag_eb_residual else [])
    df["n_flags"] = df[flags].fillna(False).astype(bool).sum(axis=1)
    return df.sort_values(["tier", "n_flags", "w"], ascending=[True, False, False]).reset_index(drop=True)


def queue_v1(ranked, n_vet, n_audit, n_controls, rng, n_active_controls=0):
    """As `queue`, but the audit draw covers the stars demoted by the residual search as well as the rest of tiers
    1-2, so what demotion removes is sampled with known probability. Controls come from all of tier 3, plus
    n_active_controls stars of ACTIVE_CLASSES below the vetting cut, labelled by star (activity, or AGN)."""
    vettable = ranked[ranked["tier"] < 3]
    top = vettable.head(n_vet).assign(role="ranked", inclusion=1.0)
    pool = pd.concat([vettable.iloc[n_vet:], ranked[ranked["explained_by"].isin(["residual", "ebfilter"])]])
    parts = [top]
    if len(pool) and n_audit:
        pi = audit.audit_inclusion(pool["w"].to_numpy(), min(n_audit, len(pool)))
        take = audit.poisson_sample(rng, pi)
        parts.append(pool[take].assign(role="audit", inclusion=pi[take]))
    taken = set(pd.concat(parts)["source_id"])
    ctrl = ranked[(ranked["tier"] == 3) & ~ranked["source_id"].isin(taken)]
    if len(ctrl) and n_controls:
        pick = rng.choice(len(ctrl), min(n_controls, len(ctrl)), replace=False)
        parts.append(ctrl.iloc[np.sort(pick)].assign(role="control", inclusion=n_controls / len(ctrl),
                                                     control_label="eclipsing_binary"))
    if n_active_controls and "best_class_name" in ranked:
        taken = set(pd.concat(parts)["source_id"])
        act = ranked[ranked["best_class_name"].isin(list(ACTIVE_CLASSES)) & ~ranked["source_id"].isin(taken)]
        if len(act):
            pick = rng.choice(len(act), min(n_active_controls, len(act)), replace=False)
            a = act.iloc[np.sort(pick)]
            parts.append(a.assign(role="control", inclusion=n_active_controls / len(act),
                                  control_label=a["best_class_name"].map(ACTIVE_CLASSES)))
    return pd.concat(parts, ignore_index=True)


def _requeue(argv):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.dayone.rank requeue")
    ap.add_argument("--ranked", required=True)
    ap.add_argument("--residual", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", choices=("residual", "filter"), default="residual")
    ap.add_argument("--flag-eb-residual", action="store_true")
    ap.add_argument("--n-vet", type=int, default=30)
    ap.add_argument("--n-audit", type=int, default=6)
    ap.add_argument("--n-controls", type=int, default=8)
    ap.add_argument("--n-active-controls", type=int, default=8)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ranked = apply_residual(pd.read_parquet(a.ranked), pd.read_parquet(a.residual), a.rule, a.flag_eb_residual)
    ranked.to_parquet(out / "ranked_v1.parquet", index=False)
    seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(8), "little")
    q = queue_v1(ranked, a.n_vet, a.n_audit, a.n_controls, np.random.default_rng(seed), a.n_active_controls)
    q.to_parquet(out / "queue_v1.parquet", index=False)
    summary = {"by_tier": ranked["tier"].value_counts().sort_index().to_dict(),
               "explained_by": ranked["explained_by"].value_counts().to_dict(),
               "eb_residual": int(ranked["eb_residual"].sum()), "aperiodic": int(ranked["aperiodic"].sum()),
               "rule": a.rule,
               "flag_eb_residual": a.flag_eb_residual, "queue": q["role"].value_counts().to_dict(), "seed": seed}
    (out / "summary_v1.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps(summary, indent=2, default=int))
    return 0


def _prefetch(argv):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.dayone.rank prefetch")
    ap.add_argument("--ranked", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    p = prefetch(pd.read_parquet(a.ranked), a.n)
    p.to_parquet(a.out, index=False)
    print(f"{len(p)} stars -> {a.out}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("requeue", "prefetch"):
        return (_requeue if argv[0] == "requeue" else _prefetch)(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.dayone.rank")
    ap.add_argument("--q", required=True)
    ap.add_argument("--lightcurves", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=float, default=0.1)
    ap.add_argument("--n-vet", type=int, default=30)
    ap.add_argument("--n-audit", type=int, default=6)
    ap.add_argument("--n-controls", type=int, default=8)
    # Drawn from urandom at run time so the audit sample cannot be predicted; pass the recorded seed to replay.
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    from cosmos_search.gaia.archive import tap

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = selections(a.q, a.level)
    ctx = context(sel["source_id"], tap())
    phys = lc_physics(read_lightcurves(a.lightcurves, sel["source_id"]), ctx)
    ranked = rank(sel, ctx, phys)
    ranked.to_parquet(out / "ranked.parquet", index=False)
    seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(8), "little")
    q = queue(ranked, a.n_vet, a.n_audit, a.n_controls, np.random.default_rng(seed))
    q.to_parquet(out / "queue.parquet", index=False)
    summary = {"selected": len(sel), "by_channel": sel["channel"].value_counts().to_dict(),
               "by_tier": ranked["tier"].value_counts().sort_index().to_dict(),
               "flags": {f: int(ranked[f].sum()) for f in ("fov", "grey", "tension", "bright")},
               "queue": q["role"].value_counts().to_dict(), "seed": seed}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps(summary, indent=2, default=int))
    return 0


if __name__ == "__main__":
    sys.exit(main())
