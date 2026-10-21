"""Assemble the star table, search half A, replicate in half B (docs/population.md §3-§5)."""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import glm, scan

LMC = (80.894, -69.756, 8.0)
SMC = (13.187, -72.829, 4.0)
ECL_P = 0.01


def _angdist(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def ecliptic_latitude(ra, dec):
    eps = np.radians(23.4392911)
    r, d = np.radians(ra), np.radians(dec)
    return np.degrees(np.arcsin(np.sin(d) * np.cos(eps) - np.cos(d) * np.sin(eps) * np.sin(r)))


def assemble(out, aux_dir, astrometry):
    out = Path(out)
    stars = pd.concat([pq.read_table(p).to_pandas() for p in sorted(glob.glob(f"{out}/stars/*.parquet"))])
    expo = pd.concat([pq.read_table(p).to_pandas() for p in sorted(glob.glob(f"{out}/exposure/*.parquet"))])
    aux = pq.read_table(f"{aux_dir}/aux_gaia_source.parquet").to_pandas()
    bl = pq.read_table(f"{aux_dir}/aux_blend.parquet").to_pandas()
    ast = pq.read_table(astrometry).to_pandas().drop(columns=["b"])
    df = stars.merge(expo, on="source_id").merge(aux, on="source_id", how="left").merge(ast, on="source_id", how="left")
    fr = []
    for b in ("bp", "rp"):
        n, k = bl[f"phot_{b}_n_obs"].astype(float), bl[f"phot_{b}_n_blended_transits"].astype(float)
        fr.append(np.where(n > 0, k / n, np.nan))
    bl["blend_frac"] = np.fmax(*fr)
    df = df.merge(bl[["source_id", "blend_frac"]], on="source_id", how="left")
    snr = df.parallax / df.parallax_error
    with np.errstate(divide="ignore", invalid="ignore"):
        df["m_g"] = np.where(snr >= 5, df.phot_g_mean_mag + 5 * np.log10(df.parallax) - 10, np.nan)
        df["log_excess"] = np.log10(df.excess)
        share = (df.E_EB_A + df.E_EB_B) / (df.E_gen_A + df.E_gen_B)
        df["log_eb_share"] = np.log10(np.where(share > 0, share, 1e-12))
    df["abs_ecl_lat"] = np.abs(ecliptic_latitude(df.ra, df.dec))
    df["excluded"] = (_angdist(df.ra, df.dec, LMC[0], LMC[1]) < LMC[2]) | (_angdist(df.ra, df.dec, SMC[0], SMC[1]) < SMC[2])
    df["pixel"] = scan.cell_pixel(df.source_id)
    df["shell"] = scan.shell_of(df.r_med_photogeo)
    df.to_parquet(out / "stars.parquet", index=False)
    return df


STAR_COLS = ["source_id", "ra", "dec", "pixel", "shell", "E_gen_A", "E_gen_B", "E_EB_A", "E_EB_B", "excluded",
             *[c for c, _ in glm.BINS.values()]]


def load_stars(out, cols=STAR_COLS):
    s = pd.read_parquet(Path(out) / "stars.parquet", columns=list(dict.fromkeys(cols)))
    return s[~s.excluded].reset_index(drop=True)


def outcome(stars, ev, half, mask, cls="SOV"):
    """Events of a class per star in one half, after the half's time mask. Returns counts aligned with stars."""
    from .run import hour_bin, mask_pixel
    e = ev[(ev.cls == cls) & (ev.half == half)]
    e = e[~mask[hour_bin(e.t_deep.to_numpy()), mask_pixel(e.source_id.to_numpy())]]
    c = e.groupby("source_id").size()
    return stars.source_id.map(c).fillna(0).to_numpy(float)


def null_model(stars, y, e, test_ecl=True, ridge=1.0):
    codes, eb_edges = glm.covariate_codes(stars)
    terms = list(glm.MAIN)
    info = {"eb_edges": [float(x) for x in eb_edges]}
    if test_ecl:
        stat, df, p = glm.lrt(codes, y, e, terms, "ecl", ridge)
        info["ecl_lrt"] = {"stat": stat, "df": df, "p": p}
        if p < ECL_P:
            terms.append("ecl")
    m = glm.fit(codes, y, e, terms, ridge)
    lam = m["predict"](codes, e)
    info.update(terms=m["terms"], converged=m["converged"], n_cells=m["n_cells"],
                coef={k: float(v) for k, v in zip(m["labels"], m["beta"])})
    return lam, info, terms


def search(stars, y, e, win=None, n_null=scan.N_NULL, rng=None, test_ecl=True, terms=None, k_max=5):
    """Half-A search: GLM, cells, scan, max-null, greedy candidates."""
    if terms is None:
        lam, info, terms = null_model(stars, y, e, test_ecl)
    else:
        codes, _ = glm.covariate_codes(stars)
        m = glm.fit(codes, y, e, terms)
        lam, info = m["predict"](codes, e), {"terms": m["terms"]}
    win = win or scan.Windows()
    px, sh = stars.pixel.to_numpy(), stars.shell.to_numpy()
    obs, lc = scan.cell_sums(px, sh, y), scan.cell_sums(px, sh, lam)
    null = scan.null_max(win, lc, n=n_null, rng=rng)
    phi, phi_scale = scan.dispersion(win, obs, lc)
    cands, top = scan.candidates(win, obs, lc, null, phi=phi, k_max=k_max)
    cands_poisson, top_poisson = scan.candidates(win, obs, lc, null)
    info.update(phi_per_scale=phi_scale.round(3).tolist(), candidates_poisson_only=cands_poisson,
                max_llr_poisson=top_poisson,_lam=lam, n_events=float(y.sum()), expected=float(lam.sum()), max_llr=top,
                null_quantiles={q: float(np.quantile(null, q)) for q in (0.5, 0.9, 0.95, 0.99)})
    return cands, info, terms, win


def window_members(stars, win, cand):
    """Boolean mask of stars inside a candidate window."""
    pix = win.M[cand["disc"]].indices
    lo, hi = cand["shells"]
    return np.isin(stars.pixel.to_numpy(), pix) & (stars.shell.to_numpy() >= lo) & (stars.shell.to_numpy() < hi)


def refit_check(stars, lam, e, terms, win, n=10, rng=None):
    """Max LLR over windows in n star-level null realisations, with and without refitting the GLM (§4)."""
    rng = rng or np.random.default_rng()
    codes, _ = glm.covariate_codes(stars)
    px, sh = stars.pixel.to_numpy(), stars.shell.to_numpy()
    lc = scan.cell_sums(px, sh, lam)
    E0 = win.totals(lc)
    out = {"no_refit": [], "refit": []}
    for _ in range(n):
        y = rng.poisson(lam).astype(float)
        O = win.totals(scan.cell_sums(px, sh, y))
        out["no_refit"].append(float(scan.llr(O, E0).max()))
        m = glm.fit(codes, y, e, terms)
        E1 = win.totals(scan.cell_sums(px, sh, m["predict"](codes, e)))
        out["refit"].append(float(scan.llr(O, E1).max()))
    return out


def permutation_p(stars, y, members, n=9999, rng=None):
    """One-sided p for a fixed window's event count when events are permuted among stars within full covariate
    cells: per cell, the window's count is hypergeometric (cell size, cell events, cell members in the window)."""
    rng = rng or np.random.default_rng()
    codes, _ = glm.covariate_codes(stars)
    key = pd.MultiIndex.from_frame(codes[list(glm.MAIN) + ["ecl"]]).codes
    cell = np.zeros(len(stars), np.int64)
    for k in key:
        cell = cell * (int(k.max()) + 1) + k
    df = pd.DataFrame({"cell": cell, "y": y, "m": members.astype(int)})
    g = df.groupby("cell").agg(n=("y", "size"), k=("y", "sum"), w=("m", "sum"))
    g = g[(g.w > 0) & (g.k > 0)]
    obs = float(y[members].sum())
    sims = np.zeros(n)
    for row in g.itertuples():
        kk = int(min(row.k, row.n))
        sims += rng.hypergeometric(kk, int(row.n) - kk, int(row.w), n)
    return float((1 + np.sum(sims >= obs)) / (1 + n)), obs, float(sims.mean())
