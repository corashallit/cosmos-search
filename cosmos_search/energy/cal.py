"""Calibration of G5 on real stars with known answers injected, split into halves A and B.

    python -m cosmos_search.energy cal --ref ref/ --half A --out cal/      # choose thresholds on A
    python -m cosmos_search.energy cal --ref ref/ --half B --out cal/      # confirm on B (after the doc commit)

A star belongs to half A or B by (source_id // 3) % 2. The photosphere model that scores a half is fitted on the
other half's reference stars only, so no star is ever its own neighbour.

Per half:
- null: hosts as they are (G >= HOST_G_MIN from the general and mid samples): the false-excess rate. A few per cent
  of real stars have real excesses or blends, so this is an upper bound on the false rate.
- families: each energy.inject family put into the same hosts: excess recall per family.
- dimmers: energy.inject.dimmer_plan (shell / thin / none arrangements): false deficits (a deficit claimed while
  the injected emitter is in the claimed range) and deficit recall on "none".
- natural: DPAC's YSO class (real dust) and ECL class (dimmers without dust), unmodified.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import inject
from .budget import BANDS, T_SUB
from .photometry import design, sigma_var_from_gaia
from .photosphere import Locus
from .run import T_DEFICIT_MIN, evaluate

HOST_G_MIN = 10.0
FRESH = 2   # "half C": hosts from reference.SAMPLES' *_c slices, scored with a photosphere fitted on all of A and B
N_GENERAL, N_MID = 3000, 1500
N_DIMMER = 6000
INVISIBLE = 0.3


def half_of(df):
    return (np.asarray(df["source_id"], np.int64) // 3) % 2


def locus_for(ref_dir, h):
    """Photosphere model fitted on the half that is not h."""
    ref_dir = Path(ref_dir)
    samples = {}
    for s in ("general", "mid", "bright"):
        df = pd.read_parquet(ref_dir / f"{s}.parquet")
        samples[s] = df[half_of(df) != h]
    return Locus().fit(samples)


def hosts(ref_dir, h, rng, n_general=N_GENERAL, n_mid=N_MID):
    ref_dir = Path(ref_dir)
    parts = []
    for s, n in (("general", n_general), ("mid", n_mid)):
        if h == FRESH:
            df = pd.read_parquet(ref_dir / f"{s}_c.parquet")
            df = df[df.phot_g_mean_mag >= HOST_G_MIN]
        else:
            df = pd.read_parquet(ref_dir / f"{s}.parquet")
            df = df[(half_of(df) == h) & (df.phot_g_mean_mag >= HOST_G_MIN)]
        parts.append(df.sample(min(n, len(df)), random_state=int(rng.integers(2**31))).assign(sample=s))
    out = pd.concat(parts, ignore_index=True)
    out["sigma_var_mag"] = sigma_var_from_gaia(out)
    return out


def run_half(ref_dir, h, seed, log=print):
    rng = np.random.default_rng(seed)
    locus = locus_for(ref_dir, h)
    host = hosts(ref_dir, h, rng)
    des = design(host, locus.predict(host))
    rows = []
    null, _ = evaluate(host, locus)
    rows.append(null.assign(kind="null", family="none", sample=host["sample"].to_numpy(),
                            g=host.phot_g_mean_mag.to_numpy()))
    log(f"half {h}: null {len(null)}")
    for fam in inject.FAMILIES:
        comps = inject.draw(fam, len(host), rng)
        res, _ = evaluate(inject.apply(host, des, comps), locus)
        res = res.assign(kind="family", family=fam, sample=host["sample"].to_numpy(), g=host.phot_g_mean_mag.to_numpy(),
                         inj_T=[c[0][0] for c in comps], inj_f=[c[0][1] for c in comps],
                         inj_max_excess=injected_max_excess(des, comps))
        rows.append(res)
        log(f"half {h}: {fam} excess {res.g5_excess.mean():.3f}")
    fam_tab = pd.concat(rows, ignore_index=True)

    dh = hosts(ref_dir, h, rng, n_general=N_DIMMER * 2 // 3, n_mid=N_DIMMER // 3)
    ddes = design(dh, locus.predict(dh))
    plan, comps = inject.dimmer_plan(ddes, rng)
    dres, _ = evaluate(inject.apply(dh, ddes, comps), locus, delta=plan["delta"].to_numpy())
    dim = pd.concat([dres, plan.add_prefix("inj_")], axis=1).assign(sample=dh["sample"].to_numpy(),
                                                                    g=dh.phot_g_mean_mag.to_numpy())
    log(f"half {h}: dimmers {len(dim)}")

    nat = []
    for s in ("yso", "ecl"):
        df = pd.read_parquet(Path(ref_dir) / f"{s}.parquet")
        df = df[half_of(df) == h] if h != FRESH else df.iloc[:0]
        res, _ = evaluate(df, locus)
        nat.append(res.assign(kind="natural", family=s, g=df.phot_g_mean_mag.to_numpy()))
    nat = pd.concat([n for n in nat if len(n)] or [pd.DataFrame(columns=["family", "g5_z_excess"])], ignore_index=True)
    return fam_tab, dim, nat


def injected_max_excess(des, comps):
    """Largest injected excess over the photosphere in any band (photospheres): below ~0.3 the injection is
    invisible to 2MASS and WISE whatever the family says."""
    phot = np.stack([des["a0"].to_numpy(float) * des[f"P_{b}"].to_numpy(float) for b in BANDS], -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmax(inject.added_flux(des, comps) / phot, axis=1)


def false_deficit(dim, geo, z_col=None):
    """Deficit claimed for an injection whose emitter sits in the range the claim covers."""
    flag = dim[f"g5_deficit_{geo}"] if z_col is None else z_col
    in_range = (dim.inj_T >= T_DEFICIT_MIN) & (dim.inj_T <= T_SUB)
    covers = in_range & (dim.inj_arrangement.isin(["shell"] if geo == "shell" else ["shell", "thin"]))
    return flag & covers, covers


def summary(fam_tab, dim, nat, z_excess, z_deficit):
    from cosmos_search.inject.recovery import wilson

    def rate(mask, n_mask=None):
        k, n = int(mask.sum()), int(len(mask) if n_mask is None else n_mask.sum())
        lo, hi = wilson(k, n) if n else (np.nan, np.nan)
        return {"k": k, "n": n, "rate": k / n if n else np.nan, "interval_95": [float(lo), float(hi)]}

    out = {"z_excess": z_excess, "z_deficit": z_deficit}
    ex = lambda t: t.g5_z_excess >= z_excess
    null = fam_tab[fam_tab.kind == "null"]
    out["false_excess_null"] = rate(ex(null))
    out["false_excess_null_by_sample"] = {s: rate(ex(g)) for s, g in null.groupby("sample")}
    out["excess_recall"] = {f: rate(ex(g)) for f, g in fam_tab[fam_tab.kind == "family"].groupby("family")}
    out["natural_excess"] = {f: rate(ex(g)) for f, g in nat.groupby("family")}
    fams = fam_tab[fam_tab.kind == "family"]
    if "inj_max_excess" in fams:
        # B5' (docs §8): injections no band can see (< 0.3 photospheres anywhere), whatever their family.
        inv = fams[fams.inj_max_excess < INVISIBLE]
        out["invisible_flagged"] = rate(ex(inv))
        out["invisible_flagged_cold"] = rate(ex(inv[inv.family == "cold"]))
    for geo in ("shell", "thin"):
        zmin = dim[f"g5_{geo}_z_min"].fillna(0)
        flag = (zmin >= z_deficit) & ~ex(dim)
        bad, covers = false_deficit(dim, geo, flag)
        out[f"false_deficit_{geo}"] = {"k": int(bad.sum()), "n": int(covers.sum()),
                                       "interval_95": [float(x) for x in wilson(int(bad.sum()), int(covers.sum()))]}
        none = dim.inj_arrangement == "none"
        out[f"deficit_{geo}_recall_none"] = rate(flag[none])
        out[f"deficit_{geo}_recall_none_by_delta"] = {
            f"{lo:g}-{hi:g}": rate(flag[none & dim.inj_delta.between(lo, hi)])
            for lo, hi in ((1e-3, 1e-2), (1e-2, 3e-2), (3e-2, 0.1), (0.1, 0.3))}
    # Claims of "excluded above T" that cut through the injected emitter.
    for geo, arr in (("shell", ["shell"]), ("thin", ["shell", "thin"])):
        col = dim[f"g5_{geo}_excluded_above_K"]
        sel = dim.inj_arrangement.isin(arr) & (dim.inj_T <= T_SUB)
        out[f"excluded_range_cuts_emitter_{geo}"] = {"k": int((sel & (dim.inj_T >= col)).sum()), "n": int(sel.sum())}
    return out


def choose_thresholds(fam_tab, dim, target_false_excess=0.01, target_false_deficit=0.005,
                      grid=np.arange(2.0, 10.01, 0.5)):
    """Half A: the smallest z_excess with the null rate <= target, then the smallest z_deficit with the false
    deficit rate (shell and thin, pooled) <= target at that z_excess."""
    null = fam_tab[fam_tab.kind == "null"].g5_z_excess.fillna(0)
    z_ex = next((z for z in grid if (null >= z).mean() <= target_false_excess), grid[-1])
    exd = dim.g5_z_excess.fillna(0) >= z_ex
    z_def = grid[-1]
    for z in grid:
        k = n = 0
        for geo in ("shell", "thin"):
            flag = (dim[f"g5_{geo}_z_min"].fillna(0) >= z) & ~exd
            bad, covers = false_deficit(dim, geo, flag)
            k, n = k + int(bad.sum()), n + int(covers.sum())
        if n and k / n <= target_false_deficit:
            z_def = z
            break
    return float(z_ex), float(z_def)
