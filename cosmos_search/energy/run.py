"""G5 for a table of stars: photosphere, fit, and the per-star summary the packets and the ranking read.

    from cosmos_search.energy import run
    locus = run.load_locus(ref_dir)                  # reference.py samples -> photosphere.Locus
    g5 = run.evaluate(raw, locus, delta=delta_lo)    # one row per star

Columns (all prefixed g5_):
- n_bands: bands that entered the fit (K, W1-W4).
- z_excess: how strongly a photosphere-plus-emitter beats the photosphere alone, maximised over T and beta.
  excess = z_excess >= Z_EXCESS.
- t_best, f_best: the emitter at that maximum (meaningful only with an excess).
- f_ul_<T>K: the largest fraction of the star's light the photometry allows to come out at T (150, 300, 1000 K),
  whatever the dimming. A property of the star alone.
- With delta (the time-averaged dimming's lower bound, dimming.mean_deficit):
  shell_excluded_above_K / thin_excluded_above_K: the absorbed light cannot be coming out at any temperature from
  this up to 1500 K in that arrangement (nan: no such range);
  deficit_shell: shell excluded over the whole of [T_DEFICIT_MIN, 1500 K] and no excess. "Whatever is dimming the
  star is not an envelope or swarm of warm absorbers around it."
  deficit_thin: the same for the least absorbing arrangement, which only very deep, frequent and hot dimming can
  reach.

Thresholds are pre-registered in docs/arms/energy-budget.md before the blind run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import budget
from .photometry import design, sigma_var_from_gaia
from .photosphere import Locus

Z_EXCESS = 5.0        # pre-registered (docs/arms/energy-budget.md §5)
Z_DEFICIT = 3.0
T_DEFICIT_MIN = 150.0
T_REPORT = (150.0, 300.0, 1000.0)
CHUNK = 400


def load_locus(ref_dir, samples=("general", "mid", "bright")):
    ref_dir = Path(ref_dir)
    return Locus().fit({s: pd.read_parquet(ref_dir / f"{s}.parquet") for s in samples})


def evaluate(raw, locus, delta=None, chunk=CHUNK):
    raw = raw.reset_index(drop=True)
    if "sigma_var_mag" not in raw:
        raw = raw.assign(sigma_var_mag=sigma_var_from_gaia(raw))
    pred = locus.predict(raw)
    des = design(raw, pred)
    delta = None if delta is None else np.asarray(delta, float)
    parts = []
    for lo in range(0, len(des), chunk):
        sl = slice(lo, lo + chunk)
        dl = None if delta is None else np.nan_to_num(delta[sl], nan=0.0)
        parts.append(summarise(budget.fit(des.iloc[sl], dl), dl))
    out = pd.concat(parts, ignore_index=True)
    if "source_id" in raw:
        out.insert(0, "source_id", raw["source_id"].to_numpy())
    return out, des


def summarise(res, delta=None):
    out = {"g5_n_bands": res["n_bands"], "g5_z_excess": res["z_excess"], "g5_t_best": res["t_best"],
           "g5_f_best": res["f_best"]}
    for T in T_REPORT:
        out[f"g5_f_ul_{int(T)}K"] = budget.f_ul_at(res["f_ul"], T)
    out["g5_excess"] = res["z_excess"] >= Z_EXCESS
    if delta is not None:
        sel = (budget.T_GRID >= T_DEFICIT_MIN) & (budget.T_GRID <= budget.T_SUB)
        for geo in ("shell", "thin"):
            z = res[f"z_{geo}"]
            out[f"g5_{geo}_excluded_above_K"] = budget.excluded_above(z, z_thr=Z_DEFICIT)
            out[f"g5_{geo}_z_min"] = z[:, sel].min(axis=1)
            out[f"g5_deficit_{geo}"] = (out[f"g5_{geo}_z_min"] >= Z_DEFICIT) & ~out["g5_excess"]
        out["g5_delta_lo"] = delta
    return pd.DataFrame(out)
