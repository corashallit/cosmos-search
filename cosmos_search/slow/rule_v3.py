"""Rule slow-v3: slow-v2 with the dimming episode read in time order and changes shared with neighbours set aside
(docs/arms/slow-changes.md, "Fades in time order").

A star is flagged as fading when
- its time-ordered episode score s_fade (`detect`: a dip, a fade to the last visit, or a drop after a bright start)
  passes the threshold of its magnitude range and its change date is not shared with other stars nearby
  (`sky.shared_changes`, level-6 HEALPix pixels), or
- its trend slope_z is below minus the trend threshold, or
- its repeated-dimming score s_rep passes the period-search threshold (unchanged from slow-v2).

A second list, `early`, holds stars that were faint at the first visit and normal later (rise_kind "jump", s_rise
above the episode threshold, change date not shared): either a brightening or a dimming already under way when
Gaia started, which Gaia alone cannot tell apart. Observations from before 2014 can.

Thresholds are the 0.1% points of the mirror statistics among quiet stars of both halves (s_rise with shared dates
set aside, slope_z, s_rep_bright). Files in prereg/rules/slow-v3/: thresholds.json (episode and trend, from DR3
shard 0/10), period_thresholds.json and drift_table.parquet (as slow-v2). On release day `release` writes the same
files from DR4's own stars.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.slow import drift, sky
from cosmos_search.slow import rule as v1
from cosmos_search.slow.rule import mag_group

NAME = "slow-v3"
RULE_DIR = Path(__file__).resolve().parents[2] / "prereg" / "rules" / NAME
SHARED_LEVEL = 6
SHARED_S_MIN = 4.0


def load(rule_dir=RULE_DIR):
    """(episode/trend thresholds, period-search thresholds, drift table), each at the 0.1% share."""
    d = Path(rule_dir)
    thr = {g: v[v1.SHARE] for g, v in json.loads((d / "thresholds.json").read_text()).items()}
    rep = {g: v[v1.SHARE] for g, v in json.loads((d / "period_thresholds.json").read_text()).items()}
    return thr, rep, pd.read_parquet(d / "drift_table.parquet")


def shared(stats, level=SHARED_LEVEL, targets=None, target_patch=None):
    """Shared-date flags for `stats` (or for `targets` placed in `target_patch`) against the stars of `stats`."""
    pix = drift.sky_pixel(stats["source_id"].to_numpy(), level)
    return sky.shared_changes(stats, pix, targets, target_patch, s_min=SHARED_S_MIN)


def apply(stats, thr, thr_p, shared_flags=None):
    """stats: detector and period-search output with gbin (and every other star of the same sky for the shared
    test). shared_flags: precomputed `shared` output, else computed from stats. Returns id, gbin, scores,
    flagged (fading), early (second list) and which score passed."""
    sh = shared(stats) if shared_flags is None else shared_flags
    th = pd.DataFrame(thr).T.reindex(mag_group(stats["gbin"]))
    s_th, z_th = th["s"].to_numpy(), th["z"].to_numpy()
    rep_th = pd.Series(thr_p).reindex(mag_group(stats["gbin"])).to_numpy()
    ep = (stats["s_fade"].to_numpy() > s_th) & ~sh["shared_fade"].to_numpy()
    tr = stats["slope_z"].to_numpy() < -z_th
    rep = stats["s_rep"].to_numpy() > rep_th
    early = ((stats["s_rise"].to_numpy() > s_th) & (stats["rise_kind"].to_numpy() == "jump")
             & ~sh["shared_rise"].to_numpy())
    out = stats[["source_id", "gbin", "n_visits", "sigma_visit", "abbe", "s_fade", "s_rise", "fade_kind",
                 "rise_kind", "slope_z", "s_rep"]].rename(columns={"source_id": "id"}).reset_index(drop=True)
    out["shared_fade"] = sh["shared_fade"].to_numpy()
    out["flagged"] = ep | tr | rep
    out["early"] = early & ~out["flagged"].to_numpy()
    by = [np.where(ep, "episode", ""), np.where(tr, "trend", ""), np.where(rep, "repeat", "")]
    out["by"] = ["+".join(x for x in t if x) for t in zip(*by)]
    return out
