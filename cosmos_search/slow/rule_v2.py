"""Rule slow-v2: slow-v1 (`rule`) or the slow period search (`period`), frozen for the period search's blind test
(docs/arms/slow-changes.md, "Slow period search").

A star is flagged when slow-v1 flags it (dimming episode or fading trend) or when its repeated-dimming score s_rep
passes the threshold of its magnitude range, set so that 0.1% of quiet DR3 stars (shard 0/10, half B) pass when
repeatedly brightening (s_rep_bright, which skips the repeat conditions and so sits a little high). Files frozen
in prereg/rules/slow-v2/: the detector thresholds and drift table of slow-v1, and period_thresholds.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.slow import rule as v1
from cosmos_search.slow.rule import _rate, mag_group

NAME = "slow-v2"
RULE_DIR = Path(__file__).resolve().parents[2] / "prereg" / "rules" / NAME
FAINT_GBIN = v1.FAINT_GBIN

# Pass criteria (pre-registered with the rule; see the doc for what each outcome leads to).
FALSE_ALARM_MAX = 0.006          # P1: natural quiet stars flagged, per magnitude range
SNR_MIN = 20.0                   # P2, P3: fakes counted at expected signal-to-noise >= this, G < 19
# P2: each new shape against the calibrated shape it resembles most, in this batch.
COUNTERPART = {"sawtooth_dips": "repeat_long_dip", "uneven_dips": "repeat_long_dip",
               "paired_dips": "repeat_long_dip", "drifting_cycle": "slow_cycle"}
HELD_OUT_MARGIN = 0.15
# P3: calibrated shapes at SNR >= 20 on shard 0 (scripts/slow_period_check.py, detector or period search).
KNOWN_CAL = {"repeat_long_dip": 0.65, "slow_cycle": 0.25}
KNOWN_MARGIN = 0.10


def load(rule_dir=RULE_DIR):
    """(detector thresholds, period-search thresholds, drift table), each at the 0.1% share."""
    thr, table = v1.load(rule_dir)
    p = json.loads((Path(rule_dir) / "period_thresholds.json").read_text())
    return thr, {g: v[v1.SHARE] for g, v in p.items()}, table


def apply(stats, thr, thr_p):
    """stats: detector and period-search output with gbin. Returns slow-v1's columns plus s_rep and which score
    passed."""
    out = v1.apply(stats, thr)
    th = pd.Series(thr_p).reindex(mag_group(stats["gbin"])).to_numpy()
    rep = stats["s_rep"].to_numpy() > th
    out["s_rep"] = stats["s_rep"].to_numpy()
    out["rep_period"] = stats["rep_period"].to_numpy()
    out["by"] = np.where(rep, np.where(out["flagged"], out["by"] + "+repeat", "repeat"), out["by"])
    out["flagged"] = out["flagged"].to_numpy() | rep
    return out


def score(m):
    """m: flags merged with the revealed truth (group natural / held_out / known / brightening, family, gbin, snr)."""
    m = m.copy()
    m["mag"] = mag_group(m["gbin"])
    out = {"rule": NAME, "criteria": {}}
    nat = m[m["group"] == "natural"]
    p1 = {g: _rate(d["flagged"]) for g, d in nat.groupby("mag")}
    for v in p1.values():
        v["pass"] = v["rate"] <= FALSE_ALARM_MAX
    out["criteria"]["P1_false_alarm"] = {**p1, "pass": all(v["pass"] for v in p1.values())}

    strong = m[(m["mag"] == "G<19") & (m["snr"] >= SNR_MIN)]
    known = {f: _rate(d["flagged"]) for f, d in strong[strong["group"] == "known"].groupby("family")}
    ho = {}
    for f, d in strong[strong["group"] == "held_out"].groupby("family"):
        ref = known.get(COUNTERPART[f], {"rate": np.nan})["rate"]
        ho[f] = {**_rate(d["flagged"]), "counterpart": COUNTERPART[f], "counterpart_rate": ref}
        ho[f]["pass"] = bool(ho[f]["rate"] >= ref - HELD_OUT_MARGIN)
    out["criteria"]["P2_held_out_recall"] = {**ho, "pass": len(ho) == len(COUNTERPART)
                                             and all(v["pass"] for v in ho.values())}
    for f, v in known.items():
        v["calibration"] = KNOWN_CAL[f]
        v["pass"] = v["rate"] >= KNOWN_CAL[f] - KNOWN_MARGIN
    out["criteria"]["P3_known_recall"] = {**known, "pass": all(v["pass"] for v in known.values())}
    out["pass"] = all(c["pass"] for c in out["criteria"].values())

    inj = m[m["group"] != "natural"].copy()
    inj["snr_bin"] = pd.cut(inj["snr"], [0, 5, 10, 20, 40, np.inf]).astype(str)
    out["recall_by_snr"] = {f"{g}/{f}": {b: _rate(dd["flagged"]) for b, dd in d.groupby("snr_bin")}
                            for (g, f), d in inj[inj["mag"] == "G<19"].groupby(["group", "family"])}
    out["found_by"] = {f"{g}/{f}": d.loc[d["flagged"], "by"].value_counts().to_dict()
                       for (g, f), d in inj.groupby(["group", "family"])}
    return out
