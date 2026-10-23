"""Rule slow-v1: which stars the slow-change detector flags, frozen for its blind test (docs/arms/slow-changes.md,
"Blind test (S6)").

A star is flagged when its dimming episode score s_dim or its fading trend -slope_z passes the threshold of its
magnitude range, set so that 0.1% of quiet DR3 stars (shard 0/10, half B) pass when brightening. Epochs are first
corrected with the drift table fitted on half A of the same stars. Both files are frozen in prereg/rules/slow-v1/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

NAME = "slow-v1"
RULE_DIR = Path(__file__).resolve().parents[2] / "prereg" / "rules" / NAME
SHARE = "0.001"
FAINT_GBIN = 8  # drift.G_EDGES[8] = 19

# Pass criteria (pre-registered with the rule; see the doc for what each outcome leads to).
FALSE_ALARM_MAX = 0.005          # P1: natural quiet stars flagged, per magnitude range
SNR_MIN = 20.0                   # P2, P3: fakes counted at expected signal-to-noise >= this, G < 19
HELD_OUT_POOLED_MIN = 0.70       # P2: pooled recall over the held-out shapes
HELD_OUT_LOWER_MIN = 0.60        # P2: and its 95% Clopper-Pearson lower bound
HELD_OUT_FAMILY_MIN = 0.50       # P2: and each held-out shape on its own
KNOWN_S3 = {"long_dip": 0.834, "ramp_fade": 0.836, "secular_fade": 1.0}  # S3 recall at SNR >= 20, shard 0
KNOWN_MARGIN = 0.10              # P3: each calibrated shape within this of its S3 recall


def load(rule_dir=RULE_DIR):
    """(thresholds per magnitude range at the 0.1% share, drift table)."""
    thr = json.loads((Path(rule_dir) / "thresholds.json").read_text())
    return {g: v[SHARE] for g, v in thr.items()}, pd.read_parquet(Path(rule_dir) / "drift_table.parquet")


def mag_group(gbin):
    return np.where(np.asarray(gbin) <= FAINT_GBIN - 1, "G<19", "G>=19")


def apply(stats, thr):
    """stats: detector output (detect.star_features) with gbin. Returns id, gbin, scores, flagged and which score
    passed."""
    th = pd.DataFrame(thr).T.reindex(mag_group(stats["gbin"]))
    ep = stats["s_dim"].to_numpy() > th["s"].to_numpy()
    tr = stats["slope_z"].to_numpy() < -th["z"].to_numpy()
    out = stats[["source_id", "gbin", "n_visits", "sigma_visit", "abbe", "s_dim", "s_bright", "slope_z"]].copy()
    out = out.rename(columns={"source_id": "id"})
    out["flagged"] = ep | tr
    out["by"] = np.where(ep & tr, "both", np.where(ep, "episode", np.where(tr, "trend", "")))
    return out


def cp_upper(k, n, conf=0.95):
    return float(beta.ppf(conf, k + 1, n - k)) if k < n else 1.0


def cp_lower(k, n, conf=0.95):
    return float(beta.ppf(1 - conf, k, n - k + 1)) if k > 0 else 0.0


def _rate(x):
    k, n = int(np.sum(x)), len(x)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lower_95": cp_lower(k, n), "upper_95": cp_upper(k, n)}


def score(m):
    """m: flags merged with the revealed truth (group natural / held_out / known / brightening, family, gbin,
    snr). Returns the three criteria and the reported-only tables."""
    m = m.copy()
    m["mag"] = mag_group(m["gbin"])
    out = {"rule": NAME, "criteria": {}}
    nat = m[m["group"] == "natural"]
    p1 = {g: _rate(d["flagged"]) for g, d in nat.groupby("mag")}
    for v in p1.values():
        v["pass"] = v["rate"] <= FALSE_ALARM_MAX
    out["criteria"]["P1_false_alarm"] = {**p1, "pass": all(v["pass"] for v in p1.values())}

    strong = m[(m["mag"] == "G<19") & (m["snr"] >= SNR_MIN)]
    ho = strong[strong["group"] == "held_out"]
    pooled = _rate(ho["flagged"])
    fam = {f: _rate(d["flagged"]) for f, d in ho.groupby("family")}
    for v in fam.values():
        v["pass"] = v["rate"] >= HELD_OUT_FAMILY_MIN
    out["criteria"]["P2_held_out_recall"] = {
        "pooled": pooled, "families": fam,
        "pass": pooled["rate"] >= HELD_OUT_POOLED_MIN and pooled["lower_95"] >= HELD_OUT_LOWER_MIN
        and all(v["pass"] for v in fam.values())}

    kn = {f: {**_rate(d["flagged"]), "s3": KNOWN_S3[f]} for f, d in strong[strong["group"] == "known"].groupby("family")}
    for v in kn.values():
        v["pass"] = v["rate"] >= v["s3"] - KNOWN_MARGIN
    out["criteria"]["P3_known_recall"] = {**kn, "pass": all(v["pass"] for v in kn.values())}
    out["pass"] = all(c["pass"] for c in out["criteria"].values())

    inj = m[m["group"] != "natural"].copy()
    inj["snr_bin"] = pd.cut(inj["snr"], [0, 5, 10, 20, 40, np.inf]).astype(str)
    out["recall_by_snr"] = {f"{g}/{f}": {b: _rate(dd["flagged"]) for b, dd in d.groupby("snr_bin")}
                            for (g, f), d in inj[inj["mag"] == "G<19"].groupby(["group", "family"])}
    out["found_by"] = {f"{g}/{f}": d.loc[d["flagged"], "by"].value_counts().to_dict()
                       for (g, f), d in inj.groupby(["group", "family"])}
    return out
